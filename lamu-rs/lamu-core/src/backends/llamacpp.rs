//! llama.cpp backend — manages llama-server subprocess.
//! Direct port of `lamu/backends/llamacpp.py`.

use crate::backends::{Backend, ChatMessage};
use crate::config::llama_bin;
use crate::scheduler::VramScheduler;
use crate::types::ModelEntry;
use crate::{Error, Result};
use async_trait::async_trait;
use futures_util::stream::{Stream, StreamExt};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::time::sleep;

pub struct LlamaCppBackend {
    bin_path: PathBuf,
    proc: Option<Child>,
    port: u16,
    model_name: String,
    client: reqwest::Client,
}

impl LlamaCppBackend {
    pub fn new(bin_path: Option<PathBuf>) -> Self {
        Self {
            bin_path: bin_path.unwrap_or_else(llama_bin),
            proc: None,
            port: 0,
            model_name: String::new(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("reqwest client"),
        }
    }

    /// Detect ngram-mod support by checking --help.
    async fn supports_ngram_mod(&self) -> bool {
        let Ok(out) = Command::new(&self.bin_path)
            .arg("--help")
            .output()
            .await
        else {
            return false;
        };
        let stdout = String::from_utf8_lossy(&out.stdout);
        stdout.contains("--spec-ngram-mod-n-match")
    }
}

#[async_trait]
impl Backend for LlamaCppBackend {
    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32> {
        self.port = port;
        self.model_name = entry.name.clone();

        if !self.bin_path.exists() {
            return Err(Error::Backend(format!(
                "llama-server not found at {}",
                self.bin_path.display()
            )));
        }

        let mut cmd = Command::new(&self.bin_path);
        cmd.arg("-m").arg(&entry.path)
            .arg("--host").arg("0.0.0.0")
            .arg("--port").arg(port.to_string())
            .arg("--ctx-size").arg(std::cmp::min(entry.context_max, 131072).to_string())
            .arg("-ngl").arg("99")
            .arg("--flash-attn").arg("on")
            .arg("--cache-type-k").arg("q4_0")
            .arg("--cache-type-v").arg("q4_0")
            .arg("--parallel").arg("1");

        if self.supports_ngram_mod().await
            && (entry.arch == "qwen35" || entry.arch == "qwen3") {
            cmd.args([
                "--spec-type", "ngram-mod",
                "--spec-ngram-mod-n-match", "24",
                "--spec-ngram-mod-n-min", "12",
                "--spec-ngram-mod-n-max", "48",
            ]);
        }

        cmd.env("CUDA_VISIBLE_DEVICES", "0")
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let child = cmd.spawn()
            .map_err(|e| Error::Backend(format!("spawn failed: {}", e)))?;

        let pid = child.id().unwrap_or(0);
        self.proc = Some(child);

        // Health poll
        for _ in 0..60 {
            sleep(Duration::from_secs(1)).await;
            if self.is_healthy().await {
                return Ok(pid);
            }
        }

        let _ = self.unload().await;
        Err(Error::Backend(format!("llama-server health timeout (port {})", port)))
    }

    async fn unload(&mut self) -> Result<()> {
        if let Some(mut p) = self.proc.take() {
            let _ = p.kill().await;
        }
        self.model_name.clear();
        Ok(())
    }

    async fn is_healthy(&self) -> bool {
        let url = format!("http://localhost:{}/health", self.port);
        let Ok(resp) = self.client.get(&url)
            .timeout(Duration::from_secs(2))
            .send().await
        else { return false; };
        let Ok(json) = resp.json::<Value>().await else { return false; };
        json.get("status").and_then(|v| v.as_str()) == Some("ok")
    }

    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String> {
        let payload = json!({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        let url = format!("http://localhost:{}/v1/chat/completions", self.port);
        let resp = self.client.post(&url).json(&payload).send().await
            .map_err(|e| Error::Backend(format!("http: {}", e)))?;
        let data: Value = resp.json().await
            .map_err(|e| Error::Backend(format!("json: {}", e)))?;
        let msg = data.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message"))
            .ok_or_else(|| Error::Backend("missing choices[0].message".into()))?;
        let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let reasoning = msg.get("reasoning_content").and_then(|v| v.as_str()).unwrap_or("");
        if !reasoning.is_empty() {
            Ok(format!("<think>\n{}\n</think>\n{}", reasoning, content))
        } else {
            Ok(content.to_string())
        }
    }

    async fn stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let payload = json!({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": true,
        });
        let url = format!("http://localhost:{}/v1/chat/completions", self.port);
        let resp = self.client.post(&url).json(&payload).send().await
            .map_err(|e| Error::Backend(format!("http: {}", e)))?;

        let byte_stream = resp.bytes_stream();
        let line_stream = byte_stream
            .map(|res| res.map_err(|e| Error::Backend(format!("stream: {}", e))));

        let decoded = async_stream::try_stream! {
            let mut buf = String::new();
            let mut s = std::pin::pin!(line_stream);
            while let Some(chunk) = s.next().await {
                let chunk = chunk?;
                buf.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(nl) = buf.find('\n') {
                    let line: String = buf.drain(..=nl).collect();
                    let line = line.trim();
                    if let Some(rest) = line.strip_prefix("data: ") {
                        if rest == "[DONE]" { return; }
                        let Ok(val) = serde_json::from_str::<Value>(rest) else { continue };
                        let Some(token) = val.get("choices")
                            .and_then(|c| c.get(0))
                            .and_then(|c| c.get("delta"))
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str())
                        else { continue };
                        if !token.is_empty() {
                            yield token.to_string();
                        }
                    }
                }
            }
        };

        Ok(Box::pin(decoded))
    }

    fn port(&self) -> u16 { self.port }
    fn model_name(&self) -> &str { &self.model_name }
}

impl LlamaCppBackend {
    /// Query actual VRAM usage via NVML PID lookup.
    pub fn get_vram_mb(&self, scheduler: &VramScheduler) -> u32 {
        let Some(p) = self.proc.as_ref() else { return 0 };
        let Some(my_pid) = p.id() else { return 0 };
        for (pid, mb) in scheduler.query_gpu_pids() {
            if pid == my_pid {
                return mb;
            }
        }
        0
    }
}
