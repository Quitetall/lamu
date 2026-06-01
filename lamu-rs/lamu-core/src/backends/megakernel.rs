//! Megakernel backend — Qwen3.5-0.8B custom server.
//!
//! Spawns `python server/megakernel_server.py --port <p>` from
//! `~/local-llm/lucebox-hub/megakernel`. Speaks OpenAI HTTP on the chosen port.

use crate::backends::{Backend, ChatMessage};
use crate::scheduler::VramScheduler;
use crate::types::ModelEntry;
use crate::{Error, Result};
use async_trait::async_trait;
use futures_util::stream::{Stream, StreamExt};
use serde_json::Value;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::time::sleep;

pub struct MegakernelBackend {
    python_bin: PathBuf,
    server_script: PathBuf,
    work_dir: PathBuf,
    proc: Option<Child>,
    port: u16,
    model_name: String,
    client: reqwest::Client,
}

impl MegakernelBackend {
    pub fn new() -> Result<Self> {
        let home = dirs::home_dir().unwrap_or_default();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| Error::Http(format!("reqwest client init: {}", e)))?;
        Ok(Self {
            python_bin: home.join("local-llm/.venv/bin/python"),
            server_script: home.join("local-llm/server/megakernel_server.py"),
            work_dir: home.join("local-llm/lucebox-hub/megakernel"),
            proc: None,
            port: 0,
            model_name: String::new(),
            client,
        })
    }
}

#[async_trait]
impl Backend for MegakernelBackend {
    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32> {
        self.port = port;
        self.model_name = entry.name.clone();

        if !self.python_bin.exists() {
            return Err(Error::Backend(format!(
                "python not found at {}",
                self.python_bin.display()
            )));
        }
        if !self.server_script.exists() {
            return Err(Error::Backend(format!(
                "megakernel server script not found at {}",
                self.server_script.display()
            )));
        }

        let mut cmd = Command::new(&self.python_bin);
        cmd.arg(&self.server_script)
            .arg("--port").arg(port.to_string())
            .current_dir(&self.work_dir)
            .env("CUDA_VISIBLE_DEVICES", crate::config::gpu_index().to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        crate::backends::harden_child_command(&mut cmd);

        let child = cmd.spawn()
            .map_err(|e| Error::Backend(format!("spawn failed: {}", e)))?;
        let pid = child.id().unwrap_or(0);
        self.proc = Some(child);

        for _ in 0..45 {
            sleep(Duration::from_secs(1)).await;
            // Bail early if the child exited during startup instead of
            // polling for the full timeout while holding the spawn gate. (#16)
            if let Some(p) = self.proc.as_mut() {
                if let Ok(Some(status)) = p.try_wait() {
                    return Err(Error::Backend(format!(
                        "megakernel server exited during startup (port {port}, {status})"
                    )));
                }
            }
            if self.is_healthy().await {
                return Ok(pid);
            }
        }

        let _ = self.unload().await;
        Err(Error::Backend(format!("megakernel health timeout (port {})", port)))
    }

    async fn unload(&mut self) -> Result<()> {
        if let Some(mut p) = self.proc.take() {
            crate::backends::graceful_kill(&mut p).await;
        }
        self.model_name.clear();
        Ok(())
    }

    async fn is_healthy(&self) -> bool {
        let url = format!("http://localhost:{}/health", self.port);
        self.client.get(&url)
            .timeout(Duration::from_secs(2))
            .send().await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String> {
        self.generate_with_opts(messages, max_tokens, temperature,
                                crate::backends::GenerateOpts::default()).await
    }

    async fn generate_with_opts(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        opts: crate::backends::GenerateOpts,
    ) -> Result<String> {
        let payload = crate::backends::build_payload(&messages, max_tokens, temperature, false, &opts);
        let url = format!("http://localhost:{}/v1/chat/completions", self.port);
        let resp = self.client.post(&url).json(&payload).send().await
            .map_err(|e| Error::Backend(format!("http: {}", e)))?;
        let data: Value = resp.json().await
            .map_err(|e| Error::Backend(format!("json: {}", e)))?;
        let msg = data.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message"))
            .ok_or_else(|| Error::Backend("missing choices[0].message".into()))?;
        Ok(msg.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string())
    }

    async fn stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let payload = crate::backends::build_payload(&messages, max_tokens, temperature, true,
                                    &crate::backends::GenerateOpts::default());
        let url = format!("http://localhost:{}/v1/chat/completions", self.port);
        let resp = self.client.post(&url).json(&payload).send().await
            .map_err(|e| Error::Backend(format!("http: {}", e)))?;

        let byte_stream = resp.bytes_stream();
        let line_stream = byte_stream
            .map(|res| res.map_err(|e| Error::Backend(format!("stream: {}", e))));

        let decoded = async_stream::try_stream! {
            let mut buf: Vec<u8> = Vec::new();
            let mut s = std::pin::pin!(line_stream);
            while let Some(chunk) = s.next().await {
                let chunk = chunk?;
                // Byte-buffer, decode whole lines only: from_utf8_lossy on a
                // raw chunk corrupts a multibyte char split across chunks.
                buf.extend_from_slice(&chunk);
                while let Some(line) = crate::sse::next_sse_line(&mut buf) {
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

impl MegakernelBackend {
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
