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

/// Argv + env for spawning a `llama-server` configured for one
/// `ModelEntry`. Pure data — caller picks tokio or std::process,
/// caller spawns + health-polls + warms up. The flag set is the
/// authoritative one; lamu-mcp and lamu-cli's swap path consume from
/// here so the three spawn paths can never drift again.
#[derive(Debug, Clone)]
pub struct LlamaSpawn {
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
}

/// Build the canonical llama-server argv + env for `entry` on `port`.
///
/// `supports_ngram` is the result of probing `llama-server --help` for
/// `--spec-ngram-mod-n-match`. Pass `false` if the binary doesn't
/// support speculative ngram-mod (older builds) — the spec flags will
/// be omitted instead of failing the spawn.
///
/// Env knobs read here:
/// - `LAMU_DEFAULT_CTX` — caps the context window (default: full).
/// - `LAMU_KV` — KV cache type. Validated against the set llama.cpp
///   actually accepts; an unknown value is rejected up front rather
///   than crashing the server at startup or silently falling back to
///   f16. Default: `q8_0` (speed/VRAM sweet spot).
/// - `LAMU_BIND_HOST` — bind address (default: `127.0.0.1`). Set to
///   `0.0.0.0` to opt in to remote exposure.
pub fn build_llama_spawn(
    entry: &ModelEntry,
    port: u16,
    supports_ngram: bool,
) -> Result<LlamaSpawn> {
    let ctx_cap = std::env::var("LAMU_DEFAULT_CTX")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(u32::MAX);
    let ctx = entry.context_max.min(ctx_cap);

    let kv_type = match std::env::var("LAMU_KV").as_deref() {
        Ok("q8_0") | Ok("q4_0") | Ok("q4_1") | Ok("q5_0") | Ok("q5_1")
            | Ok("f16") | Ok("bf16") | Ok("f32") => std::env::var("LAMU_KV").unwrap(),
        Ok(other) => {
            return Err(Error::Backend(format!(
                "LAMU_KV='{}' invalid — expected one of: q8_0, q4_0, q4_1, q5_0, q5_1, f16, bf16, f32",
                other
            )));
        }
        Err(_) => "q8_0".to_string(),
    };

    let host = std::env::var("LAMU_BIND_HOST").unwrap_or_else(|_| "127.0.0.1".into());

    let mut args: Vec<String> = vec![
        "-m".into(), entry.path.display().to_string(),
        "--host".into(), host,
        "--port".into(), port.to_string(),
        "--ctx-size".into(), ctx.to_string(),
        "-ngl".into(), "99".into(),
        "--flash-attn".into(), "on".into(),
        "--cache-type-k".into(), kv_type.clone(),
        "--cache-type-v".into(), kv_type,
        "--parallel".into(), "1".into(),
        // Larger prompt-eval batches = fewer kernel launches per turn.
        // 4096/512 keeps VRAM stable on a 24GB card.
        "--batch-size".into(), "4096".into(),
        "--ubatch-size".into(), "512".into(),
        // Reuse shared prefixes across multi-turn chat — the next
        // turn's KV starts where the last one ended, so re-eval skips
        // the system prompt + history.
        "--cache-reuse".into(), "256".into(),
    ];

    if supports_ngram && (entry.arch == "qwen35" || entry.arch == "qwen3") {
        args.extend([
            "--spec-type".into(), "ngram-mod".into(),
            "--spec-ngram-mod-n-match".into(), "24".into(),
            "--spec-ngram-mod-n-min".into(), "12".into(),
            "--spec-ngram-mod-n-max".into(), "48".into(),
        ]);
    }

    Ok(LlamaSpawn {
        args,
        envs: vec![("CUDA_VISIBLE_DEVICES".into(), "0".into())],
    })
}

/// Async helper: probe `llama-server --help` for ngram-mod support.
pub async fn detect_ngram_support(bin: &std::path::Path) -> bool {
    match Command::new(bin).arg("--help").output().await {
        Ok(o) => String::from_utf8_lossy(&o.stdout).contains("--spec-ngram-mod-n-match"),
        Err(_) => false,
    }
}

/// Sync helper for blocking callers (the TUI swap path).
pub fn detect_ngram_support_blocking(bin: &std::path::Path) -> bool {
    match std::process::Command::new(bin).arg("--help").output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).contains("--spec-ngram-mod-n-match"),
        Err(_) => false,
    }
}

pub struct LlamaCppBackend {
    bin_path: PathBuf,
    proc: Option<Child>,
    port: u16,
    model_name: String,
    client: reqwest::Client,
}

impl LlamaCppBackend {
    pub fn new(bin_path: Option<PathBuf>) -> Result<Self> {
        // reqwest::Client::build can fail (e.g. invalid TLS config). Phase C:
        // propagate as Error::Http instead of panicking.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| Error::Http(format!("reqwest client init: {}", e)))?;
        Ok(Self {
            bin_path: bin_path.unwrap_or_else(llama_bin),
            proc: None,
            port: 0,
            model_name: String::new(),
            client,
        })
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

        let supports_ngram = detect_ngram_support(&self.bin_path).await;
        let spawn = build_llama_spawn(entry, port, supports_ngram)?;

        let mut cmd = Command::new(&self.bin_path);
        cmd.args(&spawn.args);
        for (k, v) in &spawn.envs {
            cmd.env(k, v);
        }
        cmd.stdout(Stdio::null()).stderr(Stdio::null());

        let child = cmd.spawn()
            .map_err(|e| Error::Backend(format!("spawn failed: {}", e)))?;

        let pid = child.id().unwrap_or(0);
        self.proc = Some(child);

        // Health poll
        for _ in 0..60 {
            sleep(Duration::from_secs(1)).await;
            if self.is_healthy().await {
                // Warmup: one-token completion so cuBLAS / cuDNN
                // handles + kv slot are JIT-compiled before the user's
                // first real prompt. Otherwise TTFT for the first
                // message bakes in 2–4s of kernel build cost.
                let warm_url = format!("http://localhost:{}/v1/chat/completions", port);
                let _ = self.client
                    .post(&warm_url)
                    .timeout(Duration::from_secs(30))
                    .json(&json!({
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "stream": false,
                    }))
                    .send()
                    .await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BackendType, Capability, ModelFormat};
    use std::sync::Mutex;

    // Tests in this mod read/write LAMU_* env vars. They are
    // process-local but thread-shared, so cargo test's default
    // multi-thread runner would race. Serialize via this Mutex.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_env() {
        // SAFETY: ENV_LOCK held by the calling test for the duration of
        // the test body. No concurrent reads from another LAMU env user
        // exist within the lamu-core test binary.
        unsafe {
            std::env::remove_var("LAMU_BIND_HOST");
            std::env::remove_var("LAMU_KV");
            std::env::remove_var("LAMU_DEFAULT_CTX");
        }
    }

    fn dummy_entry(arch: &str, ctx_max: u32) -> ModelEntry {
        ModelEntry {
            name: "test".into(),
            path: PathBuf::from("/tmp/test.gguf"),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
            arch: arch.into(),
            params_b: 4.0,
            quant: "Q4_K_M".into(),
            vram_mb: 4096,
            context_max: ctx_max,
            capabilities: vec![Capability::Code],
            reasoning_marker: None,
            speculative: None,
            pinned: false,
            notes: String::new(),
            status: String::new(),
        }
    }

    #[test]
    fn build_llama_spawn_defaults_localhost_q8_0() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let entry = dummy_entry("llama", 8192);
        let s = build_llama_spawn(&entry, 8020, false).unwrap();
        let joined = s.args.join(" ");
        assert!(joined.contains("--host 127.0.0.1"), "{joined}");
        assert!(joined.contains("--cache-type-k q8_0"), "{joined}");
        assert!(joined.contains("--cache-type-v q8_0"), "{joined}");
        assert!(joined.contains("--port 8020"), "{joined}");
        assert!(joined.contains("--ctx-size 8192"), "{joined}");
        assert!(joined.contains("--batch-size 4096"), "{joined}");
        assert!(joined.contains("--ubatch-size 512"), "{joined}");
        assert!(joined.contains("--cache-reuse 256"), "{joined}");
        assert!(!joined.contains("--spec-type"), "{joined}");
        assert_eq!(s.envs, vec![("CUDA_VISIBLE_DEVICES".into(), "0".into())]);
    }

    #[test]
    fn build_llama_spawn_rejects_bad_kv() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { std::env::set_var("LAMU_KV", "garbage"); }
        let entry = dummy_entry("llama", 8192);
        let r = build_llama_spawn(&entry, 8020, false);
        clear_env();
        assert!(matches!(r, Err(Error::Backend(_))), "expected Backend err, got {:?}", r);
    }

    #[test]
    fn build_llama_spawn_caps_ctx_via_env() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe { std::env::set_var("LAMU_DEFAULT_CTX", "4096"); }
        let entry = dummy_entry("llama", 131072);
        let s = build_llama_spawn(&entry, 8020, false).unwrap();
        clear_env();
        assert!(s.args.join(" ").contains("--ctx-size 4096"));
    }

    #[test]
    fn build_llama_spawn_emits_ngram_for_qwen3_when_supported() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let entry = dummy_entry("qwen3", 32768);
        let s = build_llama_spawn(&entry, 8020, true).unwrap();
        let joined = s.args.join(" ");
        assert!(joined.contains("--spec-type ngram-mod"));
        assert!(joined.contains("--spec-ngram-mod-n-match 24"));
    }

    #[test]
    fn build_llama_spawn_omits_ngram_for_non_qwen() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let entry = dummy_entry("llama", 8192);
        let s = build_llama_spawn(&entry, 8020, true).unwrap();
        assert!(!s.args.join(" ").contains("--spec-type"));
    }
}
