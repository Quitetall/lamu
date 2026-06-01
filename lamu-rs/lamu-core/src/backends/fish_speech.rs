//! fish-speech (OpenAudio S2-Pro) TTS backend — local, managed subprocess.
//!
//! Spawns fish-speech's own API server:
//!   <repo>/.venv/bin/python <repo>/tools/api_server.py --mode tts
//!     --listen 127.0.0.1:<port>
//!     --llama-checkpoint-path <ckpt-dir>          (e.g. checkpoints/s2-pro)
//!     --decoder-checkpoint-path <ckpt-dir>/codec.pth
//!     --decoder-config-name modded_dac_vq
//!     --device cuda --half --max-text-length 2000 --workers 1
//! and proxies TTS requests to `POST /v1/tts` (done by the
//! `text_to_speech` MCP tool, NOT this trait — see below).
//!
//! The repo dir is derived from the entry's checkpoint path
//! (`<repo>/checkpoints/s2-pro` → repo = `../..`), so a single registry
//! `path` locates the venv, the server script, and the codec.
//!
//! `--half` + `--max-text-length 2000` are load-bearing: S2-Pro is ~16 GB
//! resident in fp16 and the codec DECODE can peak toward the full card on
//! long input (a single monolithic `from_indices` pass), so the text cap
//! keeps each decode batch bounded. See docs/design/media-modalities.md §3/§6.
//!
//! Backend::generate/stream are NOT implemented (TTS has no text-gen
//! semantics + returns audio bytes, not a String) — they return an error
//! directing the caller to the `text_to_speech` tool, which proxies to the
//! spawned server's port. The trait impl exists only for the lifecycle
//! (load / unload / is_healthy / port) the scheduler + loader drive.

use crate::backends::{Backend, ChatMessage};
use crate::types::{DevicePlacement, ModelEntry};
use crate::{Error, Result};
use async_trait::async_trait;
use futures_util::stream::Stream;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::time::sleep;

pub struct FishSpeechBackend {
    proc: Option<Child>,
    port: u16,
    model_name: String,
    client: reqwest::Client,
    cuda_index: u32,
}

impl FishSpeechBackend {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| Error::Http(format!("reqwest client init: {}", e)))?;
        Ok(Self {
            proc: None,
            port: 0,
            model_name: String::new(),
            client,
            cuda_index: crate::config::gpu_index(),
        })
    }
}

#[async_trait]
impl Backend for FishSpeechBackend {
    fn set_device(&mut self, placement: DevicePlacement) {
        self.cuda_index = placement.primary_index();
    }

    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32> {
        self.port = port;
        self.model_name = entry.name.clone();

        // entry.path is the checkpoint dir (.../fish-speech/checkpoints/s2-pro);
        // the repo (venv + tools/api_server.py) is its grandparent.
        let ckpt = &entry.path;
        let repo = ckpt
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| {
                Error::Backend(format!(
                    "fish_speech: cannot derive repo dir from checkpoint path {}",
                    ckpt.display()
                ))
            })?;
        let python_bin = repo.join(".venv/bin/python");
        let api_server = repo.join("tools/api_server.py");
        let codec = ckpt.join("codec.pth");
        for p in [&python_bin, &api_server, &codec] {
            if !p.exists() {
                return Err(Error::Backend(format!("fish_speech: missing {}", p.display())));
            }
        }

        // Capture the server's stderr to a log file (NOT a pipe — an
        // undrained pipe would deadlock the child, the #8 lesson). On a
        // crash/timeout we read its tail into the error so the failure
        // reason (CUDA OOM, missing dep, bad ckpt) isn't lost.
        let log_path = dirs::data_dir()
            .unwrap_or_else(std::env::temp_dir) // never the cwd (CI/containers)
            .join("lamu")
            .join("tts");
        if let Err(e) = std::fs::create_dir_all(&log_path) {
            tracing::warn!("fish_speech: log dir {} not creatable ({e}); startup diagnostics may be lost", log_path.display());
        }
        let log_file = log_path.join(format!("fish-speech-{port}.log"));
        let stderr_sink = std::fs::File::create(&log_file)
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());

        let mut cmd = Command::new(&python_bin);
        cmd.arg(&api_server)
            .arg("--mode")
            .arg("tts")
            .arg("--listen")
            .arg(format!("127.0.0.1:{port}"))
            .arg("--llama-checkpoint-path")
            .arg(ckpt)
            .arg("--decoder-checkpoint-path")
            .arg(&codec)
            .arg("--decoder-config-name")
            .arg("modded_dac_vq")
            .arg("--device")
            .arg("cuda")
            .arg("--half") // fp16: ~halves resident VRAM (~16GB)
            .arg("--max-text-length")
            // Generous cap: the per-batch codec decode is bounded by the
            // request's chunk_length (200), so total text length drives
            // TIME not peak VRAM — this only rejects absurd inputs.
            .arg("4000")
            .arg("--workers")
            .arg("1")
            .current_dir(repo)
            .env("CUDA_VISIBLE_DEVICES", self.cuda_index.to_string())
            .stdout(Stdio::null())
            .stderr(stderr_sink);
        crate::backends::harden_child_command(&mut cmd);

        let child = cmd
            .spawn()
            .map_err(|e| Error::Backend(format!("spawn fish-speech: {}", e)))?;
        let pid = child.id().unwrap_or(0);
        self.proc = Some(child);
        if pid == 0 {
            // pid 0 breaks the scheduler's query_gpu_pids PID match.
            let _ = self.unload().await;
            return Err(Error::Backend("fish_speech: spawned process has pid 0".into()));
        }

        // Health poll. 90 × 2s = 3 min: a ~13-16 GB model off disk + the
        // torch/CUDA import is slow; the 60-iter LLM budget is too short.
        for _ in 0..90 {
            sleep(Duration::from_secs(2)).await;
            // Bail early if the server exited during startup (bad ckpt,
            // CUDA OOM, port bound) instead of polling the full timeout.
            if let Some(p) = self.proc.as_mut() {
                if let Ok(Some(status)) = p.try_wait() {
                    return Err(Error::Backend(format!(
                        "fish_speech server exited during startup (port {port}, {status})\nstderr tail:\n{}",
                        crate::backends::read_log_tail(&log_file)
                    )));
                }
            }
            if self.is_healthy().await {
                return Ok(pid);
            }
        }

        let _ = self.unload().await;
        Err(Error::Backend(format!(
            "fish_speech health timeout (port {port})\nstderr tail:\n{}",
            crate::backends::read_log_tail(&log_file)
        )))
    }

    async fn unload(&mut self) -> Result<()> {
        // Borrow (don't take) so a failed kill retains the handle for a retry.
        if let Some(p) = self.proc.as_mut() {
            crate::backends::graceful_kill(p, self.port).await?;
        }
        self.proc = None;
        self.model_name.clear();
        Ok(())
    }

    async fn is_healthy(&self) -> bool {
        let url = format!("http://127.0.0.1:{}/v1/health", self.port);
        matches!(
            self.client.get(&url).timeout(Duration::from_secs(2)).send().await,
            Ok(r) if r.status().is_success()
        )
    }

    async fn generate(
        &self,
        _messages: Vec<ChatMessage>,
        _max_tokens: u32,
        _temperature: f32,
    ) -> Result<String> {
        Err(Error::Backend(
            "fish_speech is a TTS backend — use the text_to_speech tool, not chat generation".into(),
        ))
    }

    async fn stream(
        &self,
        _messages: Vec<ChatMessage>,
        _max_tokens: u32,
        _temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        Err(Error::Backend(
            "fish_speech does not support text streaming".into(),
        ))
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BackendType;

    #[test]
    fn backend_type_fish_speech_serde() {
        assert_eq!(
            serde_json::to_string(&BackendType::FishSpeech).unwrap(),
            "\"fish_speech\""
        );
        assert_eq!(
            serde_json::from_str::<BackendType>("\"fish_speech\"").unwrap(),
            BackendType::FishSpeech
        );
    }

    #[test]
    fn new_is_unloaded_with_no_port() {
        let b = FishSpeechBackend::new().unwrap();
        assert_eq!(b.port(), 0);
        assert_eq!(b.model_name(), "");
    }
}
