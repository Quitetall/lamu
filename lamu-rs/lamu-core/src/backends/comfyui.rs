//! ComfyUI image-generation backend — local, managed subprocess.
//!
//! Spawns ComfyUI's web server:
//!   <comfy>/.venv/bin/python <comfy>/main.py --listen 127.0.0.1 --port <port>
//! and proxies image requests to `POST /prompt` + `GET /history` + `GET /view`
//! (done by the `generate_image` MCP tool, NOT this trait).
//!
//! One ComfyUI serves MANY checkpoints — the registry `path` is the ComfyUI
//! install dir (venv + main.py + models/), and the per-request `checkpoint`
//! (a file under models/checkpoints/) is chosen in the workflow graph the
//! tool POSTs, not at spawn.
//!
//! Backend::generate/stream are NOT implemented (image gen returns PNG bytes,
//! not a String) — they return a directing error. The trait impl exists only
//! for the lifecycle (load / unload / is_healthy / port) the scheduler drives.
//! `Modality::Image` (tiered eviction: dropped before LLMs) is already wired.

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

pub struct ComfyUIBackend {
    proc: Option<Child>,
    port: u16,
    model_name: String,
    client: reqwest::Client,
    cuda_index: u32,
}

impl ComfyUIBackend {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(600)) // image gen can be slow
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
impl Backend for ComfyUIBackend {
    fn set_device(&mut self, placement: DevicePlacement) {
        self.cuda_index = placement.primary_index();
    }

    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32> {
        self.port = port;
        self.model_name = entry.name.clone();

        // entry.path is the ComfyUI install dir (venv + main.py + models/).
        let comfy = &entry.path;
        let python_bin = comfy.join(".venv/bin/python");
        let main_py = comfy.join("main.py");
        for p in [&python_bin, &main_py] {
            if !p.exists() {
                return Err(Error::Backend(format!("comfyui: missing {}", p.display())));
            }
        }

        // Capture stderr to a log (file, not a pipe — undrained pipe
        // deadlocks the child) for diagnostics on a failed startup.
        let log_path = dirs::data_dir()
            .unwrap_or_else(std::env::temp_dir) // never the cwd (CI/containers)
            .join("lamu")
            .join("images");
        if let Err(e) = std::fs::create_dir_all(&log_path) {
            tracing::warn!("comfyui: log dir {} not creatable ({e}); startup diagnostics may be lost", log_path.display());
        }
        let log_file = log_path.join(format!("comfyui-{port}.log"));
        let stderr_sink = std::fs::File::create(&log_file)
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());

        let mut cmd = Command::new(&python_bin);
        cmd.arg(&main_py)
            .arg("--listen")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .current_dir(comfy)
            .env("CUDA_VISIBLE_DEVICES", self.cuda_index.to_string())
            .stdout(Stdio::null())
            .stderr(stderr_sink);
        crate::backends::harden_child_command(&mut cmd);

        let child = cmd
            .spawn()
            .map_err(|e| Error::Backend(format!("spawn comfyui: {}", e)))?;
        let pid = child.id().unwrap_or(0);
        self.proc = Some(child);
        if pid == 0 {
            let _ = self.unload().await;
            return Err(Error::Backend("comfyui: spawned process has pid 0".into()));
        }

        // Health poll. 90 × 2s = 3 min: ComfyUI import + custom-node init is
        // slow; the checkpoint loads lazily on first /prompt, not here.
        for _ in 0..90 {
            sleep(Duration::from_secs(2)).await;
            if let Some(p) = self.proc.as_mut() {
                if let Ok(Some(status)) = p.try_wait() {
                    return Err(Error::Backend(format!(
                        "comfyui server exited during startup (port {port}, {status})\nstderr tail:\n{}",
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
            "comfyui health timeout (port {port})\nstderr tail:\n{}",
            crate::backends::read_log_tail(&log_file)
        )))
    }

    async fn unload(&mut self) -> Result<()> {
        if let Some(mut p) = self.proc.take() {
            crate::backends::graceful_kill(&mut p).await;
        }
        self.model_name.clear();
        Ok(())
    }

    async fn is_healthy(&self) -> bool {
        let url = format!("http://127.0.0.1:{}/system_stats", self.port);
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
            "comfyui is an image backend — use the generate_image tool, not chat generation".into(),
        ))
    }

    async fn stream(
        &self,
        _messages: Vec<ChatMessage>,
        _max_tokens: u32,
        _temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        Err(Error::Backend("comfyui does not support text streaming".into()))
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
    fn backend_type_comfyui_serde() {
        assert_eq!(
            serde_json::to_string(&BackendType::ComfyUI).unwrap(),
            "\"comfyui\""
        );
        assert_eq!(
            serde_json::from_str::<BackendType>("\"comfyui\"").unwrap(),
            BackendType::ComfyUI
        );
    }

    #[test]
    fn new_is_unloaded_with_no_port() {
        let b = ComfyUIBackend::new().unwrap();
        assert_eq!(b.port(), 0);
        assert_eq!(b.model_name(), "");
    }
}
