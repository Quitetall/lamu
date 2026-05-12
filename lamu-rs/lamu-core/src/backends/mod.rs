//! Backends — model lifecycle management.
//! Direct port of `lamu/backends/`.

pub mod dflash;
pub mod llamacpp;
pub mod megakernel;

use crate::types::{BackendType, ModelEntry};
use crate::Result;
use async_trait::async_trait;
use futures_util::stream::Stream;
use std::pin::Pin;

/// Construct the right backend impl for the entry's declared type.
pub fn make_backend(entry: &ModelEntry) -> Result<Box<dyn Backend>> {
    match entry.backend {
        BackendType::LlamaCpp => Ok(Box::new(llamacpp::LlamaCppBackend::new(None)?)),
        BackendType::Megakernel => Ok(Box::new(megakernel::MegakernelBackend::new()?)),
        BackendType::Dflash | BackendType::DflashLucebox => {
            Ok(Box::new(dflash::DflashBackend::new()?))
        }
    }
}

/// Per-call backend options. Backends without a corresponding feature
/// silently ignore unknown fields.
#[derive(Debug, Clone, Default)]
pub struct GenerateOpts {
    /// Qwen3.6 / Qwen3.5 reasoning toggle. `Some(false)` disables the
    /// `<think>` block via `chat_template_kwargs.enable_thinking`. `None`
    /// leaves the model's default behaviour (thinking on).
    pub enable_thinking: Option<bool>,
}

#[async_trait]
pub trait Backend: Send + Sync {
    /// Load model. Returns PID.
    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32>;

    /// Stop process and free VRAM.
    async fn unload(&mut self) -> Result<()>;

    /// Health check.
    async fn is_healthy(&self) -> bool;

    /// Generate non-streaming. Returns raw text (think blocks included).
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String>;

    /// Generate non-streaming with extended options. Default impl ignores
    /// `opts` and forwards to `generate()`. Backends override to honor
    /// per-call params like `enable_thinking`.
    async fn generate_with_opts(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        _opts: GenerateOpts,
    ) -> Result<String> {
        self.generate(messages, max_tokens, temperature).await
    }

    /// Generate streaming. Yields tokens.
    async fn stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>;

    fn port(&self) -> u16;
    fn model_name(&self) -> &str;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Graceful child shutdown: SIGTERM, wait up to 10s for the child to
/// exit cleanly, fall back to SIGKILL if it ignores TERM. Used by every
/// `Backend::unload` impl so server-side flushes (KV cache, log files,
/// etc.) get a chance to run before we tear the process down.
///
/// Why 10s: llama-server typically exits in <1s; megakernel/dflash
/// Python servers can take 2-5s to flush. 10s leaves margin without
/// hanging the MCP server when a child genuinely refuses to die.
#[cfg(unix)]
pub async fn graceful_kill(child: &mut tokio::process::Child) {
    use std::time::Duration;
    if let Some(pid) = child.id() {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGTERM,
        );
    }
    match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(_) => {
            tracing::debug!("graceful_kill: child exited cleanly after SIGTERM");
        }
        Err(_) => {
            tracing::warn!("graceful_kill: 10s SIGTERM timeout, escalating to SIGKILL");
            let _ = child.kill().await;
        }
    }
}

#[cfg(not(unix))]
pub async fn graceful_kill(child: &mut tokio::process::Child) {
    // No SIGTERM on non-Unix; just kill.
    let _ = child.kill().await;
}
