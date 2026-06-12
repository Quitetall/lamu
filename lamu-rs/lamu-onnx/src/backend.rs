//! `Backend` impl for the in-process ONNX embed engine (ADRs 0033/0034).
//!
//! Shape mirrors `lamu-tts`'s fish_speech backend — load / health / unload
//! lifecycle driven by the scheduler + loader — except the "server" is a
//! tokio task ([`lamu_inproc::spawn_embed_server`]), not a subprocess:
//!
//! - `load()` builds the ort engine in `spawn_blocking`, spawns the
//!   in-process axum server on the assigned port, health-polls it, and
//!   returns **`std::process::id()` — lamu's own pid — as the "pid"**.
//!   The Backend contract wants a pid for the scheduler's bookkeeping and
//!   GPU-pid matching; an in-process backend has no child process, and
//!   lamu's own pid is the one process the served port genuinely belongs
//!   to. The pid-kill path (`kill_pid_and_verify`) explicitly refuses to
//!   signal lamu's own pid, so this can never become a self-SIGTERM; it
//!   also means v1 in-process servers can't be torn down via MCP's
//!   pid-path unload — only an owning handle's `unload()` (or lamu
//!   exiting) stops one. With vram_mb 0 there is no VRAM to reclaim, so
//!   that's a documented non-event, not a leak.
//! - `is_healthy()` is a local TCP `GET /health` (the in-process server
//!   answers llama-server's `{"status":"ok"}` shape on both `/health` and
//!   `/v1/health`).
//! - `unload()` aborts the server task (closing the listener, releasing
//!   the port) and drops the engine.
//! - `generate`/`stream` error: this backend is embeddings-only (v1);
//!   requests proxy to `POST /v1/embeddings` per the port-proxy
//!   architecture.

use async_trait::async_trait;
use futures_util::stream::Stream;
use lamu_core::backends::{Backend, ChatMessage};
use lamu_core::types::{DevicePlacement, ModelEntry};
use lamu_core::{Error, Result};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::engine::OnnxEmbedEngine;

pub struct OnnxBackend {
    engine: Option<Arc<OnnxEmbedEngine>>,
    server: Option<tokio::task::JoinHandle<()>>,
    port: u16,
    model_name: String,
}

impl OnnxBackend {
    /// Construct (unloaded) for a registry entry. The entry is re-supplied
    /// to `load()` by the trait contract; nothing needs caching here yet —
    /// the signature takes it to match the module factory (ADR 0023) and
    /// to leave room for entry-driven config without an API break.
    pub fn new(_entry: &ModelEntry) -> Self {
        Self {
            engine: None,
            server: None,
            port: 0,
            model_name: String::new(),
        }
    }
}

#[async_trait]
impl Backend for OnnxBackend {
    /// CPU execution provider only in v1 — there is no device to pin, so
    /// placement is deliberately ignored (the scheduler sees vram_mb 0 and
    /// has no interplay with this backend anyway).
    fn set_device(&mut self, _placement: DevicePlacement) {}

    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32> {
        // The port-proxy architecture (ADR 0033) needs a deterministic
        // port: 0 would let the OS pick one nobody can find, and the
        // health poll below would just time out confusingly.
        if port == 0 {
            return Err(Error::Backend(
                "onnx backend requires a concrete port (got 0) — the loader assigns one".into(),
            ));
        }
        self.port = port;
        self.model_name = entry.name.clone();

        // ort session construction + the probe embed are blocking CPU work.
        let model_path = entry.path.clone();
        let engine = tokio::task::spawn_blocking(move || OnnxEmbedEngine::load(&model_path))
            .await
            .map_err(|e| Error::Backend(format!("onnx engine load task: {e}")))?
            .map_err(|e| Error::Backend(format!("onnx engine load: {e:#}")))?;
        let engine = Arc::new(engine);
        self.engine = Some(engine.clone());
        tracing::info!(
            model = %entry.name,
            port,
            dims = engine.dims(),
            "onnx embed engine loaded — serving in-process"
        );

        let handle = lamu_inproc::spawn_embed_server(port, entry.name.clone(), engine);
        self.server = Some(handle);

        // The server binds inside its task; confirm liveness the same way
        // the loader treats subprocess backends — poll /health. A CPU axum
        // server is up in milliseconds; 5s of budget is generous.
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            // Bind failure makes the task return immediately — bail early
            // with a useful error instead of polling out the budget.
            if self.server.as_ref().map(|h| h.is_finished()).unwrap_or(true) {
                let _ = self.unload().await;
                return Err(Error::Backend(format!(
                    "onnx in-process server exited during startup — port {port} likely already bound"
                )));
            }
            if self.is_healthy().await {
                // In-process: lamu's own pid (see module docs for why).
                return Ok(std::process::id());
            }
        }

        let _ = self.unload().await;
        Err(Error::Backend(format!(
            "onnx in-process server health timeout (port {port})"
        )))
    }

    async fn unload(&mut self) -> Result<()> {
        if let Some(handle) = self.server.take() {
            // Abort the accept loop and WAIT for the task to finish so the
            // listening socket is provably closed (port released) before we
            // report the unload done.
            handle.abort();
            let _ = handle.await;
        }
        self.engine = None;
        self.model_name.clear();
        self.port = 0; // back to the unloaded invariant
        Ok(())
    }

    async fn is_healthy(&self) -> bool {
        if self.port == 0 {
            return false;
        }
        http_health_ok(self.port).await
    }

    async fn generate(
        &self,
        _messages: Vec<ChatMessage>,
        _max_tokens: u32,
        _temperature: f32,
    ) -> Result<String> {
        Err(Error::Backend("onnx backend is embeddings-only (v1)".into()))
    }

    async fn stream(
        &self,
        _messages: Vec<ChatMessage>,
        _max_tokens: u32,
        _temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        Err(Error::Backend("onnx backend is embeddings-only (v1)".into()))
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Minimal local `GET /health` over a raw TCP stream — enough to read the
/// in-process server's `{"status":"ok"}` without pulling an HTTP client
/// into this crate. Loopback-only by construction.
async fn http_health_ok(port: u16) -> bool {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let connect = tokio::time::timeout(
        Duration::from_secs(2),
        tokio::net::TcpStream::connect(("127.0.0.1", port)),
    )
    .await;
    let Ok(Ok(mut stream)) = connect else { return false };
    let req = format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).await.is_err() {
        return false;
    }
    let mut buf = Vec::new();
    let read = tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut buf)).await;
    if !matches!(read, Ok(Ok(_))) {
        return false;
    }
    let text = String::from_utf8_lossy(&buf);
    // Status line 200 + the llama-server-shaped ok body.
    text.starts_with("HTTP/1.1 200") && text.contains("\"status\":\"ok\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamu_core::types::{BackendType, ModelFormat};

    fn entry() -> ModelEntry {
        ModelEntry {
            name: "test-onnx".into(),
            path: "/nonexistent/model.onnx".into(),
            format: ModelFormat::Onnx,
            backend: BackendType::Onnx,
            backend_kind: None,
            arch: "unknown".into(),
            params_b: 0.0,
            quant: "fp32".into(),
            vram_mb: 0,
            context_max: 512,
            capabilities: vec![],
            reasoning_marker: None,
            speculative: None,
            sampling: None,
            pinned: false,
            main: false,
            notes: String::new(),
            status: Default::default(),
            modality: Default::default(),
            system_prompt: None,
        }
    }

    #[tokio::test]
    async fn unload_before_load_is_ok() {
        let mut b = OnnxBackend::new(&entry());
        assert!(b.unload().await.is_ok(), "unload on a never-loaded backend is a no-op");
        assert_eq!(b.port(), 0);
        assert_eq!(b.model_name(), "");
    }

    #[tokio::test]
    async fn generate_and_stream_error_embeddings_only() {
        let b = OnnxBackend::new(&entry());
        let g = b.generate(vec![], 16, 0.0).await;
        match g {
            Ok(_) => panic!("generate must fail on an embeddings-only backend"),
            Err(e) => assert!(format!("{e}").contains("embeddings-only"), "got: {e}"),
        }
        assert!(b.stream(vec![], 16, 0.0).await.is_err());
    }

    #[tokio::test]
    async fn load_rejects_port_zero() {
        let mut b = OnnxBackend::new(&entry());
        match b.load(&entry(), 0).await {
            Ok(_) => panic!("port 0 must be rejected"),
            Err(e) => assert!(format!("{e}").contains("concrete port"), "got: {e}"),
        }
    }

    #[tokio::test]
    async fn load_with_missing_model_fails_cleanly() {
        let mut b = OnnxBackend::new(&entry());
        let r = b.load(&entry(), 8123).await;
        match r {
            Ok(_) => panic!("load must fail for a nonexistent model"),
            Err(e) => assert!(format!("{e}").contains("not found"), "got: {e}"),
        }
        // Failure leaves no server task behind.
        assert!(!b.is_healthy().await);
    }

    #[tokio::test]
    async fn is_healthy_false_when_unloaded() {
        let b = OnnxBackend::new(&entry());
        assert!(!b.is_healthy().await);
    }
}
