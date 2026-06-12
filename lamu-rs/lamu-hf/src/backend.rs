//! `Backend` impl for the in-process candle chat engine (ADR 0035).
//!
//! Lifecycle mirrors `lamu-onnx`'s OnnxBackend (ADRs 0033/0034): the
//! "server" is a tokio task ([`lamu_inproc::spawn_chat_server`]), not a
//! subprocess —
//!
//! - `load()` picks the device (ADR 0017 placement → `Device::new_cuda`
//!   when the `cuda` feature is compiled; never `CUDA_VISIBLE_DEVICES`,
//!   which would constrain the whole lamu process), builds the candle
//!   engine in `spawn_blocking`, spawns the in-process axum server on the
//!   assigned port, health-polls it, and returns **`std::process::id()`**
//!   as the "pid" — same in-process convention as lamu-onnx; the pid-kill
//!   path refuses to signal lamu's own pid, so in-process servers tear
//!   down only via `unload()` or lamu exiting.
//! - `unload()` aborts the server task (releasing the port) and drops the
//!   engine. Dropping the engine frees the weights (host RAM, or device
//!   memory under `cuda`). CAVEAT for the scheduler's margin math: a CUDA
//!   context, once created in-process, keeps ~300-400 MB resident until
//!   lamu itself exits — unload returns the model's weights but not that
//!   fixed overhead. Subprocess backends don't have this; it's the cost
//!   of in-process GPU serving and should be priced into vram headroom.
//! - `generate`/`stream`/`generate_with_opts` run the engine DIRECTLY
//!   (no HTTP hop) for the MCP-held-backend path; HTTP traffic proxies to
//!   `POST /v1/chat/completions` per the port-proxy architecture.
//! - `tokenize_count` is the engine tokenizer's exact count (ADR 0021
//!   engine-truth), no HTTP round-trip needed.

use async_trait::async_trait;
use futures_util::stream::Stream;
use lamu_core::backends::{Backend, ChatMessage, GenerateOpts};
use lamu_core::types::{DevicePlacement, ModelEntry};
use lamu_core::{Error, Result};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::engine::CandleEngine;
use lamu_inproc::{ChatEngine, ChatRequestIn};

pub struct HfCandleBackend {
    engine: Option<CandleEngine>,
    server: Option<tokio::task::JoinHandle<()>>,
    port: u16,
    model_name: String,
    /// Stored by `set_device` (the loader calls it after the scheduler
    /// picks a device, before `load`). Used to pick the CUDA ordinal when
    /// the `cuda` feature is compiled; ignored on CPU builds.
    placement: DevicePlacement,
}

impl HfCandleBackend {
    /// Construct (unloaded) for a registry entry — entry re-supplied to
    /// `load()` by the trait contract (signature matches the ADR 0023
    /// module factory).
    pub fn new(_entry: &ModelEntry) -> Self {
        Self {
            engine: None,
            server: None,
            port: 0,
            model_name: String::new(),
            placement: DevicePlacement::default(),
        }
    }

    /// Resolve the candle device for the stored placement.
    ///
    /// IN-PROCESS rule: pin via `Device::new_cuda(primary_index)`, never
    /// `CUDA_VISIBLE_DEVICES` — an env var would remap devices for the
    /// whole lamu process (and every other in-process engine in it).
    /// `Sharded` is treated as its primary index, same as every backend
    /// (ADR 0017 P2 placeholder).
    fn pick_device(&self) -> Result<candle_core::Device> {
        #[cfg(feature = "cuda")]
        {
            let idx = self.placement.primary_index() as usize;
            return candle_core::Device::new_cuda(idx)
                .map_err(|e| Error::Backend(format!("cuda device {idx}: {e}")));
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(candle_core::Device::Cpu)
        }
    }

    fn build_request(
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        opts: &GenerateOpts,
    ) -> ChatRequestIn {
        ChatRequestIn {
            messages: messages.into_iter().map(|m| (m.role, m.content)).collect(),
            max_tokens,
            temperature,
            top_p: opts.top_p,
            top_k: opts.top_k,
            stream: false,
        }
    }

    fn engine(&self) -> Result<CandleEngine> {
        self.engine
            .clone()
            .ok_or_else(|| Error::Backend("hf_candle backend is not loaded".into()))
    }
}

#[async_trait]
impl Backend for HfCandleBackend {
    /// STORE the placement (used by `pick_device` at load). Contrast with
    /// subprocess backends, which export `CUDA_VISIBLE_DEVICES` for the
    /// child — an in-process engine must address the device directly.
    fn set_device(&mut self, placement: DevicePlacement) {
        self.placement = placement;
    }

    async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32> {
        // Port-proxy architecture (ADR 0033) needs a deterministic port;
        // 0 would let the OS pick one nobody can find.
        if port == 0 {
            return Err(Error::Backend(
                "hf_candle backend requires a concrete port (got 0) — the loader assigns one".into(),
            ));
        }
        self.port = port;
        self.model_name = entry.name.clone();

        let device = self.pick_device()?;
        // mmap + (on cuda) weight upload is blocking work.
        let model_path = entry.path.clone();
        let engine = tokio::task::spawn_blocking(move || CandleEngine::load(&model_path, device))
            .await
            .map_err(|e| Error::Backend(format!("candle engine load task: {e}")))?
            .map_err(|e| Error::Backend(format!("candle engine load: {e:#}")))?;
        self.engine = Some(engine.clone());
        tracing::info!(
            model = %entry.name,
            port,
            arch = engine.arch(),
            context_max = engine.context_max(),
            "candle chat engine loaded — serving in-process"
        );

        let handle = lamu_inproc::spawn_chat_server(
            port,
            entry.name.clone(),
            Arc::new(engine) as Arc<dyn ChatEngine>,
        );
        self.server = Some(handle);

        // Server binds inside its task — confirm liveness like the loader
        // does for subprocess backends: poll /health, bail early if the
        // task already died (bind failure).
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if self.server.as_ref().map(|h| h.is_finished()).unwrap_or(true) {
                let _ = self.unload().await;
                return Err(Error::Backend(format!(
                    "hf_candle in-process server exited during startup — port {port} likely already bound"
                )));
            }
            if self.is_healthy().await {
                // In-process: lamu's own pid (see module docs for why).
                return Ok(std::process::id());
            }
        }

        let _ = self.unload().await;
        Err(Error::Backend(format!(
            "hf_candle in-process server health timeout (port {port})"
        )))
    }

    async fn unload(&mut self) -> Result<()> {
        if let Some(handle) = self.server.take() {
            // Abort the accept loop and WAIT so the listening socket is
            // provably closed (port released) before reporting done.
            handle.abort();
            let _ = handle.await;
        }
        // Drops the weights. The fixed CUDA-context overhead stays
        // resident in-process — see module docs.
        self.engine = None;
        self.model_name.clear();
        self.port = 0;
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
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String> {
        self.generate_with_opts(messages, max_tokens, temperature, GenerateOpts::default())
            .await
    }

    /// Direct engine call (no HTTP hop) for the MCP-held-backend path.
    /// Honors `top_p`/`top_k`; the other GenerateOpts fields have no
    /// candle equivalent yet and are ignored (documented no-op, same as
    /// every backend without the matching feature).
    async fn generate_with_opts(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
        opts: GenerateOpts,
    ) -> Result<String> {
        let engine = self.engine()?;
        let req = Self::build_request(messages, max_tokens, temperature, &opts);
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
        let task = tokio::spawn(async move { ChatEngine::generate(&engine, req, tx).await });
        let mut out = String::new();
        while let Some(frag) = rx.recv().await {
            out.push_str(&frag);
        }
        match task.await {
            Err(e) => Err(Error::Backend(format!("hf_candle generate panicked: {e}"))),
            Ok(Err(e)) => Err(Error::Backend(format!("hf_candle generate: {e:#}"))),
            Ok(Ok(_)) => Ok(out),
        }
    }

    async fn stream(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let engine = self.engine()?;
        let req = Self::build_request(messages, max_tokens, temperature, &GenerateOpts::default());
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
        tokio::spawn(async move {
            if let Err(e) = ChatEngine::generate(&engine, req, tx).await {
                // The channel is already closed/derelict at this point;
                // stream consumers see end-of-stream. Log so the failure
                // isn't silent.
                tracing::error!("hf_candle stream generation failed: {e:#}");
            }
        });
        let s = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|frag| (Ok(frag), rx))
        });
        Ok(Box::pin(s))
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Engine-truth token count (ADR 0021) straight from the in-process
    /// tokenizer — no HTTP round-trip, no fabrication possible.
    async fn tokenize_count(&self, text: &str) -> Result<u32> {
        let engine = self.engine()?;
        let text = text.to_string();
        let count = tokio::task::spawn_blocking(move || engine.tokenize_count(&text))
            .await
            .map_err(|e| Error::Backend(format!("tokenize task panicked: {e}")))?
            .map_err(|e| Error::Backend(format!("tokenize: {e:#}")))?;
        Ok(count as u32)
    }
}

/// Minimal local `GET /health` over a raw TCP stream — same private
/// helper as lamu-onnx's backend (kept duplicated on purpose: it's 20
/// lines, and hoisting it into lamu-inproc would make the server crate
/// carry a client).
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
    text.starts_with("HTTP/1.1 200") && text.contains("\"status\":\"ok\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamu_core::types::{BackendType, ModelFormat};

    fn entry() -> ModelEntry {
        ModelEntry {
            name: "test-hf".into(),
            path: "/nonexistent/model-dir".into(),
            format: ModelFormat::Safetensors,
            backend: BackendType::HfCandle,
            backend_kind: None,
            arch: "qwen2".into(),
            params_b: 0.0,
            quant: "bf16".into(),
            vram_mb: 0,
            context_max: 4096,
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
        let mut b = HfCandleBackend::new(&entry());
        assert!(b.unload().await.is_ok());
        assert_eq!(b.port(), 0);
        assert_eq!(b.model_name(), "");
    }

    #[tokio::test]
    async fn load_rejects_port_zero() {
        let mut b = HfCandleBackend::new(&entry());
        match b.load(&entry(), 0).await {
            Ok(_) => panic!("port 0 must be rejected"),
            Err(e) => assert!(format!("{e}").contains("concrete port"), "got: {e}"),
        }
    }

    #[tokio::test]
    async fn load_with_missing_model_fails_cleanly() {
        let mut b = HfCandleBackend::new(&entry());
        match b.load(&entry(), 8127).await {
            Ok(_) => panic!("load must fail for a nonexistent model dir"),
            Err(e) => assert!(format!("{e}").contains("not found"), "got: {e}"),
        }
        assert!(!b.is_healthy().await, "failed load leaves no server behind");
    }

    #[tokio::test]
    async fn generate_unloaded_is_clear_error() {
        let b = HfCandleBackend::new(&entry());
        match b.generate(vec![], 8, 0.0).await {
            Ok(_) => panic!("generate on an unloaded backend must fail"),
            Err(e) => assert!(format!("{e}").contains("not loaded"), "got: {e}"),
        }
        assert!(b.stream(vec![], 8, 0.0).await.is_err());
        assert!(b.tokenize_count("x").await.is_err());
    }

    #[tokio::test]
    async fn is_healthy_false_when_unloaded() {
        let b = HfCandleBackend::new(&entry());
        assert!(!b.is_healthy().await);
    }

    #[test]
    fn set_device_stores_placement() {
        let mut b = HfCandleBackend::new(&entry());
        b.set_device(DevicePlacement::Single(1));
        assert_eq!(b.placement, DevicePlacement::Single(1));
        b.set_device(DevicePlacement::Sharded(vec![2, 3]));
        assert_eq!(b.placement.primary_index(), 2, "Sharded → primary index");
    }
}
