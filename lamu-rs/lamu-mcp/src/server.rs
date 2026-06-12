//! MCP stdio server. Hand-rolled JSON-RPC.
//! Direct port of `lamu/mcp/server.py::LamuMcpServer`.
//!
//! Protocol: line-delimited JSON-RPC 2.0 over stdin/stdout.
//! Logs to stderr. Tools dispatched via `tools::*`.

use anyhow::Result;
use lamu_core::backends::Backend;
use lamu_core::health::HealthRegistry;
use lamu_core::queue::{RequestQueue, Strategy as QueueStrategy};
use lamu_core::registry::{load_registry, scan_directory, write_registry};
use lamu_core::router::Router;
use lamu_core::scheduler::VramScheduler;
use lamu_core::tools_ext::ToolCtxError;
use lamu_core::types::ModelEntry;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex as AsyncMutex;
use tracing::warn;

/// Clone is shallow — every field is an `Arc` (or `Copy`), so clones
/// share ALL state (scheduler, queues, routing mode). Used by the
/// ADR 0030 embedder adapter, which needs a process-lifetime handle to
/// the server's embed path.
#[derive(Clone)]
pub struct LamuMcpServer {
    pub state: Arc<Mutex<ServerState>>,
    /// Per-model request queues (separate from parking_lot state — async).
    pub queues: Arc<AsyncMutex<HashMap<String, Arc<RequestQueue<()>>>>>,
    pub queue_strategy: QueueStrategy,
    pub queue_concurrency: usize,
    /// Routing mode: 'auto', 'local-only', 'cloud-only'. Default 'auto'.
    /// `cloud-only` makes the local query path refuse and frees VRAM.
    pub routing_mode: Arc<AsyncMutex<String>>,
}

/// Shared handle to a loaded Backend. Wrapped in `Arc<TokioMutex<…>>`
/// so a query can clone the Arc out of the state lock and serialize
/// per-backend access without re-locking the whole ServerState across
/// an `.await`.
pub type BackendHandle = Arc<AsyncMutex<Box<dyn Backend>>>;

pub struct ServerState {
    pub models_dir: PathBuf,
    pub registry_path: PathBuf,
    pub scheduler: VramScheduler,
    pub entries: HashMap<String, ModelEntry>,
    pub router: Router,
    pub client: reqwest::Client,
    /// Health for every loaded backend. Shared with the router via
    /// `route(..., health_map=Some(health.all()))` so DEAD/QUARANTINED
    /// backends never get picked.
    pub health: HealthRegistry,
    /// Loaded backends keyed by model name. Each Backend impl owns its
    /// own Child + transport details; lamu-mcp only orchestrates
    /// load/unload + routes to .generate().
    pub backends: HashMap<String, BackendHandle>,
}

impl LamuMcpServer {
    pub fn new(models_dir: PathBuf, registry_path: PathBuf, scheduler: VramScheduler) -> Result<Self> {
        let mut entries_vec = load_registry(&registry_path)?;
        if entries_vec.is_empty() {
            entries_vec = scan_directory(&models_dir)?;
            write_registry(&entries_vec, &registry_path)?;
        }
        let entries: HashMap<String, ModelEntry> = entries_vec.iter()
            .map(|e| (e.name.clone(), e.clone()))
            .collect();
        let router = Router::new(&scheduler, entries_vec);
        // Phase C: propagate reqwest builder failure as Error::Http instead
        // of panicking the whole MCP server at startup.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| lamu_core::Error::Http(format!("reqwest client init: {}", e)))?;

        let queue_strategy = match std::env::var("LAMU_QUEUE_STRATEGY").as_deref() {
            Ok("lifo") => QueueStrategy::Lifo,
            Ok("priority") => QueueStrategy::Priority,
            _ => QueueStrategy::Fifo,
        };
        let queue_concurrency: usize = std::env::var("LAMU_QUEUE_CONCURRENCY")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1);

        let server = Self {
            state: Arc::new(Mutex::new(ServerState {
                models_dir,
                registry_path,
                scheduler,
                entries,
                router,
                client,
                health: HealthRegistry::new(),
                backends: HashMap::new(),
            })),
            queues: Arc::new(AsyncMutex::new(HashMap::new())),
            queue_strategy,
            queue_concurrency,
            routing_mode: Arc::new(AsyncMutex::new(
                // m22: validate the env value. Consumers match EXACTLY
                // "local-only"/"cloud-only"; a typo like "local_only" stored
                // verbatim would enforce NEITHER restriction (silent auto-like)
                // while routing_status printed the bogus mode. Fall back to auto.
                match std::env::var("LAMU_ROUTING_MODE").as_deref() {
                    Ok("local-only") => "local-only".to_string(),
                    Ok("cloud-only") => "cloud-only".to_string(),
                    Ok("auto") | Err(_) => "auto".to_string(),
                    Ok(other) => {
                        eprintln!("lamu-mcp: LAMU_ROUTING_MODE='{other}' invalid (expected auto|local-only|cloud-only); using auto");
                        "auto".to_string()
                    }
                }
            )),
        };
        server.register_local_embedder();
        Ok(server)
    }

    /// ADR 0030: register the process-global LOCAL embedder over this
    /// server's embed path (`ToolCtx::embed`: resolve the registry's
    /// `Capability::Embedding` model, ensure-load, POST its port).
    ///
    /// The memory MCP tools are `HandlerKind::Free` (no server ref) and
    /// the detached autocapture/reconcile tasks have no server either,
    /// so the registration must be process-global — `lamu_memory`'s
    /// embedder chain resolves it from any context.
    ///
    /// The chain stays STATIC: the adapter is registered only when the
    /// registry has an embedding-capable model AT STARTUP. Adding one
    /// later (scan_models / registry edit) requires a server restart to
    /// be picked up. With NO embedding-capable model, nothing is
    /// registered and the chain falls through to the keyed OpenAI leg.
    fn register_local_embedder(&self) {
        let model = self
            .state
            .lock()
            .entries
            .values()
            .filter(|e| e.capabilities.contains(&lamu_core::types::Capability::Embedding))
            .map(|e| e.name.clone())
            .min(); // deterministic when >1 embedding entry
        let Some(model) = model else { return };
        tracing::info!("ADR 0030: registering local embedder '{model}' (MCP embed path)");
        lamu_memory::embedder::set_global(Arc::new(McpServerEmbedder {
            server: self.clone(),
            model,
            dims: std::sync::atomic::AtomicUsize::new(0),
        }));
    }

    pub async fn run(self) -> Result<()> {
        // Orphan cleanup runs in lamu-cli main() — lamu-mcp is a
        // library, always invoked via `lamu start`. See
        // `lamu_core::lifecycle` for the PDEATHSIG + watchdog rationale.

        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin).lines();
        let mut writer = stdout;

        // Startup banner stays as explicit eprintln (not tracing) so it's
        // visible regardless of RUST_LOG. Some callers wait for this string
        // before sending requests.
        eprintln!("LAMU MCP server ready (rust)");

        // Install signal handlers ONCE outside the loop — each
        // `signal(SignalKind::*)` call registers a fresh handler with
        // the runtime; doing it per-iteration leaks file descriptors.
        #[cfg(unix)]
        let (mut sig_term, mut sig_int) = {
            use tokio::signal::unix::{signal, SignalKind};
            (
                signal(SignalKind::terminate())
                    .map_err(|e| anyhow::anyhow!("SIGTERM handler: {e}"))?,
                signal(SignalKind::interrupt())
                    .map_err(|e| anyhow::anyhow!("SIGINT handler: {e}"))?,
            )
        };

        loop {
            #[cfg(unix)]
            let line = tokio::select! {
                res = reader.next_line() => match res {
                    Ok(Some(l)) => l,
                    Ok(None) => {
                        tracing::info!("stdin EOF — parent harness closed pipe, shutting down");
                        break;
                    }
                    Err(e) => {
                        tracing::error!("read error: {}", e);
                        break;
                    }
                },
                _ = sig_term.recv() => {
                    tracing::info!("SIGTERM received, shutting down");
                    break;
                }
                _ = sig_int.recv() => {
                    tracing::info!("SIGINT received, shutting down");
                    break;
                }
            };
            #[cfg(not(unix))]
            let line = match reader.next_line().await {
                Ok(Some(l)) => l,
                Ok(None) => break,
                Err(e) => {
                    tracing::error!("read error: {}", e);
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }

            let request: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    warn!("bad json: {}", e);
                    continue;
                }
            };

            let id = request.get("id").cloned();
            let method = request.get("method").and_then(|v| v.as_str()).unwrap_or("");
            let params = request.get("params").cloned().unwrap_or(Value::Null);

            // Serial by design (ADR 0024): one request is fully handled
            // before the next line is read. Concurrency lives inside tools
            // (parallel_query, council) and at the per-model RequestQueue —
            // do not tokio::spawn here without superseding that ADR.
            let response = self.handle(method, params, id.clone()).await;

            // Notifications (no id) → no response
            if id.is_some() {
                if let Some(resp) = response {
                    let resp_str = serde_json::to_string(&resp)?;
                    writer.write_all(resp_str.as_bytes()).await?;
                    writer.write_all(b"\n").await?;
                    writer.flush().await?;
                }
            }
        }

        // Graceful teardown on a clean shutdown (SIGTERM / SIGINT / stdin
        // EOF — the loop `break`s above). Without this, the loaded
        // llama-server children were only reaped by Child-drop, which on
        // some exit paths never ran → leaked VRAM. We now SIGTERM each
        // backend (KV/log flush) before exit. The orphan-watchdog's
        // exit(0) path can't reach here — that's covered by the
        // children's PR_SET_PDEATHSIG — and kill_on_drop is the final
        // backstop if this drain is skipped or times out.
        self.drain_backends_on_shutdown().await;

        Ok(())
    }

    /// Best-effort graceful unload of every loaded backend at shutdown.
    /// Bounded per-backend so a wedged child can't hang the exit; the
    /// child's kill_on_drop + PDEATHSIG guarantee it dies regardless.
    async fn drain_backends_on_shutdown(&self) {
        let handles: Vec<(String, BackendHandle)> = {
            let mut st = self.state.lock();
            st.backends.drain().collect()
        };
        if handles.is_empty() {
            return;
        }
        tracing::info!("shutdown: draining {} backend(s)", handles.len());
        for (name, backend_arc) in handles {
            let mut b = backend_arc.lock().await;
            match tokio::time::timeout(std::time::Duration::from_secs(10), b.unload()).await {
                Ok(Ok(_)) => tracing::info!("shutdown: unloaded {}", name),
                Ok(Err(e)) => warn!("shutdown: unload({}) errored: {}", name, e),
                Err(_) => warn!("shutdown: unload({}) timed out — kill_on_drop/PDEATHSIG will reap", name),
            }
        }
    }

    /// Dispatch one JSON-RPC request. The stdio loop in `run()` calls
    /// this; integration tests bypass stdio and call it directly.
    pub async fn handle(&self, method: &str, params: Value, id: Option<Value>) -> Option<Value> {
        match method {
            "initialize" => Some(initialize_response(id)),
            "notifications/initialized" => None,
            "tools/list" => Some(tools_list_response(id)),
            "tools/call" => Some(self.tools_call(params, id).await),
            "ping" => Some(json!({"jsonrpc":"2.0","id":id,"result":{}})),
            _ => Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("method not found: {}", method) }
            })),
        }
    }

    async fn tools_call(&self, params: Value, id: Option<Value>) -> Value {
        let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(Value::Object(Default::default()));

        // Routing-mode gate: 'local-only' refuses cloud-LLM tools — the
        // mirror of handle_query's 'cloud-only' refusal for local
        // queries. Cloud tools are HandlerKind::Free and never receive
        // `&self`, so they can't consult routing_mode themselves; the
        // dispatcher (which DOES hold &self) enforces it via the tool's
        // own `cloud` flag. Scope the lock so the guard drops before any
        // handler `.await` — holding routing_mode across a handler could
        // deadlock set_routing_mode.
        let local_only = self.routing_mode.lock().await.as_str() == "local-only";

        // Phase 2.1: dispatcher reads from `tools::TOOLS`. Adding a new
        // tool means one entry in tools.rs; the dispatcher and the
        // tools/list response both pick it up automatically.
        let result = match crate::tools::find(tool_name) {
            Some(t) if local_only && t.cloud => format!(
                "error: routing mode is 'local-only' — cloud tool '{}' refused. \
                 Call set_routing_mode(mode='auto') to re-enable.",
                tool_name
            ),
            Some(t) => match &t.handler {
                crate::tools::HandlerKind::Stateful(f) => f(self, args).await,
                crate::tools::HandlerKind::Free(f) => f(args).await,
            },
            // ADR 0023: not a built-in — try the module-tool registry
            // (generate_image, text_to_speech, future jart tools), dispatched
            // over ToolCtx. A module tool flagged `cloud` gets the SAME
            // local-only gate as a built-in cloud tool.
            None => match lamu_core::tools_ext::find_handler(tool_name) {
                Some((_, cloud)) if local_only && cloud => format!(
                    "error: routing mode is 'local-only' — cloud tool '{}' refused. \
                     Call set_routing_mode(mode='auto') to re-enable.",
                    tool_name
                ),
                Some((h, _)) => h(self as &dyn lamu_core::tools_ext::ToolCtx, args).await,
                None => format!("Unknown tool: {}", tool_name),
            },
        };

        // Heuristic: handlers prefix error responses with "error:" or
        // "Unknown tool:". Surface that as MCP `isError: true` so
        // clients can branch on tool failure programmatically.
        let lower = result.trim_start().to_lowercase();
        let is_error = lower.starts_with("error:")
            || lower.starts_with("unknown tool:")
            || lower.starts_with("missing prompt");

        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{ "type": "text", "text": result }],
                "isError": is_error,
            }
        })
    }

    pub(crate) async fn get_or_create_queue(&self, model_name: &str) -> Arc<RequestQueue<()>> {
        let mut map = self.queues.lock().await;
        if let Some(q) = map.get(model_name) {
            return q.clone();
        }
        let q = Arc::new(RequestQueue::<()>::new(self.queue_strategy, self.queue_concurrency));
        map.insert(model_name.to_string(), q.clone());
        q
    }

}

/// ADR 0023: the server's implementation of what module tools (generate_image,
/// future tts/jart) need — modality lookup, load trigger, loaded port.
#[async_trait::async_trait]
impl lamu_core::tools_ext::ToolCtx for LamuMcpServer {
    fn model_modality(&self, model: &str) -> Option<lamu_core::types::Modality> {
        self.state.lock().entries.get(model).map(|e| e.modality)
    }
    async fn ensure_loaded(&self, model: &str) -> Result<String, ToolCtxError> {
        // Bridge: handle_load_model still speaks the legacy wire convention
        // ("error:"-prefixed string). Convert at the seam (ADR 0027).
        let s = self.handle_load_model(json!({ "name": model })).await;
        if lamu_core::tools_ext::is_wire_error(&s) {
            Err(ToolCtxError::Load(ToolCtxError::strip_wire_prefix(&s).to_string()))
        } else {
            Ok(s)
        }
    }
    fn loaded_port(&self, model: &str) -> Option<u16> {
        self.state.lock().scheduler.get_loaded(model).map(|m| m.port)
    }
    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String, ToolCtxError> {
        // Route like `council`: a model present in the local registry goes
        // through the queued/scheduled local path; anything else is treated as
        // a cloud model. `None` keeps the summarization defaults (low temp,
        // bounded length) that every short-summary caller relies on; long-form
        // callers override per call (trait contract).
        let args = json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens
                .unwrap_or(lamu_core::tools_ext::GENERATE_DEFAULT_MAX_TOKENS),
            "temperature": temperature
                .unwrap_or(lamu_core::tools_ext::GENERATE_DEFAULT_TEMPERATURE),
        });
        // Legacy-wire bridge (ADR 0027): handle_query/handle_cloud_query still
        // return "error:"-prefixed strings; convert at the seam.
        let to_result = |s: String| -> Result<String, ToolCtxError> {
            if lamu_core::tools_ext::is_wire_error(&s) {
                Err(ToolCtxError::Generate(ToolCtxError::strip_wire_prefix(&s).to_string()))
            } else {
                Ok(s)
            }
        };
        let is_local = self.state.lock().entries.contains_key(model);
        if !is_local {
            // Enforce routing at the capability seam (the trait contract promises
            // generate honors routing mode). The dispatcher's static per-tool
            // `cloud` flag can't express "depends on the model arg" — without
            // this check a module tool flagged cloud:false (e.g. `research` with
            // its default cloud summary model) would leak to the cloud under
            // local-only. Frontends that drive handlers directly (cmd_research)
            // inherit the gate here for free.
            if self.routing_mode.lock().await.as_str() == "local-only" {
                return Err(ToolCtxError::Generate(format!(
                    "routing mode is 'local-only' — cloud model '{model}' refused. \
                     Call set_routing_mode(mode='auto') to re-enable."
                )));
            }
            return to_result(crate::cloud::handle_cloud_query(args).await);
        }
        let out = self.handle_query(args.clone()).await;
        // A reasoning model can emit an all-`<think>` completion and leave the
        // visible `content` empty (handle_query strips the reasoning). Retry once
        // with thinking disabled to force visible output. Non-empty output
        // (including an error) passes straight through.
        if !out.trim().is_empty() {
            return to_result(out);
        }
        let mut retry = args;
        retry["enable_thinking"] = json!(false);
        to_result(self.handle_query(retry).await)
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ToolCtxError> {
        let embed_err = |m: String| ToolCtxError::Embed(m);
        if texts.is_empty() {
            return Ok(vec![]);
        }
        // Resolve the embedding model from the registry (capability Embedding)
        // and clone the pooled HTTP client (reuse the connection pool + the
        // configured timeout) — both under one short lock, dropped before await.
        // min() (not find over HashMap order) so the pick is DETERMINISTIC and
        // always matches the model name `register_local_embedder` stamped into
        // the ADR 0030 identity.
        let (name, client) = {
            let st = self.state.lock();
            let name = st
                .entries
                .values()
                .filter(|e| e.capabilities.contains(&lamu_core::types::Capability::Embedding))
                .map(|e| e.name.clone())
                .min();
            (name, st.client.clone())
        }; // parking_lot guard dropped before any await
        let Some(name) = name else {
            return Err(embed_err(
                "no embedding model in registry — add one with capability 'embedding'".into(),
            ));
        };
        // Ensure loaded + resolve its port.
        if let Err(e) = self.ensure_loaded(&name).await {
            return Err(embed_err(format!("load embedding model '{name}': {e}")));
        }
        let port = self
            .loaded_port(&name)
            .ok_or_else(|| embed_err(format!("embedding model '{name}' is not on a live port")))?;
        let url = format!("http://localhost:{port}/v1/embeddings");
        let resp = client
            .post(&url)
            .json(&json!({ "model": name, "input": texts }))
            .send()
            .await
            .map_err(|e| embed_err(format!("embeddings backend: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            return Err(embed_err(format!(
                "embeddings backend {s}: {}",
                resp.text().await.unwrap_or_default()
            )));
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| embed_err(format!("embeddings non-JSON: {e}")))?;
        let data = v
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| embed_err("embeddings response missing data[]".to_string()))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let emb = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| embed_err("embeddings item missing embedding[]".to_string()))?;
            out.push(emb.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect());
        }
        Ok(out)
    }
}

/// ADR 0030: [`lamu_memory::embedder::Embedder`] over the SERVER's
/// embed path. Holds a (shallow) clone of the server — every field is
/// an Arc, so the adapter shares the live scheduler/registry/queues —
/// and delegates to `ToolCtx::embed` (ensure-load the registry's
/// embedding model, POST its port's `/v1/embeddings`).
///
/// `model` is the registry embedding model's name resolved at startup;
/// `dims` is probed on the first successful embed (0 until then —
/// harmless: the storage layer records dims from the actual vectors,
/// never from the identity).
struct McpServerEmbedder {
    server: LamuMcpServer,
    model: String,
    dims: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl lamu_memory::embedder::Embedder for McpServerEmbedder {
    fn identity(&self) -> lamu_memory::embedder::EmbedderId {
        lamu_memory::embedder::EmbedderId {
            model: self.model.clone(),
            dims: self.dims.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        use lamu_core::tools_ext::ToolCtx;
        let mut out = Vec::with_capacity(texts.len());
        // Chunk so a big index_repo doesn't ship one giant payload to
        // the local backend.
        for chunk in texts.chunks(64) {
            let vecs = self
                .server
                .embed(chunk)
                .await
                .map_err(|e| anyhow::anyhow!("local embed: {e}"))?;
            out.extend(vecs);
        }
        if out.len() != texts.len() {
            return Err(anyhow::anyhow!(
                "local embed count mismatch: requested {}, got {}",
                texts.len(),
                out.len()
            ));
        }
        if let Some(first) = out.first() {
            self.dims
                .store(first.len(), std::sync::atomic::Ordering::Relaxed);
        }
        Ok(out)
    }
}

fn initialize_response(id: Option<Value>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": "lamu", "version": "2.0" }
        }
    })
}


fn tools_list_response(id: Option<Value>) -> Value {
    // Phase 2.1: iterate the single-source tool catalog. Adding a
    // new tool to crate::tools::TOOLS shows up here automatically.
    let mut tools: Vec<Value> = crate::tools::TOOLS.iter()
        .map(|t| t.to_list_entry())
        .collect();
    // ADR 0023: append module-contributed tools (generate_image, …).
    tools.extend(lamu_core::tools_ext::list_entries());

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "tools": tools }
    })
}


// ── Sandboxed file write (Phase 6.1) ────────────────────────────────
// Wraps lamu_core::sandbox::journal::safe_write so any agent
// modification gets recorded for `lamu rollback <session>`.
//
// Path scoping: caller passes a relative path; it's resolved against
// the lamu-mcp process cwd. Absolute paths and any '..' segments are
// refused so the call cannot escape cwd. Combined with the
// validate_session_id allowlist on the journal side, an attacker
// controlling the MCP arguments can't write outside cwd or escape the
// journal directory.

pub(crate) async fn handle_write_file(args: Value) -> String {
    let path_str = args["path"].as_str().unwrap_or("");
    let content = args["content"].as_str().unwrap_or("");
    let session_id = args["session_id"].as_str().unwrap_or("");

    if path_str.is_empty() {
        return "error: path is required".into();
    }
    if session_id.is_empty() {
        return "error: session_id is required".into();
    }

    let rel = std::path::PathBuf::from(path_str);
    if rel.is_absolute() {
        return format!(
            "error: absolute paths refused — pass a path relative to lamu-mcp's cwd: {}",
            path_str
        );
    }
    if rel.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return format!("error: '..' refused in path: {}", path_str);
    }
    let cwd = match std::env::current_dir() {
        Ok(c) => c,
        Err(e) => return format!("error: getcwd: {}", e),
    };
    let abs = cwd.join(&rel);

    // Symlink-escape guard: canonicalize the *parent* (which must exist)
    // and require it to live under the canonicalized cwd. This catches
    // a relative path like `link/file` where `link` is a symlink
    // pointing outside cwd — the `..` block above doesn't see those.
    // The leaf filename itself can be new (doesn't need to exist).
    let cwd_canon = match cwd.canonicalize() {
        Ok(p) => p,
        Err(e) => return format!("error: canonicalize cwd: {}", e),
    };
    let parent_to_check = abs.parent().unwrap_or(&abs);
    if let Err(e) = std::fs::create_dir_all(parent_to_check) {
        return format!("error: prepare parent dir: {}", e);
    }
    let parent_canon = match parent_to_check.canonicalize() {
        Ok(p) => p,
        Err(e) => return format!("error: canonicalize parent: {}", e),
    };
    if !parent_canon.starts_with(&cwd_canon) {
        return format!(
            "error: resolved path escapes cwd via symlink: parent {} not under {}",
            parent_canon.display(),
            cwd_canon.display()
        );
    }

    let journal = match lamu_core::sandbox::journal::Journal::open(session_id) {
        Ok(j) => j,
        Err(e) => return format!("error: open journal: {}", e),
    };

    if let Err(e) = lamu_core::sandbox::journal::safe_write(&journal, &abs, content.as_bytes()) {
        return format!("error: write_file: {}", e);
    }

    format!(
        "wrote {} bytes to {} (journaled to session {})",
        content.len(),
        rel.display(),
        session_id
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::parse_capability;
    use lamu_core::types::Capability;
    use serde_json::json;


    #[test]
    fn parse_capability_known() {
        assert_eq!(parse_capability("chat"), Some(Capability::Chat));
        assert_eq!(parse_capability("code"), Some(Capability::Code));
        assert_eq!(parse_capability("reasoning"), Some(Capability::Reasoning));
        assert_eq!(parse_capability("routing"), Some(Capability::Routing));
        assert_eq!(parse_capability("vision"), Some(Capability::Vision));
        assert_eq!(parse_capability("long_context"), Some(Capability::LongContext));
    }

    #[test]
    fn parse_capability_unknown_returns_none() {
        assert_eq!(parse_capability("totally_fake"), None);
        assert_eq!(parse_capability(""), None);
    }

    #[test]
    fn initialize_response_shape() {
        let id = Some(json!(7));
        let resp = initialize_response(id.clone());
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 7);
        assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(resp["result"]["serverInfo"]["name"], "lamu");
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_response_includes_all_tools() {
        let resp = tools_list_response(Some(json!(1)));
        let tools = resp["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        for required in [
            "query", "plan_query", "list_models", "load_model",
            "unload_model", "vram_status", "scan_models", "queue_status",
            "cloud_query", "list_cloud_models",
            "review_commit", "review_diff", "set_routing_mode", "routing_status",
            "parallel_query", "write_file",
            "remember", "recall_memory", "consolidate_memory",
        ] {
            assert!(names.contains(&required), "missing tool: {}", required);
        }
    }


    #[test]
    fn tools_list_response_query_requires_prompt() {
        let resp = tools_list_response(None);
        let query = resp["result"]["tools"].as_array().unwrap()
            .iter().find(|t| t["name"] == "query").unwrap();
        let required = query["inputSchema"]["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "prompt"));
    }

    // Phase 6.1 — write_file MCP tool. The journal scoping + path
    // validation are covered in lamu-core; these tests pin the tool's
    // input-shape rejections (which run before the journal sees
    // anything). Tests that mutate cwd serialize via WRITE_FILE_CWD_LOCK
    // since std::env::set_current_dir is process-global.
    //
    // Use tokio::sync::Mutex (not std::sync::Mutex) so the guard can be
    // safely held across the .await on handle_write_file without the
    // sync-mutex-across-await footgun.

    use tokio::sync::Mutex as TokioMutex;
    use std::sync::OnceLock;
    static WRITE_FILE_CWD_LOCK: OnceLock<TokioMutex<()>> = OnceLock::new();
    fn cwd_lock() -> &'static TokioMutex<()> {
        WRITE_FILE_CWD_LOCK.get_or_init(|| TokioMutex::new(()))
    }

    #[tokio::test]
    async fn write_file_rejects_absolute_path() {
        let r = handle_write_file(json!({
            "path": "/etc/passwd",
            "content": "x",
            "session_id": "test",
        })).await;
        assert!(r.starts_with("error:"), "got: {r}");
        assert!(r.contains("absolute"), "got: {r}");
    }

    #[tokio::test]
    async fn write_file_rejects_parent_dir_segment() {
        let r = handle_write_file(json!({
            "path": "subdir/../../escape.txt",
            "content": "x",
            "session_id": "test",
        })).await;
        assert!(r.starts_with("error:"), "got: {r}");
        assert!(r.contains(".."), "got: {r}");
    }

    #[tokio::test]
    async fn write_file_rejects_missing_path_or_session() {
        let no_path = handle_write_file(json!({
            "content": "x",
            "session_id": "test",
        })).await;
        assert!(no_path.starts_with("error: path"), "got: {no_path}");

        let no_session = handle_write_file(json!({
            "path": "ok.txt",
            "content": "x",
        })).await;
        assert!(no_session.starts_with("error: session_id"), "got: {no_session}");
    }

    #[tokio::test]
    async fn write_file_rejects_bad_session_id() {
        let r = handle_write_file(json!({
            "path": "ok.txt",
            "content": "x",
            "session_id": "../escape",
        })).await;
        // The journal validator's error string starts with "session id …"
        // and bubbles up through the "error: open journal: …" wrapper.
        assert!(r.starts_with("error: open journal:"), "got: {r}");
        assert!(r.contains("session id"), "got: {r}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_file_rejects_directory_symlink_escape() {
        // Symlinked subdir attack: cwd/escape -> outside_dir/.
        // write_file("escape/pwned.txt", ...) must refuse.
        let _g = cwd_lock().lock().await;
        let outside = tempfile::tempdir().unwrap();
        let inside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), inside.path().join("escape")).unwrap();

        let prev_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(inside.path()).unwrap();
        let r = handle_write_file(json!({
            "path": "escape/pwned.txt",
            "content": "owned",
            "session_id": "test-symlink-escape",
        })).await;
        std::env::set_current_dir(prev_cwd).unwrap();

        assert!(r.starts_with("error:"), "got: {r}");
        assert!(r.contains("escapes cwd via symlink"), "got: {r}");
        assert!(!outside.path().join("pwned.txt").exists(), "symlink escape wrote anyway");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_file_rejects_leaf_symlink_escape() {
        // Symlinked leaf attack: cwd/sneaky -> /tmp/outside/target.
        // The parent canonicalizes to cwd (passes the parent check),
        // but std::fs::write would follow the leaf symlink. Defense
        // lives in lamu_core::sandbox::journal::safe_write, which
        // refuses to follow a leaf symlink before writing.
        let _g = cwd_lock().lock().await;
        let outside = tempfile::tempdir().unwrap();
        let outside_target = outside.path().join("target.txt");
        std::fs::write(&outside_target, b"original").unwrap();
        let inside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(&outside_target, inside.path().join("sneaky")).unwrap();

        let prev_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(inside.path()).unwrap();
        let r = handle_write_file(json!({
            "path": "sneaky",
            "content": "pwned",
            "session_id": "test-leaf-symlink",
        })).await;
        std::env::set_current_dir(prev_cwd).unwrap();

        assert!(r.starts_with("error:"), "got: {r}");
        assert!(r.contains("symlink"), "got: {r}");
        // Confirm the symlink target is unchanged.
        assert_eq!(std::fs::read(&outside_target).unwrap(), b"original");
    }
}
