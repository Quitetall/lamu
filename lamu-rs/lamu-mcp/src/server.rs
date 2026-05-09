//! MCP stdio server. Hand-rolled JSON-RPC.
//! Direct port of `lamu/mcp/server.py::LamuMcpServer`.
//!
//! Protocol: line-delimited JSON-RPC 2.0 over stdin/stdout.
//! Logs to stderr. Tools dispatched via `tools::*`.

use anyhow::Result;
use lamu_core::backends::{make_backend, Backend};
use lamu_core::health::HealthRegistry;
use lamu_core::observability::{emit, new_trace_id, trace_id_from_traceparent};
use lamu_core::queue::{QueueRequest, RequestQueue, Strategy as QueueStrategy};
use lamu_core::reasoning::get_extractor;
use lamu_core::registry::{load_registry, scan_directory, write_registry};
use lamu_core::router::Router;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{BackendType, Capability, ModelEntry};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tracing::warn;

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
    /// load/unload + routes to .generate(). Wrapped in
    /// Arc<TokioMutex<…>> so handle_query can clone the Arc out of the
    /// state lock and serialize per-backend access without re-locking
    /// the whole ServerState across an await.
    pub backends: HashMap<String, Arc<AsyncMutex<Box<dyn Backend>>>>,
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

        Ok(Self {
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
                std::env::var("LAMU_ROUTING_MODE").unwrap_or_else(|_| "auto".to_string())
            )),
        })
    }

    pub async fn run(self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin).lines();
        let mut writer = stdout;

        // Startup banner stays as explicit eprintln (not tracing) so it's
        // visible regardless of RUST_LOG. Some callers wait for this string
        // before sending requests.
        eprintln!("LAMU MCP server ready (rust)");

        loop {
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

        Ok(())
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

        // Phase 2.1: dispatcher reads from `tools::TOOLS`. Adding a new
        // tool means one entry in tools.rs; the dispatcher and the
        // tools/list response both pick it up automatically.
        let result = match crate::tools::find(tool_name) {
            Some(t) => match &t.handler {
                crate::tools::HandlerKind::Stateful(f) => f(self, args).await,
                crate::tools::HandlerKind::Free(f) => f(args).await,
            },
            None => format!("Unknown tool: {}", tool_name),
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

    async fn get_or_create_queue(&self, model_name: &str) -> Arc<RequestQueue<()>> {
        let mut map = self.queues.lock().await;
        if let Some(q) = map.get(model_name) {
            return q.clone();
        }
        let q = Arc::new(RequestQueue::<()>::new(self.queue_strategy, self.queue_concurrency));
        map.insert(model_name.to_string(), q.clone());
        q
    }

    pub(crate) async fn handle_query(&self, args: Value) -> String {
        let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        if prompt.is_empty() {
            return "missing prompt".into();
        }

        // Enforce routing mode — refuse local queries when cloud-only.
        {
            let mode = self.routing_mode.lock().await.clone();
            if mode == "cloud-only" {
                return "error: routing mode is 'cloud-only' — local queries refused. Call set_routing_mode(mode='auto') to re-enable, or use cloud_query instead.".into();
            }
        }

        let model = args.get("model").and_then(|v| v.as_str());
        let caps_raw = args.get("capabilities").and_then(|v| v.as_array());
        let system = args.get("system").and_then(|v| v.as_str()).unwrap_or("");
        let max_tokens = args.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(16384) as u32;
        let temperature = args.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.7) as f32;
        let include_reasoning = args.get("include_reasoning").and_then(|v| v.as_bool()).unwrap_or(false);
        let priority = args.get("priority").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let origin = args.get("origin").and_then(|v| v.as_str()).unwrap_or("anonymous").to_string();

        // Trace ID: accept W3C traceparent from `_meta`, else generate.
        let trace_id = args
            .get("_meta")
            .and_then(|m| m.get("traceparent"))
            .and_then(|v| v.as_str())
            .and_then(trace_id_from_traceparent)
            .unwrap_or_else(new_trace_id);

        emit(
            "mcp_query_start",
            Some(&trace_id),
            json!({
                "model_hint": model,
                "capabilities": caps_raw,
                "origin": origin,
                "priority": priority,
                "prompt_len": prompt.len(),
            }),
        );

        let caps: Vec<Capability> = caps_raw
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str())
                .filter_map(parse_capability)
                .collect())
            .unwrap_or_default();
        let caps_opt = if caps.is_empty() { None } else { Some(caps.as_slice()) };

        // Route + collect target info under lock
        let (port, model_name, marker, client) = {
            let st = self.state.lock();
            let decision = st.router.route(&st.scheduler, model, caps_opt, Some(st.health.all()));

            if decision.model_name.is_empty() {
                return format!("No model available: {}", decision.reason);
            }
            if !decision.loaded {
                return format!(
                    "Model '{}' not loaded. Would need to load (evicting: {:?}). \
                     Use load_model first or query a loaded model.",
                    decision.model_name, decision.would_evict
                );
            }

            let Some(loaded) = st.scheduler.get_loaded(&decision.model_name) else {
                return "internal: lost loaded model".into();
            };
            let port = loaded.port;
            let entry = st.entries.get(&decision.model_name).cloned();
            let marker = entry.as_ref().and_then(|e| e.reasoning_marker.clone());
            let client = st.client.clone();
            (port, decision.model_name, marker, client)
        };

        // Mark used (separate lock acquisition)
        {
            let mut st = self.state.lock();
            st.scheduler.mark_used(&model_name);
        }

        // Build messages
        let mut messages = Vec::new();
        if !system.is_empty() {
            messages.push(json!({"role":"system","content":system}));
        }
        messages.push(json!({"role":"user","content":prompt}));

        let payload = json!({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        let url = format!("http://localhost:{}/v1/chat/completions", port);

        // Acquire queue slot before hitting backend
        let queue = self.get_or_create_queue(&model_name).await;
        let _guard = queue.enqueue(QueueRequest {
            payload: (),
            priority,
            enqueued_at: Instant::now(),
            origin,
        }).await;

        let resp = match client.post(&url).json(&payload).send().await {
            Ok(r) => r,
            Err(e) => {
                emit(
                    "mcp_query_failed",
                    Some(&trace_id),
                    json!({
                        "model": model_name,
                        "error_type": "request_send",
                        "error": format!("{e}"),
                    }),
                );
                return format!("Generation error: {}", e);
            }
        };
        let data: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                emit(
                    "mcp_query_failed",
                    Some(&trace_id),
                    json!({
                        "model": model_name,
                        "error_type": "json_decode",
                        "error": format!("{e}"),
                    }),
                );
                return format!("JSON decode error: {}", e);
            }
        };

        let msg = match data.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message")) {
            Some(m) => m,
            None => {
                emit(
                    "mcp_query_failed",
                    Some(&trace_id),
                    json!({"model": model_name, "error_type": "no_message"}),
                );
                return "no message in response".into();
            }
        };
        let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let reasoning_content = msg.get("reasoning_content").and_then(|v| v.as_str()).unwrap_or("");

        let extractor = get_extractor(marker);
        let (reasoning, content_clean) = if !reasoning_content.is_empty() {
            (reasoning_content.to_string(), content.to_string())
        } else {
            extractor.split(content)
        };

        let text = if include_reasoning && !reasoning.is_empty() {
            format!("**Reasoning:**\n{}\n\n**Answer:**\n{}", reasoning, content_clean)
        } else {
            content_clean.clone()
        };

        emit(
            "mcp_query_done",
            Some(&trace_id),
            json!({
                "model": model_name,
                "content_len": content_clean.len(),
                "reasoning_len": reasoning.len(),
            }),
        );

        if text.trim().is_empty() {
            format!("[Model thinking truncated — reasoning: {} chars]", reasoning.len())
        } else {
            text
        }
    }

    pub(crate) fn handle_plan_query(&self, args: Value) -> String {
        let model = args.get("model").and_then(|v| v.as_str());
        let caps_raw = args.get("capabilities").and_then(|v| v.as_array());
        let caps: Vec<Capability> = caps_raw
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str())
                .filter_map(parse_capability)
                .collect())
            .unwrap_or_default();
        let caps_opt = if caps.is_empty() { None } else { Some(caps.as_slice()) };

        let st = self.state.lock();
        let decision = st.router.route(&st.scheduler, model, caps_opt, Some(st.health.all()));
        serde_json::to_string_pretty(&json!({
            "would_route_to": decision.model_name,
            "reason": decision.reason,
            "loaded": decision.loaded,
            "would_evict": decision.would_evict,
        })).unwrap_or_else(|e| format!("serialize error: {}", e))
    }

    pub(crate) fn handle_list_models(&self) -> String {
        let st = self.state.lock();
        let mut lines = Vec::new();
        let mut names: Vec<&String> = st.entries.keys().collect();
        names.sort();
        for name in names {
            let entry = &st.entries[name];
            let loaded = st.scheduler.is_loaded(name);
            let status_glyph = if loaded { "🟢 loaded" } else { "⚪ available" };
            // Operator-curated tag glyph (defined on ModelStatus so the
            // match can never drift from the enum's variants).
            let tag = entry.status.glyph();
            let caps: Vec<&str> = entry.capabilities.iter().map(|c| match c {
                Capability::Chat => "chat",
                Capability::Code => "code",
                Capability::Reasoning => "reasoning",
                Capability::Routing => "routing",
                Capability::Vision => "vision",
                Capability::LongContext => "long_context",
            }).collect();
            let mut line = format!(
                "{} {}{} ({}B {}, {}MB, [{}])",
                status_glyph, tag, name, entry.params_b, entry.quant, entry.vram_mb, caps.join(", ")
            );
            if !entry.notes.is_empty() {
                line.push_str(&format!("\n     — {}", entry.notes));
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    pub(crate) async fn handle_load_model(&self, args: Value) -> String {
        let name = match args.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => return "error: missing 'name' argument".into(),
        };

        // Atomic plan-and-reserve: hold the state lock across (a) entry
        // lookup, (b) plan_load, (c) mark_loading. Without this,
        // concurrent load_model calls could both pass the is_loaded check
        // and both spawn a backend on the same port.
        // Also pick a name-resolution mode: exact match wins; otherwise
        // require unique substring match. Ambiguous matches return an
        // error rather than silently picking one.
        let (entry, to_evict, evict_backends) = {
            let mut st = self.state.lock();

            // 1. Resolve name: exact > unique-substring > error.
            let entry: ModelEntry = if let Some(e) = st.entries.get(&name) {
                e.clone()
            } else {
                let candidates: Vec<&ModelEntry> = st.entries.values()
                    .filter(|e| e.name.contains(&name) || name.contains(e.name.as_str()))
                    .collect();
                match candidates.len() {
                    0 => return format!(
                        "error: model '{}' not found in registry. Run scan_models.",
                        name
                    ),
                    1 => candidates[0].clone(),
                    n => {
                        let names: Vec<String> = candidates.iter().map(|e| e.name.clone()).collect();
                        return format!(
                            "error: model '{}' is ambiguous ({} matches: {}). Use the exact name.",
                            name, n, names.join(", ")
                        );
                    }
                }
            };

            if st.scheduler.is_loaded(&entry.name) {
                return format!("Model '{}' already loaded.", entry.name);
            }
            let (can, evict) = st.scheduler.plan_load(&entry);
            if !can {
                return format!(
                    "error: cannot fit '{}' ({}MB) in VRAM. Insufficient space.",
                    entry.name, entry.vram_mb
                );
            }

            // Mark loading INSIDE the lock so no concurrent caller picks
            // up the same plan. evict_backends carries Arc<Mutex<Box<dyn
            // Backend>>> handles we'll unload outside the state lock.
            let mut evict_backends: Vec<(String, Arc<AsyncMutex<Box<dyn Backend>>>)> = Vec::new();
            for evict_name in &evict {
                if let Some(b) = st.backends.remove(evict_name) {
                    evict_backends.push((evict_name.clone(), b));
                }
                st.scheduler.mark_unloaded(evict_name);
                st.health.drop(evict_name);
            }
            st.scheduler.mark_loading(entry.clone());

            (entry, evict, evict_backends)
        };

        // Phase 6.3: route eviction through Backend::unload so per-impl
        // cleanup (megakernel/dflash sigterm semantics) lives in the
        // backend, not lamu-mcp. The unload guard is bounded by a 30s
        // timeout so a stuck backend can't hang the MCP server.
        for (evict_name, backend_arc) in evict_backends {
            let mut backend = backend_arc.lock().await;
            match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                backend.unload(),
            ).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => warn!("load_model: unload({}) errored: {}", evict_name, e),
                Err(_) => warn!(
                    "load_model: unload({}) timed out — leaving zombie", evict_name
                ),
            }
        }
        // Settle period for VRAM to actually drop after kill.
        if !to_evict.is_empty() {
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }

        // Pick port
        let port: u16 = {
            let st = self.state.lock();
            if st.scheduler.loaded_models().is_empty() {
                lamu_core::config::PORT_MAIN
            } else {
                lamu_core::config::PORT_SIDECAR
            }
        };

        // Construct the right Backend for this entry. The Backend impl
        // owns spawn + health-poll + warmup — lamu-mcp doesn't manage
        // that lifecycle anymore. make_backend dispatches on
        // entry.backend (LlamaCpp / Megakernel / Dflash).
        let mut backend: Box<dyn Backend> = match make_backend(&entry) {
            Ok(b) => b,
            Err(e) => {
                let mut st = self.state.lock();
                st.scheduler.mark_unloaded(&entry.name);
                return format!("error: make_backend: {}", e);
            }
        };
        {
            let mut st = self.state.lock();
            st.scheduler.mark_loading(entry.clone());
        }

        let pid = match backend.load(&entry, port).await {
            Ok(pid) => pid,
            Err(e) => {
                let mut st = self.state.lock();
                st.scheduler.mark_unloaded(&entry.name);
                st.health.drop(&entry.name);
                return format!("error: load failed: {}", e);
            }
        };

        // Healthy + warmed up by the time backend.load returned. Confirm
        // load + insert into backends map.
        let vram = {
            let st = self.state.lock();
            let pids = st.scheduler.query_gpu_pids();
            pids.iter()
                .find(|(p, _)| *p == pid)
                .map(|(_, m)| *m)
                .unwrap_or(entry.vram_mb)
        };
        {
            let mut st = self.state.lock();
            let _ = st.scheduler.confirm_loaded(&entry.name, pid, port, vram);
            st.health.get_or_create(&entry.name).record_success();
            st.backends.insert(entry.name.clone(), Arc::new(AsyncMutex::new(backend)));
        }
        let evict_msg = if to_evict.is_empty() {
            String::new()
        } else {
            format!(" (evicted: {:?})", to_evict)
        };
        format!("Loaded '{}' on :{} ({}MB VRAM){}", entry.name, port, vram, evict_msg)
    }

    pub(crate) async fn handle_unload_model(&self, args: Value) -> String {
        let name = match args.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => return "error: missing 'name' argument".into(),
        };

        // Resolve under lock, take the Child handle out, release lock,
        // THEN wait. Avoids holding the parking_lot lock across an
        // await point.
        let dead = {
            let mut st = self.state.lock();
            let target: Option<String> = st.scheduler.loaded_models().iter()
                .find(|m| m.entry.name.contains(&name) || name.contains(m.entry.name.as_str()))
                .map(|m| m.entry.name.clone());
            let Some(target) = target else {
                return format!("Model '{}' not loaded.", name);
            };
            let backend = st.backends.remove(&target);
            st.scheduler.mark_unloaded(&target);
            st.health.drop(&target);
            (target, backend)
        };
        let (target, backend) = dead;
        if let Some(backend_arc) = backend {
            let mut b = backend_arc.lock().await;
            match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                b.unload(),
            ).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => warn!("unload({}): backend errored: {}", target, e),
                Err(_) => warn!("unload({}): timeout — leaving zombie", target),
            }
        }
        format!("Unloaded '{}'. VRAM freed.", target)
    }

    pub(crate) fn handle_vram_status(&self) -> String {
        let st = self.state.lock();
        let budget = st.scheduler.budget();
        let mut lines = vec![
            format!("VRAM: {}/{} MB ({} MB free)", budget.used_mb, budget.total_mb, budget.free_mb),
            format!("Available for models: {} MB", budget.available_mb),
            "Loaded:".into(),
        ];
        if budget.loaded_models.is_empty() {
            lines.push("  (none)".into());
        } else {
            for (name, vram) in &budget.loaded_models {
                lines.push(format!("  {}: {} MB", name, vram));
            }
        }
        lines.join("\n")
    }

    pub(crate) async fn handle_queue_status(&self) -> String {
        let strategy = match self.queue_strategy {
            QueueStrategy::Fifo => "fifo",
            QueueStrategy::Lifo => "lifo",
            QueueStrategy::Priority => "priority",
        };
        let mut lines = vec![
            format!("Strategy: {} (concurrency={})", strategy, self.queue_concurrency),
            "Per-model queue depth:".into(),
        ];
        let map = self.queues.lock().await;
        if map.is_empty() {
            lines.push("  (no queues active)".into());
        } else {
            for (name, q) in map.iter() {
                let depth = q.depth().await;
                lines.push(format!("  {}: {} pending", name, depth));
            }
        }
        lines.join("\n")
    }

    pub(crate) async fn handle_set_routing_mode(&self, args: Value) -> String {
        let mode = args["mode"].as_str().unwrap_or("auto").to_string();
        if !matches!(mode.as_str(), "auto" | "local-only" | "cloud-only") {
            return format!("error: mode must be 'auto', 'local-only', or 'cloud-only' (got '{}')", mode);
        }

        // Hold the routing-mode lock for the whole transition. Once mode
        // is set to cloud-only, handle_query refuses new local requests,
        // so no concurrent load_model can race in while we drain.
        let mut current = self.routing_mode.lock().await;
        let old = current.clone();
        *current = mode.clone();

        // cloud-only → drain backends + scheduler atomically inside the
        // state lock, THEN unload outside the lock so the per-backend
        // unload doesn't hold the state lock for 30s.
        let mut freed = Vec::new();
        let mut to_unload: Vec<(String, Arc<AsyncMutex<Box<dyn Backend>>>)> = Vec::new();
        if mode == "cloud-only" {
            let mut st = self.state.lock();
            let names: Vec<String> = st.scheduler.loaded_models()
                .iter().map(|m| m.entry.name.clone()).collect();
            for n in &names {
                if let Some(b) = st.backends.remove(n) {
                    to_unload.push((n.clone(), b));
                }
                st.scheduler.mark_unloaded(n);
                freed.push(n.clone());
            }
            drop(st);
        }
        // Routing mode still locked; release before any await on the
        // backend unload so other RPCs aren't blocked while llama-server
        // tears down.
        drop(current);

        for (name, backend_arc) in to_unload {
            let mut b = backend_arc.lock().await;
            match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                b.unload(),
            ).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => warn!("set_routing_mode: unload({}) errored: {}", name, e),
                Err(_) => warn!(
                    "set_routing_mode: unload({}) timed out after 30s — leaving zombie", name
                ),
            }
        }

        let mut msg = format!("routing mode: {} → {}", old, mode);
        if !freed.is_empty() {
            msg.push_str(&format!("\nfreed VRAM by unloading: {}", freed.join(", ")));
        }
        msg
    }

    pub(crate) async fn handle_routing_status(&self) -> String {
        let mode = self.routing_mode.lock().await.clone();
        let st = self.state.lock();
        let (used, total) = st.scheduler.query_vram();
        let loaded: Vec<String> = st.scheduler.loaded_models().iter()
            .map(|m| format!("{} ({}MB)", m.entry.name, m.vram_actual_mb))
            .collect();
        let cloud_count = crate::cloud::load_cloud_models().len();
        format!(
            "routing mode: {}\nlocal: {} models loaded ({} MB / {} MB VRAM)\n  loaded: {}\ncloud: {} models in registry",
            mode,
            loaded.len(), used, total,
            if loaded.is_empty() { "(none)".into() } else { loaded.join(", ") },
            cloud_count
        )
    }

    /// Fan out a batch of tasks. Each task gets routed via either
    /// `handle_cloud_query` (if model name matches a cloud entry) or
    /// `handle_query` (local). Concurrency is capped per-model — see
    /// `provider_concurrency` for the per-provider table. Local
    /// concurrency is always 1.
    ///
    /// Returns a JSON-shaped text body (parseable by the caller) plus
    /// a human-readable summary header.
    pub(crate) async fn handle_parallel_query(&self, args: Value) -> String {
        let tasks_arr = match args["tasks"].as_array() {
            Some(a) if !a.is_empty() => a.clone(),
            _ => return "error: 'tasks' must be a non-empty array".into(),
        };
        let default_model = args["default_model"].as_str()
            .unwrap_or("deepseek-v4-flash").to_string();
        let default_system = args["default_system"].as_str().unwrap_or("").to_string();
        let user_max = args["max_concurrency"].as_u64().map(|n| n as usize);

        let cloud = crate::cloud::load_cloud_models();

        // Build per-(model) semaphores. Same-model tasks share one
        // semaphore so the cap actually limits in-flight requests.
        let mut sems: HashMap<String, Arc<Semaphore>> = HashMap::new();

        let mut prepared = Vec::with_capacity(tasks_arr.len());
        for (idx, t) in tasks_arr.iter().enumerate() {
            let prompt = t["prompt"].as_str().unwrap_or("").to_string();
            if prompt.is_empty() {
                prepared.push(Err(format!("task[{}]: empty prompt", idx)));
                continue;
            }
            let model = t["model"].as_str().unwrap_or(&default_model).to_string();
            let task_id = t["id"].as_str().map(String::from)
                .unwrap_or_else(|| format!("task{}", idx));
            let system = t["system"].as_str().unwrap_or(&default_system).to_string();
            let max_tokens = t["max_tokens"].as_u64().unwrap_or(8192);
            let temperature = t["temperature"].as_f64().unwrap_or(0.3);
            let include_reasoning = t["include_reasoning"].as_bool().unwrap_or(false);
            // thinking_enabled: pass through ONLY if the task supplies
            // an actual bool. Explicit null → fall back to per-model
            // heuristic (treat null same as omitted).
            let thinking_enabled_arg = t.get("thinking_enabled")
                .and_then(|v| v.as_bool())
                .map(Value::Bool);

            let is_cloud = cloud.iter().any(|m| m.name == model);
            let cap = if is_cloud {
                let provider_cap = crate::cloud::provider_concurrency(&model, &cloud);
                user_max.map(|u| u.min(provider_cap)).unwrap_or(provider_cap)
            } else {
                1 // local: always sequential per project policy
            };
            let sem_key = if is_cloud { model.clone() } else { format!("local:{}", model) };
            let sem = sems.entry(sem_key)
                .or_insert_with(|| Arc::new(Semaphore::new(cap)))
                .clone();

            let mut inner_args = json!({
                "model": model.clone(),
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "include_reasoning": include_reasoning,
            });
            if let Some(te) = thinking_enabled_arg {
                inner_args["thinking_enabled"] = te;
            }

            prepared.push(Ok((idx, task_id, model, is_cloud, sem, inner_args)));
        }

        // Spawn futures (all borrow self via &self lifetime; join_all
        // holds them in a single scope so no 'static needed).
        let t0 = std::time::Instant::now();
        let futs = prepared.into_iter().map(|p| async move {
            match p {
                Err(msg) => (0usize, "error".to_string(), "(unknown)".to_string(), false, msg, 0.0),
                Ok((idx, id, model, is_cloud, sem, args)) => {
                    let t_start = std::time::Instant::now();
                    let _permit = match sem.acquire().await {
                        Ok(p) => p,
                        Err(e) => return (idx, id, model, is_cloud,
                                          format!("error: semaphore: {}", e), 0.0),
                    };
                    let result = if is_cloud {
                        crate::cloud::handle_cloud_query(args).await
                    } else {
                        self.handle_query(args).await
                    };
                    let elapsed = t_start.elapsed().as_secs_f32();
                    (idx, id, model, is_cloud, result, elapsed)
                }
            }
        });

        let mut results: Vec<_> = futures_util::future::join_all(futs).await;
        results.sort_by_key(|(idx, _, _, _, _, _)| *idx);
        let total_wall = t0.elapsed().as_secs_f32();

        // Build a JSON-shaped body so callers can machine-parse, plus
        // a header readable by humans.
        let json_results: Vec<Value> = results.iter().map(|(idx, id, model, is_cloud, text, elapsed)| {
            json!({
                "idx": idx,
                "id": id,
                "model": model,
                "via": if *is_cloud { "cloud" } else { "local" },
                "elapsed_s": elapsed,
                "result": text,
            })
        }).collect();
        let body = json!({
            "total_tasks": results.len(),
            "wall_time_s": total_wall,
            "results": json_results,
        });
        let summary = format!(
            "=== parallel_query: {} task(s) in {:.1}s wall ===",
            results.len(), total_wall
        );
        format!("{}\n{}", summary, serde_json::to_string_pretty(&body).unwrap_or_default())
    }

    pub(crate) fn handle_scan(&self) -> String {
        let mut st = self.state.lock();
        let entries = match scan_directory(&st.models_dir) {
            Ok(e) => e,
            Err(e) => return format!("scan error: {}", e),
        };
        if let Err(e) = write_registry(&entries, &st.registry_path) {
            return format!("write error: {}", e);
        }
        st.entries = entries.iter().map(|e| (e.name.clone(), e.clone())).collect();
        st.router.update_registry(entries.clone());
        format!(
            "Scanned {}: {} models found. Registry updated.",
            st.models_dir.display(), entries.len()
        )
    }
}

fn parse_capability(s: &str) -> Option<Capability> {
    match s {
        "chat" => Some(Capability::Chat),
        "code" => Some(Capability::Code),
        "reasoning" => Some(Capability::Reasoning),
        "routing" => Some(Capability::Routing),
        "vision" => Some(Capability::Vision),
        "long_context" => Some(Capability::LongContext),
        _ => None,
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
    let tools: Vec<Value> = crate::tools::TOOLS.iter()
        .map(|t| t.to_list_entry())
        .collect();

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
