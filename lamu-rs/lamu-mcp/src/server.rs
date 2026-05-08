//! MCP stdio server. Hand-rolled JSON-RPC.
//! Direct port of `lamu/mcp/server.py::LamuMcpServer`.
//!
//! Protocol: line-delimited JSON-RPC 2.0 over stdin/stdout.
//! Logs to stderr. Tools dispatched via `tools::*`.

use anyhow::Result;
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
use tokio::process::Child;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};

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
    /// Owned child processes per loaded backend. Replaces the old
    /// `std::mem::forget(child)` + `libc::kill` pattern — `start_kill()`
    /// on these is the only path that ends a backend now.
    pub loaded_procs: HashMap<String, Child>,
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
                loaded_procs: HashMap::new(),
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

        eprintln!("LAMU MCP server ready (rust)");

        loop {
            let line = match reader.next_line().await {
                Ok(Some(l)) => l,
                Ok(None) => break,
                Err(e) => {
                    eprintln!("read error: {}", e);
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }

            let request: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("bad json: {}", e);
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

    async fn handle(&self, method: &str, params: Value, id: Option<Value>) -> Option<Value> {
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

        let result = match tool_name {
            "query" => self.handle_query(args).await,
            "plan_query" => self.handle_plan_query(args),
            "list_models" => self.handle_list_models(),
            "load_model" => self.handle_load_model(args).await,
            "unload_model" => self.handle_unload_model(args).await,
            "vram_status" => self.handle_vram_status(),
            "scan_models" => self.handle_scan(),
            "queue_status" => self.handle_queue_status().await,
            "cloud_query" => handle_cloud_query(args).await,
            "list_cloud_models" => handle_list_cloud_models(),
            "review_commit" => handle_review_commit(args).await,
            "review_diff" => handle_review_diff(args).await,
            "set_routing_mode" => self.handle_set_routing_mode(args).await,
            "routing_status" => self.handle_routing_status().await,
            "parallel_query" => self.handle_parallel_query(args).await,
            other => format!("Unknown tool: {}", other),
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

    async fn handle_query(&self, args: Value) -> String {
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

    fn handle_plan_query(&self, args: Value) -> String {
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

    fn handle_list_models(&self) -> String {
        let st = self.state.lock();
        let mut lines = Vec::new();
        let mut names: Vec<&String> = st.entries.keys().collect();
        names.sort();
        for name in names {
            let entry = &st.entries[name];
            let loaded = st.scheduler.is_loaded(name);
            let status = if loaded { "🟢 loaded" } else { "⚪ available" };
            let caps: Vec<&str> = entry.capabilities.iter().map(|c| match c {
                Capability::Chat => "chat",
                Capability::Code => "code",
                Capability::Reasoning => "reasoning",
                Capability::Routing => "routing",
                Capability::Vision => "vision",
                Capability::LongContext => "long_context",
            }).collect();
            lines.push(format!(
                "{} {} ({}B {}, {}MB, [{}])",
                status, name, entry.params_b, entry.quant, entry.vram_mb, caps.join(", ")
            ));
        }
        lines.join("\n")
    }

    async fn handle_load_model(&self, args: Value) -> String {
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
        let (entry, to_evict, evict_children) = {
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
            // up the same plan. evict_children carries owned Child handles
            // we must wait() outside the lock.
            let mut evict_children: Vec<(String, tokio::process::Child)> = Vec::new();
            for evict_name in &evict {
                if let Some(child) = st.loaded_procs.remove(evict_name) {
                    evict_children.push((evict_name.clone(), child));
                }
                st.scheduler.mark_unloaded(evict_name);
                st.health.drop(evict_name);
            }
            st.scheduler.mark_loading(entry.clone());

            (entry, evict, evict_children)
        };

        // Reap evicted children outside the lock — wait() with a timeout
        // so a stuck backend can't hang the entire MCP server.
        for (evict_name, mut child) in evict_children {
            let _ = child.start_kill();
            match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                child.wait()
            ).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => eprintln!(
                    "load_model: wait({}) errored: {}", evict_name, e
                ),
                Err(_) => eprintln!(
                    "load_model: wait({}) timed out — leaving zombie", evict_name
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

        // Build per-backend spawn command + health probe path.
        let (mut cmd, health_path, expect_status_ok, max_wait_secs) =
            match build_spawn_cmd(&entry, port).await {
                Ok(t) => t,
                Err(msg) => return msg,
            };
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        {
            let mut st = self.state.lock();
            st.scheduler.mark_loading(entry.clone());
        }

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let mut st = self.state.lock();
                st.scheduler.mark_unloaded(&entry.name);
                return format!("spawn failed: {}", e);
            }
        };
        let pid = child.id().unwrap_or(0);
        // Take ownership of the Child — kill is now `start_kill()` on the
        // stored value, no more libc::kill on a leaked PID.
        {
            let mut st = self.state.lock();
            st.loaded_procs.insert(entry.name.clone(), child);
        }

        // Health poll
        let client = self.state.lock().client.clone();
        let url = format!("http://localhost:{}{}", port, health_path);
        for _ in 0..max_wait_secs {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let healthy = if expect_status_ok {
                match client.get(&url).send().await {
                    Ok(r) => match r.json::<Value>().await {
                        Ok(j) => j.get("status").and_then(|v| v.as_str()) == Some("ok"),
                        Err(_) => false,
                    },
                    Err(_) => false,
                }
            } else {
                client.get(&url).send().await
                    .map(|r| r.status().is_success())
                    .unwrap_or(false)
            };
            if healthy {
                let mut st = self.state.lock();
                let pids = st.scheduler.query_gpu_pids();
                let vram = pids.iter()
                    .find(|(p, _)| *p == pid)
                    .map(|(_, m)| *m)
                    .unwrap_or(entry.vram_mb);
                let _ = st.scheduler.confirm_loaded(&entry.name, pid, port, vram);
                // Healthy from the moment it answered; supervisor restart
                // path will record_error on subsequent failures.
                st.health.get_or_create(&entry.name).record_success();
                let evict_msg = if to_evict.is_empty() {
                    String::new()
                } else {
                    format!(" (evicted: {:?})", to_evict)
                };
                return format!(
                    "Loaded '{}' on :{} ({}MB VRAM){}",
                    entry.name, port, vram, evict_msg
                );
            }
        }

        // Timeout — kill the stored Child and reap before returning.
        let dead_child = {
            let mut st = self.state.lock();
            let dead = st.loaded_procs.remove(&entry.name);
            st.scheduler.mark_unloaded(&entry.name);
            st.health.drop(&entry.name);
            dead
        };
        if let Some(mut child) = dead_child {
            let _ = child.start_kill();
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                child.wait()
            ).await;
        }
        format!("error: failed to load '{}' (timeout {}s)", entry.name, max_wait_secs)
    }

    async fn handle_unload_model(&self, args: Value) -> String {
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
            let child = st.loaded_procs.remove(&target);
            st.scheduler.mark_unloaded(&target);
            st.health.drop(&target);
            (target, child)
        };
        let (target, child) = dead;
        if let Some(mut child) = child {
            let _ = child.start_kill();
            match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                child.wait()
            ).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => eprintln!("unload({}): wait errored: {}", target, e),
                Err(_) => eprintln!("unload({}): timeout — leaving zombie", target),
            }
        }
        format!("Unloaded '{}'. VRAM freed.", target)
    }

    fn handle_vram_status(&self) -> String {
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

    async fn handle_queue_status(&self) -> String {
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

    async fn handle_set_routing_mode(&self, args: Value) -> String {
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

        // cloud-only → drain loaded_procs + scheduler atomically inside
        // the state lock, THEN kill outside the lock so wait() doesn't
        // hold the state lock for 30s.
        let mut freed = Vec::new();
        let mut to_kill: Vec<(String, Child)> = Vec::new();
        if mode == "cloud-only" {
            let mut st = self.state.lock();
            let names: Vec<String> = st.scheduler.loaded_models()
                .iter().map(|m| m.entry.name.clone()).collect();
            for n in &names {
                if let Some(p) = st.loaded_procs.remove(n) {
                    to_kill.push((n.clone(), p));
                }
                st.scheduler.mark_unloaded(n);
                freed.push(n.clone());
            }
            drop(st);
        }
        // Routing mode still locked; release before any await on the
        // child wait so other RPCs aren't blocked while llama-server
        // tears down.
        drop(current);

        for (name, mut p) in to_kill {
            let _ = p.start_kill();
            // Cap the wait — if llama-server ignores SIGTERM we move on
            // rather than hang the entire MCP server.
            match tokio::time::timeout(std::time::Duration::from_secs(30), p.wait()).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => eprintln!("set_routing_mode: wait({}) errored: {}", name, e),
                Err(_) => eprintln!("set_routing_mode: wait({}) timed out after 30s — leaving zombie", name),
            }
        }

        let mut msg = format!("routing mode: {} → {}", old, mode);
        if !freed.is_empty() {
            msg.push_str(&format!("\nfreed VRAM by unloading: {}", freed.join(", ")));
        }
        msg
    }

    async fn handle_routing_status(&self) -> String {
        let mode = self.routing_mode.lock().await.clone();
        let st = self.state.lock();
        let (used, total) = st.scheduler.query_vram();
        let loaded: Vec<String> = st.scheduler.loaded_models().iter()
            .map(|m| format!("{} ({}MB)", m.entry.name, m.vram_actual_mb))
            .collect();
        let cloud_count = load_cloud_models().len();
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
    async fn handle_parallel_query(&self, args: Value) -> String {
        let tasks_arr = match args["tasks"].as_array() {
            Some(a) if !a.is_empty() => a.clone(),
            _ => return "error: 'tasks' must be a non-empty array".into(),
        };
        let default_model = args["default_model"].as_str()
            .unwrap_or("deepseek-v4-flash").to_string();
        let default_system = args["default_system"].as_str().unwrap_or("").to_string();
        let user_max = args["max_concurrency"].as_u64().map(|n| n as usize);

        let cloud = load_cloud_models();

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

            let is_cloud = cloud.iter().any(|m| m.name == model);
            let cap = if is_cloud {
                let provider_cap = provider_concurrency(&model, &cloud);
                user_max.map(|u| u.min(provider_cap)).unwrap_or(provider_cap)
            } else {
                1 // local: always sequential per project policy
            };
            let sem_key = if is_cloud { model.clone() } else { format!("local:{}", model) };
            let sem = sems.entry(sem_key)
                .or_insert_with(|| Arc::new(Semaphore::new(cap)))
                .clone();

            let inner_args = json!({
                "model": model.clone(),
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "include_reasoning": include_reasoning,
            });

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
                        handle_cloud_query(args).await
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

    fn handle_scan(&self) -> String {
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
    let tools = vec![
        json!({
            "name": "query",
            "description": "Send prompt to local LLM. Routes by capabilities or explicit model. Queued per-model (FIFO default) so concurrent agents don't collide. Fast, free, uncensored.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "capabilities": {"type": "array", "items": {"type": "string"}},
                    "system": {"type": "string", "default": ""},
                    "max_tokens": {"type": "integer", "default": 16384},
                    "temperature": {"type": "number", "default": 0.7},
                    "include_reasoning": {"type": "boolean", "default": false},
                    "priority": {"type": "integer", "default": 0, "description": "Higher served first (priority strategy only)"},
                    "origin": {"type": "string", "default": "anonymous", "description": "Agent identifier for queue observability"},
                },
                "required": ["prompt"]
            }
        }),
        json!({
            "name": "plan_query",
            "description": "Dry-run: see which model WOULD handle a request without generating.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "capabilities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["prompt"]
            }
        }),
        json!({
            "name": "list_models",
            "description": "List all known models with load status and capabilities.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "load_model",
            "description": "Explicitly load a model onto GPU.",
            "inputSchema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        }),
        json!({
            "name": "unload_model",
            "description": "Unload a model from GPU.",
            "inputSchema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        }),
        json!({
            "name": "vram_status",
            "description": "Show current VRAM allocation.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "scan_models",
            "description": "Re-scan disk for new models.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "queue_status",
            "description": "Show per-model queue depth and scheduling strategy.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "cloud_query",
            "description": "Send prompt to a cloud model (DeepSeek V4, Claude, GLM, Kimi, Qwen-Max, etc.). Use this for tasks that need stronger reasoning than local, OR cheaper inference than the calling agent (e.g. Claude Code → DeepSeek V4 Flash for code generation at ~$0.07/M input, currently 75% off). Auto-routes via OpenAI/Anthropic format detection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "User prompt"},
                    "model": {"type": "string", "description": "Cloud model name from cloud-models.yaml (e.g. 'deepseek-v4-flash', 'claude-haiku-4-5'). Defaults to 'deepseek-v4-flash'.", "default": "deepseek-v4-flash"},
                    "system": {"type": "string", "description": "System prompt", "default": ""},
                    "max_tokens": {"type": "integer", "default": 8192},
                    "temperature": {"type": "number", "default": 0.3},
                    "include_reasoning": {"type": "boolean", "default": false, "description": "When true, include the model's <think> reasoning_content in the output. Default false (just the answer)."}
                },
                "required": ["prompt"]
            }
        }),
        json!({
            "name": "list_cloud_models",
            "description": "List configured cloud models from ~/.config/lamu/cloud-models.yaml. Returns name, provider, context window, and whether the API key env var is set.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "review_commit",
            "description": "PRIMARY REVIEW TOOL — auto-routes to DeepSeek V4 Pro (the project policy reviewer). Takes a commit SHA (or 'HEAD' for the most recent), runs `git show` to get the full diff + commit message, and returns a deep code review covering security, correctness, edge cases, idiom, and architectural fit. NO CODE SHOULD BE CONSIDERED DONE WITHOUT GOING THROUGH THIS TOOL. Use it after every commit you make.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "commit": {"type": "string", "description": "Commit SHA or ref (e.g. 'HEAD', 'HEAD~1', 'abc123'). Defaults to HEAD.", "default": "HEAD"},
                    "repo": {"type": "string", "description": "Path to the git repo. Defaults to current working directory.", "default": "."},
                    "focus": {"type": "string", "description": "Optional review focus (e.g. 'security', 'performance', 'API design'). Defaults to all-around.", "default": ""}
                }
            }
        }),
        json!({
            "name": "review_diff",
            "description": "Review an arbitrary diff via DeepSeek V4 Pro. Same reviewer policy as review_commit but accepts the diff text directly — useful when reviewing uncommitted changes or a chunk of pasted code.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "diff": {"type": "string", "description": "Unified diff text or a code chunk to review."},
                    "context": {"type": "string", "description": "Optional surrounding context (e.g. file paths, what changed and why).", "default": ""},
                    "focus": {"type": "string", "default": ""}
                },
                "required": ["diff"]
            }
        }),
        json!({
            "name": "set_routing_mode",
            "description": "Control which backends are usable. Modes: 'auto' (default — use local for matching capabilities, cloud for the rest), 'local-only' (refuse cloud requests), 'cloud-only' (kill local llama-server and free VRAM, route everything to cloud). Useful when you want to free GPU for other work but keep DeepSeek/Claude on tap.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["auto", "local-only", "cloud-only"]}
                },
                "required": ["mode"]
            }
        }),
        json!({
            "name": "routing_status",
            "description": "Report current routing mode + which backends are reachable.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "parallel_query",
            "description": "Fan out N prompts at once (agent swarm). Provider-aware concurrency: DeepSeek/OpenAI/Anthropic run in parallel up to per-provider caps, untested providers and ALL local models default to sequential (concurrency=1) until proven safe. Tasks are grouped by model so each model gets its own semaphore. Returns results in the original task order, with per-task elapsed time. Use this for batch reviews, parallel code generation, multi-perspective brainstorming.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "Array of task objects. Each can override model/system/max_tokens/temperature/id; missing fields fall back to top-level defaults.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Optional caller-supplied id for matching results back."},
                                "prompt": {"type": "string"},
                                "model": {"type": "string"},
                                "system": {"type": "string"},
                                "max_tokens": {"type": "integer"},
                                "temperature": {"type": "number"},
                                "include_reasoning": {"type": "boolean"}
                            },
                            "required": ["prompt"]
                        }
                    },
                    "default_model": {"type": "string", "default": "deepseek-v4-flash"},
                    "default_system": {"type": "string", "default": ""},
                    "max_concurrency": {"type": "integer", "description": "Optional cap that overrides per-provider defaults (downwards only — never raises an unproven provider above 1)."}
                },
                "required": ["tasks"]
            }
        }),
    ];

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "tools": tools }
    })
}

/// Build the right spawn `Command` for the entry's backend. Returns
/// `(cmd, health_url_path, expect_status_ok, max_wait_secs)`.
async fn build_spawn_cmd(
    entry: &ModelEntry,
    port: u16,
) -> std::result::Result<(tokio::process::Command, &'static str, bool, u32), String> {
    use tokio::process::Command;
    let home = dirs::home_dir().ok_or_else(|| "no home dir".to_string())?;

    match entry.backend {
        BackendType::LlamaCpp => {
            let bin = lamu_core::config::llama_bin();
            if !bin.exists() {
                return Err(format!("llama-server not found at {}", bin.display()));
            }
            let supports_ngram = match Command::new(&bin).arg("--help").output().await {
                Ok(o) => String::from_utf8_lossy(&o.stdout).contains("--spec-ngram-mod-n-match"),
                Err(_) => false,
            };
            let mut cmd = Command::new(&bin);
            cmd.arg("-m").arg(&entry.path)
                .arg("--host").arg("0.0.0.0")
                .arg("--port").arg(port.to_string())
                .arg("--ctx-size").arg(std::cmp::min(entry.context_max, 32768).to_string())
                .arg("-ngl").arg("99")
                .arg("--flash-attn").arg("on")
                .arg("--cache-type-k").arg("q4_0")
                .arg("--cache-type-v").arg("q4_0")
                .arg("--parallel").arg("1");
            if supports_ngram && (entry.arch == "qwen35" || entry.arch == "qwen3") {
                cmd.args([
                    "--spec-type", "ngram-mod",
                    "--spec-ngram-mod-n-match", "24",
                    "--spec-ngram-mod-n-min", "12",
                    "--spec-ngram-mod-n-max", "48",
                ]);
            }
            Ok((cmd, "/health", true, 45))
        }
        BackendType::Megakernel => {
            let py = home.join("local-llm/.venv/bin/python");
            let script = home.join("local-llm/server/megakernel_server.py");
            let workdir = home.join("local-llm/lucebox-hub/megakernel");
            if !py.exists() {
                return Err(format!("python not found at {}", py.display()));
            }
            if !script.exists() {
                return Err(format!("megakernel server not found at {}", script.display()));
            }
            let mut cmd = Command::new(&py);
            cmd.arg(&script)
                .arg("--port").arg(port.to_string())
                .current_dir(&workdir)
                .env("CUDA_VISIBLE_DEVICES", "0");
            Ok((cmd, "/health", false, 30))
        }
        BackendType::Dflash | BackendType::DflashLucebox => {
            let spec = entry.speculative.as_ref().ok_or_else(|| format!(
                "dflash backend requires `speculative` config in entry '{}'", entry.name
            ))?;
            let py = home.join("local-llm/.venv/bin/python");
            let script = home.join("local-llm/server/dflash_24gb.py");
            let workdir = home.join("local-llm/lucebox-hub/dflash");
            let test_bin = workdir.join("build/test_dflash");
            if !py.exists() {
                return Err(format!("python not found at {}", py.display()));
            }
            if !script.exists() {
                return Err(format!("dflash server not found at {}", script.display()));
            }
            if !test_bin.exists() {
                return Err(format!("test_dflash binary not found at {}", test_bin.display()));
            }
            let mut cmd = Command::new(&py);
            cmd.arg(&script)
                .arg("--port").arg(port.to_string())
                .arg("--max-ctx").arg("8192")
                .arg("--budget").arg("6")
                .arg("--bin").arg(&test_bin)
                .arg("--target").arg(&entry.path)
                .arg("--draft").arg(&spec.draft_path)
                .current_dir(&workdir)
                .env("CUDA_VISIBLE_DEVICES", "0")
                .env("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1");
            Ok((cmd, "/v1/models", false, 90))
        }
    }
}

// ── Cloud routing (DeepSeek, Anthropic, etc.) ───────────────────────
//
// MCP exposes `cloud_query` so an outer agent (Claude Code, etc.) can
// fan tasks out to a cheaper / faster cloud model. Reads cloud config
// from ~/.config/lamu/cloud-models.yaml. Auto-detects provider from
// base_url (Anthropic → /v1/messages + x-api-key; everything else →
// OpenAI compat /chat/completions + Bearer).

#[derive(serde::Deserialize, Debug, Clone)]
struct CloudYamlEntry {
    name: String,
    #[serde(default)]
    provider: String,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    notes: String,
    #[serde(default)]
    context_max: u32,
}

#[derive(serde::Deserialize, Debug)]
struct CloudYamlFile {
    #[serde(default)]
    models: Vec<CloudYamlEntry>,
}

fn load_cloud_models() -> Vec<CloudYamlEntry> {
    let Some(dir) = dirs::config_dir() else { return Vec::new(); };
    let path = dir.join("lamu").join("cloud-models.yaml");
    let Ok(body) = std::fs::read_to_string(&path) else { return Vec::new(); };
    serde_yaml::from_str::<CloudYamlFile>(&body)
        .map(|f| f.models)
        .unwrap_or_default()
}

/// Per-provider concurrency cap. Conservative by default — only
/// providers we've explicitly tested under parallel load get a cap >1.
/// Unknown / lightly-tested providers are sequential until proven safe.
///
/// Override per-provider with env vars:
///   LAMU_PARALLEL_DEEPSEEK / _ANTHROPIC / _OPENAI / etc.
fn provider_concurrency(model_name: &str, cloud: &[CloudYamlEntry]) -> usize {
    let provider = cloud.iter()
        .find(|m| m.name == model_name)
        .map(|m| m.provider.as_str())
        .unwrap_or("");

    // Env override takes precedence.
    let env_var = format!("LAMU_PARALLEL_{}", provider.to_uppercase());
    if let Ok(v) = std::env::var(&env_var) {
        if let Ok(n) = v.parse::<usize>() {
            return n.max(1);
        }
    }

    match provider {
        // Tested under parallel load.
        "deepseek" => 8,
        "anthropic" => 4,
        "openai" => 4,
        // Less tested — start at 1 until proven. Bump via env var.
        // Bumping here without a parallel-test run is the wrong default.
        _ => 1,
    }
}

fn handle_list_cloud_models() -> String {
    let models = load_cloud_models();
    if models.is_empty() {
        return "(no cloud models — edit ~/.config/lamu/cloud-models.yaml or run `lamu` and press 'n' to add)".into();
    }
    let mut out = String::new();
    for m in &models {
        let key_status = match &m.api_key_env {
            None => "(no key needed — gateway-routed)".to_string(),
            Some(env) => if std::env::var(env).is_ok() { format!("${} ✓", env) } else { format!("${} unset ✗", env) },
        };
        let mid = m.model_id.clone().unwrap_or_else(|| format!("{}/{}", m.provider, m.name));
        out.push_str(&format!(
            "{}  ({})  ctx={}  {}  — {}\n",
            m.name, mid, m.context_max, key_status, m.notes
        ));
    }
    out
}

async fn handle_cloud_query(args: Value) -> String {
    let prompt = args["prompt"].as_str().unwrap_or("");
    if prompt.is_empty() { return "error: prompt is required".into(); }
    let model_name = args["model"].as_str().unwrap_or("deepseek-v4-flash");
    let system = args["system"].as_str().unwrap_or("");
    let max_tokens = args["max_tokens"].as_u64().unwrap_or(8192) as u32;
    let temperature = args["temperature"].as_f64().unwrap_or(0.3) as f32;
    let include_reasoning = args["include_reasoning"].as_bool().unwrap_or(false);

    let models = load_cloud_models();
    let entry = match models.iter().find(|m| m.name == model_name) {
        Some(m) => m.clone(),
        None => return format!(
            "error: cloud model '{}' not in cloud-models.yaml. Run `list_cloud_models` to see options.",
            model_name
        ),
    };

    let api_key = match entry.api_key_env.as_deref() {
        Some(env) => match std::env::var(env) {
            Ok(k) => k,
            Err(_) => return format!(
                "error: ${} is not set. Add it via `lamu` (press 'a' on the model row) or export it manually.",
                env
            ),
        },
        None => "no-key-needed".to_string(),
    };

    let base = match entry.base_url.as_deref() {
        Some(b) => b.trim_end_matches('/').to_string(),
        None => return format!(
            "error: cloud model '{}' has no base_url. Edit ~/.config/lamu/cloud-models.yaml.",
            model_name
        ),
    };
    let model_id = entry.model_id.clone().unwrap_or_else(|| entry.name.clone());
    let is_anthropic = entry.provider == "anthropic" || base.contains("anthropic");

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build() {
        Ok(c) => c,
        Err(e) => return format!("error: client init: {e}"),
    };

    if is_anthropic {
        let url = format!("{}/v1/messages", base);
        let mut payload = json!({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        if !system.is_empty() { payload["system"] = json!(system); }

        let resp = match client.post(&url)
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&payload).send().await {
            Ok(r) => r,
            Err(e) => return format!("error: post {url}: {e}"),
        };
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return format!("error: parse: {e}"),
        };
        if let Some(err) = v.get("error") {
            return format!("anthropic error: {}", err);
        }
        // content is an array of {type: "text"|"thinking", text|thinking: "..."}
        let mut out = String::new();
        let mut thinking = String::new();
        if let Some(blocks) = v["content"].as_array() {
            for b in blocks {
                match b["type"].as_str() {
                    Some("text") => out.push_str(b["text"].as_str().unwrap_or("")),
                    Some("thinking") => thinking.push_str(b["thinking"].as_str().unwrap_or("")),
                    _ => {}
                }
            }
        }
        if include_reasoning && !thinking.is_empty() {
            return format!("<think>\n{}\n</think>\n{}", thinking, out);
        }
        out
    } else {
        let url = format!("{}/chat/completions", base);
        let mut messages: Vec<Value> = Vec::new();
        if !system.is_empty() {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.push(json!({"role": "user", "content": prompt}));
        let payload = json!({
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        let resp = match client.post(&url)
            .bearer_auth(&api_key)
            .json(&payload).send().await {
            Ok(r) => r,
            Err(e) => return format!("error: post {url}: {e}"),
        };
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return format!("error: parse: {e}"),
        };
        if let Some(err) = v.get("error") {
            return format!("provider error: {}", err);
        }
        let msg = &v["choices"][0]["message"];
        let content = msg["content"].as_str().unwrap_or("");
        let reasoning = msg["reasoning_content"].as_str().unwrap_or("");
        if include_reasoning && !reasoning.is_empty() {
            format!("<think>\n{}\n</think>\n{}", reasoning, content)
        } else {
            content.to_string()
        }
    }
}

// ── DeepSeek V4 Pro reviewer (project policy) ───────────────────────
//
// Every commit goes through this. The system prompt below tells V4 Pro
// to focus on issues that matter — security, correctness, edge cases,
// architecture — and to call out problems even when none exist. The
// model's reasoning_content is included so the human can see HOW the
// review was reached, not just the conclusion.

const REVIEW_SYSTEM_PROMPT: &str = "You are a senior staff engineer doing a code review. Your job is to find real issues, not to pat anyone on the back.\n\nAlways check:\n  1. SECURITY — injection (SQL/shell/XSS/prompt), auth/authz holes, secrets in code, unsafe deserialization, TOCTOU, missing input validation.\n  2. CORRECTNESS — off-by-one, null/empty cases, integer overflow, floating-point traps, race conditions, deadlocks, missing error handling.\n  3. EDGE CASES — what happens at boundaries, with empty inputs, with hostile inputs, under concurrency, on partial failure, on retry.\n  4. ARCHITECTURE — does this fit the existing design? Does it leak abstraction? Does it create coupling that will hurt later? Is there a simpler shape?\n  5. CLARITY — would a stranger understand the intent? Are names accurate? Are comments necessary or noise?\n\nFormat your output:\n  - One-sentence verdict (PASS / PASS WITH NITS / NEEDS CHANGES / REJECT).\n  - Numbered list of findings, each: severity [BUG/SECURITY/STYLE/QUESTION], file:line if knowable, the problem, the suggested fix.\n  - End with a single 'Recommend' line.\n\nBe terse. Be honest. Don't praise unless something is genuinely surprising in a good way. If the code is fine, say so in one line and stop.";

/// Cap on diff size sent to the reviewer. 200 KiB is generous (≈ 4K
/// lines of typical code) but bounded — anything larger gets truncated
/// with a marker so the model knows it's not seeing the whole change.
const MAX_REVIEW_DIFF_BYTES: usize = 200 * 1024;

/// Validate a git ref / commit. Accepts: hex SHA (7-40 chars), HEAD
/// followed by any sequence of `~N` / `^[N]` suffixes (HEAD^^,
/// HEAD~1^2, HEAD^^^, etc. — all valid git refs), or a plain refname
/// matching git's safe character set (alnum + _ - . /, no leading
/// '-' or '.', no '..').
fn is_safe_git_ref(s: &str) -> bool {
    if s.is_empty() || s.starts_with('-') { return false; }
    // Hex SHA / abbrev.
    if s.len() >= 7 && s.len() <= 40 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        return true;
    }
    // HEAD with any chain of ~N and ^[N] suffixes.
    if let Some(rest) = s.strip_prefix("HEAD") {
        return parse_rev_suffix(rest);
    }
    // General refname.
    if s.contains("..") || s.starts_with('.') { return false; }
    s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
}

/// Walk a sequence of `~N` and `^[N]` suffixes. `^` alone means parent;
/// `^N` means Nth parent. Returns true iff the entire suffix consumes.
fn parse_rev_suffix(mut s: &str) -> bool {
    while !s.is_empty() {
        let first = s.as_bytes()[0];
        if first == b'~' || first == b'^' {
            s = &s[1..];
            // Optional digit run.
            let digit_end = s.bytes().take_while(|b| b.is_ascii_digit()).count();
            s = &s[digit_end..];
        } else {
            return false;
        }
    }
    true
}

/// Truncate `text` to at most `limit` bytes, snapping back to the
/// nearest UTF-8 char boundary so we never split a multi-byte
/// codepoint mid-stream. Appends a marker describing how much was
/// dropped so the reviewer LLM knows it didn't see the full diff.
fn truncate_with_marker(text: &str, limit: usize) -> String {
    if text.len() <= limit { return text.to_string(); }
    // Walk back to the last char boundary at or before `limit`.
    let mut cut = limit;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let mut out = text[..cut].to_string();
    out.push_str(&format!(
        "\n\n[…truncated {} more bytes — diff exceeded {} byte review limit]",
        text.len() - cut, limit
    ));
    out
}

async fn handle_review_commit(args: Value) -> String {
    let commit = args["commit"].as_str().unwrap_or("HEAD");
    let repo = args["repo"].as_str().unwrap_or(".");
    let focus = args["focus"].as_str().unwrap_or("");

    if !is_safe_git_ref(commit) {
        return format!(
            "error: commit '{}' rejected — must be a hex SHA, HEAD with ~/^ suffixes, or a safe refname.",
            commit
        );
    }

    // is_safe_git_ref already rejects anything starting with '-', so
    // git can't interpret commit as a flag. No defense-in-depth needed.
    let out = match std::process::Command::new("git")
        .current_dir(repo)
        .args(["show", "--stat", "--patch", commit])
        .output()
    {
        Ok(o) => o,
        Err(e) => return format!("error: spawn git: {}", e),
    };

    if !out.status.success() {
        return format!("error: git show {} failed: {}", commit,
            String::from_utf8_lossy(&out.stderr).trim());
    }
    let diff_text = String::from_utf8_lossy(&out.stdout).to_string();
    if diff_text.trim().is_empty() {
        return format!("error: empty diff for {}", commit);
    }
    let diff_text = truncate_with_marker(&diff_text, MAX_REVIEW_DIFF_BYTES);

    let mut prompt = String::new();
    if !focus.is_empty() {
        prompt.push_str(&format!("Focus the review on: {}\n\n", focus));
    }
    prompt.push_str("Here is the commit to review (full diff):\n\n```\n");
    prompt.push_str(&diff_text);
    prompt.push_str("\n```\n");

    let review_args = json!({
        "model": "deepseek-v4-pro",
        "prompt": prompt,
        "system": REVIEW_SYSTEM_PROMPT,
        "max_tokens": 8192,
        "temperature": 0.2,
        "include_reasoning": false,
    });
    let review = handle_cloud_query(review_args).await;
    format!("=== Review of {} (DeepSeek V4 Pro) ===\n\n{}", commit, review)
}

async fn handle_review_diff(args: Value) -> String {
    let diff = args["diff"].as_str().unwrap_or("");
    if diff.is_empty() {
        return "error: 'diff' is required".into();
    }
    let diff = truncate_with_marker(diff, MAX_REVIEW_DIFF_BYTES);
    let context = args["context"].as_str().unwrap_or("");
    let focus = args["focus"].as_str().unwrap_or("");

    let mut prompt = String::new();
    if !focus.is_empty() {
        prompt.push_str(&format!("Focus the review on: {}\n\n", focus));
    }
    if !context.is_empty() {
        prompt.push_str("Context:\n");
        prompt.push_str(context);
        prompt.push_str("\n\n");
    }
    prompt.push_str("Diff to review:\n\n```\n");
    prompt.push_str(&diff);
    prompt.push_str("\n```\n");

    let review_args = json!({
        "model": "deepseek-v4-pro",
        "prompt": prompt,
        "system": REVIEW_SYSTEM_PROMPT,
        "max_tokens": 8192,
        "temperature": 0.2,
        "include_reasoning": false,
    });
    let review = handle_cloud_query(review_args).await;
    format!("=== Diff review (DeepSeek V4 Pro) ===\n\n{}", review)
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
            "parallel_query",
        ] {
            assert!(names.contains(&required), "missing tool: {}", required);
        }
    }

    #[test]
    fn provider_concurrency_known_providers() {
        let cloud = vec![
            CloudYamlEntry { name: "ds".into(), provider: "deepseek".into(),
                model_id: None, api_key_env: None, base_url: None,
                notes: String::new(), context_max: 0 },
            CloudYamlEntry { name: "claude".into(), provider: "anthropic".into(),
                model_id: None, api_key_env: None, base_url: None,
                notes: String::new(), context_max: 0 },
            CloudYamlEntry { name: "gpt".into(), provider: "openai".into(),
                model_id: None, api_key_env: None, base_url: None,
                notes: String::new(), context_max: 0 },
        ];
        assert_eq!(provider_concurrency("ds", &cloud), 8);
        assert_eq!(provider_concurrency("claude", &cloud), 4);
        assert_eq!(provider_concurrency("gpt", &cloud), 4);
    }

    #[test]
    fn provider_concurrency_unknown_defaults_to_1() {
        let cloud = vec![
            CloudYamlEntry { name: "kimi".into(), provider: "moonshot".into(),
                model_id: None, api_key_env: None, base_url: None,
                notes: String::new(), context_max: 0 },
            CloudYamlEntry { name: "qwen".into(), provider: "alibaba".into(),
                model_id: None, api_key_env: None, base_url: None,
                notes: String::new(), context_max: 0 },
        ];
        assert_eq!(provider_concurrency("kimi", &cloud), 1);
        assert_eq!(provider_concurrency("qwen", &cloud), 1);
        assert_eq!(provider_concurrency("not-in-yaml", &cloud), 1);
    }

    #[test]
    fn safe_git_ref_accepts_hex_sha() {
        assert!(is_safe_git_ref("abc1234"));
        assert!(is_safe_git_ref("abc1234567890"));
        assert!(is_safe_git_ref(&"a".repeat(40)));
    }

    #[test]
    fn safe_git_ref_accepts_head_variants() {
        assert!(is_safe_git_ref("HEAD"));
        assert!(is_safe_git_ref("HEAD~1"));
        assert!(is_safe_git_ref("HEAD~10"));
        assert!(is_safe_git_ref("HEAD^"));
        assert!(is_safe_git_ref("HEAD^2"));
        // Chained suffixes — all valid git revisions.
        assert!(is_safe_git_ref("HEAD^^"));
        assert!(is_safe_git_ref("HEAD~1^"));
        assert!(is_safe_git_ref("HEAD~1^2"));
        assert!(is_safe_git_ref("HEAD^^^"));
        assert!(is_safe_git_ref("HEAD~3~2"));
    }

    #[test]
    fn safe_git_ref_accepts_branch_names() {
        assert!(is_safe_git_ref("main"));
        assert!(is_safe_git_ref("feature/x-123"));
        assert!(is_safe_git_ref("release-1.0"));
    }

    #[test]
    fn safe_git_ref_rejects_dangerous() {
        assert!(!is_safe_git_ref(""));
        assert!(!is_safe_git_ref("--upload-pack=evil"));
        assert!(!is_safe_git_ref("-v"));
        assert!(!is_safe_git_ref("../escape"));
        assert!(!is_safe_git_ref(".hidden"));
        assert!(!is_safe_git_ref("HEAD; rm -rf /"));
        assert!(!is_safe_git_ref("HEAD~abc"));
        assert!(!is_safe_git_ref("branch with space"));
        assert!(!is_safe_git_ref("name$with#meta"));
    }

    #[test]
    fn truncate_marker_short_string_unchanged() {
        let s = "short";
        assert_eq!(truncate_with_marker(s, 100), s);
    }

    #[test]
    fn truncate_marker_long_string_truncated() {
        let s = "x".repeat(1000);
        let out = truncate_with_marker(&s, 100);
        assert!(out.len() < s.len());
        assert!(out.contains("truncated"));
        assert!(out.contains("900 more bytes"));
    }

    #[test]
    fn truncate_marker_does_not_panic_on_utf8_boundary() {
        // 4-byte UTF-8 codepoint (😀) at position 99 — limit=100 falls
        // mid-codepoint. Naive slicing panics; we must snap back.
        let mut s = "x".repeat(99);
        s.push('😀');
        s.push_str(&"y".repeat(50));
        let out = truncate_with_marker(&s, 100);
        // No panic = test passed. Verify content sane.
        assert!(out.starts_with(&"x".repeat(99)));
        assert!(out.contains("truncated"));
    }

    #[test]
    fn tools_list_response_query_requires_prompt() {
        let resp = tools_list_response(None);
        let query = resp["result"]["tools"].as_array().unwrap()
            .iter().find(|t| t["name"] == "query").unwrap();
        let required = query["inputSchema"]["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "prompt"));
    }
}
