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
use tokio::sync::Mutex as AsyncMutex;

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
            "unload_model" => self.handle_unload_model(args),
            "vram_status" => self.handle_vram_status(),
            "scan_models" => self.handle_scan(),
            "queue_status" => self.handle_queue_status().await,
            "cloud_query" => handle_cloud_query(args).await,
            "list_cloud_models" => handle_list_cloud_models(),
            "review_commit" => handle_review_commit(args).await,
            "review_diff" => handle_review_diff(args).await,
            "set_routing_mode" => self.handle_set_routing_mode(args).await,
            "routing_status" => self.handle_routing_status().await,
            other => format!("Unknown tool: {}", other),
        };

        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{ "type": "text", "text": result }],
                "isError": false
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
            None => return "missing name".into(),
        };

        // Find entry + plan eviction
        let (entry, to_evict) = {
            let st = self.state.lock();
            let entry: Option<ModelEntry> = st.entries.iter()
                .find(|(n, _)| n.contains(&name) || name.contains(n.as_str()))
                .map(|(_, e)| e.clone());
            let Some(entry) = entry else {
                return format!("Model '{}' not found in registry. Run scan_models.", name);
            };
            if st.scheduler.is_loaded(&entry.name) {
                return format!("Model '{}' already loaded.", entry.name);
            }
            let (can, evict) = st.scheduler.plan_load(&entry);
            if !can {
                return format!(
                    "Cannot fit '{}' ({}MB) in VRAM. Insufficient space.",
                    entry.name, entry.vram_mb
                );
            }
            (entry, evict)
        };

        // Evict — start_kill on the owned Child, then drop it. Health entry
        // for the evicted backend is removed so its supervisor lifecycle ends.
        if !to_evict.is_empty() {
            let mut st = self.state.lock();
            for evict_name in &to_evict {
                if let Some(mut child) = st.loaded_procs.remove(evict_name) {
                    let _ = child.start_kill();
                }
                st.scheduler.mark_unloaded(evict_name);
                st.health.drop(evict_name);
            }
        }
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

        // Timeout — kill the stored Child and unregister scheduler/health.
        let mut st = self.state.lock();
        if let Some(mut child) = st.loaded_procs.remove(&entry.name) {
            let _ = child.start_kill();
        }
        st.scheduler.mark_unloaded(&entry.name);
        st.health.drop(&entry.name);
        format!("Failed to load '{}' (timeout {}s)", entry.name, max_wait_secs)
    }

    fn handle_unload_model(&self, args: Value) -> String {
        let name = match args.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => return "missing name".into(),
        };

        let mut st = self.state.lock();
        let target: Option<String> = st.scheduler.loaded_models().iter()
            .find(|m| m.entry.name.contains(&name) || name.contains(m.entry.name.as_str()))
            .map(|m| m.entry.name.clone());

        let Some(target) = target else {
            return format!("Model '{}' not loaded.", name);
        };

        if let Some(mut child) = st.loaded_procs.remove(&target) {
            let _ = child.start_kill();
        }
        st.scheduler.mark_unloaded(&target);
        st.health.drop(&target);
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
        let mut current = self.routing_mode.lock().await;
        let old = current.clone();
        *current = mode.clone();
        drop(current);

        // cloud-only → free all local VRAM by unloading every loaded backend.
        let mut freed = Vec::new();
        if mode == "cloud-only" {
            let names: Vec<String> = {
                let st = self.state.lock();
                st.scheduler.loaded_models().iter().map(|m| m.entry.name.clone()).collect()
            };
            for n in names {
                let mut st = self.state.lock();
                if let Some(mut p) = st.loaded_procs.remove(&n) {
                    drop(st);
                    let _ = p.start_kill();
                    let _ = p.wait().await;
                    let mut st2 = self.state.lock();
                    st2.scheduler.mark_unloaded(&n);
                    freed.push(n);
                } else {
                    st.scheduler.mark_unloaded(&n);
                    freed.push(n);
                }
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

async fn handle_review_commit(args: Value) -> String {
    let commit = args["commit"].as_str().unwrap_or("HEAD");
    let repo = args["repo"].as_str().unwrap_or(".");
    let focus = args["focus"].as_str().unwrap_or("");

    // Get diff + commit message via git show.
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
    prompt.push_str(diff);
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
}
