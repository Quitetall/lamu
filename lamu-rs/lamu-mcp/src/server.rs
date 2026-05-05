//! MCP stdio server. Hand-rolled JSON-RPC.
//! Direct port of `lamu/mcp/server.py::LamuMcpServer`.
//!
//! Protocol: line-delimited JSON-RPC 2.0 over stdin/stdout.
//! Logs to stderr. Tools dispatched via `tools::*`.

use anyhow::{Context, Result};
use lamu_core::reasoning::get_extractor;
use lamu_core::registry::{load_registry, scan_directory, write_registry};
use lamu_core::router::Router;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{Capability, ModelEntry};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

pub struct LamuMcpServer {
    pub state: Arc<Mutex<ServerState>>,
}

pub struct ServerState {
    pub models_dir: PathBuf,
    pub registry_path: PathBuf,
    pub scheduler: VramScheduler,
    pub entries: HashMap<String, ModelEntry>,
    pub router: Router,
    pub client: reqwest::Client,
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
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("reqwest");

        Ok(Self {
            state: Arc::new(Mutex::new(ServerState {
                models_dir,
                registry_path,
                scheduler,
                entries,
                router,
                client,
            })),
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

    async fn handle_query(&self, args: Value) -> String {
        let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        if prompt.is_empty() {
            return "missing prompt".into();
        }

        let model = args.get("model").and_then(|v| v.as_str());
        let caps_raw = args.get("capabilities").and_then(|v| v.as_array());
        let system = args.get("system").and_then(|v| v.as_str()).unwrap_or("");
        let max_tokens = args.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(16384) as u32;
        let temperature = args.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.7) as f32;
        let include_reasoning = args.get("include_reasoning").and_then(|v| v.as_bool()).unwrap_or(false);

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
            let decision = st.router.route(&st.scheduler, model, caps_opt);

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
        let resp = match client.post(&url).json(&payload).send().await {
            Ok(r) => r,
            Err(e) => return format!("Generation error: {}", e),
        };
        let data: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return format!("JSON decode error: {}", e),
        };

        let msg = match data.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message")) {
            Some(m) => m,
            None => return "no message in response".into(),
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
        let decision = st.router.route(&st.scheduler, model, caps_opt);
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

        // Evict
        if !to_evict.is_empty() {
            let mut st = self.state.lock();
            for evict_name in &to_evict {
                if let Some(loaded) = st.scheduler.get_loaded(evict_name) {
                    if let Some(pid) = loaded.pid {
                        if pid > 0 {
                            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
                        }
                    }
                }
                st.scheduler.mark_unloaded(evict_name);
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

        // Spawn llama-server
        let bin = lamu_core::config::llama_bin();
        if !bin.exists() {
            return format!("llama-server not found at {}", bin.display());
        }

        let supports_ngram = {
            let out = tokio::process::Command::new(&bin).arg("--help").output().await;
            match out {
                Ok(o) => String::from_utf8_lossy(&o.stdout).contains("--spec-ngram-mod-n-match"),
                Err(_) => false,
            }
        };

        let mut cmd = tokio::process::Command::new(&bin);
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
        // Detach — we keep PID and kill later via libc
        std::mem::forget(child);

        // Health poll
        let client = self.state.lock().client.clone();
        for _ in 0..45 {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let url = format!("http://localhost:{}/health", port);
            if let Ok(r) = client.get(&url).send().await {
                if let Ok(j) = r.json::<Value>().await {
                    if j.get("status").and_then(|v| v.as_str()) == Some("ok") {
                        // Confirm + register VRAM
                        let mut st = self.state.lock();
                        let pids = st.scheduler.query_gpu_pids();
                        let vram = pids.iter()
                            .find(|(p, _)| *p == pid)
                            .map(|(_, m)| *m)
                            .unwrap_or(entry.vram_mb);
                        let _ = st.scheduler.confirm_loaded(&entry.name, pid, port, vram);
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
            }
        }

        // Timeout
        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        let mut st = self.state.lock();
        st.scheduler.mark_unloaded(&entry.name);
        format!("Failed to load '{}' (timeout 45s)", entry.name)
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

        if let Some(loaded) = st.scheduler.get_loaded(&target) {
            if let Some(pid) = loaded.pid {
                if pid > 0 {
                    unsafe { libc::kill(pid as i32, libc::SIGKILL) };
                }
            }
        }
        st.scheduler.mark_unloaded(&target);
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
            "description": "Send prompt to local LLM. Routes by capabilities or explicit model. Fast, free, uncensored.",
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
    ];

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "tools": tools }
    })
}
