//! Phase 2.2b/c: stateful tool handlers.
//!
//! All `impl LamuMcpServer { handle_* }` methods extracted from
//! `server.rs`. Each handler is reachable via the dispatch table in
//! `tools.rs` which calls them through dispatch wrappers — never
//! directly. This file is purely the implementation surface.

use crate::server::LamuMcpServer;
use lamu_core::backends::{make_backend, Backend, ChatMessage};
use lamu_core::observability::{emit, new_trace_id, trace_id_from_traceparent};
use lamu_core::queue::{QueueRequest, Strategy as QueueStrategy};
use lamu_core::reasoning::get_extractor;
use lamu_core::registry::{scan_directory, write_registry};
use lamu_core::types::{Capability, ModelEntry};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tracing::warn;
use std::collections::HashMap;

/// Map a capability string from MCP args. Pub(crate) so the
/// dispatch / wiring layers can also parse if needed.
pub(crate) fn parse_capability(s: &str) -> Option<Capability> {
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

impl LamuMcpServer {
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

        // Refuse before any VRAM allocation if `lamu-train` (or
        // another exclusive holder) owns the GPU. `--allow-evict`
        // flag (future) flips this to a wait via await_unlock.
        if let Err(e) = lamu_core::scheduler_lock::check_unlocked() {
            return format!("error: {e}");
        }

        let model = args.get("model").and_then(|v| v.as_str());
        let caps_raw = args.get("capabilities").and_then(|v| v.as_array());
        let system = args.get("system").and_then(|v| v.as_str()).unwrap_or("");
        // Parse samplers as Option so we can distinguish "caller omitted"
        // from "caller passed the default" when merging the per-model
        // sampling profile (see lamu_core::types::SamplingProfile). The
        // builtin defaults (16384 / 0.7) are applied as the final merge
        // fallback below, preserving prior behavior when no profile + no
        // request value.
        let req_max_tokens = args.get("max_tokens").and_then(|v| v.as_u64()).map(|x| x as u32);
        let req_temperature = args.get("temperature").and_then(|v| v.as_f64()).map(|x| x as f32);
        let req_top_p = args.get("top_p").and_then(|v| v.as_f64()).map(|x| x as f32);
        let req_top_k = args.get("top_k").and_then(|v| v.as_u64()).map(|x| x as u32);
        let req_min_p = args.get("min_p").and_then(|v| v.as_f64()).map(|x| x as f32);
        let req_repeat_penalty = args.get("repeat_penalty").and_then(|v| v.as_f64()).map(|x| x as f32);
        let include_reasoning = args.get("include_reasoning").and_then(|v| v.as_bool()).unwrap_or(false);
        let priority = args.get("priority").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let origin = args.get("origin").and_then(|v| v.as_str()).unwrap_or("anonymous").to_string();
        // Qwen3.6 / Qwen3.5 reasoning toggle: when explicitly false, append
        // `/no_think` to the system message. Qwen's chat template honors
        // this directive and skips the `<think>` block, cutting wall time
        // 4× on tiny prompts and ~20% on long. Unset = leave default
        // (thinking on).
        let enable_thinking = args.get("enable_thinking").and_then(|v| v.as_bool());

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

        // Route + collect target info under lock. Backend Arc is
        // cloned out of the map so .generate() can run without holding
        // the state lock across .await.
        let (model_name, marker, sampling, backend_arc) = {
            let st = self.state.lock();
            let decision = st.router.route(&st.scheduler, model, caps_opt, Some(st.health.all()));

            if decision.model_name.is_empty() {
                return format!("error: no model available: {}", decision.reason);
            }
            if !decision.loaded {
                return format!(
                    "error: model '{}' not loaded. Would need to load (evicting: {:?}). \
                     Use load_model first or query a loaded model.",
                    decision.model_name, decision.would_evict
                );
            }

            let Some(_loaded) = st.scheduler.get_loaded(&decision.model_name) else {
                return "error: internal: lost loaded model".into();
            };
            let entry = st.entries.get(&decision.model_name).cloned();
            let marker = entry.as_ref().and_then(|e| e.reasoning_marker.clone());
            let sampling = entry.as_ref().and_then(|e| e.sampling.clone());
            let Some(backend_arc) = st.backends.get(&decision.model_name).cloned() else {
                return format!(
                    "error: internal: model '{}' marked loaded but missing from backends map",
                    decision.model_name
                );
            };
            (decision.model_name, marker, sampling, backend_arc)
        };

        // Mark used (separate lock acquisition)
        {
            let mut st = self.state.lock();
            st.scheduler.mark_used(&model_name);
        }

        // Build chat history in the unified Backend format.
        let mut chat_messages: Vec<lamu_core::backends::ChatMessage> = Vec::new();
        if !system.is_empty() {
            chat_messages.push(lamu_core::backends::ChatMessage {
                role: "system".into(),
                content: system.to_string(),
            });
        }
        chat_messages.push(lamu_core::backends::ChatMessage {
            role: "user".into(),
            content: prompt.to_string(),
        });

        // Acquire queue slot before hitting backend
        let queue = self.get_or_create_queue(&model_name).await;
        let _guard = queue.enqueue(QueueRequest {
            payload: (),
            priority,
            enqueued_at: Instant::now(),
            origin,
        }).await;

        // Phase 6.3b: dispatch through Backend::generate. Each impl
        // (LlamaCpp / Megakernel / Dflash) parses its own response
        // shape, so the OpenAI-only inline parser this used to be is
        // gone. The backend mutex is held only across .generate() —
        // queue gating limits concurrent same-model requests already.
        let raw = {
            let backend = backend_arc.lock().await;
            // Pre-check: if the backend has died (process exited, port
            // unbound, etc.) surface a clear error before .generate()
            // returns an opaque empty/timeout. Each Backend impl knows
            // how to probe its own health endpoint.
            if !backend.is_healthy().await {
                emit(
                    "mcp_query_failed",
                    Some(&trace_id),
                    json!({
                        "model": model_name,
                        "error_type": "backend_unhealthy",
                    }),
                );
                return format!(
                    "error: backend '{}' is not healthy — try `unload_model` then `load_model` to restart",
                    model_name
                );
            }
            // Merge the per-model sampling profile (if any) with the
            // caller's request values. Precedence: locked profile field >
            // request value > unlocked profile field > builtin default.
            // temperature/max_tokens are passed positionally so they
            // collapse to a concrete value (builtin default = 16384/0.7);
            // top_p/top_k/min_p/repeat_penalty stay Option so the backend
            // only sends them downstream when actually set.
            let (max_tokens, temperature, opts) = match sampling.as_ref() {
                Some(p) => {
                    let mt = p.max_tokens(req_max_tokens, 16384);
                    let temp = p.temperature(req_temperature, 0.7);
                    let opts = lamu_core::backends::GenerateOpts {
                        enable_thinking,
                        top_p: p.resolve_top_p(req_top_p),
                        top_k: p.resolve_top_k(req_top_k),
                        min_p: p.resolve_min_p(req_min_p),
                        repeat_penalty: p.resolve_repeat_penalty(req_repeat_penalty),
                    };
                    (mt, temp, opts)
                }
                None => {
                    let opts = lamu_core::backends::GenerateOpts {
                        enable_thinking,
                        top_p: req_top_p,
                        top_k: req_top_k,
                        min_p: req_min_p,
                        repeat_penalty: req_repeat_penalty,
                    };
                    (req_max_tokens.unwrap_or(16384), req_temperature.unwrap_or(0.7), opts)
                }
            };
            match backend.generate_with_opts(chat_messages, max_tokens, temperature, opts).await {
                Ok(s) => s,
                Err(e) => {
                    emit(
                        "mcp_query_failed",
                        Some(&trace_id),
                        json!({
                            "model": model_name,
                            "error_type": "backend_generate",
                            "error": format!("{e}"),
                        }),
                    );
                    return format!("error: generation failed: {}", e);
                }
            }
        };

        // Empty Ok("") from a backend means the response parsed but
        // had no content — surface as an error instead of returning
        // a silent blank string. Backend impls returning Err already
        // path through the match arm above.
        if raw.is_empty() {
            emit(
                "mcp_query_failed",
                Some(&trace_id),
                json!({"model": model_name, "error_type": "backend_empty_response"}),
            );
            return "error: backend returned empty response".into();
        }

        // LlamaCppBackend / Megakernel / Dflash all return either
        // "content" or "<think>{reasoning}</think>\n{content}".
        // The reasoning extractor handles both shapes.
        let extractor = get_extractor(marker);
        let (reasoning, content_clean) = extractor.split(&raw);

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

        // Routing-mode gate: refuse under 'cloud-only'. set_routing_mode
        // ('cloud-only') drains backends + frees VRAM precisely so the
        // GPU is available for other work; letting load_model re-spawn a
        // llama-server here would silently re-allocate that VRAM and
        // defeat the documented "free GPU" guarantee. handle_query
        // already refuses local queries under cloud-only — this closes
        // the load path too.
        if self.routing_mode.lock().await.as_str() == "cloud-only" {
            return "error: routing mode is 'cloud-only' — load_model refused (would re-allocate the VRAM you freed). Call set_routing_mode(mode='auto') first.".into();
        }

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
            let mut evict_backends: Vec<(String, crate::server::BackendHandle)> = Vec::new();
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
            // Insert into the backends map BEFORE confirm_loaded.
            // Once Backend::generate is wired in (next step), a query
            // arriving between confirm_loaded and backends.insert would
            // see the scheduler say "loaded" but find no backend in the
            // map. Doing both inside one lock acquisition with the
            // insert first removes the race entirely.
            st.backends.insert(entry.name.clone(), Arc::new(AsyncMutex::new(backend)));
            let _ = st.scheduler.confirm_loaded(&entry.name, pid, port, vram);
            st.health.get_or_create(&entry.name).record_success();
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
        let mut to_unload: Vec<(String, crate::server::BackendHandle)> = Vec::new();
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
            .unwrap_or("mimo-v2.5").to_string();
        let default_system = args["default_system"].as_str().unwrap_or("").to_string();
        let user_max = args["max_concurrency"].as_u64().map(|n| n as usize);

        // Routing-mode gate: under 'local-only' refuse the cloud tasks in
        // the batch (but keep running local ones). parallel_query calls
        // handle_cloud_query directly, bypassing the dispatcher's
        // is_cloud_tool gate, so it must enforce local-only itself.
        let local_only = self.routing_mode.lock().await.as_str() == "local-only";

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
            if is_cloud && local_only {
                prepared.push(Err(format!(
                    "task[{}]: cloud model '{}' refused — routing mode is 'local-only'",
                    idx, model
                )));
                continue;
            }
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
            Err(e) => return format!("error: scan: {}", e),
        };
        if let Err(e) = write_registry(&entries, &st.registry_path) {
            return format!("error: write registry: {}", e);
        }
        st.entries = entries.iter().map(|e| (e.name.clone(), e.clone())).collect();
        st.router.update_registry(entries.clone());
        format!(
            "Scanned {}: {} models found. Registry updated.",
            st.models_dir.display(), entries.len()
        )
    }
}
