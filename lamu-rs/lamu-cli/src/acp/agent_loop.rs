//! The ACP prompt-turn agent loop (ADR 0036).
//!
//! One `session/prompt` = one turn: build OpenAI-shaped messages from the
//! session history + the new content blocks, POST the loaded model's
//! `/v1/chat/completions` with `stream=true` and the curated tool subset,
//! forward `delta.content` as `agent_message_chunk` and
//! `delta.reasoning_content` (the ADR 0037 serve-side reasoning split)
//! as `agent_thought_chunk`, accumulate `tool_calls` deltas (the lamu-api
//! anthropic bridge's `ToolAcc` pattern), execute finished tool calls via
//! the in-process MCP dispatch, and loop until the model stops, the turn
//! is cancelled, or `max_turns` model requests have been made
//! (→ `max_turn_requests`, the spec's stop reason for the iteration cap).

use super::protocol::{self, StopReason};
use super::{AcpServer, AcpSession, PermissionDecision};
use lamu_core::sse::next_sse_line;
use lamu_core::tools_ext::ToolCtx;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use tokio::sync::watch;

/// Curated subset of the MCP tool surface exposed to the model. Kept
/// deliberately small: an editor agent wants research + memory + a write
/// primitive, not the full ops catalog (load_model, vram_status, …).
/// `web_search`/`research` are lamu-jart module tools — present only when
/// the composition root registered them (main() does; their absence just
/// shrinks the advertised set).
pub(crate) const CURATED_TOOLS: &[&str] =
    &["query", "web_search", "research", "recall_memory", "remember", "write_file"];

/// Tools that mutate something the client can see — gated behind
/// `session/request_permission` before execution.
pub(crate) const WRITE_EFFECTING_TOOLS: &[&str] = &["write_file"];

/// rawOutput cap in `tool_call_update` notifications.
const RAW_OUTPUT_MAX: usize = 4096;
/// Tool-result cap in the conversation history (context protection).
const TOOL_RESULT_HISTORY_MAX: usize = 16384;

// ── Tool schema translation (MCP → OpenAI) ──────────────────────────

/// Find a tool's MCP catalog entry `{name, description, inputSchema}` —
/// built-in TOOLS table first, then the ADR 0023 module-tool registry.
pub(crate) fn mcp_tool_entry(name: &str) -> Option<Value> {
    if let Some(t) = lamu_mcp::tools::find(name) {
        return Some(t.to_list_entry());
    }
    lamu_core::tools_ext::list_entries()
        .into_iter()
        .find(|e| e.get("name").and_then(|n| n.as_str()) == Some(name))
}

/// Translate one MCP tool entry into the OpenAI function-tool shape.
/// `session_id` is stripped from the parameters: the ACP layer injects
/// the ACP session id at dispatch (the model never chooses it).
pub(crate) fn mcp_to_openai_tool(entry: &Value) -> Value {
    let mut params = entry.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object"}));
    if let Some(props) = params.get_mut("properties").and_then(|p| p.as_object_mut()) {
        props.remove("session_id");
    }
    if let Some(req) = params.get_mut("required").and_then(|r| r.as_array_mut()) {
        req.retain(|v| v != "session_id");
    }
    json!({
        "type": "function",
        "function": {
            "name": entry.get("name").cloned().unwrap_or(Value::Null),
            "description": entry.get("description").cloned().unwrap_or(Value::Null),
            "parameters": params,
        }
    })
}

/// The curated tool subset in OpenAI shape (skips tools that aren't
/// registered in this process).
pub(crate) fn curated_openai_tools() -> Vec<Value> {
    CURATED_TOOLS
        .iter()
        .filter_map(|name| mcp_tool_entry(name).map(|e| mcp_to_openai_tool(&e)))
        .collect()
}

/// ACP `ToolKind` for a LAMU tool (drives the client's icon/grouping).
fn tool_kind(name: &str) -> &'static str {
    match name {
        "write_file" => "edit",
        "web_search" | "research" => "fetch",
        "recall_memory" => "search",
        "query" => "think",
        _ => "other",
    }
}

fn tool_title(name: &str, args: &Value) -> String {
    match name {
        "write_file" => match args.get("path").and_then(|p| p.as_str()) {
            Some(p) => format!("Write {p}"),
            None => "Write file".to_string(),
        },
        "query" => "Query local model".to_string(),
        "web_search" => "Search the web".to_string(),
        "research" => "Research".to_string(),
        "recall_memory" => "Recall memory".to_string(),
        "remember" => "Remember".to_string(),
        other => other.to_string(),
    }
}

fn system_prompt(tools: &[Value]) -> String {
    let names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str())
        .collect();
    format!(
        "You are LAMU, a local model agent embedded in the user's editor via ACP. \
         Be direct and concise. Available tools: {}. Call a tool only when the task \
         needs it; otherwise answer from context. After tool results arrive, give the \
         user a clear final answer.",
        names.join(", ")
    )
}

// ── Cancellation ────────────────────────────────────────────────────

/// Resolve when the turn's cancel flag flips true. If the sender is
/// dropped without flipping (turn finished normally elsewhere), pend
/// forever — callers only use this inside `select!`.
pub(crate) async fn wait_cancelled(rx: &mut watch::Receiver<bool>) {
    if *rx.borrow() {
        return;
    }
    loop {
        if rx.changed().await.is_err() {
            std::future::pending::<()>().await;
        }
        if *rx.borrow() {
            return;
        }
    }
}

// ── Prompt-block folding ────────────────────────────────────────────

/// Fold the ACP prompt content blocks into one model-facing user string.
/// `text` concatenates; `resource` (embeddedContext, advertised true)
/// becomes a fenced context block; `resource_link` becomes a reference
/// line; image/audio are advertised unsupported, so any that arrive
/// anyway degrade to a placeholder.
pub(crate) fn prompt_blocks_to_text(blocks: &[Value]) -> String {
    let mut out = String::new();
    for b in blocks {
        match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(t);
                }
            }
            Some("resource") => {
                let res = b.get("resource").cloned().unwrap_or(Value::Null);
                let uri = res.get("uri").and_then(|u| u.as_str()).unwrap_or("untitled");
                if let Some(text) = res.get("text").and_then(|t| t.as_str()) {
                    out.push_str(&format!("\n<context uri=\"{uri}\">\n{text}\n</context>\n"));
                } else {
                    // blob resource — not text-foldable.
                    out.push_str(&format!("\n[binary resource: {uri}]\n"));
                }
            }
            Some("resource_link") => {
                let uri = b.get("uri").and_then(|u| u.as_str()).unwrap_or("?");
                out.push_str(&format!("\n[linked resource: {uri}]\n"));
            }
            Some(other) => out.push_str(&format!("\n[unsupported content block: {other}]\n")),
            None => {}
        }
    }
    out
}

// ── Streamed tool-call accumulation (lamu-api ToolAcc pattern) ──────

/// One streamed OpenAI tool call: first delta carries `id` +
/// `function.name`, later deltas append `function.arguments` fragments.
#[derive(Default, Clone)]
pub(crate) struct ToolAcc {
    pub id: String,
    pub name: String,
    pub args: String,
}

struct StreamOutcome {
    content: String,
    tool_calls: Vec<ToolAcc>,
    cancelled: bool,
}

// ── The turn ────────────────────────────────────────────────────────

/// Run one prompt turn. `Err` becomes a JSON-RPC error response to the
/// `session/prompt` request (e.g. the model can't be loaded); `Ok`
/// carries the wire stop reason.
pub(crate) async fn run_prompt_turn(
    server: &AcpServer,
    session: &AcpSession,
    prompt_blocks: Vec<Value>,
    mut cancel: watch::Receiver<bool>,
) -> Result<StopReason, String> {
    let user_text = prompt_blocks_to_text(&prompt_blocks);
    session.history.lock().await.push(json!({ "role": "user", "content": user_text }));

    // Resolve the chat endpoint: test override, or ensure-load the
    // session model and take its live port (same resolution ToolCtx::
    // generate / `lamu repl` rely on).
    let url = match &server.cfg.chat_url {
        Some(u) => u.clone(),
        None => {
            let ctx: &dyn ToolCtx = &*server.mcp;
            ctx.ensure_loaded(&session.model)
                .await
                .map_err(|e| format!("load model '{}': {e}", session.model))?;
            let port = ctx
                .loaded_port(&session.model)
                .ok_or_else(|| format!("model '{}' loaded but has no live port", session.model))?;
            format!("http://localhost:{port}/v1/chat/completions")
        }
    };

    let tools = curated_openai_tools();
    let system = system_prompt(&tools);
    let mut call_counter: usize = 0;

    for _turn in 0..server.cfg.max_turns {
        if *cancel.borrow() {
            return Ok(StopReason::Cancelled);
        }

        let mut messages = vec![json!({ "role": "system", "content": system })];
        messages.extend(session.history.lock().await.iter().cloned());
        let mut payload = json!({
            "model": session.model,
            "messages": messages,
            "stream": true,
            "max_tokens": server.cfg.max_tokens,
        });
        if !tools.is_empty() {
            payload["tools"] = json!(tools);
            payload["tool_choice"] = json!("auto");
        }

        let outcome = stream_chat(server, session, &url, &payload, &mut cancel).await?;
        if outcome.cancelled {
            // Keep whatever visible text was produced so the next turn
            // has the partial context.
            if !outcome.content.is_empty() {
                session
                    .history
                    .lock()
                    .await
                    .push(json!({ "role": "assistant", "content": outcome.content }));
            }
            return Ok(StopReason::Cancelled);
        }

        if outcome.tool_calls.is_empty() {
            session
                .history
                .lock()
                .await
                .push(json!({ "role": "assistant", "content": outcome.content }));
            return Ok(StopReason::EndTurn);
        }

        // Assistant message carrying the tool calls (OpenAI shape), then
        // one `tool` message per executed call.
        let mut wire_calls = Vec::new();
        let mut calls = Vec::new();
        for acc in &outcome.tool_calls {
            call_counter += 1;
            let id = if acc.id.is_empty() {
                format!("call_{call_counter}")
            } else {
                acc.id.clone()
            };
            wire_calls.push(json!({
                "id": id,
                "type": "function",
                "function": { "name": acc.name, "arguments": acc.args },
            }));
            calls.push((id, acc.clone()));
        }
        {
            let mut h = session.history.lock().await;
            let content = if outcome.content.is_empty() {
                Value::Null
            } else {
                Value::String(outcome.content.clone())
            };
            h.push(json!({
                "role": "assistant",
                "content": content,
                "tool_calls": wire_calls,
            }));
        }

        for (id, acc) in calls {
            let (result, cancelled) =
                execute_tool_call(server, session, &id, &acc, &mut cancel).await;
            session.history.lock().await.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": truncate(&result, TOOL_RESULT_HISTORY_MAX),
            }));
            if cancelled {
                return Ok(StopReason::Cancelled);
            }
        }
        // Loop: feed the tool results back to the model.
    }

    Ok(StopReason::MaxTurnRequests)
}

/// POST one streaming chat completion and forward deltas as updates.
async fn stream_chat(
    server: &AcpServer,
    session: &AcpSession,
    url: &str,
    payload: &Value,
    cancel: &mut watch::Receiver<bool>,
) -> Result<StreamOutcome, String> {
    use futures_util::StreamExt;

    let mut outcome = StreamOutcome {
        content: String::new(),
        tool_calls: Vec::new(),
        cancelled: false,
    };

    let send = server.http.post(url).json(payload).send();
    let resp = tokio::select! {
        r = send => r.map_err(|e| format!("chat backend: {e}"))?,
        _ = wait_cancelled(cancel) => { outcome.cancelled = true; return Ok(outcome); }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("chat backend {status}: {}", truncate(&body, 512)));
    }

    let mut byte_stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    // BTreeMap keyed by delta index keeps the backend's call order.
    let mut acc: BTreeMap<usize, ToolAcc> = BTreeMap::new();

    'read: loop {
        let chunk = tokio::select! {
            c = byte_stream.next() => c,
            _ = wait_cancelled(cancel) => { outcome.cancelled = true; break 'read; }
        };
        let Some(chunk) = chunk else { break 'read };
        let Ok(bytes) = chunk else { break 'read };
        buf.extend_from_slice(&bytes);
        while let Some(line) = next_sse_line(&mut buf) {
            if process_sse_line(server, session, &line, &mut outcome, &mut acc) {
                break 'read; // [DONE]
            }
        }
    }

    outcome.tool_calls = acc.into_values().filter(|t| !t.name.is_empty()).collect();
    Ok(outcome)
}

/// Handle one SSE line; returns true on `[DONE]`.
fn process_sse_line(
    server: &AcpServer,
    session: &AcpSession,
    raw: &str,
    outcome: &mut StreamOutcome,
    acc: &mut BTreeMap<usize, ToolAcc>,
) -> bool {
    let line = raw.trim();
    let Some(data) = line.strip_prefix("data:") else {
        return false; // comments / event: lines / blanks
    };
    let data = data.trim();
    if data == "[DONE]" {
        return true;
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return false;
    };
    let Some(choice) = v.get("choices").and_then(|c| c.get(0)) else {
        return false; // usage-only chunk
    };
    if let Some(delta) = choice.get("delta") {
        // ADR 0037: the serve-side reasoning split surfaces structured
        // `delta.reasoning_content`; forward it as thought chunks,
        // visibly separate from the message stream.
        if let Some(r) = delta.get("reasoning_content").and_then(|x| x.as_str()) {
            if !r.is_empty() {
                server.send_update(&session.id, protocol::agent_thought_chunk(r));
            }
        }
        if let Some(c) = delta.get("content").and_then(|x| x.as_str()) {
            if !c.is_empty() {
                server.send_update(&session.id, protocol::agent_message_chunk(c));
                outcome.content.push_str(c);
            }
        }
        if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
            for tc in tcs {
                let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let slot = acc.entry(idx).or_default();
                if let Some(id) = tc.get("id").and_then(|x| x.as_str()) {
                    if !id.is_empty() {
                        slot.id = id.to_string();
                    }
                }
                if let Some(name) =
                    tc.get("function").and_then(|f| f.get("name")).and_then(|x| x.as_str())
                {
                    if !name.is_empty() {
                        slot.name = name.to_string();
                    }
                }
                if let Some(a) =
                    tc.get("function").and_then(|f| f.get("arguments")).and_then(|x| x.as_str())
                {
                    slot.args.push_str(a);
                }
            }
        }
    }
    false
}

/// Execute one finished tool call: permission gate (write-effecting
/// only), in-process dispatch, lifecycle notifications. Returns
/// `(result_text_for_history, turn_cancelled)`.
async fn execute_tool_call(
    server: &AcpServer,
    session: &AcpSession,
    id: &str,
    acc: &ToolAcc,
    cancel: &mut watch::Receiver<bool>,
) -> (String, bool) {
    let args: Value = if acc.args.trim().is_empty() {
        json!({})
    } else {
        match serde_json::from_str(&acc.args) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("error: invalid tool arguments JSON: {e}");
                server.send_update(
                    &session.id,
                    protocol::tool_call(id, &tool_title(&acc.name, &json!({})), tool_kind(&acc.name), &json!(acc.args)),
                );
                server.send_update(
                    &session.id,
                    protocol::tool_call_update(id, "failed", Some(json!({ "error": msg }))),
                );
                return (msg, false);
            }
        }
    };
    let title = tool_title(&acc.name, &args);
    let kind = tool_kind(&acc.name);

    server.send_update(&session.id, protocol::tool_call(id, &title, kind, &args));

    if WRITE_EFFECTING_TOOLS.contains(&acc.name.as_str()) {
        match server
            .request_permission(session, &acc.name, id, &title, kind, &args, cancel)
            .await
        {
            PermissionDecision::Allowed => {}
            PermissionDecision::Rejected => {
                let msg = format!("error: user rejected {} via permission prompt", acc.name);
                server.send_update(
                    &session.id,
                    protocol::tool_call_update(id, "failed", Some(json!({ "error": msg }))),
                );
                return (msg, false);
            }
            PermissionDecision::CancelledTurn => {
                server.send_update(
                    &session.id,
                    protocol::tool_call_update(id, "failed", Some(json!({ "error": "cancelled" }))),
                );
                return ("error: turn cancelled".into(), true);
            }
        }
    }

    server.send_update(&session.id, protocol::tool_call_update(id, "in_progress", None));

    let exec = dispatch_tool(server, session, &acc.name, args);
    let result = tokio::select! {
        r = exec => r,
        _ = wait_cancelled(cancel) => {
            server.send_update(
                &session.id,
                protocol::tool_call_update(id, "failed", Some(json!({ "error": "cancelled" }))),
            );
            return ("error: turn cancelled".into(), true);
        }
    };

    // The MCP dispatcher's exact failure heuristic — shared so the two
    // frontends can never disagree on what a failed tool call is.
    let failed = lamu_mcp::server::is_tool_error_text(&result);
    server.send_update(
        &session.id,
        protocol::tool_call_update(
            id,
            if failed { "failed" } else { "completed" },
            Some(json!({ "output": truncate(&result, RAW_OUTPUT_MAX) })),
        ),
    );
    (result, false)
}

/// Route one tool execution. `write_file` goes through the client's
/// filesystem (`fs/write_text_file`) when the client advertised the
/// capability — the client sees/journals the edit natively; otherwise it
/// falls back to the local journaled `write_file` handler with the ACP
/// session id injected (note: that handler resolves relative to the
/// *process* cwd, so spawn `lamu acp` in the project root — doc'd
/// follow-up is threading the session cwd through the journal seam).
/// Everything else uses `LamuMcpServer::dispatch_tool_text`, the exact
/// MCP `tools/call` path (local-only gate + module-tool fallback).
async fn dispatch_tool(
    server: &AcpServer,
    session: &AcpSession,
    name: &str,
    mut args: Value,
) -> String {
    if name == "write_file" {
        let client_fs = server.client_caps.lock().expect("acp caps lock").fs.write_text_file;
        if client_fs {
            let Some(path) = args.get("path").and_then(|p| p.as_str()).map(str::to_string) else {
                return "error: write_file: path is required".into();
            };
            let content = args
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let abs = if std::path::Path::new(&path).is_absolute() {
                std::path::PathBuf::from(&path)
            } else {
                session.cwd.join(&path)
            };
            let params = json!({
                "sessionId": session.id,
                "path": abs,
                "content": content,
            });
            let (req_id, rx) =
                server.begin_client_request(protocol::M_FS_WRITE_TEXT_FILE, params);
            return match rx.await {
                Ok(Ok(_)) => {
                    format!("wrote {} bytes to {} (client fs)", content.len(), abs.display())
                }
                Ok(Err(e)) => format!(
                    "error: fs/write_text_file: {}",
                    e.get("message").and_then(|m| m.as_str()).unwrap_or("client error")
                ),
                Err(_) => {
                    server.forget_client_request(req_id);
                    "error: fs/write_text_file: client connection dropped".into()
                }
            };
        }
        // Local fallback: inject the ACP session id so the write lands in
        // the rollback journal under this session.
        if let Some(obj) = args.as_object_mut() {
            obj.insert("session_id".into(), json!(session.id));
        }
    }
    server.mcp.dispatch_tool_text(name, args).await
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… [truncated {} bytes]", &s[..end], s.len() - end)
}
