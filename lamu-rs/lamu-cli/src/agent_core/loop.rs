//! Parameterized agent loop — protocol-agnostic core of ACP and A2A turns.
//!
//! `run_prompt_turn` was extracted from `acp/agent_loop.rs` and made generic
//! over [`UpdateSink`] + [`PermissionGate`] so both ACP and A2A drive the same
//! model-interaction loop without sharing wire shapes (ADR 0038).
//!
//! The ACP surface wires in its mpsc/oneshot adapters; A2A wires in its SSE
//! sink and `DenyWrites` gate.

use super::{LoopEvent, PermissionDecision, PermissionGate, ToolStatus, UpdateSink};
use crate::acp::agent_loop::{
    mcp_to_openai_tool, mcp_tool_entry, prompt_blocks_to_text,
    wait_cancelled, ToolAcc, WRITE_EFFECTING_TOOLS,
};
use lamu_core::sse::next_sse_line;
use lamu_core::tools_ext::ToolCtx;
use lamu_mcp::server::LamuMcpServer;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::watch;

/// Minimal config the loop needs. Supplied by the surface (ACP config or A2A
/// config). Kept here as a plain struct so neither surface has to import the
/// other's config type.
pub struct LoopConfig {
    /// Test seam: when `Some`, bypass registry/ensure-load and POST here directly.
    pub chat_url: Option<String>,
    /// Model name to load/use.
    pub model: String,
    /// Cap on inner model-request iterations.
    pub max_turns: usize,
    /// `max_tokens` per model request.
    pub max_tokens: u32,
}

/// Tool subset for the loop. Usually `CURATED_TOOLS` (ACP) or a subset that
/// excludes `write_file` (A2A). The list is filtered against the live registry
/// — tools absent from the registry are silently dropped.
pub type ToolSubset<'a> = &'a [&'a str];

/// Why a turn ended (surface-neutral; surfaces translate to their own wire enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnStop {
    EndTurn,
    Cancelled,
    MaxTurnRequests,
}

const RAW_OUTPUT_MAX: usize = 4096;
const TOOL_RESULT_HISTORY_MAX: usize = 16384;

fn tool_kind(name: &str) -> &'static str {
    match name {
        "write_file" => "edit",
        "web_search" | "research" | "deep_research" => "fetch",
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
        "deep_research" => "Deep research".to_string(),
        "recall_memory" => "Recall memory".to_string(),
        "remember" => "Remember".to_string(),
        other => other.to_string(),
    }
}

fn system_prompt_for(tools: &[Value]) -> String {
    let names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str())
        .collect();
    format!(
        "You are LAMU, a local model agent. Be direct and concise. \
         Available tools: {}. Call a tool only when the task needs it; \
         otherwise answer from context. After tool results arrive, give \
         the user a clear final answer.",
        names.join(", ")
    )
}

/// Build the OpenAI tool list from a curated name subset.
fn tools_for_subset(subset: ToolSubset<'_>) -> Vec<Value> {
    subset
        .iter()
        .filter_map(|name| mcp_tool_entry(name).map(|e| mcp_to_openai_tool(&e)))
        .collect()
}

struct StreamOutcome {
    content: String,
    tool_calls: Vec<ToolAcc>,
    cancelled: bool,
}

/// Run one prompt turn, parameterized on sink + gate.
///
/// `history` is the in/out conversation buffer (OpenAI message array). The
/// caller owns it across turns (multi-turn A2A context, ACP session history).
///
/// `Err(msg)` means a hard failure (model not reachable, etc.) — the surface
/// should map it to an error response.
pub async fn run_prompt_turn(
    mcp: &Arc<LamuMcpServer>,
    http: &Client,
    cfg: &LoopConfig,
    history: &tokio::sync::Mutex<Vec<Value>>,
    prompt_blocks: Vec<Value>,
    subset: ToolSubset<'_>,
    sink: &dyn UpdateSink,
    gate: &dyn PermissionGate,
    mut cancel: watch::Receiver<bool>,
) -> Result<TurnStop, String> {
    let user_text = prompt_blocks_to_text(&prompt_blocks);
    history.lock().await.push(json!({ "role": "user", "content": user_text }));

    let url = match &cfg.chat_url {
        Some(u) => u.clone(),
        None => {
            let ctx: &dyn ToolCtx = &**mcp;
            ctx.ensure_loaded(&cfg.model)
                .await
                .map_err(|e| format!("load model '{}': {e}", cfg.model))?;
            let port = ctx
                .loaded_port(&cfg.model)
                .ok_or_else(|| format!("model '{}' loaded but has no live port", cfg.model))?;
            format!("http://localhost:{port}/v1/chat/completions")
        }
    };

    let tools = tools_for_subset(subset);
    let system = system_prompt_for(&tools);
    let mut call_counter: usize = 0;

    for _turn in 0..cfg.max_turns {
        if *cancel.borrow() {
            return Ok(TurnStop::Cancelled);
        }

        let mut messages = vec![json!({ "role": "system", "content": system })];
        messages.extend(history.lock().await.iter().cloned());
        let mut payload = json!({
            "model": cfg.model,
            "messages": messages,
            "stream": true,
            "max_tokens": cfg.max_tokens,
        });
        if !tools.is_empty() {
            payload["tools"] = json!(tools);
            payload["tool_choice"] = json!("auto");
        }

        let outcome = stream_chat(http, &url, &payload, sink, &mut cancel).await?;
        if outcome.cancelled {
            if !outcome.content.is_empty() {
                history
                    .lock()
                    .await
                    .push(json!({ "role": "assistant", "content": outcome.content }));
            }
            return Ok(TurnStop::Cancelled);
        }

        if outcome.tool_calls.is_empty() {
            history
                .lock()
                .await
                .push(json!({ "role": "assistant", "content": outcome.content }));
            return Ok(TurnStop::EndTurn);
        }

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
            let mut h = history.lock().await;
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
                execute_tool_call(mcp, cfg, &id, &acc, sink, gate, subset, &mut cancel).await;
            history.lock().await.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": truncate(&result, TOOL_RESULT_HISTORY_MAX),
            }));
            if cancelled {
                return Ok(TurnStop::Cancelled);
            }
        }
    }

    Ok(TurnStop::MaxTurnRequests)
}

async fn stream_chat(
    http: &Client,
    url: &str,
    payload: &Value,
    sink: &dyn UpdateSink,
    cancel: &mut watch::Receiver<bool>,
) -> Result<StreamOutcome, String> {
    use futures_util::StreamExt;

    let mut outcome = StreamOutcome {
        content: String::new(),
        tool_calls: Vec::new(),
        cancelled: false,
    };

    let send = http.post(url).json(payload).send();
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
            if process_sse_line(&line, &mut outcome, &mut acc, sink) {
                break 'read;
            }
        }
    }

    outcome.tool_calls = acc.into_values().filter(|t| !t.name.is_empty()).collect();
    Ok(outcome)
}

fn process_sse_line(
    raw: &str,
    outcome: &mut StreamOutcome,
    acc: &mut BTreeMap<usize, ToolAcc>,
    sink: &dyn UpdateSink,
) -> bool {
    let line = raw.trim();
    let Some(data) = line.strip_prefix("data:") else {
        return false;
    };
    let data = data.trim();
    if data == "[DONE]" {
        return true;
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return false;
    };
    let Some(choice) = v.get("choices").and_then(|c| c.get(0)) else {
        return false;
    };
    if let Some(delta) = choice.get("delta") {
        if let Some(r) = delta.get("reasoning_content").and_then(|x| x.as_str()) {
            if !r.is_empty() {
                let _ = sink.emit(LoopEvent::ThoughtChunk(r.to_string()));
            }
        }
        if let Some(c) = delta.get("content").and_then(|x| x.as_str()) {
            if !c.is_empty() {
                let _ = sink.emit(LoopEvent::MessageChunk(c.to_string()));
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

async fn execute_tool_call(
    mcp: &Arc<LamuMcpServer>,
    cfg: &LoopConfig,
    id: &str,
    acc: &ToolAcc,
    sink: &dyn UpdateSink,
    gate: &dyn PermissionGate,
    subset: ToolSubset<'_>,
    cancel: &mut watch::Receiver<bool>,
) -> (String, bool) {
    let args: Value = if acc.args.trim().is_empty() {
        json!({})
    } else {
        match serde_json::from_str(&acc.args) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("error: invalid tool arguments JSON: {e}");
                let _ = sink.emit(LoopEvent::ToolCall {
                    id: id.to_string(),
                    title: tool_title(&acc.name, &json!(acc.args)),
                    kind: tool_kind(&acc.name).to_string(),
                    raw_input: json!(acc.args),
                });
                let _ = sink.emit(LoopEvent::ToolCallUpdate {
                    id: id.to_string(),
                    status: ToolStatus::Failed,
                    raw_output: Some(json!({ "error": msg })),
                });
                return (msg, false);
            }
        }
    };
    let title = tool_title(&acc.name, &args);
    let kind = tool_kind(&acc.name).to_string();

    // Fail closed: if the model calls a tool not in the subset, reject it.
    if !subset.contains(&acc.name.as_str()) {
        let msg = format!("error: tool '{}' is not available in this context", acc.name);
        let _ = sink.emit(LoopEvent::ToolCall {
            id: id.to_string(),
            title: title.clone(),
            kind: kind.clone(),
            raw_input: args.clone(),
        });
        let _ = sink.emit(LoopEvent::ToolCallUpdate {
            id: id.to_string(),
            status: ToolStatus::Failed,
            raw_output: Some(json!({ "error": msg })),
        });
        return (msg, false);
    }

    let _ = sink.emit(LoopEvent::ToolCall {
        id: id.to_string(),
        title: title.clone(),
        kind: kind.clone(),
        raw_input: args.clone(),
    });

    if WRITE_EFFECTING_TOOLS.contains(&acc.name.as_str()) {
        match gate.request(&acc.name, &args, cancel).await {
            PermissionDecision::Allowed => {}
            PermissionDecision::Rejected => {
                let msg = format!("error: user rejected {} via permission prompt", acc.name);
                let _ = sink.emit(LoopEvent::ToolCallUpdate {
                    id: id.to_string(),
                    status: ToolStatus::Failed,
                    raw_output: Some(json!({ "error": msg })),
                });
                return (msg, false);
            }
            PermissionDecision::CancelledTurn => {
                let _ = sink.emit(LoopEvent::ToolCallUpdate {
                    id: id.to_string(),
                    status: ToolStatus::Failed,
                    raw_output: Some(json!({ "error": "cancelled" })),
                });
                return ("error: turn cancelled".into(), true);
            }
        }
    }

    let _ = sink.emit(LoopEvent::ToolCallUpdate {
        id: id.to_string(),
        status: ToolStatus::InProgress,
        raw_output: None,
    });

    // Inject model name for `query` tool if not already specified.
    let mut dispatch_args = args.clone();
    if acc.name == "query" {
        if let Some(obj) = dispatch_args.as_object_mut() {
            obj.entry("model".to_string()).or_insert_with(|| json!(cfg.model));
        }
    }

    let exec = mcp.dispatch_tool_text(&acc.name, dispatch_args);
    let result = tokio::select! {
        r = exec => r,
        _ = wait_cancelled(cancel) => {
            let _ = sink.emit(LoopEvent::ToolCallUpdate {
                id: id.to_string(),
                status: ToolStatus::Failed,
                raw_output: Some(json!({ "error": "cancelled" })),
            });
            return ("error: turn cancelled".into(), true);
        }
    };

    let failed = lamu_mcp::server::is_tool_error_text(&result);
    let _ = sink.emit(LoopEvent::ToolCallUpdate {
        id: id.to_string(),
        status: if failed { ToolStatus::Failed } else { ToolStatus::Completed },
        raw_output: Some(json!({ "output": truncate(&result, RAW_OUTPUT_MAX) })),
    });
    (result, false)
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
