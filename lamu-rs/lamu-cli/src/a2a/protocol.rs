//! Hand-rolled A2A (Agent2Agent) protocol wire types — the v1.0.0 subset
//! the LAMU A2A server implements.
//!
//! # Spec version pinned
//!
//! **A2A Protocol Specification v1.0.0** — sourced from the canonical
//! specification site <https://a2a-protocol.org/latest/specification/> on
//! 2026-06-12. The spec identifies itself as "Latest Released Version: 1.0.0"
//! with the note "The specific version of the A2A protocol in use is
//! identified using the `Major.Minor` elements (e.g. `1.0`)."
//!
//! Normative protobuf source lives at:
//!   <https://github.com/google-a2a/A2A/blob/main/specification/a2a.proto>
//!
//! Only the HTTP/JSON-RPC surface is implemented (no gRPC). Hand-rolled on
//! the same pattern as ACP (`acp/protocol.rs`) — avoids pulling the `a2a-rs`
//! crate (new dep tree; ADR 0036 rationale applies equally here).
//!
//! Field casing: A2A uses `camelCase` JSON keys, `snake_case` enum values.
//! `messageId` is required by spec; we generate a random id when not supplied.
//! Unknown fields on input are ignored (permissive); omitted on output.

use serde_json::{json, Value};

// ── Spec-pinned method names ─────────────────────────────────────────

/// A2A JSON-RPC method: send a message, run to completion, return Task.
pub const M_MESSAGE_SEND: &str = "message/send";
/// A2A JSON-RPC method: send a message with SSE streaming.
pub const M_MESSAGE_STREAM: &str = "message/stream";
/// A2A JSON-RPC method: fetch a stored task by id.
pub const M_TASKS_GET: &str = "tasks/get";
/// A2A JSON-RPC method: request cancellation of a running task.
pub const M_TASKS_CANCEL: &str = "tasks/cancel";

// ── Well-known paths ─────────────────────────────────────────────────

/// Canonical A2A v1.0.0 well-known path (renamed from `agent.json` in 0.2.x).
pub const PATH_AGENT_CARD: &str = "/.well-known/agent-card.json";
/// 0.2.x well-known path, kept as an alias for older clients.
pub const PATH_AGENT_CARD_WK_COMPAT: &str = "/.well-known/agent.json";
/// Bare alias some clients probe.
pub const PATH_AGENT_CARD_COMPAT: &str = "/agent.json";
pub const PATH_RPC: &str = "/";

// ── Task states (spec §TaskState) ────────────────────────────────────

pub const STATE_SUBMITTED: &str = "submitted";
pub const STATE_WORKING: &str = "working";
pub const STATE_COMPLETED: &str = "completed";
pub const STATE_FAILED: &str = "failed";
pub const STATE_CANCELED: &str = "canceled";
pub const STATE_INPUT_REQUIRED: &str = "input-required";

// ── Agent Card builder ───────────────────────────────────────────────

/// Build the LAMU Agent Card (spec §AgentCard). Served at both
/// `/.well-known/agent-card.json` (v1.0.0 canonical), with
/// `/.well-known/agent.json` (0.2.x) and `/agent.json` as aliases.
///
/// `url` is the base URL of this A2A server (e.g. `http://127.0.0.1:8022`).
pub fn agent_card(url: &str, skills: &[Value]) -> Value {
    json!({
        // Required identity fields.
        "name": "LAMU",
        "description": "LAMU local model agent — research, memory, and query tools \
                        over a local LLM with MCP tool dispatch.",
        "url": url,
        "version": env!("CARGO_PKG_VERSION"),
        // Capabilities (spec §AgentCapabilities).
        "capabilities": {
            "streaming": true,
            "pushNotifications": false,
            "extendedAgentCard": false,
        },
        // Declared I/O modes (spec §AgentInterface). Text-only in v1.
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        // Skills: one per curated tool + a catch-all "chat" skill.
        "skills": skills,
        // No security schemes in v1 loopback-default mode.
        "securitySchemes": {},
        "security": [],
    })
}

/// Build the skill list for the agent card. One skill per curated tool name
/// plus a general "chat" skill.
pub fn build_skills(tool_names: &[&str]) -> Vec<Value> {
    let mut skills: Vec<Value> = tool_names
        .iter()
        .map(|name| {
            let (id, desc) = skill_meta(name);
            json!({
                "id": id,
                "name": id,
                "description": desc,
                "inputModes": ["text"],
                "outputModes": ["text"],
            })
        })
        .collect();
    // Always include a general chat skill.
    skills.push(json!({
        "id": "chat",
        "name": "chat",
        "description": "General-purpose chat with the local LAMU model.",
        "inputModes": ["text"],
        "outputModes": ["text"],
    }));
    skills
}

fn skill_meta(tool: &str) -> (&'static str, &'static str) {
    match tool {
        "query" => ("query", "Query the local model directly."),
        "web_search" => ("web_search", "Search the web for current information."),
        "research" => ("research", "In-depth research using multiple sources."),
        "deep_research" => (
            "deep_research",
            "Deep multi-source research with decomposition and verification.",
        ),
        "recall_memory" => ("recall_memory", "Search lifetime memory for relevant context."),
        "remember" => ("remember", "Store a fact or note in lifetime memory."),
        _ => ("unknown", ""),
    }
}

// ── JSON-RPC 2.0 envelope builders ──────────────────────────────────

pub fn rpc_error(id: &Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
}

pub fn rpc_result(id: &Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

// ── Task builder helpers ─────────────────────────────────────────────

/// Build a Task object. `state` is one of the `STATE_*` constants.
pub fn task(
    id: &str,
    context_id: &str,
    state: &str,
    message: Option<Value>,
    artifacts: Vec<Value>,
    history: Vec<Value>,
) -> Value {
    let status = if let Some(msg) = message {
        json!({ "state": state, "message": msg, "timestamp": utc_now() })
    } else {
        json!({ "state": state, "timestamp": utc_now() })
    };
    json!({
        "id": id,
        "contextId": context_id,
        "status": status,
        "artifacts": artifacts,
        "history": history,
    })
}

/// Build a Task with a simple text artifact (the completed-turn shape).
pub fn task_completed(
    id: &str,
    context_id: &str,
    text: &str,
    history: Vec<Value>,
) -> Value {
    let artifact = json!({
        "artifactId": format!("{id}-artifact"),
        "index": 0,
        "parts": [{ "kind": "text", "text": text }],
    });
    let status_msg = agent_message(text);
    task(
        id,
        context_id,
        STATE_COMPLETED,
        Some(status_msg),
        vec![artifact],
        history,
    )
}

// ── Message builders ─────────────────────────────────────────────────

/// Build an agent-role Message with a single text part.
pub fn agent_message(text: &str) -> Value {
    json!({
        "messageId": pseudo_id("msg-"),
        "role": "agent",
        "parts": [{ "kind": "text", "text": text }],
    })
}

/// Build a thought DataPart (streaming only — unknown parts ignored by
/// compliant clients, so this is safe to inject into the stream).
pub fn thought_part(text: &str) -> Value {
    json!({
        "kind": "data",
        "data": { "kind": "thought", "text": text },
        "metadata": { "internal": true },
    })
}

// ── SSE event builders ────────────────────────────────────────────────

/// `TaskStatusUpdateEvent` (spec §TaskStatusUpdateEvent).
pub fn status_update_event(task_id: &str, state: &str, message: Option<Value>) -> Value {
    let status = if let Some(msg) = message {
        json!({ "state": state, "message": msg, "timestamp": utc_now() })
    } else {
        json!({ "state": state, "timestamp": utc_now() })
    };
    json!({
        "taskId": task_id,
        "status": status,
    })
}

/// `TaskArtifactUpdateEvent` (spec §TaskArtifactUpdateEvent).
pub fn artifact_update_event(task_id: &str, artifact: Value) -> Value {
    json!({
        "taskId": task_id,
        "artifact": artifact,
        "timestamp": utc_now(),
    })
}

/// Wrap an event as an SSE `data:` line (JSON), ready for transmission.
/// SSE framing: `data: <json>\n\n`.
pub fn sse_line(event: &Value) -> String {
    format!("data: {}\n\n", event)
}

// ── Helpers ──────────────────────────────────────────────────────────

fn utc_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // ISO 8601 UTC without chrono dep: hand-build from epoch seconds.
    let (y, mo, d, h, mi, s) = epoch_to_parts(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn epoch_to_parts(mut secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let s = secs % 60; secs /= 60;
    let mi = secs % 60; secs /= 60;
    let h = secs % 24; secs /= 24;
    // Days since 1970-01-01. Gregorian calendar approximation.
    let mut days = secs;
    let mut y = 1970u64;
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let dy = if leap { 366 } else { 365 };
        if days < dy { break; }
        days -= dy;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let months = [31u64, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 1u64;
    for &dm in &months {
        if days < dm { break; }
        days -= dm;
        mo += 1;
    }
    (y, mo, days + 1, h, mi, s)
}

/// Generate a short opaque pseudo-id (no uuid dep; same hex pattern as ACP).
pub fn pseudo_id(prefix: &str) -> String {
    // 8 random bytes → 16 hex chars
    let mut buf = [0u8; 8];
    if getrandom::getrandom(&mut buf).is_err() {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        return format!("{prefix}{t:016x}");
    }
    format!("{prefix}{}", buf.iter().map(|b| format!("{b:02x}")).collect::<String>())
}

// ── Request deserializers ─────────────────────────────────────────────

/// Parse a `message/send` or `message/stream` request params.
/// Returns `(contextId, message_text, task_id_hint)`.
/// `contextId` defaults to a fresh id when absent (new context).
pub fn parse_send_params(params: &Value) -> Option<(String, String, Option<String>)> {
    let msg = params.get("message")?;
    // Extract text from the first text Part.
    let text = extract_text_from_message(msg)?;
    let context_id = params
        .get("contextId")
        .or_else(|| msg.get("contextId"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| pseudo_id("ctx-"));
    let task_id = params
        .get("taskId")
        .or_else(|| msg.get("taskId"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Some((context_id, text, task_id))
}

/// Extract the first text Part from a Message.
fn extract_text_from_message(msg: &Value) -> Option<String> {
    // Direct text field (simplified sender).
    if let Some(t) = msg.get("text").and_then(|v| v.as_str()) {
        return Some(t.to_string());
    }
    // Standard spec: parts array.
    let parts = msg.get("parts")?.as_array()?;
    for part in parts {
        // Spec v1.0.0 uses `kind: "text"` or legacy `type: "text"`.
        let is_text = part.get("kind").and_then(|v| v.as_str()) == Some("text")
            || part.get("type").and_then(|v| v.as_str()) == Some("text");
        if is_text {
            if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                return Some(t.to_string());
            }
        }
    }
    None
}

/// Parse a `tasks/get` request params → task id.
pub fn parse_get_params(params: &Value) -> Option<String> {
    params.get("id").and_then(|v| v.as_str()).map(str::to_string)
}

/// Parse a `tasks/cancel` request params → task id.
pub fn parse_cancel_params(params: &Value) -> Option<String> {
    params.get("id").and_then(|v| v.as_str()).map(str::to_string)
}
