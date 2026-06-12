//! Hand-rolled ACP (Agent Client Protocol) wire types — the v1 subset the
//! LAMU agent implements (ADR 0036).
//!
//! Shapes are pinned against Zed's official `agent-client-protocol` crate
//! v0.14.0 / `agent-client-protocol-schema` v0.13.6 (the protocol's source
//! of truth):
//!
//! - Framing: **newline-delimited JSON-RPC 2.0** over stdio. Verified
//!   against the official crate's `src/stdio.rs`, which writes each
//!   message as one line + `\n` and reads lines (NOT LSP `Content-Length`
//!   headers). Identical to the lamu-mcp stdio loop's framing.
//! - `protocolVersion` is an **integer** (`1` for the current stable
//!   protocol). Pre-release clients sent date strings — those parse as
//!   version 0 and we answer with our latest (the client disconnects if
//!   it can't speak it).
//! - Field casing: `camelCase` keys, `snake_case` enum values, tagged
//!   unions on `sessionUpdate` / `type` / `outcome`.
//!
//! Only the load-bearing fields are typed; permissive extras (`_meta`,
//! annotations, unknown fields) are ignored on input and omitted on
//! output, which the spec explicitly allows.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;

/// Latest ACP protocol version this agent speaks.
pub const PROTOCOL_VERSION: u64 = 1;

// ── Method names (client → agent) ───────────────────────────────────
pub const M_INITIALIZE: &str = "initialize";
pub const M_AUTHENTICATE: &str = "authenticate";
pub const M_SESSION_NEW: &str = "session/new";
pub const M_SESSION_PROMPT: &str = "session/prompt";
pub const M_SESSION_CANCEL: &str = "session/cancel";

// ── Method names (agent → client) ───────────────────────────────────
pub const M_SESSION_UPDATE: &str = "session/update";
pub const M_REQUEST_PERMISSION: &str = "session/request_permission";
pub const M_FS_WRITE_TEXT_FILE: &str = "fs/write_text_file";

/// `initialize` params (client → agent). Tolerant: `protocolVersion`
/// arrives as an integer on v1 clients, a date string on pre-release
/// ones (treated as 0).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct InitializeParams {
    pub protocol_version: Value,
    pub client_capabilities: ClientCapabilities,
}

impl InitializeParams {
    pub fn version(&self) -> u64 {
        self.protocol_version.as_u64().unwrap_or(0)
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ClientCapabilities {
    pub fs: FsCapabilities,
    pub terminal: bool,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FsCapabilities {
    pub read_text_file: bool,
    pub write_text_file: bool,
}

/// `initialize` result (agent → client). We only speak v1, so the answer
/// is always `1` regardless of the requested version (spec: respond with
/// the client's version if supported, else the agent's latest; the
/// client disconnects if it can't speak it).
pub fn initialize_result(_requested_version: u64) -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "agentCapabilities": {
            "loadSession": false,
            "promptCapabilities": {
                "image": false,
                "audio": false,
                // Resource content blocks in `session/prompt` are folded
                // into the model prompt as fenced context.
                "embeddedContext": true,
            },
        },
        "authMethods": [],
        "agentInfo": { "name": "lamu", "version": env!("CARGO_PKG_VERSION") },
    })
}

/// `session/new` params. `mcpServers` is accepted but ignored in v1 —
/// the agent's tool surface is the in-process LAMU dispatch, not
/// client-supplied MCP servers (doc-noted follow-up).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionParams {
    pub cwd: PathBuf,
    #[serde(default)]
    pub mcp_servers: Vec<Value>,
}

/// `session/prompt` params. Content blocks stay raw `Value`s — the
/// agent loop folds them to text leniently (unknown block types become
/// a placeholder instead of a hard parse error).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptParams {
    pub session_id: String,
    pub prompt: Vec<Value>,
}

/// `session/cancel` notification params.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelParams {
    pub session_id: String,
}

/// Why a prompt turn ended (`session/prompt` result `stopReason`).
/// Complete v1 wire enum; variants the loop can't currently produce
/// (`MaxTokens`) are kept so the type matches the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    /// The per-turn model-request cap (`AcpConfig::max_turns`) was hit.
    /// The spec defines this exact reason for the iteration cap, so the
    /// loop returns it rather than an `end_turn` with a prose note.
    MaxTurnRequests,
    Refusal,
    Cancelled,
}

pub fn prompt_result(stop: StopReason) -> Value {
    json!({ "stopReason": stop })
}

// ── session/update notification payload builders ────────────────────

/// Full `session/update` JSON-RPC notification.
pub fn session_update(session_id: &str, update: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": M_SESSION_UPDATE,
        "params": { "sessionId": session_id, "update": update },
    })
}

pub fn agent_message_chunk(text: &str) -> Value {
    json!({
        "sessionUpdate": "agent_message_chunk",
        "content": { "type": "text", "text": text },
    })
}

pub fn agent_thought_chunk(text: &str) -> Value {
    json!({
        "sessionUpdate": "agent_thought_chunk",
        "content": { "type": "text", "text": text },
    })
}

/// Initial `tool_call` update (status starts `pending`).
pub fn tool_call(id: &str, title: &str, kind: &str, raw_input: &Value) -> Value {
    json!({
        "sessionUpdate": "tool_call",
        "toolCallId": id,
        "title": title,
        "kind": kind,
        "status": "pending",
        "rawInput": raw_input,
    })
}

/// `tool_call_update` — status is one of `in_progress`/`completed`/`failed`.
pub fn tool_call_update(id: &str, status: &str, raw_output: Option<Value>) -> Value {
    let mut v = json!({
        "sessionUpdate": "tool_call_update",
        "toolCallId": id,
        "status": status,
    });
    if let Some(out) = raw_output {
        v["rawOutput"] = out;
    }
    v
}

// ── session/request_permission (agent → client request) ─────────────

/// Params for `session/request_permission`. `toolCall` is a
/// `ToolCallUpdate` shape (toolCallId + optional fields).
pub fn request_permission_params(
    session_id: &str,
    tool_call_id: &str,
    title: &str,
    kind: &str,
    raw_input: &Value,
) -> Value {
    json!({
        "sessionId": session_id,
        "toolCall": {
            "toolCallId": tool_call_id,
            "title": title,
            "kind": kind,
            "rawInput": raw_input,
        },
        "options": [
            { "optionId": "allow_once",    "name": "Allow once",    "kind": "allow_once" },
            { "optionId": "allow_always",  "name": "Always allow",  "kind": "allow_always" },
            { "optionId": "reject_once",   "name": "Reject once",   "kind": "reject_once" },
            { "optionId": "reject_always", "name": "Always reject", "kind": "reject_always" },
        ],
    })
}

/// Parse a `session/request_permission` result into the selected option
/// id; `None` means the client reported the turn cancelled (outcome
/// `cancelled`) or sent something unintelligible (treated as a reject).
pub fn permission_outcome_option(result: &Value) -> Option<String> {
    let outcome = result.get("outcome")?;
    if outcome.get("outcome").and_then(|v| v.as_str()) == Some("selected") {
        return outcome
            .get("optionId")
            .and_then(|v| v.as_str())
            .map(str::to_string);
    }
    None
}
