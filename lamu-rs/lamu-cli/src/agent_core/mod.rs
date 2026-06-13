//! Shared agent-loop traits (ADR 0038): `UpdateSink` + `PermissionGate`.
//!
//! Extracted from `acp/mod.rs` so the A2A surface can reuse `run_prompt_turn`
//! without depending on ACP's wire shapes. The ACP surface implements both
//! traits over its existing mpsc/oneshot channels; the A2A surface implements
//! them over per-task SSE streams.
//!
//! This is a **CLI-internal module** (lamu-cli/src/agent_core), NOT a crate.
//! Per ADR 0023/0029, frontends never depend on each other — A2A and ACP both
//! live inside lamu-cli (the composition root).

pub mod r#loop;

use async_trait::async_trait;
use serde_json::Value;

// ── LoopEvent ────────────────────────────────────────────────────────

/// Events the agent loop emits to whatever surface is watching.
#[derive(Debug, Clone)]
pub enum LoopEvent {
    /// A visible text chunk from the model's delta stream.
    MessageChunk(String),
    /// A reasoning/thought chunk (ADR 0037 `delta.reasoning_content`).
    ThoughtChunk(String),
    /// A tool call is starting: id, display title, and kind tag.
    ToolCall {
        id: String,
        title: String,
        kind: String,
        raw_input: Value,
    },
    /// A tool call changed state.
    ToolCallUpdate {
        id: String,
        status: ToolStatus,
        raw_output: Option<Value>,
    },
}

/// Tool-call lifecycle states emitted in [`LoopEvent::ToolCallUpdate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    InProgress,
    Completed,
    Failed,
}

impl ToolStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolStatus::InProgress => "in_progress",
            ToolStatus::Completed => "completed",
            ToolStatus::Failed => "failed",
        }
    }
}

// ── UpdateSink ───────────────────────────────────────────────────────

/// Abstraction over "emit a loop event to whoever is watching".
///
/// ACP implements this by serializing [`LoopEvent`]s into
/// `session/update` notifications over its mpsc sender.
/// A2A implements this by pushing SSE events into the per-task channel.
pub trait UpdateSink: Send + Sync {
    fn emit(&self, ev: LoopEvent) -> anyhow::Result<()>;
}

// ── PermissionGate ───────────────────────────────────────────────────

/// Outcome of a permission request for a write-effecting tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    Allowed,
    Rejected,
    /// The turn was cancelled while waiting for a decision.
    CancelledTurn,
}

/// Abstraction over "ask whether this tool call is permitted".
///
/// ACP implements this as `session/request_permission` over oneshot channels.
/// A2A v1 uses `DenyWrites` (no human present to answer prompts).
#[async_trait]
pub trait PermissionGate: Send + Sync {
    async fn request(
        &self,
        tool: &str,
        input: &Value,
        cancel: &mut tokio::sync::watch::Receiver<bool>,
    ) -> PermissionDecision;
}

// ── Stock implementations ────────────────────────────────────────────

/// Always allow everything. Useful for in-process dev tooling.
pub struct AlwaysAllow;

#[async_trait]
impl PermissionGate for AlwaysAllow {
    async fn request(
        &self,
        _tool: &str,
        _input: &Value,
        _cancel: &mut tokio::sync::watch::Receiver<bool>,
    ) -> PermissionDecision {
        PermissionDecision::Allowed
    }
}

/// Deny every write-effecting tool call (no human present). A2A v1 gate.
///
/// The curated A2A tool subset already excludes `write_file`; this gate is
/// the fail-closed defence if the model forges a write_file call anyway.
pub struct DenyWrites;

#[async_trait]
impl PermissionGate for DenyWrites {
    async fn request(
        &self,
        _tool: &str,
        _input: &Value,
        _cancel: &mut tokio::sync::watch::Receiver<bool>,
    ) -> PermissionDecision {
        PermissionDecision::Rejected
    }
}
