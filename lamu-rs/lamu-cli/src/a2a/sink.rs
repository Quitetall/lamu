//! A2A `UpdateSink` + `PermissionGate` implementations.
//!
//! `A2aSink` wraps an mpsc sender that pushes SSE data lines (already
//! formatted as `data: <json>\n\n`) into the per-task streaming channel.
//! The axum SSE handler drains that channel and forwards lines to the client.
//!
//! `A2aPermissionGate` = deny writes: no human present in an A2A context,
//! so write-effecting tools are unconditionally rejected. The curated A2A
//! tool subset already excludes `write_file`; this gate is the fail-closed
//! second layer if the model forges such a call.

use crate::agent_core::{LoopEvent, PermissionDecision, PermissionGate, UpdateSink};
use crate::a2a::protocol;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;

/// Forwards `LoopEvent`s from the agent loop to the per-task SSE channel
/// as `data: <json>\n\n` strings.
pub struct A2aSink {
    pub task_id: String,
    pub tx: mpsc::UnboundedSender<String>,
}

impl UpdateSink for A2aSink {
    fn emit(&self, ev: LoopEvent) -> anyhow::Result<()> {
        let sse = match ev {
            LoopEvent::MessageChunk(text) => {
                // Stream as a TaskStatusUpdateEvent with an agent message
                // carrying the chunk. Clients accumulate these.
                let event = protocol::status_update_event(
                    &self.task_id,
                    protocol::STATE_WORKING,
                    Some(json!({
                        "messageId": protocol::pseudo_id("msg-"),
                        "role": "agent",
                        "parts": [{ "kind": "text", "text": text }],
                    })),
                );
                protocol::sse_line(&event)
            }
            LoopEvent::ThoughtChunk(text) => {
                // Stream as a TaskArtifactUpdateEvent carrying a DataPart
                // `{kind:"thought"}`. Unknown artifact kinds are ignored by
                // compliant A2A clients (additive extension, spec §7.2).
                let artifact = json!({
                    "artifactId": protocol::pseudo_id("thought-"),
                    "index": 0,
                    "parts": [protocol::thought_part(&text)],
                });
                let event = protocol::artifact_update_event(&self.task_id, artifact);
                protocol::sse_line(&event)
            }
            LoopEvent::ToolCall { id, title, kind, raw_input } => {
                // Not spec-mandated; stream as a DataPart artifact for
                // transparency. Clients that don't understand it ignore it.
                let artifact = json!({
                    "artifactId": format!("tool-call-{id}"),
                    "index": 0,
                    "parts": [{
                        "kind": "data",
                        "data": {
                            "kind": "tool_call",
                            "toolCallId": id,
                            "title": title,
                            "toolKind": kind,
                            "rawInput": raw_input,
                            "status": "pending",
                        },
                    }],
                });
                let event = protocol::artifact_update_event(&self.task_id, artifact);
                protocol::sse_line(&event)
            }
            LoopEvent::ToolCallUpdate { id, status, raw_output } => {
                let mut data = json!({
                    "kind": "tool_call_update",
                    "toolCallId": id,
                    "status": status.as_str(),
                });
                if let Some(out) = raw_output {
                    data["rawOutput"] = out;
                }
                let artifact = json!({
                    "artifactId": format!("tool-call-{id}"),
                    "index": 0,
                    "parts": [{ "kind": "data", "data": data }],
                });
                let event = protocol::artifact_update_event(&self.task_id, artifact);
                protocol::sse_line(&event)
            }
        };
        let _ = self.tx.send(sse);
        Ok(())
    }
}

/// A2A v1 permission gate: deny all write-effecting tools unconditionally.
pub struct A2aPermissionGate;

#[async_trait]
impl PermissionGate for A2aPermissionGate {
    async fn request(
        &self,
        _tool: &str,
        _input: &Value,
        _cancel: &mut tokio::sync::watch::Receiver<bool>,
    ) -> PermissionDecision {
        PermissionDecision::Rejected
    }
}
