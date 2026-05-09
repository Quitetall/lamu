//! Unified internal chat-history types shared by all providers.
//!
//! Pure data — no IO, no async, no transport. The `Provider` trait
//! (sync, in lamu-cli) and `handle_cloud_query` (async, in lamu-mcp)
//! both translate these into wire-format JSON.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone)]
pub struct ToolCallRef {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments string, as the model emits it.
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// When set: this assistant message includes a tool_use block.
    /// Both visible content and tool call live on the same Message
    /// so we can reconstruct provider-specific representations.
    pub tool_call: Option<ToolCallRef>,
    /// When set: this message is a tool result. The id matches a
    /// prior assistant Message.tool_call.id.
    pub tool_result_for: Option<String>,
}

impl Message {
    pub fn plain(role: Role, content: String) -> Self {
        Self { role, content, tool_call: None, tool_result_for: None }
    }
}

#[derive(Debug)]
pub enum StreamEvent {
    /// Final-answer text token.
    Token(String),
    /// Thinking/reasoning text token (DeepSeek-V4, Qwen3-Thinking,
    /// Anthropic extended thinking).
    Reason(String),
    /// Model wants to call a tool.
    ToolCall { id: String, name: String, arguments: String },
    Done,
    Error(String),
}
