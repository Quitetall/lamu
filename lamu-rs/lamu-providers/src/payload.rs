//! Pure wire-format payload builders.
//!
//! These functions are IO-free — they take a `&[Message]` history and
//! return a `serde_json::Value` ready to be POSTed by either the sync
//! transport (lamu-cli) or the async transport (lamu-mcp).

use crate::types::{Message, Role};
use serde_json::{json, Value};

/// Build an OpenAI `/chat/completions` request body. Covers OpenAI
/// itself, DeepSeek, Moonshot, Alibaba DashScope, Zhipu, Together,
/// Groq, llama.cpp server, vLLM, sglang — anything that speaks the
/// OpenAI wire format.
pub fn build_openai_payload(model: &str, history: &[Message], search: bool) -> Value {
    let messages: Vec<Value> = history.iter().map(|m| {
        if let Some(tc) = &m.tool_call {
            json!({
                "role": "assistant",
                "content": m.content,
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                }]
            })
        } else if let Some(tid) = &m.tool_result_for {
            json!({
                "role": "tool",
                "tool_call_id": tid,
                "content": m.content,
            })
        } else {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };
            json!({"role": role, "content": m.content})
        }
    }).collect();

    let mut payload = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "max_tokens": 65536,
        "temperature": 0.7,
    });
    if search {
        payload["tools"] = json!([{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information. Use when the user asks about recent events, facts you may not know, or anything that benefits from up-to-date sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        }]);
        payload["tool_choice"] = json!("auto");
    }
    payload
}

/// Build an Anthropic `/v1/messages` request body. System messages are
/// promoted to the top-level `system` field; tool calls become
/// `tool_use` blocks; tool results become `tool_result` blocks.
pub fn build_anthropic_payload(model: &str, history: &[Message], search: bool) -> Value {
    let mut system_text = String::new();
    let mut messages: Vec<Value> = Vec::new();

    for m in history {
        if matches!(m.role, Role::System) && m.tool_result_for.is_none() {
            if !system_text.is_empty() { system_text.push_str("\n\n"); }
            system_text.push_str(&m.content);
            continue;
        }

        if let Some(tc) = &m.tool_call {
            let input: Value = serde_json::from_str(&tc.arguments)
                .unwrap_or_else(|_| json!({}));
            let mut blocks: Vec<Value> = Vec::new();
            if !m.content.trim().is_empty() {
                blocks.push(json!({"type": "text", "text": m.content}));
            }
            blocks.push(json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": input,
            }));
            messages.push(json!({
                "role": "assistant",
                "content": blocks,
            }));
            continue;
        }

        if let Some(tid) = &m.tool_result_for {
            messages.push(json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": m.content,
                }],
            }));
            continue;
        }

        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => continue,
        };
        messages.push(json!({"role": role, "content": m.content}));
    }

    let mut payload = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "max_tokens": 65536,
        "temperature": 0.7,
    });
    if !system_text.is_empty() {
        payload["system"] = json!(system_text);
    }
    if search {
        payload["tools"] = json!([{
            "name": "web_search",
            "description": "Search the web for current information. Use when the user asks about recent events, facts you may not know, or anything that benefits from up-to-date sources.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }]);
    }
    payload
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, Role, ToolCallRef};

    #[test]
    fn openai_payload_plain() {
        let history = vec![
            Message::plain(Role::User, "hi".into()),
            Message::plain(Role::Assistant, "hello".into()),
        ];
        let payload = build_openai_payload("gpt-4", &history, false);
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn openai_payload_tool_roundtrip() {
        let history = vec![
            Message::plain(Role::User, "weather?".into()),
            Message {
                role: Role::Assistant,
                content: "looking".into(),
                tool_call: Some(ToolCallRef {
                    id: "call_1".into(),
                    name: "web_search".into(),
                    arguments: r#"{"query":"weather"}"#.into(),
                }),
                tool_result_for: None,
            },
            Message {
                role: Role::User,
                content: "sunny".into(),
                tool_call: None,
                tool_result_for: Some("call_1".into()),
            },
        ];
        let payload = build_openai_payload("gpt-4", &history, true);
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_1");
        assert_eq!(payload["tools"][0]["function"]["name"], "web_search");
    }

    #[test]
    fn anthropic_payload_promotes_system() {
        let history = vec![
            Message::plain(Role::System, "You are helpful.".into()),
            Message::plain(Role::User, "hi".into()),
        ];
        let payload = build_anthropic_payload("claude-opus", &history, false);
        assert_eq!(payload["system"], "You are helpful.");
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn anthropic_payload_uses_input_schema() {
        let history = vec![Message::plain(Role::User, "hi".into())];
        let payload = build_anthropic_payload("claude-opus", &history, true);
        assert_eq!(payload["tools"][0]["name"], "web_search");
        assert!(payload["tools"][0]["input_schema"].is_object());
        assert!(payload["tools"][0]["parameters"].is_null());
    }

    #[test]
    fn anthropic_payload_tool_blocks() {
        let history = vec![
            Message::plain(Role::User, "weather?".into()),
            Message {
                role: Role::Assistant,
                content: String::new(),
                tool_call: Some(ToolCallRef {
                    id: "toolu_1".into(),
                    name: "web_search".into(),
                    arguments: r#"{"query":"weather"}"#.into(),
                }),
                tool_result_for: None,
            },
            Message {
                role: Role::User,
                content: "sunny".into(),
                tool_call: None,
                tool_result_for: Some("toolu_1".into()),
            },
        ];
        let payload = build_anthropic_payload("claude-opus", &history, true);
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["content"][0]["type"], "tool_use");
        assert_eq!(msgs[1]["content"][0]["id"], "toolu_1");
        assert_eq!(msgs[2]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[2]["content"][0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn anthropic_payload_text_then_tool_block() {
        let history = vec![
            Message::plain(Role::User, "?".into()),
            Message {
                role: Role::Assistant,
                content: "Let me check.".into(),
                tool_call: Some(ToolCallRef {
                    id: "toolu_1".into(),
                    name: "web_search".into(),
                    arguments: r#"{"query":"x"}"#.into(),
                }),
                tool_result_for: None,
            },
        ];
        let payload = build_anthropic_payload("claude-opus", &history, false);
        let blocks = payload["messages"][1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "tool_use");
    }

    #[test]
    fn anthropic_payload_skips_empty_text_block() {
        let history = vec![
            Message {
                role: Role::Assistant,
                content: "   ".into(),
                tool_call: Some(ToolCallRef {
                    id: "id1".into(),
                    name: "t".into(),
                    arguments: "{}".into(),
                }),
                tool_result_for: None,
            },
        ];
        let payload = build_anthropic_payload("claude-opus", &history, false);
        let blocks = payload["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_use");
    }
}
