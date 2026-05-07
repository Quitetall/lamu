//! Provider abstraction — chat API translation layer.
//!
//! lamu's chat_tui works with a single unified internal format
//! (`Message`, `ToolCallRef`, `StreamEvent`). Each `Provider` impl
//! translates that into one specific wire format (OpenAI compat,
//! Anthropic native, etc.) and parses streaming responses back into
//! the unified events.
//!
//! ## Adding a new provider
//!
//! 1. Define a unit struct: `pub struct MyProvider;`
//! 2. `impl Provider for MyProvider { ... }` — the four wire methods:
//!    - `name(&self) -> &'static str`
//!    - `detect(&self, url) -> bool` — true when this provider should
//!      handle the URL
//!    - `auth(&self, req, api_key)` — attach auth headers
//!    - `build_payload(&self, model, history, search) -> Value`
//!    - `parse_stream(&self, resp, tx)` — translate SSE to StreamEvents
//! 3. Add `&MyProvider` to `PROVIDERS` below — earlier entries take
//!    precedence (`detect` is checked top-down). Keep the catch-all
//!    OpenAI-compat provider last.
//! 4. If your provider needs a non-default URL path, edit
//!    `cloud_models::CloudModel::chat_url` to map your `provider:`
//!    string to the right path.
//!
//! Everything format-specific lives in this file. chat_tui never
//! sees a wire-level field.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::mpsc::Sender;

// ── Unified internal types ───────────────────────────────────────────

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

// ── Provider trait ───────────────────────────────────────────────────

pub trait Provider: Sync {
    fn name(&self) -> &'static str;

    /// Does this provider handle the given URL?
    fn detect(&self, url: &str) -> bool;

    /// Attach auth + format headers to an outgoing request.
    fn auth(
        &self,
        req: reqwest::blocking::RequestBuilder,
        api_key: &str,
    ) -> reqwest::blocking::RequestBuilder;

    /// Build the wire-format JSON payload from the unified history.
    fn build_payload(&self, model: &str, history: &[Message], search: bool) -> Value;

    /// Read the SSE stream and emit unified StreamEvents.
    fn parse_stream(&self, resp: reqwest::blocking::Response, tx: Sender<StreamEvent>);
}

// ── Registry ─────────────────────────────────────────────────────────
// Maintainers: add new providers here. Order matters — detect() is
// checked top-down, first match wins. Keep the catch-all OpenAI
// compat provider last.

pub static PROVIDERS: &[&(dyn Provider + Sync)] = &[
    &AnthropicProvider,
    &OpenAiCompatProvider,
];

/// Pick the first provider whose `detect` returns true for `url`.
/// Falls back to OpenAI compat (the last entry, which always matches).
pub fn detect(url: &str) -> &'static (dyn Provider + Sync) {
    for p in PROVIDERS {
        if p.detect(url) {
            return *p;
        }
    }
    &OpenAiCompatProvider
}

// ── OpenAI-compat provider ───────────────────────────────────────────
// Covers OpenAI itself, DeepSeek, Moonshot, Alibaba DashScope, Zhipu,
// Together, Groq, llama.cpp server, vLLM, sglang — anything that speaks
// the OpenAI /chat/completions wire format.

pub struct OpenAiCompatProvider;

impl Provider for OpenAiCompatProvider {
    fn name(&self) -> &'static str { "openai" }

    fn detect(&self, _url: &str) -> bool {
        // Catch-all: matched last in the registry, accepts anything not
        // claimed by a more specific provider above.
        true
    }

    fn auth(
        &self,
        req: reqwest::blocking::RequestBuilder,
        api_key: &str,
    ) -> reqwest::blocking::RequestBuilder {
        req.bearer_auth(api_key)
    }

    fn build_payload(&self, model: &str, history: &[Message], search: bool) -> Value {
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
            "max_tokens": 16384,
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

    fn parse_stream(&self, resp: reqwest::blocking::Response, tx: Sender<StreamEvent>) {
        let reader = BufReader::new(resp);
        let mut tool_id = String::new();
        let mut tool_name = String::new();
        let mut tool_args = String::new();
        let mut finish_tool = false;

        for line_res in reader.lines() {
            let line = match line_res {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(format!("read: {e}")));
                    let _ = tx.send(StreamEvent::Done);
                    return;
                }
            };
            let line = line.trim();
            if !line.starts_with("data:") { continue; }
            let data = line[5..].trim();
            if data == "[DONE]" { break; }
            let v: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let finish_reason = v["choices"][0]["finish_reason"].as_str().unwrap_or("");
            if finish_reason == "tool_calls" {
                finish_tool = true;
            }

            let delta = v.get("choices").and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"));

            if let Some(tc) = delta.and_then(|d| d["tool_calls"].get(0)) {
                if let Some(id) = tc["id"].as_str() { if !id.is_empty() { tool_id = id.to_string(); } }
                if let Some(n) = tc["function"]["name"].as_str() { if !n.is_empty() { tool_name = n.to_string(); } }
                if let Some(a) = tc["function"]["arguments"].as_str() { tool_args.push_str(a); }
            }

            let reason = delta.and_then(|d| d.get("reasoning_content"))
                .and_then(|s| s.as_str()).unwrap_or("").to_string();
            if !reason.is_empty() {
                if tx.send(StreamEvent::Reason(reason)).is_err() { return; }
            }
            let token = delta.and_then(|d| d.get("content"))
                .and_then(|s| s.as_str()).unwrap_or("").to_string();
            if !token.is_empty() {
                if tx.send(StreamEvent::Token(token)).is_err() { return; }
            }
        }

        if finish_tool && !tool_name.is_empty() {
            let _ = tx.send(StreamEvent::ToolCall {
                id: tool_id,
                name: tool_name,
                arguments: tool_args,
            });
        } else {
            let _ = tx.send(StreamEvent::Done);
        }
    }
}

// ── Anthropic provider ───────────────────────────────────────────────
// Covers api.anthropic.com /v1/messages and any /anthropic/-prefixed
// proxy (e.g. api.deepseek.com/anthropic).

pub struct AnthropicProvider;

impl Provider for AnthropicProvider {
    fn name(&self) -> &'static str { "anthropic" }

    fn detect(&self, url: &str) -> bool {
        let u = url.to_lowercase();
        u.contains("anthropic.com")
            || u.contains("/anthropic/")
            || u.ends_with("/anthropic")
            || u.ends_with("/v1/messages")
    }

    fn auth(
        &self,
        req: reqwest::blocking::RequestBuilder,
        api_key: &str,
    ) -> reqwest::blocking::RequestBuilder {
        req.header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
    }

    fn build_payload(&self, model: &str, history: &[Message], search: bool) -> Value {
        let mut system_text = String::new();
        let mut messages: Vec<Value> = Vec::new();

        for m in history {
            // System messages are top-level on Anthropic, not a role.
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
                Role::System => continue, // already handled above
            };
            messages.push(json!({"role": role, "content": m.content}));
        }

        let mut payload = json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "max_tokens": 16384,
            "temperature": 0.7,
        });
        if !system_text.is_empty() {
            payload["system"] = json!(system_text);
        }
        if search {
            // Anthropic tools schema: input_schema, not parameters.
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

    fn parse_stream(&self, resp: reqwest::blocking::Response, tx: Sender<StreamEvent>) {
        let reader = BufReader::new(resp);
        let mut current_block_type = String::new();
        let mut tool_id = String::new();
        let mut tool_name = String::new();
        let mut tool_args = String::new();
        let mut got_tool = false;

        for line_res in reader.lines() {
            let line = match line_res {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(format!("read: {e}")));
                    let _ = tx.send(StreamEvent::Done);
                    return;
                }
            };
            let line = line.trim();
            if !line.starts_with("data:") { continue; }
            let data = line[5..].trim();
            if data.is_empty() { continue; }
            let v: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let typ = v["type"].as_str().unwrap_or("");
            match typ {
                "content_block_start" => {
                    let cb = &v["content_block"];
                    current_block_type = cb["type"].as_str().unwrap_or("").to_string();
                    if current_block_type == "tool_use" {
                        tool_id = cb["id"].as_str().unwrap_or("").to_string();
                        tool_name = cb["name"].as_str().unwrap_or("").to_string();
                        tool_args.clear();
                    }
                }
                "content_block_delta" => {
                    let delta = &v["delta"];
                    let dt = delta["type"].as_str().unwrap_or("");
                    match dt {
                        "text_delta" => {
                            let t = delta["text"].as_str().unwrap_or("").to_string();
                            if !t.is_empty() {
                                if tx.send(StreamEvent::Token(t)).is_err() { return; }
                            }
                        }
                        "thinking_delta" => {
                            let t = delta["thinking"].as_str().unwrap_or("").to_string();
                            if !t.is_empty() {
                                if tx.send(StreamEvent::Reason(t)).is_err() { return; }
                            }
                        }
                        "input_json_delta" => {
                            if let Some(p) = delta["partial_json"].as_str() {
                                tool_args.push_str(p);
                            }
                        }
                        _ => {}
                    }
                }
                "content_block_stop" => {}
                "message_delta" => {
                    if v["delta"]["stop_reason"].as_str() == Some("tool_use") {
                        got_tool = true;
                    }
                }
                "message_stop" => break,
                "error" => {
                    let msg = v["error"]["message"].as_str()
                        .unwrap_or("anthropic stream error");
                    let _ = tx.send(StreamEvent::Error(msg.to_string()));
                    let _ = tx.send(StreamEvent::Done);
                    return;
                }
                _ => {}
            }
        }

        if got_tool && !tool_name.is_empty() {
            let _ = tx.send(StreamEvent::ToolCall {
                id: tool_id,
                name: tool_name,
                arguments: tool_args,
            });
        } else {
            let _ = tx.send(StreamEvent::Done);
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_anthropic_urls() {
        assert_eq!(detect("https://api.anthropic.com/v1/messages").name(), "anthropic");
        assert_eq!(detect("https://api.deepseek.com/anthropic/v1/messages").name(), "anthropic");
        assert_eq!(detect("https://gateway.example.com/anthropic").name(), "anthropic");
    }

    #[test]
    fn detect_openai_falls_through() {
        assert_eq!(detect("https://api.deepseek.com/chat/completions").name(), "openai");
        assert_eq!(detect("https://api.openai.com/v1/chat/completions").name(), "openai");
        assert_eq!(detect("http://localhost:8020/v1/chat/completions").name(), "openai");
    }

    #[test]
    fn openai_payload_plain() {
        let history = vec![
            Message::plain(Role::User, "hi".into()),
            Message::plain(Role::Assistant, "hello".into()),
        ];
        let payload = OpenAiCompatProvider.build_payload("gpt-4", &history, false);
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
        let payload = OpenAiCompatProvider.build_payload("gpt-4", &history, true);
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
        let payload = AnthropicProvider.build_payload("claude-opus", &history, false);
        assert_eq!(payload["system"], "You are helpful.");
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn anthropic_payload_uses_input_schema() {
        let history = vec![Message::plain(Role::User, "hi".into())];
        let payload = AnthropicProvider.build_payload("claude-opus", &history, true);
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
        let payload = AnthropicProvider.build_payload("claude-opus", &history, true);
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
        let payload = AnthropicProvider.build_payload("claude-opus", &history, false);
        let blocks = payload["messages"][1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "tool_use");
    }
}
