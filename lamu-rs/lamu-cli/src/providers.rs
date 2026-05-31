//! Provider abstraction — chat API translation layer.
//!
//! This crate's chat_tui works with a single unified internal format
//! (`Message`, `ToolCallRef`, `StreamEvent` — all defined in
//! `lamu-providers`). Each `Provider` impl translates that into one
//! specific wire format (OpenAI compat, Anthropic native, etc.) and
//! parses streaming responses back into the unified events.
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
//! Pure payload construction lives in `lamu_providers::payload`. This
//! module is the *sync transport* layer — auth headers + SSE parsing
//! against `reqwest::blocking`.

use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::sync::mpsc::Sender;

// Re-export the unified types so existing `use crate::providers::{Message, …}`
// imports keep working. The canonical definitions live in lamu-providers.
pub use lamu_providers::{
    anthropic_beta_header, build_anthropic_payload, build_openai_payload, Message, Role,
    StreamEvent, ToolCallRef,
};

// ── Provider trait ───────────────────────────────────────────────────

pub trait Provider: Sync {
    /// Provider id ("openai"/"anthropic") — reserved for logging/introspection.
    #[allow(dead_code)]
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
        build_openai_payload(model, history, search)
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
        let mut req = req
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");
        // Opt-in 1M-context beta. Set ANTHROPIC_BETA in env to engage:
        //   ANTHROPIC_BETA=context-1m-2025-08-07
        if let Some(val) = anthropic_beta_header() {
            req = req.header("anthropic-beta", val);
        }
        req
    }

    fn build_payload(&self, model: &str, history: &[Message], search: bool) -> Value {
        build_anthropic_payload(model, history, search)
    }

    fn parse_stream(&self, resp: reqwest::blocking::Response, tx: Sender<StreamEvent>) {
        let reader = BufReader::new(resp);
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
                    let current_block_type = cb["type"].as_str().unwrap_or("");
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
// Payload-shape tests live in `lamu-providers`. Tests here cover the
// sync transport: detect() routing.

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
}
