//! `web_search` MCP tool — general keyless web search via a self-hosted SearXNG
//! instance (metasearch over many engines). The lookup backend behind the
//! "don't answer from parametric memory — look it up" default: a small local
//! model (or the outer agent) can ground any factual claim against fresh hits.
//!
//! Keyless: hits `$SEARXNG_URL` (default `http://127.0.0.1:8888`) `/search`
//! with `format=json`. No model call — pure retrieval — so it's usable in
//! local-only routing. Returns sanitized title/url/snippet/engine per result.
//!
//! The fetch + injection-sanitization live in `lamu_core::web_search` so this
//! tool and the agentic `answer` loop share one hardened backend (audit B7).

use lamu_core::tools_ext::ToolCtx;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// JSON schema for the `web_search` MCP tool.
pub fn schema_web_search() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "default": 8, "description": "Max results to return (1-25)."},
            "categories": {"type": "string", "description": "Optional SearXNG categories, e.g. 'general', 'science', 'news'. Default: general."}
        },
        "required": ["query"]
    })
}

/// `web_search` ignores the ctx (pure retrieval, no model call) but keeps the
/// `ModuleToolHandler` shape so it registers like every other module tool.
pub fn dispatch_web_search<'a>(
    _ctx: &'a dyn ToolCtx,
    args: Value,
) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(handle_web_search(args))
}

pub async fn handle_web_search(args: Value) -> String {
    let query = args["query"].as_str().unwrap_or("").trim().to_string();
    if query.is_empty() {
        return "error: web_search requires a non-empty `query`".into();
    }
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 25) as usize;
    let categories = args["categories"].as_str().unwrap_or("general");

    match lamu_core::web_search::searxng_search(&query, limit, categories, Duration::from_secs(20)).await {
        Ok(hits) => {
            let results: Vec<Value> = hits.iter().map(|h| h.to_json()).collect();
            json!({"query": query, "count": results.len(), "results": results}).to_string()
        }
        Err(e) => format!("error: {e}"),
    }
}
