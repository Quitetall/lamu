//! `web_search` MCP tool — general keyless web search via a self-hosted SearXNG
//! instance (metasearch over many engines). The lookup backend behind the
//! "don't answer from parametric memory — look it up" default: a small local
//! model (or the outer agent) can ground any factual claim against fresh hits.
//!
//! Keyless: hits `$SEARXNG_URL` (default `http://127.0.0.1:8888`) `/search`
//! with `format=json`. No model call — pure retrieval — so it's usable in
//! local-only routing. Returns title/url/snippet/engine per result.

use lamu_core::tools_ext::ToolCtx;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// Base URL of the SearXNG instance. `SEARXNG_URL` overrides the local default;
/// it must be an `http(s)://` URL (trimmed) — a malformed value (spaces,
/// newlines, a non-URL) is ignored rather than corrupting the request.
fn searxng_url() -> String {
    std::env::var("SEARXNG_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| s.starts_with("http://") || s.starts_with("https://"))
        .unwrap_or_else(|| "http://127.0.0.1:8888".to_string())
}

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

    match searxng_search(&query, limit, categories).await {
        Ok(results) => json!({"query": query, "count": results.len(), "results": results}).to_string(),
        Err(e) => format!("error: {e}"),
    }
}

/// Run one SearXNG query and return up to `limit` parsed results
/// ({title,url,snippet,engine}). Shared by the `web_search` tool and the
/// agentic `answer` loop. Err string on transport / non-200 / parse failure.
pub(crate) async fn searxng_search(
    query: &str,
    limit: usize,
    categories: &str,
) -> Result<Vec<Value>, String> {
    let base = searxng_url();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let resp = client
        .get(format!("{}/search", base.trim_end_matches('/')))
        .query(&[("q", query), ("format", "json"), ("categories", categories)])
        .send()
        .await
        .map_err(|e| {
            format!("SearXNG unreachable at {base} ({e}) — is the container up? set SEARXNG_URL to override.")
        })?;
    if !resp.status().is_success() {
        return Err(format!(
            "SearXNG HTTP {} (JSON format may be disabled — settings.yml search.formats must include 'json')",
            resp.status().as_u16()
        ));
    }
    let body: Value = resp.json().await.map_err(|e| format!("parse SearXNG response: {e}"))?;
    Ok(parse_results(&body, limit))
}

/// Extract the top-`limit` results into a compact shape. Pure over the parsed
/// body so it's unit-testable without a live SearXNG.
fn parse_results(body: &Value, limit: usize) -> Vec<Value> {
    body["results"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter(|r| r.get("url").and_then(|u| u.as_str()).is_some())
                .take(limit)
                .map(|r| {
                    json!({
                        "title": r["title"].as_str().unwrap_or(""),
                        "url": r["url"].as_str().unwrap_or(""),
                        "snippet": r["content"].as_str().unwrap_or(""),
                        "engine": r["engine"].as_str().unwrap_or(""),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_results_shapes_and_caps() {
        let body = json!({"results": [
            {"title": "A", "url": "https://a", "content": "snip a", "engine": "duckduckgo"},
            {"title": "B", "url": "https://b", "content": "snip b", "engine": "google"},
            {"title": "no url dropped"},
            {"title": "C", "url": "https://c", "content": "snip c", "engine": "bing"}
        ]});
        let out = parse_results(&body, 2);
        assert_eq!(out.len(), 2); // capped at limit
        assert_eq!(out[0]["url"], "https://a");
        assert_eq!(out[1]["title"], "B");
    }

    #[test]
    fn parse_results_empty_on_no_results() {
        assert!(parse_results(&json!({}), 8).is_empty());
        assert!(parse_results(&json!({"results": []}), 8).is_empty());
    }

    #[test]
    fn searxng_url_defaults_local() {
        // env not set in the test process -> local default.
        assert_eq!(searxng_url(), "http://127.0.0.1:8888");
    }
}
