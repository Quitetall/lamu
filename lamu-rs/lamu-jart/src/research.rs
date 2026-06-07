//! `research` MCP tool — fans out jart's scrapers for a query, then (optionally)
//! summarizes the results in-process via lamu-core.
//!
//! ADR 0023: lives in the lamu-jart MODULE and runs against a `&dyn ToolCtx`. It
//! uses jart's `core` library for the scrape/fan-out (HuggingFace, PubMed,
//! bioRxiv, Semantic Scholar over stdio Python adapters) and `ctx.generate` for
//! the summary — no self-HTTP round-trip to `:8020`.

use jart::core::{ai, cache::Cache, config::Topic, feed, ratelimit::Pacer};
use lamu_core::tools_ext::ToolCtx;
use serde_json::{json, Value};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

const SUMMARY_PROMPT: &str =
    "You are a research assistant. Summarize the key themes across these papers \
     in 4-6 sentences, then list the 3 most notable items by title.";

/// Where jart's Python scrapers live. `JART_SCRAPERS_DIR` overrides; otherwise
/// the standalone jart checkout's `scrapers/` (this module depends on that repo).
pub(crate) fn scrapers_dir() -> PathBuf {
    if let Ok(p) = std::env::var("JART_SCRAPERS_DIR") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    dirs::home_dir()
        .unwrap_or_default()
        .join("Desktop/jart/scrapers")
}

/// Default model for research summarization/synthesis. `LAMU_RESEARCH_MODEL`
/// overrides — set it to a local registry model (e.g. the installed
/// `tongyi-deepresearch-30b-a3b`) to keep research fully local; otherwise the
/// cloud workhorse. The chosen model is loaded on demand (ensure_loaded), so a
/// cold local model just costs a one-time load.
pub(crate) fn default_research_model() -> String {
    std::env::var("LAMU_RESEARCH_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "mimo-v2.5".to_string())
}

/// JSON schema for the `research` MCP tool.
pub fn schema_research() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Topic/search query (used as both the HuggingFace and PubMed search term)."},
            "limit": {"type": "integer", "default": 8, "description": "Max items per source (1-25)."},
            "summarize": {"type": "boolean", "default": true, "description": "Summarize the results in-process via the model below."},
            "model": {"type": "string", "description": "Model for the summary — a local registry model or a cloud model (honors routing mode). Defaults to $LAMU_RESEARCH_MODEL, else mimo-v2.5."}
        },
        "required": ["query"]
    })
}

/// The `ModuleToolHandler` wrapper registered into lamu-core's tool registry.
pub fn dispatch_research<'a>(
    ctx: &'a dyn ToolCtx,
    args: Value,
) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(handle_research(ctx, args))
}

/// Tool entrypoint.
pub async fn handle_research(ctx: &dyn ToolCtx, args: Value) -> String {
    let query = args["query"].as_str().unwrap_or("").trim().to_string();
    if query.is_empty() {
        return "error: research requires a non-empty `query`".into();
    }
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 25) as usize;
    let summarize = args["summarize"].as_bool().unwrap_or(true);
    let model = args["model"].as_str().map(String::from).unwrap_or_else(default_research_model);

    let sdir = scrapers_dir();
    if !sdir.exists() {
        return format!(
            "error: jart scrapers dir not found at {} — set JART_SCRAPERS_DIR to the jart checkout's scrapers/",
            sdir.display()
        );
    }

    // One topic from the query: jart maps `hf` + `pubmed` as the per-source
    // search terms. id/label are cosmetic.
    let topic = Topic {
        id: "research".into(),
        label: query.clone(),
        hf: query.clone(),
        pubmed: query.clone(),
    };
    let cache = Cache::new();
    let pacer = Pacer::new();
    let f = feed::load(&sdir, std::slice::from_ref(&topic), limit, &cache, &pacer).await;

    // Build the JSON result from the feed, adding a summary when asked.
    let mut out = match serde_json::to_value(&f) {
        Ok(v) => v,
        Err(e) => return format!("error: serialize feed: {e}"),
    };
    out["query"] = Value::String(query);

    if summarize && !f.papers.is_empty() {
        // Ground the summary on title + abstract of each paper.
        let items: Vec<String> = f
            .papers
            .iter()
            .map(|p| format!("{}\n{}", p.title, p.grounding))
            .collect();
        let content = ai::build_grounded_content(SUMMARY_PROMPT, &items);
        // A LOCAL registry model must be loaded before generate — handle_query
        // does NOT auto-load a cold model (it returns "error: model ... not
        // loaded"). model_modality is Some only for local models; cloud models
        // are routed by generate and need no load.
        let load_err = if ctx.model_modality(&model).is_some() {
            let status = ctx.ensure_loaded(&model).await;
            // handle_load_model returns "error: ..." (with colon) on failure.
            if status.trim_start().to_lowercase().starts_with("error:") {
                Some(status)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(status) = load_err {
            out["summary_error"] = Value::String(status);
        } else {
            let summary = ctx.generate(&model, &content).await;
            // ctx.generate returns an "error:"-prefixed string on failure (the MCP
            // convention — matches the server's own is_error check, which keys on
            // "error:" WITH the colon so prose like "Error bars on the chart…" isn't
            // misread). Surface a failure in a dedicated field rather than failing
            // the whole tool — the feed is still useful without the summary.
            if summary.trim_start().to_lowercase().starts_with("error:") {
                out["summary_error"] = Value::String(summary);
            } else {
                out["summary"] = Value::String(summary);
            }
        }
    }

    out.to_string()
}
