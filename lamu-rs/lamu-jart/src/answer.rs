//! `answer` MCP tool — agentic web-grounded Q&A (Phase 2 of the grounding work).
//!
//! The loop, all through `ctx.generate` (no function-calling infra needed):
//!   1. DECIDE  — ask the model whether the question needs a web lookup and, if
//!      so, for 1-3 search queries (JSON array; `[]` = pure reasoning, no facts).
//!   2. SEARCH  — run each query through SearXNG (`web_search::searxng_search`),
//!      merge + dedup by URL into a numbered source list.
//!   3. GROUND  — answer using ONLY those sources, citing `[N]`; resolve each
//!      `[N]` → `sources[N-1].url` IN CODE (hallucinated links impossible).
//! When DECIDE returns no queries, the model answers directly from reasoning
//! (still told to flag uncertainty) — so opinion/math/code questions don't pay
//! for a needless search.

use crate::research::default_research_model;
use crate::web_search::searxng_search;
use jart::core::ai::build_grounded_content;
use lamu_core::tools_ext::ToolCtx;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;

const DECIDE_PROMPT: &str =
    "Decide whether answering the user's question needs a WEB LOOKUP — i.e. it \
     depends on specific facts, names, dates, numbers, prices, current events, or \
     particular documents/APIs you may not reliably know. If it does, reply with \
     ONLY a JSON array of 1-3 short keyword search queries (2-6 words each). If \
     the question is pure reasoning, math, opinion, or code that needs no external \
     facts, reply with ONLY an empty array []. No prose.";

const ANSWER_GROUNDED_PROMPT: &str =
    "Answer the user's question using ONLY the numbered web sources below. Cite \
     every factual claim inline as [N] (the source's id). Never cite a number not \
     provided, never invent a source. If the sources don't actually answer the \
     question, say so plainly. Lead with the direct answer, then detail.";

const ANSWER_DIRECT_PROMPT: &str =
    "Answer the user's question. This is reasoning/analysis that needs no external \
     facts — but if any specific fact you're unsure of slips in, flag it as \
     uncertain rather than stating it confidently.";

/// JSON schema for the `answer` MCP tool.
pub fn schema_answer() -> Value {
    json!({
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to answer."},
            "model": {"type": "string", "description": "Model to use. Defaults to $LAMU_RESEARCH_MODEL, else mimo-v2.5."},
            "max_queries": {"type": "integer", "default": 3, "description": "Max web searches to run (1-5)."},
            "per_query": {"type": "integer", "default": 5, "description": "Results per search (1-10)."}
        },
        "required": ["question"]
    })
}

pub fn dispatch_answer<'a>(
    ctx: &'a dyn ToolCtx,
    args: Value,
) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(handle_answer(ctx, args))
}

pub async fn handle_answer(ctx: &dyn ToolCtx, args: Value) -> String {
    let question = args["question"].as_str().unwrap_or("").trim().to_string();
    if question.is_empty() {
        return "error: answer requires a non-empty `question`".into();
    }
    let model = args["model"].as_str().map(String::from).unwrap_or_else(default_research_model);
    let max_q = args["max_queries"].as_u64().unwrap_or(3).clamp(1, 5) as usize;
    let per_q = args["per_query"].as_u64().unwrap_or(5).clamp(1, 10) as usize;

    // A local model must be loaded before generate (cloud routes via generate).
    if ctx.model_modality(&model).is_some() {
        let status = ctx.ensure_loaded(&model).await;
        if status.trim_start().to_lowercase().starts_with("error:") {
            return format!("error: load model '{model}': {status}");
        }
    }

    // 1. DECIDE.
    let decide = ctx
        .generate(&model, &format!("{DECIDE_PROMPT}\n\nQuestion: {question}"))
        .await;
    if decide.trim_start().to_lowercase().starts_with("error:") {
        return format!("error: decide step: {decide}");
    }
    let queries: Vec<String> = parse_queries(&decide).into_iter().take(max_q).collect();

    // No lookup needed → direct reasoning answer.
    if queries.is_empty() {
        let ans = ctx
            .generate(&model, &format!("{ANSWER_DIRECT_PROMPT}\n\nQuestion: {question}"))
            .await;
        if ans.trim_start().to_lowercase().starts_with("error:") {
            return format!("error: answer step: {ans}");
        }
        return json!({
            "question": question, "searched": [], "sources": [],
            "answer": ans, "citations": [], "grounded": false
        })
        .to_string();
    }

    // 2. SEARCH — merge + dedup by URL.
    let mut sources: Vec<Value> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut search_failed: Vec<Value> = Vec::new();
    for q in &queries {
        match searxng_search(q, per_q, "general").await {
            Ok(results) => {
                for r in results {
                    let url = r["url"].as_str().unwrap_or("").to_string();
                    if url.is_empty() || !seen.insert(url) {
                        continue;
                    }
                    sources.push(r);
                }
            }
            Err(e) => search_failed.push(json!({"query": q, "error": e})),
        }
    }

    if sources.is_empty() {
        return json!({
            "question": question, "searched": queries, "sources": [],
            "answer": "", "citations": [], "grounded": false,
            "search_failed": search_failed,
            "note": "no web results — SearXNG may be down or the queries returned nothing"
        })
        .to_string();
    }

    // 3. GROUND + answer. Fence each source as <source id="N">; cite [N].
    let items: Vec<String> = sources
        .iter()
        .map(|s| {
            format!(
                "{}\n{}",
                s["title"].as_str().unwrap_or(""),
                s["snippet"].as_str().unwrap_or("")
            )
        })
        .collect();
    let instruction = format!("{ANSWER_GROUNDED_PROMPT}\n\nQuestion: {question}");
    let content = build_grounded_content(&instruction, &items);
    let answer = ctx.generate(&model, &content).await;
    if answer.trim_start().to_lowercase().starts_with("error:") {
        return format!("error: answer step: {answer}");
    }

    // Resolve [N] → sources[N-1].url IN CODE.
    let citations = cited_urls(&answer, &sources);

    json!({
        "question": question,
        "searched": queries,
        "sources": sources_json(&sources),
        "answer": answer,
        "citations": citations,
        "grounded": true,
        "search_failed": search_failed,
    })
    .to_string()
}

/// Tolerant parse of the DECIDE reply into search queries. Extracts the first
/// `[`..last `]` JSON array of strings; empty / unparseable / "error:" → no
/// queries (so an undecidable reply errs toward a direct answer, not a bad
/// search). Drops blanks.
fn parse_queries(text: &str) -> Vec<String> {
    if text.trim_start().to_lowercase().starts_with("error:") {
        return Vec::new();
    }
    let (a, b) = match (text.find('['), text.rfind(']')) {
        (Some(a), Some(b)) if b > a => (a, b),
        _ => return Vec::new(),
    };
    serde_json::from_str::<Vec<String>>(&text[a..=b])
        .map(|v| {
            v.into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Compact source list with 1-based ids for the response.
fn sources_json(sources: &[Value]) -> Vec<Value> {
    sources
        .iter()
        .enumerate()
        .map(|(i, s)| {
            json!({
                "idx": i + 1,
                "title": s["title"].as_str().unwrap_or(""),
                "url": s["url"].as_str().unwrap_or(""),
                "engine": s["engine"].as_str().unwrap_or(""),
            })
        })
        .collect()
}

/// Extract `[N]` citations from the answer and map each to its source URL.
/// 1-based, in-range only, de-duplicated in first-appearance order — a number
/// the model wasn't given yields no link.
fn cited_urls(answer: &str, sources: &[Value]) -> Vec<Value> {
    let bytes = answer.as_bytes();
    let mut out = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && j < bytes.len() && bytes[j] == b']' {
                if let Ok(n) = answer[i + 1..j].parse::<usize>() {
                    if n >= 1 && n <= sources.len() && seen.insert(n) {
                        out.push(json!({
                            "idx": n,
                            "url": sources[n - 1]["url"].as_str().unwrap_or(""),
                            "title": sources[n - 1]["title"].as_str().unwrap_or(""),
                        }));
                    }
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_queries_extracts_and_drops_blanks() {
        assert_eq!(
            parse_queries(r#"Sure: ["a b", "", "c d e"]"#),
            vec!["a b", "c d e"]
        );
    }

    #[test]
    fn parse_queries_empty_means_no_search() {
        assert!(parse_queries("[]").is_empty());
        assert!(parse_queries("no json").is_empty());
        assert!(parse_queries("error: model down").is_empty());
    }

    #[test]
    fn cited_urls_maps_only_real_in_range() {
        let sources = vec![
            json!({"url": "https://a", "title": "A"}),
            json!({"url": "https://b", "title": "B"}),
        ];
        let c = cited_urls("Foo [1] bar [2] baz [9] qux [1].", &sources);
        assert_eq!(c.len(), 2); // [9] out of range, [1] deduped
        assert_eq!(c[0]["url"], "https://a");
        assert_eq!(c[1]["idx"], 2);
    }
}
