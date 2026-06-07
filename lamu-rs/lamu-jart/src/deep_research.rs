//! `deep_research` MCP tool — multi-step research orchestrator (ADR 0023, R1).
//!
//! query → decompose into sub-questions → CONCURRENT multi-source search per
//! sub-question (jart's HuggingFace/PubMed/bioRxiv/Semantic fan-out) → merge +
//! dedup + index a citeable corpus → cited synthesis via `ctx.generate`.
//! Citations are resolved `[N]` → `corpus[N-1].link` in CODE, so every reference
//! is a real retrieved paper — hallucinated links are structurally impossible.
//!
//! Embeddings-rank (R3), follow-up chat (R4), per-claim verification (R5), and
//! web search (R6) layer on later. Fan-out is N concurrent `ctx.generate` /
//! `feed::load` — the per-provider concurrency cap is enforced inside LAMU, so
//! the module never reimplements throttling (ADR 0023 / cloud.rs semaphore).

use crate::research::{default_research_model, scrapers_dir};
use jart::core::{ai, cache::Cache, config::Topic, feed, model::Paper, ratelimit::Pacer};
use lamu_core::tools_ext::ToolCtx;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;

const DECOMPOSE_PROMPT: &str =
    "You are planning a literature search. Break the user's research question into \
     a few focused, non-overlapping sub-questions that together cover it. Reply \
     with ONLY a JSON array of strings (the sub-questions) and nothing else.";

const SYNTH_PROMPT: &str =
    "You are a research assistant writing a grounded briefing. Using ONLY the \
     numbered sources provided, answer the user's question in a few short \
     paragraphs. Cite every claim inline as [N], where N is the source's id \
     number; never cite a number that wasn't provided and never invent a source. \
     Lead with the direct answer, then the supporting detail.";

/// JSON schema for the `deep_research` MCP tool.
pub fn schema_deep_research() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The research question to investigate."},
            "sub_questions": {"type": "integer", "default": 4, "description": "How many sub-questions to decompose into (1-8)."},
            "limit_per_source": {"type": "integer", "default": 6, "description": "Max items per source per sub-question (1-15)."},
            "decompose_model": {"type": "string", "description": "Model for query decomposition. Defaults to $LAMU_RESEARCH_MODEL, else mimo-v2.5."},
            "synthesis_model": {"type": "string", "description": "Model for the final cited synthesis. Defaults to $LAMU_RESEARCH_MODEL, else mimo-v2.5."}
        },
        "required": ["query"]
    })
}

pub fn dispatch_deep_research<'a>(
    ctx: &'a dyn ToolCtx,
    args: Value,
) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(handle_deep_research(ctx, args))
}

pub async fn handle_deep_research(ctx: &dyn ToolCtx, args: Value) -> String {
    let query = args["query"].as_str().unwrap_or("").trim().to_string();
    if query.is_empty() {
        return "error: deep_research requires a non-empty `query`".into();
    }
    let n_sub = args["sub_questions"].as_u64().unwrap_or(4).clamp(1, 8) as usize;
    let limit = args["limit_per_source"].as_u64().unwrap_or(6).clamp(1, 15) as usize;
    let decompose_model = args["decompose_model"].as_str().map(String::from).unwrap_or_else(default_research_model);
    let synthesis_model = args["synthesis_model"].as_str().map(String::from).unwrap_or_else(default_research_model);

    let sdir = scrapers_dir();
    if !sdir.exists() {
        return format!(
            "error: jart scrapers dir not found at {} — set JART_SCRAPERS_DIR to the jart checkout's scrapers/",
            sdir.display()
        );
    }

    // 1. Decompose (1 generate). Falls back to [query] on any failure.
    let subqs = decompose(ctx, &decompose_model, &query, n_sub).await;

    // 2. Concurrent multi-source search — one feed::load per sub-question, all
    //    sharing one Cache + Pacer (so the pacer throttles per-source globally
    //    and the cache de-dups repeated fetches across sub-questions).
    let cache = Cache::new();
    let pacer = Pacer::new();
    let topics: Vec<Topic> = subqs
        .iter()
        .map(|sq| Topic { id: "dr".into(), label: sq.clone(), hf: sq.clone(), pubmed: sq.clone() })
        .collect();
    let searches = topics
        .iter()
        .map(|t| feed::load(&sdir, std::slice::from_ref(t), limit, &cache, &pacer));
    let feeds = futures::future::join_all(searches).await;

    // 3. Merge + dedup (by normalized title) + collect source failures.
    let mut corpus: Vec<Paper> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut sources_failed: Vec<Value> = Vec::new();
    for f in feeds {
        for e in f.errors {
            sources_failed.push(json!({"source": e.source, "message": e.message}));
        }
        for p in f.papers {
            let key = title_key(&p.title);
            if !key.is_empty() && !seen.insert(key) {
                continue;
            }
            corpus.push(p);
        }
    }
    corpus.sort_by(|a, b| b.ts.cmp(&a.ts));

    if corpus.is_empty() {
        return json!({
            "query": query, "sub_questions": subqs, "corpus": [],
            "report": "", "citations": [], "sources_failed": sources_failed,
            "note": "no papers matched the sub-questions"
        })
        .to_string();
    }

    // Bound the synthesis context: when the corpus is large, rank by relevance to
    // the query via embeddings (RAG) and keep the top-K; else use it as-is. On
    // any embedding failure this falls back to the recency order.
    let corpus = rank_corpus(ctx, &query, corpus).await;

    // 4. Cited synthesis (1 generate). build_grounded_content fences each item as
    //    <source id="N">; the prompt instructs the model to cite [N].
    let items: Vec<String> = corpus
        .iter()
        .map(|p| format!("Title: {}\nAbstract: {}", p.title, p.grounding))
        .collect();
    let synth_instruction = format!("{SYNTH_PROMPT}\n\nUser question: {query}");
    let content = ai::build_grounded_content(&synth_instruction, &items);

    // A local synthesis model must be loaded first (cloud routes via generate).
    if ctx.model_modality(&synthesis_model).is_some() {
        let status = ctx.ensure_loaded(&synthesis_model).await;
        if status.trim_start().to_lowercase().starts_with("error:") {
            return with_corpus(&query, &subqs, &corpus, &sources_failed, "synthesis_error", &status);
        }
    }
    let report = ctx.generate(&synthesis_model, &content).await;
    if report.trim_start().to_lowercase().starts_with("error:") {
        return with_corpus(&query, &subqs, &corpus, &sources_failed, "synthesis_error", &report);
    }

    // 5. Resolve cited [N] → corpus[N-1].link IN CODE (only real, in-range refs).
    let citations = cited_links(&report, &corpus);

    json!({
        "query": query,
        "sub_questions": subqs,
        "corpus": corpus_json(&corpus),
        "report": report,
        "citations": citations,
        "sources_failed": sources_failed
    })
    .to_string()
}

/// Max papers fed to synthesis. Above this, RAG-rank and keep the top-K so the
/// synthesis prompt stays bounded.
const MAX_SYNTH: usize = 18;

/// Rank the corpus by cosine similarity to the query (embeddings RAG) and keep
/// the top `MAX_SYNTH`. No-op when the corpus already fits; falls back to the
/// existing (recency) order on any embedding failure.
async fn rank_corpus(ctx: &dyn ToolCtx, query: &str, mut corpus: Vec<Paper>) -> Vec<Paper> {
    if corpus.len() <= MAX_SYNTH {
        return corpus;
    }
    let mut texts: Vec<String> = Vec::with_capacity(corpus.len() + 1);
    texts.push(query.to_string());
    texts.extend(corpus.iter().map(|p| format!("{} {}", p.title, p.grounding)));
    let embs = match ctx.embed(&texts).await {
        Ok(e) if e.len() == corpus.len() + 1 => e,
        _ => {
            corpus.truncate(MAX_SYNTH); // recency fallback
            return corpus;
        }
    };
    let q = embs[0].clone();
    let mut scored: Vec<(f32, Paper)> = corpus
        .into_iter()
        .enumerate()
        .map(|(i, p)| (cosine(&q, &embs[i + 1]), p))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(MAX_SYNTH).map(|(_, p)| p).collect()
}

/// Cosine similarity; 0 for a zero vector or length mismatch.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Normalized dedup key for a title (lowercase ascii-alphanumeric, capped).
fn title_key(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(120)
        .collect()
}

fn corpus_json(corpus: &[Paper]) -> Vec<Value> {
    corpus
        .iter()
        .enumerate()
        .map(|(i, p)| {
            json!({"idx": i + 1, "title": p.title, "link": p.link, "source": p.source, "date": p.date_label})
        })
        .collect()
}

/// Build a result that keeps the corpus but reports a synthesis failure in
/// `field` — the searched studies are still useful without the writeup.
fn with_corpus(
    query: &str,
    subqs: &[String],
    corpus: &[Paper],
    sources_failed: &[Value],
    field: &str,
    err: &str,
) -> String {
    json!({
        "query": query,
        "sub_questions": subqs,
        "corpus": corpus_json(corpus),
        "report": "",
        "citations": [],
        "sources_failed": sources_failed,
        field: err
    })
    .to_string()
}

/// Decompose the query into sub-questions; fall back to [query] on any failure.
async fn decompose(ctx: &dyn ToolCtx, model: &str, query: &str, n: usize) -> Vec<String> {
    if ctx.model_modality(model).is_some() {
        let status = ctx.ensure_loaded(model).await;
        if status.trim_start().to_lowercase().starts_with("error:") {
            return vec![query.to_string()];
        }
    }
    let prompt = format!("{DECOMPOSE_PROMPT}\n\nResearch question: {query}\n\n(at most {n} sub-questions)");
    let out = ctx.generate(model, &prompt).await;
    parse_subquestions(&out, query, n)
}

/// Tolerant parse: extract a JSON array of strings from model output (first `[`
/// to last `]`); fall back to `[query]`. Caps at `n`, drops blanks.
pub(crate) fn parse_subquestions(text: &str, query: &str, n: usize) -> Vec<String> {
    let fallback = || vec![query.to_string()];
    if text.trim_start().to_lowercase().starts_with("error:") {
        return fallback();
    }
    let (a, b) = match (text.find('['), text.rfind(']')) {
        (Some(a), Some(b)) if b > a => (a, b),
        _ => return fallback(),
    };
    match serde_json::from_str::<Vec<String>>(&text[a..=b]) {
        Ok(v) => {
            let cleaned: Vec<String> = v
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .take(n)
                .collect();
            if cleaned.is_empty() { fallback() } else { cleaned }
        }
        Err(_) => fallback(),
    }
}

/// Extract `[N]` citations from the report and map each to its corpus paper.
/// 1-based, in-range only, de-duplicated in first-appearance order — so a model
/// that cites a number it wasn't given simply produces no link.
pub(crate) fn cited_links(report: &str, corpus: &[Paper]) -> Vec<Value> {
    let bytes = report.as_bytes();
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
                if let Ok(n) = report[i + 1..j].parse::<usize>() {
                    if n >= 1 && n <= corpus.len() && seen.insert(n) {
                        let p = &corpus[n - 1];
                        out.push(json!({"idx": n, "link": p.link, "title": p.title}));
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

    fn paper(title: &str, link: &str) -> Paper {
        Paper {
            kind: "paper".into(), source: "HF".into(), topic: "t".into(),
            title: title.into(), link: link.into(), date_label: "2026-01-01".into(),
            ts: 1, summary: "s".into(), grounding: "g".into(),
        }
    }

    #[test]
    fn parse_subquestions_extracts_json_array() {
        let out = parse_subquestions(r#"Sure: ["a?", "b?", "c?"]"#, "q", 4);
        assert_eq!(out, vec!["a?", "b?", "c?"]);
    }

    #[test]
    fn parse_subquestions_caps_and_drops_blanks() {
        let out = parse_subquestions(r#"["x", "", "y", "z", "w"]"#, "q", 2);
        assert_eq!(out, vec!["x", "y"]);
    }

    #[test]
    fn parse_subquestions_falls_back_on_garbage_or_error() {
        assert_eq!(parse_subquestions("no json here", "the query", 4), vec!["the query"]);
        assert_eq!(parse_subquestions("error: model down", "the query", 4), vec!["the query"]);
    }

    #[test]
    fn cited_links_maps_only_real_in_range_indices() {
        let corpus = vec![paper("A", "https://a"), paper("B", "https://b")];
        // [1] and [2] valid; [9] out of range -> dropped; [1] repeat -> deduped.
        let cites = cited_links("Foo [1] bar [2] baz [9] qux [1].", &corpus);
        assert_eq!(cites.len(), 2);
        assert_eq!(cites[0]["idx"], 1);
        assert_eq!(cites[0]["link"], "https://a");
        assert_eq!(cites[1]["idx"], 2);
    }

    #[test]
    fn cited_links_empty_when_no_citations() {
        let corpus = vec![paper("A", "https://a")];
        assert!(cited_links("No citations in this prose.", &corpus).is_empty());
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0); // zero vector
        assert_eq!(cosine(&[1.0], &[1.0, 1.0]), 0.0); // length mismatch
    }
}
