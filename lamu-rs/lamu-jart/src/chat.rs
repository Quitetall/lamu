//! `research_chat` MCP tool (ADR 0023, R4) — follow-up Q&A grounded in a
//! `deep_research` session's corpus via embeddings retrieval.
//!
//! Embed the user's message, cosine-rank the session corpus (RAG), ground the
//! top-K studies via `build_grounded_content`, and answer with `[N]` citations
//! that resolve (in code) to the ranked studies — so chat refs are real papers
//! too. Stateless per call: the session corpus + embeddings live in `session`.

use crate::deep_research::{cited_links, cosine};
use crate::research::default_research_model;
use crate::session;
use jart::core::{ai, model::Paper};
use lamu_core::tools_ext::ToolCtx;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;

const CHAT_PROMPT: &str =
    "You are continuing a research conversation. Answer the user's follow-up using \
     ONLY the numbered sources below (retrieved earlier for them). Cite every claim \
     inline as [N] using the source id; never invent a source. If the sources don't \
     cover the question, say so plainly.";

/// How many of the session's studies to ground each chat turn on.
const TOP_K: usize = 8;

pub fn schema_research_chat() -> Value {
    json!({
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "A session id returned by deep_research."},
            "message": {"type": "string", "description": "The follow-up question."},
            "model": {"type": "string", "description": "Model for the answer. Defaults to $LAMU_RESEARCH_MODEL, else mimo-v2.5."}
        },
        "required": ["session_id", "message"]
    })
}

pub fn dispatch_research_chat<'a>(
    ctx: &'a dyn ToolCtx,
    args: Value,
) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(handle_research_chat(ctx, args))
}

pub async fn handle_research_chat(ctx: &dyn ToolCtx, args: Value) -> String {
    let sid = args["session_id"].as_str().unwrap_or("").trim().to_string();
    if sid.is_empty() {
        return "error: research_chat requires a `session_id` (from deep_research)".into();
    }
    let message = args["message"].as_str().unwrap_or("").trim().to_string();
    if message.is_empty() {
        return "error: research_chat requires a non-empty `message`".into();
    }
    let model = args["model"].as_str().map(String::from).unwrap_or_else(default_research_model);

    let Some((corpus, embeddings)) = session::get(&sid) else {
        return format!("error: unknown or expired session '{sid}' — run deep_research first");
    };

    // RAG: rank the corpus by relevance to the message, keep top-K (the numbered
    // sources the model will see + cite).
    let ranked = rank_by_message(ctx, &message, &corpus, &embeddings).await;

    let items: Vec<String> = ranked
        .iter()
        .map(|p| format!("Title: {}\nAbstract: {}", p.title, p.grounding))
        .collect();
    let instruction = format!("{CHAT_PROMPT}\n\nFollow-up question: {message}");
    let content = ai::build_grounded_content(&instruction, &items);

    if ctx.model_modality(&model).is_some() {
        let status = ctx.ensure_loaded(&model).await;
        if status.trim_start().to_lowercase().starts_with("error:") {
            return format!("error: load model '{model}': {status}");
        }
    }
    let answer = ctx.generate(&model, &content).await;
    if answer.trim_start().to_lowercase().starts_with("error:") {
        return format!("error: {answer}");
    }

    // Citations resolve against the RANKED subset (the numbered sources shown).
    let citations = cited_links(&answer, &ranked);
    json!({
        "session_id": sid,
        "answer": answer,
        "citations": citations
    })
    .to_string()
}

/// Rank the session corpus by cosine similarity of the message to each paper's
/// stored embedding; return the top-K papers. Falls back to the first K papers
/// when embeddings are unavailable/mismatched or the message can't be embedded.
async fn rank_by_message(
    ctx: &dyn ToolCtx,
    message: &str,
    corpus: &[Paper],
    embeddings: &[Vec<f32>],
) -> Vec<Paper> {
    if corpus.len() <= TOP_K || embeddings.len() != corpus.len() {
        return corpus.iter().take(TOP_K).cloned().collect();
    }
    let qemb = match ctx.embed(&[message.to_string()]).await {
        Ok(e) if !e.is_empty() => e[0].clone(),
        _ => return corpus.iter().take(TOP_K).cloned().collect(),
    };
    let mut scored: Vec<(f32, &Paper)> = corpus
        .iter()
        .enumerate()
        .map(|(i, p)| (cosine(&qemb, &embeddings[i]), p))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(TOP_K).map(|(_, p)| p.clone()).collect()
}
