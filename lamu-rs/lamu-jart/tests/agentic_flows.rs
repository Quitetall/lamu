//! Agentic-flow regression tests over a scripted ToolCtx (FakeCtx) — the
//! first coverage that exercises answer/deep_research/research_chat THROUGH
//! the seam (generate/ensure_loaded/embed) instead of only their parsers.
//! No model, no GPU; the one networked step (SearXNG) is served by an
//! in-test TCP stub.
//!
//! Env discipline: SEARXNG_URL / LAMU_RESEARCH_DIR / JART_SCRAPERS_DIR are
//! process-global, so every test that touches them holds ENV_LOCK
//! (the config.rs pattern — other test binaries are separate processes).

use lamu_core::test_support::FakeCtx;
use lamu_core::tools_ext::ToolCtxError;
use lamu_jart::answer::handle_answer;
use lamu_jart::deep_research::handle_deep_research;
use lamu_jart::chat::handle_research_chat;
use serde_json::{json, Value};
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Minimal one-shot SearXNG stub: accepts ONE connection, returns `body` as
/// an HTTP 200 JSON response, then exits. Returns the bound URL.
async fn searxng_stub(body: String) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        if let Ok((mut sock, _)) = listener.accept().await {
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf).await; // drain the request
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes()).await;
        }
    });
    format!("http://{addr}")
}

fn searx_body(hits: &[(&str, &str, &str)]) -> String {
    let results: Vec<Value> = hits
        .iter()
        .map(|(t, u, s)| json!({"title": t, "url": u, "content": s, "engine": "test"}))
        .collect();
    json!({ "results": results }).to_string()
}

// ───────────────────────────── answer ─────────────────────────────

#[tokio::test]
async fn answer_direct_path_no_search() {
    let _g = ENV_LOCK.lock().unwrap();
    // DECIDE replies [] → direct answer, grounded=false, no network at all.
    let ctx = FakeCtx::new().gen_ok("[]").gen_ok("Pure reasoning answer.");
    let out = handle_answer(&ctx, json!({"question": "Is 7 prime?"})).await;
    let v: Value = serde_json::from_str(&out).expect("json output");
    assert_eq!(v["grounded"], false);
    assert_eq!(v["answer"], "Pure reasoning answer.");
    assert!(v["sources"].as_array().unwrap().is_empty());
    // The direct ANSWER step must carry the long-form budget; DECIDE the default.
    let calls = ctx.generate_calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].2, None, "DECIDE keeps the seam default");
    assert_eq!(calls[1].2, Some(4096), "answer step passes ANSWER_MAX_TOKENS");
}

#[tokio::test]
async fn answer_decide_error_propagates() {
    let _g = ENV_LOCK.lock().unwrap();
    let ctx = FakeCtx::new().gen_err("model down");
    let out = handle_answer(&ctx, json!({"question": "anything"})).await;
    assert!(out.starts_with("error: decide step:"), "got: {out}");
    assert!(out.contains("model down"));
}

#[tokio::test]
async fn answer_local_load_failure_short_circuits() {
    let _g = ENV_LOCK.lock().unwrap();
    let ctx = FakeCtx::new()
        .with_modality("qwen", lamu_core::types::Modality::Llm)
        .enqueue_ensure_loaded(Err(ToolCtxError::Load("out of VRAM".into())));
    let out = handle_answer(&ctx, json!({"question": "q", "model": "qwen"})).await;
    assert!(out.starts_with("error: load model 'qwen'"), "got: {out}");
    assert!(out.contains("out of VRAM"));
    assert!(ctx.generate_calls.lock().unwrap().is_empty(), "no generate after failed load");
}

#[tokio::test]
async fn answer_search_failure_reports_not_errors() {
    let _g = ENV_LOCK.lock().unwrap();
    // DECIDE wants a lookup, but SearXNG is unreachable (closed port) →
    // graceful "no web results" JSON, search_failed populated, no error.
    unsafe { std::env::set_var("SEARXNG_URL", "http://127.0.0.1:9") };
    let ctx = FakeCtx::new().gen_ok(r#"["rust async traits"]"#);
    let out = handle_answer(&ctx, json!({"question": "What are Rust async traits?"})).await;
    unsafe { std::env::remove_var("SEARXNG_URL") };
    let v: Value = serde_json::from_str(&out).expect("json output: {out}");
    assert_eq!(v["grounded"], false);
    assert!(!v["search_failed"].as_array().unwrap().is_empty());
    assert!(v["note"].as_str().unwrap().contains("no web results"));
}

#[tokio::test]
async fn answer_grounded_happy_path_with_citations() {
    let _g = ENV_LOCK.lock().unwrap();
    let drafts = tempfile::tempdir().unwrap();
    let url = searxng_stub(searx_body(&[
        ("Tokio docs", "https://tokio.rs", "Tokio is an async runtime for Rust."),
        ("Async book", "https://rust-lang.github.io/async-book", "Async/await in Rust."),
    ]))
    .await;
    unsafe {
        std::env::set_var("SEARXNG_URL", &url);
        std::env::set_var("LAMU_RESEARCH_DIR", drafts.path());
    }
    let ctx = FakeCtx::new()
        .gen_ok(r#"["tokio async runtime"]"#) // DECIDE
        .gen_ok("Tokio is Rust's async runtime [1]. See also the async book [2]."); // GROUND
    let out = handle_answer(&ctx, json!({"question": "What is Tokio?"})).await;
    unsafe {
        std::env::remove_var("SEARXNG_URL");
        std::env::remove_var("LAMU_RESEARCH_DIR");
    }
    let v: Value = serde_json::from_str(&out).expect("json output");
    assert_eq!(v["grounded"], true);
    assert_eq!(v["sources"].as_array().unwrap().len(), 2);
    let cits = v["citations"].as_array().unwrap();
    assert_eq!(cits.len(), 2);
    assert_eq!(cits[0]["url"], "https://tokio.rs");
    assert_eq!(cits[1]["idx"], 2);
    // Draft auto-saved into the temp research dir.
    let saved = v["saved_to"].as_str().unwrap_or("");
    assert!(saved.starts_with(drafts.path().to_str().unwrap()), "draft in temp dir: {saved}");
    // The grounded prompt must fence the sources as untrusted content.
    let calls = ctx.generate_calls.lock().unwrap();
    assert!(calls[1].1.contains("Tokio is an async runtime"), "snippet reached the prompt");
}

// ─────────────────────────── deep_research ───────────────────────────

#[tokio::test]
async fn deep_research_missing_scrapers_dir_errors_clearly() {
    let _g = ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("JART_SCRAPERS_DIR", "/nonexistent/scrapers-xyz") };
    let ctx = FakeCtx::new();
    let out = handle_deep_research(&ctx, json!({"query": "EEG compression"})).await;
    unsafe { std::env::remove_var("JART_SCRAPERS_DIR") };
    assert!(out.starts_with("error:"), "got: {out}");
    assert!(out.contains("scrapers dir"), "names the problem: {out}");
}

#[tokio::test]
async fn deep_research_empty_corpus_degrades_with_note() {
    let _g = ENV_LOCK.lock().unwrap();
    // Scrapers dir EXISTS but is empty → every source fails → corpus empty →
    // the structured no-papers JSON (not an error string). decompose succeeds.
    let scrapers = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("JART_SCRAPERS_DIR", scrapers.path()) };
    let ctx = FakeCtx::new().gen_ok(r#"["eeg compression"]"#);
    let out = handle_deep_research(&ctx, json!({"query": "EEG compression", "sub_questions": 1})).await;
    unsafe { std::env::remove_var("JART_SCRAPERS_DIR") };
    let v: Value = serde_json::from_str(&out).expect("json output, got: {out}");
    assert_eq!(v["report"], "");
    assert!(v["corpus"].as_array().unwrap().is_empty());
    assert!(v["note"].as_str().unwrap().contains("no papers"));
    assert_eq!(v["sub_questions"][0], "eeg compression");
}

#[tokio::test]
async fn deep_research_decompose_failure_falls_back_to_query() {
    let _g = ENV_LOCK.lock().unwrap();
    let scrapers = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("JART_SCRAPERS_DIR", scrapers.path()) };
    let ctx = FakeCtx::new().gen_err("decompose model down");
    let out = handle_deep_research(&ctx, json!({"query": "spike sorting", "sub_questions": 2})).await;
    unsafe { std::env::remove_var("JART_SCRAPERS_DIR") };
    let v: Value = serde_json::from_str(&out).expect("json output");
    // Fallback: the query itself is the single sub-question.
    assert_eq!(v["sub_questions"].as_array().unwrap().len(), 1);
    assert_eq!(v["sub_questions"][0], "spike sorting");
}

// ─────────────────────────── research_chat ───────────────────────────

#[tokio::test]
async fn research_chat_validates_args() {
    let ctx = FakeCtx::new();
    let out = handle_research_chat(&ctx, json!({"message": "hi"})).await;
    assert!(out.starts_with("error:") && out.contains("session_id"), "got: {out}");
    let out = handle_research_chat(&ctx, json!({"session_id": "s1"})).await;
    assert!(out.starts_with("error:") && out.contains("message"), "got: {out}");
}

#[tokio::test]
async fn research_chat_unknown_session_errors() {
    let ctx = FakeCtx::new();
    let out = handle_research_chat(
        &ctx,
        json!({"session_id": "nope", "message": "what did we find?"}),
    )
    .await;
    assert!(out.contains("unknown or expired session"), "got: {out}");
}
