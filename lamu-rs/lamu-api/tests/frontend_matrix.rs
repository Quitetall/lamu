//! Frontend integration matrix (SCALE-TEST P2).
//!
//! For a *running* `lamu serve`, exercise the exact call shape each documented
//! frontend uses (docs/API.md "Point your frontend at LAMU") and assert the
//! response shape that frontend's client library reads:
//!
//!   | Frontend                  | Surface                    | Shape asserted               |
//!   |---------------------------|----------------------------|------------------------------|
//!   | Open WebUI / Continue     | POST /v1/chat/completions  | choices[0].message.content   |
//!   | Open WebUI (model list)   | GET  /v1/models            | data[].id                    |
//!   | Claude Code (non-stream)  | POST /v1/messages          | content[0].text              |
//!   | Claude Code (stream)      | POST /v1/messages stream   | content_block_delta text     |
//!   | AnythingLLM/OWUI (Ollama) | POST /api/chat stream:false| message.content              |
//!   | AnythingLLM (model probe) | GET  /api/tags             | models[].name                |
//!   | RAG front-ends            | POST /v1/embeddings        | data[0].embedding (float[])  |
//!
//! Same gating contract as spec_e2e.rs: skips (returns Ok) when `lamu` is not
//! on PATH, the registry is missing, or no candidate model survives. The chat
//! surfaces additionally skip on host-VRAM contention. The embeddings surface
//! skips when no `embedding`-capability entry exists (the common laptop case),
//! since RAG is opt-in.
//!
//! Slow (real serve + real model load) → `#[ignore]`. Run with:
//!
//! ```bash
//! cargo test --test frontend_matrix -- --ignored --nocapture
//! ```

use serde_json::Value;
use std::time::Duration;

mod common;
use common::{
    body_or_vram_skip, ephemeral_port, lamu_binary, pick_embedding_model, pick_test_model,
    start_lamu_serve, wait_for_health,
};

// ── the matrix ──────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "spawns real lamu serve + loads a real model; run with --ignored"]
async fn frontend_matrix_against_real_serve() {
    let Some(binary) = lamu_binary() else {
        eprintln!("frontend_matrix: `lamu` not on PATH — skipping.");
        return;
    };
    let Some(model) = pick_test_model() else {
        eprintln!("frontend_matrix: no suitable chat model — skipping.");
        return;
    };
    let port = ephemeral_port();
    let _serve = start_lamu_serve(&binary, port);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("client");

    // A cold 27B Q5_K_M load can take 60s+ on a laptop; default 90s, tunable.
    let health_secs = std::env::var("LAMU_MATRIX_HEALTH_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(90);
    if !wait_for_health(&client, port, Duration::from_secs(health_secs)).await {
        panic!("frontend_matrix: /health never came up on :{port} within {health_secs}s (set LAMU_MATRIX_HEALTH_SECS to extend)");
    }
    let base = format!("http://127.0.0.1:{port}");
    eprintln!("frontend_matrix: model '{model}' on :{port}");

    // ── Open WebUI model list: GET /v1/models → data[].id present ───────────
    {
        let v: Value = client
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .expect("GET /v1/models")
            .json()
            .await
            .expect("models json");
        assert_eq!(v["object"], "list", "/v1/models object field");
        let ids: Vec<&str> = v["data"]
            .as_array()
            .expect("/v1/models data array")
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        assert!(!ids.is_empty(), "/v1/models returned no model ids");
        eprintln!("frontend_matrix: /v1/models ok ({} ids)", ids.len());
    }

    // ── AnythingLLM Ollama probe: GET /api/tags → models[].name present ─────
    {
        let v: Value = client
            .get(format!("{base}/api/tags"))
            .send()
            .await
            .expect("GET /api/tags")
            .json()
            .await
            .expect("tags json");
        let names: Vec<&str> = v["models"]
            .as_array()
            .expect("/api/tags models array")
            .iter()
            .filter_map(|m| m["name"].as_str())
            .collect();
        assert!(!names.is_empty(), "/api/tags returned no model names");
        eprintln!("frontend_matrix: /api/tags ok ({} names)", names.len());
    }

    // ── Open WebUI / Continue: POST /v1/chat/completions →
    //    choices[0].message.content ──────────────────────────────────────────
    {
        let resp = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "reply with just ok"}],
                "max_tokens": 64,
                "temperature": 0.0,
                "enable_thinking": false,
            }))
            .send()
            .await
            .expect("POST /v1/chat/completions");
        match body_or_vram_skip(resp, "OpenAI").await {
            Ok(v) => {
                let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
                assert!(!content.is_empty(), "OpenAI choices[0].message.content empty");
                eprintln!("frontend_matrix: /v1/chat/completions ok");
            }
            Err(reason) => {
                eprintln!("frontend_matrix: SKIP rest — {reason}");
                return;
            }
        }
    }

    // ── Claude Code non-stream: POST /v1/messages → content[0].text ─────────
    {
        let resp = client
            .post(format!("{base}/v1/messages"))
            .json(&serde_json::json!({
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "reply with just ok"}],
                "enable_thinking": false,
            }))
            .send()
            .await
            .expect("POST /v1/messages");
        match body_or_vram_skip(resp, "Anthropic").await {
            Ok(v) => {
                assert_eq!(v["type"], "message", "Anthropic top-level type");
                let text = v["content"][0]["text"].as_str().unwrap_or("");
                assert!(!text.is_empty(), "Anthropic content[0].text empty");
                eprintln!("frontend_matrix: /v1/messages (non-stream) ok");
            }
            Err(reason) => {
                eprintln!("frontend_matrix: SKIP rest — {reason}");
                return;
            }
        }
    }

    // ── Claude Code stream: POST /v1/messages stream:true → at least one
    //    content_block_delta carrying text_delta. Read the raw SSE body and
    //    scan for the named-event payload (docs/API.md:448-449). No SSE parser
    //    dep — substring scan over the buffered body is enough to prove the
    //    wire shape a streaming client reads. ──────────────────────────────
    {
        let resp = client
            .post(format!("{base}/v1/messages"))
            .json(&serde_json::json!({
                "model": model,
                "max_tokens": 64,
                "stream": true,
                "messages": [{"role": "user", "content": "reply with just ok"}],
                "enable_thinking": false,
            }))
            .send()
            .await
            .expect("POST /v1/messages stream");
        if resp.status().is_success() {
            let ct = resp
                .headers()
                .get("content-type")
                .and_then(|h| h.to_str().ok())
                .unwrap_or("")
                .to_string();
            assert!(
                ct.starts_with("text/event-stream"),
                "Anthropic stream content-type = {ct}"
            );
            let body = resp.text().await.expect("stream body");
            assert!(
                body.contains("event: content_block_delta"),
                "no content_block_delta event in Anthropic stream:\n{body}"
            );
            assert!(
                body.contains("text_delta"),
                "no text_delta in Anthropic stream:\n{body}"
            );
            assert!(
                body.contains("event: message_stop"),
                "Anthropic stream not terminated by message_stop:\n{body}"
            );
            eprintln!("frontend_matrix: /v1/messages (stream) ok");
        } else {
            let b = resp.text().await.unwrap_or_default();
            if b.contains("VRAM") || b.contains("requires evicting") || b.contains("vram exhausted")
            {
                eprintln!("frontend_matrix: SKIP stream — VRAM contention");
            } else {
                panic!("Anthropic stream non-2xx: {b}");
            }
        }
    }

    // ── AnythingLLM / Open WebUI Ollama mode: POST /api/chat stream:false →
    //    message.content ───────────────────────────────────────────────────
    {
        let resp = client
            .post(format!("{base}/api/chat"))
            .json(&serde_json::json!({
                "model": model,
                "stream": false,
                "messages": [{"role": "user", "content": "reply with just ok"}],
                "enable_thinking": false,
            }))
            .send()
            .await
            .expect("POST /api/chat");
        match body_or_vram_skip(resp, "Ollama").await {
            Ok(v) => {
                let content = v["message"]["content"].as_str().unwrap_or("");
                assert!(!content.is_empty(), "Ollama message.content empty");
                assert_eq!(v["done"], true, "Ollama non-stream done flag");
                eprintln!("frontend_matrix: /api/chat (stream:false) ok");
            }
            Err(reason) => {
                eprintln!("frontend_matrix: SKIP rest — {reason}");
                return;
            }
        }
    }

    // ── RAG front-ends: POST /v1/embeddings → data[0].embedding (float[]).
    //    Opt-in: only run when an `embedding`-capability entry exists, else
    //    skip (most laptop registries have none). ─────────────────────────
    match pick_embedding_model() {
        None => {
            eprintln!("frontend_matrix: no embedding-capability model — SKIP /v1/embeddings");
        }
        Some(emb) => {
            let resp = client
                .post(format!("{base}/v1/embeddings"))
                .json(&serde_json::json!({"model": emb, "input": "hello world"}))
                .send()
                .await
                .expect("POST /v1/embeddings");
            // embeddings is a near-blind passthrough (docs/API.md:356-377):
            // backend status is forwarded. Treat a 5xx as a skip (backend
            // load/VRAM), a 2xx must carry data[0].embedding as a float array.
            let status = resp.status();
            let v: Value = resp.json().await.unwrap_or(Value::Null);
            if status.is_success() {
                let emb_arr = v["data"][0]["embedding"]
                    .as_array()
                    .expect("data[0].embedding array");
                assert!(!emb_arr.is_empty(), "embedding vector empty");
                assert!(
                    emb_arr.iter().all(|x| x.is_number()),
                    "embedding entries must be numbers"
                );
                eprintln!(
                    "frontend_matrix: /v1/embeddings ok (dim {})",
                    emb_arr.len()
                );
            } else {
                eprintln!(
                    "frontend_matrix: SKIP /v1/embeddings — backend status {status}, body {v}"
                );
            }
        }
    }

    drop(_serve);
}
