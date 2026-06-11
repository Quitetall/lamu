//! End-to-end spec test for `lamu serve`.
//!
//! Spawns the real `lamu` binary on an ephemeral port, probes all three
//! HTTP surfaces (OpenAI, Anthropic, Ollama), and asserts the spawn /
//! status / pidfile contract end-to-end. Slow (~15-30s including model
//! load) so gated behind `#[ignore]` — run with:
//!
//! ```bash
//! cargo test --test spec_e2e -- --ignored --nocapture
//! ```
//!
//! Skips automatically (returns Ok) when:
//! - the `lamu` binary isn't on $PATH (CI without prior `cargo install`)
//! - the live registry (~/.local/share/lamu/models.yaml, ADR 0025) is missing
//! - no model in the registry fits in available VRAM (laptop / non-GPU
//!   runners — explicit skip rather than a hang)
//!
//! Failure here means the wire-up regressed end-to-end, even if every
//! unit test still passes. This is the integration backstop.

use serde_json::Value;
use std::time::Duration;

mod common;
use common::{
    ephemeral_port, lamu_binary, pick_test_model, start_lamu_serve, wait_for_health,
};

#[tokio::test]
#[ignore = "spawns real lamu serve + loads a real model; run with --ignored"]
async fn three_surfaces_round_trip_against_real_serve() {
    let Some(binary) = lamu_binary() else {
        eprintln!("spec_e2e: `lamu` not on PATH — skipping. \
                   Install with `cargo install --path lamu-cli` first.");
        return;
    };
    let Some(model) = pick_test_model() else {
        eprintln!("spec_e2e: no suitable test model — registry missing, \
                   empty, or every candidate filtered (no chat capability, \
                   draft arch, or GGUF missing on disk). Skipping.");
        return;
    };
    let port = ephemeral_port();
    let _serve = start_lamu_serve(&binary, port);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))  // model load can be slow
        .build()
        .expect("client");

    if !wait_for_health(&client, port, Duration::from_secs(30)).await {
        panic!("spec_e2e: /health never came up on :{port}");
    }

    eprintln!("spec_e2e: testing against model '{model}' on :{port}");

    /// Returns Ok(json_body) on 2xx. On error: if body looks like host-VRAM
    /// contention, returns Err(skip_reason) so the caller skips; otherwise
    /// panics with the full response.
    async fn assert_2xx_or_skip(
        resp: reqwest::Response, surface: &str,
    ) -> std::result::Result<Value, String> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp.json::<Value>().await.expect("success body should parse"));
        }
        let body = resp.text().await.unwrap_or_default();
        if body.contains("vram exhausted")
            || body.contains("requires evicting")
            || body.contains("VRAM")
        {
            return Err(format!("{surface}: host VRAM contention — {body}"));
        }
        panic!("{surface} returned {status}; body: {body}");
    }

    // OpenAI surface.
    let oai_resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "reply with just ok"}],
            "max_tokens": 200,
            "temperature": 0.0,
            "enable_thinking": false,
        }))
        .send()
        .await
        .expect("openai POST");
    let body = match assert_2xx_or_skip(oai_resp, "OpenAI").await {
        Ok(b) => b,
        Err(reason) => { eprintln!("spec_e2e: SKIP — {reason}"); return; }
    };
    let content = body["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(!content.is_empty(), "OpenAI surface returned empty content");

    // Anthropic surface (non-stream).
    let anthro_resp = client
        .post(format!("http://127.0.0.1:{port}/v1/messages"))
        .json(&serde_json::json!({
            "model": model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "reply with just ok"}],
            "enable_thinking": false,
        }))
        .send()
        .await
        .expect("anthropic POST");
    let body = match assert_2xx_or_skip(anthro_resp, "Anthropic").await {
        Ok(b) => b,
        Err(reason) => { eprintln!("spec_e2e: SKIP — {reason}"); return; }
    };
    let text = body["content"][0]["text"].as_str().unwrap_or("");
    assert!(!text.is_empty(), "Anthropic surface returned empty text");

    // Ollama surface (non-stream).
    let ollama_resp = client
        .post(format!("http://127.0.0.1:{port}/api/chat"))
        .json(&serde_json::json!({
            "model": model,
            "stream": false,
            "messages": [{"role": "user", "content": "reply with just ok"}],
            "enable_thinking": false,
        }))
        .send()
        .await
        .expect("ollama POST");
    let body = match assert_2xx_or_skip(ollama_resp, "Ollama").await {
        Ok(b) => b,
        Err(reason) => { eprintln!("spec_e2e: SKIP — {reason}"); return; }
    };
    let content = body["message"]["content"].as_str().unwrap_or("");
    assert!(!content.is_empty(), "Ollama surface returned empty content");

    // /health now reports the loaded model.
    let h: Value = client
        .get(format!("http://127.0.0.1:{port}/health"))
        .send().await.expect("health").json().await.expect("health json");
    let n = h["models_loaded"].as_u64().unwrap_or(0);
    assert!(n >= 1, "after 3 round-trips, models_loaded must be >= 1; got {n}");

    // Cleanup: ServeHandle Drop kills the child, which should unlink
    // the pidfile. Give it a moment + best-effort verify.
    drop(_serve);
    tokio::time::sleep(Duration::from_millis(500)).await;
    if let Some(rt) = dirs::runtime_dir() {
        let pf = rt.join(format!("lamu-serve-{port}.pid"));
        assert!(!pf.exists(), "pidfile should be unlinked after SIGTERM; still at {}", pf.display());
    }
}
