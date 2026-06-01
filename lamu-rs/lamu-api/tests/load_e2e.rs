//! Concurrent load harness (SCALE-TEST P1).
//!
//! Each test spawns ONE real `lamu serve` on an ephemeral port and fans
//! concurrency *inside* that single server (N tasks against one backend),
//! deliberately avoiding the ephemeral-port TOCTOU that spinning up many
//! servers would multiply (see common::ephemeral_port). The shared serve
//! harness lives in `tests/common/mod.rs`.
//!
//! Same gating contract as spec_e2e.rs / frontend_matrix.rs: every test
//! returns Ok (skips) when `lamu` is not on PATH, the registry is missing
//! or yields no candidate model, or a leg hits host-VRAM contention. Slow
//! (real serve + real model load under concurrency) → `#[ignore]`. Run:
//!
//! ```bash
//! cargo test --test load_e2e -- --ignored --nocapture
//! # tune fan-out:
//! LAMU_LOAD_CONCURRENCY=32 cargo test --test load_e2e -- --ignored --nocapture
//! ```
//!
//! Concurrency defaults to 8 (LAMU_LOAD_CONCURRENCY). Health timeout is 90s
//! (LAMU_LOAD_HEALTH_SECS) since a cold 27B load dominates wall time.

mod common;

use common::{
    body_or_vram_skip, ephemeral_port, is_vram_contention, lamu_binary, models_loaded,
    pick_test_model, start_lamu_serve, wait_for_health,
};
use std::time::Duration;

fn concurrency() -> usize {
    std::env::var("LAMU_LOAD_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(8)
}

fn health_secs() -> u64 {
    std::env::var("LAMU_LOAD_HEALTH_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(90)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(120)) // model load can be slow under load
        .build()
        .expect("client")
}

/// Common preamble: resolve binary + model + bring up one serve. Returns
/// (model, port, client, ServeHandle) or None (already-logged skip). The
/// ServeHandle MUST be held by the caller for the server's lifetime.
async fn boot(test: &str) -> Option<(String, u16, reqwest::Client, common::ServeHandle)> {
    let Some(binary) = lamu_binary() else {
        eprintln!("{test}: `lamu` not on PATH — skipping.");
        return None;
    };
    let Some(model) = pick_test_model() else {
        eprintln!("{test}: no suitable chat model — skipping.");
        return None;
    };
    let port = ephemeral_port();
    let serve = start_lamu_serve(&binary, port);
    let client = client();
    let secs = health_secs();
    if !wait_for_health(&client, port, Duration::from_secs(secs)).await {
        panic!("{test}: /health never came up on :{port} within {secs}s (set LAMU_LOAD_HEALTH_SECS to extend)");
    }
    eprintln!("{test}: model '{model}' on :{port}, concurrency={}", concurrency());
    Some((model, port, client, serve))
}

// ── 1. N concurrent /v1/chat/completions to the SAME model → all succeed
//       (or VRAM-skip), and the server holds exactly ONE backend for that
//       name (single-flight: no duplicate spawn). ──────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "spawns real lamu serve + loads a real model under concurrency; run with --ignored"]
async fn concurrent_chat_same_model_single_flight() {
    let Some((model, port, client, _serve)) = boot("load_e2e::single_flight").await else {
        return;
    };
    let n = concurrency();
    let base = format!("http://127.0.0.1:{port}");

    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let client = client.clone();
        let base = base.clone();
        let model = model.clone();
        handles.push(tokio::spawn(async move {
            let resp = client
                .post(format!("{base}/v1/chat/completions"))
                .json(&serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": format!("reply with just ok (req {i})")}],
                    "max_tokens": 64,
                    "temperature": 0.0,
                    "enable_thinking": false,
                }))
                .send()
                .await
                .expect("concurrent /v1/chat/completions");
            body_or_vram_skip(resp, "OpenAI").await
        }));
    }

    let mut ok = 0usize;
    for h in handles {
        match h.await.expect("a concurrent chat task panicked") {
            Ok(v) => {
                let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
                assert!(!content.is_empty(), "OpenAI choices[0].message.content empty under load");
                ok += 1;
            }
            Err(reason) => {
                // Host-VRAM contention: skip the WHOLE test (a partial run can't
                // prove the single-flight invariant).
                eprintln!("load_e2e::single_flight: SKIP — {reason}");
                return;
            }
        }
    }
    assert_eq!(ok, n, "all {n} concurrent requests must succeed when none skipped");

    // Single-flight: N concurrent loads of ONE name must collapse to ONE
    // backend. `/health` models_loaded counts distinct loaded backends
    // (scheduler.loaded_models().len()); >1 here would mean a duplicate
    // spawn of the same model leaked past the loader's per-name gate.
    let loaded = models_loaded(&client, port).await.expect("models_loaded probe");
    assert_eq!(
        loaded, 1,
        "single model under {n}-way concurrency must yield exactly 1 backend; got {loaded}"
    );
}

// ── 2. /v1/chat/completions + /v1/messages + /api/chat concurrently against
//       one serve → every surface answers (or the whole test VRAM-skips). ───
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "spawns real lamu serve + loads a real model under concurrency; run with --ignored"]
async fn mixed_surface_concurrent() {
    let Some((model, port, client, _serve)) = boot("load_e2e::mixed_surface").await else {
        return;
    };
    let base = format!("http://127.0.0.1:{port}");

    // Fire one request per surface concurrently and join. Each closure
    // returns Result<(), String>: Err = VRAM skip (whole test skips).
    let oai = {
        let client = client.clone();
        let base = base.clone();
        let model = model.clone();
        async move {
            let resp = client
                .post(format!("{base}/v1/chat/completions"))
                .json(&serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": "reply with just ok"}],
                    "max_tokens": 64, "temperature": 0.0, "enable_thinking": false,
                }))
                .send().await.expect("POST /v1/chat/completions");
            body_or_vram_skip(resp, "OpenAI").await.map(|v| {
                assert!(
                    !v["choices"][0]["message"]["content"].as_str().unwrap_or("").is_empty(),
                    "OpenAI content empty"
                );
            })
        }
    };
    let anthro = {
        let client = client.clone();
        let base = base.clone();
        let model = model.clone();
        async move {
            let resp = client
                .post(format!("{base}/v1/messages"))
                .json(&serde_json::json!({
                    "model": model, "max_tokens": 64,
                    "messages": [{"role": "user", "content": "reply with just ok"}],
                    "enable_thinking": false,
                }))
                .send().await.expect("POST /v1/messages");
            body_or_vram_skip(resp, "Anthropic").await.map(|v| {
                assert_eq!(v["type"], "message", "Anthropic top-level type");
                assert!(
                    !v["content"][0]["text"].as_str().unwrap_or("").is_empty(),
                    "Anthropic text empty"
                );
            })
        }
    };
    let ollama = {
        let client = client.clone();
        let base = base.clone();
        let model = model.clone();
        async move {
            let resp = client
                .post(format!("{base}/api/chat"))
                .json(&serde_json::json!({
                    "model": model, "stream": false,
                    "messages": [{"role": "user", "content": "reply with just ok"}],
                    "enable_thinking": false,
                }))
                .send().await.expect("POST /api/chat");
            body_or_vram_skip(resp, "Ollama").await.map(|v| {
                assert!(
                    !v["message"]["content"].as_str().unwrap_or("").is_empty(),
                    "Ollama content empty"
                );
                assert_eq!(v["done"], true, "Ollama non-stream done flag");
            })
        }
    };

    let (r_oai, r_anthro, r_ollama) = tokio::join!(oai, anthro, ollama);
    for r in [r_oai, r_anthro, r_ollama] {
        if let Err(reason) = r {
            eprintln!("load_e2e::mixed_surface: SKIP — {reason}");
            return;
        }
    }

    // All three surfaces target the same model name → still one backend.
    let loaded = models_loaded(&client, port).await.expect("models_loaded probe");
    assert_eq!(
        loaded, 1,
        "three surfaces against one model must share one backend; got {loaded}"
    );
    eprintln!("load_e2e::mixed_surface: all three surfaces ok");
}

// ── 3. N concurrent STREAMS, one per surface format, validated terminators:
//       OpenAI `data: [DONE]` (text/event-stream), Anthropic `event:
//       message_stop` (text/event-stream), Ollama `"done":true`
//       (application/x-ndjson). ──────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "spawns real lamu serve + loads a real model under concurrency; run with --ignored"]
async fn streaming_under_load() {
    let Some((model, port, client, _serve)) = boot("load_e2e::streaming").await else {
        return;
    };
    let n = concurrency();
    let base = format!("http://127.0.0.1:{port}");

    // Round-robin the three streaming formats across N concurrent streams.
    #[derive(Clone, Copy)]
    enum Fmt {
        OpenAi,
        Anthropic,
        Ollama,
    }
    let fmts = [Fmt::OpenAi, Fmt::Anthropic, Fmt::Ollama];

    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let client = client.clone();
        let base = base.clone();
        let model = model.clone();
        let fmt = fmts[i % fmts.len()];
        handles.push(tokio::spawn(async move {
            // Returns Ok(true)=verified, Ok(false)=VRAM skip, panics on regression.
            match fmt {
                Fmt::OpenAi => {
                    let resp = client
                        .post(format!("{base}/v1/chat/completions"))
                        .json(&serde_json::json!({
                            "model": model, "stream": true, "max_tokens": 64,
                            "temperature": 0.0, "enable_thinking": false,
                            "messages": [{"role": "user", "content": "reply with just ok"}],
                        }))
                        .send().await.expect("POST /v1/chat/completions stream");
                    if !resp.status().is_success() {
                        let b = resp.text().await.unwrap_or_default();
                        if is_vram_contention(&b) { return false; }
                        panic!("OpenAI stream non-2xx: {b}");
                    }
                    let ct = resp.headers().get("content-type")
                        .and_then(|h| h.to_str().ok()).unwrap_or("").to_string();
                    assert!(ct.starts_with("text/event-stream"),
                        "OpenAI stream content-type = {ct}");
                    let body = resp.text().await.expect("openai stream body");
                    assert!(body.contains("data: [DONE]"),
                        "OpenAI stream not terminated by [DONE]:\n{body}");
                    true
                }
                Fmt::Anthropic => {
                    let resp = client
                        .post(format!("{base}/v1/messages"))
                        .json(&serde_json::json!({
                            "model": model, "stream": true, "max_tokens": 64,
                            "enable_thinking": false,
                            "messages": [{"role": "user", "content": "reply with just ok"}],
                        }))
                        .send().await.expect("POST /v1/messages stream");
                    if !resp.status().is_success() {
                        let b = resp.text().await.unwrap_or_default();
                        if is_vram_contention(&b) { return false; }
                        panic!("Anthropic stream non-2xx: {b}");
                    }
                    let ct = resp.headers().get("content-type")
                        .and_then(|h| h.to_str().ok()).unwrap_or("").to_string();
                    assert!(ct.starts_with("text/event-stream"),
                        "Anthropic stream content-type = {ct}");
                    let body = resp.text().await.expect("anthropic stream body");
                    assert!(body.contains("event: content_block_delta"),
                        "no content_block_delta in Anthropic stream:\n{body}");
                    assert!(body.contains("text_delta"),
                        "no text_delta in Anthropic stream:\n{body}");
                    assert!(body.contains("event: message_stop"),
                        "Anthropic stream not terminated by message_stop:\n{body}");
                    true
                }
                Fmt::Ollama => {
                    let resp = client
                        .post(format!("{base}/api/chat"))
                        .json(&serde_json::json!({
                            "model": model, "stream": true,
                            "enable_thinking": false,
                            "messages": [{"role": "user", "content": "reply with just ok"}],
                        }))
                        .send().await.expect("POST /api/chat stream");
                    if !resp.status().is_success() {
                        let b = resp.text().await.unwrap_or_default();
                        if is_vram_contention(&b) { return false; }
                        panic!("Ollama stream non-2xx: {b}");
                    }
                    let ct = resp.headers().get("content-type")
                        .and_then(|h| h.to_str().ok()).unwrap_or("").to_string();
                    assert!(ct.starts_with("application/x-ndjson"),
                        "Ollama stream content-type = {ct}");
                    let body = resp.text().await.expect("ollama stream body");
                    // Final NDJSON line carries `"done":true` (json! has no spaces).
                    assert!(body.contains("\"done\":true"),
                        "Ollama stream not terminated by done:true:\n{body}");
                    true
                }
            }
        }));
    }

    let mut verified = 0usize;
    for h in handles {
        if h.await.expect("a concurrent stream task panicked") {
            verified += 1;
        } else {
            eprintln!("load_e2e::streaming: SKIP — VRAM contention on at least one stream");
            return;
        }
    }
    assert_eq!(verified, n, "all {n} concurrent streams must verify their terminator");
    let loaded = models_loaded(&client, port).await.expect("models_loaded probe");
    assert_eq!(loaded, 1, "streaming load against one model must share one backend; got {loaded}");
}

// ── 4. Eviction refused under parallel load: a model that would require
//       evicting another must be refused (ADR-0006, loader.rs:168-174) with
//       503 + "won't auto-evict", and must NOT leak a `mark_loading` slot
//       (models_loaded stays put). Skips when the registry can't construct
//       the contention (no second model that fits only after eviction). ─────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "spawns real lamu serve + loads a real model under concurrency; run with --ignored"]
async fn eviction_refused_under_parallel_load() {
    let Some((model, port, client, _serve)) = boot("load_e2e::eviction").await else {
        return;
    };
    let base = format!("http://127.0.0.1:{port}");

    // Warm the primary model with one request so a backend is resident and
    // holding VRAM (so a second large model would need eviction).
    {
        let resp = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "reply with just ok"}],
                "max_tokens": 16, "temperature": 0.0, "enable_thinking": false,
            }))
            .send().await.expect("warm POST");
        match body_or_vram_skip(resp, "OpenAI(warm)").await {
            Ok(_) => {}
            Err(reason) => {
                eprintln!("load_e2e::eviction: SKIP — {reason}");
                return;
            }
        }
    }

    // Find a SECOND distinct chat model whose VRAM, added on top of the
    // resident one, would force eviction. We can't compute exact host VRAM
    // here, so we drive load via the registry's other chat entries and look
    // for the ADR-0006 refusal. If no other chat model exists OR every one
    // fits alongside the first (no eviction needed), skip — the invariant
    // can only be asserted when contention is actually provoked.
    let Some(other) = common::pick_test_model().and_then(|primary| {
        // pick_test_model() returns the same primary; re-scan the registry
        // for a different chat candidate (largest, to maximize eviction need).
        let yaml = std::fs::read_to_string(common::registry_path()).ok()?;
        let parsed: serde_yaml::Value = serde_yaml::from_str(&yaml).ok()?;
        let models = parsed.get("models")?.as_mapping()?;
        let mut others: Vec<(String, u64)> = models
            .iter()
            .filter_map(|(k, v)| {
                let name = k.as_str()?.to_string();
                if name == primary {
                    return None;
                }
                let vram = v.get("vram_mb")?.as_u64()?;
                let arch = v.get("arch").and_then(|x| x.as_str()).unwrap_or("");
                if arch.starts_with("dflash") || arch == "gpt2" {
                    return None;
                }
                let caps = v.get("capabilities").and_then(|c| c.as_sequence())?;
                if !caps.iter().any(|c| c.as_str() == Some("chat")) {
                    return None;
                }
                let path = v.get("path").and_then(|p| p.as_str())?;
                if !std::path::Path::new(path).exists() {
                    return None;
                }
                Some((name, vram))
            })
            .collect();
        // Largest first — most likely to require eviction.
        others.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
        others.into_iter().next().map(|(n, _)| n)
    }) else {
        eprintln!("load_e2e::eviction: no second chat model on disk to provoke eviction — SKIP");
        return;
    };

    let before = models_loaded(&client, port).await.expect("models_loaded before");

    // Fire N concurrent requests at the SECOND model. If it needs eviction,
    // every one must get the ADR-0006 refusal (503, won't auto-evict). If it
    // happens to fit (no eviction), the requests succeed and we can't assert
    // the refusal — skip the assertion in that case (load test stays honest).
    let n = concurrency();
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let client = client.clone();
        let base = base.clone();
        let other = other.clone();
        handles.push(tokio::spawn(async move {
            let resp = client
                .post(format!("{base}/v1/chat/completions"))
                .json(&serde_json::json!({
                    "model": other,
                    "messages": [{"role": "user", "content": "reply with just ok"}],
                    "max_tokens": 16, "temperature": 0.0, "enable_thinking": false,
                }))
                .send().await.expect("POST second model");
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            (status, body)
        }));
    }

    let mut saw_eviction_refusal = false;
    let mut all_fit = true;
    for h in handles {
        let (status, body) = h.await.expect("eviction task panicked");
        if body.contains("won't auto-evict") || body.contains("requires evicting") {
            // ADR-0006: 503 SERVICE_UNAVAILABLE with spawn_failed envelope.
            assert_eq!(status, 503,
                "eviction refusal must be HTTP 503; got {status}, body: {body}");
            assert!(body.contains("won't auto-evict"),
                "refusal body must name the won't-auto-evict policy; got: {body}");
            saw_eviction_refusal = true;
            all_fit = false;
        } else if status == 200 {
            // The second model fit without eviction on this host.
        } else if is_vram_contention(&body) {
            // Plain VRAM-exhausted (no eviction candidate) also counts as the
            // server correctly refusing to overcommit; not the ADR-0006 path,
            // so don't assert the eviction string, but it's not a regression.
            all_fit = false;
        } else {
            panic!("second-model request unexpected status {status}: {body}");
        }
    }

    if !saw_eviction_refusal {
        eprintln!("load_e2e::eviction: second model fit without eviction (all_fit={all_fit}) — \
                   ADR-0006 path not provoked on this host; SKIP assertion");
        return;
    }

    // No leaked mark_loading: a refused eviction returns BEFORE
    // `mark_loading` (loader.rs:159-178), so the loaded count must be
    // unchanged — the refused model never occupies a slot.
    let after = models_loaded(&client, port).await.expect("models_loaded after");
    assert_eq!(
        after, before,
        "refused eviction must not leak a loading slot: models_loaded {before} → {after}"
    );
    eprintln!("load_e2e::eviction: ADR-0006 refusal observed under {n}-way load, no leaked slot");
}
