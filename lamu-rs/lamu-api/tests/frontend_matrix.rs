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
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

// ── shared serve harness (mirrors spec_e2e.rs; kept self-contained so the two
//    test binaries don't need a shared module) ──────────────────────────────

fn ephemeral_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let p = l.local_addr().expect("local_addr").port();
    drop(l);
    p
}

fn lamu_binary() -> Option<PathBuf> {
    which::which("lamu").ok()
}

fn registry_path() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("local-llm")
        .join("config")
        .join("models.yaml")
}

/// Same model-pick strategy as spec_e2e.rs: prefer `main: true`, else the
/// smallest standalone chat entry (>=3B, not a draft/gpt2, GGUF on disk).
fn pick_test_model() -> Option<String> {
    let yaml = std::fs::read_to_string(registry_path()).ok()?;
    let parsed: serde_yaml::Value = serde_yaml::from_str(&yaml).ok()?;
    let models = parsed.get("models")?.as_mapping()?;

    for (k, v) in models {
        let name = k.as_str()?.to_string();
        if v.get("main").and_then(|m| m.as_bool()) == Some(true) {
            return Some(name);
        }
    }

    let mut candidates: Vec<(String, u64)> = models
        .iter()
        .filter_map(|(k, v)| {
            let name = k.as_str()?.to_string();
            let vram = v.get("vram_mb")?.as_u64()?;
            let arch = v.get("arch").and_then(|x| x.as_str()).unwrap_or("");
            let params_b = v
                .get("params_b")
                .and_then(|x| {
                    x.as_f64()
                        .or_else(|| x.as_u64().map(|u| u as f64))
                        .or_else(|| x.as_i64().map(|i| i as f64))
                })
                .unwrap_or(0.0);
            if arch.starts_with("dflash") || arch == "gpt2" {
                return None;
            }
            if params_b < 3.0 {
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
    candidates.sort_by_key(|(_, v)| *v);
    candidates.into_iter().next().map(|(n, _)| n)
}

/// First registry entry advertising the `embedding` capability whose GGUF
/// exists on disk, else None. Mirrors the server's resolution
/// (openai_compat.rs:212-216) closely enough to know whether to run the
/// embeddings leg of the matrix.
fn pick_embedding_model() -> Option<String> {
    let yaml = std::fs::read_to_string(registry_path()).ok()?;
    let parsed: serde_yaml::Value = serde_yaml::from_str(&yaml).ok()?;
    let models = parsed.get("models")?.as_mapping()?;
    for (k, v) in models {
        let name = k.as_str()?.to_string();
        let caps = match v.get("capabilities").and_then(|c| c.as_sequence()) {
            Some(c) => c,
            None => continue,
        };
        if !caps.iter().any(|c| c.as_str() == Some("embedding")) {
            continue;
        }
        if let Some(p) = v.get("path").and_then(|p| p.as_str()) {
            if !std::path::Path::new(p).exists() {
                continue;
            }
        }
        return Some(name);
    }
    None
}

struct ServeHandle {
    child: Child,
}

impl Drop for ServeHandle {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use nix::sys::signal::{kill, Signal};
            use nix::unistd::Pid;
            let pid = self.child.id();
            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while std::time::Instant::now() < deadline {
                if let Ok(Some(_)) = self.child.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_lamu_serve(binary: &PathBuf, port: u16) -> ServeHandle {
    let child = Command::new(binary)
        .args(["serve", "--port", &port.to_string()])
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn lamu serve");
    ServeHandle { child }
}

async fn wait_for_health(client: &reqwest::Client, port: u16, timeout: Duration) -> bool {
    let url = format!("http://127.0.0.1:{}/health", port);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(r) = client.get(&url).timeout(Duration::from_secs(1)).send().await {
            if let Ok(j) = r.json::<Value>().await {
                if j.get("status").and_then(|v| v.as_str()) == Some("ok") {
                    return true;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    false
}

/// Ok(body) on 2xx; Err(reason) when the body looks like host-VRAM contention
/// (caller skips that leg). Panics on any other non-2xx so a real wire-up
/// regression still fails the test.
async fn body_or_vram_skip(resp: reqwest::Response, surface: &str) -> Result<Value, String> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp
            .json::<Value>()
            .await
            .expect("success body should parse"));
    }
    let body = resp.text().await.unwrap_or_default();
    if body.contains("vram exhausted") || body.contains("requires evicting") || body.contains("VRAM")
    {
        return Err(format!("{surface}: host VRAM contention — {body}"));
    }
    panic!("{surface} returned {status}; body: {body}");
}

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

    if !wait_for_health(&client, port, Duration::from_secs(30)).await {
        panic!("frontend_matrix: /health never came up on :{port}");
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
