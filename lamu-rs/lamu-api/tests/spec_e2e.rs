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
//! - $HOME/local-llm/config/models.yaml is missing
//! - no model in the registry fits in available VRAM (laptop / non-GPU
//!   runners — explicit skip rather than a hang)
//!
//! Failure here means the wire-up regressed end-to-end, even if every
//! unit test still passes. This is the integration backstop.

use serde_json::Value;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Allocate an ephemeral port by binding, capturing, then dropping.
/// There's a TOCTOU window between drop and the child's bind but it
/// only widens if some other test races us on the same port; SO_REUSEADDR
/// in `lamu serve` covers the common TIME_WAIT case.
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

/// Choose a model to exercise the surfaces with. Strategy:
///
///   1. Prefer the `main: true` entry — it's the one `lamu serve`
///      preloaded on startup, so the round-trip skips a fresh spawn
///      AND any per-test VRAM contention against that preload.
///   2. Fall back to the smallest standalone chat entry that:
///       - has `chat` capability,
///       - has `params_b >= 3.0` (excludes drafts / speculators that
///         technically declare chat but produce useless output),
///       - is not a `dflash*` arch (drafts),
///       - is not `gpt2` (project-origin proxy, low quality),
///       - whose GGUF path exists on disk.
///
/// Returns `None` if the registry is unreadable or no candidate
/// survives — the test then skips with a message rather than failing.
fn pick_test_model() -> Option<String> {
    let yaml = std::fs::read_to_string(registry_path()).ok()?;
    let parsed: serde_yaml::Value = serde_yaml::from_str(&yaml).ok()?;
    let models = parsed.get("models")?.as_mapping()?;

    // 1. main: true wins outright.
    for (k, v) in models {
        let name = k.as_str()?.to_string();
        if v.get("main").and_then(|m| m.as_bool()) == Some(true) {
            return Some(name);
        }
    }

    // 2. Smallest standalone chat fallback.
    let mut candidates: Vec<(String, u64)> = models
        .iter()
        .filter_map(|(k, v)| {
            let name = k.as_str()?.to_string();
            let vram = v.get("vram_mb")?.as_u64()?;
            let arch = v.get("arch").and_then(|x| x.as_str()).unwrap_or("");
            // `as_f64()` returns None when the YAML field is a plain
            // integer (e.g. `params_b: 7`). Fall back to as_u64/as_i64
            // so integer-typed entries aren't silently filtered out.
            let params_b = v.get("params_b")
                .and_then(|x| x.as_f64()
                    .or_else(|| x.as_u64().map(|u| u as f64))
                    .or_else(|| x.as_i64().map(|i| i as f64)))
                .unwrap_or(0.0);
            if arch.starts_with("dflash") || arch == "gpt2" { return None; }
            if params_b < 3.0 { return None; }
            let caps = v.get("capabilities").and_then(|c| c.as_sequence())?;
            if !caps.iter().any(|c| c.as_str() == Some("chat")) { return None; }
            let path = v.get("path").and_then(|p| p.as_str())?;
            if !std::path::Path::new(path).exists() { return None; }
            Some((name, vram))
        })
        .collect();
    candidates.sort_by_key(|(_, v)| *v);
    candidates.into_iter().next().map(|(n, _)| n)
}

struct ServeHandle {
    child: Child,
}

impl Drop for ServeHandle {
    fn drop(&mut self) {
        // tokio::process::Child::kill() sends SIGKILL on Unix, which
        // skips our graceful-shutdown path (and leaves the pidfile
        // behind). Send SIGTERM directly so the binary's
        // tokio::signal handler runs `with_graceful_shutdown` and the
        // `Drop` on `PidFile` unlinks the file. Fall back to SIGKILL
        // after a brief grace period if the child hangs.
        #[cfg(unix)]
        {
            use nix::sys::signal::{kill, Signal};
            use nix::unistd::Pid;
            let pid = self.child.id();
            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
            // Poll for exit up to 5 seconds — graceful shutdown can be
            // slow on a busy host (flush logs, run preload teardown,
            // unlink pidfile). 5s is generous enough that the SIGKILL
            // escalation only fires on a genuinely-stuck binary; the
            // test then runs its pidfile assertion against the result
            // of a real graceful exit, not a half-killed process.
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while std::time::Instant::now() < deadline {
                if let Ok(Some(_)) = self.child.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        // SIGTERM didn't take or non-Unix: hard kill.
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
