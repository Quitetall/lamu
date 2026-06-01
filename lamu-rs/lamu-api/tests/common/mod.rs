//! Shared real-port serve harness for the ignore-gated e2e/load test
//! binaries (`spec_e2e.rs`, `frontend_matrix.rs`, `load_e2e.rs`).
//!
//! This file lives at `tests/common/mod.rs`, so Cargo auto-includes it as a
//! *module* in any integration test that does `mod common;` — it is NOT
//! compiled as its own test binary (no "running 0 tests" noise). The
//! module-level `#![allow(dead_code)]` is required: each test binary compiles
//! the whole module but calls only a subset of the helpers (e.g. only
//! `frontend_matrix.rs` uses `pick_embedding_model`), and without it the
//! unused helpers would trip `cargo clippy -- -D warnings` per-binary.

#![allow(dead_code)]

use serde_json::Value;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Allocate an ephemeral port by binding, capturing, then dropping.
/// There's a TOCTOU window between drop and the child's bind but it
/// only widens if some other test races us on the same port; SO_REUSEADDR
/// in `lamu serve` covers the common TIME_WAIT case. Per the load-harness
/// SPEC, each test allocates exactly ONE port and fans concurrency inside
/// the single server, so this window is hit at most once per test.
pub fn ephemeral_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let p = l.local_addr().expect("local_addr").port();
    drop(l);
    p
}

pub fn lamu_binary() -> Option<PathBuf> {
    which::which("lamu").ok()
}

pub fn registry_path() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("local-llm")
        .join("config")
        .join("models.yaml")
}

/// Choose a model to exercise the surfaces with. Strategy:
///
///   1. Prefer the `main: true` entry — it's the one `lamu serve`
///      preloaded on startup, so the round-trip skips a fresh spawn
///      AND any per-test VRAM contention against that preload. The load
///      harness relies on this: with the model preloaded, N concurrent
///      requests must collapse onto exactly one backend (single-flight).
///   2. Fall back to the smallest standalone chat entry that:
///       - has `chat` capability,
///       - has `params_b >= 3.0` (excludes drafts / speculators that
///         technically declare chat but produce useless output),
///       - is not a `dflash*` arch (drafts),
///       - is not `gpt2` (project-origin proxy, low quality),
///       - whose GGUF path exists on disk.
///
/// Returns `None` if the registry is unreadable or no candidate
/// survives — the caller then skips with a message rather than failing.
pub fn pick_test_model() -> Option<String> {
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
/// (openai_compat.rs resolve path) closely enough to know whether to run
/// the embeddings leg of the frontend matrix.
pub fn pick_embedding_model() -> Option<String> {
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

pub struct ServeHandle {
    pub child: Child,
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
            // escalation only fires on a genuinely-stuck binary.
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

pub fn start_lamu_serve(binary: &PathBuf, port: u16) -> ServeHandle {
    let child = Command::new(binary)
        .args(["serve", "--port", &port.to_string()])
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn lamu serve");
    ServeHandle { child }
}

pub async fn wait_for_health(client: &reqwest::Client, port: u16, timeout: Duration) -> bool {
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

/// `models_loaded` count from `/health` (openai_compat.rs:233-235), or
/// `None` if the probe fails. The load harness uses this to assert the
/// single-flight contract (exactly one backend per distinct model name)
/// and the "no leaked mark_loading" invariant after a refused eviction.
pub async fn models_loaded(client: &reqwest::Client, port: u16) -> Option<u64> {
    let url = format!("http://127.0.0.1:{}/health", port);
    let r = client.get(&url).send().await.ok()?;
    let j = r.json::<Value>().await.ok()?;
    j.get("models_loaded").and_then(|v| v.as_u64())
}

/// Ok(body) on 2xx; Err(reason) when the status/body looks like host-VRAM
/// contention (caller skips that leg). Panics on any other non-2xx so a
/// real wire-up regression still fails the test. Shared by every surface
/// across spec_e2e.rs / frontend_matrix.rs / load_e2e.rs.
pub async fn body_or_vram_skip(resp: reqwest::Response, surface: &str) -> Result<Value, String> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp
            .json::<Value>()
            .await
            .expect("success body should parse"));
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

/// True when a non-2xx body indicates host-VRAM contention rather than a
/// wire-up regression. For streaming legs where the caller already holds
/// the body text (SSE/NDJSON) and can't reuse `body_or_vram_skip`.
pub fn is_vram_contention(body: &str) -> bool {
    body.contains("vram exhausted")
        || body.contains("requires evicting")
        || body.contains("VRAM")
}
