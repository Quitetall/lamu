//! Repeatable perf benchmark (SCALE-TEST P2).
//!
//! Two ignore-gated harnesses, both emitting req/s + p50/p99:
//!
//!   1. `bench_read_path_oneshot` — in-process axum oneshot over the read-path
//!      endpoints (/health, /v1/models, /metrics). No real model, no port, no
//!      `lamu` binary needed → runs anywhere, deterministic, the regression
//!      tripwire for routing/lock/serialization overhead. Reuses the exact
//!      AppState fixture shape from http.rs (make_state) so it measures the
//!      same stack the unit tests cover.
//!
//!   2. `bench_concurrent_read_path` — the same read-path under N concurrent
//!      tasks (mirrors http.rs:168 concurrent_requests_no_deadlock) to measure
//!      throughput + tail latency under the shared parking_lot lock contention
//!      that the multi-user track (ADR 0018) will stress.
//!
//! A plain tokio timer, NOT criterion (see SPEC decision note). Tunable via
//! env: LAMU_BENCH_ITERS (default 2000), LAMU_BENCH_CONCURRENCY (default 64).
//!
//! Run:
//! ```bash
//! cargo test --release --test perf_bench -- --ignored --nocapture
//! # or tune:
//! LAMU_BENCH_ITERS=10000 LAMU_BENCH_CONCURRENCY=128 \
//!   cargo test --release --test perf_bench -- --ignored --nocapture
//! ```
//! `--release` matters: the read path is cheap enough that debug-build
//! overhead dominates and the numbers are meaningless without it.

use axum::body::Body;
use axum::http::Request;
use lamu_api::metrics::LamuMetrics;
use lamu_api::openai_compat::{build_app, AppState, AuthMode};
use lamu_core::health::HealthRegistry;
use lamu_core::router::Router;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{BackendType, Capability, ModelEntry, ModelFormat};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tower::util::ServiceExt;

// ── fixture (copy of http.rs sample_entry/make_state; kept local so the two
//    test binaries stay independent) ────────────────────────────────────────

fn sample_entry(name: &str) -> ModelEntry {
    ModelEntry {
        name: name.to_string(),
        path: PathBuf::from(format!("/tmp/{name}.gguf")),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp,
        arch: "qwen35".into(),
        params_b: 27.0,
        quant: "Q5_K_M".into(),
        vram_mb: 18000,
        context_max: 131072,
        capabilities: vec![Capability::Chat, Capability::Code],
        reasoning_marker: None,
        speculative: None,
        sampling: None,
        pinned: false,
        main: false,
        notes: String::new(),
        status: lamu_core::types::ModelStatus::default(),
        modality: lamu_core::types::Modality::Llm,
    }
}

fn make_state() -> AppState {
    let entries = vec![sample_entry("qwen35-27b"), sample_entry("qwen35-0.8b")];
    let entries_map: HashMap<String, ModelEntry> =
        entries.iter().map(|e| (e.name.clone(), e.clone())).collect();
    let scheduler = VramScheduler::new();
    let router = Router::new(&scheduler, entries.clone());
    let client = reqwest::Client::new();
    AppState {
        scheduler: Arc::new(Mutex::new(scheduler)),
        router: Arc::new(Mutex::new(router)),
        entries: Arc::new(entries_map),
        client,
        health: Arc::new(Mutex::new(HealthRegistry::new())),
        metrics: Arc::new(LamuMetrics::new().unwrap()),
        http_port: 8020,
        auth: Arc::new(AuthMode::Off),
        quota: Arc::new(lamu_api::quota::QuotaManager::new()),
        priority_queue: None, // P3 OFF — keep the perf baseline byte-identical
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// p-quantile (nearest-rank) over already-collected latency samples.
fn pctl(sorted_us: &[u128], q: f64) -> u128 {
    if sorted_us.is_empty() {
        return 0;
    }
    let rank = (q * (sorted_us.len() as f64 - 1.0)).round() as usize;
    sorted_us[rank.min(sorted_us.len() - 1)]
}

fn report(label: &str, lat_us: &mut Vec<u128>, wall: std::time::Duration) {
    lat_us.sort_unstable();
    let n = lat_us.len();
    let rps = n as f64 / wall.as_secs_f64();
    let p50 = pctl(lat_us, 0.50) as f64 / 1000.0;
    let p99 = pctl(lat_us, 0.99) as f64 / 1000.0;
    let mean = lat_us.iter().sum::<u128>() as f64 / n as f64 / 1000.0;
    eprintln!(
        "[{label}] n={n} wall={:.3}s req/s={rps:.0} \
         lat_ms(mean={mean:.3} p50={p50:.3} p99={p99:.3})",
        wall.as_secs_f64()
    );
}

const READ_PATHS: [&str; 3] = ["/health", "/v1/models", "/metrics"];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "perf benchmark; run with --release ... -- --ignored --nocapture"]
async fn bench_read_path_oneshot() {
    let iters = env_usize("LAMU_BENCH_ITERS", 2000);
    let app = build_app(make_state());

    // warmup — first oneshot pays lazy-init costs we don't want in the sample.
    for p in READ_PATHS {
        let _ = app
            .clone()
            .oneshot(Request::builder().uri(p).body(Body::empty()).unwrap())
            .await
            .unwrap();
    }

    let mut lat_us = Vec::with_capacity(iters);
    let start = Instant::now();
    for i in 0..iters {
        let uri = READ_PATHS[i % READ_PATHS.len()];
        let t0 = Instant::now();
        let resp = app
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        // Drain the body so we time the full read path, not just headers.
        let _ = axum::body::to_bytes(resp.into_body(), 256 * 1024)
            .await
            .unwrap();
        lat_us.push(t0.elapsed().as_micros());
    }
    let wall = start.elapsed();
    report("read_path_oneshot", &mut lat_us, wall);
    // sanity floor — if this regresses below ~1k req/s the read path grew a real
    // bottleneck. The floor is calibrated for --release; debug-build overhead can
    // legitimately miss it (see file header), so the assert is release-only. In
    // debug we only warn so the harness still works as a measurement tool.
    let rps = iters as f64 / wall.as_secs_f64();
    if cfg!(debug_assertions) {
        eprintln!("[read_path_oneshot] debug build — skipping {rps:.0} req/s floor assert (run --release to gate)");
    } else {
        assert!(rps > 1000.0, "read path throughput collapsed: {rps:.0} req/s (run with --release)");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "perf benchmark; run with --release ... -- --ignored --nocapture"]
async fn bench_concurrent_read_path() {
    let total = env_usize("LAMU_BENCH_ITERS", 2000);
    let conc = env_usize("LAMU_BENCH_CONCURRENCY", 64);
    let app = build_app(make_state());

    for p in READ_PATHS {
        let _ = app
            .clone()
            .oneshot(Request::builder().uri(p).body(Body::empty()).unwrap())
            .await
            .unwrap();
    }

    // Spawn `total` requests with at most `conc` in flight via a semaphore.
    let sem = Arc::new(tokio::sync::Semaphore::new(conc));
    let lat = Arc::new(Mutex::new(Vec::<u128>::with_capacity(total)));
    let start = Instant::now();
    let mut handles = Vec::with_capacity(total);
    for i in 0..total {
        let app = app.clone();
        let sem = sem.clone();
        let lat = lat.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let uri = READ_PATHS[i % READ_PATHS.len()];
            let t0 = Instant::now();
            let resp = app
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            let _ = axum::body::to_bytes(resp.into_body(), 256 * 1024)
                .await
                .unwrap();
            lat.lock().push(t0.elapsed().as_micros());
        }));
    }
    // Check every task: join_all returns Vec<Result<_, JoinError>> and a
    // swallowed panic (e.g. a failed oneshot .unwrap()) would otherwise let the
    // bench "pass" with fewer-than-`total` samples and bogus throughput.
    for r in futures_util::future::join_all(handles).await {
        r.expect("a concurrent bench task panicked");
    }
    let wall = start.elapsed();
    let mut lat_us = Arc::try_unwrap(lat).unwrap().into_inner();
    report(
        &format!("concurrent_read_path[conc={conc}]"),
        &mut lat_us,
        wall,
    );
    // Same release-only floor as the single-threaded bench — guards against a
    // concurrency/lock-contention regression going unnoticed.
    let rps = total as f64 / wall.as_secs_f64();
    if cfg!(debug_assertions) {
        eprintln!("[concurrent_read_path] debug build — skipping {rps:.0} req/s floor assert (run --release to gate)");
    } else {
        assert!(rps > 1000.0, "concurrent read path throughput collapsed: {rps:.0} req/s (run with --release)");
    }
}
