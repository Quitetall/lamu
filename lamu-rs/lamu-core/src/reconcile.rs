//! Liveness reconciliation — the load-bearing guarantee that the
//! scheduler's `loaded` state can never silently drift from process/GPU
//! ground truth.
//!
//! The scheduler is in-memory bookkeeping: `confirm_loaded` flips a model
//! to `Loaded`, `mark_unloaded` removes it. Nothing in that path observes
//! the real world, so a backend that dies on its own (CUDA OOM, external
//! SIGKILL, host crash-restart of llama-server, a botched preload) leaves
//! the map saying `loaded:true` forever — `/health` reports
//! `models_loaded:1`, `/v1/models` reports `loaded:true`, while the GPU is
//! empty and no server answers the port. That phantom state is exactly
//! what this loop exists to make impossible.
//!
//! Invariant enforced every tick: **a model is `Loaded` IFF its backend is
//! observably alive** — either its PID is holding VRAM (NVML compute-process
//! list) OR its port answers `/health`. A model that satisfies NEITHER is
//! dead; we `mark_unloaded` it and free its VRAM accounting. We keep a model
//! that holds VRAM even when `/health` momentarily times out (a busy server
//! under load must not be falsely evicted) and keep one that answers `/health`
//! even if NVML lags — alive on either signal, dead only on both.
//!
//! We never KILL here: a process that is already gone needs no kill, and a
//! legitimately-live backend we lost the handle to is still serving
//! correctly. Killing live orphans is the unload path's job (and only on
//! explicit request). Reconciliation's sole mandate is: make the reported
//! state true.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::health::HealthRegistry;
use crate::scheduler::VramScheduler;
use crate::types::ModelState;

/// Default reconcile cadence. Short enough that a dead backend is reflected
/// in `/health` and `/v1/models` within a few seconds; long enough that the
/// per-tick probe cost (one cheap `/health` GET per loaded model) is noise.
pub const DEFAULT_INTERVAL_SECS: u64 = 7;

/// Per-request timeout for the liveness `/health` probe. The AppState client
/// carries a 300s generation timeout; reconciliation must never inherit that
/// — a dead port should fail fast, not stall the whole loop.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// A model we found dead this tick, with the reason for the audit log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evicted {
    pub name: String,
    pub reason: &'static str,
}

/// Probe `127.0.0.1:port/health` with a tight timeout. True on any 2xx —
/// a llama-server / python-server that is up answers here. Any transport
/// error, timeout, or non-2xx is "not answering".
async fn port_healthy(client: &reqwest::Client, port: u16) -> bool {
    if port == 0 {
        return false;
    }
    let url = format!("http://127.0.0.1:{port}/health");
    match client.get(&url).timeout(PROBE_TIMEOUT).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// One reconciliation pass. Returns the models evicted this tick (empty when
/// everything checks out). Pure of any background machinery so it can be
/// unit-tested directly.
///
/// Lock discipline: snapshot under the scheduler lock, run all async probes
/// with NO lock held (the codebase forbids holding the parking_lot guard
/// across an await), then re-acquire to evict. The re-acquire re-validates
/// that the dead model is still the SAME instance (pid+port unchanged) so a
/// model legitimately reloaded between snapshot and eviction is never
/// clobbered.
pub async fn reconcile_once(
    scheduler: &Arc<Mutex<VramScheduler>>,
    health: &Arc<Mutex<HealthRegistry>>,
    client: &reqwest::Client,
) -> Vec<Evicted> {
    // 1. Snapshot loaded models + the current NVML compute-PID set. No await.
    struct Probe {
        name: String,
        pid: Option<u32>,
        port: u16,
    }
    let (probes, gpu_pids): (Vec<Probe>, HashSet<u32>) = {
        let s = scheduler.lock();
        let gpu_pids: HashSet<u32> = s.query_gpu_pids().into_iter().map(|(p, _)| p).collect();
        let probes = s
            .loaded_models()
            .iter()
            .filter(|m| matches!(m.state, ModelState::Loaded))
            .map(|m| Probe {
                name: m.entry.name.clone(),
                pid: m.pid,
                port: m.port,
            })
            .collect();
        (probes, gpu_pids)
    };

    // 2. Probe liveness with no lock held. Alive iff pid holds VRAM OR port
    //    answers /health; dead only when BOTH signals are absent. A pid that
    //    already holds VRAM needs no network probe (alive). The remaining
    //    /health probes run CONCURRENTLY — serial 2s timeouts would otherwise
    //    sum to 2*N seconds and blow past the reconcile interval when several
    //    backends are down at once.
    let dead: Vec<(String, Option<u32>, u16)> = {
        let needing_probe = probes
            .iter()
            .filter(|p| !p.pid.is_some_and(|pid| gpu_pids.contains(&pid)));
        let futs = needing_probe.map(|p| async move {
            if port_healthy(client, p.port).await {
                None
            } else {
                Some((p.name.clone(), p.pid, p.port))
            }
        });
        futures_util::future::join_all(futs)
            .await
            .into_iter()
            .flatten()
            .collect()
    };
    if dead.is_empty() {
        return Vec::new();
    }

    // 3. Evict, re-validating instance identity. The scheduler and health
    //    locks are taken SEQUENTIALLY (never nested) so this can't form an
    //    ABBA cycle with any path that touches health. Only evict a model
    //    still Loaded AND still carrying the exact (pid, port) we probed —
    //    otherwise a concurrent reload re-confirmed it and we must not touch
    //    the fresh instance.
    let mut evicted_names: Vec<String> = Vec::new();
    {
        let mut s = scheduler.lock();
        for (name, probed_pid, probed_port) in &dead {
            let still_same = s.get_loaded(name).is_some_and(|m| {
                matches!(m.state, ModelState::Loaded)
                    && m.pid == *probed_pid
                    && m.port == *probed_port
            });
            if still_same {
                s.mark_unloaded(name);
                evicted_names.push(name.clone());
            }
        }
    }
    if !evicted_names.is_empty() {
        let mut h = health.lock();
        for name in &evicted_names {
            HealthRegistry::drop(&mut h, name);
        }
    }
    evicted_names
        .into_iter()
        .map(|name| Evicted {
            name,
            reason: "backend dead: pid holds no VRAM and port did not answer /health",
        })
        .collect()
}

/// Run reconciliation forever on `interval`. Spawned once at `lamu serve`
/// startup. Every eviction is logged at WARN with the reason so an operator
/// has an audit trail for "why did my model disappear from /v1/models".
pub async fn run_reconcile_loop(
    scheduler: Arc<Mutex<VramScheduler>>,
    health: Arc<Mutex<HealthRegistry>>,
    client: reqwest::Client,
    interval: Duration,
) {
    tracing::info!(
        interval_secs = interval.as_secs(),
        "reconcile: liveness loop started — loaded-state will track process/GPU ground truth"
    );
    loop {
        tokio::time::sleep(interval).await;
        let evicted = reconcile_once(&scheduler, &health, &client).await;
        for e in evicted {
            tracing::warn!(
                model = %e.name,
                reason = e.reason,
                "reconcile: evicted phantom-loaded model (state said loaded, backend was dead)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        BackendType, Capability, Modality, ModelEntry, ModelFormat, ModelStatus,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn entry(name: &str, vram: u32) -> ModelEntry {
        ModelEntry {
            name: name.to_string(),
            path: "/dev/null".into(),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
            backend_kind: None,
            arch: "qwen35".to_string(),
            params_b: 1.0,
            quant: "Q4_K_M".to_string(),
            vram_mb: vram,
            context_max: 4096,
            capabilities: vec![Capability::Chat],
            reasoning_marker: None,
            speculative: None,
            sampling: None,
            pinned: false,
            main: false,
            notes: String::new(),
            status: ModelStatus::Unspecified,
            modality: Modality::Llm,
        }
    }

    fn state() -> (Arc<Mutex<VramScheduler>>, Arc<Mutex<HealthRegistry>>, reqwest::Client) {
        let mut s = VramScheduler::new();
        s.set_total_mb_for_tests(24_000);
        (
            Arc::new(Mutex::new(s)),
            Arc::new(Mutex::new(HealthRegistry::new())),
            reqwest::Client::builder().build().unwrap(),
        )
    }

    /// A minimal one-shot HTTP server that answers ANY request with 200.
    /// Stands in for a live llama-server `/health`. Returns the bound port;
    /// the task serves a single connection then exits.
    async fn spawn_ok_server() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                    .await;
                let _ = sock.flush().await;
            }
        });
        port
    }

    #[tokio::test]
    async fn evicts_model_with_dead_port_and_no_gpu_pid() {
        // The observed bug: loaded:true, no NVML pid (NVML null in tests),
        // port not listening. Must be evicted so /health stops lying.
        let (sched, health, client) = state();
        // Port 1 is never an open llama-server in CI; pid present but NVML
        // (nulled by set_total_mb_for_tests) reports no compute processes.
        sched.lock().register_loaded(entry("ghost", 16_000), Some(424242), 1, 16_000);
        assert!(sched.lock().is_loaded("ghost"));

        let evicted = reconcile_once(&sched, &health, &client).await;

        assert_eq!(evicted.len(), 1, "dead-port model must be evicted");
        assert_eq!(evicted[0].name, "ghost");
        assert!(!sched.lock().is_loaded("ghost"), "scheduler must reflect ground truth");
    }

    #[tokio::test]
    async fn keeps_model_whose_port_answers_health() {
        // A live backend (port answers /health) must never be evicted even
        // though NVML is null in tests (no pid-on-gpu signal available).
        let (sched, health, client) = state();
        let port = spawn_ok_server().await;
        sched.lock().register_loaded(entry("live", 16_000), Some(999), port, 16_000);

        let evicted = reconcile_once(&sched, &health, &client).await;

        assert!(evicted.is_empty(), "a /health-answering backend is alive: {evicted:?}");
        assert!(sched.lock().is_loaded("live"));
    }

    #[tokio::test]
    async fn keeps_live_instance_after_reload() {
        // A model re-registered with a live port answers /health and is kept.
        // (The true mid-flight TOCTOU race — found dead during the probe, then
        // reloaded with a new pid+port before the eviction lock — is guarded by
        // the pid+port identity re-check in step 3; it can't be exercised
        // deterministically without splitting reconcile_once, so the e2e covers
        // it. This test pins the simpler "live instance is never evicted".)
        let (sched, health, client) = state();
        let live_port = spawn_ok_server().await;
        sched.lock().register_loaded(entry("flap", 16_000), Some(222), live_port, 16_000);

        let evicted = reconcile_once(&sched, &health, &client).await;
        assert!(evicted.is_empty(), "live instance must not be clobbered: {evicted:?}");
        assert!(sched.lock().is_loaded("flap"));
    }

    #[tokio::test]
    async fn no_loaded_models_is_noop() {
        let (sched, health, client) = state();
        let evicted = reconcile_once(&sched, &health, &client).await;
        assert!(evicted.is_empty());
    }
}
