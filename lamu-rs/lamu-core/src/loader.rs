//! Backend lifecycle loader.
//!
//! Spawns llama-server subprocesses and registers them in the VRAM
//! scheduler. Used by:
//!
//! - `lamu-mcp` (handle_load_model): retains `Box<dyn Backend>` so later
//!   eviction can `unload()` cleanly. Uses `spawn_one` directly.
//! - `lamu-api` (HTTP handlers): drops the trait object after spawn. The
//!   llama-server subprocess survives independently; subsequent requests
//!   proxy via `scheduler.get_loaded(name).port`. Uses `ensure_loaded`.
//!
//! Concurrent `ensure_loaded` calls are serialized via a process-global
//! tokio mutex so two parallel HTTP requests for the same model don't
//! double-spawn.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;

use crate::backends::{make_backend, Backend};
use crate::config::{PORT_MAIN, PORT_SIDECAR};
use crate::health::HealthRegistry;
use crate::scheduler::VramScheduler;
use crate::types::{DevicePlacement, LoadedModel, ModelEntry, ModelState};
use crate::{Error, Result};

/// How long to wait after an auto-evict kill before re-planning, so NVML's
/// used-VRAM reading reflects the freed memory. A fixed heuristic: llama-server
/// releases VRAM on exit within ~1s, but NVML's per-process accounting can lag
/// a beat; 3s is the same settle the MCP eviction path uses. The re-plan is
/// conservative regardless — if usage hasn't dropped it returns VramExhausted
/// rather than over-committing.
const EVICT_SETTLE: std::time::Duration = std::time::Duration::from_secs(3);

fn spawn_gate() -> &'static AsyncMutex<()> {
    static GATE: OnceLock<AsyncMutex<()>> = OnceLock::new();
    GATE.get_or_init(|| AsyncMutex::new(()))
}

/// Spawn a single backend on `port`, pinned to `placement` (ADR 0017
/// P2). Pure primitive — caller is responsible for scheduler
/// bookkeeping. `set_device` runs before `load` so the spawned child's
/// `CUDA_VISIBLE_DEVICES` matches the scheduler's placement; on a
/// single-GPU rig `placement` is `Single(0)` and this is byte-identical
/// to before. Returns the backend trait object plus the subprocess PID.
pub async fn spawn_one(
    entry: &ModelEntry,
    port: u16,
    placement: DevicePlacement,
) -> Result<(Box<dyn Backend>, u32)> {
    let mut backend = make_backend(entry)?;
    backend.set_device(placement);
    let pid = backend.load(entry, port).await?;
    Ok((backend, pid))
}

/// Pick the port for a fresh backend spawn. Excludes the caller's own
/// HTTP listener port (if any) and any port already claimed by a
/// loading/loaded model. Tries PORT_MAIN, PORT_SIDECAR, then 8002..8010.
pub fn pick_backend_port(
    scheduler: &VramScheduler,
    http_serve_port: Option<u16>,
) -> Option<u16> {
    let mut occupied: std::collections::HashSet<u16> = scheduler
        .loaded_models()
        .iter()
        .filter(|lm| matches!(lm.state, ModelState::Loaded | ModelState::Loading))
        .map(|lm| lm.port)
        .filter(|p| *p != 0)
        .collect();
    if let Some(p) = http_serve_port {
        occupied.insert(p);
    }
    for candidate in [
        PORT_MAIN,
        PORT_SIDECAR,
        8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009,
    ] {
        if !occupied.contains(&candidate) {
            return Some(candidate);
        }
    }
    // m10: all candidate ports occupied. Return None so the caller refuses the
    // load cleanly instead of spawning onto an occupied port (bind failure /
    // cross-model port aliasing).
    None
}

/// HTTP-path spawn + register. Idempotent on concurrent same-name calls.
///
/// Refuses to evict — the HTTP path doesn't retain backend handles so it
/// can't `unload()` cleanly. If the planner demands eviction, caller
/// must unload via MCP first.
///
/// On success, the llama-server subprocess is registered in `scheduler`
/// and the `Box<dyn Backend>` is dropped. Subsequent requests proxy via
/// `scheduler.get_loaded(name).unwrap().port`.
pub async fn ensure_loaded(
    name: &str,
    entries: &HashMap<String, ModelEntry>,
    scheduler: &Arc<Mutex<VramScheduler>>,
    health: &Arc<Mutex<HealthRegistry>>,
    http_serve_port: Option<u16>,
) -> Result<LoadedModel> {
    // GPU exclusive-lock gate: refuse while lamu-train (or another exclusive
    // holder) owns the card. This is the single chokepoint for every HTTP spawn
    // path — lamu-api chat / anthropic-stream / ollama-stream / /v1/embeddings
    // all funnel through here, and only non-stream chat_completions checked the
    // lock before. Even an already-loaded model would run inference on the GPU,
    // so gate at the entry. (Test paths use `ensure_loaded_with` directly and
    // are unaffected.)
    crate::scheduler_lock::check_unlocked()?;
    // Clone for the async spawn closure; `scheduler` itself moves into
    // `ensure_loaded_with` below.
    let sched_for_spawn = scheduler.clone();
    ensure_loaded_with(
        name, entries, scheduler, health, http_serve_port,
        move |entry, port| {
            let sched = sched_for_spawn.clone();
            async move {
                // `ensure_loaded_with` records placement via `mark_loading`
                // inside its VRAM gate before invoking this closure, so
                // `placement_of` is normally `Some`. `unwrap_or_default()` →
                // `Single(0)` is the safe fallback (and the single-GPU value).
                // The lock is taken and dropped on this line — the owned
                // `DevicePlacement` outlives the guard, so nothing is held
                // across the `spawn_one` await.
                let placement = sched
                    .lock()
                    .placement_of(&entry.name)
                    .unwrap_or_default();
                spawn_one(&entry, port, placement).await
            }
        },
    ).await
}

/// `ensure_loaded` parameterised on the spawn primitive so tests can
/// inject a `FakeBackend` factory without going through `make_backend`
/// (which requires a real GGUF on disk for `LlamaCppBackend::load`).
///
/// The closure receives the chosen port and an owned registry entry; it
/// returns the spawned backend trait object + PID, same contract as
/// `spawn_one`. Taking `ModelEntry` by value keeps the returned future
/// `'static`, sidestepping closure-lifetime gymnastics.
pub async fn ensure_loaded_with<F, Fut>(
    name: &str,
    entries: &HashMap<String, ModelEntry>,
    scheduler: &Arc<Mutex<VramScheduler>>,
    health: &Arc<Mutex<HealthRegistry>>,
    http_serve_port: Option<u16>,
    spawn: F,
) -> Result<LoadedModel>
where
    F: FnOnce(ModelEntry, u16) -> Fut,
    Fut: std::future::Future<Output = Result<(Box<dyn Backend>, u32)>>,
{
    if let Some(lm) = scheduler.lock().get_loaded(name).cloned() {
        if matches!(lm.state, ModelState::Loaded) {
            return Ok(lm);
        }
    }
    let _gate = spawn_gate().lock().await;
    if let Some(lm) = scheduler.lock().get_loaded(name).cloned() {
        if matches!(lm.state, ModelState::Loaded) {
            return Ok(lm);
        }
    }

    let Some(entry) = entries.get(name).cloned() else {
        return Err(Error::ModelNotFound(name.to_string()));
    };

    // Plan the load. When it needs VRAM reclaimed: refuse by default (ADR
    // 0006 — a shared endpoint must not surprise-kill a model another client
    // uses), or, when LAMU_HTTP_AUTOEVICT is on (single-user desktop opt-in),
    // capture the eviction targets and reclaim them with a VERIFIED kill before
    // reserving. We only ever evict THIS scheduler's own resident models
    // (handle-less HTTP spawns); VRAM held by another process (training / MCP)
    // shows up only via NVML and is never an eviction candidate.
    let evict_targets: Vec<(String, Option<u32>, u16)> = {
        let sched = scheduler.lock();
        let (can, evict) = sched.plan_load(&entry);
        if !can {
            return Err(Error::VramExhausted {
                need_mb: entry.vram_mb,
                have_mb: sched.available_mb(),
            });
        }
        if evict.is_empty() {
            Vec::new()
        } else if !crate::config::http_autoevict() {
            return Err(Error::Config(format!(
                "loading '{}' requires evicting {:?}; HTTP path won't auto-evict — \
                 set LAMU_HTTP_AUTOEVICT=1 to allow it, or unload them via MCP first",
                entry.name, evict
            )));
        } else {
            evict
                .iter()
                .map(|n| {
                    let (pid, port) =
                        sched.get_loaded(n).map(|m| (m.pid, m.port)).unwrap_or((None, 0));
                    (n.clone(), pid, port)
                })
                .collect()
        }
    };

    // Verified eviction. The kill awaits run with NO scheduler lock held; the
    // brief per-target mark_unloaded re-locks but never across an await. The
    // global spawn_gate (held since the top of this fn) serializes every load
    // in this process, so no concurrent ensure_loaded can race this eviction.
    // mark_unloaded fires only AFTER kill_pid_and_verify confirms death — and
    // we refuse the load rather than spawn onto VRAM we couldn't reclaim.
    let mut already_evicted: Vec<&str> = Vec::new();
    for (ev_name, ev_pid, ev_port) in &evict_targets {
        let Some(pid) = ev_pid else {
            // A Loaded model with no recorded pid (e.g. adopted via
            // auto_register) — we cannot VERIFY a kill, so refuse rather than
            // mark it unloaded and over-commit VRAM against a live process.
            return Err(Error::Backend(format!(
                "auto-evict of '{ev_name}': no pid recorded — unload it via MCP first"
            )));
        };
        if let Err(e) = crate::backends::kill_pid_and_verify(*pid, *ev_port).await {
            if !already_evicted.is_empty() {
                tracing::warn!(
                    evicted = ?already_evicted,
                    "loader: auto-evict aborted mid-batch — listed models were already killed before '{}' failed; scheduler freed their VRAM",
                    ev_name
                );
            }
            return Err(Error::Backend(format!("auto-evict of '{ev_name}' failed: {e}")));
        }
        scheduler.lock().mark_unloaded(ev_name);
        HealthRegistry::drop(&mut health.lock(), ev_name);
        already_evicted.push(ev_name.as_str());
        tracing::info!(model = %ev_name, "loader: auto-evicted to make room (LAMU_HTTP_AUTOEVICT)");
    }
    if !evict_targets.is_empty() {
        // Let NVML's used-VRAM reading drop before we re-plan against it.
        tokio::time::sleep(EVICT_SETTLE).await;
    }

    // Re-plan + reserve under the lock. After eviction the load must now fit;
    // if a concurrent load raced in and took the freed room, surface it as
    // VramExhausted rather than spawning into an over-committed device.
    let port = {
        let mut sched = scheduler.lock();
        let (can, evict) = sched.plan_load(&entry);
        if !can || !evict.is_empty() {
            return Err(Error::VramExhausted {
                need_mb: entry.vram_mb,
                have_mb: sched.available_mb(),
            });
        }
        let Some(port) = pick_backend_port(&sched, http_serve_port) else {
            return Err(Error::Backend(
                "no free backend port available (all candidate ports 8000-8009 occupied)".into(),
            ));
        };
        sched.mark_loading(entry.clone());
        port
    };

    // Cold load: holds the process-global single-flight gate for the
    // duration of `spawn` (up to ~90s for a large GGUF). Concurrent
    // `ensure_loaded` calls for OTHER models that fit are serialized behind
    // this — intentional VRAM/port safety on a single card, but otherwise
    // invisible. Log it so a slow first-touch is explainable in the journal.
    tracing::info!(
        model = %entry.name, port, required_vram_mb = entry.vram_mb,
        "loader: cold load starting (single-flight gate held until spawn returns)"
    );

    // Cancellation-safety (B2): if this future is dropped while awaiting
    // `spawn` (e.g. an HTTP client disconnects during the ~90s cold load),
    // nothing past this point runs — so without a guard the `mark_loading`
    // entry would linger forever, reserving `entry.vram_mb` and blocking
    // eviction (it's never evictable while Loading). The guard rolls that
    // back on Drop (cancel OR error) and is disarmed only after a successful
    // confirm_loaded.
    struct LoadRollback<'a> {
        name: String,
        scheduler: &'a Arc<Mutex<VramScheduler>>,
        health: &'a Arc<Mutex<HealthRegistry>>,
        armed: bool,
    }
    impl Drop for LoadRollback<'_> {
        fn drop(&mut self) {
            if self.armed {
                self.scheduler.lock().mark_unloaded(&self.name);
                HealthRegistry::drop(&mut self.health.lock(), &self.name);
            }
        }
    }
    let mut rollback = LoadRollback {
        name: entry.name.clone(),
        scheduler,
        health,
        armed: true,
    };

    let (_backend, pid) = match spawn(entry.clone(), port).await {
        Ok(pair) => pair,
        // The guard's Drop performs the mark_unloaded + health cleanup on this
        // early return, so no manual rollback here.
        Err(e) => return Err(e),
    };

    let vram = {
        let sched = scheduler.lock();
        sched.query_gpu_pids()
            .iter()
            .find(|(p, _)| *p == pid)
            .map(|(_, m)| *m)
            .unwrap_or(entry.vram_mb)
    };
    let lm = {
        let mut sched = scheduler.lock();
        let _ = sched.confirm_loaded(&entry.name, pid, port, vram);
        sched.get_loaded(&entry.name)
            .cloned()
            .ok_or_else(|| Error::ModelNotFound(entry.name.clone()))?
    };
    // Load committed: disarm the rollback so Drop is a no-op.
    rollback.armed = false;
    health.lock().get_or_create(&entry.name).record_success();
    Ok(lm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{Backend, ChatMessage};
    use crate::types::{BackendType, Capability, ModelFormat, ModelStatus};
    use async_trait::async_trait;
    use futures_util::stream::Stream;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Serializes the two tests that read LAMU_HTTP_AUTOEVICT so one's env
    /// mutation can't race the other (process-global env). No other loader
    /// test reaches the eviction branch, so they don't need the lock.
    static EVICT_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn fake_entry(name: &str, vram: u32) -> ModelEntry {
        ModelEntry {
            name: name.to_string(),
            path: "/dev/null".into(),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
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
            modality: crate::types::Modality::Llm,
        }
    }

    /// Backend that never spawns anything. Tracks load() invocations so
    /// single-flight tests can assert exactly-one spawn under concurrency.
    /// The PID returned is the post-incremented load count; tests using a
    /// single FakeBackend instance see distinct PIDs per call.
    struct FakeBackend {
        load_calls: Arc<AtomicU32>,
        port: parking_lot::Mutex<u16>,
        name: parking_lot::Mutex<String>,
        load_should_fail: bool,
    }

    impl FakeBackend {
        fn new(load_calls: Arc<AtomicU32>) -> Self {
            Self {
                load_calls,
                port: parking_lot::Mutex::new(0),
                name: parking_lot::Mutex::new(String::new()),
                load_should_fail: false,
            }
        }
        fn failing(load_calls: Arc<AtomicU32>) -> Self {
            Self {
                load_calls,
                port: parking_lot::Mutex::new(0),
                name: parking_lot::Mutex::new(String::new()),
                load_should_fail: true,
            }
        }
    }

    #[async_trait]
    impl Backend for FakeBackend {
        async fn load(&mut self, entry: &ModelEntry, port: u16) -> Result<u32> {
            let n = self.load_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.load_should_fail {
                return Err(Error::Backend(format!("fake spawn failure for '{}'", entry.name)));
            }
            *self.port.lock() = port;
            *self.name.lock() = entry.name.clone();
            Ok(100_000 + n)  // synthetic, never collides with real PIDs
        }
        async fn unload(&mut self) -> Result<()> { Ok(()) }
        async fn is_healthy(&self) -> bool { true }
        async fn generate(&self, _: Vec<ChatMessage>, _: u32, _: f32) -> Result<String> {
            Ok(String::new())
        }
        async fn stream(
            &self, _: Vec<ChatMessage>, _: u32, _: f32,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
            use futures_util::stream;
            Ok(Box::pin(stream::empty()))
        }
        fn port(&self) -> u16 { *self.port.lock() }
        fn model_name(&self) -> &str { "fake" }
    }

    fn one_entry_registry(entry: ModelEntry) -> HashMap<String, ModelEntry> {
        let mut m = HashMap::new();
        m.insert(entry.name.clone(), entry);
        m
    }

    fn fresh_state(total_vram_mb: u32) -> (Arc<Mutex<VramScheduler>>, Arc<Mutex<HealthRegistry>>) {
        let mut sched = VramScheduler::new();
        sched.set_total_mb_for_tests(total_vram_mb);
        (
            Arc::new(Mutex::new(sched)),
            Arc::new(Mutex::new(HealthRegistry::new())),
        )
    }

    #[test]
    fn pick_backend_port_skips_http_listener() {
        let sched = VramScheduler::new();
        let port = pick_backend_port(&sched, Some(PORT_MAIN)).expect("a free port exists");
        assert_ne!(port, PORT_MAIN);
    }

    #[test]
    fn pick_backend_port_skips_loaded() {
        let mut sched = VramScheduler::new();
        let e = fake_entry("a", 100);
        sched.register_loaded(e, None, PORT_MAIN, 100);
        let port = pick_backend_port(&sched, None).expect("a free port exists");
        assert_ne!(port, PORT_MAIN);
    }

    #[tokio::test]
    async fn ensure_loaded_idempotent() {
        let entry = fake_entry("m", 1000);
        let entries = one_entry_registry(entry.clone());
        let (sched, health) = fresh_state(24_000);
        let calls = Arc::new(AtomicU32::new(0));
        let spawn = |e: ModelEntry, port: u16| {
            let calls = calls.clone();
            let e = e.clone();
            Box::pin(async move {
                let mut b = FakeBackend::new(calls);
                let pid = b.load(&e, port).await?;
                Ok::<_, Error>((Box::new(b) as Box<dyn Backend>, pid))
            })
        };
        let lm1 = ensure_loaded_with(
            "m", &entries, &sched, &health, None, spawn.clone(),
        ).await.unwrap();
        let lm2 = ensure_loaded_with(
            "m", &entries, &sched, &health, None, spawn,
        ).await.unwrap();
        assert_eq!(lm1.entry.name, "m");
        assert_eq!(lm1.port, lm2.port);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "second call must hit fast path");
    }

    #[tokio::test]
    async fn ensure_loaded_single_flight() {
        let entry = fake_entry("m", 1000);
        let entries = one_entry_registry(entry);
        let (sched, health) = fresh_state(24_000);
        let calls = Arc::new(AtomicU32::new(0));
        // 10 parallel calls — gate must serialize them; second-through-tenth
        // hit the inside-gate fast path. Total backend loads must equal 1.
        let mut handles = Vec::new();
        for _ in 0..10 {
            let entries = entries.clone();
            let sched = sched.clone();
            let health = health.clone();
            let calls = calls.clone();
            handles.push(tokio::spawn(async move {
                let spawn = |e: ModelEntry, port: u16| {
                    let calls = calls.clone();
                    let e = e.clone();
                    Box::pin(async move {
                        let mut b = FakeBackend::new(calls);
                        let pid = b.load(&e, port).await?;
                        Ok::<_, Error>((Box::new(b) as Box<dyn Backend>, pid))
                    })
                };
                ensure_loaded_with("m", &entries, &sched, &health, None, spawn).await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1,
            "single-flight gate must collapse N concurrent calls into 1 spawn");
    }

    #[tokio::test]
    async fn ensure_loaded_refuses_vram_exhausted() {
        let entry = fake_entry("huge", 30_000);  // exceeds total
        let entries = one_entry_registry(entry);
        let (sched, health) = fresh_state(24_000);
        let calls = Arc::new(AtomicU32::new(0));
        let spawn = |e: ModelEntry, port: u16| {
            let calls = calls.clone();
            let e = e.clone();
            Box::pin(async move {
                let mut b = FakeBackend::new(calls);
                let pid = b.load(&e, port).await?;
                Ok::<_, Error>((Box::new(b) as Box<dyn Backend>, pid))
            })
        };
        let err = ensure_loaded_with(
            "huge", &entries, &sched, &health, None, spawn,
        ).await.unwrap_err();
        assert!(matches!(err, Error::VramExhausted { .. }), "got: {err:?}");
        assert_eq!(calls.load(Ordering::SeqCst), 0, "spawn must not run when VRAM exhausted");
    }

    #[tokio::test]
    async fn ensure_loaded_refuses_eviction_required() {
        let _g = EVICT_ENV_LOCK.lock().unwrap();
        // SAFETY: lock held; the only other env reader (the autoevict test)
        // is serialized behind the same lock.
        unsafe { std::env::remove_var("LAMU_HTTP_AUTOEVICT"); }
        let occupier = fake_entry("occupier", 20_000);
        let new_entry = fake_entry("new", 10_000);  // forces evicting 'occupier'
        let mut entries = HashMap::new();
        entries.insert(occupier.name.clone(), occupier.clone());
        entries.insert(new_entry.name.clone(), new_entry);
        let (sched, health) = fresh_state(24_000);
        sched.lock().register_loaded(occupier, None, PORT_MAIN, 20_000);
        let calls = Arc::new(AtomicU32::new(0));
        let spawn = |e: ModelEntry, port: u16| {
            let calls = calls.clone();
            let e = e.clone();
            Box::pin(async move {
                let mut b = FakeBackend::new(calls);
                let pid = b.load(&e, port).await?;
                Ok::<_, Error>((Box::new(b) as Box<dyn Backend>, pid))
            })
        };
        let err = ensure_loaded_with(
            "new", &entries, &sched, &health, None, spawn,
        ).await.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("evict"), "error must mention eviction: {msg}");
        assert!(msg.contains("MCP"), "error must direct user to MCP: {msg}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn ensure_loaded_autoevicts_when_enabled() {
        let _g = EVICT_ENV_LOCK.lock().unwrap();
        // SAFETY: lock held; serialized against the refuses test above.
        unsafe { std::env::set_var("LAMU_HTTP_AUTOEVICT", "1"); }

        // Occupier on a CLOSED port → kill_pid_and_verify sees the port silent
        // and confirms death immediately (no real process is signalled).
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead_port = listener.local_addr().unwrap().port();
        drop(listener);

        let occupier = fake_entry("occupier", 20_000);
        let new_entry = fake_entry("new", 10_000); // forces evicting 'occupier'
        let mut entries = HashMap::new();
        entries.insert(occupier.name.clone(), occupier.clone());
        entries.insert(new_entry.name.clone(), new_entry);
        let (sched, health) = fresh_state(24_000);
        sched.lock().register_loaded(occupier, Some(987654), dead_port, 20_000);

        let calls = Arc::new(AtomicU32::new(0));
        let spawn = |e: ModelEntry, port: u16| {
            let calls = calls.clone();
            let e = e.clone();
            Box::pin(async move {
                let mut b = FakeBackend::new(calls);
                let pid = b.load(&e, port).await?;
                Ok::<_, Error>((Box::new(b) as Box<dyn Backend>, pid))
            })
        };
        let lm = ensure_loaded_with("new", &entries, &sched, &health, None, spawn)
            .await
            .expect("autoevict should reclaim VRAM and load 'new'");
        unsafe { std::env::remove_var("LAMU_HTTP_AUTOEVICT"); }

        assert_eq!(lm.entry.name, "new");
        assert!(sched.lock().is_loaded("new"), "new model loaded after auto-evict");
        assert!(!sched.lock().is_loaded("occupier"), "occupier was auto-evicted");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "exactly one spawn — the new model");
    }

    #[tokio::test]
    async fn ensure_loaded_skips_http_port() {
        let entry = fake_entry("m", 1000);
        let entries = one_entry_registry(entry);
        let (sched, health) = fresh_state(24_000);
        let calls = Arc::new(AtomicU32::new(0));
        let spawn = |e: ModelEntry, port: u16| {
            let calls = calls.clone();
            let e = e.clone();
            Box::pin(async move {
                let mut b = FakeBackend::new(calls);
                let pid = b.load(&e, port).await?;
                Ok::<_, Error>((Box::new(b) as Box<dyn Backend>, pid))
            })
        };
        let lm = ensure_loaded_with(
            "m", &entries, &sched, &health, Some(PORT_MAIN), spawn,
        ).await.unwrap();
        assert_ne!(lm.port, PORT_MAIN, "must not collide with the HTTP listener's port");
    }

    #[tokio::test]
    async fn ensure_loaded_rolls_back_on_spawn_failure() {
        let entry = fake_entry("m", 1000);
        let entries = one_entry_registry(entry);
        let (sched, health) = fresh_state(24_000);
        let calls = Arc::new(AtomicU32::new(0));
        let spawn = |e: ModelEntry, _port: u16| {
            let calls = calls.clone();
            let e = e.clone();
            Box::pin(async move {
                let mut b = FakeBackend::failing(calls);
                let pid = b.load(&e, 0).await?;
                Ok::<_, Error>((Box::new(b) as Box<dyn Backend>, pid))
            })
        };
        let err = ensure_loaded_with(
            "m", &entries, &sched, &health, None, spawn,
        ).await.unwrap_err();
        assert!(matches!(err, Error::Backend(_)));
        assert!(!sched.lock().is_loaded("m"),
            "scheduler must NOT carry a phantom Loading entry after spawn failure");
    }

    #[tokio::test]
    async fn ensure_loaded_rolls_back_on_cancellation() {
        // B2: dropping the load future mid-spawn (e.g. an HTTP client
        // disconnects during the ~90s cold load) must NOT leave a stuck
        // Loading entry reserving VRAM + blocking eviction forever.
        let entry = fake_entry("m", 1000);
        let entries = one_entry_registry(entry);
        let (sched, health) = fresh_state(24_000);
        let spawn = |_e: ModelEntry, _port: u16| {
            Box::pin(async move {
                // A cold load that never completes — the caller cancels first.
                futures_util::future::pending::<Result<(Box<dyn Backend>, u32)>>().await
            })
        };
        let fut = ensure_loaded_with("m", &entries, &sched, &health, None, spawn);
        // Drive to the spawn await; the timeout firing drops `fut` (cancellation).
        let _ = tokio::time::timeout(std::time::Duration::from_millis(50), fut).await;
        assert!(!sched.lock().is_loaded("m"),
            "a cancelled cold load must roll back the Loading entry (B2)");
    }

    // ── Property tests ────────────────────────────────────────────────

    proptest::proptest! {
        // For any combination of (already-loaded ports, http_serve_port),
        // the picked port must not collide with anything occupied or with
        // the listener's own port — as long as the candidate range has
        // any free slot. With ≤10 candidate ports and ≤6 occupied
        // + http_serve, there's always slack.
        #[test]
        fn pick_backend_port_never_collides(
            // 0..=6 distinct ports from the candidate set
            occupied_count in 0usize..=6,
            http_port_choice in 0u8..=3,
        ) {
            let candidates = [
                PORT_MAIN, PORT_SIDECAR, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009,
            ];
            let occupied: std::collections::HashSet<u16> =
                candidates.iter().take(occupied_count).copied().collect();
            let mut sched = VramScheduler::new();
            for (i, p) in occupied.iter().enumerate() {
                let e = fake_entry(&format!("occ{i}"), 1);
                sched.register_loaded(e, None, *p, 1);
            }
            let http_port = match http_port_choice {
                0 => None,
                1 => Some(PORT_MAIN),
                2 => Some(PORT_SIDECAR),
                _ => Some(8002),
            };
            let picked = pick_backend_port(&sched, http_port)
                .expect("≤7 occupied of 10 candidates → a free port always exists");
            proptest::prop_assert!(!occupied.contains(&picked),
                "picked {picked} collides with occupied {:?}", occupied);
            if let Some(hp) = http_port {
                proptest::prop_assert_ne!(picked, hp,
                    "picked {} collides with http_serve_port {}", picked, hp);
            }
        }
    }
}
