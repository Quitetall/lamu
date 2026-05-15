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
use crate::types::{LoadedModel, ModelEntry, ModelState};
use crate::{Error, Result};

fn spawn_gate() -> &'static AsyncMutex<()> {
    static GATE: OnceLock<AsyncMutex<()>> = OnceLock::new();
    GATE.get_or_init(|| AsyncMutex::new(()))
}

/// Spawn a single backend on `port`. Pure primitive — caller is
/// responsible for scheduler bookkeeping. Returns the backend trait
/// object plus the spawned subprocess PID.
pub async fn spawn_one(entry: &ModelEntry, port: u16) -> Result<(Box<dyn Backend>, u32)> {
    let mut backend = make_backend(entry)?;
    let pid = backend.load(entry, port).await?;
    Ok((backend, pid))
}

/// Pick the port for a fresh backend spawn. Excludes the caller's own
/// HTTP listener port (if any) and any port already claimed by a
/// loading/loaded model. Tries PORT_MAIN, PORT_SIDECAR, then 8002..8010.
pub fn pick_backend_port(
    scheduler: &VramScheduler,
    http_serve_port: Option<u16>,
) -> u16 {
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
            return candidate;
        }
    }
    // All slots taken — caller will likely fail to bind, but return
    // something deterministic for the error path rather than panicking.
    PORT_SIDECAR
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
    ensure_loaded_with(
        name, entries, scheduler, health, http_serve_port,
        |entry, port| async move { spawn_one(&entry, port).await },
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

    let port = {
        let mut sched = scheduler.lock();
        let (can, evict) = sched.plan_load(&entry);
        if !can {
            return Err(Error::VramExhausted {
                need_mb: entry.vram_mb,
                have_mb: sched.available_mb(),
            });
        }
        if !evict.is_empty() {
            return Err(Error::Config(format!(
                "loading '{}' requires evicting {:?}; HTTP path won't auto-evict — \
                 unload them via MCP first",
                entry.name, evict
            )));
        }
        let port = pick_backend_port(&sched, http_serve_port);
        sched.mark_loading(entry.clone());
        port
    };

    let (_backend, pid) = match spawn(entry.clone(), port).await {
        Ok(pair) => pair,
        Err(e) => {
            scheduler.lock().mark_unloaded(&entry.name);
            HealthRegistry::drop(&mut health.lock(), &entry.name);
            return Err(e);
        }
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
            pinned: false,
            main: false,
            notes: String::new(),
            status: ModelStatus::Unspecified,
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
        let port = pick_backend_port(&sched, Some(PORT_MAIN));
        assert_ne!(port, PORT_MAIN);
    }

    #[test]
    fn pick_backend_port_skips_loaded() {
        let mut sched = VramScheduler::new();
        let e = fake_entry("a", 100);
        sched.register_loaded(e, None, PORT_MAIN, 100);
        let port = pick_backend_port(&sched, None);
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
            let picked = pick_backend_port(&sched, http_port);
            proptest::prop_assert!(!occupied.contains(&picked),
                "picked {picked} collides with occupied {:?}", occupied);
            if let Some(hp) = http_port {
                proptest::prop_assert_ne!(picked, hp,
                    "picked {} collides with http_serve_port {}", picked, hp);
            }
        }
    }
}
