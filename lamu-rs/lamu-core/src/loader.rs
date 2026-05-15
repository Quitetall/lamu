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

    let (_backend, pid) = match spawn_one(&entry, port).await {
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
    use crate::types::{BackendType, Capability, ModelFormat, ModelStatus};

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
}
