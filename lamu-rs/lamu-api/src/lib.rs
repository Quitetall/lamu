//! lamu-api — OpenAI-compatible HTTP layer.
//! Direct port of `lamu/api/openai_compat.py`.

pub mod auth;
pub mod keys;
pub mod memory_api;
pub mod metrics;
pub mod openai_compat;
pub mod quota;

use lamu_core::config::registry_path;
use std::net::SocketAddr;
use std::path::PathBuf;

pub async fn serve(port: u16) -> anyhow::Result<()> {
    let pidfile = PidFile::acquire(port)?;
    let state = openai_compat::build_state(&registry_path(), port)?;
    openai_compat::auto_register(&state).await;
    spawn_main_preload(state.clone());
    spawn_reconcile_loop(&state);
    // Resolve auth once: the value the middleware will enforce (in state.auth)
    // is the same one the off-loopback gate checks below. KeyStore counts as
    // "configured" ONLY with ≥1 active key — an empty store off-loopback must
    // still hard-fail like no-token (ADR 0018).
    let auth_configured = match state.auth.as_ref() {
        openai_compat::AuthMode::Off => false,
        openai_compat::AuthMode::StaticToken(_) => true,
        openai_compat::AuthMode::KeyStore(ks) => ks.has_active_key(),
    };
    let app = openai_compat::build_app(state);

    // Default to LOCALHOST — the OpenAI-compat API is unauthenticated, so it
    // must not be reachable off-host unless the operator explicitly opts in
    // via LAMU_BIND_HOST (same env the backends use), e.g. =0.0.0.0. Binding
    // 0.0.0.0 by default exposed an unauthenticated API from day one.
    let host = std::env::var("LAMU_BIND_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let addr: SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
        anyhow::anyhow!("LAMU_BIND_HOST='{host}' is not a valid IP (need e.g. 127.0.0.1 or 0.0.0.0): {e}")
    })?;
    // Off-loopback bind is only allowed with a token (ADR 0012) — otherwise the
    // unauthenticated API would be reachable on the LAN. LAMU_ALLOW_INSECURE=1
    // is the explicit, loud escape hatch.
    if !addr.ip().is_loopback() && !auth_configured {
        if std::env::var("LAMU_ALLOW_INSECURE").ok().as_deref() == Some("1") {
            tracing::warn!(
                "LAMU_ALLOW_INSECURE=1: serving an UNAUTHENTICATED API on {} — anyone reachable \
                 can drive inference and spend cloud credits. You asked for this.",
                addr
            );
        } else {
            anyhow::bail!(
                "LAMU_BIND_HOST='{host}' is off-loopback but no API token is set — the \
                 OpenAI-compat API would be unauthenticated on the LAN. Run `lamu auth init` \
                 (writes ~/.config/lamu/api-token) or set LAMU_API_TOKEN, then retry. To expose \
                 without auth on purpose, set LAMU_ALLOW_INSECURE=1."
            );
        }
    }
    let listener = bind_reuseaddr(addr)?;
    tracing::info!("LAMU OpenAI-compat listening on {} (pid {})", addr, std::process::id());

    // Graceful shutdown on SIGINT/SIGTERM so the pidfile gets cleaned up.
    let shutdown = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
            let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler");
            tokio::select! {
                _ = term.recv() => tracing::info!("SIGTERM received, shutting down"),
                _ = int.recv() => tracing::info!("SIGINT received, shutting down"),
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    // Flush any dirty persistent vector-index to disk before the process
    // exits (Item 2: flush_all on shutdown). This runs after the server
    // has stopped accepting requests so no concurrent index mutations occur.
    lamu_memory::tv_store::flush_all();

    // PidFile::Drop unlinks the file.
    drop(pidfile);
    Ok(())
}

/// Bind a listener with SO_REUSEADDR so a fast restart after SIGTERM
/// doesn't trip on TIME_WAIT sockets.
fn bind_reuseaddr(addr: SocketAddr) -> anyhow::Result<tokio::net::TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    let sock = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))?;
    sock.set_reuse_address(true)?;
    sock.set_nonblocking(true)?;
    sock.bind(&addr.into())?;
    sock.listen(1024)?;
    let std_listener: std::net::TcpListener = sock.into();
    Ok(tokio::net::TcpListener::from_std(std_listener)?)
}

/// RAII pidfile. Refuses startup if a live lamu serve already holds the
/// port; cleans up stale entries left by SIGKILLed predecessors.
///
/// Path: `$XDG_RUNTIME_DIR/lamu-serve-{port}.pid` if available, else
/// `/tmp/lamu-serve-{port}.pid`.
#[derive(Debug)]
struct PidFile {
    path: PathBuf,
}

impl PidFile {
    fn acquire(port: u16) -> anyhow::Result<Self> {
        PidFile::acquire_at(pidfile_path(port), Some(port))
    }

    /// Path-injected version of `acquire`. Pulled out so tests can pass
    /// `tempdir().path().join("pid")` and exercise the atomic-create +
    /// stale-reclaim + live-refuse + lost-race paths without touching
    /// `$XDG_RUNTIME_DIR`.
    ///
    /// `port` is informational — only surfaces in the live-holder error
    /// message. Pass `None` from tests that don't care.
    pub(crate) fn acquire_at(path: PathBuf, port: Option<u16>) -> anyhow::Result<Self> {
        // Up to two attempts: first tries atomic O_CREAT|O_EXCL; if that
        // fails with AlreadyExists we inspect the holder, reclaim if dead,
        // and try once more. Two parallel `lamu serve` invocations can't
        // both win — `create_new` is atomic at the syscall level.
        for attempt in 0..2 {
            match std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
            {
                Ok(mut f) => {
                    use std::io::Write as _;
                    write!(f, "{}\n", std::process::id())?;
                    return Ok(PidFile { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if attempt == 1 {
                        // Second attempt already lost the race against a
                        // concurrent winner. Don't reclaim again — the new
                        // holder is presumed live; surface a clear message.
                        anyhow::bail!(
                            "lamu serve: pidfile {} acquisition raced — \
                             another `lamu serve` won the second attempt. \
                             Run `lamu status` or `cat {}` to find the holder.",
                            path.display(), path.display()
                        );
                    }
                    // attempt == 0: inspect the existing pidfile. Live
                    // holder → bail. Dead/unreadable → reclaim + retry.
                    match std::fs::read_to_string(&path) {
                        Ok(s) => {
                            if let Ok(pid) = s.trim().parse::<i32>() {
                                if is_process_alive(pid) {
                                    let port_str = port
                                        .map(|p| format!(":{p} "))
                                        .unwrap_or_default();
                                    anyhow::bail!(
                                        "lamu serve already running on {}(pid {}). \
                                         Use `lamu status` to inspect, then `kill {}` to stop.",
                                        port_str, pid, pid
                                    );
                                }
                                tracing::warn!(
                                    "stale pidfile at {} (pid {} dead) — reclaiming",
                                    path.display(), pid
                                );
                            } else {
                                tracing::warn!(
                                    "unparseable pidfile at {} — reclaiming",
                                    path.display()
                                );
                            }
                        }
                        Err(read_err) => {
                            tracing::warn!(
                                "pidfile at {} unreadable ({}) — reclaiming",
                                path.display(), read_err
                            );
                        }
                    }
                    let _ = std::fs::remove_file(&path);
                    // fall through to next loop iteration
                }
                Err(e) => return Err(e.into()),
            }
        }
        // Unreachable: both attempts cover every Err case explicitly.
        unreachable!("pidfile retry loop exited without returning")
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn pidfile_path(port: u16) -> PathBuf {
    if let Some(rt) = dirs::runtime_dir() {
        return rt.join(format!("lamu-serve-{}.pid", port));
    }
    PathBuf::from(format!("/tmp/lamu-serve-{}.pid", port))
}

#[cfg(unix)]
fn is_process_alive(pid: i32) -> bool {
    use nix::sys::signal::kill;
    use nix::unistd::Pid;
    // kill(pid, 0) reports existence without sending a signal.
    matches!(kill(Pid::from_raw(pid), None), Ok(()))
}

#[cfg(not(unix))]
fn is_process_alive(_pid: i32) -> bool { false }

/// Fire-and-forget preload of the `main: true` registry entry, if any.
/// Doesn't block the HTTP listener — the first request that arrives
/// either hits the still-in-flight load (and waits on the loader's
/// per-name single-flight gate) or finds the model already loaded.
///
/// On failure (e.g. VRAM exhausted) we log a warning and let request-
/// driven loading try a smaller model later. Skipping preload is not
/// fatal.
fn spawn_main_preload(state: openai_compat::AppState) {
    let main_name = state.entries.values()
        .filter(|e| e.main)
        .min_by(|a, b| a.name.cmp(&b.name)) // deterministic when >1 main (#22)
        .map(|e| e.name.clone());
    let Some(name) = main_name else {
        tracing::info!("preload: no `main: true` entry in registry, skipping");
        return;
    };
    if state.scheduler.lock().is_loaded(&name) {
        tracing::info!("preload: '{}' already loaded (auto_register found it), skipping", name);
        return;
    }
    tokio::spawn(async move {
        tracing::info!("preload: spawning '{}'", name);
        match lamu_core::loader::ensure_loaded(
            &name,
            state.entries.as_ref(),
            &state.scheduler,
            &state.health,
            Some(state.http_port),
        ).await {
            Ok(lm) => tracing::info!(
                "preload: '{}' loaded on :{} ({}MB)",
                lm.entry.name, lm.port, lm.vram_actual_mb
            ),
            Err(e) => tracing::warn!(
                "preload: '{}' failed: {} — request-driven load will retry",
                name, e
            ),
        }
    });
}

/// Spawn the liveness reconciliation loop (lamu-core). Every
/// `LAMU_RECONCILE_SECS` (default 7) it probes each `Loaded` model and
/// `mark_unloaded`s any whose backend is dead — so `/health` and
/// `/v1/models` can never report a phantom-loaded model that crashed,
/// got OOM-killed, or was torn down out-of-band. Set the env to `0` to
/// disable (e.g. a deterministic test harness that asserts raw state).
fn spawn_reconcile_loop(state: &openai_compat::AppState) {
    let secs = std::env::var("LAMU_RECONCILE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(lamu_core::reconcile::DEFAULT_INTERVAL_SECS);
    if secs == 0 {
        tracing::info!("reconcile: disabled (LAMU_RECONCILE_SECS=0)");
        return;
    }
    let scheduler = state.scheduler.clone();
    let health = state.health.clone();
    let client = state.client.clone();
    tokio::spawn(lamu_core::reconcile::run_reconcile_loop(
        scheduler,
        health,
        client,
        std::time::Duration::from_secs(secs),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_pidfile() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("lamu-serve-9999.pid");
        (dir, path)
    }

    #[test]
    fn pidfile_acquire_creates_file_with_our_pid() {
        let (_dir, path) = temp_pidfile();
        let _pf = PidFile::acquire_at(path.clone(), Some(9999)).expect("acquire");
        let body = std::fs::read_to_string(&path).expect("read");
        let pid: u32 = body.trim().parse().expect("parse pid");
        assert_eq!(pid, std::process::id());
    }

    #[test]
    fn pidfile_acquire_refuses_live_holder() {
        let (_dir, path) = temp_pidfile();
        // Write our OWN pid — guaranteed live.
        std::fs::write(&path, format!("{}\n", std::process::id())).unwrap();
        let err = PidFile::acquire_at(path, Some(9999)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("already running"), "got: {msg}");
        assert!(msg.contains(&format!("{}", std::process::id())),
            "error must name the live holder PID; got: {msg}");
    }

    #[test]
    fn pidfile_acquire_reclaims_stale_pid() {
        let (_dir, path) = temp_pidfile();
        // `i32::MAX` (2^31 - 1) is never assigned by the Linux kernel:
        // PID_MAX_LIMIT is 2^22 on 64-bit, much smaller. The check is
        // `kill(pid, 0)` which returns ESRCH for unassigned PIDs.
        let dead_pid: i32 = i32::MAX;
        std::fs::write(&path, format!("{}\n", dead_pid)).unwrap();
        let _pf = PidFile::acquire_at(path.clone(), Some(9999))
            .expect("must reclaim stale pidfile");
        let body = std::fs::read_to_string(&path).unwrap();
        let pid: u32 = body.trim().parse().expect("parse pid");
        assert_eq!(pid, std::process::id());
    }

    #[test]
    fn pidfile_acquire_reclaims_unparseable() {
        let (_dir, path) = temp_pidfile();
        std::fs::write(&path, b"this is not a pid number").unwrap();
        let _pf = PidFile::acquire_at(path.clone(), Some(9999))
            .expect("must reclaim unparseable pidfile");
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.trim().parse::<u32>().is_ok());
    }

    #[test]
    fn pidfile_drop_unlinks() {
        let (_dir, path) = temp_pidfile();
        {
            let _pf = PidFile::acquire_at(path.clone(), Some(9999)).unwrap();
            assert!(path.exists());
        }
        assert!(!path.exists(), "Drop must unlink the pidfile");
    }

    #[test]
    fn pidfile_port_field_in_error_message() {
        let (_dir, path) = temp_pidfile();
        std::fs::write(&path, format!("{}\n", std::process::id())).unwrap();
        let err = PidFile::acquire_at(path.clone(), Some(12345)).unwrap_err();
        assert!(format!("{err}").contains("12345"),
            "port should surface in the error; got: {err}");
        // None case: just confirm it doesn't panic + still mentions the live PID.
        std::fs::write(&path, format!("{}\n", std::process::id())).unwrap();
        let err = PidFile::acquire_at(path, None).unwrap_err();
        assert!(format!("{err}").contains("already running"));
    }
}
