//! lamu-api — OpenAI-compatible HTTP layer.
//! Direct port of `lamu/api/openai_compat.py`.

pub mod metrics;
pub mod openai_compat;

use lamu_core::config::registry_path;
use std::net::SocketAddr;
use std::path::PathBuf;

pub async fn serve(port: u16) -> anyhow::Result<()> {
    let pidfile = PidFile::acquire(port)?;
    let state = openai_compat::build_state(&registry_path(), port)?;
    openai_compat::auto_register(&state).await;
    spawn_main_preload(state.clone());
    let app = openai_compat::build_app(state);

    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
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
struct PidFile {
    path: PathBuf,
}

impl PidFile {
    fn acquire(port: u16) -> anyhow::Result<Self> {
        let path = pidfile_path(port);
        if path.exists() {
            if let Ok(s) = std::fs::read_to_string(&path) {
                if let Ok(pid) = s.trim().parse::<i32>() {
                    if is_process_alive(pid) {
                        anyhow::bail!(
                            "lamu serve already running on :{} (pid {}). \
                             Use `lamu status` to inspect, then `kill {}` to stop.",
                            port, pid, pid
                        );
                    }
                    tracing::warn!(
                        "stale pidfile at {} (pid {} dead) — reclaiming",
                        path.display(), pid
                    );
                }
            }
            let _ = std::fs::remove_file(&path);
        }
        std::fs::write(&path, format!("{}\n", std::process::id()))?;
        Ok(PidFile { path })
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
        .find(|e| e.main)
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
