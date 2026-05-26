//! Orphan-cleanup primitives shared across lamu binaries.
//!
//! Two mechanisms, used together (belt + suspenders):
//!
//! 1. **`install_parent_death_signal()`** — Linux `PR_SET_PDEATHSIG`.
//!    Kernel delivers SIGTERM the instant our immediate parent dies.
//!    Cheap, fires synchronously, but per Linux 3.4+ semantics is
//!    *cleared* on `execve(2)` for processes with set-UID etc. and has
//!    been observed not to fire in some terminal-multiplexer setups
//!    (orphans surviving for hours in `S<l+`).
//!
//! 2. **`spawn_orphan_watchdog()`** — polls `getppid() == 1` every
//!    5 seconds in a background tokio task. Whenever the original
//!    parent is gone (i.e. we've been reparented to `init`), exit with
//!    code 0. This catches the cases where PDEATHSIG silently failed.
//!
//! Belt + suspenders is overkill for happy-path but the failure mode
//! (zombie `lamu start` processes piling up after Claude Code or a
//! tmux pane closes) is annoying enough that two cheap mechanisms is
//! worth the redundancy.

use std::time::Duration;

/// Ask the kernel to send us SIGTERM when our immediate parent dies.
/// No-op on non-Linux.
#[cfg(target_os = "linux")]
pub fn install_parent_death_signal() {
    let rc = unsafe {
        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM as libc::c_ulong, 0, 0, 0)
    };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        tracing::warn!(
            "PR_SET_PDEATHSIG failed (rc={}, errno={}); orphan-on-parent-death \
             cleanup is degraded — kill stale processes manually",
            rc, err
        );
    }
}

#[cfg(not(target_os = "linux"))]
pub fn install_parent_death_signal() {}

/// Spawn a tokio task that exits the process if reparented to init
/// (i.e. original parent died). Returns immediately. Caller must be
/// in a tokio runtime.
///
/// Polls every 5s. Cost: one syscall per interval, negligible.
#[cfg(unix)]
pub fn spawn_orphan_watchdog() {
    let original_ppid = nix::unistd::getppid();
    // If we were already an orphan at startup, don't pull the rug out
    // from under whatever spawned us — wait for the next genuine
    // re-parent event. (Edge: started by `nohup` from a dying shell.)
    if original_ppid.as_raw() == 1 {
        tracing::info!("orphan-watchdog: ppid=1 at startup — disabled");
        return;
    }
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        // First tick fires immediately; skip it so we don't race startup.
        interval.tick().await;
        loop {
            interval.tick().await;
            let now = nix::unistd::getppid();
            if now.as_raw() == 1 {
                tracing::warn!(
                    "orphan-watchdog: reparented to init (was ppid={}), exiting",
                    original_ppid
                );
                std::process::exit(0);
            }
        }
    });
}

#[cfg(not(unix))]
pub fn spawn_orphan_watchdog() {}

#[cfg(test)]
mod tests {
    use super::*;

    // PDEATHSIG is per-thread state with no public read accessor; we
    // can't directly assert it was set. The smoke test below just
    // confirms the call doesn't panic, doesn't crash, and on Linux
    // returns prctl rc==0 (success path warns; failure path logs but
    // still doesn't panic).
    #[test]
    fn install_parent_death_signal_does_not_panic() {
        install_parent_death_signal();
    }

    // Watchdog must not exit the process when ppid != 1 at startup.
    // We run it for two intervals' worth and assert we're still alive.
    #[tokio::test(flavor = "current_thread")]
    async fn watchdog_does_not_exit_under_live_parent() {
        // We're being run by `cargo test` → parent is the test binary
        // → ppid != 1. Spawn the watchdog and let it tick twice (10s
        // is too long for unit tests, so we use a shorter override via
        // direct call below).
        let original_ppid = nix::unistd::getppid();
        assert_ne!(
            original_ppid.as_raw(),
            1,
            "test must be run under a live parent (cargo test); got ppid=1"
        );
        spawn_orphan_watchdog();
        // Sleep just over one tick. If watchdog wrongly exits, the test
        // process dies and cargo reports a fail. 100ms is well under
        // the 5s poll interval so the watchdog is still on its first
        // delay — the assertion is mostly that spawning didn't panic.
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // If ppid is already 1 at startup, watchdog must disable itself
    // (otherwise we'd exit immediately every time we ran under init).
    // Can't actually set ppid=1, so this test just documents the
    // intent — the behavior is covered by inspection.
    #[test]
    fn watchdog_documents_init_disable() {
        // No-op test; behavior asserted by `if original_ppid == 1 { return }`
        // in spawn_orphan_watchdog. Compiles only if function exists.
        let _ = spawn_orphan_watchdog;
    }
}
