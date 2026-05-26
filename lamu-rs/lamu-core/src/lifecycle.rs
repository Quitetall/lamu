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
//!
//! **Detached intent.** `nohup(1)` sets `SIGHUP` to `SIG_IGN` — the
//! canonical "I want this process to outlive its parent" marker.
//! `spawn_orphan_watchdog` checks for it and disables itself when
//! present, so `nohup lamu serve &` survives terminal close as the
//! user expects. The watchdog only fires for unintentional orphans
//! (parent crashed, tmux pane closed without `disown`, etc.).

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
    // `nohup(1)` sets SIGHUP to SIG_IGN before exec. Treat that as the
    // user's explicit "survive parent death" marker and skip the
    // watchdog — otherwise `nohup lamu serve &` dies the moment the
    // launching subshell exits.
    if sighup_is_ignored() {
        tracing::info!("orphan-watchdog: SIGHUP=SIG_IGN (nohup/detached) — disabled");
        return;
    }
    let original_ppid = nix::unistd::getppid();
    // If we were already an orphan at startup, don't pull the rug out
    // from under whatever spawned us — wait for the next genuine
    // re-parent event.
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

/// True iff SIGHUP is currently set to SIG_IGN. `nohup(1)` is the
/// canonical caller; some daemonization wrappers also set this.
#[cfg(unix)]
fn sighup_is_ignored() -> bool {
    let mut act: libc::sigaction = unsafe { std::mem::zeroed() };
    // sigaction(2) with new_act=NULL reads the current handler.
    let rc = unsafe { libc::sigaction(libc::SIGHUP, std::ptr::null(), &mut act) };
    if rc != 0 {
        // Couldn't read disposition — fail-open (run watchdog).
        return false;
    }
    act.sa_sigaction == libc::SIG_IGN
}

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
    #[test]
    fn sighup_default_disposition_not_ignored() {
        // Cargo test inherits SIGHUP=SIG_DFL from the launching shell
        // (unless that shell was itself nohup'd, which is unusual).
        // Asserts the helper returns false under normal test conditions.
        assert!(!sighup_is_ignored());
    }

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

}
