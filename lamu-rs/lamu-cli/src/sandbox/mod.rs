//! Sandboxing — defense in depth against prompt injection + agent
//! mistakes. Four layers, each independent so the failure of any one
//! doesn't bypass the others.
//!
//! 1. `snap`     — git snapshot at session start, `lamu undo` on exit.
//! 2. `journal`  — every agent fs op recorded with before-bytes,
//!                 `lamu rollback <session>` walks the journal in
//!                 reverse and restores.
//! 3. `gate`     — risky tool-call patterns (rm, dd, curl|sh, etc.)
//!                 require user confirmation in the chat TUI.
//! 4. `launcher` — `lamu agent <cmd>` wraps with bubblewrap (or
//!                 firejail), strict bind mounts, allow-listed network.
//!
//! All sandbox state lives at `~/.local/share/lamu/sandbox/`.

pub mod gate;
pub mod journal;
pub mod launcher;
pub mod snap;

use std::path::PathBuf;

/// Root directory for everything sandbox-related.
pub fn sandbox_root() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lamu")
        .join("sandbox")
}

/// Filesystem-friendly session id. Format: `YYYYMMDD-HHMMSS-rand4`.
pub fn new_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs;
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;
    let year = 1970 + days / 365;
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    // Pseudo-random 4-char suffix from nanos so concurrent sessions
    // don't collide.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let suffix: u32 = nanos % 100_000;
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}-{:05}",
        year, month.min(12), day.min(31), hour, min, sec, suffix
    )
}
