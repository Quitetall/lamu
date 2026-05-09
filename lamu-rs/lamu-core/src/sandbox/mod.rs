//! Sandbox primitives shared between lamu-cli and lamu-mcp.
//!
//! Today this only houses the filesystem journal (`safe_write`,
//! `safe_delete`, `safe_create_dir`, `rollback`). The other layers —
//! git snapshots, tool-call gating, bubblewrap launcher — live in
//! `lamu-cli/src/sandbox/` because they couple to the interactive
//! TUI surface. If MCP grows a `write_file` tool (Phase 6.1), it
//! consumes from here so the journal record is identical to what
//! the CLI writes.

pub mod journal;

use std::path::PathBuf;

/// Root directory for everything sandbox-related. Mirrors the path
/// the lamu-cli sandbox root resolved to so existing journal files
/// from earlier sessions remain readable through this module.
pub fn sandbox_root() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lamu")
        .join("sandbox")
}
