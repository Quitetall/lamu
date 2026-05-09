//! Filesystem journal shim — the canonical implementation lives in
//! `lamu-core::sandbox::journal` so lamu-mcp's `write_file` tool
//! (Phase 6.1) can consume the same record format. This module
//! re-exports the public surface for callers in lamu-cli.
//!
//! Phase 6.0: ownership moved from lamu-cli to lamu-core. The
//! `journal_for(session_id)` helper that returned a `Journal` rooted
//! under `sandbox::sandbox_root()` is preserved, just delegating.

pub use lamu_core::sandbox::journal::{
    rollback, rollback_one, safe_create_dir, safe_delete, safe_write, EncodedBlob, Journal,
    JournalEntry,
};
