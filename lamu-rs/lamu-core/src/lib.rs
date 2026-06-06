//! lamu-core — types, registry, scheduler, router, reasoning extraction.
//!
//! Mirrors `lamu/core/` Python package. Each module = direct port of
//! its Python sibling. See `docs/PORT_PLAN.md` for translation guide.

pub mod types;
pub mod registry;
pub mod scheduler;
pub mod scheduler_lock;
pub mod router;
pub mod reasoning;
pub mod sse;
pub mod config;
/// Confined-output path resolver (symlink/traversal-safe). Shared by the media
/// modules (lamu-image, lamu-tts); lived in lamu-mcp before ADR 0023.
pub mod media_paths;
pub mod cookbook;
pub mod error;
pub mod health;
pub mod backends;
pub mod loader;
pub mod reconcile;
pub mod observability;
pub mod queue;
pub mod sandbox;
pub mod supervisor;
pub mod lifecycle;

pub use error::{Error, Result};
