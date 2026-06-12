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
/// Module-tool extension seam (ADR 0023): ToolCtx + ModuleTool registry so
/// modules contribute MCP tools without a lamu-mcp dependency.
pub mod tools_ext;
/// Scripted ToolCtx double for agentic-flow tests (feature `test-support`;
/// consumer crates enable it in dev-dependencies — never in production).
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod cookbook;
/// Shared SearXNG retrieval + prompt-injection sanitization (ADR audit B7):
/// one keyless metasearch backend + `sanitize_field` for both lamu-jart's
/// `web_search`/`answer` and lamu-api's auto-grounding.
pub mod web_search;
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
