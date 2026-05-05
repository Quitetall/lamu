//! lamu-core — types, registry, scheduler, router, reasoning extraction.
//!
//! Mirrors `lamu/core/` Python package. Each module = direct port of
//! its Python sibling. See `docs/PORT_PLAN.md` for translation guide.

pub mod types;
pub mod registry;
pub mod scheduler;
pub mod router;
pub mod reasoning;
pub mod config;
pub mod error;

pub use error::{Error, Result};
