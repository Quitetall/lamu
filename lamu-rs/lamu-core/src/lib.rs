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
pub mod config;
pub mod error;
pub mod health;
pub mod backends;
pub mod loader;
pub mod observability;
pub mod queue;
pub mod sandbox;
pub mod supervisor;
pub mod lifecycle;

pub use error::{Error, Result};
