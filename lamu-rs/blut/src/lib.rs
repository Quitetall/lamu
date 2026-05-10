//! lamu-train — local fine-tuning subsystem.
//!
//! This crate owns the data shapes and lifecycle types for a training
//! run. The Python trainer is launched in a later step; for now the
//! crate is types-only, designed to be the byte-stable contract
//! between every component that talks about a training job:
//!
//!   - `lamu-train` binary (CLI entry point, future)
//!   - `lamu-mcp::train_tool` (MCP `train_from_conversations` handler)
//!   - on-disk job files (`spec.json`, `status.jsonl`)
//!   - eventually a Python trainer subprocess (one StatusUpdate JSON per
//!     line on stdout)
//!
//! Design rules:
//!
//!   - All public types serialize to/from JSON with stable field names.
//!     Anything user-visible (CLI flags, MCP schema, on-disk records)
//!     reuses these types directly — there's no second representation.
//!   - Errors flow through `TrainError`. Anything user-facing speaks
//!     `Display` (the inner string is the error message that lands in
//!     the user's terminal or MCP response).
//!   - The `TrainBackend` trait is intentionally narrow: `run` and
//!     `cancel`. New backends (e.g. a future "rented GPU" backend)
//!     plug in by implementing it; callers don't change.
//!   - No I/O in this crate beyond the trait contract. Spec validation
//!     is pure; protocol is pure; even the trait method is `async`
//!     only because the future implementations need it.

// Production code stays unsafe-free; tests use unsafe env-var
// mutations for hermetic env-injection patterns.
#![cfg_attr(not(test), forbid(unsafe_code))]

pub mod artifacts;
pub mod backend;
pub mod conversations;
pub mod convert;
pub mod datasets_db;
pub mod error;
pub mod framework;
pub mod jobs;
pub mod paths;
pub mod policy;
pub mod protocol;
pub mod python_backend;
pub mod recipes;
pub mod spec;
pub mod stages;

/// Process-wide lock for tests that mutate environment variables.
/// Multiple test modules touch `LAMU_TRAIN_*` env vars; without a
/// shared mutex parallel test execution races on the global env.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub use backend::{TrainArtifact, TrainBackend};
pub use error::TrainError;
pub use protocol::StatusUpdate;
pub use python_backend::PythonTrainBackend;
pub use spec::{DatasetSource, Method, Optim, TrainSpec};
