//! Provider wire-format adapters.
//!
//! Shared types + payload builders + a single env-driven header
//! helper. No HTTP client, no tokio, no blocking transport. The
//! payload builders are pure (input → JSON); `headers::anthropic_beta_header`
//! is the one exception that touches IO, by reading the
//! `ANTHROPIC_BETA` environment variable.
//!
//! The `Provider` trait (sync transport) lives in `lamu-cli`; the
//! async transport for MCP lives in `lamu-mcp`. Both consume from
//! here so the wire format stays in lock-step.

pub mod cloud_config;
pub mod headers;
pub mod payload;
pub mod types;

pub use cloud_config::{config_path, load_or_empty, CloudModel, CloudModelList, QuotaState};
pub use headers::anthropic_beta_header;
pub use payload::{build_anthropic_payload, build_openai_payload};
pub use types::{Message, Role, StreamEvent, ToolCallRef};
