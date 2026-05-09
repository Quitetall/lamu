//! Pure provider wire-format adapters.
//!
//! This crate is intentionally minimal: shared types + payload
//! builders + header helpers. No HTTP client, no tokio, no blocking.
//! The `Provider` trait (sync transport) lives in `lamu-cli`; the
//! async transport for MCP lives in `lamu-mcp`. Both consume from
//! here so the wire format stays in lock-step.

pub mod headers;
pub mod payload;
pub mod types;

pub use headers::anthropic_beta_header;
pub use payload::{build_anthropic_payload, build_openai_payload};
pub use types::{Message, Role, StreamEvent, ToolCallRef};
