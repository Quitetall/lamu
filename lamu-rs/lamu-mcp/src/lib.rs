//! lamu-mcp — MCP stdio server.
//!
//! Port of `lamu/mcp/server.py`.
//! Exposes 7 tools: query, plan_query, list_models, load_model,
//! unload_model, vram_status, scan_models.
//!
//! Transport: JSON-RPC over stdio.

pub mod auto_context;
pub mod cloud;
pub mod compact;
pub mod context;
pub mod cookbook_tool;
pub mod council;
pub mod handlers;
/// Memory/persistence storage moved to the lamu-memory crate (ADR 0029);
/// the `memory` / `rag` / `lifetime_memory` modules below are frontend
/// shims that re-export it plus the MCP-shaped pieces (tool handlers,
/// cloud-judged orchestration, untrusted fencing). The vector_index seam
/// lives at `lamu_memory::vector_index`.
pub mod lifetime_memory;
/// Re-export: media_paths moved to lamu-core (ADR 0023). Keeps
/// `crate::media_paths::…` working for the in-tree tts/image tools.
pub use lamu_core::media_paths;
pub mod memory;
pub mod rag;
pub mod server;
pub mod tools;
pub mod train_tool;
pub mod untrusted;
