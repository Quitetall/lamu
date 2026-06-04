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
pub mod image;
pub mod lifetime_memory;
pub mod media_paths;
pub mod memory;
pub mod rag;
pub mod server;
pub mod tools;
pub mod train_tool;
pub mod tts;
pub mod untrusted;
pub mod vector_index;
