//! Repo retrieval (RAG) — MCP frontend shim.
//!
//! The implementation (ripgrep + OpenAI-embedding semantic search over
//! `~/.local/share/lamu/embeddings.db`) moved to `lamu_memory::rag`
//! (ADR 0029: retrieval/persistence is a shared capability used by
//! multiple frontends, so it lives in a module crate that depends only
//! on external crates). This re-export keeps every existing
//! `crate::rag::X` call site — `search` / `index_repo` / `SearchMode` /
//! `SearchHit` from the tool dispatch in cloud.rs — compiling unchanged.

pub use lamu_memory::rag::*;
