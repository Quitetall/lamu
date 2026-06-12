//! lamu-memory — shared memory/persistence capability.
//!
//! ADR 0029: memory is a shared capability used by two frontends —
//! lamu-mcp's MCP tools today, lamu-api's HTTP memory routes next wave.
//! A frontend must not depend on another frontend, so storage lives in
//! this module crate, which depends only on external crates (rusqlite,
//! reqwest, …) — never on a frontend.
//!
//! What lives here:
//! - [`memory`] — per-`conversation_id` append-only turn log
//!   (`conversations.db`).
//! - [`lifetime_memory`] — the GLOBAL cross-session fact store
//!   (`memory.db`): schema + temporal migration, remember / recall /
//!   supersede / forget, novelty dedup, graphify corpus export.
//! - [`rag`] — repo retrieval (`embeddings.db`): ripgrep + OpenAI
//!   embedding semantic search.
//! - [`vector_index`] — the vector-similarity seam (brute-force cosine
//!   default, opt-in `turbovec` backend).
//!
//! What does NOT live here (stays in the frontends): MCP tool handlers,
//! cloud-judged orchestration (fact extraction, auto-contradiction —
//! they call lamu-mcp's cloud plumbing), and untrusted-content fencing
//! (ADR 0011) — fencing is a frontend wire concern. Wire contracts are
//! frozen; this extraction is behavior-preserving.

pub mod lifetime_memory;
pub mod memory;
pub mod rag;
pub mod vector_index;
