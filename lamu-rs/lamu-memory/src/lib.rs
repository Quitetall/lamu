//! lamu-memory — shared memory/persistence capability.
//!
//! ADR 0029: memory is a shared capability used by two frontends —
//! lamu-mcp's MCP tools today, lamu-api's HTTP memory routes next wave.
//! A frontend must not depend on another frontend, so storage lives in
//! this module crate, which depends only on external crates (rusqlite,
//! reqwest, …) — never on a frontend.
//!
//! ADR 0028: all persistence lives in ONE schema-versioned SQLite
//! database, `~/.local/share/lamu/lamu.db` ([`store`]), brought up to
//! date by a real migration framework ([`migrate`]) with a one-time
//! import of the three legacy files (`conversations.db`, `memory.db`,
//! `embeddings.db`). Public fn signatures and MCP wire contracts are
//! unchanged.
//!
//! What lives here:
//! - [`store`] — the unified `lamu.db`: path resolution (`$LAMU_DB`
//!   override), the shared connection singleton, open flow + legacy
//!   import (ADR 0028).
//! - [`migrate`] — versioned schema migrations + the `schema_version`
//!   ledger (ADR 0028).
//! - [`memory`] — per-`conversation_id` append-only turn log
//!   (`conversations` / `turns` tables).
//! - [`lifetime_memory`] — the GLOBAL cross-session fact store
//!   (`memories` table): remember / recall / supersede / forget,
//!   novelty dedup, graphify corpus export.
//! - [`rag`] — repo retrieval (`chunks` table): ripgrep + OpenAI
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
pub mod migrate;
pub mod rag;
pub mod store;
pub mod vector_index;
