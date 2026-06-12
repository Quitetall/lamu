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
//! ADR 0030: embeddings resolve through a local-first chain
//! ([`embedder`]) — env escape hatch → process-global local embedder
//! (registered by each frontend's composition root) → keyed OpenAI →
//! none. Every embedded row records its `embedding_model`; vector
//! recall is model-filtered and fused with an FTS5 lexical leg
//! ([`hybrid`]); [`reembed`] converges rows after a model switch.
//!
//! What lives here:
//! - [`store`] — the unified `lamu.db`: path resolution (`$LAMU_DB`
//!   override), the shared connection singleton, open flow + legacy
//!   import (ADR 0028).
//! - [`migrate`] — versioned schema migrations + the `schema_version`
//!   ledger (ADR 0028).
//! - [`embedder`] — the embedder chain: trait, global registration,
//!   OpenAI + `lamu serve` impls (ADR 0030).
//! - [`hybrid`] — reciprocal-rank fusion + FTS5 query sanitizing
//!   (ADR 0030).
//! - [`reembed`] — `lamu memory reembed`'s lib-level core (ADR 0030).
//! - [`memory`] — per-`conversation_id` append-only turn log
//!   (`conversations` / `turns` tables).
//! - [`lifetime_memory`] — the GLOBAL cross-session fact store
//!   (`memories` table): remember / hybrid recall / supersede / forget,
//!   novelty dedup, graphify corpus export.
//! - [`rag`] — repo retrieval (`chunks` table): ripgrep + embedding
//!   semantic search.
//! - [`vector_index`] — the vector-similarity seam (brute-force cosine
//!   default, opt-in `turbovec` backend).
//! - [`tv_store`] — the persistent `.tv` index lifecycle (ADR 0031):
//!   load-or-rebuild, catch-up, throttled atomic persist, stale
//!   accounting. Feature-on default; inert in a feature-off build.
//!
//! What does NOT live here (stays in the frontends): MCP tool handlers,
//! cloud-judged orchestration (fact extraction, auto-contradiction —
//! they call lamu-mcp's cloud plumbing), and untrusted-content fencing
//! (ADR 0011) — fencing is a frontend wire concern. Wire contracts are
//! frozen; this extraction is behavior-preserving.

pub mod embedder;
pub mod hybrid;
pub mod lifetime_memory;
pub mod memory;
pub mod migrate;
pub mod rag;
pub mod reembed;
pub mod store;
pub mod tv_store;
pub mod vector_index;
