//! Lifetime cross-session memory — storage core.
//!
//! Where `memory.rs` keys turns by `conversation_id` (strictly
//! per-conversation), this module is a GLOBAL fact store that spans
//! every conversation. Facts are extracted from conversations (via
//! MiMo — that orchestration lives in the lamu-mcp frontend, ADR 0029)
//! or added explicitly, embedded via the embedder chain (ADR 0030:
//! local registry model first, OpenAI escape hatch / fallback — see
//! `crate::embedder::resolve`), and recalled by HYBRID cross-session
//! search: a model-filtered vector leg over the existing
//! `crate::vector_index::BruteForceCosine` seam fused with an FTS5
//! lexical leg via reciprocal-rank fusion (`crate::hybrid`).
//!
//! ADR 0029: this crate holds only the storage capability (schema +
//! temporal migration, remember / recall / supersede / forget, novelty
//! dedup, corpus export). The cloud-judged orchestration (fact
//! extraction, auto-contradiction) and the MCP tool handlers stay in
//! lamu-mcp's `lifetime_memory` module, which re-exports everything
//! here so its call sites compile unchanged.
//!
//! ## Storage
//!
//! The `memories` table of the unified `lamu.db` (ADR 0028;
//! previously its own `memory.db`). The schema is owned by
//! `crate::migrate`; the connection is the shared `crate::store`
//! singleton. The pre-0028 in-place helpers ([`LEGACY_SCHEMA`],
//! [`migrate_temporal_columns`], [`open_legacy_memory_db`]) survive
//! only to normalize a legacy `memory.db` before its one-time import.
//!
//! ## Degradation without an embedder
//!
//! `remember` stores the memory with `embedding = NULL` when the chain
//! resolves no embedder — it never fails on a missing backend.
//! `recall_memory` ranks embedding-bearing rows semantically (vector
//! leg, filtered to the current embedder's model) fused with the FTS5
//! lexical leg; with no embedder at all it degrades to FTS + recency
//! (pre-0030 it was recency-only — strictly better now).
//!
//! ## Per-store identity (ADR 0030)
//!
//! Every embedded row records its `embedding_model`; the vector leg
//! ranks ONLY rows matching the current embedder's model (vectors from
//! different models live in different spaces). The `embedding_stores`
//! row for 'memories' tracks the store's adopted identity; on a model
//! switch the old row is kept (warn once) until `lamu memory reembed`
//! converges the rows.

use anyhow::{Context, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Arc;

use crate::embedder::EmbedderId;
use crate::hybrid::{rrf_merge, sanitize_fts_query};
use crate::rag::{blob_to_vec, vec_to_blob};
use crate::vector_index::{cosine, vector_backend, BruteForceCosine, Scored, VectorBackend, VectorIndex};

/// This module's persistent-index store id (ADR 0031).
const TV_STORE: crate::tv_store::Store = crate::tv_store::Store::Memories;

/// The default owner for every MCP/local caller (ADR 0032; the schema
/// default the `owner` column has carried since ADR 0028). HTTP callers
/// under `AuthMode::KeyStore` pass their key's user instead; everything
/// else — MCP tool handlers, autocapture/reconcile, the CLI — passes
/// this constant.
pub const LOCAL_OWNER: &str = "local";

/// The PRE-ADR-0028 standalone `memory.db` schema. Kept ONLY for the
/// legacy open path ([`open_legacy_memory_db`]) that normalizes an old
/// file before its one-time import into `lamu.db` — the live schema is
/// owned by `crate::migrate` (migration 001 adds `owner` +
/// `embedding_model` on top of this shape).
///
/// `idx_memories_valid` is deliberately NOT in this batch: on a
/// pre-temporal file the `valid_until` column doesn't exist yet, so
/// creating the index here would fail before [`migrate_temporal_columns`]
/// (which adds the column AND creates that index) gets to run.
const LEGACY_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding BLOB,
    kind TEXT NOT NULL DEFAULT 'fact',
    source TEXT,
    ts INTEGER NOT NULL,
    valid_from INTEGER NOT NULL DEFAULT 0,
    valid_until INTEGER,
    supersedes INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories(ts);
";

/// Idempotent valid-time migration for an EXISTING `memories` table.
///
/// `CREATE TABLE IF NOT EXISTS` in [`SCHEMA`] only adds the three
/// temporal columns to a FRESH database — it does NOT alter a pre-existing
/// table. This brings an old `memory.db` up to the temporal schema:
///
/// 1. Read `PRAGMA table_info(memories)` to see which columns exist.
/// 2. For each of `valid_from` / `valid_until` / `supersedes` that is
///    missing, `ALTER TABLE memories ADD COLUMN ...`. (SQLite `ADD COLUMN`
///    is not natively idempotent — it errors if the column already exists —
///    so we gate on the PRAGMA check, which makes the whole migration safe
///    to run on EVERY startup.)
/// 3. Backfill: pre-migration rows added the column with the default `0`
///    for `valid_from`; set `valid_from = ts` for those so historical rows
///    carry a sane validity start.
/// 4. Ensure the `idx_memories_valid` index exists.
///
/// No row is ever dropped or rewritten beyond the backfill UPDATE; the
/// migration cannot lose data and is a no-op once the columns are present
/// and backfilled.
fn migrate_temporal_columns(conn: &Connection) -> Result<()> {
    // Existing column names on the table.
    let existing: std::collections::HashSet<String> = {
        let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
        let cols = stmt.query_map([], |r| r.get::<_, String>(1))?;
        let mut set = std::collections::HashSet::new();
        for c in cols {
            set.insert(c?);
        }
        set
    };

    // ADD COLUMN for each temporal column that is not already present.
    // The column definitions mirror SCHEMA so fresh + migrated DBs match.
    if !existing.contains("valid_from") {
        conn.execute_batch(
            "ALTER TABLE memories ADD COLUMN valid_from INTEGER NOT NULL DEFAULT 0",
        )?;
    }
    if !existing.contains("valid_until") {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN valid_until INTEGER")?;
    }
    if !existing.contains("supersedes") {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN supersedes INTEGER")?;
    }

    // Backfill: rows that predate the migration got valid_from = 0 (the
    // column default). Give them a sane validity start = their ts. New rows
    // set valid_from = ts explicitly so this only ever touches old rows.
    conn.execute_batch("UPDATE memories SET valid_from = ts WHERE valid_from = 0")?;

    // The valid-time recall filter scans valid_until; index it. IF NOT
    // EXISTS keeps this a no-op after the first run.
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_until)",
    )?;
    Ok(())
}

/// Build the SQL WHERE-clause fragment that restricts a memory query to
/// currently-valid facts (`include_expired = false`) or to all facts
/// regardless of validity (`include_expired = true`).
///
/// A fact is currently valid when its `valid_until` is NULL (never expired)
/// or lies strictly in the future relative to `now`. `now` is inlined as a
/// literal so the fragment can be concatenated into a prepared statement
/// without juggling bind-parameter ordering against the other binds
/// (`embedding IS NOT NULL`, `LIMIT ?`). Pure + unit-testable.
pub(crate) fn valid_time_clause(include_expired: bool, now: i64) -> String {
    if include_expired {
        String::new()
    } else {
        format!("(valid_until IS NULL OR valid_until > {now})")
    }
}

// ── Memory DB handle ───────────────────────────────────────────────

/// Open a PRE-ADR-0028 standalone `memory.db` through its historical
/// open path: legacy schema applied idempotently, then
/// [`migrate_temporal_columns`] normalizes a pre-temporal file. Used
/// only by `crate::store`'s one-time legacy import (so the
/// INSERT…SELECT can name the valid-time columns unconditionally) —
/// the live store opens via `crate::store` instead.
pub(crate) fn open_legacy_memory_db(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.execute_batch(LEGACY_SCHEMA)?;
    // LEGACY_SCHEMA's CREATE TABLE IF NOT EXISTS only adds the temporal
    // columns to a FRESH db; bring an existing memory.db up to the
    // valid-time schema (idempotent, safe to run on every open).
    migrate_temporal_columns(&conn)?;
    Ok(conn)
}

/// The fact store's connection — a thin delegate to the unified
/// `lamu.db` singleton (`crate::store`, ADR 0028). Kept under its
/// historical name so the storage fns below read unchanged.
fn memory_db() -> Result<Arc<Mutex<Connection>>> {
    crate::store::shared_handle()
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ── Records ────────────────────────────────────────────────────────

/// One memory returned from a recall. `score` is `Some` for the
/// semantic (embedding) path and `None` for the recency fallback.
/// `valid_until` is `None` for a currently-valid fact; `Some` only when
/// the hit came from an `include_expired` recall (ADR 0032 surfaces it
/// on the HTTP recall response).
#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub id: i64,
    pub text: String,
    pub kind: String,
    pub source: Option<String>,
    pub ts: i64,
    pub score: Option<f32>,
    pub valid_until: Option<i64>,
}

/// A memory row loaded for ranking: `(id, text, kind, source, ts,
/// embedding)`. Aliased to tame clippy's `type_complexity` lint on the
/// `rank_memories` / `recall_memory` plumbing without changing the
/// underlying shape.
pub(crate) type MemoryRow = (i64, String, String, Option<String>, i64, Vec<f32>);

/// Payload carried through `BruteForceCosine` during ranking. Must be
/// `Clone` (the seam clones payloads out on search).
#[derive(Debug, Clone)]
pub(crate) struct MemoryPayload {
    pub id: i64,
    pub text: String,
    pub kind: String,
    pub source: Option<String>,
    pub ts: i64,
}

// ── Storage ────────────────────────────────────────────────────────

/// Insert one memory row, returning its rowid. Factored out of
/// `remember` so tests can store a row with a KNOWN embedding (or none)
/// against an in-memory connection without touching any embedder.
/// `embedding_model` tags the row's provenance (ADR 0030) — pass the
/// embedder's identity model when `embedding` is `Some`, `None` when
/// the row carries no embedding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_memory(
    conn: &Connection,
    text: &str,
    embedding: Option<&[f32]>,
    embedding_model: Option<&str>,
    kind: &str,
    source: &str,
    ts: i64,
    owner: &str,
) -> Result<i64> {
    // A brand-new fact is valid from `ts`, has no expiry, and supersedes
    // nothing. supersede() uses insert_memory_full to set `supersedes`.
    insert_memory_full(conn, text, embedding, embedding_model, kind, source, ts, ts, None, owner)
}

/// Insert one memory row with full control over the temporal columns,
/// returning its rowid. `valid_from`/`valid_until`/`supersedes` are set
/// verbatim. [`insert_memory`] is the common case (valid_from = ts,
/// valid_until = NULL, supersedes = NULL); [`supersede`] uses this directly
/// to record `supersedes = Some(old_id)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_memory_full(
    conn: &Connection,
    text: &str,
    embedding: Option<&[f32]>,
    embedding_model: Option<&str>,
    kind: &str,
    source: &str,
    ts: i64,
    valid_from: i64,
    supersedes: Option<i64>,
    owner: &str,
) -> Result<i64> {
    let blob = embedding.map(vec_to_blob);
    // Only stamp a model when an embedding is actually present — a NULL
    // embedding with a model tag would look reembeddable-but-done.
    // `owner` is named explicitly (ADR 0032 plumb-through) — relying on
    // the schema default would silently mis-attribute a KeyStore write.
    let model = embedding.and(embedding_model);
    conn.execute(
        "INSERT INTO memories (owner, text, embedding, embedding_model, kind, source, ts, valid_from, valid_until, supersedes) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
        params![owner, text, blob, model, kind, source, ts, valid_from, supersedes],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Embed `text` via the chain. `Ok(None)` when no embedder resolves
/// (store unembedded — never fail on a missing backend); `Err` when an
/// embedder exists but the embed itself fails (same contract the keyed
/// OpenAI path had).
async fn embed_via_chain(text: &str) -> Result<Option<(Vec<f32>, EmbedderId)>> {
    let Some(embedder) = crate::embedder::resolve() else {
        return Ok(None);
    };
    let mut vecs = embedder.embed(std::slice::from_ref(&text.to_string())).await?;
    let v = vecs
        .pop()
        .ok_or_else(|| anyhow::anyhow!("embedder returned no vector"))?;
    Ok(Some((v, embedder.identity())))
}

/// Store a fact in the lifetime memory. Embeds via the embedder chain
/// when one resolves; stores `embedding = NULL` otherwise (never fails
/// on a missing backend). Returns the new rowid. `owner` scopes the
/// fact (ADR 0032): MCP/local callers pass [`LOCAL_OWNER`]; the HTTP
/// surface passes the KeyStore principal's user.
pub async fn remember(text: &str, kind: &str, source: &str, owner: &str) -> Result<i64> {
    let embedded = embed_via_chain(text).await?;
    let arc = memory_db()?;
    let id = {
        let conn = arc.lock();
        let now = now_secs();
        let id = insert_memory(
            &conn,
            text,
            embedded.as_ref().map(|(v, _)| v.as_slice()),
            embedded.as_ref().map(|(_, ident)| ident.model.as_str()),
            kind,
            source,
            now,
            owner,
        )?;
        if let Some((v, ident)) = &embedded {
            crate::store::record_store_identity(&conn, "memories", &ident.model, v.len(), now);
            // ADR 0031: append to the live persistent index (no-op when
            // the persistent path is inactive or the index isn't loaded).
            crate::tv_store::note_added(&conn, TV_STORE, id, v, &ident.model);
        }
        id
    }; // DB lock released — persist (file I/O) must not run under it.
    crate::tv_store::maybe_persist(TV_STORE);
    Ok(id)
}

// ── Recall ─────────────────────────────────────────────────────────

/// Rank embedding-bearing rows against a query embedding via the
/// `BruteForceCosine` seam, returning the top-`k` as `MemoryHit`s with a
/// `Some(score)`. Pure (no I/O, no network) so it is unit-testable.
///
/// `rows` carries only rows that HAVE an embedding (the caller filters
/// out NULL-embedding rows before building this list).
/// Build the selected [`VectorIndex`] backend from payload rows and return
/// the top-`k` scored payloads. Brute is the unchanged default; TurboVec is
/// reachable only with the `turbovec` feature compiled in AND
/// `LAMU_VECTOR_BACKEND=turbovec` at runtime (see [`vector_backend`]). Both
/// branches add the same rows + run the same `.search`, so `rank_memories`'
/// `MemoryHit` mapping is backend-agnostic.
fn rank_rows(
    rows: Vec<(Vec<f32>, MemoryPayload)>,
    query_emb: &[f32],
    k: usize,
) -> Vec<Scored<MemoryPayload>> {
    fn fill<I: VectorIndex<MemoryPayload>>(
        mut index: I,
        rows: Vec<(Vec<f32>, MemoryPayload)>,
        query_emb: &[f32],
        k: usize,
    ) -> Vec<Scored<MemoryPayload>> {
        for (emb, payload) in rows {
            index.add(emb, payload);
        }
        index.search(query_emb, k)
    }
    match vector_backend() {
        VectorBackend::Brute => {
            fill(BruteForceCosine::with_capacity(rows.len()), rows, query_emb, k)
        }
        VectorBackend::TurboVec => {
            #[cfg(feature = "turbovec")]
            {
                fill(
                    crate::vector_index::TurboVecIndex::with_capacity(rows.len()),
                    rows,
                    query_emb,
                    k,
                )
            }
            #[cfg(not(feature = "turbovec"))]
            {
                fill(BruteForceCosine::with_capacity(rows.len()), rows, query_emb, k)
            }
        }
    }
}

pub(crate) fn rank_memories(
    query_emb: &[f32],
    rows: Vec<MemoryRow>,
    k: usize,
) -> Vec<MemoryHit> {
    // SEAM: same vector-index seam rag.rs uses for repo search. Build the
    // payload rows once, then let the selector pick the backend; the
    // result-mapping below is identical for both branches.
    let payload_rows: Vec<(Vec<f32>, MemoryPayload)> = rows
        .into_iter()
        .map(|(id, text, kind, source, ts, emb)| {
            (
                emb,
                MemoryPayload {
                    id,
                    text,
                    kind,
                    source,
                    ts,
                },
            )
        })
        .collect();
    rank_rows(payload_rows, query_emb, k)
        .into_iter()
        .map(|hit| MemoryHit {
            id: hit.payload.id,
            text: hit.payload.text,
            kind: hit.payload.kind,
            source: hit.payload.source,
            ts: hit.payload.ts,
            score: Some(hit.score),
            // Ranking is only ever fed currently-valid rows (or the
            // caller hydrates validity afterwards) — the leg itself
            // doesn't carry the temporal column.
            valid_until: None,
        })
        .collect()
}

/// Recall the top-`k` memories most relevant to `query` — HYBRID
/// vector + FTS5 recall fused by reciprocal-rank fusion (ADR 0030).
///
/// - With an embedder: the vector leg embeds the query and cosine-ranks
///   the embedding-bearing rows WHOSE `embedding_model` MATCHES the
///   current embedder's model (mixed-model vectors never rank against
///   each other); the FTS leg bm25-ranks `memories_fts`; the two ranked
///   lists are RRF-merged. `score` carries the COSINE similarity for
///   hits the vector leg scored (preserving the pre-0030 score
///   semantics — `lamu-mcp`'s reconcile compares it against the novelty
///   threshold) and `None` for FTS-only hits.
/// - Without an embedder: the recency list substitutes for the vector
///   leg, so recall degrades to FTS + recency (`score = None`) —
///   strictly better than the pre-0030 recency-only fallback. An embed
///   FAILURE (backend down mid-flight) degrades the same way with a
///   warning instead of failing the whole recall.
///
/// VALID-TIME SEMANTICS: by default (`include_expired = false`) recall
/// returns ONLY currently-valid facts — rows whose `valid_until` is NULL
/// (never expired) or lies in the future. This is the intended temporal
/// behaviour: facts that were superseded or soft-deleted (forgotten) drop
/// out of default recall but are NEVER removed from the store. Pass
/// `include_expired = true` for historical recall over the full timeline.
///
/// OWNER SCOPING (ADR 0032): every leg — vector (per-query scan AND the
/// persistent-index post-filter), FTS, recency — is restricted to rows
/// whose `owner` matches; cross-owner facts can never surface.
pub async fn recall_memory(
    query: &str,
    k: usize,
    include_expired: bool,
    owner: &str,
) -> Result<Vec<MemoryHit>> {
    // Embed BEFORE taking the lock (network/backend await must not hold
    // the shared store mutex).
    let embedded = match crate::embedder::resolve() {
        Some(e) => match e.embed(std::slice::from_ref(&query.to_string())).await {
            Ok(mut v) if !v.is_empty() => Some((v.remove(0), e.identity().model)),
            Ok(_) => None,
            Err(err) => {
                tracing::warn!("recall: query embed failed ({err}) — degrading to FTS + recency");
                None
            }
        },
        None => None,
    };
    let arc = memory_db()?;
    let now = now_secs();
    let conn = arc.lock();
    recall_hybrid_conn(
        &conn,
        query,
        embedded.as_ref().map(|(v, _)| v.as_slice()),
        embedded.as_ref().map(|(_, m)| m.as_str()),
        k,
        include_expired,
        now,
        owner,
    )
}

/// Connection-level core of [`recall_memory`]: vector leg (when
/// `qvec`/`model` are present) + FTS leg, RRF-merged, hydrated in merge
/// order. Factored out so tests drive it against a tempdir connection
/// with known embeddings and `now`.
///
/// NOTE: the cosine pass runs under the caller's lock — same trade-off
/// `remember_if_novel` already accepted: fine while the store is small;
/// revisit if it grows large.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recall_hybrid_conn(
    conn: &Connection,
    query: &str,
    qvec: Option<&[f32]>,
    model: Option<&str>,
    k: usize,
    include_expired: bool,
    now: i64,
    owner: &str,
) -> Result<Vec<MemoryHit>> {
    if k == 0 {
        return Ok(Vec::new());
    }
    let valid = valid_time_clause(include_expired, now);

    // ── Leg 1: vector (model-filtered) or recency substitute ───────
    // The cosine score per id is kept so hydration can surface it.
    let mut cosine_by_id: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
    let leg1: Vec<(i64, f32)> = if let (Some(qvec), Some(model)) = (qvec, model) {
        // ADR 0031: the persistent index serves the leg when active. Its
        // raw hits are over-fetched (k * OVERFETCH) and post-filtered
        // against SQLite (validity + model + owner) — expired rows STAY
        // in the .tv (invalidation-without-delete), the filter hides
        // them; the same post-filter is what enforces owner scoping
        // (the .tv is built per (store, model), NOT per owner).
        // `None` (inactive / dims not %8 / error) → the per-query scan.
        if let Some(raw) = crate::tv_store::search_persistent(conn, TV_STORE, qvec, model, k) {
            let kept = filter_indexed_candidates(conn, &raw, model, &valid, k, owner)?;
            for (id, s) in &kept {
                cosine_by_id.insert(*id, *s);
            }
            kept
        } else {
            let where_clause = if valid.is_empty() {
                "WHERE embedding IS NOT NULL AND embedding_model = ?1 AND owner = ?2".to_string()
            } else {
                format!(
                    "WHERE embedding IS NOT NULL AND embedding_model = ?1 AND owner = ?2 AND {valid}"
                )
            };
            let sql = format!(
                "SELECT id, text, kind, source, ts, embedding FROM memories {where_clause}"
            );
            let rows: Vec<MemoryRow> = {
                let mut stmt = conn.prepare(&sql)?;
                let mapped = stmt.query_map(params![model, owner], |r| {
                    let id: i64 = r.get(0)?;
                    let text: String = r.get(1)?;
                    let kind: String = r.get(2)?;
                    let source: Option<String> = r.get(3)?;
                    let ts: i64 = r.get(4)?;
                    let emb: Vec<u8> = r.get(5)?;
                    Ok((id, text, kind, source, ts, blob_to_vec(&emb)))
                })?;
                let mut rows = Vec::new();
                for row in mapped {
                    rows.push(row?);
                }
                rows
            };
            rank_memories(qvec, rows, k)
                .into_iter()
                .map(|h| {
                    let s = h.score.unwrap_or(0.0);
                    cosine_by_id.insert(h.id, s);
                    (h.id, s)
                })
                .collect()
        }
    } else {
        // No vector leg → recency list takes its slot in the fusion.
        let where_clause = if valid.is_empty() {
            "WHERE owner = ?1 ".to_string()
        } else {
            format!("WHERE owner = ?1 AND {valid} ")
        };
        let sql = format!(
            "SELECT id FROM memories {where_clause}ORDER BY ts DESC, id DESC LIMIT ?2"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mapped = stmt.query_map(params![owner, k as i64], |r| r.get::<_, i64>(0))?;
        let mut ids = Vec::new();
        for id in mapped {
            ids.push((id?, 0.0_f32));
        }
        ids
    };

    // ── Leg 2: FTS5 bm25 over memories_fts ─────────────────────────
    // bm25() is "smaller is better" (negative for good matches), so
    // ascending ORDER BY puts the best first — RRF only needs the rank.
    // The leg fetches more than k candidates so fusion has room.
    // OWNER: memories_fts has no owner column (external-content FTS over
    // text only), so the filter rides the JOIN against `memories`.
    let fts_leg: Vec<(i64, f64)> = match sanitize_fts_query(query) {
        None => Vec::new(),
        Some(match_expr) => {
            let and_valid = if valid.is_empty() {
                String::new()
            } else {
                format!("AND {valid} ")
            };
            let sql = format!(
                "SELECT memories_fts.rowid, bm25(memories_fts) \
                 FROM memories_fts JOIN memories ON memories.id = memories_fts.rowid \
                 WHERE memories_fts MATCH ?1 AND memories.owner = ?2 {and_valid}\
                 ORDER BY bm25(memories_fts) ASC LIMIT ?3"
            );
            let limit = (k * 4).max(16) as i64;
            let run = || -> Result<Vec<(i64, f64)>> {
                let mut stmt = conn.prepare(&sql)?;
                let mapped = stmt.query_map(params![match_expr, owner, limit], |r| {
                    Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
                })?;
                let mut out = Vec::new();
                for row in mapped {
                    out.push(row?);
                }
                Ok(out)
            };
            match run() {
                Ok(v) => v,
                Err(e) => {
                    // The sanitizer should make MATCH syntax-safe; if a
                    // pathological query still errors, drop the leg
                    // rather than the whole recall.
                    tracing::warn!("recall: FTS leg failed ({e}) — vector/recency leg only");
                    Vec::new()
                }
            }
        }
    };

    // ── Fuse + hydrate in merge order ───────────────────────────────
    let merged = rrf_merge(&leg1, &fts_leg, k);
    if merged.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = vec!["?"; merged.len()].join(",");
    let and_valid = if valid.is_empty() {
        String::new()
    } else {
        format!(" AND {valid}")
    };
    let sql = format!(
        "SELECT id, text, kind, source, ts, valid_until FROM memories \
         WHERE id IN ({placeholders}) AND owner = ?{owner_pos}{and_valid}",
        owner_pos = merged.len() + 1
    );
    let mut stmt = conn.prepare(&sql)?;
    let bind: Vec<rusqlite::types::Value> = merged
        .iter()
        .map(|(id, _)| rusqlite::types::Value::Integer(*id))
        .chain(std::iter::once(rusqlite::types::Value::Text(owner.to_string())))
        .collect();
    let mapped = stmt.query_map(rusqlite::params_from_iter(bind), |r| {
        Ok(MemoryHit {
            id: r.get(0)?,
            text: r.get(1)?,
            kind: r.get(2)?,
            source: r.get(3)?,
            ts: r.get(4)?,
            score: None,
            valid_until: r.get(5)?,
        })
    })?;
    let mut by_id: std::collections::HashMap<i64, MemoryHit> = std::collections::HashMap::new();
    for h in mapped {
        let h = h?;
        by_id.insert(h.id, h);
    }
    Ok(merged
        .iter()
        .filter_map(|(id, _)| by_id.remove(id))
        .map(|mut h| {
            h.score = cosine_by_id.get(&h.id).copied();
            h
        })
        .collect())
}

/// Post-filter raw persistent-index candidates against SQLite: keep ids
/// that are still embedding-bearing under the CURRENT `model` AND pass
/// the `valid` time clause (expired rows stay in the .tv — this filter is
/// what hides them) AND belong to `owner` (the .tv is built per
/// (store, model) with NO owner partitioning — this post-filter is the
/// owner fence, ADR 0032), preserve the index's score order, dedup by id
/// (chunk-style rowid reuse can alias two slots onto one id; first =
/// best-scored wins), truncate to `k` (ADR 0031).
///
/// FOLLOW-UP (not v1): the index over-fetches `k * OVERFETCH` across ALL
/// owners; under heavy multi-tenant usage one owner's rows could crowd a
/// small owner's out of the candidate set. Per-owner over-fetch scaling
/// (or per-owner indexes) is the documented follow-up if that bites.
fn filter_indexed_candidates(
    conn: &Connection,
    raw: &[(i64, f32)],
    model: &str,
    valid: &str,
    k: usize,
    owner: &str,
) -> Result<Vec<(i64, f32)>> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = vec!["?"; raw.len()].join(",");
    let and_valid = if valid.is_empty() {
        String::new()
    } else {
        format!(" AND {valid}")
    };
    let sql = format!(
        "SELECT id FROM memories WHERE id IN ({placeholders}) \
         AND embedding IS NOT NULL AND embedding_model = ?{model_pos} \
         AND owner = ?{owner_pos}{and_valid}",
        model_pos = raw.len() + 1,
        owner_pos = raw.len() + 2
    );
    let mut stmt = conn.prepare(&sql)?;
    let bind: Vec<rusqlite::types::Value> = raw
        .iter()
        .map(|(id, _)| rusqlite::types::Value::Integer(*id))
        .chain(std::iter::once(rusqlite::types::Value::Text(model.to_string())))
        .chain(std::iter::once(rusqlite::types::Value::Text(owner.to_string())))
        .collect();
    let mapped = stmt.query_map(rusqlite::params_from_iter(bind), |r| r.get::<_, i64>(0))?;
    let mut keep = std::collections::HashSet::new();
    for id in mapped {
        keep.insert(id?);
    }
    let mut seen = std::collections::HashSet::new();
    Ok(raw
        .iter()
        .filter(|(id, _)| keep.contains(id) && seen.insert(*id))
        .take(k)
        .copied()
        .collect())
}

/// Load the EXACT stored embeddings for a set of candidate ids
/// (validity- and model-filtered). The novelty gate compares exact
/// cosine against [`NOVELTY_THRESHOLD`] — quantized index scores are
/// only used to pick WHICH rows to compare, never as the similarity.
/// Keyed by rowid DELIBERATELY: SQLite returns IN-list rows in arbitrary
/// order, and the caller's `raw` is score-ordered — a positional Vec would
/// invite silent misalignment the moment anyone zips them. `is_novel`
/// consumes the values as a bag, so today only the keys' SET matters.
fn load_candidate_embeddings(
    conn: &Connection,
    raw: &[(i64, f32)],
    model: &str,
    valid: &str,
    owner: &str,
) -> Result<std::collections::HashMap<i64, Vec<f32>>> {
    if raw.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let placeholders = vec!["?"; raw.len()].join(",");
    let and_valid = if valid.is_empty() {
        String::new()
    } else {
        format!(" AND {valid}")
    };
    let sql = format!(
        "SELECT id, embedding FROM memories WHERE id IN ({placeholders}) \
         AND embedding IS NOT NULL AND embedding_model = ?{model_pos} \
         AND owner = ?{owner_pos}{and_valid}",
        model_pos = raw.len() + 1,
        owner_pos = raw.len() + 2
    );
    let mut stmt = conn.prepare(&sql)?;
    let bind: Vec<rusqlite::types::Value> = raw
        .iter()
        .map(|(id, _)| rusqlite::types::Value::Integer(*id))
        .chain(std::iter::once(rusqlite::types::Value::Text(model.to_string())))
        .chain(std::iter::once(rusqlite::types::Value::Text(owner.to_string())))
        .collect();
    let mapped = stmt.query_map(rusqlite::params_from_iter(bind), |r| {
        let id: i64 = r.get(0)?;
        let blob: Vec<u8> = r.get(1)?;
        Ok((id, blob_to_vec(&blob)))
    })?;
    let mut out = std::collections::HashMap::new();
    for row in mapped {
        let (id, v) = row?;
        out.insert(id, v);
    }
    Ok(out)
}

// ── Fact-extraction parsing (pure half of consolidation) ───────────

/// Parse MiMo's extraction output into individual facts. One fact per
/// non-empty line, leading bullets/numbering stripped. A line that is
/// exactly `NONE` (case-insensitive) is dropped; if the whole output is
/// NONE / empty, the result is an empty vec. Pure + unit-testable.
///
/// The extraction orchestration itself (prompting MiMo over a transcript)
/// is a frontend concern and lives in lamu-mcp (ADR 0029); only this pure
/// parser is shared.
pub fn parse_extracted_facts(raw: &str) -> Vec<String> {
    let mut facts = Vec::new();
    for line in raw.lines() {
        let mut s = line.trim();
        if s.is_empty() {
            continue;
        }
        // Strip a single leading bullet ("- ", "* ", "• ").
        if let Some(rest) = s
            .strip_prefix("- ")
            .or_else(|| s.strip_prefix("* "))
            .or_else(|| s.strip_prefix("• "))
        {
            s = rest.trim_start();
        } else {
            // Strip leading "N. " or "N) " numbering.
            let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() {
                let after = &s[digits.len()..];
                if let Some(rest) = after.strip_prefix(". ").or_else(|| after.strip_prefix(") ")) {
                    s = rest.trim_start();
                }
            }
        }
        let s = s.trim();
        if s.is_empty() {
            continue;
        }
        if s.eq_ignore_ascii_case("NONE") {
            continue;
        }
        facts.push(s.to_string());
    }
    facts
}

// ── Autocapture (novelty dedup) ─────────────────────────────────────

/// Cosine-similarity threshold at/above which a candidate fact is
/// considered a duplicate of an existing memory and skipped. Chosen high
/// (0.92) so only near-identical restatements are dropped — distinct but
/// related facts still land. Public so lamu-mcp's auto-contradiction
/// judge can exclude near-duplicates from its candidate set.
pub const NOVELTY_THRESHOLD: f32 = 0.92;

/// How many nearest candidates the persistent-index novelty probe asks
/// for (over-fetched ×4 by the index itself). Only the WHICH-rows choice
/// — the novelty decision always runs exact cosine on stored embeddings.
const NOVELTY_PROBE_K: usize = 16;

/// PURE novelty test: a candidate embedding is novel iff its MAX cosine
/// similarity against every `existing` embedding is strictly below
/// `threshold`. Empty `existing` → always novel (true). Reuses
/// [`crate::vector_index::cosine`] (no re-implementation). Unit-testable.
pub(crate) fn is_novel(new_emb: &[f32], existing: &[Vec<f32>], threshold: f32) -> bool {
    !existing
        .iter()
        .any(|e| cosine(new_emb, e) >= threshold)
}

/// Store `text` as a memory only if it is NOVEL relative to what is
/// already remembered, returning the new rowid on insert or `None` when
/// it was skipped as a near-duplicate.
///
/// - Without an embedder: dedup is impossible (no embeddings), so fall
///   back to an unconditional [`remember`] and return `Ok(Some(id))`.
/// - With one: embed `text`, load every embedding-bearing row WHOSE
///   `embedding_model` matches the current embedder (cosine across
///   models is meaningless — ADR 0030), and if [`is_novel`] is false
///   return `Ok(None)`. Otherwise insert the row WITH its embedding +
///   model tag and return `Ok(Some(id))`.
///
/// OWNER SCOPING (ADR 0032): the novelty scan compares ONLY against
/// `owner`'s rows — the same fact independently asserted by two owners
/// is novel for each (cross-owner dedup would leak fact existence).
pub async fn remember_if_novel(
    text: &str,
    kind: &str,
    source: &str,
    owner: &str,
) -> Result<Option<i64>> {
    let Some((emb, ident)) = embed_via_chain(text).await? else {
        // No embedder → can't embed/dedup; store unconditionally.
        return remember(text, kind, source, owner).await.map(Some);
    };

    let arc = memory_db()?;
    let now = now_secs();
    // Dedup only against CURRENTLY-VALID facts. The old SELECT had no
    // valid-time filter, so a re-asserted fact whose near-duplicate had
    // been superseded/forgotten (row still present, just expired) was
    // dropped as a "duplicate" — but default recall hides the expired row,
    // so the now-current fact silently vanished.
    let valid = valid_time_clause(false, now);
    // Hold ONE guard across SELECT + is_novel + insert. is_novel (cosine)
    // and insert_memory are synchronous (no await), so this is safe and
    // closes the TOCTOU window where two concurrent autocapture threads
    // both passed the novelty check against the same pre-insert snapshot
    // and both inserted. Trade-off: the cosine scan now runs under the
    // lock, serializing concurrent novelty checks — fine while memory.db
    // is small + autocapture is bounded; revisit if the store grows large.
    let inserted = {
        let conn = arc.lock();
        // ADR 0031: with the persistent index active, probe it for the
        // nearest candidates instead of scanning every row — the novelty
        // gate then runs EXACT cosine over just those rows' stored
        // embeddings (quantized scores never decide novelty). The probe
        // over-fetches ×4 internally; a ≥-threshold near-duplicate is by
        // definition the nearest neighbor, so top-NOVELTY_PROBE_K cannot
        // realistically miss it. Inactive/unusable → full scan as before.
        let probe =
            crate::tv_store::search_persistent(&conn, TV_STORE, &emb, &ident.model, NOVELTY_PROBE_K);
        let existing: Vec<Vec<f32>> = match probe {
            Some(raw) => load_candidate_embeddings(&conn, &raw, &ident.model, &valid, owner)?
                .into_values()
                .collect(),
            None => {
                let sql = format!(
                    "SELECT embedding FROM memories \
                     WHERE embedding IS NOT NULL AND embedding_model = ?1 \
                     AND owner = ?2 AND {valid}"
                );
                let mut stmt = conn.prepare(&sql)?;
                let mapped = stmt.query_map(params![ident.model, owner], |r| {
                    let blob: Vec<u8> = r.get(0)?;
                    Ok(blob_to_vec(&blob))
                })?;
                let mut rows = Vec::new();
                for row in mapped {
                    rows.push(row?);
                }
                rows
            }
        };

        if !is_novel(&emb, &existing, NOVELTY_THRESHOLD) {
            None
        } else {
            let id = insert_memory(
                &conn, text, Some(&emb), Some(&ident.model), kind, source, now, owner,
            )?;
            crate::store::record_store_identity(&conn, "memories", &ident.model, emb.len(), now);
            crate::tv_store::note_added(&conn, TV_STORE, id, &emb, &ident.model);
            Some(id)
        }
    }; // DB lock released — persist (file I/O) must not run under it.
    crate::tv_store::maybe_persist(TV_STORE);
    Ok(inserted)
}

// ── Supersession + soft-delete (temporal) ──────────────────────────

/// Replace fact `old_id` with a NEW fact (`new_text`): the new fact is
/// inserted with `supersedes = Some(old_id)` and `valid_from = now`, and
/// the old fact is expired (`valid_until = now`). Returns
/// `Ok(Some(new_id))` on success.
///
/// This is the "user moved X → Y" operation: the old fact becomes
/// historical (still in the store, recallable with `include_expired`) and
/// the new fact takes its place in default recall. The old row is only
/// expired if it is CURRENTLY valid (`valid_until IS NULL`) — re-superseding
/// an already-expired fact leaves its earlier `valid_until` intact.
///
/// OWNER SCOPING (ADR 0032): `old_id` must belong to `owner`. A missing
/// id and ANOTHER owner's id both return `Ok(None)` with NOTHING
/// inserted — indistinguishable on purpose (no existence leak).
///
/// Embeds `new_text` exactly like [`remember`] (NULL embedding when no
/// embedder resolves), so the new fact is semantically recallable.
pub async fn supersede(
    old_id: i64,
    new_text: &str,
    kind: &str,
    source: &str,
    owner: &str,
) -> Result<Option<i64>> {
    let embedded = embed_via_chain(new_text).await?;
    let now = now_secs();
    let arc = memory_db()?;
    let id = {
        let mut conn = arc.lock();
        // Will this call actually expire an INDEXED row? Checked under the
        // same guard as the supersede itself, so it's race-free. Drives
        // the stale-count bump below (ADR 0031): only rows that carry an
        // embedding ever entered the .tv.
        let old_indexed: bool = {
            use rusqlite::OptionalExtension;
            conn.query_row(
                "SELECT embedding IS NOT NULL FROM memories \
                 WHERE id = ?1 AND owner = ?2 AND valid_until IS NULL",
                params![old_id, owner],
                |r| r.get(0),
            )
            .optional()?
            .unwrap_or(false)
        };
        let id = supersede_conn(
            &mut conn,
            old_id,
            new_text,
            embedded.as_ref().map(|(v, _)| v.as_slice()),
            embedded.as_ref().map(|(_, ident)| ident.model.as_str()),
            kind,
            source,
            now,
            owner,
        )?;
        if let Some(id) = id {
            if let Some((v, ident)) = &embedded {
                crate::store::record_store_identity(&conn, "memories", &ident.model, v.len(), now);
                crate::tv_store::note_added(&conn, TV_STORE, id, v, &ident.model);
            }
            if old_indexed {
                // Invalidation WITHOUT delete: the expired row's vector stays
                // in the .tv; searches post-filter it out, and the stale
                // counter drives the >25% rebuild threshold.
                crate::tv_store::note_stale(&conn, TV_STORE, 1);
            }
        }
        id
    }; // DB lock released — persist (file I/O) must not run under it.
    crate::tv_store::maybe_persist(TV_STORE);
    Ok(id)
}

/// Connection-level core of [`supersede`]: verify `old_id` belongs to
/// `owner`, insert the new fact with `supersedes = Some(old_id)` /
/// `valid_from = now`, then expire the old fact if it is currently
/// valid. Returns `Ok(None)` — with NOTHING inserted — when no row with
/// (`old_id`, `owner`) exists, which covers both a genuinely missing id
/// and another owner's id (deliberately indistinguishable, ADR 0032).
/// Factored out so tests can drive it against an in-memory connection
/// with a known embedding and `now`.
///
/// ATOMICITY: the ownership check, INSERT (new fact) and UPDATE (expire
/// old fact) run in a single SQLite transaction. Without it, a crash or
/// error between the statements would leave the new fact inserted while
/// the old fact is still `valid_until IS NULL` — both then appear in
/// default recall, violating the "exactly one valid version" invariant
/// supersession exists to enforce.
#[allow(clippy::too_many_arguments)]
pub(crate) fn supersede_conn(
    conn: &mut Connection,
    old_id: i64,
    new_text: &str,
    embedding: Option<&[f32]>,
    embedding_model: Option<&str>,
    kind: &str,
    source: &str,
    now: i64,
    owner: &str,
) -> Result<Option<i64>> {
    let tx = conn.transaction()?;
    // Ownership gate: the old row must be `owner`'s. Validity is NOT
    // required here — re-superseding an owned-but-already-expired fact
    // keeps its earlier `valid_until` (pre-0032 behavior, unchanged).
    let owned: i64 = tx.query_row(
        "SELECT COUNT(*) FROM memories WHERE id = ?1 AND owner = ?2",
        params![old_id, owner],
        |r| r.get(0),
    )?;
    if owned == 0 {
        return Ok(None);
    }
    let new_id = insert_memory_full(
        &tx,
        new_text,
        embedding,
        embedding_model,
        kind,
        source,
        now,
        now,
        Some(old_id),
        owner,
    )?;
    tx.execute(
        "UPDATE memories SET valid_until = ? WHERE id = ? AND owner = ? AND valid_until IS NULL",
        params![now, old_id, owner],
    )?;
    tx.commit()?;
    Ok(Some(new_id))
}

/// Soft-delete fact `id`: set `valid_until = now` so it drops out of
/// default recall but remains in the store (recoverable, and the timeline
/// survives). Returns `true` if a currently-valid row with that id was
/// expired, `false` if no such row existed (already expired or absent).
///
/// OWNER SCOPING (ADR 0032): the row must belong to `owner`; another
/// owner's id returns `false` — the same response as a missing id (no
/// existence leak).
///
/// No fact is ever hard-deleted; `forget` only moves a fact into history.
pub fn forget(id: i64, owner: &str) -> Result<bool> {
    let now = now_secs();
    let arc = memory_db()?;
    let conn = arc.lock();
    // Indexed = currently valid AND embedding-bearing — only those rows
    // ever entered the .tv, so only they count toward stale (ADR 0031).
    let was_indexed: bool = {
        use rusqlite::OptionalExtension;
        conn.query_row(
            "SELECT embedding IS NOT NULL FROM memories \
             WHERE id = ?1 AND owner = ?2 AND valid_until IS NULL",
            params![id, owner],
            |r| r.get(0),
        )
        .optional()?
        .unwrap_or(false)
    };
    let affected = forget_conn(&conn, id, now, owner)?;
    if affected && was_indexed {
        crate::tv_store::note_stale(&conn, TV_STORE, 1);
    }
    Ok(affected)
}

/// Connection-level core of [`forget`]: expire the row if it is currently
/// valid AND belongs to `owner`, returning whether a row was affected.
/// Factored out for testing against an in-memory connection with a known
/// `now`.
pub(crate) fn forget_conn(conn: &Connection, id: i64, now: i64, owner: &str) -> Result<bool> {
    let affected = conn.execute(
        "UPDATE memories SET valid_until = ? WHERE id = ? AND owner = ? AND valid_until IS NULL",
        params![now, id, owner],
    )?;
    Ok(affected > 0)
}

// ── graphify corpus exporter ───────────────────────────────────────

/// Sanitize a frontmatter scalar for single-line YAML: drop newlines and
/// trailing whitespace so the `---` block stays well-formed, and
/// double-quote the value when it would otherwise be misread by a YAML
/// parser (empty, a YAML-special bare word like `null`/`true`/`yes`/`no`,
/// or containing `:` / `#` / a leading quote). Plain words (the common
/// case — `fact`, `preference`, a `[A-Za-z0-9_-.]` source) pass through
/// unquoted. The body (fact text) is written as-is below the block.
fn yaml_scalar(s: &str) -> String {
    let s = s.replace(['\n', '\r'], " ");
    let s = s.trim();
    let needs_quote = s.is_empty()
        || s.contains(':')
        || s.contains('#')
        || s.starts_with(['"', '\'', '[', '{', '*', '&', '!', '|', '>', '@', '`'])
        // Numeric-looking bare scalars (e.g. a kind of "123" or "3.14")
        // would be read as a YAML number, not a string — quote them too.
        || s.parse::<i64>().is_ok()
        || s.parse::<f64>().is_ok()
        || matches!(
            s.to_ascii_lowercase().as_str(),
            "null" | "~" | "true" | "false" | "yes" | "no" | "on" | "off"
        );
    if needs_quote {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        s.to_string()
    }
}

/// Export the fact store as a graphify-ready corpus: one markdown file per
/// memory at `<dir>/mem_<id>.md`, each led by a YAML frontmatter block and
/// followed by the fact text. Creates `dir` if missing. Returns the count
/// of files written.
///
/// By default (`include_expired = false`) only currently-valid facts are
/// exported; `include_expired = true` exports every fact (so the graph
/// shows the full timeline, including superseded/forgotten facts).
///
/// lamu does NOT extract entities, edges, hyperedges, or communities — it
/// only emits the corpus. The user then runs `/graphify <dir>` (or
/// `graphify <dir>`); graphify's LLM extraction + clustering pipeline pulls
/// entities/hyperedges/communities from these files.
///
/// OWNER SCOPING (ADR 0032): only `owner`'s facts are exported. The MCP
/// `export_memory_graph` handler passes [`LOCAL_OWNER`], so the graphify
/// corpus never carries HTTP tenants' facts.
pub fn export_graph_corpus(dir: &Path, include_expired: bool, owner: &str) -> Result<usize> {
    let arc = memory_db()?;
    let now = now_secs();
    // Load all rows under the lock, then release before doing filesystem
    // writes — same don't-hold-the-mutex-across-I/O discipline as recall.
    let rows = {
        let conn = arc.lock();
        load_corpus_rows(&conn, include_expired, now, owner)?
    };
    write_corpus_rows(dir, &rows)
}

/// One row loaded for corpus export.
pub(crate) struct CorpusRow {
    pub id: i64,
    pub text: String,
    pub kind: String,
    pub source: Option<String>,
    pub ts: i64,
    pub valid_from: i64,
    pub valid_until: Option<i64>,
    pub supersedes: Option<i64>,
}

/// Load the rows to export (currently-valid only unless `include_expired`;
/// `owner`'s rows only), ordered by id. Connection-level so tests drive it
/// without the singleton.
pub(crate) fn load_corpus_rows(
    conn: &Connection,
    include_expired: bool,
    now: i64,
    owner: &str,
) -> Result<Vec<CorpusRow>> {
    let valid = valid_time_clause(include_expired, now);
    let where_clause = if valid.is_empty() {
        "WHERE owner = ?1 ".to_string()
    } else {
        format!("WHERE owner = ?1 AND {valid} ")
    };
    let sql = format!(
        "SELECT id, text, kind, source, ts, valid_from, valid_until, supersedes \
         FROM memories {where_clause}ORDER BY id ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mapped = stmt.query_map(params![owner], |r| {
        Ok(CorpusRow {
            id: r.get(0)?,
            text: r.get(1)?,
            kind: r.get(2)?,
            source: r.get(3)?,
            ts: r.get(4)?,
            valid_from: r.get(5)?,
            valid_until: r.get(6)?,
            supersedes: r.get(7)?,
        })
    })?;
    let mut rows = Vec::new();
    for row in mapped {
        rows.push(row?);
    }
    Ok(rows)
}

/// Write each row as `<dir>/mem_<id>.md` with graphify-honored YAML
/// frontmatter + the fact text as the body. Creates `dir` if missing.
/// Returns the count written. Connection-free so it is directly testable.
pub(crate) fn write_corpus_rows(dir: &Path, rows: &[CorpusRow]) -> Result<usize> {
    use std::io::Write;

    std::fs::create_dir_all(dir)
        .with_context(|| format!("create corpus dir {}", dir.display()))?;

    // Prune stale `mem_<id>.md` from a prior export: forgotten/expired
    // facts excluded from `rows` would otherwise linger on disk and get
    // re-ingested by graphify (the export dir is reused across runs).
    // Only our own `mem_<digits>.md` files are touched — foreign files in
    // the dir are left alone.
    let keep: std::collections::HashSet<String> =
        rows.iter().map(|r| format!("mem_{}.md", r.id)).collect();
    // Best-effort prune: failures are logged, not fatal — exporting the
    // current facts (below) is the load-bearing part. But DO log, because a
    // silent failure means a forgotten fact lingers and #6 re-surfaces.
    match std::fs::read_dir(dir) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let fname = entry.file_name();
                let Some(name) = fname.to_str() else { continue };
                // Only our own mem_<digits>.md (ids are SQLite rowids).
                let is_ours = name
                    .strip_prefix("mem_")
                    .and_then(|s| s.strip_suffix(".md"))
                    .map_or(false, |d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()));
                if is_ours && !keep.contains(name) {
                    if let Err(e) = std::fs::remove_file(entry.path()) {
                        tracing::warn!(
                            "prune stale corpus file {}: {e}",
                            entry.path().display()
                        );
                    }
                }
            }
        }
        Err(e) => tracing::warn!("prune: read_dir {} failed: {e}", dir.display()),
    }

    let mut written = 0usize;
    for row in rows {
        let path = dir.join(format!("mem_{}.md", row.id));
        let valid_until = match row.valid_until {
            Some(v) => v.to_string(),
            None => "current".to_string(),
        };
        let supersedes = match row.supersedes {
            Some(s) => s.to_string(),
            None => String::new(),
        };
        let source = yaml_scalar(row.source.as_deref().unwrap_or(""));
        let content = format!(
            "---\n\
             id: {}\n\
             kind: {}\n\
             source: {}\n\
             ts: {}\n\
             valid_from: {}\n\
             valid_until: {}\n\
             supersedes: {}\n\
             ---\n\
             {}\n",
            row.id,
            yaml_scalar(&row.kind),
            source,
            row.ts,
            row.valid_from,
            valid_until,
            supersedes,
            row.text,
        );
        let mut f = std::fs::File::create(&path)
            .with_context(|| format!("create {}", path.display()))?;
        f.write_all(content.as_bytes())
            .with_context(|| format!("write {}", path.display()))?;
        written += 1;
    }
    Ok(written)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory db on the UNIFIED schema (ADR 0028) — what every
    /// storage fn now runs against. Legacy-shape fixtures (for the
    /// temporal migration) build [`OLD_SCHEMA`] explicitly instead.
    fn open_test_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        crate::migrate::migrate(&mut conn).unwrap();
        conn
    }

    #[test]
    fn insert_and_rank_with_known_embeddings() {
        let conn = open_test_db();
        // Hand-crafted 3-dim embeddings.
        let id_x = insert_memory(&conn, "x-axis fact", Some(&[1.0, 0.0, 0.0]), Some("test-model"), "fact", "manual", 100, "local")
            .unwrap();
        let id_y = insert_memory(&conn, "y-axis fact", Some(&[0.0, 1.0, 0.0]), Some("test-model"), "fact", "manual", 200, "local")
            .unwrap();
        let id_near =
            insert_memory(&conn, "near-x fact", Some(&[0.9, 0.1, 0.0]), Some("test-model"), "fact", "manual", 300, "local")
                .unwrap();
        assert_eq!(id_x, 1);
        assert_eq!(id_y, 2);
        assert_eq!(id_near, 3);

        // Round-trip rows out of the storage (verifies blob persistence).
        let mut stmt = conn
            .prepare("SELECT id, text, kind, source, ts, embedding FROM memories")
            .unwrap();
        let mapped = stmt
            .query_map([], |r| {
                let id: i64 = r.get(0)?;
                let text: String = r.get(1)?;
                let kind: String = r.get(2)?;
                let source: Option<String> = r.get(3)?;
                let ts: i64 = r.get(4)?;
                let emb: Vec<u8> = r.get(5)?;
                Ok((id, text, kind, source, ts, blob_to_vec(&emb)))
            })
            .unwrap();
        let rows: Vec<_> = mapped.map(|r| r.unwrap()).collect();
        assert_eq!(rows.len(), 3);

        // Query close to the x-axis: expect "x-axis fact" first, then
        // "near-x fact", then "y-axis fact".
        let query = [1.0, 0.0, 0.0];
        let hits = rank_memories(&query, rows.clone(), 3);
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, id_x);
        assert_eq!(hits[1].id, id_near);
        assert_eq!(hits[2].id, id_y);
        // Scores descending.
        assert!(hits[0].score.unwrap() >= hits[1].score.unwrap());
        assert!(hits[1].score.unwrap() >= hits[2].score.unwrap());
        // Every hit carries a Some(score) in the ranked path.
        assert!(hits.iter().all(|h| h.score.is_some()));

        // top-k cap honored.
        let top1 = rank_memories(&query, rows, 1);
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].id, id_x);
    }

    #[test]
    fn null_embedding_round_trip() {
        let conn = open_test_db();
        let id = insert_memory(&conn, "no-embedding fact", None, None, "fact", "manual", 42, "local").unwrap();
        let emb: Option<Vec<u8>> = conn
            .query_row("SELECT embedding FROM memories WHERE id = ?", params![id], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(emb.is_none(), "embedding column should be NULL when no key");
    }

    #[test]
    fn parse_facts_strips_bullets_and_numbering() {
        let raw = "- prefers terse responses\n* uses CachyOS\n1. building LamQuant\n2) ADHD";
        let facts = parse_extracted_facts(raw);
        assert_eq!(
            facts,
            vec![
                "prefers terse responses".to_string(),
                "uses CachyOS".to_string(),
                "building LamQuant".to_string(),
                "ADHD".to_string(),
            ]
        );
    }

    #[test]
    fn parse_facts_none_yields_empty() {
        assert!(parse_extracted_facts("NONE").is_empty());
        assert!(parse_extracted_facts("none").is_empty());
        assert!(parse_extracted_facts("  None  ").is_empty());
        assert!(parse_extracted_facts("").is_empty());
        assert!(parse_extracted_facts("\n\n   \n").is_empty());
    }

    #[test]
    fn parse_facts_drops_blank_lines_and_splits_multi() {
        let raw = "fact one\n\n   \nfact two\n";
        let facts = parse_extracted_facts(raw);
        assert_eq!(facts, vec!["fact one".to_string(), "fact two".to_string()]);
    }

    #[test]
    fn is_novel_empty_existing_is_always_novel() {
        // Nothing to compare against → novel regardless of the candidate.
        assert!(is_novel(&[1.0, 0.0, 0.0], &[], NOVELTY_THRESHOLD));
    }

    #[test]
    fn is_novel_near_duplicate_is_not_novel() {
        let existing = vec![vec![1.0, 0.0, 0.0]];
        // cosine([1,0,0],[1,0,0]) == 1.0 >= 0.92 → duplicate.
        assert!(!is_novel(&[1.0, 0.0, 0.0], &existing, NOVELTY_THRESHOLD));
        // A vector only slightly off-axis: cosine ≈ 0.9999 >= 0.92 → dup.
        assert!(!is_novel(&[0.999, 0.01, 0.0], &existing, NOVELTY_THRESHOLD));
    }

    #[test]
    fn is_novel_dissimilar_is_novel() {
        let existing = vec![vec![1.0, 0.0, 0.0]];
        // Orthogonal → cosine 0.0 < 0.92 → novel.
        assert!(is_novel(&[0.0, 1.0, 0.0], &existing, NOVELTY_THRESHOLD));
    }

    #[test]
    fn is_novel_max_over_many_existing() {
        // Novel against several dissimilar rows, but NOT novel once a
        // near-duplicate is present — the MAX similarity is what gates.
        let dissimilar = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        assert!(is_novel(&[1.0, 0.0, 0.0], &dissimilar, NOVELTY_THRESHOLD));
        let with_dup = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
        assert!(!is_novel(&[1.0, 0.0, 0.0], &with_dup, NOVELTY_THRESHOLD));
    }

    #[test]
    fn parse_facts_none_among_real_facts_is_dropped() {
        let raw = "- real fact\nNONE\n- another fact";
        let facts = parse_extracted_facts(raw);
        assert_eq!(
            facts,
            vec!["real fact".to_string(), "another fact".to_string()]
        );
    }

    // ── Temporal (valid-time) tests ────────────────────────────────

    /// Old pre-temporal schema: the `memories` table WITHOUT the three
    /// valid-time columns + WITHOUT the valid index. Used to drive the
    /// migration.
    const OLD_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding BLOB,
    kind TEXT NOT NULL DEFAULT 'fact',
    source TEXT,
    ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories(ts);
";

    fn column_names(conn: &Connection) -> std::collections::HashSet<String> {
        let mut stmt = conn.prepare("PRAGMA table_info(memories)").unwrap();
        let cols = stmt.query_map([], |r| r.get::<_, String>(1)).unwrap();
        cols.map(|c| c.unwrap()).collect()
    }

    #[test]
    fn migration_adds_columns_backfills_and_is_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(OLD_SCHEMA).unwrap();

        // Old schema lacks all three temporal columns.
        let before = column_names(&conn);
        assert!(!before.contains("valid_from"));
        assert!(!before.contains("valid_until"));
        assert!(!before.contains("supersedes"));

        // Seed a pre-migration row (only old columns exist).
        conn.execute(
            "INSERT INTO memories (text, embedding, kind, source, ts) VALUES (?, ?, ?, ?, ?)",
            params!["old fact", Option::<Vec<u8>>::None, "fact", "manual", 12345i64],
        )
        .unwrap();

        // Run the migration.
        migrate_temporal_columns(&conn).unwrap();

        // All three columns now present.
        let after = column_names(&conn);
        assert!(after.contains("valid_from"));
        assert!(after.contains("valid_until"));
        assert!(after.contains("supersedes"));

        // Backfill: the pre-migration row's valid_from is now its ts (not 0).
        let (valid_from, valid_until, supersedes): (i64, Option<i64>, Option<i64>) = conn
            .query_row(
                "SELECT valid_from, valid_until, supersedes FROM memories WHERE text = 'old fact'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(valid_from, 12345);
        assert!(valid_until.is_none(), "backfilled row must stay currently-valid");
        assert!(supersedes.is_none());

        // A second migration run is a no-op (no error, columns unchanged).
        migrate_temporal_columns(&conn).unwrap();
        let after2 = column_names(&conn);
        assert_eq!(after, after2);
        // valid_from must NOT be re-touched (it is no longer 0, so the
        // backfill UPDATE matches nothing).
        let valid_from2: i64 = conn
            .query_row(
                "SELECT valid_from FROM memories WHERE text = 'old fact'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(valid_from2, 12345);
    }

    #[test]
    fn migration_idempotent_on_fresh_schema() {
        // A db already on the new SCHEMA: migration must be a clean no-op.
        let conn = open_test_db();
        migrate_temporal_columns(&conn).unwrap();
        let cols = column_names(&conn);
        assert!(cols.contains("valid_from"));
        assert!(cols.contains("valid_until"));
        assert!(cols.contains("supersedes"));
    }

    #[test]
    fn valid_time_clause_shape() {
        // include_expired = true drops the filter entirely.
        assert_eq!(valid_time_clause(true, 1000), "");
        // include_expired = false produces the NULL-or-future filter.
        assert_eq!(
            valid_time_clause(false, 1000),
            "(valid_until IS NULL OR valid_until > 1000)"
        );
    }

    /// Helper: load (id, text) for currently-valid / all rows via the same
    /// recency-path SELECT shape recall_memory's no-key branch builds.
    fn load_default(conn: &Connection, include_expired: bool, now: i64) -> Vec<(i64, String)> {
        let valid = valid_time_clause(include_expired, now);
        let where_clause = if valid.is_empty() {
            String::new()
        } else {
            format!("WHERE {valid} ")
        };
        let sql = format!(
            "SELECT id, text FROM memories {where_clause}ORDER BY ts DESC, id DESC"
        );
        let mut stmt = conn.prepare(&sql).unwrap();
        let mapped = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
            .unwrap();
        mapped.map(|r| r.unwrap()).collect()
    }

    #[test]
    fn valid_time_recall_hides_expired_by_default() {
        let conn = open_test_db();
        // Current fact (valid_until NULL via insert_memory).
        let id_current =
            insert_memory(&conn, "current fact", None, None, "fact", "manual", 100, "local").unwrap();
        // Expired fact: insert then expire it with valid_until in the past.
        let id_expired =
            insert_memory(&conn, "expired fact", None, None, "fact", "manual", 50, "local").unwrap();
        conn.execute(
            "UPDATE memories SET valid_until = ? WHERE id = ?",
            params![60i64, id_expired],
        )
        .unwrap();

        // `now` = 1000 is well past valid_until=60, so the expired row is
        // filtered by default but the current row (valid_until NULL) stays.
        let now = 1000;
        let default_rows = load_default(&conn, false, now);
        assert_eq!(default_rows.len(), 1, "default load hides expired");
        assert_eq!(default_rows[0].0, id_current);

        // include_expired = true returns BOTH.
        let all_rows = load_default(&conn, true, now);
        assert_eq!(all_rows.len(), 2, "include_expired returns all");
        let ids: std::collections::HashSet<i64> = all_rows.iter().map(|r| r.0).collect();
        assert!(ids.contains(&id_current));
        assert!(ids.contains(&id_expired));
    }

    #[test]
    fn supersede_expires_old_and_links_new() {
        let mut conn = open_test_db();
        let old_id = insert_memory(&conn, "lives in SF", None, None, "fact", "manual", 100, "local").unwrap();

        // Supersede at now = 500.
        let new_id =
            supersede_conn(&mut conn, old_id, "lives in NYC", None, None, "fact", "manual", 500, "local")
                .unwrap()
                .expect("old_id is owned — supersede must insert");
        assert_ne!(old_id, new_id);

        // Old row: valid_until now set to 500.
        let old_valid_until: Option<i64> = conn
            .query_row(
                "SELECT valid_until FROM memories WHERE id = ?",
                params![old_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(old_valid_until, Some(500));

        // New row: supersedes = old_id, valid_until NULL (currently valid),
        // valid_from = now.
        let (supersedes, valid_until, valid_from): (Option<i64>, Option<i64>, i64) = conn
            .query_row(
                "SELECT supersedes, valid_until, valid_from FROM memories WHERE id = ?",
                params![new_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(supersedes, Some(old_id));
        assert!(valid_until.is_none());
        assert_eq!(valid_from, 500);

        // INVARIANT (atomic supersede): after the transaction commits there
        // is EXACTLY ONE currently-valid row — the new fact. The old fact is
        // expired in the same transaction, so default recall never sees both.
        let valid_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE valid_until IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(valid_count, 1, "exactly one valid version after supersede");
    }

    #[test]
    fn yaml_scalar_quotes_special_values() {
        // Plain words pass through unquoted (the common case).
        assert_eq!(yaml_scalar("fact"), "fact");
        assert_eq!(yaml_scalar("sess-1_2.3"), "sess-1_2.3");
        // YAML-special bare words get quoted so they aren't read as bool/null.
        assert_eq!(yaml_scalar("null"), "\"null\"");
        assert_eq!(yaml_scalar("true"), "\"true\"");
        assert_eq!(yaml_scalar("Yes"), "\"Yes\"");
        // Values with structural chars get quoted.
        assert_eq!(yaml_scalar("a: b"), "\"a: b\"");
        assert_eq!(yaml_scalar("#tag"), "\"#tag\"");
        // Empty gets quoted (empty bare scalar is null in YAML).
        assert_eq!(yaml_scalar(""), "\"\"");
        // Numeric-looking values get quoted so they stay strings.
        assert_eq!(yaml_scalar("123"), "\"123\"");
        assert_eq!(yaml_scalar("3.14"), "\"3.14\"");
        // Newlines collapsed to a space before the quote decision.
        assert_eq!(yaml_scalar("line\nbreak"), "line break");
    }

    #[test]
    fn forget_soft_deletes_then_is_idempotent_noop() {
        let conn = open_test_db();
        let id = insert_memory(&conn, "transient", None, None, "fact", "manual", 100, "local").unwrap();

        // First forget: row is currently valid → affected → true.
        assert!(forget_conn(&conn, id, 777, "local").unwrap());
        let valid_until: Option<i64> = conn
            .query_row(
                "SELECT valid_until FROM memories WHERE id = ?",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(valid_until, Some(777));

        // Row is NOT hard-deleted — it still exists.
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories WHERE id = ?", params![id], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, 1, "forget must not hard-delete");

        // Second forget of the same id: already expired → false, and
        // valid_until is unchanged.
        assert!(!forget_conn(&conn, id, 888, "local").unwrap());
        let valid_until2: Option<i64> = conn
            .query_row(
                "SELECT valid_until FROM memories WHERE id = ?",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(valid_until2, Some(777), "re-forget must not overwrite");
    }

    #[test]
    fn export_graph_corpus_writes_frontmatter_and_respects_include_expired() {
        let conn = open_test_db();
        let id_current =
            insert_memory(&conn, "current fact", None, None, "preference", "manual", 100, "local").unwrap();
        let id_expired =
            insert_memory(&conn, "expired fact", None, None, "fact", "sess-1", 50, "local").unwrap();
        conn.execute(
            "UPDATE memories SET valid_until = ? WHERE id = ?",
            params![60i64, id_expired],
        )
        .unwrap();

        let tmp = tempfile::tempdir().unwrap();

        // Default: only the current fact is exported.
        let now = 1000;
        let rows = load_corpus_rows(&conn, false, now, "local").unwrap();
        let n = write_corpus_rows(tmp.path(), &rows).unwrap();
        assert_eq!(n, 1, "default export = currently-valid only");

        let current_path = tmp.path().join(format!("mem_{id_current}.md"));
        assert!(current_path.exists());
        let body = std::fs::read_to_string(&current_path).unwrap();
        // Frontmatter fields present.
        assert!(body.starts_with("---\n"));
        assert!(body.contains(&format!("id: {id_current}")));
        assert!(body.contains("kind: preference"));
        assert!(body.contains("source: manual"));
        assert!(body.contains("ts: 100"));
        assert!(body.contains("valid_from: 100"));
        assert!(body.contains("valid_until: current"));
        assert!(body.contains("supersedes: "));
        // Body is the fact text.
        assert!(body.contains("current fact"));
        // The expired fact was NOT exported by default.
        assert!(!tmp.path().join(format!("mem_{id_expired}.md")).exists());

        // include_expired = true: BOTH files written, expired carries its
        // numeric valid_until.
        let tmp2 = tempfile::tempdir().unwrap();
        let rows_all = load_corpus_rows(&conn, true, now, "local").unwrap();
        let n2 = write_corpus_rows(tmp2.path(), &rows_all).unwrap();
        assert_eq!(n2, 2, "include_expired exports all");
        let expired_body =
            std::fs::read_to_string(tmp2.path().join(format!("mem_{id_expired}.md"))).unwrap();
        assert!(expired_body.contains("valid_until: 60"));
        assert!(expired_body.contains("source: sess-1"));
    }

    #[test]
    fn export_graph_corpus_creates_missing_dir() {
        let conn = open_test_db();
        insert_memory(&conn, "a fact", None, None, "fact", "manual", 100, "local").unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("does").join("not").join("exist");
        let rows = load_corpus_rows(&conn, false, 1000, "local").unwrap();
        let n = write_corpus_rows(&nested, &rows).unwrap();
        assert_eq!(n, 1);
        assert!(nested.join("mem_1.md").exists());
    }

    #[test]
    fn export_prunes_stale_files_but_keeps_foreign() {
        let conn = open_test_db();
        let id1 = insert_memory(&conn, "fact one", None, None, "fact", "manual", 100, "local").unwrap();
        let id2 = insert_memory(&conn, "fact two", None, None, "fact", "manual", 100, "local").unwrap();
        let id3 = insert_memory(&conn, "fact three", None, None, "fact", "manual", 100, "local").unwrap();
        let tmp = tempfile::tempdir().unwrap();

        // First export: all three written.
        let rows = load_corpus_rows(&conn, false, 1000, "local").unwrap();
        assert_eq!(write_corpus_rows(tmp.path(), &rows).unwrap(), 3);
        // A foreign file the prune must never touch.
        std::fs::write(tmp.path().join("README.md"), b"keep me").unwrap();

        // Forget id2, then re-export to the SAME dir.
        conn.execute(
            "UPDATE memories SET valid_until = ? WHERE id = ?",
            params![500i64, id2],
        )
        .unwrap();
        let rows2 = load_corpus_rows(&conn, false, 1000, "local").unwrap();
        assert_eq!(write_corpus_rows(tmp.path(), &rows2).unwrap(), 2);

        assert!(tmp.path().join(format!("mem_{id1}.md")).exists());
        assert!(
            !tmp.path().join(format!("mem_{id2}.md")).exists(),
            "forgotten fact's stale file must be pruned"
        );
        assert!(tmp.path().join(format!("mem_{id3}.md")).exists());
        assert!(
            tmp.path().join("README.md").exists(),
            "foreign file must be left alone"
        );
    }

    // ── ADR 0030: identity enforcement + hybrid recall ─────────────

    /// Tempdir lamu.db with the FULL unified schema — the FTS leg needs
    /// the real `memories_fts` virtual table + triggers (which the
    /// migrations create); file-backed per the e2e spec.
    fn open_tmp_lamu_db() -> (tempfile::TempDir, Connection) {
        let td = tempfile::tempdir().unwrap();
        let conn = crate::store::open_at(&td.path().join("lamu.db")).unwrap();
        (td, conn)
    }

    #[test]
    fn vector_recall_ranks_only_current_model_rows() {
        let (_td, conn) = open_tmp_lamu_db();
        // Two rows under model A, two under model B — all near the same
        // query vector so ANY of them would rank if not filtered.
        let a1 = insert_memory(&conn, "alpha one", Some(&[1.0, 0.0]), Some("model-a"), "fact", "manual", 10, "local").unwrap();
        let a2 = insert_memory(&conn, "alpha two", Some(&[0.9, 0.1]), Some("model-a"), "fact", "manual", 20, "local").unwrap();
        let b1 = insert_memory(&conn, "bravo one", Some(&[1.0, 0.0]), Some("model-b"), "fact", "manual", 30, "local").unwrap();
        let b2 = insert_memory(&conn, "bravo two", Some(&[0.8, 0.2]), Some("model-b"), "fact", "manual", 40, "local").unwrap();

        // Query text shares no tokens with any fact → the FTS leg is
        // empty and the result isolates the vector leg.
        let hits = recall_hybrid_conn(
            &conn,
            "zzzqqq",
            Some(&[1.0, 0.0]),
            Some("model-b"),
            10,
            false,
            1000,
            "local",
        )
        .unwrap();
        let ids: std::collections::HashSet<i64> = hits.iter().map(|h| h.id).collect();
        assert!(ids.contains(&b1) && ids.contains(&b2), "both B rows rank");
        assert!(
            !ids.contains(&a1) && !ids.contains(&a2),
            "A rows must not rank under model B"
        );
        // Vector-leg hits carry their cosine score.
        assert!(hits.iter().all(|h| h.score.is_some()));

        // Switch identity to model A → only A rows rank.
        let hits_a = recall_hybrid_conn(
            &conn,
            "zzzqqq",
            Some(&[1.0, 0.0]),
            Some("model-a"),
            10,
            false,
            1000,
            "local",
        )
        .unwrap();
        let ids_a: std::collections::HashSet<i64> = hits_a.iter().map(|h| h.id).collect();
        assert_eq!(ids_a, [a1, a2].into_iter().collect());
    }

    #[test]
    fn hybrid_recall_surfaces_lexical_and_semantic_hits() {
        let (_td, conn) = open_tmp_lamu_db();
        // Lexical fact: matches the query text via FTS but its vector is
        // orthogonal to the query embedding.
        let lex = insert_memory(
            &conn,
            "the zanzibar deployment protocol",
            Some(&[0.0, 1.0, 0.0]),
            Some("fake"),
            "fact",
            "manual",
            10,
            "local",
        )
        .unwrap();
        // Semantic fact: no token overlap with the query, but its vector
        // is what the (fake) query embedding points at.
        let sem = insert_memory(
            &conn,
            "ship the island rollout plan",
            Some(&[1.0, 0.0, 0.0]),
            Some("fake"),
            "fact",
            "manual",
            20,
            "local",
        )
        .unwrap();
        // Noise that matches neither leg well.
        insert_memory(&conn, "unrelated grocery list", Some(&[0.0, 0.0, 1.0]), Some("fake"), "fact", "manual", 30, "local").unwrap();

        let hits = recall_hybrid_conn(
            &conn,
            "zanzibar",             // FTS finds `lex`
            Some(&[1.0, 0.0, 0.0]), // cosine ranks `sem` first
            Some("fake"),
            2,
            false,
            1000,
            "local",
        )
        .unwrap();
        let ids: std::collections::HashSet<i64> = hits.iter().map(|h| h.id).collect();
        assert!(ids.contains(&lex), "lexical (FTS) hit must surface");
        assert!(ids.contains(&sem), "semantic (vector) hit must surface");
        // Score semantics preserved: vector-scored hits carry cosine.
        let sem_hit = hits.iter().find(|h| h.id == sem).unwrap();
        assert!(sem_hit.score.unwrap() > 0.99, "cosine of identical vectors ≈ 1");
    }

    #[test]
    fn hybrid_recall_without_embedder_is_fts_plus_recency() {
        let (_td, conn) = open_tmp_lamu_db();
        // Old lexical match vs newer unrelated facts. Pre-0030 the
        // no-key fallback was recency-only and would bury the match.
        let lex = insert_memory(&conn, "the zanzibar deployment protocol", None, None, "fact", "manual", 10, "local").unwrap();
        for i in 0..5 {
            insert_memory(&conn, &format!("filler fact {i}"), None, None, "fact", "manual", 100 + i, "local").unwrap();
        }
        let hits = recall_hybrid_conn(&conn, "zanzibar", None, None, 3, false, 1000, "local").unwrap();
        assert!(
            hits.iter().any(|h| h.id == lex),
            "FTS leg must surface the lexical match even without an embedder"
        );
        // Degraded path: no cosine scores anywhere.
        assert!(hits.iter().all(|h| h.score.is_none()));
    }

    #[test]
    fn hybrid_recall_hides_expired_in_both_legs() {
        let (_td, conn) = open_tmp_lamu_db();
        let id = insert_memory(&conn, "zanzibar expired fact", Some(&[1.0, 0.0]), Some("fake"), "fact", "manual", 10, "local").unwrap();
        conn.execute("UPDATE memories SET valid_until = 50 WHERE id = ?", params![id]).unwrap();
        let hits =
            recall_hybrid_conn(&conn, "zanzibar", Some(&[1.0, 0.0]), Some("fake"), 5, false, 1000, "local")
                .unwrap();
        assert!(hits.is_empty(), "expired fact hidden from both legs");
        let hits_all =
            recall_hybrid_conn(&conn, "zanzibar", Some(&[1.0, 0.0]), Some("fake"), 5, true, 1000, "local")
                .unwrap();
        assert_eq!(hits_all.len(), 1, "include_expired surfaces it");
    }

    #[test]
    fn hybrid_recall_query_with_fts_operators_does_not_error() {
        let (_td, conn) = open_tmp_lamu_db();
        insert_memory(&conn, "plain fact", None, None, "fact", "manual", 10, "local").unwrap();
        // Raw quotes/operators would be FTS5 syntax errors un-sanitized.
        for q in ["\"unbalanced", "NEAR(", "a -b OR (c", "*"] {
            let r = recall_hybrid_conn(&conn, q, None, None, 5, false, 1000, "local");
            assert!(r.is_ok(), "query {q:?} must not error: {r:?}");
        }
    }

    /// Full public-API e2e through the GLOBAL chain: register a
    /// FakeEmbedder, `remember()` facts, `recall_memory()` them back.
    /// This test CLAIMS the process-wide store singleton by pointing
    /// `$LAMU_DB` at a tempdir BEFORE first touch — it is the only test
    /// in this binary that touches `shared_handle` (everything else is
    /// connection-level), and it holds the chain lock so the env
    /// mutation can't race the other env-touching tests.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn remember_and_recall_e2e_through_global_chain() {
        use crate::embedder::testutil::{chain_lock, reset_chain, FakeEmbedder};
        let _g = chain_lock();
        reset_chain();
        let td = tempfile::tempdir().unwrap();
        // SAFETY: serialized by chain_lock.
        unsafe { std::env::set_var("LAMU_DB", td.path().join("lamu.db")) };

        let fake = FakeEmbedder::new("fake-e2e", vec![0.0, 1.0])
            .with("the zanzibar deployment protocol", vec![0.0, 1.0])
            .with("ship the island rollout plan", vec![1.0, 0.0])
            .with("rollout", vec![1.0, 0.0]); // query → semantic neighbor of `sem`
        crate::embedder::set_global(std::sync::Arc::new(fake));

        let lex = remember("the zanzibar deployment protocol", "fact", "manual", LOCAL_OWNER)
            .await
            .unwrap();
        let sem = remember("ship the island rollout plan", "fact", "manual", LOCAL_OWNER)
            .await
            .unwrap();

        // Rows carry the chain identity; embedding_stores adopted it.
        {
            let arc = memory_db().unwrap();
            let conn = arc.lock();
            let models: Vec<String> = {
                let mut stmt = conn
                    .prepare("SELECT embedding_model FROM memories ORDER BY id")
                    .unwrap();
                let mapped = stmt.query_map([], |r| r.get::<_, String>(0)).unwrap();
                mapped.map(|r| r.unwrap()).collect()
            };
            assert_eq!(models, vec!["fake-e2e".to_string(), "fake-e2e".to_string()]);
            let (store_model, dims): (String, i64) = conn
                .query_row(
                    "SELECT model, dims FROM embedding_stores WHERE store = 'memories'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert_eq!(store_model, "fake-e2e");
            assert_eq!(dims, 2);
        }

        // ADR 0032: novelty dedup is owner-scoped. The same fact text is
        // a duplicate for LOCAL_OWNER (its embedding is already stored
        // under "local") but NOVEL for a different owner — cross-owner
        // dedup would leak fact existence between tenants.
        let dup = remember_if_novel("the zanzibar deployment protocol", "fact", "manual", LOCAL_OWNER)
            .await
            .unwrap();
        assert!(dup.is_none(), "near-duplicate for the same owner is skipped");
        let other = remember_if_novel("the zanzibar deployment protocol", "fact", "manual", "katana-user")
            .await
            .unwrap();
        assert!(other.is_some(), "same text is novel for a different owner");
        // ...and the local recall below must NOT surface the other
        // owner's copy (owner filter on the FTS/vector legs).
        let other_id = other.unwrap();

        // Hybrid recall: "rollout" embeds onto `sem`'s vector (semantic
        // leg) AND lexically matches `sem`'s text; "zanzibar" would be
        // FTS-only. Query "rollout zanzibar" surfaces BOTH.
        let mut fake2 = FakeEmbedder::new("fake-e2e", vec![0.0, 0.0]);
        fake2.map.insert("rollout zanzibar".into(), vec![1.0, 0.0]);
        crate::embedder::set_global(std::sync::Arc::new(fake2));
        let hits = recall_memory("rollout zanzibar", 2, false, LOCAL_OWNER).await.unwrap();
        let ids: std::collections::HashSet<i64> = hits.iter().map(|h| h.id).collect();
        assert!(ids.contains(&sem), "vector leg surfaces the semantic hit");
        assert!(ids.contains(&lex), "FTS leg surfaces the lexical hit");
        assert!(!ids.contains(&other_id), "another owner's copy never surfaces locally");

        // No embedder at all → FTS still finds the lexical match.
        crate::embedder::clear_global();
        let hits = recall_memory("zanzibar", 2, false, LOCAL_OWNER).await.unwrap();
        assert!(hits.iter().any(|h| h.id == lex));
        assert!(hits.iter().all(|h| h.score.is_none()));

        reset_chain();
        unsafe { std::env::remove_var("LAMU_DB") };
        // NOTE: the singleton stays pinned to the (now-removed) tempdir
        // db for the rest of the process — fine: no other test in this
        // binary touches shared_handle.
    }

    #[test]
    fn record_store_identity_upserts_and_pins_on_mismatch() {
        let (_td, conn) = open_tmp_lamu_db();
        // Absent → insert.
        crate::store::record_store_identity(&conn, "memories", "model-a", 384, 100);
        let (model, dims, at): (String, i64, i64) = conn
            .query_row(
                "SELECT model, dims, updated_at FROM embedding_stores WHERE store = 'memories'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!((model.as_str(), dims, at), ("model-a", 384, 100));
        // Matching model → refresh dims + updated_at.
        crate::store::record_store_identity(&conn, "memories", "model-a", 512, 200);
        let (dims2, at2): (i64, i64) = conn
            .query_row(
                "SELECT dims, updated_at FROM embedding_stores WHERE store = 'memories'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!((dims2, at2), (512, 200));
        // Mismatch → row STAYS pinned to the old model (warn-once path).
        crate::store::record_store_identity(&conn, "memories", "model-b", 768, 300);
        let (model3, at3): (String, i64) = conn
            .query_row(
                "SELECT model, updated_at FROM embedding_stores WHERE store = 'memories'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(model3, "model-a", "mismatch must not flip the store row");
        assert_eq!(at3, 200, "mismatch must not touch updated_at");
        // Exercise the warn path a second time (covers the once-per-store set).
        crate::store::record_store_identity(&conn, "memories", "model-b", 768, 400);
    }

    // ── ADR 0032: owner scoping ─────────────────────────────────────

    /// Seed one embedded + one unembedded fact per owner. Both owners'
    /// embedded rows sit on the SAME model and near the same vector, and
    /// both owners' texts share the token "tenancy" — so every leg
    /// (vector, FTS, recency) would surface BOTH owners if the owner
    /// filter were missing.
    fn seed_two_owners(conn: &Connection) -> (i64, i64, i64, i64) {
        let a_vec = insert_memory(conn, "tenancy fact alpha vec", Some(&[1.0, 0.0]), Some("fake"), "fact", "manual", 10, "alice").unwrap();
        let a_fts = insert_memory(conn, "tenancy fact alpha fts", None, None, "fact", "manual", 20, "alice").unwrap();
        let b_vec = insert_memory(conn, "tenancy fact bravo vec", Some(&[0.9, 0.1]), Some("fake"), "fact", "manual", 30, "bob").unwrap();
        let b_fts = insert_memory(conn, "tenancy fact bravo fts", None, None, "fact", "manual", 40, "bob").unwrap();
        (a_vec, a_fts, b_vec, b_fts)
    }

    #[test]
    fn owner_filters_vector_and_fts_legs() {
        let (_td, conn) = open_tmp_lamu_db();
        let (a_vec, a_fts, b_vec, b_fts) = seed_two_owners(&conn);

        // Vector + FTS hybrid as alice: only alice's rows.
        let hits = recall_hybrid_conn(
            &conn, "tenancy", Some(&[1.0, 0.0]), Some("fake"), 10, false, 1000, "alice",
        )
        .unwrap();
        let ids: std::collections::HashSet<i64> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids, [a_vec, a_fts].into_iter().collect(), "alice sees only alice");

        // Same query as bob: only bob's rows.
        let hits_b = recall_hybrid_conn(
            &conn, "tenancy", Some(&[1.0, 0.0]), Some("fake"), 10, false, 1000, "bob",
        )
        .unwrap();
        let ids_b: std::collections::HashSet<i64> = hits_b.iter().map(|h| h.id).collect();
        assert_eq!(ids_b, [b_vec, b_fts].into_iter().collect(), "bob sees only bob");

        // A third owner sees nothing at all.
        let hits_c = recall_hybrid_conn(
            &conn, "tenancy", Some(&[1.0, 0.0]), Some("fake"), 10, false, 1000, "carol",
        )
        .unwrap();
        assert!(hits_c.is_empty(), "stranger sees nothing");
    }

    #[test]
    fn owner_filters_recency_leg() {
        let (_td, conn) = open_tmp_lamu_db();
        let (a_vec, a_fts, _b_vec, _b_fts) = seed_two_owners(&conn);
        // No qvec + a query with no FTS match → pure recency leg.
        let hits = recall_hybrid_conn(&conn, "zzzqqq", None, None, 10, false, 1000, "alice")
            .unwrap();
        let ids: std::collections::HashSet<i64> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids, [a_vec, a_fts].into_iter().collect(), "recency leg owner-scoped");
    }

    #[test]
    fn forget_cross_owner_is_affected_zero() {
        let (_td, conn) = open_tmp_lamu_db();
        let (a_vec, ..) = seed_two_owners(&conn);
        // bob cannot forget alice's fact — same response as a missing id.
        assert!(!forget_conn(&conn, a_vec, 500, "bob").unwrap());
        let valid_until: Option<i64> = conn
            .query_row("SELECT valid_until FROM memories WHERE id = ?", params![a_vec], |r| r.get(0))
            .unwrap();
        assert!(valid_until.is_none(), "cross-owner forget must not expire the row");
        // alice can.
        assert!(forget_conn(&conn, a_vec, 500, "alice").unwrap());
    }

    #[test]
    fn supersede_cross_owner_inserts_nothing() {
        let (_td, mut conn) = open_tmp_lamu_db();
        let (a_vec, ..) = seed_two_owners(&conn);
        let before: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        // bob superseding alice's fact: None, nothing inserted, old intact.
        let r = supersede_conn(&mut conn, a_vec, "bob's takeover", None, None, "fact", "manual", 500, "bob")
            .unwrap();
        assert!(r.is_none(), "cross-owner supersede must return None");
        let after: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(before, after, "cross-owner supersede must not insert the new fact");
        let valid_until: Option<i64> = conn
            .query_row("SELECT valid_until FROM memories WHERE id = ?", params![a_vec], |r| r.get(0))
            .unwrap();
        assert!(valid_until.is_none(), "old fact stays current");

        // Missing id behaves identically (no existence leak).
        let r2 = supersede_conn(&mut conn, 999_999, "ghost", None, None, "fact", "manual", 500, "bob")
            .unwrap();
        assert!(r2.is_none());

        // alice superseding her own fact works and the new row is hers.
        let new_id = supersede_conn(&mut conn, a_vec, "alpha vec v2", None, None, "fact", "manual", 600, "alice")
            .unwrap()
            .expect("own supersede inserts");
        let owner: String = conn
            .query_row("SELECT owner FROM memories WHERE id = ?", params![new_id], |r| r.get(0))
            .unwrap();
        assert_eq!(owner, "alice");
    }

    #[test]
    fn export_corpus_is_owner_scoped() {
        let (_td, conn) = open_tmp_lamu_db();
        let (a_vec, a_fts, b_vec, b_fts) = seed_two_owners(&conn);
        let rows_a = load_corpus_rows(&conn, false, 1000, "alice").unwrap();
        let ids_a: std::collections::HashSet<i64> = rows_a.iter().map(|r| r.id).collect();
        assert_eq!(ids_a, [a_vec, a_fts].into_iter().collect());
        let rows_b = load_corpus_rows(&conn, false, 1000, "bob").unwrap();
        let ids_b: std::collections::HashSet<i64> = rows_b.iter().map(|r| r.id).collect();
        assert_eq!(ids_b, [b_vec, b_fts].into_iter().collect());
        // include_expired stays owner-scoped too.
        forget_conn(&conn, a_vec, 500, "alice").unwrap();
        let rows_all = load_corpus_rows(&conn, true, 1000, "alice").unwrap();
        let ids_all: std::collections::HashSet<i64> = rows_all.iter().map(|r| r.id).collect();
        assert_eq!(ids_all, [a_vec, a_fts].into_iter().collect());
    }

    #[test]
    fn recall_hydrates_valid_until_on_expired_hits() {
        let (_td, conn) = open_tmp_lamu_db();
        let id = insert_memory(&conn, "tenancy expiring fact", None, None, "fact", "manual", 10, "alice").unwrap();
        forget_conn(&conn, id, 500, "alice").unwrap();
        let hits =
            recall_hybrid_conn(&conn, "tenancy", None, None, 5, true, 1000, "alice").unwrap();
        let hit = hits.iter().find(|h| h.id == id).expect("expired hit surfaces");
        assert_eq!(hit.valid_until, Some(500));
        // A current fact hydrates None.
        let id2 = insert_memory(&conn, "tenancy current fact", None, None, "fact", "manual", 20, "alice").unwrap();
        let hits2 =
            recall_hybrid_conn(&conn, "tenancy", None, None, 5, false, 1000, "alice").unwrap();
        let hit2 = hits2.iter().find(|h| h.id == id2).expect("current hit surfaces");
        assert!(hit2.valid_until.is_none());
    }
}
