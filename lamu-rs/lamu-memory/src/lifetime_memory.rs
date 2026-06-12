//! Lifetime cross-session memory — storage core.
//!
//! Where `memory.rs` keys turns by `conversation_id` (strictly
//! per-conversation), this module is a GLOBAL fact store that spans
//! every conversation. Facts are extracted from conversations (via
//! MiMo — that orchestration lives in the lamu-mcp frontend, ADR 0029)
//! or added explicitly, embedded with OpenAI's
//! `text-embedding-3-small`, and recalled by cross-session semantic
//! search over the existing `crate::vector_index::BruteForceCosine`
//! seam — this is the seam's first cross-session consumer.
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
//! A separate SQLite at `~/.local/share/lamu/memory.db` (NOT
//! `conversations.db`). We mirror `rag.rs`'s
//! `OnceLock<Arc<Mutex<Connection>>>` + WAL pattern with our own
//! static + accessor; the established codebase convention is
//! per-module duplication of the connection singleton rather than a
//! shared helper.
//!
//! ## Degradation without an embedding key
//!
//! `remember` stores the memory with `embedding = NULL` when there is
//! no `OPENAI_API_KEY` — it never fails on a missing key. `recall_memory`
//! ranks embedding-bearing rows semantically when a key is present, and
//! falls back to most-recent-k by `ts` when no key is available at all
//! (so the query itself cannot be embedded).

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::rag::{blob_to_vec, embed_one, openai_key, vec_to_blob};
use crate::vector_index::{cosine, vector_backend, BruteForceCosine, Scored, VectorBackend, VectorIndex};

const SCHEMA: &str = "
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
CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_until);
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

fn memory_db_path() -> Result<PathBuf> {
    let dir = dirs::data_local_dir()
        .ok_or_else(|| anyhow!("no data_local_dir"))?
        .join("lamu");
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir.join("memory.db"))
}

fn open_memory_db(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.execute_batch(SCHEMA)?;
    // SCHEMA's CREATE TABLE IF NOT EXISTS only adds the temporal columns
    // to a FRESH db; bring an existing memory.db up to the valid-time
    // schema (idempotent, safe to run every startup).
    migrate_temporal_columns(&conn)?;
    Ok(conn)
}

static MEMORY_DB: OnceLock<Arc<Mutex<Connection>>> = OnceLock::new();

fn memory_db() -> Result<Arc<Mutex<Connection>>> {
    // Fast path: already initialized.
    if let Some(d) = MEMORY_DB.get() {
        return Ok(d.clone());
    }
    // Open outside the OnceLock so a failed open doesn't poison the
    // cell, then publish via get_or_init — under a race the loser's
    // connection is dropped (not leaked) and everyone shares the winner.
    let path = memory_db_path()?;
    let conn = open_memory_db(&path)?;
    let arc = MEMORY_DB.get_or_init(|| Arc::new(Mutex::new(conn)));
    Ok(arc.clone())
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
#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub id: i64,
    pub text: String,
    pub kind: String,
    pub source: Option<String>,
    pub ts: i64,
    pub score: Option<f32>,
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
/// against an in-memory connection without hitting OpenAI.
pub(crate) fn insert_memory(
    conn: &Connection,
    text: &str,
    embedding: Option<&[f32]>,
    kind: &str,
    source: &str,
    ts: i64,
) -> Result<i64> {
    // A brand-new fact is valid from `ts`, has no expiry, and supersedes
    // nothing. supersede() uses insert_memory_full to set `supersedes`.
    insert_memory_full(conn, text, embedding, kind, source, ts, ts, None)
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
    kind: &str,
    source: &str,
    ts: i64,
    valid_from: i64,
    supersedes: Option<i64>,
) -> Result<i64> {
    let blob = embedding.map(vec_to_blob);
    conn.execute(
        "INSERT INTO memories (text, embedding, kind, source, ts, valid_from, valid_until, supersedes) \
         VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
        params![text, blob, kind, source, ts, valid_from, supersedes],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Store a fact in the lifetime memory. Embeds via OpenAI when a key is
/// present; stores `embedding = NULL` otherwise (never fails on a
/// missing key). Returns the new rowid.
pub async fn remember(text: &str, kind: &str, source: &str) -> Result<i64> {
    let embedding = match openai_key() {
        Some(key) => Some(embed_one(text, &key).await?),
        None => None,
    };
    let arc = memory_db()?;
    let conn = arc.lock();
    insert_memory(
        &conn,
        text,
        embedding.as_deref(),
        kind,
        source,
        now_secs(),
    )
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
        })
        .collect()
}

/// Recall the top-`k` memories most relevant to `query`.
///
/// - With an `OPENAI_API_KEY`: embed the query, load all rows, rank the
///   embedding-bearing rows via the seam (NULL-embedding rows are
///   skipped in the ranked path).
/// - Without a key (the query itself cannot be embedded): fall back to
///   the most-recent `k` rows by `ts`, descending, with `score = None`.
///
/// VALID-TIME SEMANTICS: by default (`include_expired = false`) recall
/// returns ONLY currently-valid facts — rows whose `valid_until` is NULL
/// (never expired) or lies in the future. This is the intended temporal
/// behaviour: facts that were superseded or soft-deleted (forgotten) drop
/// out of default recall but are NEVER removed from the store. Pass
/// `include_expired = true` for historical recall over the full timeline.
pub async fn recall_memory(query: &str, k: usize, include_expired: bool) -> Result<Vec<MemoryHit>> {
    let arc = memory_db()?;
    let now = now_secs();
    let valid = valid_time_clause(include_expired, now);

    match openai_key() {
        Some(key) => {
            let qvec = embed_one(query, &key).await?;
            // Collect rows into a Vec, then release the lock BEFORE
            // ranking — never hold the mutex across the cosine pass, so
            // concurrent remember/recall don't serialize behind it.
            // `embedding IS NOT NULL` skips embedding-less rows in SQL
            // (the ranked path can't use them anyway); the valid-time
            // clause (when not include_expired) hides expired facts.
            let where_clause = if valid.is_empty() {
                "WHERE embedding IS NOT NULL".to_string()
            } else {
                format!("WHERE embedding IS NOT NULL AND {valid}")
            };
            let sql = format!(
                "SELECT id, text, kind, source, ts, embedding FROM memories {where_clause}"
            );
            let rows: Vec<MemoryRow> = {
                let conn = arc.lock();
                let mut stmt = conn.prepare(&sql)?;
                let mapped = stmt.query_map([], |r| {
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
            Ok(rank_memories(&qvec, rows, k))
        }
        None => {
            // No key — query can't be embedded; fall back to recency.
            let where_clause = if valid.is_empty() {
                String::new()
            } else {
                format!("WHERE {valid} ")
            };
            let sql = format!(
                "SELECT id, text, kind, source, ts FROM memories \
                 {where_clause}ORDER BY ts DESC, id DESC LIMIT ?"
            );
            let conn = arc.lock();
            let mut stmt = conn.prepare(&sql)?;
            let mapped = stmt.query_map(params![k as i64], |r| {
                Ok(MemoryHit {
                    id: r.get(0)?,
                    text: r.get(1)?,
                    kind: r.get(2)?,
                    source: r.get(3)?,
                    ts: r.get(4)?,
                    score: None,
                })
            })?;
            let mut hits = Vec::new();
            for h in mapped {
                hits.push(h?);
            }
            Ok(hits)
        }
    }
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
/// - Without an `OPENAI_API_KEY`: dedup is impossible (no embeddings), so
///   fall back to an unconditional [`remember`] and return `Ok(Some(id))`.
/// - With a key: embed `text`, load every embedding-bearing row (same
///   SELECT + `blob_to_vec` the recall path uses), and if
///   [`is_novel`] is false return `Ok(None)`. Otherwise insert the row
///   WITH its embedding and return `Ok(Some(id))`.
pub async fn remember_if_novel(text: &str, kind: &str, source: &str) -> Result<Option<i64>> {
    let key = match openai_key() {
        // No key → can't embed/dedup; store unconditionally.
        None => return remember(text, kind, source).await.map(Some),
        Some(k) => k,
    };

    let emb = embed_one(text, &key).await?; // embed BEFORE taking the lock
    let arc = memory_db()?;
    let now = now_secs();
    // Dedup only against CURRENTLY-VALID facts. The old SELECT had no
    // valid-time filter, so a re-asserted fact whose near-duplicate had
    // been superseded/forgotten (row still present, just expired) was
    // dropped as a "duplicate" — but default recall hides the expired row,
    // so the now-current fact silently vanished.
    let valid = valid_time_clause(false, now);
    let sql =
        format!("SELECT embedding FROM memories WHERE embedding IS NOT NULL AND {valid}");
    // Hold ONE guard across SELECT + is_novel + insert. is_novel (cosine)
    // and insert_memory are synchronous (no await), so this is safe and
    // closes the TOCTOU window where two concurrent autocapture threads
    // both passed the novelty check against the same pre-insert snapshot
    // and both inserted. Trade-off: the cosine scan now runs under the
    // lock, serializing concurrent novelty checks — fine while memory.db
    // is small + autocapture is bounded; revisit if the store grows large.
    let conn = arc.lock();
    let existing: Vec<Vec<f32>> = {
        let mut stmt = conn.prepare(&sql)?;
        let mapped = stmt.query_map([], |r| {
            let blob: Vec<u8> = r.get(0)?;
            Ok(blob_to_vec(&blob))
        })?;
        let mut rows = Vec::new();
        for row in mapped {
            rows.push(row?);
        }
        rows
    };

    if !is_novel(&emb, &existing, NOVELTY_THRESHOLD) {
        return Ok(None);
    }

    let id = insert_memory(&conn, text, Some(&emb), kind, source, now)?;
    Ok(Some(id))
}

// ── Supersession + soft-delete (temporal) ──────────────────────────

/// Replace fact `old_id` with a NEW fact (`new_text`): the new fact is
/// inserted with `supersedes = Some(old_id)` and `valid_from = now`, and
/// the old fact is expired (`valid_until = now`). Returns the new fact's
/// rowid.
///
/// This is the "user moved X → Y" operation: the old fact becomes
/// historical (still in the store, recallable with `include_expired`) and
/// the new fact takes its place in default recall. The old row is only
/// expired if it is CURRENTLY valid (`valid_until IS NULL`) — re-superseding
/// an already-expired fact leaves its earlier `valid_until` intact.
///
/// Embeds `new_text` exactly like [`remember`] (NULL embedding when no
/// `OPENAI_API_KEY`), so the new fact is semantically recallable.
pub async fn supersede(old_id: i64, new_text: &str, kind: &str, source: &str) -> Result<i64> {
    let embedding = match openai_key() {
        Some(key) => Some(embed_one(new_text, &key).await?),
        None => None,
    };
    let now = now_secs();
    let arc = memory_db()?;
    let mut conn = arc.lock();
    supersede_conn(&mut conn, old_id, new_text, embedding.as_deref(), kind, source, now)
}

/// Connection-level core of [`supersede`]: insert the new fact with
/// `supersedes = Some(old_id)` / `valid_from = now`, then expire the old
/// fact if it is currently valid. Factored out so tests can drive it
/// against an in-memory connection with a known embedding and `now`.
///
/// ATOMICITY: the INSERT (new fact) and UPDATE (expire old fact) run in a
/// single SQLite transaction. Without it, a crash or error between the two
/// statements would leave the new fact inserted while the old fact is still
/// `valid_until IS NULL` — both then appear in default recall, violating the
/// "exactly one valid version" invariant supersession exists to enforce.
#[allow(clippy::too_many_arguments)]
pub(crate) fn supersede_conn(
    conn: &mut Connection,
    old_id: i64,
    new_text: &str,
    embedding: Option<&[f32]>,
    kind: &str,
    source: &str,
    now: i64,
) -> Result<i64> {
    let tx = conn.transaction()?;
    let new_id = insert_memory_full(&tx, new_text, embedding, kind, source, now, now, Some(old_id))?;
    tx.execute(
        "UPDATE memories SET valid_until = ? WHERE id = ? AND valid_until IS NULL",
        params![now, old_id],
    )?;
    tx.commit()?;
    Ok(new_id)
}

/// Soft-delete fact `id`: set `valid_until = now` so it drops out of
/// default recall but remains in the store (recoverable, and the timeline
/// survives). Returns `true` if a currently-valid row with that id was
/// expired, `false` if no such row existed (already expired or absent).
///
/// No fact is ever hard-deleted; `forget` only moves a fact into history.
pub fn forget(id: i64) -> Result<bool> {
    let now = now_secs();
    let arc = memory_db()?;
    let conn = arc.lock();
    forget_conn(&conn, id, now)
}

/// Connection-level core of [`forget`]: expire the row if it is currently
/// valid, returning whether a row was affected. Factored out for testing
/// against an in-memory connection with a known `now`.
pub(crate) fn forget_conn(conn: &Connection, id: i64, now: i64) -> Result<bool> {
    let affected = conn.execute(
        "UPDATE memories SET valid_until = ? WHERE id = ? AND valid_until IS NULL",
        params![now, id],
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
pub fn export_graph_corpus(dir: &Path, include_expired: bool) -> Result<usize> {
    let arc = memory_db()?;
    let now = now_secs();
    // Load all rows under the lock, then release before doing filesystem
    // writes — same don't-hold-the-mutex-across-I/O discipline as recall.
    let rows = {
        let conn = arc.lock();
        load_corpus_rows(&conn, include_expired, now)?
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

/// Load the rows to export (currently-valid only unless `include_expired`),
/// ordered by id. Connection-level so tests drive it without the singleton.
pub(crate) fn load_corpus_rows(
    conn: &Connection,
    include_expired: bool,
    now: i64,
) -> Result<Vec<CorpusRow>> {
    let valid = valid_time_clause(include_expired, now);
    let where_clause = if valid.is_empty() {
        String::new()
    } else {
        format!("WHERE {valid} ")
    };
    let sql = format!(
        "SELECT id, text, kind, source, ts, valid_from, valid_until, supersedes \
         FROM memories {where_clause}ORDER BY id ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mapped = stmt.query_map([], |r| {
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

    fn open_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(SCHEMA).unwrap();
        conn
    }

    #[test]
    fn insert_and_rank_with_known_embeddings() {
        let conn = open_test_db();
        // Hand-crafted 3-dim embeddings.
        let id_x = insert_memory(&conn, "x-axis fact", Some(&[1.0, 0.0, 0.0]), "fact", "manual", 100)
            .unwrap();
        let id_y = insert_memory(&conn, "y-axis fact", Some(&[0.0, 1.0, 0.0]), "fact", "manual", 200)
            .unwrap();
        let id_near =
            insert_memory(&conn, "near-x fact", Some(&[0.9, 0.1, 0.0]), "fact", "manual", 300)
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
        let id = insert_memory(&conn, "no-embedding fact", None, "fact", "manual", 42).unwrap();
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
            insert_memory(&conn, "current fact", None, "fact", "manual", 100).unwrap();
        // Expired fact: insert then expire it with valid_until in the past.
        let id_expired =
            insert_memory(&conn, "expired fact", None, "fact", "manual", 50).unwrap();
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
        let old_id = insert_memory(&conn, "lives in SF", None, "fact", "manual", 100).unwrap();

        // Supersede at now = 500.
        let new_id =
            supersede_conn(&mut conn, old_id, "lives in NYC", None, "fact", "manual", 500).unwrap();
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
        let id = insert_memory(&conn, "transient", None, "fact", "manual", 100).unwrap();

        // First forget: row is currently valid → affected → true.
        assert!(forget_conn(&conn, id, 777).unwrap());
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
        assert!(!forget_conn(&conn, id, 888).unwrap());
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
            insert_memory(&conn, "current fact", None, "preference", "manual", 100).unwrap();
        let id_expired =
            insert_memory(&conn, "expired fact", None, "fact", "sess-1", 50).unwrap();
        conn.execute(
            "UPDATE memories SET valid_until = ? WHERE id = ?",
            params![60i64, id_expired],
        )
        .unwrap();

        let tmp = tempfile::tempdir().unwrap();

        // Default: only the current fact is exported.
        let now = 1000;
        let rows = load_corpus_rows(&conn, false, now).unwrap();
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
        let rows_all = load_corpus_rows(&conn, true, now).unwrap();
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
        insert_memory(&conn, "a fact", None, "fact", "manual", 100).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("does").join("not").join("exist");
        let rows = load_corpus_rows(&conn, false, 1000).unwrap();
        let n = write_corpus_rows(&nested, &rows).unwrap();
        assert_eq!(n, 1);
        assert!(nested.join("mem_1.md").exists());
    }

    #[test]
    fn export_prunes_stale_files_but_keeps_foreign() {
        let conn = open_test_db();
        let id1 = insert_memory(&conn, "fact one", None, "fact", "manual", 100).unwrap();
        let id2 = insert_memory(&conn, "fact two", None, "fact", "manual", 100).unwrap();
        let id3 = insert_memory(&conn, "fact three", None, "fact", "manual", 100).unwrap();
        let tmp = tempfile::tempdir().unwrap();

        // First export: all three written.
        let rows = load_corpus_rows(&conn, false, 1000).unwrap();
        assert_eq!(write_corpus_rows(tmp.path(), &rows).unwrap(), 3);
        // A foreign file the prune must never touch.
        std::fs::write(tmp.path().join("README.md"), b"keep me").unwrap();

        // Forget id2, then re-export to the SAME dir.
        conn.execute(
            "UPDATE memories SET valid_until = ? WHERE id = ?",
            params![500i64, id2],
        )
        .unwrap();
        let rows2 = load_corpus_rows(&conn, false, 1000).unwrap();
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
}
