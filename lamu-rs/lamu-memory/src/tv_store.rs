//! Persistent turbovec index lifecycle (ADR 0031).
//!
//! The per-query [`crate::vector_index::TurboVecIndex`] rebuilds its
//! quantized index from SQLite rows on every search — fine for a first
//! cut, pure waste once the corpus grows. This module gives each vector
//! store (`memories`, `chunks`) ONE long-lived quantized index that is
//! loaded from disk at first use, caught up from SQLite, appended to by
//! the write paths, and persisted back with tmp-file + atomic rename.
//!
//! ## Files
//!
//! Per store under `<data_dir>/lamu/index/` (derived from the lamu.db
//! path's PARENT, so a `$LAMU_DB` redirect in tests redirects the index
//! too):
//!
//! - `<store>.tv` — the quantized `TurboQuantIndex` (turbovec's format)
//! - `<store>.ids` — JSON array of SQLite rowids, parallel to the .tv
//!   slot order (slots are append-only; we never `swap_remove`)
//! - `<store>.meta.json` — `{model, dims, bit_width, last_rowid}`
//!
//! Persist order is .tv → .ids → .meta.json, each via tmp + atomic
//! rename; the meta is the commit point. A crash between renames leaves
//! a length/identity mismatch that the next load detects → full rebuild.
//!
//! ## Lifecycle
//!
//! - **Load-or-rebuild (first use per store per process):** load the
//!   three files, validate the meta against the CURRENT embedder
//!   identity (model + dims) AND `vector_index_state`; any mismatch or
//!   load error → discard the files and rebuild from SQLite. The rebuild
//!   SELECT filters on `embedding_model = current` and `embedding NOT
//!   NULL` only — validity (expiry) is deliberately NOT filtered at
//!   build time, because expiring a fact must never force a rebuild;
//!   expired rows are hidden by the search-time SQL post-filter instead.
//! - **Catch-up:** when `meta.last_rowid < MAX(rowid)` for the store,
//!   only the missing rows are added (same model filter).
//! - **Incremental add:** write paths append the (normalized) vector +
//!   rowid to the live index and bump `last_indexed_rowid` (in
//!   `vector_index_state` and, at persist time, the meta). Persisting is
//!   THROTTLED: every [`PERSIST_EVERY_ADDS`] adds or [`PERSIST_MAX_AGE`],
//!   whichever first, checked at the `maybe_persist` hook points (no
//!   background task in v1); [`flush_all`] forces it on demand.
//! - **Invalidation without delete:** `forget`/`supersede`/chunk
//!   re-index bump `vector_index_state.stale_count`; the expired row's
//!   vector stays in the .tv. Searches over-fetch `k *` [`OVERFETCH`]
//!   and the call sites post-filter hits against SQLite (validity +
//!   model). When `stale_count` exceeds 25% of indexed rows the next
//!   search performs a SYNCHRONOUS full rebuild (v1 trade-off: simple +
//!   correct; a background rebuild task is a documented follow-up).
//!
//! ## Locking (load-bearing)
//!
//! Two locks exist: the shared DB connection mutex (`crate::store`) and
//! ONE mutex per store around the in-memory index. Lock ORDER: **DB
//! first, index second** — every path that needs both (`note_added`,
//! `search_persistent`, the load/rebuild inside it) is entered while the
//! caller already holds the DB guard and only then takes the index lock.
//! No code under an index lock ever calls `crate::store::shared_handle`
//! or otherwise acquires the DB mutex (it only uses the `&Connection`
//! the caller passed in), so the inverse order cannot occur and the pair
//! cannot deadlock. `maybe_persist` / `flush_all` take ONLY the index
//! lock — file I/O never runs while the DB lock is held.
//!
//! ## Known v1 limitations (documented, accepted)
//!
//! - In-place re-embedding under the SAME model (no rowid change, no
//!   model change) is invisible to catch-up. The shipped `reembed` flow
//!   always switches models, which the identity check catches → rebuild.
//! - `chunks` rowids can be reused by SQLite after a re-index DELETE; a
//!   stale slot can then hydrate the new row's content until the stale
//!   threshold triggers a rebuild. Hits are deduped by rowid (best score
//!   wins) so this only mildly perturbs ranking, never correctness of
//!   the hydrated payloads.
//! - Cross-process staleness: another process's inserts are picked up at
//!   load/catch-up time, not live. Same trade-off the per-query brute
//!   scan never had, accepted for v1.

// The lifecycle core is GENERIC over [`QuantBackend`] precisely so it can
// be unit-tested in the default feature-off build (the turbovec dep links
// OpenBLAS); in a feature-off NON-test build only the inert hook shells
// are reachable, so the core would otherwise trip dead_code.
#![cfg_attr(not(any(feature = "turbovec", test)), allow(dead_code))]

use anyhow::{anyhow, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::rag::blob_to_vec;
use crate::vector_index::normalize;

/// Quantization width for the persistent index (4 bits = best recall per
/// turbovec's docs; mirrors `TurboVecIndex::DEFAULT_BIT_WIDTH`).
//
// Only the feature-gated hook bodies (+ the turbovec tests) read this, so
// a feature-off test build would otherwise flag it as dead.
#[cfg_attr(not(feature = "turbovec"), allow(dead_code))]
pub(crate) const DEFAULT_BIT_WIDTH: usize = 4;

/// Persist after this many incremental adds…
pub(crate) const PERSIST_EVERY_ADDS: u32 = 32;

/// …or after this much time with ≥1 unpersisted add, whichever first.
pub(crate) const PERSIST_MAX_AGE: Duration = Duration::from_secs(30);

/// Over-fetch factor for searches: stale (expired-but-still-indexed)
/// vectors are filtered AFTER the ANN search, so fetch `k * OVERFETCH`
/// candidates to keep k survivors likely.
pub(crate) const OVERFETCH: usize = 4;

// ── Store identity ──────────────────────────────────────────────────

/// Which vector store an index covers. The two embedding-bearing tables
/// of the unified lamu.db.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Store {
    Memories,
    Chunks,
}

impl Store {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Store::Memories => "memories",
            Store::Chunks => "chunks",
        }
    }
}

// ── On-disk meta + DB bookkeeping rows ──────────────────────────────

/// `<store>.meta.json` — the on-disk index's identity + watermark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct IndexMeta {
    pub model: String,
    pub dims: usize,
    pub bit_width: usize,
    /// Watermark: every rowid ≤ this has been CONSIDERED for indexing
    /// (indexed if it carried a matching-model embedding). Catch-up adds
    /// rows above it.
    pub last_rowid: i64,
}

/// One `vector_index_state` row (the W2b bookkeeping table).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisState {
    pub last_indexed_rowid: i64,
    pub stale_count: i64,
    pub model: Option<String>,
    pub dims: Option<i64>,
}

pub(crate) fn read_state(conn: &Connection, store: Store) -> Result<Option<VisState>> {
    conn.query_row(
        "SELECT last_indexed_rowid, stale_count, model, dims \
         FROM vector_index_state WHERE store = ?1",
        params![store.name()],
        |r| {
            Ok(VisState {
                last_indexed_rowid: r.get(0)?,
                stale_count: r.get(1)?,
                model: r.get(2)?,
                dims: r.get(3)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

/// Monotonic upsert of `last_indexed_rowid` (never moves backwards).
pub(crate) fn bump_last_indexed(conn: &Connection, store: Store, rowid: i64) -> Result<()> {
    conn.execute(
        "INSERT INTO vector_index_state (store, last_indexed_rowid, stale_count) \
         VALUES (?1, ?2, 0) \
         ON CONFLICT(store) DO UPDATE SET \
             last_indexed_rowid = MAX(last_indexed_rowid, excluded.last_indexed_rowid)",
        params![store.name(), rowid],
    )?;
    Ok(())
}

/// Add `n` to `stale_count` (rows expired/replaced but still in the .tv).
pub(crate) fn bump_stale(conn: &Connection, store: Store, n: i64) -> Result<()> {
    conn.execute(
        "INSERT INTO vector_index_state (store, last_indexed_rowid, stale_count) \
         VALUES (?1, 0, ?2) \
         ON CONFLICT(store) DO UPDATE SET stale_count = stale_count + excluded.stale_count",
        params![store.name(), n],
    )?;
    Ok(())
}

/// Record a completed full (re)build: adopt the identity, zero the stale
/// counter, stamp `built_at`, set the watermark.
pub(crate) fn record_built(
    conn: &Connection,
    store: Store,
    model: &str,
    dims: usize,
    watermark: i64,
    now: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO vector_index_state \
             (store, last_indexed_rowid, stale_count, model, dims, built_at) \
         VALUES (?1, ?2, 0, ?3, ?4, ?5) \
         ON CONFLICT(store) DO UPDATE SET \
             last_indexed_rowid = excluded.last_indexed_rowid, \
             stale_count = 0, \
             model = excluded.model, \
             dims = excluded.dims, \
             built_at = excluded.built_at",
        params![store.name(), watermark, model, dims as i64, now],
    )?;
    Ok(())
}

// ── Pure decisions (unit-tested feature-off) ────────────────────────

/// Should the on-disk index be discarded and rebuilt? `Some(reason)` when
/// yes. Pure — every input is read by the caller.
///
/// NOT a rebuild: `meta.last_rowid < max_rowid` (that's catch-up) and a
/// missing `vector_index_state` row (post-filtering keeps search correct;
/// the first build writes the row).
pub(crate) fn rebuild_needed(
    meta: &IndexMeta,
    current_model: &str,
    current_dims: usize,
    expected_bit_width: usize,
    state: Option<&VisState>,
    max_rowid: i64,
) -> Option<&'static str> {
    if meta.model != current_model {
        return Some("embedder model changed");
    }
    if meta.dims != current_dims {
        return Some("embedding dims changed");
    }
    if meta.bit_width != expected_bit_width {
        return Some("quantization bit_width changed");
    }
    if meta.last_rowid > max_rowid {
        return Some("index watermark ahead of the table (store reset?)");
    }
    if let Some(s) = state {
        if s.model.as_deref().is_some_and(|m| m != current_model) {
            return Some("vector_index_state model mismatch");
        }
        if s.dims.is_some_and(|d| d != current_dims as i64) {
            return Some("vector_index_state dims mismatch");
        }
        if s.last_indexed_rowid < meta.last_rowid {
            return Some("vector_index_state behind the on-disk meta (state reset?)");
        }
    }
    None
}

/// Stale-threshold math: rebuild when stale_count > 25% of indexed rows.
/// An all-stale empty index (indexed 0, stale > 0) also rebuilds.
pub(crate) fn stale_rebuild_due(stale_count: i64, indexed_rows: usize) -> bool {
    if stale_count <= 0 {
        return false;
    }
    if indexed_rows == 0 {
        return true;
    }
    stale_count * 4 > indexed_rows as i64
}

// ── SQL row selection (shared by rebuild + catch-up) ────────────────

/// Highest rowid currently in the store's table (0 when empty). The
/// watermark a build/catch-up advances `last_rowid` to.
pub(crate) fn max_rowid(conn: &Connection, store: Store) -> Result<i64> {
    let sql = match store {
        Store::Memories => "SELECT COALESCE(MAX(id), 0) FROM memories",
        Store::Chunks => "SELECT COALESCE(MAX(rowid), 0) FROM chunks",
    };
    conn.query_row(sql, [], |r| r.get(0)).map_err(Into::into)
}

/// Rows that belong in the index but lie above `after_rowid`: embedding
/// present + embedded with the CURRENT model. `after_rowid = 0` is the
/// full-rebuild SELECT. Deliberately NO validity filter — expiring a fact
/// must not require a rebuild; expired rows are hidden at search time by
/// the SQL post-filter (see module docs).
pub(crate) fn pending_rows(
    conn: &Connection,
    store: Store,
    model: &str,
    after_rowid: i64,
) -> Result<Vec<(i64, Vec<f32>)>> {
    let sql = match store {
        Store::Memories => {
            "SELECT id, embedding FROM memories \
             WHERE id > ?1 AND embedding IS NOT NULL AND embedding_model = ?2 \
             ORDER BY id"
        }
        Store::Chunks => {
            "SELECT rowid, embedding FROM chunks \
             WHERE rowid > ?1 AND embedding IS NOT NULL AND embedding_model = ?2 \
             ORDER BY rowid"
        }
    };
    let mut stmt = conn.prepare(sql)?;
    let mapped = stmt.query_map(params![after_rowid, model], |r| {
        let id: i64 = r.get(0)?;
        let blob: Vec<u8> = r.get(1)?;
        Ok((id, blob_to_vec(&blob)))
    })?;
    let mut rows = Vec::new();
    for row in mapped {
        rows.push(row?);
    }
    Ok(rows)
}

// ── Paths + atomic file plumbing ────────────────────────────────────

/// `<data_dir>/lamu/index/` — derived from the lamu.db path's PARENT so a
/// `$LAMU_DB` redirect (tests, sandboxes) redirects the index files too.
//
// Only the feature-gated hook bodies call this (tests pass explicit dirs),
// so a feature-off test build would otherwise flag it as dead.
#[cfg_attr(not(feature = "turbovec"), allow(dead_code))]
pub(crate) fn index_dir() -> PathBuf {
    crate::store::lamu_db_path()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default()
        .join("index")
}

fn tv_path(dir: &Path, store: Store) -> PathBuf {
    dir.join(format!("{}.tv", store.name()))
}
fn ids_path(dir: &Path, store: Store) -> PathBuf {
    dir.join(format!("{}.ids", store.name()))
}
fn meta_path(dir: &Path, store: Store) -> PathBuf {
    dir.join(format!("{}.meta.json", store.name()))
}

/// Sibling tmp path for the atomic write dance (pid-suffixed so two
/// processes can't collide on the same tmp).
fn tmp_sibling(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(format!(".tmp.{}", std::process::id()));
    PathBuf::from(s)
}

/// Write `bytes` to `path` via tmp + atomic rename. On any failure the
/// tmp is removed and `path` is left untouched (old content or absent).
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = tmp_sibling(path);
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("rename {} -> {}", tmp.display(), path.display()));
    }
    Ok(())
}

/// Best-effort removal of all three index files for a store.
fn discard_files(dir: &Path, store: Store) {
    for p in [tv_path(dir, store), ids_path(dir, store), meta_path(dir, store)] {
        let _ = std::fs::remove_file(p);
    }
}

fn any_file_exists(dir: &Path, store: Store) -> bool {
    tv_path(dir, store).exists()
        || ids_path(dir, store).exists()
        || meta_path(dir, store).exists()
}

// ── The quantized-backend seam ──────────────────────────────────────

/// What the lifecycle needs from a quantized index. The real impl is
/// [`TurboBackend`] (feature `turbovec`); tests use a brute-force
/// stand-in so ALL lifecycle logic (meta validation, catch-up, stale
/// accounting, atomic persist) runs in the default feature-off build.
///
/// Contract: slots are append-only and positional — `search` returns
/// `(slot, score)` with slot parallel to the add order. Vectors handed
/// in are already L2-normalized (scores are cosine-comparable).
pub(crate) trait QuantBackend: Sized + Send {
    fn create(dims: usize, bit_width: usize) -> Result<Self>;
    fn load(path: &Path) -> Result<Self>;
    /// Append `n` vectors flattened into `flat` (`flat.len() == n * dims`).
    fn add_batch(&mut self, flat: &[f32], n: usize);
    /// Top-`k` `(slot, score)` by descending similarity.
    fn search(&self, normalized_query: &[f32], k: usize) -> Vec<(usize, f32)>;
    fn len(&self) -> usize;
    fn dims(&self) -> usize;
    fn bit_width(&self) -> usize;
    fn save(&self, path: &Path) -> Result<()>;
    /// Optional eager cache warm-up after load/build.
    fn prepare(&self) {}
}

// ── PersistentIndex — the per-store lifecycle state ─────────────────

/// One store's live in-memory index + its persistence bookkeeping.
/// Generic over [`QuantBackend`] so the lifecycle is testable without
/// the `turbovec` feature (and its OpenBLAS link dependency).
pub(crate) struct PersistentIndex<B> {
    backend: B,
    /// Slot → SQLite rowid, parallel to the backend's append order.
    rowids: Vec<i64>,
    pub(crate) meta: IndexMeta,
    /// Adds since the last successful persist.
    pub(crate) pending_adds: u32,
    /// Persist on the next opportunity regardless of throttle (set by a
    /// fresh build so it reaches disk before the next 32 adds).
    pub(crate) force_persist: bool,
    pub(crate) last_persist: Instant,
}

impl<B: QuantBackend> PersistentIndex<B> {
    /// Build from already-selected `(rowid, embedding)` rows in ONE batch
    /// add (turbovec fits its TQ+ calibration on the first batch — a
    /// batched build calibrates on the whole corpus). Rows whose length
    /// differs from `dims` are skipped with a warning (degenerate data
    /// must not abort the build).
    pub(crate) fn build(
        rows: &[(i64, Vec<f32>)],
        model: &str,
        dims: usize,
        bit_width: usize,
        watermark: i64,
    ) -> Result<Self> {
        let mut backend = B::create(dims, bit_width)?;
        let mut rowids = Vec::with_capacity(rows.len());
        let mut flat = Vec::with_capacity(rows.len() * dims);
        for (rid, v) in rows {
            if v.len() != dims {
                tracing::warn!(
                    "index build: rowid {rid} has {} dims (expected {dims}) — skipped",
                    v.len()
                );
                continue;
            }
            flat.extend_from_slice(&normalize(v));
            rowids.push(*rid);
        }
        if !rowids.is_empty() {
            backend.add_batch(&flat, rowids.len());
        }
        Ok(Self {
            backend,
            rowids,
            meta: IndexMeta {
                model: model.to_string(),
                dims,
                bit_width,
                last_rowid: watermark,
            },
            pending_adds: 0,
            force_persist: true,
            last_persist: Instant::now(),
        })
    }

    /// Load the three files. Errors on any missing/corrupt file or on a
    /// .tv/.ids length or identity disagreement (a crash between the
    /// persist renames) — the caller treats every error as
    /// discard-and-rebuild.
    pub(crate) fn load_from(dir: &Path, store: Store) -> Result<Self> {
        let meta: IndexMeta = serde_json::from_slice(
            &std::fs::read(meta_path(dir, store))
                .with_context(|| format!("read {}", meta_path(dir, store).display()))?,
        )
        .context("parse index meta")?;
        let rowids: Vec<i64> = serde_json::from_slice(
            &std::fs::read(ids_path(dir, store))
                .with_context(|| format!("read {}", ids_path(dir, store).display()))?,
        )
        .context("parse index ids")?;
        let backend = B::load(&tv_path(dir, store))
            .with_context(|| format!("load {}", tv_path(dir, store).display()))?;
        if backend.len() != rowids.len() {
            return Err(anyhow!(
                ".tv has {} vectors but .ids has {} rowids",
                backend.len(),
                rowids.len()
            ));
        }
        if backend.bit_width() != meta.bit_width {
            return Err(anyhow!(
                ".tv bit_width {} disagrees with meta {}",
                backend.bit_width(),
                meta.bit_width
            ));
        }
        if backend.len() > 0 && backend.dims() != meta.dims {
            return Err(anyhow!(
                ".tv dims {} disagrees with meta {}",
                backend.dims(),
                meta.dims
            ));
        }
        Ok(Self {
            backend,
            rowids,
            meta,
            pending_adds: 0,
            force_persist: false,
            last_persist: Instant::now(),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rowids.len()
    }

    /// Append one vector (normalized here) + its rowid; advance the
    /// watermark and the persist-throttle counter.
    pub(crate) fn add(&mut self, rowid: i64, vector: &[f32]) {
        if vector.len() != self.meta.dims {
            tracing::warn!(
                "index add: rowid {rowid} has {} dims (expected {}) — skipped",
                vector.len(),
                self.meta.dims
            );
            return;
        }
        self.backend.add_batch(&normalize(vector), 1);
        self.rowids.push(rowid);
        if rowid > self.meta.last_rowid {
            self.meta.last_rowid = rowid;
        }
        self.pending_adds += 1;
    }

    /// Top-`k` `(rowid, score)` — raw, UNfiltered: the caller post-filters
    /// against SQLite (validity + model) and truncates.
    pub(crate) fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        if k == 0 || self.rowids.is_empty() || query.len() != self.meta.dims {
            return Vec::new();
        }
        let qn = normalize(query);
        self.backend
            .search(&qn, k)
            .into_iter()
            .filter_map(|(slot, score)| self.rowids.get(slot).map(|&rid| (rid, score)))
            .collect()
    }

    pub(crate) fn dirty(&self) -> bool {
        self.force_persist || self.pending_adds > 0
    }

    /// Throttle check: persist when forced, every [`PERSIST_EVERY_ADDS`]
    /// adds, or when ≥1 add has waited [`PERSIST_MAX_AGE`].
    pub(crate) fn persist_due(&self) -> bool {
        self.force_persist
            || self.pending_adds >= PERSIST_EVERY_ADDS
            || (self.pending_adds > 0 && self.last_persist.elapsed() >= PERSIST_MAX_AGE)
    }

    /// Write .tv → .ids → .meta.json, each via tmp + atomic rename (meta
    /// last = commit point). On failure nothing is half-written: the tmp
    /// is removed and the previous generation of every already-renamed
    /// file remains a consistent (older) snapshot. MUST be called without
    /// the DB lock held (file I/O; see module docs on lock order).
    pub(crate) fn persist(&mut self, dir: &Path, store: Store) -> Result<()> {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let tvp = tv_path(dir, store);
        let tmp = tmp_sibling(&tvp);
        if let Err(e) = self.backend.save(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e).with_context(|| format!("save {}", tmp.display()));
        }
        if let Err(e) = std::fs::rename(&tmp, &tvp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e)
                .with_context(|| format!("rename {} -> {}", tmp.display(), tvp.display()));
        }
        write_atomic(&ids_path(dir, store), &serde_json::to_vec(&self.rowids)?)?;
        write_atomic(&meta_path(dir, store), &serde_json::to_vec_pretty(&self.meta)?)?;
        self.pending_adds = 0;
        self.force_persist = false;
        self.last_persist = Instant::now();
        Ok(())
    }
}

// ── Generic lifecycle core (testable feature-off) ───────────────────

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Full rebuild from SQLite into `slot` (model-filtered, validity
/// deliberately ignored — see module docs) + reset the bookkeeping row.
fn rebuild_impl<B: QuantBackend>(
    conn: &Connection,
    store: Store,
    model: &str,
    dims: usize,
    bit_width: usize,
    slot: &mut Option<PersistentIndex<B>>,
    watermark: i64,
) -> Result<()> {
    let rows = pending_rows(conn, store, model, 0)?;
    let idx = PersistentIndex::<B>::build(&rows, model, dims, bit_width, watermark)?;
    record_built(conn, store, model, dims, watermark, now_secs())?;
    idx.backend.prepare();
    *slot = Some(idx);
    Ok(())
}

/// Load-or-rebuild + catch-up for one store. Idempotent: returns fast
/// when `slot` is already live under the current identity. The caller
/// holds the DB lock (this only uses the passed `conn`) and the store's
/// index lock (`slot` is behind it).
pub(crate) fn ensure_loaded_impl<B: QuantBackend>(
    conn: &Connection,
    dir: &Path,
    store: Store,
    model: &str,
    dims: usize,
    bit_width: usize,
    slot: &mut Option<PersistentIndex<B>>,
) -> Result<()> {
    if slot.as_ref().is_some_and(|idx| {
        idx.meta.model == model && idx.meta.dims == dims && idx.meta.bit_width == bit_width
    }) {
        return Ok(());
    }
    *slot = None; // identity changed mid-process (or first use)

    let watermark = max_rowid(conn, store)?;
    let state = read_state(conn, store)?;

    match PersistentIndex::<B>::load_from(dir, store) {
        Ok(mut idx) => {
            match rebuild_needed(&idx.meta, model, dims, bit_width, state.as_ref(), watermark) {
                Some(reason) => {
                    tracing::info!(
                        "{} index discarded ({reason}); rebuilding from SQLite",
                        store.name()
                    );
                    discard_files(dir, store);
                }
                None => {
                    // Catch-up: rows written while no live index was around.
                    if idx.meta.last_rowid < watermark {
                        let missing = pending_rows(conn, store, model, idx.meta.last_rowid)?;
                        for (rid, v) in &missing {
                            idx.add(*rid, v);
                        }
                        // Watermark covers non-matching rows too — they were
                        // CONSIDERED (and excluded by the model filter).
                        idx.meta.last_rowid = watermark;
                        bump_last_indexed(conn, store, watermark)?;
                    }
                    idx.backend.prepare();
                    *slot = Some(idx);
                    return Ok(());
                }
            }
        }
        Err(e) => {
            if any_file_exists(dir, store) {
                tracing::warn!(
                    "{} index load failed ({e:#}); discarding files and rebuilding",
                    store.name()
                );
                discard_files(dir, store);
            }
            // else: first run, nothing on disk — quiet rebuild below.
        }
    }
    rebuild_impl(conn, store, model, dims, bit_width, slot, watermark)
}

/// Search core: ensure-loaded → stale-threshold check (synchronous full
/// rebuild when due — v1) → over-fetched raw `(rowid, score)` hits. The
/// CALLER post-filters against SQLite and truncates to `k`.
//
// 8 args: the lifecycle plumbing (conn/dir/store/identity/slot) IS the
// signature — same call the repo's other connection-level cores make.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_impl<B: QuantBackend>(
    conn: &Connection,
    dir: &Path,
    store: Store,
    qvec: &[f32],
    model: &str,
    bit_width: usize,
    k: usize,
    slot: &mut Option<PersistentIndex<B>>,
) -> Result<Vec<(i64, f32)>> {
    let dims = qvec.len();
    ensure_loaded_impl(conn, dir, store, model, dims, bit_width, slot)?;

    let stale = read_state(conn, store)?.map_or(0, |s| s.stale_count);
    let indexed = slot.as_ref().map_or(0, PersistentIndex::len);
    if stale_rebuild_due(stale, indexed) {
        tracing::info!(
            "{}: stale_count {stale} exceeds 25% of {indexed} indexed rows — synchronous rebuild",
            store.name()
        );
        let watermark = max_rowid(conn, store)?;
        rebuild_impl(conn, store, model, dims, bit_width, slot, watermark)?;
    }

    let idx = slot.as_ref().expect("ensure_loaded_impl populates the slot");
    Ok(idx.search(qvec, k.saturating_mul(OVERFETCH)))
}

// ── TurboVec backend + process-global handles (feature-gated) ───────

#[cfg(feature = "turbovec")]
mod turbo {
    use super::*;
    use parking_lot::Mutex;

    /// [`QuantBackend`] over turbovec's `TurboQuantIndex` (TurboQuant
    /// 2-4 bit compression + SIMD MIPS search). Vectors arrive already
    /// normalized, so the inner-product scores equal cosine similarity.
    pub(crate) struct TurboBackend {
        inner: turbovec::TurboQuantIndex,
    }

    impl QuantBackend for TurboBackend {
        fn create(dims: usize, bit_width: usize) -> Result<Self> {
            Ok(Self {
                inner: turbovec::TurboQuantIndex::new(dims, bit_width)
                    .map_err(|e| anyhow!("turbovec construct: {e}"))?,
            })
        }

        fn load(path: &Path) -> Result<Self> {
            Ok(Self {
                inner: turbovec::TurboQuantIndex::load(path)?,
            })
        }

        fn add_batch(&mut self, flat: &[f32], n: usize) {
            debug_assert_eq!(flat.len(), n * self.inner.dim().max(1));
            let _ = n;
            self.inner.add(flat);
        }

        fn search(&self, normalized_query: &[f32], k: usize) -> Vec<(usize, f32)> {
            if self.inner.is_empty()
                || k == 0
                || normalized_query.len() != self.inner.dim()
            {
                return Vec::new();
            }
            let res = self.inner.search(normalized_query, k.min(self.inner.len()));
            let scores = res.scores_for_query(0);
            let indices = res.indices_for_query(0);
            scores
                .iter()
                .zip(indices.iter())
                // Negative slots are turbovec's pad sentinel — skip.
                .filter_map(|(s, &i)| (i >= 0).then_some((i as usize, *s)))
                .collect()
        }

        fn len(&self) -> usize {
            self.inner.len()
        }
        fn dims(&self) -> usize {
            self.inner.dim()
        }
        fn bit_width(&self) -> usize {
            self.inner.bit_width()
        }
        fn save(&self, path: &Path) -> Result<()> {
            self.inner.write(path).map_err(Into::into)
        }
        fn prepare(&self) {
            self.inner.prepare();
        }
    }

    /// One process-wide slot per store, each behind its OWN mutex (the
    /// second lock in the DB-then-index order — module docs).
    static MEMORIES: Mutex<Option<PersistentIndex<TurboBackend>>> = Mutex::new(None);
    static CHUNKS: Mutex<Option<PersistentIndex<TurboBackend>>> = Mutex::new(None);

    pub(super) fn handle(store: Store) -> &'static Mutex<Option<PersistentIndex<TurboBackend>>> {
        match store {
            Store::Memories => &MEMORIES,
            Store::Chunks => &CHUNKS,
        }
    }
}

// ── Hooks (the seam the write/read paths call) ──────────────────────

/// True when the persistent TurboVec path is the active backend: the
/// `turbovec` feature is compiled in AND the runtime selector resolves
/// TurboVec (which is now the feature-on DEFAULT; `LAMU_VECTOR_BACKEND=
/// brute` still forces brute). Feature-off this is always false, keeping
/// the brute path byte-identical.
pub(crate) fn persistent_active() -> bool {
    matches!(
        crate::vector_index::vector_backend(),
        crate::vector_index::VectorBackend::TurboVec
    )
}

/// Writer hook: a row with `embedding` was just inserted as `rowid`.
/// Call while HOLDING the DB lock (uses `conn` for the bookkeeping bump);
/// pair with [`maybe_persist`] after releasing it. Appends to the live
/// in-memory index when one is loaded under the same identity; when none
/// is loaded the load-time catch-up covers the row instead. Best-effort:
/// never fails the write that produced the row.
pub(crate) fn note_added(
    conn: &Connection,
    store: Store,
    rowid: i64,
    embedding: &[f32],
    model: &str,
) {
    if !persistent_active() || embedding.is_empty() || embedding.len() % 8 != 0 {
        return;
    }
    if let Err(e) = bump_last_indexed(conn, store, rowid) {
        tracing::warn!("{} vector_index_state bump: {e}", store.name());
    }
    #[cfg(feature = "turbovec")]
    {
        let mut guard = turbo::handle(store).lock();
        if let Some(idx) = guard.as_mut() {
            if idx.meta.model == model && idx.meta.dims == embedding.len() {
                idx.add(rowid, embedding);
            } else {
                // Embedder identity changed mid-process: drop the live
                // index; the next search rebuilds under the new identity.
                *guard = None;
            }
        }
    }
    #[cfg(not(feature = "turbovec"))]
    let _ = model;
}

/// Writer hook: `n` indexed rows were expired/replaced (forget,
/// supersede, chunk re-index). The vectors stay in the .tv; this only
/// bumps the stale accounting that drives the rebuild threshold. Call
/// while holding the DB lock. Best-effort.
pub(crate) fn note_stale(conn: &Connection, store: Store, n: i64) {
    if n <= 0 || !persistent_active() {
        return;
    }
    if let Err(e) = bump_stale(conn, store, n) {
        tracing::warn!("{} stale_count bump: {e}", store.name());
    }
}

/// Throttled persist check — call AFTER releasing the DB lock (file I/O
/// runs under the index lock only). No-op when nothing is loaded, the
/// throttle isn't due, or the feature is off.
pub(crate) fn maybe_persist(store: Store) {
    #[cfg(feature = "turbovec")]
    {
        let mut guard = turbo::handle(store).lock();
        if let Some(idx) = guard.as_mut() {
            if idx.persist_due() {
                if let Err(e) = idx.persist(&index_dir(), store) {
                    tracing::warn!("{} index persist failed: {e:#}", store.name());
                }
            }
        }
    }
    #[cfg(not(feature = "turbovec"))]
    let _ = store;
}

/// Force-persist every loaded index regardless of throttle — the
/// on-demand `flush()` (e.g. frontend shutdown). Takes only the index
/// locks; safe anywhere the DB lock is not held by the calling thread…
/// and safe even then (it never touches the DB), just slower.
pub fn flush_all() {
    #[cfg(feature = "turbovec")]
    for store in [Store::Memories, Store::Chunks] {
        let mut guard = turbo::handle(store).lock();
        if let Some(idx) = guard.as_mut() {
            if idx.dirty() {
                if let Err(e) = idx.persist(&index_dir(), store) {
                    tracing::warn!("{} index flush failed: {e:#}", store.name());
                }
            }
        }
    }
}

/// Reader hook: over-fetched raw `(rowid, score)` candidates from the
/// persistent index, or `None` when the persistent path is inactive /
/// unusable for this query (feature off, `LAMU_VECTOR_BACKEND=brute`,
/// dims not a positive multiple of 8) or it errored — the caller then
/// falls back to the per-query scan. Call while HOLDING the DB lock
/// (load/rebuild reads SQLite through `conn`); the returned rowids MUST
/// be post-filtered against SQLite (validity + model) before use.
pub(crate) fn search_persistent(
    conn: &Connection,
    store: Store,
    qvec: &[f32],
    model: &str,
    k: usize,
) -> Option<Vec<(i64, f32)>> {
    if !persistent_active() || k == 0 || qvec.is_empty() || qvec.len() % 8 != 0 {
        return None;
    }
    #[cfg(feature = "turbovec")]
    {
        let mut guard = turbo::handle(store).lock();
        match search_impl(
            conn,
            &index_dir(),
            store,
            qvec,
            model,
            DEFAULT_BIT_WIDTH,
            k,
            &mut guard,
        ) {
            Ok(hits) => Some(hits),
            Err(e) => {
                tracing::warn!(
                    "{} persistent index search failed ({e:#}); falling back to brute scan",
                    store.name()
                );
                *guard = None;
                None
            }
        }
    }
    #[cfg(not(feature = "turbovec"))]
    {
        let _ = (conn, store, model);
        None
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force [`QuantBackend`] stand-in: exact cosine over stored
    /// rows, JSON (de)serialization. Lets every lifecycle test run in the
    /// DEFAULT feature-off build — the turbovec backend itself is
    /// exercised by the feature-gated tests at the bottom.
    #[derive(Serialize, Deserialize)]
    struct BruteBackend {
        dims: usize,
        bit_width: usize,
        rows: Vec<Vec<f32>>,
    }

    impl QuantBackend for BruteBackend {
        fn create(dims: usize, bit_width: usize) -> Result<Self> {
            Ok(Self { dims, bit_width, rows: Vec::new() })
        }
        fn load(path: &Path) -> Result<Self> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        }
        fn add_batch(&mut self, flat: &[f32], n: usize) {
            for i in 0..n {
                self.rows.push(flat[i * self.dims..(i + 1) * self.dims].to_vec());
            }
        }
        fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
            let mut scored: Vec<(usize, f32)> = self
                .rows
                .iter()
                .enumerate()
                .map(|(i, v)| (i, crate::vector_index::cosine(q, v)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            scored
        }
        fn len(&self) -> usize {
            self.rows.len()
        }
        fn dims(&self) -> usize {
            self.dims
        }
        fn bit_width(&self) -> usize {
            self.bit_width
        }
        fn save(&self, path: &Path) -> Result<()> {
            std::fs::write(path, serde_json::to_vec(self)?)?;
            Ok(())
        }
    }

    /// Backend whose save always fails — drives the atomic-persist
    /// "no partial files on failure" test.
    struct FailingSave(BruteBackend);
    impl QuantBackend for FailingSave {
        fn create(dims: usize, bit_width: usize) -> Result<Self> {
            Ok(Self(BruteBackend::create(dims, bit_width)?))
        }
        fn load(_path: &Path) -> Result<Self> {
            Err(anyhow!("simulated load failure"))
        }
        fn add_batch(&mut self, flat: &[f32], n: usize) {
            self.0.add_batch(flat, n);
        }
        fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
            self.0.search(q, k)
        }
        fn len(&self) -> usize {
            self.0.len()
        }
        fn dims(&self) -> usize {
            self.0.dims()
        }
        fn bit_width(&self) -> usize {
            self.0.bit_width()
        }
        fn save(&self, _path: &Path) -> Result<()> {
            Err(anyhow!("simulated save failure"))
        }
    }

    fn open_test_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        crate::migrate::migrate(&mut conn).unwrap();
        conn
    }

    fn meta(model: &str, dims: usize, bw: usize, last: i64) -> IndexMeta {
        IndexMeta { model: model.into(), dims, bit_width: bw, last_rowid: last }
    }

    // ── Pure decision logic ─────────────────────────────────────────

    #[test]
    fn rebuild_needed_matrix() {
        let m = meta("model-a", 8, 4, 10);
        let ok_state = VisState {
            last_indexed_rowid: 10,
            stale_count: 0,
            model: Some("model-a".into()),
            dims: Some(8),
        };
        // Healthy: identity matches, watermark ≤ max → no rebuild.
        assert_eq!(rebuild_needed(&m, "model-a", 8, 4, Some(&ok_state), 10), None);
        // Watermark BELOW max is catch-up, not rebuild.
        assert_eq!(rebuild_needed(&m, "model-a", 8, 4, Some(&ok_state), 99), None);
        // Missing state row is acceptable (post-filter keeps search correct).
        assert_eq!(rebuild_needed(&m, "model-a", 8, 4, None, 10), None);
        // Identity mismatches → rebuild.
        assert!(rebuild_needed(&m, "model-b", 8, 4, None, 10).is_some(), "model");
        assert!(rebuild_needed(&m, "model-a", 16, 4, None, 10).is_some(), "dims");
        assert!(rebuild_needed(&m, "model-a", 8, 2, None, 10).is_some(), "bit_width");
        // Index ahead of the table → the DB was reset → rebuild.
        assert!(rebuild_needed(&m, "model-a", 8, 4, None, 9).is_some(), "ahead");
        // State rows that disagree with the current identity → rebuild.
        let bad_model = VisState { model: Some("model-b".into()), ..ok_state.clone() };
        assert!(rebuild_needed(&m, "model-a", 8, 4, Some(&bad_model), 10).is_some());
        let bad_dims = VisState { dims: Some(16), ..ok_state.clone() };
        assert!(rebuild_needed(&m, "model-a", 8, 4, Some(&bad_dims), 10).is_some());
        // State watermark BEHIND the meta → state was reset → rebuild.
        let behind = VisState { last_indexed_rowid: 5, ..ok_state.clone() };
        assert!(rebuild_needed(&m, "model-a", 8, 4, Some(&behind), 10).is_some());
        // NULL model/dims in state (import-seeded) are not a mismatch.
        let nulls = VisState { model: None, dims: None, ..ok_state };
        assert_eq!(rebuild_needed(&m, "model-a", 8, 4, Some(&nulls), 10), None);
    }

    #[test]
    fn stale_threshold_math() {
        assert!(!stale_rebuild_due(0, 100), "no stale rows → never");
        assert!(!stale_rebuild_due(-3, 100), "negative is clamped");
        assert!(!stale_rebuild_due(25, 100), "exactly 25% is NOT over");
        assert!(stale_rebuild_due(26, 100), "26% is over");
        assert!(stale_rebuild_due(1, 3), "1 of 3 > 25%");
        assert!(!stale_rebuild_due(1, 4), "1 of 4 == 25%");
        assert!(stale_rebuild_due(1, 0), "stale rows but empty index → rebuild");
    }

    // ── Row selection SQL ───────────────────────────────────────────

    #[test]
    fn pending_rows_filters_model_and_rowid_but_not_validity() {
        let conn = open_test_db();
        let blob = |v: &[f32]| crate::rag::vec_to_blob(v);
        // 5 memories: 2 model-a, 1 model-b, 1 unembedded, 1 model-a EXPIRED.
        for (text, emb, model) in [
            ("a1", Some([1.0f32; 8]), Some("model-a")),
            ("b1", Some([0.5f32; 8]), Some("model-b")),
            ("a2", Some([0.0f32; 8]), Some("model-a")),
            ("plain", None, None),
        ] {
            conn.execute(
                "INSERT INTO memories (text, embedding, embedding_model, kind, source, ts, valid_from) \
                 VALUES (?1, ?2, ?3, 'fact', 'manual', 1, 1)",
                params![text, emb.map(|e| blob(&e)), model],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO memories (text, embedding, embedding_model, kind, source, ts, valid_from, valid_until) \
             VALUES ('a3-expired', ?1, 'model-a', 'fact', 'manual', 1, 1, 5)",
            params![blob(&[0.25f32; 8])],
        )
        .unwrap();

        assert_eq!(max_rowid(&conn, Store::Memories).unwrap(), 5);

        // Full select (after 0): model-a rows only, INCLUDING the expired
        // one (validity filtering happens at search time, never at build).
        let all = pending_rows(&conn, Store::Memories, "model-a", 0).unwrap();
        assert_eq!(all.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![1, 3, 5]);
        assert_eq!(all[0].1, vec![1.0f32; 8], "embedding round-trips");

        // Catch-up select (after 3): only the rows above the watermark.
        let missing = pending_rows(&conn, Store::Memories, "model-a", 3).unwrap();
        assert_eq!(missing.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![5]);

        // Chunks go by implicit rowid with the same shape.
        conn.execute(
            "INSERT INTO chunks (path, chunk_idx, content, embedding, embedding_model, mtime) \
             VALUES ('a.rs', 0, 'fn a() {}', ?1, 'model-a', 1)",
            params![blob(&[1.0f32; 8])],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO chunks (path, chunk_idx, content, embedding, embedding_model, mtime) \
             VALUES ('b.rs', 0, 'fn b() {}', ?1, 'model-b', 1)",
            params![blob(&[0.5f32; 8])],
        )
        .unwrap();
        assert_eq!(max_rowid(&conn, Store::Chunks).unwrap(), 2);
        let chunk_rows = pending_rows(&conn, Store::Chunks, "model-a", 0).unwrap();
        assert_eq!(chunk_rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![1]);
    }

    // ── vector_index_state bookkeeping ──────────────────────────────

    #[test]
    fn state_bookkeeping_upserts() {
        let conn = open_test_db();
        assert_eq!(read_state(&conn, Store::Memories).unwrap(), None);

        // bump_last_indexed is a monotonic upsert.
        bump_last_indexed(&conn, Store::Memories, 5).unwrap();
        bump_last_indexed(&conn, Store::Memories, 3).unwrap(); // must NOT regress
        let s = read_state(&conn, Store::Memories).unwrap().unwrap();
        assert_eq!(s.last_indexed_rowid, 5);
        assert_eq!(s.stale_count, 0);

        // bump_stale accumulates.
        bump_stale(&conn, Store::Memories, 2).unwrap();
        bump_stale(&conn, Store::Memories, 3).unwrap();
        let s = read_state(&conn, Store::Memories).unwrap().unwrap();
        assert_eq!(s.stale_count, 5);
        assert_eq!(s.last_indexed_rowid, 5, "stale bump must not touch the watermark");

        // record_built adopts identity, zeroes stale, sets watermark.
        record_built(&conn, Store::Memories, "model-a", 8, 42, 1000).unwrap();
        let s = read_state(&conn, Store::Memories).unwrap().unwrap();
        assert_eq!(s.last_indexed_rowid, 42);
        assert_eq!(s.stale_count, 0);
        assert_eq!(s.model.as_deref(), Some("model-a"));
        assert_eq!(s.dims, Some(8));
        let built_at: i64 = conn
            .query_row(
                "SELECT built_at FROM vector_index_state WHERE store = 'memories'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(built_at, 1000);

        // Per-store isolation: chunks row untouched.
        assert_eq!(read_state(&conn, Store::Chunks).unwrap(), None);
    }

    // ── Atomic persist + load round-trip ────────────────────────────

    fn sep_rows() -> Vec<(i64, Vec<f32>)> {
        // Well-separated 8-dim directions keyed by rowid.
        let mut rows = Vec::new();
        for (i, axis) in [0usize, 1, 2, 3].iter().enumerate() {
            let mut v = vec![0.0f32; 8];
            v[*axis] = 1.0;
            rows.push(((i + 1) as i64 * 10, v)); // rowids 10, 20, 30, 40
        }
        rows
    }

    fn axis_query(axis: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; 8];
        q[axis] = 2.0; // un-normalized on purpose — search normalizes
        q
    }

    #[test]
    fn persist_then_load_round_trips_and_leaves_no_tmp() {
        let td = tempfile::tempdir().unwrap();
        let rows = sep_rows();
        let mut idx =
            PersistentIndex::<BruteBackend>::build(&rows, "model-a", 8, 4, 40).unwrap();
        assert!(idx.force_persist, "fresh build wants a persist");
        assert!(idx.persist_due());
        idx.persist(td.path(), Store::Memories).unwrap();
        assert!(!idx.persist_due(), "persist resets the throttle");

        // All three files exist; no tmp leftovers anywhere.
        for f in ["memories.tv", "memories.ids", "memories.meta.json"] {
            assert!(td.path().join(f).exists(), "missing {f}");
        }
        let tmp_left = std::fs::read_dir(td.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains(".tmp."));
        assert!(!tmp_left, "no tmp files after a successful persist");

        let loaded = PersistentIndex::<BruteBackend>::load_from(td.path(), Store::Memories).unwrap();
        assert_eq!(loaded.meta, meta("model-a", 8, 4, 40));
        assert_eq!(loaded.len(), 4);
        // Search maps slots back to the ORIGINAL rowids.
        let hits = loaded.search(&axis_query(2), 1);
        assert_eq!(hits[0].0, 30, "axis-2 row carries rowid 30");
        assert!(hits[0].1 > 0.99);
    }

    #[test]
    fn failed_persist_leaves_no_partial_files() {
        let td = tempfile::tempdir().unwrap();
        let mut idx =
            PersistentIndex::<FailingSave>::build(&sep_rows(), "model-a", 8, 4, 40).unwrap();
        assert!(idx.persist(td.path(), Store::Memories).is_err());
        // NOTHING was published — .tv save failed before any rename, so
        // ids/meta were never attempted (write order: .tv → .ids → .meta).
        let entries: Vec<String> = std::fs::read_dir(td.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(entries.is_empty(), "no partial/tmp files on failure: {entries:?}");
        assert!(idx.dirty(), "failed persist keeps the index dirty");
    }

    #[test]
    fn load_rejects_ids_index_length_mismatch() {
        let td = tempfile::tempdir().unwrap();
        let mut idx =
            PersistentIndex::<BruteBackend>::build(&sep_rows(), "model-a", 8, 4, 40).unwrap();
        idx.persist(td.path(), Store::Memories).unwrap();
        // Corrupt the .ids (simulates a crash between the persist renames).
        std::fs::write(td.path().join("memories.ids"), b"[10, 20]").unwrap();
        let err = PersistentIndex::<BruteBackend>::load_from(td.path(), Store::Memories)
            .err()
            .expect("length mismatch must fail the load")
            .to_string();
        assert!(err.contains("4 vectors but .ids has 2"), "{err}");
    }

    // ── Throttle math ───────────────────────────────────────────────

    #[test]
    fn persist_throttle_every_32_adds_or_30s() {
        let td = tempfile::tempdir().unwrap();
        let mut idx =
            PersistentIndex::<BruteBackend>::build(&[], "model-a", 8, 4, 0).unwrap();
        idx.persist(td.path(), Store::Memories).unwrap(); // clear the fresh-build force

        assert!(!idx.persist_due(), "clean index → not due");
        for i in 0..(PERSIST_EVERY_ADDS - 1) {
            idx.add(i as i64 + 1, &[1.0; 8]);
        }
        assert!(!idx.persist_due(), "31 adds → below the add threshold");
        idx.add(PERSIST_EVERY_ADDS as i64, &[1.0; 8]);
        assert!(idx.persist_due(), "32nd add trips the threshold");
        idx.persist(td.path(), Store::Memories).unwrap();
        assert!(!idx.persist_due());

        // Time leg: one add + an old last_persist stamp.
        idx.add(99, &[1.0; 8]);
        assert!(!idx.persist_due(), "1 fresh add → not due yet");
        idx.last_persist = Instant::now() - PERSIST_MAX_AGE;
        assert!(idx.persist_due(), "30s with a pending add → due");

        // The time leg alone never fires with zero pending adds.
        idx.persist(td.path(), Store::Memories).unwrap();
        idx.last_persist = Instant::now() - PERSIST_MAX_AGE;
        assert!(!idx.persist_due(), "stale clock but nothing pending → not due");
    }

    // ── Lifecycle: load-or-rebuild, catch-up, stale rebuild ─────────

    /// Insert one embedded memory row, returning its rowid.
    fn insert_mem(conn: &Connection, emb: &[f32], model: &str) -> i64 {
        conn.execute(
            "INSERT INTO memories (text, embedding, embedding_model, kind, source, ts, valid_from) \
             VALUES ('t', ?1, ?2, 'fact', 'manual', 1, 1)",
            params![crate::rag::vec_to_blob(emb), model],
        )
        .unwrap();
        conn.last_insert_rowid()
    }

    #[test]
    fn ensure_loaded_builds_persists_then_catches_up_offline_rows() {
        let conn = open_test_db();
        let td = tempfile::tempdir().unwrap();
        let id1 = insert_mem(&conn, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "m");
        let id2 = insert_mem(&conn, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "m");

        // First use: nothing on disk → full rebuild from SQLite.
        let mut slot: Option<PersistentIndex<BruteBackend>> = None;
        ensure_loaded_impl(&conn, td.path(), Store::Memories, "m", 8, 4, &mut slot).unwrap();
        let idx = slot.as_mut().unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.meta.last_rowid, 2);
        // The build recorded its identity + watermark in vector_index_state.
        let s = read_state(&conn, Store::Memories).unwrap().unwrap();
        assert_eq!((s.last_indexed_rowid, s.model.as_deref()), (2, Some("m")));
        idx.persist(td.path(), Store::Memories).unwrap();

        // "Offline" insert: a row written while no index was live.
        let id3 = insert_mem(&conn, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "m");
        // …plus a row another model wrote (considered but excluded).
        insert_mem(&conn, &[0.5; 8], "other-model");

        // Fresh process: load from disk + catch-up.
        let mut slot2: Option<PersistentIndex<BruteBackend>> = None;
        ensure_loaded_impl(&conn, td.path(), Store::Memories, "m", 8, 4, &mut slot2).unwrap();
        let idx2 = slot2.as_ref().unwrap();
        assert_eq!(idx2.len(), 3, "catch-up adds ONLY the matching-model row");
        assert_eq!(idx2.meta.last_rowid, 4, "watermark covers the excluded row too");
        let s = read_state(&conn, Store::Memories).unwrap().unwrap();
        assert_eq!(s.last_indexed_rowid, 4);

        let hits = idx2.search(&axis_query(2), 1);
        assert_eq!(hits[0].0, id3, "caught-up row is searchable");
        let _ = (id1, id2);
    }

    #[test]
    fn ensure_loaded_rebuilds_on_identity_mismatch() {
        let conn = open_test_db();
        let td = tempfile::tempdir().unwrap();
        insert_mem(&conn, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "model-a");
        let b_id = insert_mem(&conn, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "model-b");

        let mut slot: Option<PersistentIndex<BruteBackend>> = None;
        ensure_loaded_impl(&conn, td.path(), Store::Memories, "model-a", 8, 4, &mut slot)
            .unwrap();
        assert_eq!(slot.as_ref().unwrap().len(), 1);
        slot.as_mut().unwrap().persist(td.path(), Store::Memories).unwrap();

        // Same slot, NEW model → in-memory discarded, on-disk meta
        // mismatch detected, files discarded, rebuilt with model-b rows.
        ensure_loaded_impl(&conn, td.path(), Store::Memories, "model-b", 8, 4, &mut slot)
            .unwrap();
        let idx = slot.as_ref().unwrap();
        assert_eq!(idx.meta.model, "model-b");
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.search(&axis_query(1), 1)[0].0, b_id);
        let s = read_state(&conn, Store::Memories).unwrap().unwrap();
        assert_eq!(s.model.as_deref(), Some("model-b"), "state re-adopted");
    }

    #[test]
    fn ensure_loaded_rebuilds_on_corrupt_files() {
        let conn = open_test_db();
        let td = tempfile::tempdir().unwrap();
        insert_mem(&conn, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "m");
        std::fs::write(td.path().join("memories.meta.json"), b"not json").unwrap();
        std::fs::write(td.path().join("memories.tv"), b"garbage").unwrap();

        let mut slot: Option<PersistentIndex<BruteBackend>> = None;
        ensure_loaded_impl(&conn, td.path(), Store::Memories, "m", 8, 4, &mut slot).unwrap();
        assert_eq!(slot.as_ref().unwrap().len(), 1, "rebuilt from SQLite");
    }

    #[test]
    fn search_impl_overfetches_and_runs_stale_rebuild() {
        let conn = open_test_db();
        let td = tempfile::tempdir().unwrap();
        for axis in 0..4 {
            let mut v = vec![0.0f32; 8];
            v[axis] = 1.0;
            insert_mem(&conn, &v, "m");
        }
        let mut slot: Option<PersistentIndex<BruteBackend>> = None;

        // k=1 over-fetches ×4 → all 4 rows come back raw.
        let hits = search_impl(
            &conn, td.path(), Store::Memories, &axis_query(0), "m", 4, 1, &mut slot,
        )
        .unwrap();
        assert_eq!(hits.len(), 4, "k * OVERFETCH raw candidates");
        assert_eq!(hits[0].0, 1, "best candidate first");

        // Stale 2 of 4 (> 25%) → the next search rebuilds synchronously
        // and zeroes the counter. (Rows still exist in SQLite, so the
        // rebuilt index has the same 4 rows — the ACCOUNTING resets.)
        bump_stale(&conn, Store::Memories, 2).unwrap();
        let built_before: i64 = conn
            .query_row(
                "SELECT COALESCE(built_at, 0) FROM vector_index_state WHERE store='memories'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let hits = search_impl(
            &conn, td.path(), Store::Memories, &axis_query(0), "m", 4, 1, &mut slot,
        )
        .unwrap();
        assert_eq!(hits.len(), 4);
        let s = read_state(&conn, Store::Memories).unwrap().unwrap();
        assert_eq!(s.stale_count, 0, "rebuild zeroes the stale counter");
        let built_after: i64 = conn
            .query_row(
                "SELECT COALESCE(built_at, 0) FROM vector_index_state WHERE store='memories'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(built_after >= built_before, "rebuild stamped built_at");

        // Below threshold: 1 of 4 (= 25%) does NOT rebuild.
        bump_stale(&conn, Store::Memories, 1).unwrap();
        search_impl(&conn, td.path(), Store::Memories, &axis_query(0), "m", 4, 1, &mut slot)
            .unwrap();
        let s = read_state(&conn, Store::Memories).unwrap().unwrap();
        assert_eq!(s.stale_count, 1, "at-threshold stale count survives");
    }

    // ── TurboVec round-trips (feature-gated) ────────────────────────
    //
    // LINK-GAP NOTE: `cargo test --features turbovec` used to die at link
    // time (`undefined symbol: cblas_sgemm`) because openblas-src's link
    // directives did not reach test binaries. lamu-memory's build.rs now
    // re-emits `-lopenblas` when the feature is on, which fixed it on this
    // machine (system OpenBLAS via pacman). If these tests ever fail to
    // LINK on another setup, that's the documented gap resurfacing — the
    // lifecycle logic above is fully covered feature-off; only the
    // turbovec quantization round-trip below would lose coverage.

    /// Quantized round-trip: write/load/search parity vs brute top-1 on a
    /// clearly-separated synthetic set.
    #[cfg(feature = "turbovec")]
    #[test]
    fn turbo_round_trip_matches_brute_top1() {
        use super::turbo::TurboBackend;
        let td = tempfile::tempdir().unwrap();
        let rows = sep_rows();
        let mut idx =
            PersistentIndex::<TurboBackend>::build(&rows, "m", 8, DEFAULT_BIT_WIDTH, 40)
                .unwrap();
        idx.persist(td.path(), Store::Memories).unwrap();
        let loaded =
            PersistentIndex::<TurboBackend>::load_from(td.path(), Store::Memories).unwrap();
        assert_eq!(loaded.len(), 4);

        let brute = PersistentIndex::<BruteBackend>::build(&rows, "m", 8, 4, 40).unwrap();
        for axis in 0..4 {
            let q = axis_query(axis);
            let t = loaded.search(&q, 1);
            let b = brute.search(&q, 1);
            assert_eq!(t[0].0, b[0].0, "axis {axis}: turbovec top-1 == brute top-1");
            assert!(t[0].1.is_finite());
        }
    }

    /// Catch-up after an offline insert, on the REAL quantized backend.
    #[cfg(feature = "turbovec")]
    #[test]
    fn turbo_catch_up_after_offline_insert() {
        use super::turbo::TurboBackend;
        let conn = open_test_db();
        let td = tempfile::tempdir().unwrap();
        insert_mem(&conn, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "m");
        let mut slot: Option<PersistentIndex<TurboBackend>> = None;
        ensure_loaded_impl(&conn, td.path(), Store::Memories, "m", 8, DEFAULT_BIT_WIDTH, &mut slot)
            .unwrap();
        slot.as_mut().unwrap().persist(td.path(), Store::Memories).unwrap();

        let id2 = insert_mem(&conn, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "m");
        let mut slot2: Option<PersistentIndex<TurboBackend>> = None;
        ensure_loaded_impl(&conn, td.path(), Store::Memories, "m", 8, DEFAULT_BIT_WIDTH, &mut slot2)
            .unwrap();
        let idx = slot2.as_ref().unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.search(&axis_query(3), 1)[0].0, id2, "offline row searchable");
    }

    /// A dims change (new embedder) discards + rebuilds the quantized index.
    #[cfg(feature = "turbovec")]
    #[test]
    fn turbo_rebuild_on_dims_change() {
        use super::turbo::TurboBackend;
        let conn = open_test_db();
        let td = tempfile::tempdir().unwrap();
        insert_mem(&conn, &[1.0; 8], "m");
        let mut slot: Option<PersistentIndex<TurboBackend>> = None;
        ensure_loaded_impl(&conn, td.path(), Store::Memories, "m", 8, DEFAULT_BIT_WIDTH, &mut slot)
            .unwrap();
        slot.as_mut().unwrap().persist(td.path(), Store::Memories).unwrap();
        assert_eq!(slot.as_ref().unwrap().meta.dims, 8);

        // The embedder now produces 16-dim vectors (same model name to
        // isolate the dims check); the 8-dim row no longer matches and is
        // skipped at build time → empty 16-dim index, meta re-stamped.
        let mut slot2: Option<PersistentIndex<TurboBackend>> = None;
        ensure_loaded_impl(&conn, td.path(), Store::Memories, "m", 16, DEFAULT_BIT_WIDTH, &mut slot2)
            .unwrap();
        let idx = slot2.as_ref().unwrap();
        assert_eq!(idx.meta.dims, 16, "meta re-stamped for the new dims");
        assert_eq!(idx.len(), 0, "old-dims rows skipped (reembed converges them)");
    }
}
