//! Repo retrieval (RAG): ripgrep + optional embedding fallback.
//!
//! ## Modes
//!
//! - **ripgrep** — fixed-string / regex grep across `git ls-files`.
//!   Instant, zero-setup, lossy on semantic recall but 90% of typical
//!   "find the function" queries are spelled exactly.
//! - **semantic** — query embedding via the embedder chain (ADR 0030:
//!   local registry model first, OpenAI escape hatch / fallback — see
//!   `crate::embedder::resolve`). Brute-force cosine-sim against the
//!   `chunks` table of the unified `lamu.db` (ADR 0028; previously its
//!   own `embeddings.db`), filtered to rows embedded with the CURRENT
//!   embedder's model so mixed-model vectors never rank against each
//!   other. Index is built on first semantic query if missing; explicit
//!   `index_repo` tool also builds it on demand.
//! - **auto** — ripgrep first; if it returns < k hits, augment with
//!   semantic. Default.
//!
//! ## Why brute-force cosine
//!
//! lamu-rs is small (~10K lines, ~150 files at typical chunk size).
//! 150 * 1536-dim cosine = 230K floats per query = sub-millisecond on
//! any modern CPU. HNSW / sqlite-vss / DuckDB-vss are overkill until
//! the index hits 10K+ rows.

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use crate::vector_index::{vector_backend, BruteForceCosine, Scored, VectorBackend, VectorIndex};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

/// The pre-ADR-0030 hardcoded model — every legacy row was embedded
/// with it, so the one-time import (store.rs) backfills
/// `embedding_model` with this. Live writes stamp whatever
/// `crate::embedder::resolve()` returns instead.
pub(crate) use crate::embedder::OPENAI_EMBED_MODEL as EMBED_MODEL;

/// This module's persistent-index store id (ADR 0031).
const TV_STORE: crate::tv_store::Store = crate::tv_store::Store::Chunks;

/// Chunk size in characters. ~1KB hits a sweet spot: large enough to
/// preserve local context, small enough to keep per-chunk embeddings
/// meaningful + the index manageable.
const CHUNK_BYTES: usize = 1024;

/// Cap on rg hits returned from a ripgrep search.
const RIPGREP_LIMIT: usize = 50;

#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Ripgrep,
    Semantic,
    Auto,
}

impl SearchMode {
    pub fn parse(s: &str) -> Self {
        match s {
            "ripgrep" => SearchMode::Ripgrep,
            "semantic" => SearchMode::Semantic,
            _ => SearchMode::Auto,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchHit {
    pub path: String,
    pub line: Option<usize>,
    pub snippet: String,
    pub score: Option<f32>,
    pub source: &'static str,
}

// ── Index DB handle ────────────────────────────────────────────────

/// The chunk index's connection — a thin delegate to the unified
/// `lamu.db` singleton (`crate::store`, ADR 0028). Kept under its
/// historical name so the call sites below read unchanged.
fn index_db() -> Result<Arc<Mutex<Connection>>> {
    crate::store::shared_handle()
}

// ── Ripgrep mode ───────────────────────────────────────────────────

pub fn ripgrep_search(query: &str, repo: &Path, k: usize) -> Result<Vec<SearchHit>> {
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let out = Command::new("rg")
        .current_dir(repo)
        .args([
            "-n",
            "--no-heading",
            "--color=never",
            "--max-count",
            "5",
            "-i",
            "--", // end of options: a query starting with '-' is a pattern, not a flag
            query,
        ])
        .output();
    let out = match out {
        Ok(o) => o,
        Err(_) => return Ok(Vec::new()), // rg not installed
    };
    if !out.status.success() {
        return Ok(Vec::new()); // exit 1 = no matches
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut hits = Vec::new();
    for line in stdout.lines() {
        // Format: path:line:content
        let mut parts = line.splitn(3, ':');
        let Some(path) = parts.next() else { continue };
        let Some(lineno) = parts.next().and_then(|s| s.parse::<usize>().ok()) else {
            continue;
        };
        let Some(content) = parts.next() else { continue };
        hits.push(SearchHit {
            path: path.to_string(),
            line: Some(lineno),
            snippet: truncate_utf8(content, 200),
            score: None,
            source: "ripgrep",
        });
        if hits.len() >= k.min(RIPGREP_LIMIT) {
            break;
        }
    }
    Ok(hits)
}

// ── Semantic mode ──────────────────────────────────────────────────
//
// The OpenAI plumbing (key resolution, pooled client, embed_one /
// embed_batch) moved to `crate::embedder` (ADR 0030); semantic mode now
// resolves whatever the embedder chain provides.

pub(crate) fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    buf
}

pub(crate) fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(b.len() / 4);
    for chunk in b.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

/// Walk `git ls-files` from `repo`, chunk each text file, and embed
/// the chunks. Existing chunks for unchanged files (matching mtime)
/// are skipped. Returns count of chunks indexed.
///
/// Each row is stamped with the current embedder's model
/// (`embedding_model`), and the `embedding_stores` bookkeeping is
/// upserted for the 'chunks' store (ADR 0030).
pub async fn index_repo(repo: &Path, force: bool) -> Result<usize> {
    let embedder = crate::embedder::resolve().ok_or_else(|| {
        anyhow!(
            "no embedder available — register a local embedding model (capability \
             'embedding') or set OPENAI_API_KEY. Use mode='ripgrep' for grep-only search."
        )
    })?;
    let files = git_ls_files(repo)?;
    let arc = index_db()?;

    // Collect (path, chunk_idx, content, mtime) tuples that need
    // embedding. We embed in batch outside the lock.
    let mut to_embed: Vec<(String, usize, String, i64)> = Vec::new();
    {
        let conn = arc.lock();
        for path in &files {
            let abs = repo.join(path);
            if !abs.is_file() {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&abs) else {
                continue;
            };
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            // Skip files that are already indexed at this mtime.
            if !force {
                let existing_mtime: Option<i64> = conn
                    .query_row(
                        "SELECT mtime FROM chunks WHERE path = ? LIMIT 1",
                        params![path],
                        |r| r.get(0),
                    )
                    .ok();
                if existing_mtime == Some(mtime) {
                    continue;
                }
            }
            // Skip binary / non-text files. Heuristic: read up to 8KB,
            // look for NUL.
            let Ok(body) = std::fs::read_to_string(&abs) else {
                continue;
            };
            if body.contains('\0') {
                continue;
            }
            // Stale chunks for this path are cleared inside the final write
            // transaction (after embed succeeds), NOT here — see #7.
            for (idx, chunk) in chunk_text(&body, CHUNK_BYTES).into_iter().enumerate() {
                to_embed.push((path.clone(), idx, chunk, mtime));
            }
        }
    }

    if to_embed.is_empty() {
        return Ok(0);
    }

    // Batch embed via the chain (each impl handles its own batching).
    let texts: Vec<String> = to_embed.iter().map(|(_, _, c, _)| c.clone()).collect();
    let embeddings = embedder.embed(&texts).await?;
    if embeddings.len() != to_embed.len() {
        return Err(anyhow!(
            "embed count mismatch: requested {}, got {}",
            to_embed.len(),
            embeddings.len()
        ));
    }
    let model = embedder.identity().model;
    let dims = embeddings.first().map(|e| e.len()).unwrap_or(0);

    // Bulk replace under one transaction: clear each affected path's old
    // chunks, THEN insert the freshly embedded ones — atomic with embed
    // SUCCESS. The DELETE used to run eagerly in autocommit BEFORE the
    // await on embed_batch above; a transient embed failure (429/timeout/
    // network) then committed the deletes with no inserts, silently
    // dropping the path's chunks until a later successful re-index (#7).
    let mut conn = arc.lock();
    let tx = conn.transaction()?;
    let mut cleared: std::collections::HashSet<&str> = std::collections::HashSet::new();
    // Rows replaced by this re-index are STALE in the persistent .tv
    // (their vectors stay; searches post-filter them) — count them for
    // the stale accounting (ADR 0031).
    let mut replaced = 0i64;
    for (path, _, _, _) in &to_embed {
        if cleared.insert(path.as_str()) {
            replaced += tx.execute("DELETE FROM chunks WHERE path = ?", params![path])? as i64;
        }
    }
    // Rowids of the fresh inserts, parallel to `embeddings` — the
    // persistent-index hooks below need them.
    let mut inserted_rowids: Vec<i64> = Vec::with_capacity(to_embed.len());
    for ((path, idx, content, mtime), emb) in to_embed.iter().zip(embeddings.iter()) {
        tx.execute(
            "INSERT OR REPLACE INTO chunks (path, chunk_idx, content, embedding, embedding_model, mtime) VALUES (?, ?, ?, ?, ?, ?)",
            params![path, *idx as i64, content, vec_to_blob(emb), model, mtime],
        )?;
        inserted_rowids.push(tx.last_insert_rowid());
    }
    // Per-store identity bookkeeping (ADR 0030) — same transaction, so
    // the rows + the store row land atomically.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    crate::store::record_store_identity(&tx, "chunks", &model, dims, now);
    tx.commit()?;

    // Persistent-index hooks AFTER commit (rows are durable) but still
    // under the conn guard — note_* writes vector_index_state through the
    // same connection (lock order: DB first, index second; ADR 0031).
    for (rowid, emb) in inserted_rowids.iter().zip(embeddings.iter()) {
        crate::tv_store::note_added(&conn, TV_STORE, *rowid, emb, &model);
    }
    crate::tv_store::note_stale(&conn, TV_STORE, replaced);
    drop(conn); // persist (file I/O) must not run under the DB lock
    crate::tv_store::maybe_persist(TV_STORE);

    Ok(to_embed.len())
}

/// Build the selected [`VectorIndex`] backend from already-loaded rows and
/// return the top-`k` scored payloads. The Brute branch is the unchanged
/// default path; the TurboVec branch is reachable only with the `turbovec`
/// feature compiled in AND `LAMU_VECTOR_BACKEND=turbovec` at runtime (see
/// [`vector_backend`]). Both branches add the same rows + run the same
/// `.search`, so the caller's result-mapping is backend-agnostic.
fn search_rows(
    rows: Vec<(Vec<f32>, (String, String))>,
    qvec: &[f32],
    k: usize,
) -> Vec<Scored<(String, String)>> {
    fn fill<I: VectorIndex<(String, String)>>(
        mut index: I,
        rows: Vec<(Vec<f32>, (String, String))>,
        qvec: &[f32],
        k: usize,
    ) -> Vec<Scored<(String, String)>> {
        for (emb, payload) in rows {
            index.add(emb, payload);
        }
        index.search(qvec, k)
    }
    match vector_backend() {
        VectorBackend::Brute => fill(BruteForceCosine::new(), rows, qvec, k),
        VectorBackend::TurboVec => {
            #[cfg(feature = "turbovec")]
            {
                fill(crate::vector_index::TurboVecIndex::new(), rows, qvec, k)
            }
            // `vector_backend()` never returns TurboVec without the feature,
            // but keep the arm total if that ever changes.
            #[cfg(not(feature = "turbovec"))]
            {
                fill(BruteForceCosine::new(), rows, qvec, k)
            }
        }
    }
}

pub async fn semantic_search(query: &str, k: usize) -> Result<Vec<SearchHit>> {
    let embedder = crate::embedder::resolve().ok_or_else(|| {
        anyhow!(
            "no embedder available — register a local embedding model (capability \
             'embedding') or set OPENAI_API_KEY."
        )
    })?;
    let mut qvecs = embedder.embed(std::slice::from_ref(&query.to_string())).await?;
    let qvec = qvecs
        .pop()
        .ok_or_else(|| anyhow!("embedder returned no vector for the query"))?;
    let model = embedder.identity().model;
    let arc = index_db()?;
    let hits = {
        let conn = arc.lock();
        // ADR 0031: persistent index first — raw over-fetched candidates,
        // hydrated + model-filtered by rowid below. `None` (persistent
        // path inactive / dims not %8 / error) → the per-query scan.
        if let Some(raw) =
            crate::tv_store::search_persistent(&conn, TV_STORE, &qvec, &model, k)
        {
            hydrate_chunk_hits(&conn, &raw, &model, k)?
        } else {
            // Model filter (ADR 0030): only rank rows embedded with the
            // CURRENT model — vectors from different models live in
            // different spaces.
            let mut stmt = conn.prepare(
                "SELECT path, chunk_idx, content, embedding FROM chunks WHERE embedding_model = ?1",
            )?;
            let rows = stmt.query_map(params![model], |r| {
                let path: String = r.get(0)?;
                let content: String = r.get(2)?; // chunk_idx (col 1) not needed in SearchHit
                let emb_blob: Vec<u8> = r.get(3)?;
                Ok((path, content, emb_blob))
            })?;
            // SEAM: swap BruteForceCosine for an ANN/quantized index when the
            // corpus outgrows brute-force (see crate::vector_index for the why).
            // Load the rows once, then build whichever backend the selector picks;
            // the result-mapping below is identical for both branches.
            let mut loaded: Vec<(Vec<f32>, (String, String))> = Vec::new();
            for row in rows {
                let (path, content, emb_blob) = row?;
                loaded.push((blob_to_vec(&emb_blob), (path, content)));
            }
            search_rows(loaded, &qvec, k)
                .into_iter()
                .map(|hit| {
                    let (path, content) = hit.payload;
                    SearchHit {
                        path,
                        line: None,
                        snippet: truncate_utf8(&content, 400),
                        score: Some(hit.score),
                        source: "semantic",
                    }
                })
                .collect()
        }
    }; // DB lock released — persist (file I/O) must not run under it.
    crate::tv_store::maybe_persist(TV_STORE);
    Ok(hits)
}

/// Hydrate raw persistent-index candidates `(rowid, score)` into
/// [`SearchHit`]s: SELECT by rowid with the model filter (a stale rowid
/// whose row was deleted by a re-index simply doesn't hydrate), keep the
/// index's score order, dedup by rowid, truncate to `k` (ADR 0031).
fn hydrate_chunk_hits(
    conn: &Connection,
    raw: &[(i64, f32)],
    model: &str,
    k: usize,
) -> Result<Vec<SearchHit>> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = vec!["?"; raw.len()].join(",");
    let sql = format!(
        "SELECT rowid, path, content FROM chunks \
         WHERE rowid IN ({placeholders}) AND embedding_model = ?{model_pos}",
        model_pos = raw.len() + 1
    );
    let mut stmt = conn.prepare(&sql)?;
    let bind: Vec<rusqlite::types::Value> = raw
        .iter()
        .map(|(id, _)| rusqlite::types::Value::Integer(*id))
        .chain(std::iter::once(rusqlite::types::Value::Text(model.to_string())))
        .collect();
    let mapped = stmt.query_map(rusqlite::params_from_iter(bind), |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
    })?;
    let mut by_rowid: std::collections::HashMap<i64, (String, String)> =
        std::collections::HashMap::new();
    for row in mapped {
        let (rowid, path, content) = row?;
        by_rowid.insert(rowid, (path, content));
    }
    let mut hits = Vec::new();
    for (rowid, score) in raw {
        let Some((path, content)) = by_rowid.remove(rowid) else {
            continue; // stale / wrong-model rowid — filtered out here
        };
        hits.push(SearchHit {
            path,
            line: None,
            snippet: truncate_utf8(&content, 400),
            score: Some(*score),
            source: "semantic",
        });
        if hits.len() >= k {
            break;
        }
    }
    Ok(hits)
}

// ── Public dispatch ────────────────────────────────────────────────

pub async fn search(
    query: &str,
    mode: SearchMode,
    k: usize,
    repo: &Path,
) -> Result<Vec<SearchHit>> {
    match mode {
        SearchMode::Ripgrep => ripgrep_search(query, repo, k),
        SearchMode::Semantic => semantic_search(query, k).await,
        SearchMode::Auto => {
            let mut hits = ripgrep_search(query, repo, k)?;
            if hits.len() < k && crate::embedder::resolve().is_some() {
                if let Ok(extra) = semantic_search(query, k - hits.len()).await {
                    // Avoid path collisions — semantic chunks may
                    // already appear in the ripgrep results; dedupe by
                    // path.
                    let seen: std::collections::HashSet<String> =
                        hits.iter().map(|h| h.path.clone()).collect();
                    for h in extra {
                        if !seen.contains(&h.path) {
                            hits.push(h);
                        }
                    }
                }
            }
            Ok(hits)
        }
    }
}

// ── helpers ────────────────────────────────────────────────────────

fn git_ls_files(repo: &Path) -> Result<Vec<String>> {
    let out = Command::new("git")
        .current_dir(repo)
        .args(["ls-files"])
        .output()?;
    if !out.status.success() {
        return Err(anyhow!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(|s| s.to_string())
        .collect())
}

fn chunk_text(body: &str, max_bytes: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in body.lines() {
        if current.len() + line.len() + 1 > max_bytes && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn truncate_utf8(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut cut = max;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &s[..cut])
}

#[cfg(test)]
mod tests {
    use super::*;

    // cosine() + its unit tests moved to crate::vector_index (the seam).

    #[test]
    fn vec_blob_round_trip() {
        let v = vec![1.0_f32, -2.5, 3.14, 0.0];
        let b = vec_to_blob(&v);
        let back = blob_to_vec(&b);
        assert_eq!(v, back);
    }

    #[test]
    fn chunk_text_respects_max() {
        let body = "line1\nline2\nline3\nline4\n";
        let chunks = chunk_text(body, 12); // tiny cap forces splits
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.len() <= 12 + 6); // each chunk is one line at most past cap
        }
    }

    #[test]
    fn chunk_text_preserves_full_body() {
        let body = "abcde\nfghij\nklmno\n";
        let chunks = chunk_text(body, 1024);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], body);
    }

    #[test]
    fn search_mode_parse() {
        assert!(matches!(SearchMode::parse("ripgrep"), SearchMode::Ripgrep));
        assert!(matches!(SearchMode::parse("semantic"), SearchMode::Semantic));
        assert!(matches!(SearchMode::parse("auto"), SearchMode::Auto));
        assert!(matches!(SearchMode::parse(""), SearchMode::Auto));
        assert!(matches!(SearchMode::parse("unknown"), SearchMode::Auto));
    }
}
