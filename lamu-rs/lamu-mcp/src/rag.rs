//! Repo retrieval (RAG): ripgrep + optional cloud-embedding fallback.
//!
//! ## Modes
//!
//! - **ripgrep** — fixed-string / regex grep across `git ls-files`.
//!   Instant, zero-setup, lossy on semantic recall but 90% of typical
//!   "find the function" queries are spelled exactly.
//! - **semantic** — query embedding via OpenAI's
//!   `text-embedding-3-small` (cheap, ~$0.02/M tokens). Brute-force
//!   cosine-sim against an on-disk index at
//!   `~/.local/share/lamu/embeddings.db`. Index is built on first
//!   semantic query if missing; explicit `index_repo` tool also
//!   builds it on demand.
//! - **auto** — ripgrep first; if it returns < k hits, augment with
//!   semantic. Default.
//!
//! ## Why brute-force cosine
//!
//! lamu-rs is small (~10K lines, ~150 files at typical chunk size).
//! 150 * 1536-dim cosine = 230K floats per query = sub-millisecond on
//! any modern CPU. HNSW / sqlite-vss / DuckDB-vss are overkill until
//! the index hits 10K+ rows.

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use crate::vector_index::{BruteForceCosine, VectorIndex};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS chunks (
    path TEXT NOT NULL,
    chunk_idx INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    mtime INTEGER NOT NULL,
    PRIMARY KEY (path, chunk_idx)
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
";

/// Embedding model + dimension. Hardcoded to OpenAI's
/// text-embedding-3-small (1536 dims, $0.02/M tokens). If we ever
/// support multiple models, every embedding row has the same shape so
/// they're never mixed within one DB.
pub(crate) const EMBED_MODEL: &str = "text-embedding-3-small";

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

fn index_db_path() -> Result<PathBuf> {
    let dir = dirs::data_local_dir()
        .ok_or_else(|| anyhow!("no data_local_dir"))?
        .join("lamu");
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join("embeddings.db"))
}

fn open_index_db(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.execute_batch(SCHEMA)?;
    Ok(conn)
}

static INDEX_DB: OnceLock<Arc<Mutex<Connection>>> = OnceLock::new();

fn index_db() -> Result<Arc<Mutex<Connection>>> {
    if let Some(d) = INDEX_DB.get() {
        return Ok(d.clone());
    }
    let path = index_db_path()?;
    let conn = open_index_db(&path)?;
    let arc = Arc::new(Mutex::new(conn));
    let _ = INDEX_DB.set(arc.clone());
    Ok(arc)
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

/// Resolve the OpenAI API key. If unset, semantic mode is unavailable.
pub(crate) fn openai_key() -> Option<String> {
    std::env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty())
}

/// POST a single string to OpenAI's `/embeddings` endpoint. Returns
/// the 1536-dim vector for text-embedding-3-small.
pub(crate) async fn embed_one(text: &str, key: &str) -> Result<Vec<f32>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    let body = json!({
        "model": EMBED_MODEL,
        "input": text,
    });
    let resp = client
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(key)
        .json(&body)
        .send()
        .await?;
    let v: Value = resp.json().await?;
    let arr = v["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| anyhow!("embeddings response missing data[0].embedding"))?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        out.push(x.as_f64().unwrap_or(0.0) as f32);
    }
    Ok(out)
}

/// Batch-embed many strings. OpenAI accepts an `input` array up to
/// 2048 items per call. We cap at 96 for safety + smaller payloads.
async fn embed_batch(texts: &[String], key: &str) -> Result<Vec<Vec<f32>>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let mut all = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(96) {
        let body = json!({
            "model": EMBED_MODEL,
            "input": chunk,
        });
        let resp = client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(key)
            .json(&body)
            .send()
            .await?;
        let v: Value = resp.json().await?;
        let arr = v["data"]
            .as_array()
            .ok_or_else(|| anyhow!("embeddings response missing data"))?;
        for entry in arr {
            let emb = entry["embedding"]
                .as_array()
                .ok_or_else(|| anyhow!("missing entry.embedding"))?;
            let mut out = Vec::with_capacity(emb.len());
            for x in emb {
                out.push(x.as_f64().unwrap_or(0.0) as f32);
            }
            all.push(out);
        }
    }
    Ok(all)
}

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
pub async fn index_repo(repo: &Path, force: bool) -> Result<usize> {
    let key = openai_key().ok_or_else(|| {
        anyhow!("OPENAI_API_KEY unset — semantic indexing requires it. Use mode='ripgrep' for grep-only search.")
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
            // Drop existing chunks for this path so re-index is clean.
            let _ = conn.execute("DELETE FROM chunks WHERE path = ?", params![path]);
            for (idx, chunk) in chunk_text(&body, CHUNK_BYTES).into_iter().enumerate() {
                to_embed.push((path.clone(), idx, chunk, mtime));
            }
        }
    }

    if to_embed.is_empty() {
        return Ok(0);
    }

    // Batch embed.
    let texts: Vec<String> = to_embed.iter().map(|(_, _, c, _)| c.clone()).collect();
    let embeddings = embed_batch(&texts, &key).await?;
    if embeddings.len() != to_embed.len() {
        return Err(anyhow!(
            "embed count mismatch: requested {}, got {}",
            to_embed.len(),
            embeddings.len()
        ));
    }

    // Bulk insert under one transaction.
    let mut conn = arc.lock();
    let tx = conn.transaction()?;
    for ((path, idx, content, mtime), emb) in to_embed.iter().zip(embeddings.iter()) {
        tx.execute(
            "INSERT OR REPLACE INTO chunks (path, chunk_idx, content, embedding, mtime) VALUES (?, ?, ?, ?, ?)",
            params![path, *idx as i64, content, vec_to_blob(emb), mtime],
        )?;
    }
    tx.commit()?;

    Ok(to_embed.len())
}

pub async fn semantic_search(query: &str, k: usize) -> Result<Vec<SearchHit>> {
    let key = openai_key().ok_or_else(|| {
        anyhow!("OPENAI_API_KEY unset — semantic search requires it.")
    })?;
    let qvec = embed_one(query, &key).await?;
    let arc = index_db()?;
    let conn = arc.lock();
    let mut stmt =
        conn.prepare("SELECT path, chunk_idx, content, embedding FROM chunks")?;
    let rows = stmt.query_map([], |r| {
        let path: String = r.get(0)?;
        let content: String = r.get(2)?; // chunk_idx (col 1) not needed in SearchHit
        let emb_blob: Vec<u8> = r.get(3)?;
        Ok((path, content, emb_blob))
    })?;
    // SEAM: swap BruteForceCosine for an ANN/quantized index when the
    // corpus outgrows brute-force (see crate::vector_index for the why).
    let mut index: BruteForceCosine<(String, String)> = BruteForceCosine::new();
    for row in rows {
        let (path, content, emb_blob) = row?;
        index.add(blob_to_vec(&emb_blob), (path, content));
    }
    Ok(index
        .search(&qvec, k)
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
        .collect())
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
            if hits.len() < k && openai_key().is_some() {
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
