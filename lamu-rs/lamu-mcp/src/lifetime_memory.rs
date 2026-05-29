//! Lifetime cross-session memory.
//!
//! Where `memory.rs` keys turns by `conversation_id` (strictly
//! per-conversation), this module is a GLOBAL fact store that spans
//! every conversation. Facts are extracted from conversations (via
//! MiMo) or added explicitly, embedded with OpenAI's
//! `text-embedding-3-small`, and recalled by cross-session semantic
//! search over the existing `crate::vector_index::BruteForceCosine`
//! seam — this is the seam's first cross-session consumer.
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
use crate::vector_index::{BruteForceCosine, VectorIndex};

const SCHEMA: &str = "
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

/// System prompt for fact extraction. Instructs MiMo to keep only
/// durable, user-specific facts worth remembering across sessions.
const EXTRACTION_PROMPT: &str = "\
You extract DURABLE, user-specific facts worth remembering across future \
sessions from the conversation transcript below. Keep only stable, \
re-usable facts: the user's identity, preferences, project facts, \
tooling/environment, and decisions they have made. Drop ephemeral \
chit-chat, one-off questions, transient state, and anything that will \
not matter next session.\n\
\n\
Output ONE fact per line, with no preamble, no numbering, no bullets, and \
no commentary. Each line must be a self-contained statement that reads \
correctly with no surrounding context. If nothing in the transcript is \
worth keeping, output exactly NONE and nothing else.";

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
    let blob = embedding.map(vec_to_blob);
    conn.execute(
        "INSERT INTO memories (text, embedding, kind, source, ts) VALUES (?, ?, ?, ?, ?)",
        params![text, blob, kind, source, ts],
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
pub(crate) fn rank_memories(
    query_emb: &[f32],
    rows: Vec<MemoryRow>,
    k: usize,
) -> Vec<MemoryHit> {
    // SEAM: same brute-force cosine index rag.rs uses for repo search.
    let mut index: BruteForceCosine<MemoryPayload> = BruteForceCosine::with_capacity(rows.len());
    for (id, text, kind, source, ts, emb) in rows {
        index.add(
            emb,
            MemoryPayload {
                id,
                text,
                kind,
                source,
                ts,
            },
        );
    }
    index
        .search(query_emb, k)
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
pub async fn recall_memory(query: &str, k: usize) -> Result<Vec<MemoryHit>> {
    let arc = memory_db()?;

    match openai_key() {
        Some(key) => {
            let qvec = embed_one(query, &key).await?;
            // Collect rows into a Vec, then release the lock BEFORE
            // ranking — never hold the mutex across the cosine pass, so
            // concurrent remember/recall don't serialize behind it.
            // `embedding IS NOT NULL` skips embedding-less rows in SQL
            // (the ranked path can't use them anyway).
            let rows: Vec<MemoryRow> = {
                let conn = arc.lock();
                let mut stmt = conn.prepare(
                    "SELECT id, text, kind, source, ts, embedding FROM memories \
                     WHERE embedding IS NOT NULL",
                )?;
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
            let conn = arc.lock();
            let mut stmt = conn.prepare(
                "SELECT id, text, kind, source, ts FROM memories ORDER BY ts DESC, id DESC LIMIT ?",
            )?;
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

// ── Consolidation (fact extraction) ────────────────────────────────

/// Parse MiMo's extraction output into individual facts. One fact per
/// non-empty line, leading bullets/numbering stripped. A line that is
/// exactly `NONE` (case-insensitive) is dropped; if the whole output is
/// NONE / empty, the result is an empty vec. Pure + unit-testable.
pub(crate) fn parse_extracted_facts(raw: &str) -> Vec<String> {
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

/// Same caller-supplied-id allowlist `write_file` / `memory.rs` enforce
/// — fail fast with a clear message rather than relying on the inner
/// `recall` validation, so the error surfaces at the tool boundary.
fn validate_conversation_id(id: &str) -> Result<()> {
    if id.is_empty() {
        return Err(anyhow!("conversation_id is empty"));
    }
    if id.starts_with('.') {
        return Err(anyhow!("conversation_id cannot start with '.': {id}"));
    }
    if id.contains("..") {
        return Err(anyhow!("conversation_id contains '..': {id}"));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(anyhow!(
            "conversation_id contains forbidden character — allowed: [A-Za-z0-9_-.]: {id}"
        ));
    }
    Ok(())
}

/// Extract durable facts from a conversation via MiMo and store each as
/// a fact memory keyed by the conversation id. Returns the count stored.
///
/// `recall(id, 0)` uses `limit = 0` to mean "no cap" (full transcript).
/// NOTE: the transcript is sent to MiMo with the spec-mandated
/// `max_tokens: 1024` for the *output*; an extremely long input
/// conversation could exceed the model's context window. Input
/// truncation is a deliberate follow-up (see commit message).
pub async fn consolidate(conversation_id: &str) -> Result<usize> {
    validate_conversation_id(conversation_id)?;
    let mem = crate::memory::shared()?;
    let turns = mem.recall(conversation_id, 0)?;
    if turns.is_empty() {
        return Ok(0);
    }
    // Compact "role: content" transcript — same shape memory.rs embeds.
    let transcript = turns
        .iter()
        .map(|t| format!("{}: {}", t.role, t.content))
        .collect::<Vec<_>>()
        .join("\n");

    let args = serde_json::json!({
        "model": "mimo-v2.5",
        "system": EXTRACTION_PROMPT,
        "prompt": transcript,
        "max_tokens": 1024,
        "temperature": 0.2,
        "include_reasoning": false,
    });
    let resp = crate::cloud::handle_cloud_query(args).await;
    if resp.starts_with("error:") {
        return Err(anyhow!("fact extraction failed: {resp}"));
    }

    let facts = parse_extracted_facts(&resp);
    let mut stored = 0usize;
    for fact in facts {
        remember(&fact, "fact", conversation_id).await?;
        stored += 1;
    }
    Ok(stored)
}

// ── MCP tool handlers ──────────────────────────────────────────────

/// `remember` tool handler. Local store, embedding optional.
/// Args: `text` (required), `kind` (default "fact"), `source` (default "manual").
pub(crate) async fn handle_remember(args: serde_json::Value) -> String {
    let text = args["text"].as_str().unwrap_or("").trim();
    if text.is_empty() {
        return "error: text is required".to_string();
    }
    let kind = match args["kind"].as_str() {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => "fact",
    };
    let source = match args["source"].as_str() {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => "manual",
    };
    match remember(text, kind, source).await {
        Ok(id) => format!("remembered memory #{id} (kind={kind}, source={source})"),
        Err(e) => format!("error: {e}"),
    }
}

/// `recall_memory` tool handler. Local read; degrades to recency
/// without a key. Args: `query` (required), `k` (default 8).
pub(crate) async fn handle_recall_memory(args: serde_json::Value) -> String {
    let query = args["query"].as_str().unwrap_or("").trim();
    if query.is_empty() {
        return "error: query is required".to_string();
    }
    // Cap k so a pathological request can't ask for an unbounded result
    // set (the underlying scan is already bounded; this bounds output).
    let k = (args["k"].as_u64().unwrap_or(8) as usize).min(100);
    match recall_memory(query, k).await {
        Ok(hits) if hits.is_empty() => "(no memories matched)".to_string(),
        Ok(hits) => {
            let mut out = String::new();
            for h in hits {
                let score = match h.score {
                    Some(s) => format!("{s:.3}"),
                    None => "recency".to_string(),
                };
                let source = h.source.as_deref().unwrap_or("?");
                out.push_str(&format!(
                    "#{} [{}] (source={}, score={}) {}\n",
                    h.id, h.kind, source, score, h.text
                ));
            }
            out
        }
        Err(e) => format!("error: {e}"),
    }
}

/// `consolidate_memory` tool handler. Cloud (requires MiMo extraction)
/// — gated under local-only by the dispatcher. Args: `conversation_id`
/// (required).
pub(crate) async fn handle_consolidate_memory(args: serde_json::Value) -> String {
    let conversation_id = args["conversation_id"].as_str().unwrap_or("").trim();
    if conversation_id.is_empty() {
        return "error: conversation_id is required".to_string();
    }
    match consolidate(conversation_id).await {
        Ok(n) => format!("stored {n} memories from {conversation_id}"),
        Err(e) => format!("error: {e}"),
    }
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
    fn validate_conversation_id_allowlist() {
        assert!(validate_conversation_id("sess-1_2.3").is_ok());
        assert!(validate_conversation_id("").is_err());
        assert!(validate_conversation_id(".hidden").is_err());
        assert!(validate_conversation_id("a..b").is_err());
        assert!(validate_conversation_id("a/b").is_err());
        assert!(validate_conversation_id("a b").is_err());
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
}
