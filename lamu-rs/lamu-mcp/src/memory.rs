//! Conversation memory — append-only per-`conversation_id` log
//! backed by SQLite at `~/.local/share/lamu/conversations.db`.
//!
//! ## Why SQLite
//!
//! Plain-file JSONL would have done it, but the user picked sqlite so
//! we get filter-by-role / time-range queries cheap, and the file-size
//! cap can be enforced via `LIMIT N` on read instead of streaming the
//! whole log. One DB file holds every conversation; tables are
//! `conversations(id, created_at)` + `turns(conversation_id, idx,
//! role, content, ts, metadata)`.
//!
//! ## API
//!
//! - `Memory::open(path)` — explicit path; used by tests with
//!   tempfiles.
//! - `memory::shared()` — process-wide instance pointing at the
//!   production data dir; used by handlers.
//!
//! Per-call work is a single INSERT or a single SELECT. We hold the
//! parking_lot::Mutex briefly; never across .await.

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    conversation_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts INTEGER NOT NULL,
    metadata TEXT,
    PRIMARY KEY (conversation_id, idx),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(conversation_id, ts);
";

/// `conversation_id` allowlist. Matches `validate_session_id` in
/// `lamu_core::sandbox::journal` — the same threat (caller-controlled
/// string used to key persistent state) deserves the same defense.
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
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) {
        return Err(anyhow!(
            "conversation_id contains forbidden character — allowed: [A-Za-z0-9_-.]: {id}"
        ));
    }
    Ok(())
}

fn db_path() -> Result<PathBuf> {
    let dir = dirs::data_local_dir()
        .ok_or_else(|| anyhow!("no data_local_dir on this platform"))?
        .join("lamu");
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir.join("conversations.db"))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// One turn from a recall.
#[derive(Debug, Clone)]
pub struct Turn {
    pub idx: i64,
    pub role: String,
    pub content: String,
    pub ts: i64,
    pub metadata: Option<String>,
}

/// Conversation-memory handle. Wraps one SQLite connection; methods
/// take `&self` so the handle can be shared via Arc.
pub struct Memory {
    conn: Arc<Mutex<Connection>>,
}

impl Memory {
    /// Open (or create) the conversation database at `path`. Schema
    /// is applied idempotently.
    ///
    /// Pragmas applied at open:
    /// - `journal_mode=WAL` — concurrent readers + one writer. Lamu's
    ///   MCP server and a separate `lamu` CLI invocation can both read
    ///   the conversations log without serializing on a global lock.
    ///   ~10× write throughput vs the default rollback-journal mode.
    /// - `synchronous=NORMAL` — fdatasync at commit instead of fsync
    ///   at every write. Worst-case crash loss = the most recent
    ///   uncommitted turn. Acceptable for conversation logs (not for
    ///   financial ledgers). ~2× faster writes.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Append many turns under one transaction. Cheaper than calling
    /// `append_turn` N times when ingesting historical / migrating —
    /// each call there is its own implicit tx + per-row sync.
    pub fn append_turns(
        &self,
        conversation_id: &str,
        turns: &[(&str, &str, Option<&str>)],
    ) -> Result<()> {
        validate_conversation_id(conversation_id)?;
        if turns.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        let ts = now_secs() as i64;
        tx.execute(
            "INSERT OR IGNORE INTO conversations (id, created_at) VALUES (?, ?)",
            params![conversation_id, ts],
        )?;
        let mut next_idx: i64 = tx.query_row(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM turns WHERE conversation_id = ?",
            params![conversation_id],
            |r| r.get(0),
        )?;
        for (role, content, metadata) in turns {
            tx.execute(
                "INSERT INTO turns (conversation_id, idx, role, content, ts, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                params![conversation_id, next_idx, role, content, ts, metadata],
            )?;
            next_idx += 1;
        }
        tx.commit()?;
        Ok(())
    }

    /// Append one turn. Creates the conversation row on first use.
    pub fn append_turn(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
        metadata: Option<&str>,
    ) -> Result<()> {
        validate_conversation_id(conversation_id)?;
        let conn = self.conn.lock();
        let ts = now_secs() as i64;
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, created_at) VALUES (?, ?)",
            params![conversation_id, ts],
        )?;
        let next_idx: i64 = conn.query_row(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM turns WHERE conversation_id = ?",
            params![conversation_id],
            |r| r.get(0),
        )?;
        conn.execute(
            "INSERT INTO turns (conversation_id, idx, role, content, ts, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            params![conversation_id, next_idx, role, content, ts, metadata],
        )?;
        Ok(())
    }

    /// Read up to `limit` most-recent turns for a conversation,
    /// oldest-first. Pass `limit = 0` for no cap.
    pub fn recall(&self, conversation_id: &str, limit: usize) -> Result<Vec<Turn>> {
        validate_conversation_id(conversation_id)?;
        let conn = self.conn.lock();
        let sql = if limit == 0 {
            "SELECT idx, role, content, ts, metadata FROM turns WHERE conversation_id = ? ORDER BY idx ASC".to_string()
        } else {
            format!(
                "SELECT idx, role, content, ts, metadata FROM (\
                    SELECT idx, role, content, ts, metadata \
                    FROM turns WHERE conversation_id = ? \
                    ORDER BY idx DESC LIMIT {}\
                ) ORDER BY idx ASC",
                limit
            )
        };
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![conversation_id], |r| {
            Ok(Turn {
                idx: r.get(0)?,
                role: r.get(1)?,
                content: r.get(2)?,
                ts: r.get(3)?,
                metadata: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }
}

/// Process-wide instance pointing at the production data dir.
/// Lazy-initialized on first use; subsequent calls reuse the same
/// SQLite connection.
pub fn shared() -> Result<&'static Memory> {
    static M: OnceLock<Memory> = OnceLock::new();
    if let Some(m) = M.get() {
        return Ok(m);
    }
    let path = db_path()?;
    let mem = Memory::open(&path)?;
    let _ = M.set(mem);
    M.get().ok_or_else(|| anyhow!("memory init race"))
}

/// V4 Batch 4: rank turns by semantic similarity to a query, return
/// the top-K most relevant + the most-recent few. Falls back to
/// chronological "last K" when OPENAI_API_KEY is unset or embedding
/// fails. Used by cloud_query when conversation_id has many turns —
/// raw chronological recall buries relevant prior turns under
/// recency. Only kicks in when total turns > KEEP_RECENT + KEEP_TOP.
pub async fn recall_ranked(
    mem: &Memory,
    conversation_id: &str,
    query: &str,
    keep_top: usize,
    keep_recent: usize,
) -> Result<Vec<Turn>> {
    let all = mem.recall(conversation_id, 0)?;
    let total = all.len();
    if total <= keep_top + keep_recent {
        return Ok(all);
    }
    let key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            // No embeddings available — fall back to chronological tail.
            return mem.recall(conversation_id, keep_top + keep_recent);
        }
    };

    // Embed the query + each prior turn. Reuse the rag module's
    // helpers via a thin shim: build the same payload it builds.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| anyhow::anyhow!(e))?;

    let mut texts: Vec<String> = Vec::with_capacity(total + 1);
    texts.push(query.to_string());
    for t in &all {
        texts.push(format!("{}: {}", t.role, t.content));
    }

    let body = serde_json::json!({
        "model": "text-embedding-3-small",
        "input": texts,
    });
    let resp = client
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(&key)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!(e))?;
    let v: serde_json::Value = resp.json().await.map_err(|e| anyhow::anyhow!(e))?;
    let arr = v["data"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("embeddings missing data"))?;
    if arr.len() != texts.len() {
        return mem.recall(conversation_id, keep_top + keep_recent);
    }

    let parse_vec = |val: &serde_json::Value| -> Vec<f32> {
        val["embedding"]
            .as_array()
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_default()
    };
    let q_vec = parse_vec(&arr[0]);

    // Score each turn (skip the most-recent KEEP_RECENT — those
    // always make the cut regardless of score).
    let cutoff_recent_start = total.saturating_sub(keep_recent);
    let mut scored: Vec<(f32, usize)> = (0..cutoff_recent_start)
        .map(|i| {
            let t_vec = parse_vec(&arr[i + 1]); // arr[0] is query
            let score = cosine_local(&q_vec, &t_vec);
            (score, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep_idxs: std::collections::BTreeSet<usize> = scored
        .into_iter()
        .take(keep_top)
        .map(|(_, i)| i)
        .collect();
    for i in cutoff_recent_start..total {
        keep_idxs.insert(i);
    }

    let out: Vec<Turn> = keep_idxs.into_iter().map(|i| all[i].clone()).collect();
    Ok(out)
}

fn cosine_local(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Render a recalled transcript as a Markdown string suitable for
/// dropping into the Tactical tier of the context layer.
pub fn render_for_context(turns: &[Turn]) -> String {
    if turns.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(turns.iter().map(|t| t.content.len() + 32).sum());
    out.push_str("Prior conversation turns:\n\n");
    for t in turns {
        out.push_str(&format!("**{}**: {}\n\n", t.role, t.content));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> (tempfile::TempDir, Memory) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conv.db");
        let mem = Memory::open(&path).unwrap();
        (dir, mem)
    }

    #[test]
    fn validate_rejects_path_traversal() {
        assert!(validate_conversation_id("../escape").is_err());
        assert!(validate_conversation_id("a..b").is_err());
        assert!(validate_conversation_id(".hidden").is_err());
        assert!(validate_conversation_id("").is_err());
        assert!(validate_conversation_id("ok-id_42").is_ok());
        assert!(validate_conversation_id("test.session.1").is_ok());
    }

    #[test]
    fn append_then_recall_preserves_order() {
        let (_td, mem) = fresh();
        mem.append_turn("c1", "user", "first", None).unwrap();
        mem.append_turn("c1", "assistant", "second", None).unwrap();
        mem.append_turn("c1", "user", "third", None).unwrap();
        let turns = mem.recall("c1", 0).unwrap();
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].content, "first");
        assert_eq!(turns[1].content, "second");
        assert_eq!(turns[2].content, "third");
        assert_eq!(turns[0].idx, 0);
        assert_eq!(turns[2].idx, 2);
    }

    #[test]
    fn recall_with_limit_returns_tail() {
        let (_td, mem) = fresh();
        for i in 0..5 {
            mem.append_turn("c2", "user", &format!("msg-{i}"), None).unwrap();
        }
        let last2 = mem.recall("c2", 2).unwrap();
        assert_eq!(last2.len(), 2);
        assert_eq!(last2[0].content, "msg-3");
        assert_eq!(last2[1].content, "msg-4");
    }

    #[test]
    fn recall_unknown_conversation_returns_empty() {
        let (_td, mem) = fresh();
        let turns = mem.recall("never-created", 10).unwrap();
        assert!(turns.is_empty());
    }

    #[test]
    fn separate_conversations_are_isolated() {
        let (_td, mem) = fresh();
        mem.append_turn("a", "user", "alpha", None).unwrap();
        mem.append_turn("b", "user", "beta", None).unwrap();
        let a_turns = mem.recall("a", 0).unwrap();
        let b_turns = mem.recall("b", 0).unwrap();
        assert_eq!(a_turns.len(), 1);
        assert_eq!(b_turns.len(), 1);
        assert_eq!(a_turns[0].content, "alpha");
        assert_eq!(b_turns[0].content, "beta");
    }

    #[test]
    fn render_for_context_empty_returns_empty_string() {
        assert!(render_for_context(&[]).is_empty());
    }

    #[test]
    fn render_for_context_includes_role_and_content() {
        let turns = vec![Turn {
            idx: 0,
            role: "user".into(),
            content: "hello".into(),
            ts: 0,
            metadata: None,
        }];
        let s = render_for_context(&turns);
        assert!(s.contains("user"));
        assert!(s.contains("hello"));
    }

    #[test]
    fn metadata_round_trips() {
        let (_td, mem) = fresh();
        mem.append_turn("m1", "assistant", "reply", Some("model=v4-pro"))
            .unwrap();
        let turns = mem.recall("m1", 0).unwrap();
        assert_eq!(turns[0].metadata.as_deref(), Some("model=v4-pro"));
    }

    #[test]
    fn append_turns_batch_orders_correctly() {
        let (_td, mem) = fresh();
        mem.append_turns(
            "batch-1",
            &[
                ("user", "u1", None),
                ("assistant", "a1", Some("m=flash")),
                ("user", "u2", None),
            ],
        )
        .unwrap();
        let turns = mem.recall("batch-1", 0).unwrap();
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].content, "u1");
        assert_eq!(turns[1].content, "a1");
        assert_eq!(turns[1].metadata.as_deref(), Some("m=flash"));
        assert_eq!(turns[2].content, "u2");
        // Indices are contiguous, starting at 0.
        assert_eq!((turns[0].idx, turns[1].idx, turns[2].idx), (0, 1, 2));
    }

    #[test]
    fn wal_mode_is_active_after_open() {
        let (_td, mem) = fresh();
        let conn = mem.conn.lock();
        let mode: String = conn
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
    }
}
