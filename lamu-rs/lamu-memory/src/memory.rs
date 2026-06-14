//! Conversation memory — append-only per-`conversation_id` log
//! backed by the `conversations` / `turns` tables of the unified
//! `lamu.db` (ADR 0028; previously its own `conversations.db`).
//!
//! ## Why SQLite
//!
//! Plain-file JSONL would have done it, but the user picked sqlite so
//! we get filter-by-role / time-range queries cheap, and the file-size
//! cap can be enforced via `LIMIT N` on read instead of streaming the
//! whole log. One DB file holds every conversation; tables are
//! `conversations(id, owner, created_at)` + `turns(conversation_id,
//! idx, role, content, ts, metadata)`. Schema is owned by
//! `crate::migrate` (ADR 0028).
//!
//! ## API
//!
//! - `Memory::open(path)` — explicit path; used by tests with
//!   tempfiles. Runs the full migration set on that path, so a temp
//!   `lamu.db` gets the complete unified schema.
//! - `memory::shared()` — process-wide instance over the shared
//!   `crate::store` connection; used by handlers.
//!
//! Per-call work is a single INSERT or a single SELECT. We hold the
//! parking_lot::Mutex briefly; never across .await.

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::{Arc, OnceLock};

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
    /// Open (or create) a lamu database at `path` and wrap a
    /// conversation-memory handle around it. The full migration set is
    /// applied idempotently (`crate::store::open_at`), so a tempfile
    /// here carries the complete unified schema — WAL +
    /// `synchronous=NORMAL` pragmas included (see `store::open_at` for
    /// the rationale).
    pub fn open(path: &Path) -> Result<Self> {
        let conn = crate::store::open_at(path)?;
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

    /// Pull every turn whose `ts >= cutoff_unix_secs`, grouped under
    /// its conversation id. Returns `(conversation_id, Turn)` pairs
    /// in the natural sort order — same conversation's turns appear
    /// contiguous and ordered by idx, conversations themselves sort
    /// lexicographically by id.
    ///
    /// Used by `lamu-train --from-conversations` (read directly via
    /// rusqlite to avoid linking lamu-mcp into lamu-train) and by
    /// the future `train_from_conversations` MCP tool dry-run.
    /// O(N) in returned rows; cap on the caller's side if needed.
    pub fn recall_since(&self, cutoff_unix_secs: i64) -> Result<Vec<(String, Turn)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT conversation_id, idx, role, content, ts, metadata \
             FROM turns WHERE ts >= ? ORDER BY conversation_id, idx ASC",
        )?;
        let rows = stmt.query_map(params![cutoff_unix_secs], |r| {
            Ok((
                r.get::<_, String>(0)?,
                Turn {
                    idx: r.get(1)?,
                    role: r.get(2)?,
                    content: r.get(3)?,
                    ts: r.get(4)?,
                    metadata: r.get(5)?,
                },
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
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
        Ok(apply_supersedes(out))
    }
}

/// ADR 0021 C5: hide turns superseded by a compaction summary. A summary turn
/// carries metadata `{"kind":"compaction_summary","supersedes":[lo,hi]}`; every
/// turn whose `idx` falls in any such range is dropped. The summary itself is
/// appended AFTER its range so it keeps a higher idx and survives — and is
/// dropped only if a LATER summary supersedes it, which is exactly right for
/// re-compaction. No markers present → returns the input unchanged (the
/// overwhelming-majority path: zero allocation beyond the scan, zero behavior
/// change for non-compacted conversations).
fn apply_supersedes(turns: Vec<Turn>) -> Vec<Turn> {
    let mut ranges: Vec<(i64, i64)> = Vec::new();
    for t in &turns {
        if let Some(md) = &t.metadata {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(md) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("compaction_summary") {
                    if let (Some(lo), Some(hi)) = (
                        v.get("supersedes").and_then(|s| s.get(0)).and_then(|x| x.as_i64()),
                        v.get("supersedes").and_then(|s| s.get(1)).and_then(|x| x.as_i64()),
                    ) {
                        ranges.push((lo, hi));
                    }
                }
            }
        }
    }
    if ranges.is_empty() {
        return turns;
    }
    turns
        .into_iter()
        .filter(|t| !ranges.iter().any(|(lo, hi)| t.idx >= *lo && t.idx <= *hi))
        .collect()
}

/// Process-wide instance over the shared unified store
/// (`crate::store`, ADRs 0028/0041). Lazy-initialized on first use.
/// Each method call acquires its own pool connection (not stored here);
/// this static only ensures the pool is initialized once.
pub fn shared() -> Result<&'static Memory> {
    static M: OnceLock<Memory> = OnceLock::new();
    if let Some(m) = M.get() {
        return Ok(m);
    }
    // Trigger pool init (runs migration + legacy import once). The Memory
    // struct holds an Arc<Mutex<Connection>> for its own private methods;
    // we use the deprecated shim here to satisfy the struct field type
    // during the pool migration transition.
    #[allow(deprecated)]
    let conn = crate::store::shared_handle()?;
    let _ = M.set(Memory { conn });
    M.get().ok_or_else(|| anyhow!("memory init race"))
}

/// V4 Batch 4: rank turns by semantic similarity to a query, return
/// the top-K most relevant + the most-recent few. Falls back to
/// chronological "last K" when no embedder resolves (ADR 0030 chain:
/// local registry model → OpenAI fallback) or embedding fails. Used by
/// cloud_query when conversation_id has many turns — raw chronological
/// recall buries relevant prior turns under recency. Only kicks in
/// when total turns > KEEP_RECENT + KEEP_TOP. Turn embeddings are
/// computed fresh per call (never persisted), so no model-tag
/// filtering applies here.
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
    let Some(embedder) = crate::embedder::resolve() else {
        // No embedder available — fall back to chronological tail.
        // Log once so degraded (non-semantic) recall is visible.
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "conversation recall: no embedder resolved — chronological tail, not semantic ranking"
            )
        });
        return mem.recall(conversation_id, keep_top + keep_recent);
    };

    // Embed the query + each prior turn via the chain.
    let mut texts: Vec<String> = Vec::with_capacity(total + 1);
    texts.push(query.to_string());
    for t in &all {
        texts.push(format!("{}: {}", t.role, t.content));
    }
    let embs = match embedder.embed(&texts).await {
        Ok(e) if e.len() == texts.len() => e,
        Ok(e) => {
            tracing::warn!(
                "conversation recall: embed count mismatch ({} != {}) — chronological tail",
                e.len(),
                texts.len()
            );
            return mem.recall(conversation_id, keep_top + keep_recent);
        }
        Err(e) => {
            tracing::warn!("conversation recall: embed failed ({e}) — chronological tail");
            return mem.recall(conversation_id, keep_top + keep_recent);
        }
    };
    let q_vec = &embs[0];

    // Score each turn (skip the most-recent KEEP_RECENT — those
    // always make the cut regardless of score).
    let cutoff_recent_start = total.saturating_sub(keep_recent);
    let mut scored: Vec<(f32, usize)> = (0..cutoff_recent_start)
        .map(|i| {
            let score = cosine_local(q_vec, &embs[i + 1]); // embs[0] is query
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

/// Render a recalled transcript as a raw Markdown body — one
/// `**role**: content` block per turn — suitable for dropping into the
/// Tactical tier of the context layer AFTER fencing.
///
/// This is the pure half of what lamu-mcp's `render_for_context` does:
/// prior turns are attacker-influenceable (a poisoned earlier message
/// could carry an injection), so each FRONTEND must fence the returned
/// body as untrusted data before a downstream prompt sees it (ADR 0011).
/// The fencing is a frontend wire concern and stays out of this crate
/// (ADR 0029) — lamu-mcp's `memory::render_for_context` wraps this body
/// with `untrusted::wrap_untrusted("prior conversation turns", …)`.
pub fn render_turns(turns: &[Turn]) -> String {
    if turns.is_empty() {
        return String::new();
    }
    let mut body = String::with_capacity(turns.iter().map(|t| t.content.len() + 32).sum());
    for t in turns {
        body.push_str(&format!("**{}**: {}\n\n", t.role, t.content));
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(idx: i64, meta: Option<&str>) -> Turn {
        Turn { idx, role: "user".into(), content: format!("t{idx}"), ts: 0, metadata: meta.map(String::from) }
    }

    #[test]
    fn apply_supersedes_noop_without_markers() {
        let t = vec![turn(0, None), turn(1, None), turn(2, None)];
        assert_eq!(apply_supersedes(t).len(), 3);
    }

    #[test]
    fn apply_supersedes_hides_range_keeps_summary() {
        // summary at idx 5 supersedes [1,3]; originals 1,2,3 hidden, 0/4/5 kept.
        let md = r#"{"kind":"compaction_summary","supersedes":[1,3]}"#;
        let t = vec![turn(0, None), turn(1, None), turn(2, None), turn(3, None), turn(4, None), turn(5, Some(md))];
        let idxs: Vec<i64> = apply_supersedes(t).into_iter().map(|x| x.idx).collect();
        assert_eq!(idxs, vec![0, 4, 5]);
    }

    #[test]
    fn apply_supersedes_recompaction_drops_old_summary() {
        // S1@5 supersedes [1,3]; S2@7 supersedes [1,5] → S1 (idx5) now hidden.
        let s1 = r#"{"kind":"compaction_summary","supersedes":[1,3]}"#;
        let s2 = r#"{"kind":"compaction_summary","supersedes":[1,5]}"#;
        let t = vec![turn(0, None), turn(4, None), turn(5, Some(s1)), turn(6, None), turn(7, Some(s2))];
        let idxs: Vec<i64> = apply_supersedes(t).into_iter().map(|x| x.idx).collect();
        assert_eq!(idxs, vec![0, 6, 7]);
    }

    #[test]
    fn recall_applies_supersedes_end_to_end() {
        let (_d, mem) = fresh();
        for i in 0..5 {
            mem.append_turn("c", "user", &format!("turn{i}"), None).unwrap();
        }
        // Supersede the first 3 (idx 0..=2) with a summary turn.
        let md = r#"{"kind":"compaction_summary","supersedes":[0,2]}"#;
        mem.append_turn("c", "system", "[summary]", Some(md)).unwrap();
        let got = mem.recall("c", 0).unwrap();
        // 5 originals + 1 summary = 6 rows on disk; recall hides idx 0,1,2 → 3.
        let idxs: Vec<i64> = got.iter().map(|t| t.idx).collect();
        assert_eq!(idxs, vec![3, 4, 5]);
    }

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
    fn recall_since_filters_by_ts() {
        // Two conversations, three turns each at ts = 100, 200, 300.
        // recall_since(200) should keep the last 2 of each = 4 rows
        // total.
        use rusqlite::params;
        let td = tempfile::tempdir().unwrap();
        let mem = Memory::open(&td.path().join("test.db")).unwrap();
        for conv in ["alpha", "bravo"] {
            mem.append_turn(conv, "user", "ignored", None).unwrap();
        }
        // Backdate the rows: append created turns at "now"; rewrite ts
        // directly via a raw query so the test controls the times.
        let conn = mem.conn.lock();
        conn.execute("DELETE FROM turns", []).unwrap();
        for (conv, ts) in [
            ("alpha", 100i64),
            ("alpha", 200),
            ("alpha", 300),
            ("bravo", 100),
            ("bravo", 200),
            ("bravo", 300),
        ] {
            conn.execute(
                "INSERT INTO turns (conversation_id, idx, role, content, ts) \
                 VALUES (?, ?, ?, ?, ?)",
                params![conv, ts, "user", format!("at-{ts}"), ts],
            )
            .unwrap();
        }
        drop(conn);

        let rows = mem.recall_since(200).unwrap();
        assert_eq!(rows.len(), 4, "expect 2 turns per conv after cutoff");
        // Conv-grouped, idx-ordered.
        assert_eq!(rows[0].0, "alpha");
        assert_eq!(rows[3].0, "bravo");
        assert_eq!(rows[0].1.ts, 200);
        assert_eq!(rows[1].1.ts, 300);
    }

    #[test]
    fn recall_since_empty_when_cutoff_in_future() {
        let td = tempfile::tempdir().unwrap();
        let mem = Memory::open(&td.path().join("test.db")).unwrap();
        mem.append_turn("conv1", "user", "hi", None).unwrap();
        let rows = mem.recall_since(i64::MAX).unwrap();
        assert!(rows.is_empty());
    }

    // The fenced (wrap_untrusted) rendering is tested in lamu-mcp's
    // `memory::render_for_context` shim — the wrap is a frontend concern
    // (ADR 0029); here we only test the pure body.
    #[test]
    fn render_turns_empty_returns_empty_string() {
        assert!(render_turns(&[]).is_empty());
    }

    #[test]
    fn render_turns_includes_role_and_content() {
        let turns = vec![Turn {
            idx: 0,
            role: "user".into(),
            content: "hello".into(),
            ts: 0,
            metadata: None,
        }];
        let s = render_turns(&turns);
        assert!(s.contains("**user**: hello"));
        // The raw body is UNfenced — fencing happens in the frontend.
        assert!(!s.contains("LAMU_UNTRUSTED"));
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
