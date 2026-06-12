//! Re-embedding maintenance (ADR 0030) — the convergence path after an
//! embedder switch.
//!
//! Per-store identity enforcement means vector recall covers only rows
//! whose `embedding_model` matches the CURRENT embedder; rows embedded
//! under a previous model (or never embedded at all — NULL embedding)
//! fall back to the FTS leg. `lamu memory reembed` walks those stale
//! rows, re-embeds them in batches, and flips the `embedding_stores`
//! row to the new identity once rows actually converged.
//!
//! This module is the LIB-LEVEL core the CLI subcommand drives (and
//! tests call directly): [`plan`] is the dry-run report, [`run`] the
//! batched execution. Both take an explicit connection/embedder so
//! nothing here touches the process singleton or the global chain.
//!
//! OWNER SCOPING (ADR 0032): reembed deliberately takes NO owner and
//! operates across ALL owners' rows. It is an operator action, and the
//! embedding-model identity is store-wide — leaving one tenant's rows
//! on a stale model would silently drop them out of vector recall.

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::sync::Arc;

use crate::embedder::{Embedder, EmbedderId};
use crate::rag::vec_to_blob;

/// Which store(s) to operate on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreSel {
    Memories,
    Chunks,
    All,
}

impl StoreSel {
    /// Parse the CLI's `--store` value. `None`/empty → All.
    pub fn parse(s: Option<&str>) -> Result<Self> {
        match s.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("all") => Ok(StoreSel::All),
            Some("memories") => Ok(StoreSel::Memories),
            Some("chunks") => Ok(StoreSel::Chunks),
            Some(other) => Err(anyhow!(
                "unknown store '{other}' — expected memories | chunks | all"
            )),
        }
    }

    fn stores(self) -> &'static [&'static str] {
        match self {
            StoreSel::Memories => &["memories"],
            StoreSel::Chunks => &["chunks"],
            StoreSel::All => &["memories", "chunks"],
        }
    }
}

/// Dry-run numbers for one store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReembedPlan {
    pub store: &'static str,
    /// Rows whose `embedding_model` differs from the current identity —
    /// including NULL-embedding rows (never embedded) for `memories`.
    pub stale: u64,
    pub total: u64,
}

/// Execution result for one store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReembedReport {
    pub store: &'static str,
    pub reembedded: u64,
}

/// Count stale rows per selected store against the current identity.
///
/// A `memories` row is stale when its embedding is NULL (it was never
/// embedded — e.g. written while no embedder resolved) or its model tag
/// differs from `identity.model`. `chunks.embedding` is NOT NULL by
/// schema, so only the model tag is checked there.
pub fn plan(conn: &Connection, identity: &EmbedderId, sel: StoreSel) -> Result<Vec<ReembedPlan>> {
    let mut out = Vec::new();
    for store in sel.stores() {
        let (total_sql, stale_sql) = match *store {
            "memories" => (
                "SELECT COUNT(*) FROM memories",
                "SELECT COUNT(*) FROM memories \
                 WHERE embedding IS NULL OR embedding_model IS NULL OR embedding_model != ?1",
            ),
            "chunks" => (
                "SELECT COUNT(*) FROM chunks",
                "SELECT COUNT(*) FROM chunks \
                 WHERE embedding_model IS NULL OR embedding_model != ?1",
            ),
            _ => unreachable!(),
        };
        let total: i64 = conn.query_row(total_sql, [], |r| r.get(0))?;
        let stale: i64 = conn.query_row(stale_sql, params![identity.model], |r| r.get(0))?;
        out.push(ReembedPlan {
            store,
            stale: stale as u64,
            total: total as u64,
        });
    }
    Ok(out)
}

/// Re-embed every stale row in the selected store(s) in batches of
/// `batch` (the CLI uses 32), then flip the `embedding_stores` row to
/// the embedder's identity. The lock is NEVER held across an embed
/// await: each iteration selects a batch under the lock, embeds with
/// the lock released, then writes the batch back in one transaction.
///
/// On a mid-run embed failure the rows already converged stay converged
/// (each batch commits independently) — re-running resumes where it
/// stopped, because staleness is recomputed from the rows themselves.
pub async fn run(
    arc: &Arc<Mutex<Connection>>,
    embedder: &dyn Embedder,
    sel: StoreSel,
    batch: usize,
) -> Result<Vec<ReembedReport>> {
    let batch = batch.max(1);
    let identity = embedder.identity();
    let mut out = Vec::new();
    for store in sel.stores() {
        let n = match *store {
            "memories" => run_memories(arc, embedder, &identity, batch).await?,
            "chunks" => run_chunks(arc, embedder, &identity, batch).await?,
            _ => unreachable!(),
        };
        out.push(ReembedReport {
            store,
            reembedded: n,
        });
    }
    Ok(out)
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

async fn run_memories(
    arc: &Arc<Mutex<Connection>>,
    embedder: &dyn Embedder,
    identity: &EmbedderId,
    batch: usize,
) -> Result<u64> {
    let mut reembedded = 0u64;
    let mut dims_seen: Option<usize> = None;
    loop {
        let rows: Vec<(i64, String)> = {
            let conn = arc.lock();
            let mut stmt = conn.prepare(
                "SELECT id, text FROM memories \
                 WHERE embedding IS NULL OR embedding_model IS NULL OR embedding_model != ?1 \
                 ORDER BY id LIMIT ?2",
            )?;
            let mapped = stmt.query_map(params![identity.model, batch as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })?;
            let mut rows = Vec::new();
            for row in mapped {
                rows.push(row?);
            }
            rows
        }; // lock released before the embed await
        if rows.is_empty() {
            break;
        }
        let texts: Vec<String> = rows.iter().map(|(_, t)| t.clone()).collect();
        let embs = embedder.embed(&texts).await?;
        if embs.len() != rows.len() {
            return Err(anyhow!(
                "reembed(memories): embed count mismatch ({} != {})",
                embs.len(),
                rows.len()
            ));
        }
        let mut conn = arc.lock();
        let tx = conn.transaction()?;
        for ((id, _), emb) in rows.iter().zip(embs.iter()) {
            tx.execute(
                "UPDATE memories SET embedding = ?1, embedding_model = ?2 WHERE id = ?3",
                params![vec_to_blob(emb), identity.model, id],
            )?;
        }
        tx.commit()?;
        dims_seen = embs.first().map(|e| e.len()).or(dims_seen);
        reembedded += rows.len() as u64;
    }
    if reembedded > 0 {
        flip_store_row(arc, "memories", &identity.model, dims_seen.unwrap_or(identity.dims))?;
    }
    Ok(reembedded)
}

async fn run_chunks(
    arc: &Arc<Mutex<Connection>>,
    embedder: &dyn Embedder,
    identity: &EmbedderId,
    batch: usize,
) -> Result<u64> {
    let mut reembedded = 0u64;
    let mut dims_seen: Option<usize> = None;
    loop {
        // chunks' PK is (path, chunk_idx) — address rows via the
        // implicit rowid (the table is not WITHOUT ROWID).
        let rows: Vec<(i64, String)> = {
            let conn = arc.lock();
            let mut stmt = conn.prepare(
                "SELECT rowid, content FROM chunks \
                 WHERE embedding_model IS NULL OR embedding_model != ?1 \
                 ORDER BY rowid LIMIT ?2",
            )?;
            let mapped = stmt.query_map(params![identity.model, batch as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })?;
            let mut rows = Vec::new();
            for row in mapped {
                rows.push(row?);
            }
            rows
        };
        if rows.is_empty() {
            break;
        }
        let texts: Vec<String> = rows.iter().map(|(_, t)| t.clone()).collect();
        let embs = embedder.embed(&texts).await?;
        if embs.len() != rows.len() {
            return Err(anyhow!(
                "reembed(chunks): embed count mismatch ({} != {})",
                embs.len(),
                rows.len()
            ));
        }
        let mut conn = arc.lock();
        let tx = conn.transaction()?;
        for ((rowid, _), emb) in rows.iter().zip(embs.iter()) {
            tx.execute(
                "UPDATE chunks SET embedding = ?1, embedding_model = ?2 WHERE rowid = ?3",
                params![vec_to_blob(emb), identity.model, rowid],
            )?;
        }
        tx.commit()?;
        dims_seen = embs.first().map(|e| e.len()).or(dims_seen);
        reembedded += rows.len() as u64;
    }
    if reembedded > 0 {
        flip_store_row(arc, "chunks", &identity.model, dims_seen.unwrap_or(identity.dims))?;
    }
    Ok(reembedded)
}

/// After a successful reembed the store's adopted identity flips to the
/// new model — this is the documented release of the "old model stays
/// pinned until a reembed" rule in `store::record_store_identity`.
fn flip_store_row(arc: &Arc<Mutex<Connection>>, store: &str, model: &str, dims: usize) -> Result<()> {
    let conn = arc.lock();
    conn.execute(
        "INSERT INTO embedding_stores (store, model, dims, updated_at) VALUES (?1, ?2, ?3, ?4) \
         ON CONFLICT(store) DO UPDATE SET model = ?2, dims = ?3, updated_at = ?4",
        params![store, model, dims as i64, now_secs()],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::testutil::FakeEmbedder;
    use crate::lifetime_memory::insert_memory;

    fn open_tmp_db() -> (tempfile::TempDir, Arc<Mutex<Connection>>) {
        let td = tempfile::tempdir().unwrap();
        let conn = crate::store::open_at(&td.path().join("lamu.db")).unwrap();
        (td, Arc::new(Mutex::new(conn)))
    }

    fn seed_memories(conn: &Connection) {
        // One NULL-embedding row, one wrong-model row, one current row.
        insert_memory(conn, "never embedded", None, None, "fact", "manual", 10, "local").unwrap();
        insert_memory(
            conn,
            "old model row",
            Some(&[9.0, 9.0]),
            Some("old-model"),
            "fact",
            "manual",
            20,
            "local",
        )
        .unwrap();
        insert_memory(
            conn,
            "current row",
            Some(&[1.0, 0.0]),
            Some("fake-new"),
            "fact",
            "manual",
            30,
            "local",
        )
        .unwrap();
    }

    fn seed_chunks(conn: &Connection) {
        conn.execute(
            "INSERT INTO chunks (path, chunk_idx, content, embedding, embedding_model, mtime) \
             VALUES ('a.rs', 0, 'fn a() {}', ?1, 'old-model', 1)",
            params![vec_to_blob(&[3.0, 3.0])],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO chunks (path, chunk_idx, content, embedding, embedding_model, mtime) \
             VALUES ('b.rs', 0, 'fn b() {}', ?1, 'fake-new', 1)",
            params![vec_to_blob(&[1.0, 0.0])],
        )
        .unwrap();
    }

    #[test]
    fn store_sel_parse() {
        assert_eq!(StoreSel::parse(None).unwrap(), StoreSel::All);
        assert_eq!(StoreSel::parse(Some("all")).unwrap(), StoreSel::All);
        assert_eq!(StoreSel::parse(Some("Memories")).unwrap(), StoreSel::Memories);
        assert_eq!(StoreSel::parse(Some("chunks")).unwrap(), StoreSel::Chunks);
        assert!(StoreSel::parse(Some("bogus")).is_err());
    }

    #[test]
    fn plan_counts_null_and_wrong_model_rows() {
        let (_td, arc) = open_tmp_db();
        {
            let conn = arc.lock();
            seed_memories(&conn);
            seed_chunks(&conn);
        }
        let identity = EmbedderId {
            model: "fake-new".into(),
            dims: 2,
        };
        let conn = arc.lock();
        let plans = plan(&conn, &identity, StoreSel::All).unwrap();
        assert_eq!(
            plans,
            vec![
                ReembedPlan { store: "memories", stale: 2, total: 3 },
                ReembedPlan { store: "chunks", stale: 1, total: 2 },
            ]
        );
        // Store-scoped plan only reports that store.
        let only_mem = plan(&conn, &identity, StoreSel::Memories).unwrap();
        assert_eq!(only_mem.len(), 1);
        assert_eq!(only_mem[0].store, "memories");
    }

    #[tokio::test]
    async fn run_converges_rows_and_flips_store_row() {
        let (_td, arc) = open_tmp_db();
        {
            let conn = arc.lock();
            seed_memories(&conn);
            seed_chunks(&conn);
            // Pre-existing stores row pinned to the OLD model — run()
            // must flip it.
            conn.execute(
                "INSERT INTO embedding_stores (store, model, dims, updated_at) \
                 VALUES ('memories', 'old-model', 2, 1)",
                [],
            )
            .unwrap();
        }
        let fake = FakeEmbedder::new("fake-new", vec![0.5, 0.5]);
        let reports = run(&arc, &fake, StoreSel::All, 1).await.unwrap(); // batch=1 exercises the loop
        assert_eq!(
            reports,
            vec![
                ReembedReport { store: "memories", reembedded: 2 },
                ReembedReport { store: "chunks", reembedded: 1 },
            ]
        );

        let conn = arc.lock();
        // Every row now carries the new model + a non-NULL embedding.
        let stale: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories \
                 WHERE embedding IS NULL OR embedding_model IS NULL OR embedding_model != 'fake-new'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stale, 0);
        let stale_chunks: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM chunks WHERE embedding_model != 'fake-new'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stale_chunks, 0);
        // The untouched current row kept its ORIGINAL embedding (only
        // stale rows are rewritten).
        let blob: Vec<u8> = conn
            .query_row(
                "SELECT embedding FROM memories WHERE text = 'current row'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(crate::rag::blob_to_vec(&blob), vec![1.0, 0.0]);
        // embedding_stores flipped for both stores.
        let (model, dims): (String, i64) = conn
            .query_row(
                "SELECT model, dims FROM embedding_stores WHERE store = 'memories'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(model, "fake-new");
        assert_eq!(dims, 2);
        let chunk_model: String = conn
            .query_row(
                "SELECT model FROM embedding_stores WHERE store = 'chunks'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(chunk_model, "fake-new");
    }

    #[tokio::test]
    async fn run_with_nothing_stale_is_a_noop() {
        let (_td, arc) = open_tmp_db();
        {
            let conn = arc.lock();
            insert_memory(
                &conn,
                "current row",
                Some(&[1.0, 0.0]),
                Some("fake-new"),
                "fact",
                "manual",
                30,
            "local",
            )
            .unwrap();
        }
        let fake = FakeEmbedder::new("fake-new", vec![0.5, 0.5]);
        let reports = run(&arc, &fake, StoreSel::Memories, 32).await.unwrap();
        assert_eq!(reports[0].reembedded, 0);
        // No store row invented for a no-op.
        let conn = arc.lock();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM embedding_stores", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }
}
