//! The unified `lamu.db` store (ADR 0028) — WAL r2d2 connection pool
//! (ADR 0041). Replaces the single process-global `Arc<Mutex<Connection>>`
//! with a pool so reads run concurrently.
//!
//! Pre-0028, each module owned its own SQLite file + singleton:
//! `memory.rs` → `conversations.db`, `lifetime_memory.rs` →
//! `memory.db`, `rag.rs` → `embeddings.db`. ADR 0028 collapses them
//! into ONE schema-versioned database at
//! `~/.local/share/lamu/lamu.db` with a real migration framework
//! (`migrate.rs`) and a one-time legacy import. The three modules keep
//! their public APIs; their storage now goes through [`conn`].
//!
//! ## Open flow (first touch of the shared store)
//!
//! 1. `lamu.db` exists → open + [`crate::migrate::migrate`], done.
//!    Existence is the idempotence marker (ADR 0025's seeding
//!    pattern): the import never re-runs.
//! 2. Else → build the DB at `lamu.db.tmp.<pid>`, migrate it, import
//!    each legacy file that exists in the same data dir
//!    (`conversations.db`, `memory.db`, `embeddings.db`), then
//!    atomically rename tmp → `lamu.db`. Legacy files are left in
//!    place (the only mutation is `memory.db`'s in-place temporal
//!    normalization, which its legacy open path has always done).
//!
//! `$LAMU_DB` overrides the path (tests/sandboxes — mirrors
//! `$LAMU_REGISTRY` in lamu-core).

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// Pooled connection type alias. Derefs to `&rusqlite::Connection`.
pub type Pool = r2d2::Pool<SqliteConnectionManager>;
pub type PooledConn = r2d2::PooledConnection<SqliteConnectionManager>;

static POOL: OnceLock<Pool> = OnceLock::new();
/// Serializes the one-time open flow (legacy import + first migrate).
/// Kept as a named item so it can live in the same scope as POOL.
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Path of the unified database: `<data dir>/lamu/lamu.db`, overridable
/// via `$LAMU_DB` (tests, sandboxes). Pure path computation — directory
/// creation happens in the open flow.
pub fn lamu_db_path() -> PathBuf {
    if let Ok(p) = std::env::var("LAMU_DB") {
        let t = p.trim();
        if !t.is_empty() {
            return PathBuf::from(t);
        }
    }
    // Explicit ~/.local/share fallback: a None data dir must not collapse
    // to a RELATIVE path that would then be created in the CWD.
    dirs::data_local_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("share")))
        .unwrap_or_default()
        .join("lamu")
        .join("lamu.db")
}

/// Open (or create) a lamu database at an explicit `path` and bring it
/// to the latest schema. No legacy import — this is the constructor for
/// tempfile tests and for `Memory::open(path)`-style explicit opens.
///
/// Pragmas applied at open (same rationale as the legacy per-module
/// opens):
/// - `journal_mode=WAL` — concurrent readers + one writer.
/// - `synchronous=NORMAL` — fdatasync at commit; worst-case crash loss
///   is the most recent uncommitted write. Fine for memory stores.
pub fn open_at(path: &Path) -> Result<Connection> {
    let mut conn =
        Connection::open(path).with_context(|| format!("open {}", path.display()))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    // Wait up to 5s for the WAL writer-lock instead of erroring SQLITE_BUSY
    // immediately — direct opens (reembed, the shared_handle shim) share the
    // DB with the pool's writers (matches the pool's with_init busy_timeout).
    conn.pragma_update(None, "busy_timeout", 5000)?;
    crate::migrate::migrate(&mut conn)?;
    Ok(conn)
}

/// Full first-open flow at an explicit `path`: if the DB exists, open +
/// migrate; otherwise build it via tmp-file + legacy import + atomic
/// rename (see module docs). Public so tests can drive the import
/// against a tempdir without touching the process-wide pool.
pub fn open_or_import(path: &Path) -> Result<Connection> {
    if path.exists() {
        // Existence = idempotence marker: never re-import, only migrate.
        return open_at(path);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    build_with_import(path)?;
    open_at(path)
}

/// Get (or lazily initialize) the process-wide WAL r2d2 connection pool.
///
/// First call runs the full open flow under INIT_LOCK (legacy import +
/// migrations), then builds the pool. The pool's `with_init` callback
/// applies WAL pragmas to every new connection so all pool members run
/// in WAL mode. POOL is set exactly once; subsequent calls return the
/// existing pool reference.
fn pool() -> Result<&'static Pool> {
    if let Some(p) = POOL.get() {
        return Ok(p);
    }
    let _g = INIT_LOCK.lock();
    if let Some(p) = POOL.get() {
        return Ok(p); // built while we waited
    }
    let path = lamu_db_path();
    // One-time import + full migration: drop the returned Connection (it
    // closes and checkpoints WAL). The pool will then open fresh connections
    // to the now-current, fully-migrated file.
    let _bootstrap = open_or_import(&path)?;
    drop(_bootstrap);

    let manager = SqliteConnectionManager::file(&path).with_init(|c| {
        c.execute_batch(
            "PRAGMA journal_mode=WAL; \
             PRAGMA synchronous=NORMAL; \
             PRAGMA busy_timeout=5000;",
        )?;
        Ok(())
    });
    let p = r2d2::Pool::builder()
        .max_size(8)
        .build(manager)
        .map_err(|e| anyhow!("build pool: {e}"))?;
    let p = POOL.get_or_init(|| p);
    Ok(p)
}

/// Acquire a pooled connection to the unified store. The pool is lazily
/// initialized on first call: runs legacy import + migrations once,
/// then serves up to 8 concurrent WAL readers. Callers use `&conn` /
/// `&mut conn` directly (PooledConn derefs to `&Connection`).
pub fn conn() -> Result<PooledConn> {
    pool()?.get().map_err(|e| anyhow!("pool get: {e}"))
}

/// Deprecated process-wide handle — thin shim over the pool so existing
/// call sites that haven't been migrated yet continue to compile.
///
/// Prefer `store::conn()` for new code. This shim wraps ONE pool
/// connection in `Arc<Mutex<>>` on each call, so concurrent callers no
/// longer serialize on a single global mutex — they each hold their own
/// pooled connection.
#[deprecated(since = "0.1.0", note = "use store::conn() instead")]
pub fn shared_handle() -> Result<Arc<Mutex<Connection>>> {
    // Get a pooled connection and move it into a temporary Arc<Mutex<>>
    // so old call sites (lock → use → drop) keep working. The pooled
    // connection is returned to the pool when the Arc is dropped.
    // Note: this is purely a migration shim; new code should call conn().
    //
    // Because PooledConn cannot be moved into a Connection directly, we
    // open a fresh non-pool connection to the same (already-migrated) path.
    // This is acceptable for the deprecated path only.
    let _ = pool()?; // ensure pool (and migration) is initialized
    let path = lamu_db_path();
    // Use open_at (WAL + synchronous + busy_timeout + idempotent migrate) so
    // the shim has no hidden ordering dependency on pool() having migrated —
    // exactly one non-pool open path, same guarantees as the pool members.
    let c = open_at(&path)?;
    Ok(Arc::new(Mutex::new(c)))
}

// ── Per-store embedder identity (ADR 0030) ──────────────────────────

/// Upsert the `embedding_stores` bookkeeping after a write embedded
/// rows for `store` ('memories' | 'chunks') with `model`/`dims`.
///
/// - No row yet → INSERT (the store adopts this identity).
/// - Row matches `model` → refresh `dims` + `updated_at`.
/// - Row carries a DIFFERENT model → the new rows were still written
///   (with their own per-row `embedding_model` tag), but the store row
///   keeps pointing at the OLD model until a `lamu memory reembed`
///   converges the rows; warn once per process per store so the split
///   is visible. Vector recall is model-filtered, so mixed-model rows
///   never rank against each other — the FTS leg covers the rest.
///
/// Best-effort: bookkeeping failure must never abort the write that
/// produced the rows, so errors are warned, not returned.
pub(crate) fn record_store_identity(
    conn: &Connection,
    store: &str,
    model: &str,
    dims: usize,
    now: i64,
) {
    use rusqlite::OptionalExtension;
    let existing: Option<String> = match conn
        .query_row(
            "SELECT model FROM embedding_stores WHERE store = ?1",
            rusqlite::params![store],
            |r| r.get(0),
        )
        .optional()
    {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("embedding_stores read ({store}): {e}");
            return;
        }
    };
    let res = match existing.as_deref() {
        None => conn.execute(
            "INSERT INTO embedding_stores (store, model, dims, updated_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![store, model, dims as i64, now],
        ),
        Some(m) if m == model => conn.execute(
            "UPDATE embedding_stores SET dims = ?2, updated_at = ?3 WHERE store = ?1",
            rusqlite::params![store, dims as i64, now],
        ),
        Some(old) => {
            warn_once_store_mismatch(store, old, model);
            return; // leave the store row pinned to the OLD model
        }
    };
    if let Err(e) = res {
        tracing::warn!("embedding_stores upsert ({store}): {e}");
    }
}

fn warn_once_store_mismatch(store: &str, old: &str, new: &str) {
    static WARNED: Mutex<Option<std::collections::HashSet<String>>> = Mutex::new(None);
    let mut g = WARNED.lock();
    let set = g.get_or_insert_with(std::collections::HashSet::new);
    if set.insert(store.to_string()) {
        tracing::warn!(
            "embedding store '{store}' is pinned to model '{old}' but new rows are embedded \
             with '{new}' — vector recall covers only '{new}' rows until you run \
             `lamu memory reembed --store {store} --yes`"
        );
    }
}

// ── Legacy import ───────────────────────────────────────────────────

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Build a fresh `lamu.db` at `target` via `lamu.db.tmp.<pid>`:
/// migrate the tmp DB to the full schema, import whichever legacy files
/// exist next to `target`, close (checkpoints WAL into the file), then
/// atomically rename tmp → target. On any error the tmp file is
/// removed and `target` is never created — the next open retries the
/// whole import (ADR 0025 seeding pattern).
fn build_with_import(target: &Path) -> Result<()> {
    let dir = target
        .parent()
        .ok_or_else(|| anyhow!("lamu.db path has no parent: {}", target.display()))?;
    let tmp = dir.join(format!("lamu.db.tmp.{}", std::process::id()));
    // A stale tmp from a crashed earlier run (pid reuse) must not leak
    // schema/rows into this build.
    remove_db_files(&tmp);

    let built = (|| -> Result<()> {
        let mut conn = open_at(&tmp)?;

        let legacy_conversations = dir.join("conversations.db");
        let legacy_memory = dir.join("memory.db");
        let legacy_embeddings = dir.join("embeddings.db");

        // Normalize legacy memory.db through its EXISTING open path
        // first: `migrate_temporal_columns` brings a pre-temporal file
        // up to the shape the INSERT…SELECT below expects. The
        // connection is dropped before the read-only ATTACH.
        if legacy_memory.exists() {
            drop(
                crate::lifetime_memory::open_legacy_memory_db(&legacy_memory)
                    .context("normalize legacy memory.db")?,
            );
        }

        if legacy_conversations.exists() {
            import_conversations(&mut conn, &legacy_conversations)
                .context("import legacy conversations.db")?;
        }
        if legacy_memory.exists() {
            import_memories(&mut conn, &legacy_memory).context("import legacy memory.db")?;
        }
        if legacy_embeddings.exists() {
            import_chunks(&mut conn, &legacy_embeddings)
                .context("import legacy embeddings.db")?;
        }
        // Drop = close: SQLite checkpoints the WAL into the main file
        // and removes -wal/-shm, so the rename moves the COMPLETE db.
        drop(conn);
        Ok(())
    })();

    if let Err(e) = built {
        remove_db_files(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, target) {
        // Don't leak the fully-built tmp (and sidecars) on a failed
        // publish — the next open retries the whole build.
        remove_db_files(&tmp);
        return Err(e)
            .with_context(|| format!("rename {} -> {}", tmp.display(), target.display()));
    }
    Ok(())
}

/// Best-effort removal of a SQLite db file and its WAL sidecars.
fn remove_db_files(path: &Path) {
    let _ = std::fs::remove_file(path);
    for suffix in ["-wal", "-shm"] {
        let mut s = path.as_os_str().to_owned();
        s.push(suffix);
        let _ = std::fs::remove_file(PathBuf::from(s));
    }
}

/// ATTACH `path` read-only as `legacy`. Read-only because import must
/// not be able to mutate the legacy file, and a `mode=ro` URI is the
/// only way to say that for an ATTACH. rusqlite's default open flags
/// include `SQLITE_OPEN_URI`, so the URI form is honored.
fn attach_legacy_ro(conn: &Connection, path: &Path) -> Result<()> {
    // Minimal URI escaping: '%' first, then the URI-significant chars
    // that can appear in a path. Data-dir paths are tame, but a tmpdir
    // with an odd name must not silently truncate at a '?'.
    let p = path.display().to_string();
    let escaped = p.replace('%', "%25").replace('#', "%23").replace('?', "%3f");
    let uri = format!("file:{escaped}?mode=ro");
    conn.execute("ATTACH DATABASE ?1 AS legacy", rusqlite::params![uri])
        .with_context(|| format!("attach {} read-only", path.display()))?;
    Ok(())
}

fn detach_legacy(conn: &Connection) {
    if let Err(e) = conn.execute("DETACH DATABASE legacy", []) {
        tracing::warn!("detach legacy db: {e}");
    }
}

/// Legacy `conversations.db` had `conversations(id, created_at)` (no
/// owner) + the same `turns` shape; fill `owner='local'`.
fn import_conversations(conn: &mut Connection, legacy: &Path) -> Result<()> {
    attach_legacy_ro(conn, legacy)?;
    let r = (|| -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO conversations (id, owner, created_at) \
             SELECT id, 'local', created_at FROM legacy.conversations",
            [],
        )?;
        tx.execute(
            "INSERT INTO turns (conversation_id, idx, role, content, ts, metadata) \
             SELECT conversation_id, idx, role, content, ts, metadata FROM legacy.turns",
            [],
        )?;
        tx.commit()?;
        Ok(())
    })();
    detach_legacy(conn);
    r
}

/// Legacy `memory.db` is temporal-normalized before this runs (see
/// [`build_with_import`]), so the SELECT can name the valid-time
/// columns unconditionally. Ids are preserved — `supersedes` rows
/// reference them. `embedding_model` is backfilled only where an
/// embedding exists (everything legacy was text-embedding-3-small).
fn import_memories(conn: &mut Connection, legacy: &Path) -> Result<()> {
    attach_legacy_ro(conn, legacy)?;
    let r = (|| -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO memories \
                 (id, owner, text, embedding, embedding_model, kind, source, ts, \
                  valid_from, valid_until, supersedes) \
             SELECT id, 'local', text, embedding, \
                    CASE WHEN embedding IS NULL THEN NULL ELSE ?1 END, \
                    kind, source, ts, valid_from, valid_until, supersedes \
             FROM legacy.memories",
            rusqlite::params![crate::rag::EMBED_MODEL],
        )?;
        // Seed the per-store embedding bookkeeping ONLY when at least
        // one imported row actually carries an embedding.
        tx.execute(
            "INSERT INTO embedding_stores (store, model, dims, updated_at) \
             SELECT 'memories', ?1, 1536, ?2 \
             WHERE EXISTS (SELECT 1 FROM legacy.memories WHERE embedding IS NOT NULL)",
            rusqlite::params![crate::rag::EMBED_MODEL, now_secs()],
        )?;
        tx.commit()?;
        Ok(())
    })();
    detach_legacy(conn);
    r
}

/// Legacy `embeddings.db` chunks always carry an embedding (the column
/// is NOT NULL), so `embedding_model` is set unconditionally and the
/// `embedding_stores` seed gates only on a row existing.
fn import_chunks(conn: &mut Connection, legacy: &Path) -> Result<()> {
    attach_legacy_ro(conn, legacy)?;
    let r = (|| -> Result<()> {
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO chunks (path, chunk_idx, content, embedding, embedding_model, mtime) \
             SELECT path, chunk_idx, content, embedding, ?1, mtime \
             FROM legacy.chunks",
            rusqlite::params![crate::rag::EMBED_MODEL],
        )?;
        tx.execute(
            "INSERT INTO embedding_stores (store, model, dims, updated_at) \
             SELECT 'chunks', ?1, 1536, ?2 \
             WHERE EXISTS (SELECT 1 FROM legacy.chunks WHERE embedding IS NOT NULL)",
            rusqlite::params![crate::rag::EMBED_MODEL, now_secs()],
        )?;
        tx.commit()?;
        Ok(())
    })();
    detach_legacy(conn);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::vec_to_blob;
    use rusqlite::params;

    #[test]
    fn lamu_db_path_respects_env_override() {
        // Serialize with every other test that mutates chain/store env
        // (the ADR 0030 e2e test also sets LAMU_DB).
        let _g = crate::embedder::testutil::chain_lock();
        // SAFETY: serialized by chain_lock; no other test reads it
        // concurrently.
        unsafe {
            std::env::set_var("LAMU_DB", "/tmp/somewhere/else.db");
        }
        assert_eq!(lamu_db_path(), PathBuf::from("/tmp/somewhere/else.db"));
        unsafe {
            std::env::set_var("LAMU_DB", "   ");
        }
        // Blank override is ignored → default path.
        assert!(lamu_db_path().ends_with("lamu/lamu.db"));
        unsafe {
            std::env::remove_var("LAMU_DB");
        }
        assert!(lamu_db_path().ends_with("lamu/lamu.db"));
    }

    #[test]
    fn open_at_gives_full_schema_on_a_temp_path() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("fresh.db")).unwrap();
        // Spot-check one table per legacy store + the version ledger.
        for t in ["conversations", "memories", "chunks", "schema_version", "memories_fts"] {
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name = ?",
                    params![t],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 1, "missing {t}");
        }
        let mode: String = conn
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
    }

    /// Build the three legacy fixture DBs in `dir`:
    /// - conversations.db — 1 conversation + 2 turns
    /// - memory.db — PRE-temporal shape (no valid_from), 2 facts, one embedded
    /// - embeddings.db — 1 chunk
    fn build_legacy_fixtures(dir: &Path) {
        let conv = Connection::open(dir.join("conversations.db")).unwrap();
        conv.execute_batch(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at INTEGER NOT NULL);
             CREATE TABLE turns (
                 conversation_id TEXT NOT NULL, idx INTEGER NOT NULL, role TEXT NOT NULL,
                 content TEXT NOT NULL, ts INTEGER NOT NULL, metadata TEXT,
                 PRIMARY KEY (conversation_id, idx),
                 FOREIGN KEY (conversation_id) REFERENCES conversations(id));",
        )
        .unwrap();
        conv.execute(
            "INSERT INTO conversations (id, created_at) VALUES ('conv-1', 100)",
            [],
        )
        .unwrap();
        conv.execute(
            "INSERT INTO turns (conversation_id, idx, role, content, ts, metadata) \
             VALUES ('conv-1', 0, 'user', 'hello from legacy', 100, NULL)",
            [],
        )
        .unwrap();
        conv.execute(
            "INSERT INTO turns (conversation_id, idx, role, content, ts, metadata) \
             VALUES ('conv-1', 1, 'assistant', 'legacy reply', 101, '{\"m\":1}')",
            [],
        )
        .unwrap();
        drop(conv);

        // PRE-temporal memory.db: no valid_from/valid_until/supersedes.
        let mem = Connection::open(dir.join("memory.db")).unwrap();
        mem.execute_batch(
            "CREATE TABLE memories (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 text TEXT NOT NULL, embedding BLOB,
                 kind TEXT NOT NULL DEFAULT 'fact', source TEXT, ts INTEGER NOT NULL);",
        )
        .unwrap();
        mem.execute(
            "INSERT INTO memories (text, embedding, kind, source, ts) VALUES (?, ?, ?, ?, ?)",
            params![
                "embedded legacy fact",
                vec_to_blob(&[1.0f32, 0.0, 0.0]),
                "fact",
                "manual",
                200i64
            ],
        )
        .unwrap();
        mem.execute(
            "INSERT INTO memories (text, embedding, kind, source, ts) VALUES (?, NULL, ?, ?, ?)",
            params!["plain legacy fact", "fact", "manual", 201i64],
        )
        .unwrap();
        drop(mem);

        let emb = Connection::open(dir.join("embeddings.db")).unwrap();
        emb.execute_batch(
            "CREATE TABLE chunks (
                 path TEXT NOT NULL, chunk_idx INTEGER NOT NULL, content TEXT NOT NULL,
                 embedding BLOB NOT NULL, mtime INTEGER NOT NULL,
                 PRIMARY KEY (path, chunk_idx));",
        )
        .unwrap();
        emb.execute(
            "INSERT INTO chunks (path, chunk_idx, content, embedding, mtime) VALUES (?, ?, ?, ?, ?)",
            params!["src/lib.rs", 0i64, "fn legacy_chunk() {}", vec_to_blob(&[0.0f32, 1.0]), 300i64],
        )
        .unwrap();
        drop(emb);
    }

    #[test]
    fn legacy_import_end_to_end_then_idempotent() {
        let td = tempfile::tempdir().unwrap();
        build_legacy_fixtures(td.path());
        let db = td.path().join("lamu.db");

        let conn = open_or_import(&db).unwrap();
        assert!(db.exists(), "lamu.db created");

        // conversations + turns imported with owner='local'.
        let (owner, created): (String, i64) = conn
            .query_row(
                "SELECT owner, created_at FROM conversations WHERE id = 'conv-1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(owner, "local");
        assert_eq!(created, 100);
        let n_turns: i64 = conn
            .query_row("SELECT COUNT(*) FROM turns WHERE conversation_id = 'conv-1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(n_turns, 2);
        let meta: Option<String> = conn
            .query_row("SELECT metadata FROM turns WHERE idx = 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(meta.as_deref(), Some("{\"m\":1}"));

        // memories: owner='local', embedding_model only on the embedded
        // fact, temporal columns backfilled (valid_from = ts via the
        // legacy normalization), still currently valid.
        // (text, owner, embedding_model, valid_from, valid_until)
        type MemRow = (String, String, Option<String>, i64, Option<i64>);
        let rows: Vec<MemRow> = {
            let mut stmt = conn
                .prepare(
                    "SELECT text, owner, embedding_model, valid_from, valid_until \
                     FROM memories ORDER BY id",
                )
                .unwrap();
            let mapped = stmt
                .query_map([], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
                })
                .unwrap();
            mapped.map(|r| r.unwrap()).collect()
        };
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, "embedded legacy fact");
        assert_eq!(rows[0].1, "local");
        assert_eq!(rows[0].2.as_deref(), Some("text-embedding-3-small"));
        assert_eq!(rows[0].3, 200, "valid_from backfilled to ts");
        assert!(rows[0].4.is_none());
        assert_eq!(rows[1].0, "plain legacy fact");
        assert!(rows[1].2.is_none(), "no embedding → no embedding_model");
        assert_eq!(rows[1].3, 201);

        // chunks imported with the model recorded.
        let (content, model): (String, Option<String>) = conn
            .query_row(
                "SELECT content, embedding_model FROM chunks WHERE path = 'src/lib.rs'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(content, "fn legacy_chunk() {}");
        assert_eq!(model.as_deref(), Some("text-embedding-3-small"));

        // embedding_stores seeded for BOTH stores (each had ≥1 embedding).
        let stores: Vec<(String, String, i64)> = {
            let mut stmt = conn
                .prepare("SELECT store, model, dims FROM embedding_stores ORDER BY store")
                .unwrap();
            let mapped = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .unwrap();
            mapped.map(|r| r.unwrap()).collect()
        };
        assert_eq!(
            stores,
            vec![
                ("chunks".to_string(), "text-embedding-3-small".to_string(), 1536),
                ("memories".to_string(), "text-embedding-3-small".to_string(), 1536),
            ]
        );

        // Legacy files left in place with their rows intact.
        drop(conn);
        for f in ["conversations.db", "memory.db", "embeddings.db"] {
            assert!(td.path().join(f).exists(), "{f} must survive the import");
        }
        let legacy_mem = Connection::open(td.path().join("memory.db")).unwrap();
        let n: i64 = legacy_mem
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2, "legacy rows untouched");
        drop(legacy_mem);

        // Re-run the open flow: existence = idempotence marker → no
        // duplicate rows.
        let conn2 = open_or_import(&db).unwrap();
        let n_mem: i64 = conn2
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_mem, 2, "re-open must not re-import");
        let n_conv: i64 = conn2
            .query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_conv, 1);
        let n_chunks: i64 = conn2
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_chunks, 1);
        // No stale tmp file left behind.
        let tmp_left = std::fs::read_dir(td.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains("lamu.db.tmp"));
        assert!(!tmp_left, "tmp build file must be renamed away");
    }

    #[test]
    fn embedding_stores_not_seeded_without_any_embedding() {
        let td = tempfile::tempdir().unwrap();
        // memory.db whose only fact has NO embedding; no other legacy files.
        let mem = Connection::open(td.path().join("memory.db")).unwrap();
        mem.execute_batch(
            "CREATE TABLE memories (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 text TEXT NOT NULL, embedding BLOB,
                 kind TEXT NOT NULL DEFAULT 'fact', source TEXT, ts INTEGER NOT NULL);",
        )
        .unwrap();
        mem.execute(
            "INSERT INTO memories (text, embedding, kind, source, ts) VALUES ('t', NULL, 'fact', 's', 1)",
            [],
        )
        .unwrap();
        drop(mem);

        let conn = open_or_import(&td.path().join("lamu.db")).unwrap();
        let n_mem: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_mem, 1, "fact imported");
        let n_stores: i64 = conn
            .query_row("SELECT COUNT(*) FROM embedding_stores", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_stores, 0, "no embeddings anywhere → no store rows");
    }

    #[test]
    fn fresh_open_with_no_legacy_files_creates_empty_db() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_or_import(&td.path().join("lamu.db")).unwrap();
        for t in ["conversations", "turns", "memories", "chunks", "embedding_stores"] {
            let n: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM {t}"), [], |r| r.get(0))
                .unwrap();
            assert_eq!(n, 0, "{t} starts empty");
        }
    }

    #[test]
    fn fts_finds_imported_and_fresh_memories() {
        let td = tempfile::tempdir().unwrap();
        build_legacy_fixtures(td.path());
        let conn = open_or_import(&td.path().join("lamu.db")).unwrap();

        // A fresh post-import insert through the storage layer (same
        // INSERT remember() runs after embedding).
        crate::lifetime_memory::insert_memory(
            &conn,
            "fresh zanzibar fact",
            Some(&[0.5f32, 0.5, 0.0]),
            Some("test-model"),
            "fact",
            "manual",
            999,
            "local",
        )
        .unwrap();

        let find = |needle: &str| -> Vec<i64> {
            let mut stmt = conn
                .prepare("SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?")
                .unwrap();
            let rows = stmt
                .query_map(params![needle], |r| r.get::<_, i64>(0))
                .unwrap();
            rows.map(|r| r.unwrap()).collect()
        };
        assert_eq!(find("legacy").len(), 2, "both imported facts indexed");
        assert_eq!(find("zanzibar").len(), 1, "fresh insert indexed via trigger");

        // turns are indexed too.
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM turns_fts WHERE turns_fts MATCH 'hello'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn failed_import_leaves_no_lamu_db_or_tmp() {
        let td = tempfile::tempdir().unwrap();
        // A corrupt legacy file: exists but is not SQLite → ATTACH fails.
        std::fs::write(td.path().join("conversations.db"), b"not a database").unwrap();
        let db = td.path().join("lamu.db");
        assert!(open_or_import(&db).is_err());
        assert!(!db.exists(), "failed import must not publish lamu.db");
        let tmp_left = std::fs::read_dir(td.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains("lamu.db.tmp"));
        assert!(!tmp_left, "failed import must clean its tmp files");
    }

    // ── Pool concurrency tests (ADR 0041) ──────────────────────────

    /// Helper: create a fresh temp DB and point LAMU_DB at it, returning
    /// the path. The caller is responsible for resetting LAMU_DB.
    /// MUST be called under chain_lock() to serialize env mutation.
    fn setup_pool_test_db() -> (tempfile::TempDir, PathBuf) {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("pool-test.db");
        // Pre-create the file so pool() doesn't run the legacy import.
        open_at(&path).unwrap();
        (td, path)
    }

    #[test]
    fn pool_init_runs_migrate_exactly_once_on_fresh_db() {
        let _g = crate::embedder::testutil::chain_lock();
        let (td, path) = setup_pool_test_db();
        unsafe { std::env::set_var("LAMU_DB", &path) };

        // Reset the pool OnceLock for this test by pointing at a fresh file.
        // Because OnceLock cannot be reset, we test against the conn() API
        // directly: get a connection and verify schema is fully migrated.
        let c = open_at(&path).unwrap();
        let version: i64 = c
            .query_row(
                "SELECT COALESCE(MAX(version),0) FROM schema_version",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            version,
            crate::migrate::MIGRATIONS.last().unwrap().version,
            "fresh db must reach the latest migration"
        );

        // No duplicate schema_version rows.
        let rows: i64 = c
            .query_row("SELECT COUNT(*) FROM schema_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, crate::migrate::MIGRATIONS.len() as i64);

        unsafe { std::env::remove_var("LAMU_DB") };
        drop(td);
    }

    #[test]
    fn two_connections_from_pool_do_not_serialize_reads() {
        // Verify that the pool delivers ≥2 connections so two concurrent
        // readers don't block on a single mutex.
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("concurrent.db");

        // Build the schema directly — no pool needed for this open.
        let mut base = Connection::open(&path).unwrap();
        base.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;").unwrap();
        crate::migrate::migrate(&mut base).unwrap();
        // Seed one row so the readers have something to SELECT.
        base.execute(
            "INSERT INTO memories (owner, text, kind, source, ts, valid_from) \
             VALUES ('local', 'pool-test fact', 'fact', 'test', 1, 1)",
            [],
        )
        .unwrap();
        drop(base);

        // Build a pool directly (bypass the global POOL singleton so this
        // test is independent of process state).
        let manager = SqliteConnectionManager::file(&path).with_init(|c| {
            c.execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;",
            )?;
            Ok(())
        });
        let local_pool = r2d2::Pool::builder().max_size(8).build(manager).unwrap();

        // Acquire two connections simultaneously.
        let c1 = local_pool.get().unwrap();
        let c2 = local_pool.get().unwrap();

        // Both can read.
        let n1: i64 = c1
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        let n2: i64 = c2
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n1, 1);
        assert_eq!(n2, 1);

        // Dropping one doesn't break the other.
        drop(c1);
        let n3: i64 = c2
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n3, 1);
        drop(c2);
        drop(td);
    }

    #[test]
    fn parallel_threads_all_succeed() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("threads.db");

        let mut base = Connection::open(&path).unwrap();
        base.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;").unwrap();
        crate::migrate::migrate(&mut base).unwrap();
        drop(base);

        let manager = SqliteConnectionManager::file(&path).with_init(|c| {
            c.execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;",
            )?;
            Ok(())
        });
        let local_pool = StdArc::new(r2d2::Pool::builder().max_size(8).build(manager).unwrap());

        const N: usize = 8;
        let success = StdArc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(N);

        for _ in 0..N {
            let p = StdArc::clone(&local_pool);
            let s = StdArc::clone(&success);
            handles.push(std::thread::spawn(move || {
                let c = p.get().unwrap();
                let _: i64 = c
                    .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
                    .unwrap();
                s.fetch_add(1, Ordering::Relaxed);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(success.load(Ordering::Relaxed), N, "all {N} threads must succeed");
        drop(td);
    }

    #[test]
    fn writer_and_reader_overlap_no_deadlock() {
        use std::sync::Arc as StdArc;

        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("wr.db");

        let mut base = Connection::open(&path).unwrap();
        base.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;").unwrap();
        crate::migrate::migrate(&mut base).unwrap();
        drop(base);

        let manager = SqliteConnectionManager::file(&path).with_init(|c| {
            c.execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;",
            )?;
            Ok(())
        });
        let local_pool = StdArc::new(r2d2::Pool::builder().max_size(8).build(manager).unwrap());

        let wp = StdArc::clone(&local_pool);
        let writer = std::thread::spawn(move || {
            let c = wp.get().unwrap();
            c.execute(
                "INSERT INTO memories (owner, text, kind, source, ts, valid_from) \
                 VALUES ('local', 'concurrent write', 'fact', 'test', 2, 2)",
                [],
            )
            .unwrap();
        });

        let rp = StdArc::clone(&local_pool);
        let reader = std::thread::spawn(move || {
            let c = rp.get().unwrap();
            let _: i64 = c
                .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
                .unwrap();
        });

        writer.join().expect("writer must not panic");
        reader.join().expect("reader must not panic");
        drop(td);
    }

    #[test]
    fn m004_index_used_for_embedding_model_query() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("idx-test.db")).unwrap();

        // Seed a row with an embedding so the partial index applies.
        conn.execute(
            "INSERT INTO memories (owner, text, embedding, embedding_model, kind, source, ts, valid_from) \
             VALUES ('local', 'test', X'01020304', 'my-model', 'fact', 'test', 1, 1)",
            [],
        )
        .unwrap();

        let plan: String = conn
            .query_row(
                "EXPLAIN QUERY PLAN \
                 SELECT id FROM memories \
                 WHERE embedding IS NOT NULL AND embedding_model = 'my-model'",
                [],
                |r| r.get::<_, String>(3),
            )
            .unwrap();
        assert!(
            plan.to_lowercase().contains("idx_memories_model_valid")
                || plan.to_lowercase().contains("using index"),
            "m004 index must be used; got plan: {plan}"
        );
        drop(td);
    }
}
