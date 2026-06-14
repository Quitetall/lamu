//! Versioned schema migrations for the unified `lamu.db` (ADR 0028).
//!
//! ADR 0028 replaces the three per-module SQLite files
//! (`conversations.db`, `memory.db`, `embeddings.db`) with ONE
//! schema-versioned database. This module owns the version ledger:
//!
//! - [`Migration`] — one schema step: a version, a name, and an `up`
//!   function that runs inside its own transaction.
//! - [`MIGRATIONS`] — the static, append-only list. New schema work is
//!   a NEW entry with the next version; existing entries are immutable
//!   once shipped (a shipped migration has already run on user DBs).
//! - [`migrate`] — bring a connection up to the latest version.
//!   Idempotent: applies only versions greater than the DB's current
//!   max, each in its own transaction, recording
//!   `(version, applied_at, description)` in `schema_version`.
//!
//! Pre-ADR-0028 in-place migrations (e.g. `memory.db`'s
//! `migrate_temporal_columns`) stay where they are — they normalize a
//! LEGACY file before its one-time import (see `store.rs`), they are
//! not part of this ledger.
//!
//! ## Error policy
//!
//! A malformed [`MIGRATIONS`] list (duplicate or out-of-order version)
//! is a programmer error, but we surface it as an `Err` rather than a
//! panic so a broken build refuses to touch the DB instead of aborting
//! mid-process — the store simply fails to open. A DB whose recorded
//! version is NEWER than this binary's latest migration is left alone
//! (warn + no-op): downgraded binaries read forward-compatible.

use anyhow::{anyhow, Context, Result};
use rusqlite::Connection;

/// One schema step. `up` runs inside a transaction owned by
/// [`migrate`]; it must not BEGIN/COMMIT itself.
pub struct Migration {
    pub version: i64,
    pub name: &'static str,
    pub up: fn(&rusqlite::Transaction) -> rusqlite::Result<()>,
}

/// The append-only migration ledger. Versions must be strictly
/// increasing; [`migrate`] validates this before touching the DB.
pub static MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "unified tables (conversations, turns, memories, chunks, embedding_stores, vector_index_state)",
        up: m001_unified_tables,
    },
    Migration {
        version: 2,
        name: "external-content FTS5 over memories(text) + turns(content)",
        up: m002_fts5,
    },
    Migration {
        version: 3,
        name: "causal event hypergraph (events, hyperedges, hyperedge_members)",
        up: m003_causal_graph,
    },
    Migration {
        version: 4,
        name: "embedding_model partial index",
        up: m004_embedding_model_index,
    },
];

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Bring `conn` up to the latest schema version. Creates the
/// `schema_version` table if absent, then applies every migration with
/// `version > MAX(schema_version.version)`, each in its own
/// transaction, recording `(version, applied_at, description)`.
/// Re-running on an up-to-date DB is a no-op.
pub fn migrate(conn: &mut Connection) -> Result<()> {
    migrate_with(conn, MIGRATIONS)
}

/// [`migrate`] over an explicit migration list. Factored out so tests
/// can drive validation/application with a local list; production code
/// always goes through [`migrate`] + [`MIGRATIONS`].
pub(crate) fn migrate_with(conn: &mut Connection, migrations: &[Migration]) -> Result<()> {
    // Validate BEFORE touching the DB: strictly increasing versions.
    // Duplicate or out-of-order entries are a programmer error — error
    // out (see module docs for the error-vs-panic call).
    let mut prev = 0i64;
    for m in migrations {
        if m.version <= prev {
            return Err(anyhow!(
                "MIGRATIONS not strictly increasing: version {} ({:?}) follows {}",
                m.version,
                m.name,
                prev
            ));
        }
        prev = m.version;
    }

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (\
             version INTEGER PRIMARY KEY, \
             applied_at INTEGER NOT NULL, \
             description TEXT NOT NULL\
         );",
    )
    .context("create schema_version table")?;

    let current: i64 = conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version",
        [],
        |r| r.get(0),
    )?;
    let latest = migrations.last().map(|m| m.version).unwrap_or(0);
    if current > latest {
        tracing::warn!(
            "lamu.db schema_version {current} is newer than this binary's latest \
             migration {latest}; leaving the schema alone (forward-compatible read)"
        );
        return Ok(());
    }

    for m in migrations.iter().filter(|m| m.version > current) {
        let tx = conn
            .transaction()
            .with_context(|| format!("begin migration {:03}", m.version))?;
        (m.up)(&tx).with_context(|| format!("apply migration {:03} ({})", m.version, m.name))?;
        tx.execute(
            "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
            rusqlite::params![m.version, now_secs(), m.name],
        )
        .with_context(|| format!("record migration {:03}", m.version))?;
        tx.commit()
            .with_context(|| format!("commit migration {:03}", m.version))?;
    }
    Ok(())
}

// ── Migration 001 — unified tables ──────────────────────────────────

/// The three legacy stores' current shapes, unified into one DB, plus
/// the new columns: `owner` (multi-user NEXT wave; everything is
/// 'local' today), `embedding_model` (provenance for mixed-model
/// futures), and the forward-looking `embedding_stores` /
/// `vector_index_state` bookkeeping tables (consumed by the persistent
/// vector-index work, ADR 0033/0034).
///
/// NOTE: the `turns -> conversations` FOREIGN KEY is declarative only —
/// `PRAGMA foreign_keys` stays at SQLite's default (OFF), matching the
/// legacy stores' behavior. Turning it on would make the legacy import
/// fail on historically-dangling turns. Enforce it only with a
/// dedicated migration that first repairs orphans.
fn m001_unified_tables(tx: &rusqlite::Transaction) -> rusqlite::Result<()> {
    tx.execute_batch(
        "
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL DEFAULT 'local',
    created_at INTEGER NOT NULL
);
CREATE TABLE turns (
    conversation_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts INTEGER NOT NULL,
    metadata TEXT,
    PRIMARY KEY (conversation_id, idx),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX idx_turns_ts ON turns(conversation_id, ts);
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL DEFAULT 'local',
    text TEXT NOT NULL,
    embedding BLOB,
    embedding_model TEXT,
    kind TEXT NOT NULL DEFAULT 'fact',
    source TEXT,
    ts INTEGER NOT NULL,
    valid_from INTEGER NOT NULL DEFAULT 0,
    valid_until INTEGER,
    supersedes INTEGER
);
CREATE INDEX idx_memories_ts ON memories(ts);
CREATE INDEX idx_memories_valid ON memories(valid_until);
CREATE INDEX idx_memories_owner ON memories(owner, valid_until);
CREATE TABLE chunks (
    path TEXT NOT NULL,
    chunk_idx INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_model TEXT,
    mtime INTEGER NOT NULL,
    PRIMARY KEY (path, chunk_idx)
);
CREATE TABLE embedding_stores (
    store TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    dims INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE vector_index_state (
    store TEXT PRIMARY KEY,
    last_indexed_rowid INTEGER NOT NULL DEFAULT 0,
    stale_count INTEGER NOT NULL DEFAULT 0,
    model TEXT,
    dims INTEGER,
    built_at INTEGER
);
",
    )
}

// ── Migration 002 — FTS5 ────────────────────────────────────────────

/// External-content FTS5 over `memories(text)` and `turns(content)`.
///
/// `turns` has a composite PK, so its FTS rides on SQLite's IMPLICIT
/// rowid (`content_rowid='rowid'`, triggers use `new.rowid`). That
/// implies `turns` must never become a WITHOUT ROWID table — doing so
/// silently breaks these triggers.
/// rusqlite's bundled SQLite ships FTS5 unconditionally, so no feature
/// probe is needed. Insert/delete/update triggers keep the index in
/// sync; the trailing `('rebuild')` backfills rows that existed before
/// this migration ran (e.g. a version-1 DB that already carried the
/// legacy import). `turns` has no INTEGER PRIMARY KEY alias, so its
/// implicit rowid is the content_rowid.
fn m002_fts5(tx: &rusqlite::Transaction) -> rusqlite::Result<()> {
    tx.execute_batch(
        "
CREATE VIRTUAL TABLE memories_fts USING fts5(text, content='memories', content_rowid='id');
CREATE TRIGGER memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER memories_fts_au AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE VIRTUAL TABLE turns_fts USING fts5(content, content='turns', content_rowid='rowid');
CREATE TRIGGER turns_fts_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER turns_fts_ad AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER turns_fts_au AFTER UPDATE OF content ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO turns_fts(rowid, content) VALUES (new.rowid, new.content);
END;
INSERT INTO memories_fts(memories_fts) VALUES ('rebuild');
INSERT INTO turns_fts(turns_fts) VALUES ('rebuild');
",
    )
}

// ── Migration 003 — causal event hypergraph ─────────────────────────

/// Three tables for the causal event hypergraph (ADR 0039):
/// - `events` — content-addressed fact nodes (b3: hash key, owner-scoped,
///   valid-time columns matching the `memories` pattern).
/// - `hyperedges` — n-ary directional causal relations (one row per
///   relation instance; owner-scoped; valid-time).
/// - `hyperedge_members` — members of each hyperedge with a `role` tag
///   ('cause' or 'effect').
///
/// FK enforcement is OFF (SQLite default; PRAGMA foreign_keys stays OFF):
/// orphan members + dangling `memory_id` references are valid states.
fn m003_causal_graph(tx: &rusqlite::Transaction) -> rusqlite::Result<()> {
    tx.execute_batch(
        "
CREATE TABLE events (
    node_hash  TEXT NOT NULL PRIMARY KEY,
    owner      TEXT NOT NULL DEFAULT 'local',
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    valid_from INTEGER NOT NULL DEFAULT 0,
    valid_until INTEGER,
    memory_id  INTEGER
);
CREATE INDEX idx_events_owner_valid ON events(owner, valid_until);
CREATE INDEX idx_events_owner_ts    ON events(owner, ts DESC);
CREATE INDEX idx_events_memory_id   ON events(memory_id) WHERE memory_id IS NOT NULL;

CREATE TABLE hyperedges (
    id         INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    owner      TEXT NOT NULL DEFAULT 'local',
    relation   TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    valid_from INTEGER NOT NULL DEFAULT 0,
    valid_until INTEGER
);
CREATE INDEX idx_hyperedges_owner_valid ON hyperedges(owner, valid_until);

CREATE TABLE hyperedge_members (
    hyperedge_id INTEGER NOT NULL,
    node_hash    TEXT NOT NULL,
    role         TEXT NOT NULL,
    PRIMARY KEY (hyperedge_id, node_hash, role)
);
CREATE INDEX idx_members_cause  ON hyperedge_members(node_hash, role) WHERE role = 'cause';
CREATE INDEX idx_members_effect ON hyperedge_members(node_hash, role) WHERE role = 'effect';
CREATE INDEX idx_members_edge   ON hyperedge_members(hyperedge_id);
",
    )
}

// ── Migration 004 — embedding_model partial index ───────────────────

/// Partial index on `memories(embedding_model, valid_until)` where an
/// embedding is present. The model-filtered vector leg in recall
/// (`WHERE embedding IS NOT NULL AND embedding_model = ?`) and the
/// novelty probe both benefit: only embedded rows enter the index, so
/// the scan is proportional to the embedded subset, not the whole table.
fn m004_embedding_model_index(tx: &rusqlite::Transaction) -> rusqlite::Result<()> {
    tx.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_memories_model_valid \
         ON memories(embedding_model, valid_until) \
         WHERE embedding IS NOT NULL;",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_names(conn: &Connection) -> std::collections::HashSet<String> {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type IN ('table','index','trigger')")
            .unwrap();
        let rows = stmt.query_map([], |r| r.get::<_, String>(0)).unwrap();
        rows.map(|r| r.unwrap()).collect()
    }

    #[test]
    fn fresh_db_gets_all_migrations_in_order() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();

        // Every unified table + the FTS layer + the causal graph is present.
        let names = table_names(&conn);
        for t in [
            "schema_version",
            "conversations",
            "turns",
            "memories",
            "chunks",
            "embedding_stores",
            "vector_index_state",
            "memories_fts",
            "turns_fts",
            "idx_turns_ts",
            "idx_memories_ts",
            "idx_memories_valid",
            "idx_memories_owner",
            "memories_fts_ai",
            "turns_fts_ai",
            // m003 — causal event hypergraph
            "events",
            "hyperedges",
            "hyperedge_members",
            "idx_events_owner_valid",
            "idx_events_owner_ts",
            "idx_events_memory_id",
            "idx_hyperedges_owner_valid",
            "idx_members_cause",
            "idx_members_effect",
            "idx_members_edge",
            // m004 — embedding_model partial index
            "idx_memories_model_valid",
        ] {
            assert!(names.contains(t), "missing schema object: {t}");
        }

        // schema_version rows recorded in ascending order with metadata.
        let mut stmt = conn
            .prepare("SELECT version, applied_at, description FROM schema_version ORDER BY version")
            .unwrap();
        let rows: Vec<(i64, i64, String)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(rows.len(), MIGRATIONS.len());
        for (row, m) in rows.iter().zip(MIGRATIONS) {
            assert_eq!(row.0, m.version);
            assert!(row.1 > 0, "applied_at recorded");
            assert_eq!(row.2, m.name);
        }
    }

    #[test]
    fn rerun_is_a_noop() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        let before: i64 = conn
            .query_row("SELECT COUNT(*) FROM schema_version", [], |r| r.get(0))
            .unwrap();
        migrate(&mut conn).unwrap();
        let after: i64 = conn
            .query_row("SELECT COUNT(*) FROM schema_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(before, after, "re-run must not re-apply or re-record");
    }

    #[test]
    fn duplicate_version_is_an_error() {
        fn noop(_tx: &rusqlite::Transaction) -> rusqlite::Result<()> {
            Ok(())
        }
        let bad = [
            Migration { version: 1, name: "a", up: noop },
            Migration { version: 1, name: "b", up: noop },
        ];
        let mut conn = Connection::open_in_memory().unwrap();
        let err = migrate_with(&mut conn, &bad).unwrap_err();
        assert!(err.to_string().contains("strictly increasing"), "{err}");
    }

    #[test]
    fn out_of_order_version_is_an_error() {
        fn noop(_tx: &rusqlite::Transaction) -> rusqlite::Result<()> {
            Ok(())
        }
        let bad = [
            Migration { version: 2, name: "b", up: noop },
            Migration { version: 1, name: "a", up: noop },
        ];
        let mut conn = Connection::open_in_memory().unwrap();
        assert!(migrate_with(&mut conn, &bad).is_err());
    }

    #[test]
    fn newer_db_than_binary_is_left_alone() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        // Simulate a DB stamped by a future binary.
        conn.execute(
            "INSERT INTO schema_version (version, applied_at, description) VALUES (999, 1, 'future')",
            [],
        )
        .unwrap();
        // No error, no new rows below 999.
        migrate(&mut conn).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM schema_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, MIGRATIONS.len() as i64 + 1);
    }

    #[test]
    fn each_migration_runs_in_its_own_transaction() {
        // A failing migration 2 must leave migration 1 applied + recorded.
        fn ok(tx: &rusqlite::Transaction) -> rusqlite::Result<()> {
            tx.execute_batch("CREATE TABLE t1 (x)")
        }
        fn boom(tx: &rusqlite::Transaction) -> rusqlite::Result<()> {
            tx.execute_batch("CREATE TABLE t2 (x); THIS IS NOT SQL;")
        }
        let list = [
            Migration { version: 1, name: "ok", up: ok },
            Migration { version: 2, name: "boom", up: boom },
        ];
        let mut conn = Connection::open_in_memory().unwrap();
        assert!(migrate_with(&mut conn, &list).is_err());
        let max: i64 = conn
            .query_row("SELECT COALESCE(MAX(version),0) FROM schema_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(max, 1, "migration 1 committed, migration 2 rolled back");
        let names = table_names(&conn);
        assert!(names.contains("t1"));
        assert!(!names.contains("t2"), "failed migration's DDL rolled back");
    }
}
