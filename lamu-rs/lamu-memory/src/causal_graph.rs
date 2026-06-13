//! Causal event hypergraph — storage core (ADR 0039).
//!
//! Three tables live in `lamu.db` after migration 003:
//! - `events` — content-addressed fact nodes keyed by a BLAKE3 hash.
//! - `hyperedges` — n-ary directional causal relations.
//! - `hyperedge_members` — cause/effect membership for each edge.
//!
//! ## Lock discipline
//!
//! Storage functions take `&Connection` / `&mut Connection` directly.
//! The `shared_handle()` + lock happen in the MCP handler layer; these
//! fns never acquire the parking_lot guard themselves and must never hold
//! it across a cross-module call.
//!
//! ## Idempotence
//!
//! `record_event` uses `INSERT OR IGNORE` so the same (kind, text) pair
//! always returns the same hash regardless of how many times it is
//! called. `link_events` is NOT idempotent by design — each call
//! records a distinct causal assertion; callers that want dedup must
//! check first.
//!
//! ## Cycle safety
//!
//! `trace_causal` guards against cycles with a path string that tracks
//! visited hashes (`'|' || hash || '|'`). The BLAKE3 hex output is
//! lowercase ASCII hex, which never contains `|`, making the sentinel
//! unambiguous.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};

// ── Helpers ────────────────────────────────────────────────────────

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ── Public types ────────────────────────────────────────────────────

/// Walk direction for [`trace_causal`] and [`neighbors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceDirection {
    /// Follow cause→effect edges outward from the seed.
    Downstream,
    /// Follow effect→cause edges back toward causes of the seed.
    Upstream,
}

/// One node returned by a causal traversal.
#[derive(Debug, Clone)]
pub struct CausalNode {
    pub node_hash: String,
    pub depth: u32,
    pub kind: String,
    pub text: String,
    pub ts: i64,
    pub valid_until: Option<i64>,
    pub memory_id: Option<i64>,
}

// ── Core storage functions ──────────────────────────────────────────

/// Compute the canonical content hash for an event.
///
/// Canonical bytes: `"{kind_trimmed}\x00{text_whitespace_normalized}"`.
/// Returns `"b3:{hex}"`. Deterministic and whitespace-normalised so that
/// equivalent events with cosmetic formatting differences map to the same
/// node in the graph.
pub fn content_hash(kind: &str, text: &str) -> String {
    let normalized_text = text
        .split_ascii_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let canonical = format!("{}\x00{}", kind.trim(), normalized_text);
    let hash = blake3::hash(canonical.as_bytes());
    format!("b3:{}", hash.to_hex())
}

/// Record an event node, returning its content hash.
///
/// Uses `INSERT OR IGNORE` so the call is idempotent: if the node
/// already exists the hash is returned unchanged. `memory_id` links the
/// event to a row in the `memories` table (optional; FK not enforced).
pub fn record_event(
    conn: &Connection,
    kind: &str,
    text: &str,
    owner: &str,
    memory_id: Option<i64>,
) -> Result<String> {
    let hash = content_hash(kind, text);
    let ts = now_secs();
    conn.execute(
        "INSERT OR IGNORE INTO events \
         (node_hash, owner, kind, text, ts, valid_from, valid_until, memory_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, 0, NULL, ?6)",
        params![hash, owner, kind.trim(), text, ts, memory_id],
    )
    .context("insert OR IGNORE into events")?;
    Ok(hash)
}

/// Link a set of event nodes with a named causal relation.
///
/// `members` is a slice of `(node_hash, role)` pairs. Roles are
/// typically `"cause"` or `"effect"`, but the schema places no
/// constraint beyond the partial indexes that target those two values.
///
/// The entire insertion (one `hyperedges` row + all `hyperedge_members`
/// rows) is atomic. On any member-insert failure the transaction is
/// rolled back and the error is returned. Returns the new edge id.
pub fn link_events(
    conn: &mut Connection,
    relation: &str,
    owner: &str,
    members: &[(String, String)],
) -> Result<i64> {
    let ts = now_secs();
    let tx = conn.transaction().context("begin link_events transaction")?;
    tx.execute(
        "INSERT INTO hyperedges (owner, relation, ts, valid_from, valid_until) \
         VALUES (?1, ?2, ?3, 0, NULL)",
        params![owner, relation, ts],
    )
    .context("insert hyperedge row")?;
    let edge_id = tx.last_insert_rowid();
    for (node_hash, role) in members {
        tx.execute(
            "INSERT INTO hyperedge_members (hyperedge_id, node_hash, role) VALUES (?1, ?2, ?3)",
            params![edge_id, node_hash, role],
        )
        .with_context(|| {
            format!("insert hyperedge_member (edge={edge_id}, hash={node_hash}, role={role})")
        })?;
    }
    tx.commit().context("commit link_events transaction")?;
    Ok(edge_id)
}

// ── SQL templates for trace_causal ─────────────────────────────────
//
// Two strings compiled at call-time (via string substitution at function
// scope) so the literal role strings bake in and the partial indexes on
// `hyperedge_members(node_hash, role) WHERE role = 'cause'|'effect'` fire.
//
// Bindings: ?1=start_hash, ?2=max_depth, ?3=owner, ?4=now

const TRACE_SQL_DOWNSTREAM: &str = "
WITH RECURSIVE causal_walk(node_hash, depth, path) AS (
    SELECT e.node_hash, 0, '|' || e.node_hash || '|'
    FROM events e
    WHERE e.node_hash = ?1 AND e.owner = ?3
      AND (e.valid_until IS NULL OR e.valid_until > ?4)
    UNION ALL
    SELECT next_e.node_hash, cw.depth + 1, cw.path || next_e.node_hash || '|'
    FROM causal_walk cw
    JOIN hyperedge_members hm_seed
        ON hm_seed.node_hash = cw.node_hash AND hm_seed.role = 'cause'
    JOIN hyperedges he
        ON he.id = hm_seed.hyperedge_id AND he.owner = ?3
        AND (he.valid_until IS NULL OR he.valid_until > ?4)
    JOIN hyperedge_members hm_harvest
        ON hm_harvest.hyperedge_id = hm_seed.hyperedge_id AND hm_harvest.role = 'effect'
    JOIN events next_e
        ON next_e.node_hash = hm_harvest.node_hash AND next_e.owner = ?3
        AND (next_e.valid_until IS NULL OR next_e.valid_until > ?4)
    WHERE cw.depth < ?2
      AND INSTR(cw.path, '|' || next_e.node_hash || '|') = 0
)
SELECT cw.node_hash, cw.depth, e.kind, e.text, e.ts, e.valid_until, e.memory_id
FROM causal_walk cw JOIN events e ON e.node_hash = cw.node_hash
WHERE cw.depth > 0
ORDER BY cw.depth ASC, e.ts ASC;
";

const TRACE_SQL_UPSTREAM: &str = "
WITH RECURSIVE causal_walk(node_hash, depth, path) AS (
    SELECT e.node_hash, 0, '|' || e.node_hash || '|'
    FROM events e
    WHERE e.node_hash = ?1 AND e.owner = ?3
      AND (e.valid_until IS NULL OR e.valid_until > ?4)
    UNION ALL
    SELECT next_e.node_hash, cw.depth + 1, cw.path || next_e.node_hash || '|'
    FROM causal_walk cw
    JOIN hyperedge_members hm_seed
        ON hm_seed.node_hash = cw.node_hash AND hm_seed.role = 'effect'
    JOIN hyperedges he
        ON he.id = hm_seed.hyperedge_id AND he.owner = ?3
        AND (he.valid_until IS NULL OR he.valid_until > ?4)
    JOIN hyperedge_members hm_harvest
        ON hm_harvest.hyperedge_id = hm_seed.hyperedge_id AND hm_harvest.role = 'cause'
    JOIN events next_e
        ON next_e.node_hash = hm_harvest.node_hash AND next_e.owner = ?3
        AND (next_e.valid_until IS NULL OR next_e.valid_until > ?4)
    WHERE cw.depth < ?2
      AND INSTR(cw.path, '|' || next_e.node_hash || '|') = 0
)
SELECT cw.node_hash, cw.depth, e.kind, e.text, e.ts, e.valid_until, e.memory_id
FROM causal_walk cw JOIN events e ON e.node_hash = cw.node_hash
WHERE cw.depth > 0
ORDER BY cw.depth ASC, e.ts ASC;
";

/// Walk the causal graph from `start_hash` up to `max_depth` hops.
///
/// Returns all reachable nodes (depth > 0), deduped by `node_hash`
/// keeping the minimum depth (UNION ALL can visit the same node via
/// multiple paths), then sorted by `(depth asc, ts asc)`.
///
/// - Downstream: seed is a cause; harvests effects.
/// - Upstream: seed is an effect; harvests causes.
///
/// Cycles are terminated by the `INSTR(path, '|' || hash || '|') = 0`
/// guard in the CTE — a node already in the path is never re-queued.
/// `max_depth` is clamped to 100 to prevent runaway queries on
/// pathological graphs.
///
/// `now` is passed explicitly so callers (tests and handlers) control
/// the validity cutoff; pass `0` to include all nodes regardless of
/// expiry (equivalent to `include_expired = true`).
pub fn trace_causal(
    conn: &Connection,
    start_hash: &str,
    direction: TraceDirection,
    max_depth: u32,
    owner: &str,
    now: i64,
) -> Result<Vec<CausalNode>> {
    let depth_cap = max_depth.min(100) as i64;
    let sql = match direction {
        TraceDirection::Downstream => TRACE_SQL_DOWNSTREAM,
        TraceDirection::Upstream => TRACE_SQL_UPSTREAM,
    };
    let mut stmt = conn.prepare(sql).context("prepare trace_causal")?;
    let rows = stmt
        .query_map(params![start_hash, depth_cap, owner, now], |r| {
            Ok(CausalNode {
                node_hash: r.get(0)?,
                depth: r.get::<_, i64>(1)? as u32,
                kind: r.get(2)?,
                text: r.get(3)?,
                ts: r.get(4)?,
                valid_until: r.get(5)?,
                memory_id: r.get(6)?,
            })
        })
        .context("execute trace_causal")?;

    // Collect all rows (UNION ALL may repeat a node via multiple paths).
    let mut all: Vec<CausalNode> = Vec::new();
    for row in rows {
        all.push(row.context("read trace_causal row")?);
    }

    // Dedup: keep minimum depth per node_hash.
    let mut best: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for n in &all {
        let e = best.entry(n.node_hash.clone()).or_insert(n.depth);
        if n.depth < *e {
            *e = n.depth;
        }
    }
    let mut deduped: Vec<CausalNode> = all
        .into_iter()
        .filter(|n| best.get(&n.node_hash) == Some(&n.depth))
        // After the filter there may still be multiple rows at the same
        // minimum depth if UNION ALL emitted the node twice at the same
        // depth. Deduplicate by keeping the first occurrence (which will
        // be the earliest by `ts` after the final sort, since the CTE
        // ORDER BY emits depth-ascending + ts-ascending rows).
        .collect();
    // Now do a final pass to guarantee uniqueness at the hash level.
    {
        let mut seen = std::collections::HashSet::new();
        deduped.retain(|n| seen.insert(n.node_hash.clone()));
    }
    deduped.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.ts.cmp(&b.ts)));
    Ok(deduped)
}

/// Return the immediate neighbors of `start_hash` at depth 1.
///
/// Equivalent to `trace_causal(…, max_depth=1)` but implemented as a
/// flat two-join query (no CTE) for efficiency. Depth is always 1 for
/// every returned node.
///
/// See [`trace_causal`] for `now` and direction semantics.
pub fn neighbors(
    conn: &Connection,
    start_hash: &str,
    direction: TraceDirection,
    owner: &str,
    now: i64,
) -> Result<Vec<CausalNode>> {
    let (seed_role, harvest_role) = match direction {
        TraceDirection::Downstream => ("cause", "effect"),
        TraceDirection::Upstream => ("effect", "cause"),
    };
    let sql = format!(
        "SELECT DISTINCT next_e.node_hash, 1 AS depth, next_e.kind, next_e.text,
                next_e.ts, next_e.valid_until, next_e.memory_id
         FROM events seed_e
         JOIN hyperedge_members hm_seed
             ON hm_seed.node_hash = seed_e.node_hash AND hm_seed.role = '{seed_role}'
         JOIN hyperedges he
             ON he.id = hm_seed.hyperedge_id AND he.owner = ?2
             AND (he.valid_until IS NULL OR he.valid_until > ?3)
         JOIN hyperedge_members hm_harvest
             ON hm_harvest.hyperedge_id = hm_seed.hyperedge_id AND hm_harvest.role = '{harvest_role}'
         JOIN events next_e
             ON next_e.node_hash = hm_harvest.node_hash AND next_e.owner = ?2
             AND (next_e.valid_until IS NULL OR next_e.valid_until > ?3)
         WHERE seed_e.node_hash = ?1 AND seed_e.owner = ?2
           AND (seed_e.valid_until IS NULL OR seed_e.valid_until > ?3)
         ORDER BY next_e.ts ASC"
    );
    let mut stmt = conn.prepare(&sql).context("prepare neighbors")?;
    let rows = stmt
        .query_map(params![start_hash, owner, now], |r| {
            Ok(CausalNode {
                node_hash: r.get(0)?,
                depth: 1,
                kind: r.get(2)?,
                text: r.get(3)?,
                ts: r.get(4)?,
                valid_until: r.get(5)?,
                memory_id: r.get(6)?,
            })
        })
        .context("execute neighbors")?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.context("read neighbors row")?);
    }
    Ok(out)
}

/// Return `true` if an event with `node_hash` owned by `owner` exists
/// in the `events` table (regardless of validity).
pub fn event_exists(conn: &Connection, node_hash: &str, owner: &str) -> Result<bool> {
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE node_hash = ?1 AND owner = ?2",
            params![node_hash, owner],
            |r| r.get(0),
        )
        .context("event_exists query")?;
    Ok(n > 0)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn open_test_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        crate::migrate::migrate(&mut conn).unwrap();
        conn
    }

    // ── content_hash dedupe ─────────────────────────────────────────

    #[test]
    fn same_kind_text_same_hash() {
        let h1 = content_hash("fact", "the sky is blue");
        let h2 = content_hash("fact", "the sky is blue");
        assert_eq!(h1, h2);
        assert!(h1.starts_with("b3:"), "hash prefix: {h1}");
    }

    #[test]
    fn whitespace_normalized_equal_hash() {
        let h1 = content_hash("fact", "the  sky   is\tblue");
        let h2 = content_hash("fact", "the sky is blue");
        assert_eq!(h1, h2, "whitespace normalization must collapse to same hash");
    }

    #[test]
    fn kind_trim_equal_hash() {
        let h1 = content_hash("  fact  ", "the sky is blue");
        let h2 = content_hash("fact", "the sky is blue");
        assert_eq!(h1, h2, "kind is trimmed before hashing");
    }

    #[test]
    fn different_kind_different_hash() {
        let h1 = content_hash("fact", "the sky is blue");
        let h2 = content_hash("preference", "the sky is blue");
        assert_ne!(h1, h2, "different kind must produce different hash");
    }

    #[test]
    fn different_text_different_hash() {
        let h1 = content_hash("fact", "the sky is blue");
        let h2 = content_hash("fact", "the grass is green");
        assert_ne!(h1, h2);
    }

    // ── record_event idempotence ────────────────────────────────────

    #[test]
    fn record_event_idempotent() {
        let conn = open_test_db();
        let h1 = record_event(&conn, "fact", "the sky is blue", "local", None).unwrap();
        let h2 = record_event(&conn, "fact", "the sky is blue", "local", None).unwrap();
        assert_eq!(h1, h2);
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1, "INSERT OR IGNORE must not duplicate the row");
    }

    #[test]
    fn record_event_stores_memory_id() {
        let conn = open_test_db();
        let h = record_event(&conn, "fact", "linked fact", "local", Some(42)).unwrap();
        let mid: Option<i64> = conn
            .query_row("SELECT memory_id FROM events WHERE node_hash = ?1", [&h], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(mid, Some(42));
    }

    // ── n-ary downstream + upstream ────────────────────────────────

    /// Build: causes A, B, C → effects D, E (one edge with 5 members).
    fn build_nary_fixture(conn: &mut Connection) -> (String, String, String, String, String) {
        let a = record_event(conn, "fact", "A", "local", None).unwrap();
        let b = record_event(conn, "fact", "B", "local", None).unwrap();
        let c = record_event(conn, "fact", "C", "local", None).unwrap();
        let d = record_event(conn, "fact", "D", "local", None).unwrap();
        let e = record_event(conn, "fact", "E", "local", None).unwrap();
        let members = vec![
            (a.clone(), "cause".to_string()),
            (b.clone(), "cause".to_string()),
            (c.clone(), "cause".to_string()),
            (d.clone(), "effect".to_string()),
            (e.clone(), "effect".to_string()),
        ];
        link_events(conn, "causes", "local", &members).unwrap();
        (a, b, c, d, e)
    }

    #[test]
    fn nary_downstream_from_a() {
        let mut conn = open_test_db();
        let (a, _b, _c, d, e) = build_nary_fixture(&mut conn);
        let now = now_secs();
        let results = trace_causal(&conn, &a, TraceDirection::Downstream, 1, "local", now).unwrap();
        let hashes: std::collections::HashSet<_> = results.iter().map(|n| &n.node_hash).collect();
        assert!(hashes.contains(&d), "D must be downstream of A");
        assert!(hashes.contains(&e), "E must be downstream of A");
        assert!(!hashes.contains(&a), "seed must not appear in results");
    }

    #[test]
    fn nary_upstream_from_d() {
        let mut conn = open_test_db();
        let (a, b, c, d, _e) = build_nary_fixture(&mut conn);
        let now = now_secs();
        let results = trace_causal(&conn, &d, TraceDirection::Upstream, 1, "local", now).unwrap();
        let hashes: std::collections::HashSet<_> = results.iter().map(|n| &n.node_hash).collect();
        assert!(hashes.contains(&a), "A must be upstream of D");
        assert!(hashes.contains(&b), "B must be upstream of D");
        assert!(hashes.contains(&c), "C must be upstream of D");
        assert!(!hashes.contains(&d), "seed must not appear in results");
    }

    // ── multi-hop ──────────────────────────────────────────────────

    /// Build A→B→C (two edges each with one cause→effect pair).
    fn build_chain(conn: &mut Connection) -> (String, String, String) {
        let a = record_event(conn, "fact", "chain-A", "local", None).unwrap();
        let b = record_event(conn, "fact", "chain-B", "local", None).unwrap();
        let c = record_event(conn, "fact", "chain-C", "local", None).unwrap();
        link_events(
            conn,
            "causes",
            "local",
            &[(a.clone(), "cause".to_string()), (b.clone(), "effect".to_string())],
        )
        .unwrap();
        link_events(
            conn,
            "causes",
            "local",
            &[(b.clone(), "cause".to_string()), (c.clone(), "effect".to_string())],
        )
        .unwrap();
        (a, b, c)
    }

    #[test]
    fn multihop_downstream_depth_2() {
        let mut conn = open_test_db();
        let (a, b, c) = build_chain(&mut conn);
        let now = now_secs();
        let results = trace_causal(&conn, &a, TraceDirection::Downstream, 2, "local", now).unwrap();
        let by_hash: std::collections::HashMap<_, _> =
            results.iter().map(|n| (&n.node_hash, n.depth)).collect();
        assert_eq!(by_hash.get(&b), Some(&1), "B at depth 1");
        assert_eq!(by_hash.get(&c), Some(&2), "C at depth 2");
    }

    #[test]
    fn multihop_downstream_depth_1_stops_at_b() {
        let mut conn = open_test_db();
        let (a, b, c) = build_chain(&mut conn);
        let now = now_secs();
        let results = trace_causal(&conn, &a, TraceDirection::Downstream, 1, "local", now).unwrap();
        let hashes: std::collections::HashSet<_> = results.iter().map(|n| &n.node_hash).collect();
        assert!(hashes.contains(&b));
        assert!(!hashes.contains(&c), "depth 1 must not reach C");
    }

    // ── cycle guard ────────────────────────────────────────────────

    #[test]
    fn cycle_guard_terminates_and_bounded() {
        let mut conn = open_test_db();
        let a = record_event(&conn, "fact", "cycle-A", "local", None).unwrap();
        let b = record_event(&conn, "fact", "cycle-B", "local", None).unwrap();
        // A→B and B→A
        link_events(
            &mut conn,
            "causes",
            "local",
            &[(a.clone(), "cause".to_string()), (b.clone(), "effect".to_string())],
        )
        .unwrap();
        link_events(
            &mut conn,
            "causes",
            "local",
            &[(b.clone(), "cause".to_string()), (a.clone(), "effect".to_string())],
        )
        .unwrap();
        let now = now_secs();
        // Large max_depth must not loop forever.
        let results =
            trace_causal(&conn, &a, TraceDirection::Downstream, 10, "local", now).unwrap();
        let hashes: std::collections::HashSet<_> = results.iter().map(|n| &n.node_hash).collect();
        // B is reachable at depth 1; A is the seed (depth 0, excluded).
        assert!(hashes.contains(&b), "B must be reachable");
        assert!(!hashes.contains(&a), "seed A must not appear in results");
        // No node appears more than once.
        assert_eq!(
            results.len(),
            hashes.len(),
            "no duplicates in cycle output"
        );
    }

    #[test]
    fn self_loop_terminates() {
        let mut conn = open_test_db();
        let a = record_event(&conn, "fact", "self-loop-A", "local", None).unwrap();
        // A is both cause and effect in the same edge.
        link_events(
            &mut conn,
            "causes",
            "local",
            &[
                (a.clone(), "cause".to_string()),
                (a.clone(), "effect".to_string()),
            ],
        )
        .unwrap();
        let now = now_secs();
        let results =
            trace_causal(&conn, &a, TraceDirection::Downstream, 10, "local", now).unwrap();
        // The self-loop produces A at depth 1, but the cycle guard stops
        // re-enqueuing it. Result: just A@1 (seed=A@0 is excluded).
        // However since A is the seed and the path guard blocks revisit,
        // the result should be empty (A@0 path = "|A|", and A@1 would
        // be blocked by INSTR check).
        // Actually: the effect A (same as seed) would be found at depth 1
        // if the guard permits it. The guard is INSTR(cw.path, '|A|') = 0
        // where cw.path = '|A|' at depth 0. INSTR('|A|', '|A|') = 1 ≠ 0.
        // So the self-loop IS blocked. Result must be empty.
        assert!(
            results.is_empty(),
            "self-loop: A cannot appear as its own effect (cycle guard blocks it)"
        );
    }

    // ── valid-time ─────────────────────────────────────────────────

    #[test]
    fn expired_effect_node_hidden() {
        let mut conn = open_test_db();
        let cause = record_event(&conn, "fact", "cause-node", "local", None).unwrap();
        let effect = record_event(&conn, "fact", "effect-node", "local", None).unwrap();
        link_events(
            &mut conn,
            "causes",
            "local",
            &[
                (cause.clone(), "cause".to_string()),
                (effect.clone(), "effect".to_string()),
            ],
        )
        .unwrap();
        // Expire the effect node: set valid_until to a past timestamp.
        conn.execute(
            "UPDATE events SET valid_until = 1 WHERE node_hash = ?1",
            [&effect],
        )
        .unwrap();
        let now = now_secs();
        let results =
            trace_causal(&conn, &cause, TraceDirection::Downstream, 5, "local", now).unwrap();
        assert!(results.is_empty(), "expired effect must be hidden");
    }

    #[test]
    fn expired_edge_hidden() {
        let mut conn = open_test_db();
        let cause = record_event(&conn, "fact", "cause-edge-test", "local", None).unwrap();
        let effect = record_event(&conn, "fact", "effect-edge-test", "local", None).unwrap();
        let edge_id = link_events(
            &mut conn,
            "causes",
            "local",
            &[
                (cause.clone(), "cause".to_string()),
                (effect.clone(), "effect".to_string()),
            ],
        )
        .unwrap();
        // Expire the edge.
        conn.execute(
            "UPDATE hyperedges SET valid_until = 1 WHERE id = ?1",
            [edge_id],
        )
        .unwrap();
        let now = now_secs();
        let results =
            trace_causal(&conn, &cause, TraceDirection::Downstream, 5, "local", now).unwrap();
        assert!(results.is_empty(), "expired edge must be hidden");
    }

    #[test]
    fn include_expired_via_now_zero() {
        let mut conn = open_test_db();
        let cause = record_event(&conn, "fact", "cause-expired-incl", "local", None).unwrap();
        let effect = record_event(&conn, "fact", "effect-expired-incl", "local", None).unwrap();
        link_events(
            &mut conn,
            "causes",
            "local",
            &[
                (cause.clone(), "cause".to_string()),
                (effect.clone(), "effect".to_string()),
            ],
        )
        .unwrap();
        // Expire both the node and the edge to a past timestamp.
        conn.execute("UPDATE events SET valid_until = 1 WHERE node_hash = ?1", [&effect]).unwrap();
        // With now=0: valid_until > 0 is true for valid_until=1, so expired row is visible.
        let results = trace_causal(&conn, &cause, TraceDirection::Downstream, 5, "local", 0).unwrap();
        let hashes: std::collections::HashSet<_> = results.iter().map(|n| &n.node_hash).collect();
        assert!(hashes.contains(&effect), "include_expired (now=0) must surface expired effect");
    }

    // ── owner isolation ────────────────────────────────────────────

    #[test]
    fn owner_isolation_nodes() {
        let conn = open_test_db();
        // Alice records a node; local queries must not see it.
        let alice_hash = record_event(&conn, "fact", "alice fact", "alice", None).unwrap();
        assert!(!event_exists(&conn, &alice_hash, "local").unwrap());
        assert!(event_exists(&conn, &alice_hash, "alice").unwrap());
    }

    #[test]
    fn cross_owner_edge_not_traversed() {
        let mut conn = open_test_db();
        // local owns the cause; alice owns the effect; edge owned by alice.
        let cause = record_event(&conn, "fact", "cross-cause", "local", None).unwrap();
        let effect = record_event(&conn, "fact", "cross-effect", "alice", None).unwrap();
        link_events(
            &mut conn,
            "causes",
            "alice",
            &[
                (cause.clone(), "cause".to_string()),
                (effect.clone(), "effect".to_string()),
            ],
        )
        .unwrap();
        // Querying as 'local': the edge is alice's → hidden from local.
        let now = now_secs();
        let results =
            trace_causal(&conn, &cause, TraceDirection::Downstream, 5, "local", now).unwrap();
        assert!(results.is_empty(), "cross-owner edge must not be traversed by local");
    }

    // ── orphan / dangling ──────────────────────────────────────────

    #[test]
    fn orphan_member_link_events_ok_traversal_skips() {
        let mut conn = open_test_db();
        let real = record_event(&conn, "fact", "real node", "local", None).unwrap();
        let ghost = "b3:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef".to_string();
        // Link: real (cause) → ghost (effect). Ghost does not exist in events.
        let result = link_events(
            &mut conn,
            "causes",
            "local",
            &[
                (real.clone(), "cause".to_string()),
                (ghost.clone(), "effect".to_string()),
            ],
        );
        // link_events must succeed (no FK check).
        assert!(result.is_ok(), "orphan member must not fail link_events");
        // Traversal from real: ghost is not in events, so the JOIN fails
        // to find it and it is silently skipped.
        let now = now_secs();
        let results =
            trace_causal(&conn, &real, TraceDirection::Downstream, 5, "local", now).unwrap();
        assert!(results.is_empty(), "orphan member must be skipped in traversal");
    }

    #[test]
    fn dangling_memory_id_stored_verbatim() {
        let conn = open_test_db();
        // memory_id = 9999 — no row in memories table, but FK is OFF so it stores fine.
        let h = record_event(&conn, "fact", "dangling link", "local", Some(9999)).unwrap();
        let mid: Option<i64> = conn
            .query_row("SELECT memory_id FROM events WHERE node_hash = ?1", [&h], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(mid, Some(9999), "dangling memory_id stored verbatim");
    }

    #[test]
    fn empty_traversal_on_missing_start() {
        let conn = open_test_db();
        let ghost = "b3:0000000000000000000000000000000000000000000000000000000000000000";
        let now = now_secs();
        let results =
            trace_causal(&conn, ghost, TraceDirection::Downstream, 5, "local", now).unwrap();
        assert!(results.is_empty(), "missing start hash → empty result");
    }

    #[test]
    fn empty_traversal_on_foreign_start() {
        let conn = open_test_db();
        let alice_node = record_event(&conn, "fact", "alice node", "alice", None).unwrap();
        let now = now_secs();
        let results =
            trace_causal(&conn, &alice_node, TraceDirection::Downstream, 5, "local", now).unwrap();
        assert!(results.is_empty(), "foreign owner start → empty result for local");
    }

    // ── neighbors == trace depth 1 ─────────────────────────────────

    #[test]
    fn neighbors_eq_trace_depth_1_downstream() {
        let mut conn = open_test_db();
        let (a, _b, _c, _d, _e) = build_nary_fixture(&mut conn);
        let now = now_secs();
        let via_trace =
            trace_causal(&conn, &a, TraceDirection::Downstream, 1, "local", now).unwrap();
        let via_neighbors = neighbors(&conn, &a, TraceDirection::Downstream, "local", now).unwrap();
        let trace_set: std::collections::HashSet<_> =
            via_trace.iter().map(|n| &n.node_hash).collect();
        let neigh_set: std::collections::HashSet<_> =
            via_neighbors.iter().map(|n| &n.node_hash).collect();
        assert_eq!(trace_set, neigh_set, "neighbors must equal trace depth=1 node set");
    }

    #[test]
    fn neighbors_eq_trace_depth_1_upstream() {
        let mut conn = open_test_db();
        let (_a, _b, _c, d, _e) = build_nary_fixture(&mut conn);
        let now = now_secs();
        let via_trace =
            trace_causal(&conn, &d, TraceDirection::Upstream, 1, "local", now).unwrap();
        let via_neighbors = neighbors(&conn, &d, TraceDirection::Upstream, "local", now).unwrap();
        let trace_set: std::collections::HashSet<_> =
            via_trace.iter().map(|n| &n.node_hash).collect();
        let neigh_set: std::collections::HashSet<_> =
            via_neighbors.iter().map(|n| &n.node_hash).collect();
        assert_eq!(trace_set, neigh_set, "upstream neighbors must equal upstream trace depth=1");
    }

    // ── event_exists ───────────────────────────────────────────────

    #[test]
    fn event_exists_present_and_absent() {
        let conn = open_test_db();
        let h = record_event(&conn, "fact", "exists test", "local", None).unwrap();
        assert!(event_exists(&conn, &h, "local").unwrap());
        assert!(!event_exists(&conn, "b3:notexist", "local").unwrap());
    }

    #[test]
    fn max_depth_clamped_to_100() {
        // Verify the clamp doesn't panic and returns a valid result.
        let conn = open_test_db();
        let now = now_secs();
        let results = trace_causal(&conn, "b3:none", TraceDirection::Downstream, u32::MAX, "local", now);
        assert!(results.is_ok(), "max_depth=u32::MAX must not panic");
    }
}
