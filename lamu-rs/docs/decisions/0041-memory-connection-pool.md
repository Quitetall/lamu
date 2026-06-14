# ADR 0041: Memory storage on a WAL connection pool (M1)

## Status

Accepted 2026-06-14

## Context

An architecture audit of the memory layer found the load-bearing scaling
ceiling: `lamu-memory/src/store.rs` held **one** process-global
`OnceLock<Arc<parking_lot::Mutex<Connection>>>` — a single SQLite connection
behind a single mutex. **Every** read and write (recall, FTS, vector search,
novelty scan, remember/supersede/forget, causal-graph ops) serialized through
it. WAL mode (`journal_mode=WAL`) gave **zero** in-process benefit because WAL's
concurrent-reader property needs *separate connections*, and there was exactly
one. For memory-as-a-service (ADR 0032) backing concurrent agents — where recall
is the hot path — the store was effectively single-threaded at the SQLite
boundary. This is M1 of a five-wave memory-scaling program (M1 pool → M2
rebuild/durability → M3 multi-tenant → M4 graph features → M5 geometry).

## Decision

Replace the singleton with an **r2d2 WAL connection pool** (`r2d2` +
`r2d2_sqlite` 0.25 — pinned to re-export the workspace's rusqlite 0.32; 0.24→0.31,
0.26→0.33 would not unify the `Connection` type).

- `store.rs`: `POOL: OnceLock<Pool>` built once under the retained `INIT_LOCK`
  after the existing one-time legacy-import + migrate (a bootstrap connection from
  `open_or_import`, then dropped). The pool's `with_init` applies
  `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000` to **every**
  connection; `max_size=8`. New API: `conn() -> Result<PooledConn>` (derefs to
  `&Connection` / `&mut Connection`, so transaction call sites are unchanged).
  **Reads now run concurrently across pooled connections; writes serialize at
  SQLite's WAL writer-lock** (correct + fast), not at an in-process mutex.
- 13 `shared_handle().lock()` call sites migrated to `conn()` (lamu-memory
  `lifetime_memory` ×6 / `rag` ×3, lamu-mcp `lifetime_memory` ×3, lamu-cli
  `memory_admin` ×1). The `insert → record_store_identity → note_added` ordering
  is preserved on one connection per write.
- `open_at` (the non-pool open used by reembed, the shim, and tests) also sets
  `busy_timeout` — so all three non-pool open paths wait on the writer-lock
  instead of erroring `SQLITE_BUSY`, matching the pool members.
- Migration **m004**: partial index `idx_memories_model_valid ON
  memories(embedding_model, valid_until) WHERE embedding IS NOT NULL` — the brute
  scan + index-rebuild SELECT (`WHERE embedding IS NOT NULL AND embedding_model=?`)
  had no covering index.

**Novelty TOCTOU (review-driven, load-bearing).** Removing the global mutex
re-opened a real regression: `remember_if_novel` used to be atomic because every
caller shared one connection-mutex across SELECT-novelty → INSERT. With the pool,
two concurrent callers read their **own** WAL snapshots, both pass the novelty
gate, and both insert a near-duplicate. A dedicated `NOVELTY_WRITE_LOCK` now
serializes **only** the novelty-gated check+insert critical section — reads,
recall, and plain `remember()` stay fully parallel (the whole point of the pool).
Invariant documented at the lock: take the lock first, then the pool conn *inside*
the section (snapshot can't go stale), and no pool-conn holder may block on it
(deadlock-free). M3's content-hash `UNIQUE` guard will replace this with
finer-grained concurrency.

## Rationale

- A WAL pool is the standard SQLite scaling answer: it removes the in-process
  serialization for reads (the hot path) while SQLite's own writer-lock keeps
  writes correct. No application-level write coordination is needed.
- `conn()` deref-ing to `&mut Connection` made the call-site migration mechanical
  (transactions, `&conn` SELECTs unchanged).
- Serializing only the novelty path preserves dedup correctness without
  reintroducing the read bottleneck the pool exists to remove.

## Alternatives Considered

- **Single writer conn + a read-conn pool** — more bespoke than r2d2 for no gain;
  SQLite already serializes writers at the WAL lock.
- **`RwLock<Connection>`** — a rusqlite `Connection` isn't usable from multiple
  threads concurrently regardless of an outer RwLock; true parallel reads need
  distinct connections. Rejected.
- **Content-hash UNIQUE dedup now (instead of the novelty lock)** — the correct
  long-term fix, but it's a schema migration + write-path change scoped to M3; the
  lock closes the regression in M1 with zero schema churn.

## Consequences

- Concurrent recall no longer serializes; throughput scales with the pool size
  and SQLite's WAL reader concurrency.
- **Known M2/M3 cleanup (documented, not regressions — all WAL-safe via
  busy_timeout):** the `Memory` struct (conversations/turns) still holds one
  non-pool `Arc<Mutex<Connection>>` via the `#[deprecated]` `shared_handle` shim
  (a `OnceLock` singleton → opened once) — M2 migrates the struct;
  `memory_admin` reembed opens a non-pool connection (one-off admin op);
  `rag::index_repo` spans two pool connections across its async embed (the old
  code also released the mutex across the await, so not a new regression).
- The concurrency unit tests build a local pool (the global `POOL` `OnceLock`
  can't be reset between tests); the production `pool()→conn()` path is exercised
  by every shared-handle-path test running against a temp `$LAMU_DB`.

## Related Decisions

ADR 0028 (the unified `lamu.db` + migration framework m004 extends), ADR 0029
(lamu-memory owns storage), ADR 0030/0031 (the embedder + turbovec index the pool
now feeds concurrently — the process-global index mutex is unchanged), ADR 0032
(memory-as-a-service, whose concurrent consumers this unblocks).

## Validation

`two_connections_from_pool_do_not_serialize_reads` (the headline proof: a second
`conn()` returns while the first is held, both SELECT), `parallel_threads_all_succeed`
(8 threads), `pool_init_runs_migrate_exactly_once_on_fresh_db`,
`m004_index_used_for_embedding_model_query` (EXPLAIN QUERY PLAN). lamu-memory 139 +
lamu-mcp 118 green. MiMo review: NEEDS CHANGES (the novelty TOCTOU) → fixed →
PASS WITH NITS. Live gate (GPU free): real concurrent HTTP memory-API + MCP recall
load showing parallel reads.
