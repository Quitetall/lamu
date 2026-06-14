# ADR 0042: Background index rebuild + flush-on-shutdown + tv_store v2 (M2)

## Status

Accepted 2026-06-14

## Context

M2 of the memory-scaling program, on M1's WAL pool (ADR 0041). The audit
flagged three index-durability/latency gaps in `lamu-memory/src/tv_store.rs`:

1. **Synchronous stale rebuild** — when `stale_count > 25%` of indexed rows,
   `search_persistent` rebuilt the whole index INLINE while holding the slot
   mutex (an O(N) SQLite scan + re-quantize), blocking every concurrent
   searcher/writer until it finished.
2. **`flush_all` not wired to shutdown** — the in-memory index could lose up
   to a throttle-window of incremental adds on exit (rows stayed durable in
   SQLite; only the index cache was lossy).
3. **tv_store v1 stale accounting** (task #192) — a single bulk `stale_count`
   integer, bumped in a separate statement after the expiry UPDATE (losable on
   a crash), with no way to suppress a known-dead slot before the 25% rebuild.

M1's pool is what makes the fix safe: a background thread can take its OWN
pooled connection, so a rebuild off the request path no longer contends a
single shared connection.

## Decision

**Background async rebuild.** On `stale_rebuild_due`, `maybe_spawn_bg_rebuild`
CAS-claims a per-store `REBUILD_IN_FLIGHT` flag and spawns ONE `std::thread`
that: (1) takes a fresh pool conn, (2) snapshots the watermark, (3) builds the
new `PersistentIndex` from SQLite **off the slot lock** (the O(N) work), (4)
persists it while it still **owns** the index (so a concurrent identity swap
can't make it persist a different one), (5) takes the slot lock only for the
fast swap **and clears the skip-set in the same critical section**, (6)
compare-and-decrements `stale_count` by the start-of-build snapshot, then
records the build's watermark/model/dims/built_at WITHOUT re-zeroing stale. A
RAII drop-guard clears the in-flight flag even on panic/early-return. The
triggering search SERVES THE CURRENT (stale) index immediately — it never
blocks. Rows added during the build (rowid > watermark) land in the old index
via `note_added` and are caught up by `ensure_loaded_impl` after the swap
(bounded + self-healing).

`record_built` gains a `reset_stale: bool`: a SYNCHRONOUS full rebuild zeroes
`stale_count`; a BACKGROUND rebuild passes `false` so it leaves the
already-subtracted count intact (and keeps `vector_index_state` consistent with
the new on-disk meta, so the next restart sees no `state-behind-meta` mismatch
and skips a spurious full rebuild).

**flush-on-shutdown.** `tv_store::flush_all()` (persists every dirty index) now
runs after `axum::serve` returns (lamu-api graceful shutdown) and after the
`lamu start` MCP stdio loop exits.

**tv_store v2 (task #192).** A per-store in-memory **skip-set** of stale
rowids: expired rowids are filtered from ANN hits immediately (not at the next
25% threshold), cleared on rebuild swap — purely an over-fetch optimization;
the SQLite validity post-filter remains the correctness gate. The stale signal
is now **tx-coupled**: `bump_stale` runs INSIDE the same transaction that
expires the row in `forget`/`supersede_conn`, so it can't be lost if the
process dies between the UPDATE commit and the bump.

**Memory struct → pool** (M1-deferred cleanup). `Memory` now holds a `Source`
enum {`Pool` | `Explicit(conn)`}: `shared()` takes a pooled connection per call
(`with_conn`/`with_conn_mut`), `open(path)` keeps an explicit connection for
tests. The `#[deprecated] store::shared_handle` shim has ZERO internal callers.

## Rationale

- Building off the slot lock and swapping under it turns an O(N) all-readers
  stall into an O(1) lock hold; the rebuild's cost moves entirely off the
  request path.
- Compare-and-decrement (not zero) is required because the rebuild runs
  concurrently with writers: an expiry that bumps `stale_count` during the
  build must survive, or that staleness is silently lost.
- Persisting before the swap (while the thread owns the index) removes the
  re-lock-and-persist-whatever's-there hazard.
- tx-coupling the stale bump closes the crash window the v1 separate-statement
  bump had.

## Alternatives Considered

- **tokio task instead of std::thread** — the rebuild is pure SQLite + CPU and
  must run from sync storage call sites that may not be in a runtime; a
  detached std thread + pooled conn is simpler and runtime-agnostic.
- **Validity-filtered rebuild (purge expired rows)** — would let the index
  shed dead rows, but ADR 0031 keeps expired rows so `include_expired` vector
  recall works; the skip-set is the chosen mitigation instead. Unchanged here.
- **Persist under one lock across swap** — the persist is bounded I/O but still
  longer than the swap; persisting before the swap keeps the lock hold minimal
  and the index uniquely owned.

## Consequences

- Stale rebuilds no longer stall concurrent recall; the index swap is atomic
  and the bookkeeping is restart-consistent.
- The skip-set is best-effort and in-memory (not persisted) — it resets on
  restart and is cleared on rebuild; correctness always rests on the SQLite
  post-filter. A row expired DURING a build may briefly reappear in over-fetch
  until the next rebuild (post-filtered out meanwhile).
- `forget_conn` is now test-only; production `forget`/`supersede` carry the
  tx-coupled bump.
- Known follow-up: the memories index still retains expired rows (ADR 0031) —
  the rebuild reduces the stale *signal* but not the dead-slot count; the
  skip-set is the live mitigation.

## Related Decisions

ADR 0041 (the pool that lets a bg thread take its own conn), ADR 0031
(persistent index lifecycle + the keep-expired-rows rule this works within),
ADR 0028 (vector_index_state bookkeeping), ADR 0032 (the memory-as-a-service
consumers that benefit from non-stalling recall).

## Validation

bg-rebuild stale-reset, skip-set filter, flush idempotency,
`subtract_stale_compare_and_decrement` (partial / floor-at-0 / concurrent-bump
survival), Memory pool/explicit dispatch. lamu-memory 142 + lamu-mcp 118 +
lamu-api 117 green. MiMo review: NEEDS CHANGES (record_built re-zeroed the
compare-and-decrement) → fixed (reset_stale) → PASS. Live gate (GPU free): a
heavy supersede/forget churn workload showing recall stays responsive while a
rebuild runs in the background.
