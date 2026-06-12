# ADR 0031: Persistent turbovec index lifecycle (.tv sidecars, catch-up, stale-rebuild)

## Status

Accepted 2026-06-12

## Context

The TurboVec backend (2-4 bit quantized SIMD search) was lazy-built per
query and never persisted — every semantic recall over a non-trivial
corpus re-quantized the whole store, erasing the point of the
accelerator. turbovec 0.7 already ships `write`/`load` + id maps, so
persistence was wiring. Constraints: turbovec has no delete (temporal
expiry must not force rebuilds), embedder identity can change
(ADR 0030), and index files must never hold the unified-DB lock during
I/O.

## Decision

Per-store sidecars under `<lamu.db parent>/index/` — `<store>.tv` +
`<store>.ids` (parallel rowid array; slots are append-only) +
`<store>.meta.json` `{model, dims, bit_width, last_rowid}` — with a
generic `PersistentIndex<B: QuantBackend>` core so the LIFECYCLE
(validation, catch-up, stale math, atomic persist, throttle) is tested
feature-off against a Brute backend, and only the turbovec binding is
feature-gated. Load-or-rebuild at first use: meta validated against the
current embedder identity AND `vector_index_state`; any mismatch or
corrupt file → discard + full rebuild from SQLite (model-filtered, NO
validity filter — expiry never rebuilds). Catch-up adds rows above
`last_rowid`; the watermark is the table MAX so excluded rows aren't
rescanned. Writers append in-memory + bump bookkeeping under the DB
guard (lock order: DB → index, never reversed); throttled persist
(32 adds or 30s, tmp+rename per file, meta written last as the commit
point) runs AFTER the guard drops. Expiry bumps `stale_count` (only for
currently-valid embedding-bearing rows); searches over-fetch k×4 and
post-filter against SQLite; stale > 25% → synchronous rebuild on next
search (v1). Novelty dedup uses the index only for candidate ids and
re-scores with exact cosine on stored vectors — quantized scores never
gate writes. Default flips: feature compiled + dims%8==0 → persistent
TurboVec; `LAMU_VECTOR_BACKEND=brute` forces the (unchanged,
per-query) Brute path, which needs no persistence — a brute search IS a
full scan.

The long-standing `cargo test --features turbovec` link failure
(`cblas_sgemm` undefined in test binaries) is fixed by a `build.rs`
re-emitting the OpenBLAS link directive when the feature is on.

## Rationale

- Generic-core/feature-gated-binding keeps CI meaningful: the logic that
  can rot (validation, catch-up, throttle, atomicity) runs in every
  default test pass; only the quantizer round-trip needs the feature.
- Exact-cosine re-scoring for novelty: a 2-bit quantized similarity is
  fine for ranking candidates, not for a 0.92-threshold write gate.
- meta-last persist ordering makes a torn write self-healing: stale meta
  → validation mismatch → rebuild from SQLite, the index being a pure
  cache of it.

## Alternatives Considered

- **turbovec IdMapIndex** — internal id mapping, but the three-file
  layout with an explicit parallel id array is exact, append-only, and
  independently inspectable. Rejected.
- **Background rebuild thread v1** — a timer/задача lifecycle for a
  rebuild that takes tens of ms at current corpus sizes. Deferred;
  synchronous-on-next-search documented.
- **Tombstone filtering inside the index** — turbovec has no delete;
  faking it with masks duplicates what the SQLite post-filter already
  guarantees. Rejected.

## Consequences

- Recall stops paying a per-query rebuild; the index survives restarts
  and catches up on rows written while it was cold.
- Known v1 limits (documented in module docs): same-model in-place
  re-embeds are invisible to the index (the shipped `reembed` always
  switches models → identity rebuild); cross-process writes are seen at
  next load; `flush_all()` is not yet wired into shutdown (bounded loss:
  the index is a cache).
- `vector_index_state` is now live-maintained, completing the W2b
  bookkeeping groundwork.

## Related Decisions

ADR 0028 (bookkeeping tables), ADR 0030 (identity that gates loads),
ADR 0002 (turbovec as opt-in lean-build policy).

## Validation

15 lifecycle tests (12 feature-off incl. meta-validation matrix, torn-
persist rejection, throttle, over-fetch + stale-rebuild; 3 feature-on
incl. brute-parity round-trip and dims-change rebuild). Workspace green
BOTH feature states — including `cargo test --features turbovec`, green
for the first time.
