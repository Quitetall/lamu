# ADR 0028: One schema-versioned `lamu.db` with versioned migrations + legacy import

## Status

Accepted 2026-06-12

## Context

Persistence had fragmented into three SQLite files, each with its own
open path, singleton, and ad-hoc evolution: `conversations.db` (turns +
compaction markers), `memory.db` (temporal fact store — already carrying
one hand-rolled ALTER-TABLE migration), `embeddings.db` (RAG chunks).
Three connections to one logical store meant no cross-store transactions,
three places to add a column, and no schema-version ground truth — the
memory.db temporal migration ran as a startup side effect on every open.
The katana-readiness plan adds owner scoping (ADR 0032) and per-store
embedder identity (ADR 0030), which would have meant three more hand
migrations. The user decision (vs an event-log rewrite or formalizing the
split): one database, real migrations.

## Decision

One `~/.local/share/lamu/lamu.db` (`$LAMU_DB` override for tests/
sandboxes), owned by `lamu-memory`: a `Migration {version, name, up}`
framework applies ordered migrations each in its own transaction,
recording rows in `schema_version`. Migration 001 creates the unified
tables — current shapes plus `owner TEXT DEFAULT 'local'` (ADR 0032
groundwork), per-row `embedding_model` (ADR 0030 groundwork), and the
`embedding_stores` / `vector_index_state` bookkeeping tables (ADR 0031
groundwork). Migration 002 adds external-content FTS5 over
`memories(text)` and `turns(content)` with triggers + rebuild backfill.

First open follows the ADR 0025 seeding pattern: `lamu.db` exists →
migrate only (existence is the idempotence marker); else build at
`lamu.db.tmp.<pid>`, migrate, normalize the legacy `memory.db` through
the existing temporal-migration path, `ATTACH` each legacy DB read-only
and bulk `INSERT…SELECT` (ids preserved so `supersedes` links survive;
`embedding_model` recorded only where an embedding exists), then atomic
rename. Legacy files are left untouched (rollback safety); a failed
import never publishes a half-built db. One shared
`OnceLock<Arc<Mutex<Connection>>>` replaces the three singletons;
`open_at(path)` runs the full migration chain so temp-path tests get the
real schema.

## Rationale

- One logical store deserves one transactional boundary: consolidation
  writes turns-derived facts; owner scoping spans every table; a single
  connection makes those atomic instead of best-effort.
- Versioned migrations turn "ALTER on every open" into auditable history;
  the old temporal migration survives only as the legacy-import
  normalizer.
- FTS5 costs no dependency (bundled in rusqlite) and gives keyless recall
  a real keyword leg — groundwork for hybrid recall (ADR 0030) and the
  no-embedder degradation path.
- Forward columns (owner, embedding_model) land in 001 so waves 3-4 need
  data backfill, not schema churn.
- DB-newer-than-binary warns and no-ops rather than failing: an old
  binary must not brick a newer store.

## Alternatives Considered

- **Event-log + CAS (the katana data model)** — architecturally aligned
  with the harness, but a rewrite of working recall/ranking/temporal code
  for no near-term capability gain; katana keeps its own log regardless.
  Rejected by user decision.
- **Keep three DBs, formalize each** — three migration frameworks or one
  shared one applied thrice; cross-store transactions still impossible.
  Rejected.
- **ATTACH the three permanently under one connection** — single
  connection but schema still fragmented across files; migrations would
  need per-file versioning anyway. Rejected.

## Consequences

- `lamu-memory` owns schema evolution; new columns/tables = a new
  `Migration` entry, nothing else.
- Legacy files linger until `lamu clean` learns to prune them (follow-up
  noted); MCP tool description strings still naming old paths are a
  cosmetic follow-up (wire schemas unchanged).
- `embedding_stores`/`vector_index_state` are written by import seeding
  only; live maintenance lands with ADR 0030/0031 (Wave 3).
- Latent pre-existing bug fixed in passing: the legacy open path created
  `idx_memories_valid` before the temporal migration could add
  `valid_until` on a genuinely pre-temporal file — real pre-temporal
  opens would have failed.

## Related Decisions

ADR 0021 (compaction markers ride in turns.metadata, unchanged),
ADR 0025 (the seeding/migration pattern this mirrors), ADR 0029 (the
crate that owns this), ADRs 0030/0031/0032 (consume the groundwork).

## Validation

- 13 new tests: migration ordering / no-op re-run / duplicate +
  out-of-order rejection / future-db tolerance / per-transaction
  isolation; full legacy-import e2e (pre-temporal memory.db fixture,
  owner + embedding_model assertions, idempotent re-run, legacy files
  byte-untouched, no-seed-without-embeddings, failed-import cleanup);
  FTS matches imported AND fresh rows. lamu-memory 63, workspace green.
- Live gate: first `lamu` run after deploy imports the real three-file
  data dir; `recall_memory`/`recall_conversation`/`search_repo` answer
  from lamu.db. (Queued behind the BLUT training run.)
