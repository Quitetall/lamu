# ADR 0029: `lamu-memory` module crate — storage out of the MCP frontend

## Status

Accepted 2026-06-12

## Context

All persistence lived inside lamu-mcp (conversation turns, the temporal
fact store, the RAG chunk index, the Brute/TurboVec vector seam) because
MCP tools were its only consumer. The katana-readiness plan adds a second
frontend consumer: lamu-api serves `/v1/memory/*` (ADR 0032) so an
external harness can use LAMU's memory as an out-of-process extension. In
the ADR 0023 taxonomy lamu-api and lamu-mcp are both FRONTENDS, and a
frontend depending on another frontend inverts the architecture. The
unified `lamu.db` work (ADR 0028) also needs one owner for schema and
migrations rather than three per-module singletons.

## Decision

New module crate `lamu-memory`, depending only on external crates (no
lamu-core, no frontends). It owns: `vector_index` (Brute + feature-gated
TurboVec), `rag` (chunk store + OpenAI embed plumbing), `memory`
(conversation turns + compaction-supersede filtering, with the pure
`render_turns` body renderer), and the `lifetime_memory` storage core
(schema + temporal migration, insert/remember/remember_if_novel, recall +
ranking, supersede/forget, corpus export, `parse_extracted_facts`). The
`turbovec` cargo feature moves here; lamu-mcp forwards it.

lamu-mcp KEEPS: the five MCP tool handlers (wire contracts frozen), the
cloud-judged orchestration (`consolidate`, `extract_from_exchange`,
`reconcile_memory` — they call `handle_cloud_query`), the prompts +
contradiction-id parsing, and the prompt-injection fencing
(`wrap_untrusted` stays at the tool boundary; `render_for_context` is now
a thin fence around `lamu_memory::memory::render_turns`). Thin re-export
shims keep every existing `crate::memory::X` / `crate::rag::X` /
`crate::lifetime_memory::X` call site compiling unchanged.

## Rationale

- Dependency direction: module crates may be consumed by any frontend;
  frontends never consume each other (ADR 0023). Memory is now a shared
  capability, so it must live where both lamu-mcp and lamu-api can reach.
- The untrusted-content fence is a TOOL-BOUNDARY concern (ADR 0011): the
  storage layer returns raw data; whoever ships it into a prompt wraps it.
  Splitting `render_for_context` made that boundary explicit.
- Cloud-judged consolidation stays a frontend concern: it composes the
  cloud tool surface, which is lamu-mcp's, not storage's.
- Re-export shims make the move zero-churn for callers and keep the diff
  reviewable; they can be retired opportunistically.

## Alternatives Considered

- **lamu-api depends on lamu-mcp** — drags the whole MCP server, cloud
  client, and tool registry into the HTTP binary for three SQL calls;
  inverts ADR 0023. Rejected.
- **Duplicate the storage code in lamu-api** — two writers, one schema,
  guaranteed drift. Rejected.
- **Move handlers too** — would push `wrap_untrusted` + cloud calls into
  the storage crate, recreating the coupling one layer down. Rejected.

## Consequences

- `lamu-memory` is the single home for ADR 0028's unified `lamu.db` +
  migrations and ADR 0031's persistent index lifecycle.
- The `turbovec` feature is selected through lamu-mcp's forward; the
  known pre-existing OpenBLAS link gap in `cargo test --features
  turbovec` (cblas_sgemm undefined in test binaries) is unchanged and
  tracked as a TODO.
- Pre-existing environmental caveat made visible while verifying: three
  tests read the REAL data dir's scheduler lock and fail while a training
  run holds it; they pass under an isolated `XDG_DATA_HOME`.

## Related Decisions

ADR 0011 (untrusted envelope), ADR 0023 (module taxonomy), ADR 0028
(unified lamu.db — lands next), ADR 0032 (memory HTTP surface).

## Validation

- Zero-behavior-change gate: full workspace tests green post-move
  (lamu-memory 50, lamu-mcp 99 + 18 dispatch_smoke wire-contract tests
  byte-stable, lamu-api 82+30, lamu-core 205+30).
- `cargo tree -p lamu-mcp` shows no turbovec by default; feature build
  green.
