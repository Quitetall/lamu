# ADR 0039: Causal event hypergraph (graph-lite world-model v1)

## Status

Accepted 2026-06-13

## Context

An expert review proposed a three-geometry stratified world model for LAMU:
(1) an event **hypergraph** for structural/relational facts, (2) **turbovec**
Euclidean embeddings for semantic-content similarity (already shipped — ADR
0031), (3) **Poincaré** hyperbolic embeddings of the hypergraph structure for
episodic/structural similarity. The review's own build order was explicit:
**causal event graph FIRST, hyperbolic later** — "turbovec is already handling
the semantic layer; the causal graph is the missing piece right now, not the
geometry."

Verified against the code: LAMU's metadata-filtered vector recall
(`search_persistent` over-fetch + SQLite post-filter, ADR 0031/0032) can do
ATTRIBUTE SELECTION — owner, embedding_model, valid-time — but CANNOT do
relational traversal. A `WHERE` predicate is per-row; "what did B cause" /
"ancestors of E" are recursive edge-walks. So the causal graph is genuinely
additive, not approximable by filtering turbovec.

Katana boundary (from its spec): katana owns its own append-only event log +
CAS (`~/.katana/objects/`, BLAKE3 `b3:<hex>`) and lists "a memory product (no
built-in vector store)" as a NON-goal — durable memory/graph is a **LAMU concern
consumed via MCP tools** ("LAMU is the armory, Katana is the sword"). A graph
built inside LAMU is invisible to katana except as `Tool(name)` capabilities.

## Decision

A graph-lite causal hypergraph in the EXISTING `lamu.db` (no separate graph
store), in a new `lamu-memory/src/causal_graph.rs` module, surfaced as MCP tools.

**Schema (migration m003), reusing the `memories` temporal + owner conventions:**
- `events` — nodes, PK `node_hash = 'b3:<hex>'` (BLAKE3 content address). `owner`,
  `valid_from`/`valid_until`, `kind`, `text`, `ts`, and a NULLABLE `memory_id` FK
  to `memories.id` — the structural↔semantic bridge. No FK enforcement (PRAGMA
  foreign_keys stays OFF workspace-wide): orphan members and dangling `memory_id`
  are valid states, skipped at traversal time by the `JOIN events`.
- `hyperedges` — n-ary directional relations (`relation`, `owner`, valid-time).
- `hyperedge_members` — junction `(hyperedge_id, node_hash, role)`; `role` is in
  the PK so self-loops are representable. Partial indexes on `role = 'cause'` and
  `role = 'effect'`.

**Storage (sync rusqlite, `&Connection` — the caller holds the lock):**
- `content_hash(kind, text)` = BLAKE3 of `"{kind_trimmed}\x00{ws-collapsed text}"`
  → `"b3:<hex>"`. Same content → same node; `record_event` is `INSERT OR IGNORE`
  idempotent. `link_events` writes the edge + all members in one atomic tx.
- `trace_causal` — recursive CTE, **two pre-compiled SQL strings** selected by
  direction (downstream seed `cause`/harvest `effect`; upstream the reverse) with
  LITERAL roles so the partial indexes fire. **Cycle guard** = a pipe-delimited
  visited-path string with `INSTR(path, '|'||hash||'|') = 0` (b3-hex never
  contains `|`) — terminates on cycles and self-loops regardless of depth.
  Owner + valid-time filter BOTH the edge and harvest-node joins. `max_depth`
  clamped to 100. `UNION ALL` can revisit a node via multiple paths, so the Rust
  layer dedups by minimum depth. `neighbors` is the flat depth-1 form.

**Interface = BLAKE3 `b3:` CAS**, byte-identical to katana's CAS convention: a
katana event and a LAMU graph node can share a hash, so katana can reference a
LAMU node and ask LAMU to trace causality without a translation layer.

**Surface: MCP tools only** in v1 — `record_event`, `link_events`, `trace_causal`
(table-driven `TOOLS` entries; `handle_*` acquires `shared_handle()`, locks, calls
the sync storage fn, drops the guard with no `.await` held). `owner = LOCAL_OWNER`.

## Rationale

- Graph-lite in `lamu.db` avoids a second datastore; recursive CTEs give 1..N-hop
  traversal that metadata filtering structurally cannot.
- Content addressing dedups nodes and gives the cross-system key the expert's
  stratification (and katana interop) rests on.
- Reusing the `memories` valid-time + owner model means temporal expiry,
  supersession-style forgetting, and per-owner isolation come for free and behave
  identically to facts.
- MCP-only matches the katana boundary (graph is a consumed capability, not a
  contract surface) and defers HTTP until a consumer holds a bearer for it.

## Alternatives Considered

- **Separate graph DB (Neo4j/etc.)** — operational weight for 1–2 hop traversal
  over a store that already holds the facts. Rejected for v1.
- **Approximate the graph via turbovec metadata filtering** — the review's own
  question; filtering is attribute selection, not transitive traversal. Can't
  express multi-hop. Rejected (and the reason recorded).
- **Events as rows in `memories` (kind='event')** — would reuse the embedding
  path but conflate semantic facts with structural nodes and pollute recall.
  Dedicated tables keep the strata clean. Rejected.
- **Embeddings on events now (turbovec `Store::Events`)** — the geometry the
  review said to defer; needs a 3rd `Store` variant + 2 SQL arms + a static slot.
  Deferred.
- **App-side BFS instead of a recursive CTE** — the correct escape hatch if the
  graph grows dense (the `INSTR` guard's intermediate sets balloon on wide nodes).
  Documented as the v2 path; CTE is fine at v1 corpus sizes.

## Consequences

- LAMU gains a queryable causal/relational layer; katana (or Claude Code) records
  events, links them, and traces cause→effect / effect→cause chains over MCP.
- The structural↔semantic join exists (a node's `memory_id` → a turbovec-indexed
  fact), even though recall ACROSS the join is not yet wired.
- **Known v1 limits:** `UNION ALL` + Rust min-depth dedup (SQLite has no recursive
  `UNION DISTINCT`); the `INSTR` path guard is O(depth) string growth and balloons
  on very wide nodes (app-side BFS is the documented v2 fix); `trace_causal` holds
  the store mutex for the whole walk (correct for sync SQLite; a contention ceiling
  under concurrent use); `role` accepts arbitrary strings (non-cause/effect roles
  are inert to trace, visible to neighbors — deliberate extensibility); no input
  length caps / `max_depth=0` returns empty.
- **Explicit deferrals:** HTTP `/v1/graph/*` (ADR 0032 mirror), turbovec
  `Store::Events` for event semantic search, Poincaré/hyperbolic structural
  embeddings (geometry layer 3), graph export for graphify, blut training-flywheel
  export of the graph.

## Related Decisions

ADR 0031 (turbovec — the semantic stratum this complements), ADR 0032 (owner
scoping + the HTTP-memory pattern a future `/v1/graph` mirrors), ADR 0028 (the
unified `lamu.db` + migration framework m003 extends), ADR 0029 (lamu-memory owns
storage; frontends consume), ADR 0024 (serial dispatch — the sync storage model).

## Validation

23 unit tests (in-memory SQLite + `migrate()`, synchronous, no model load):
content-hash dedupe (whitespace/kind normalization), `record_event` idempotency
(COUNT=1), n-ary traversal both directions, multi-hop depth labelling + depth cap,
**cycle + self-loop termination on max_depth=10**, expired node AND expired edge
hidden, `include_expired` via now=0, owner isolation + cross-owner edge not
traversed, orphan member skipped, dangling `memory_id` stored verbatim, empty
traversal on missing/foreign start, `neighbors == trace(depth 1)`, `max_depth`
clamp. lamu-memory 134 green; lamu-mcp 118 green. MiMo review: PASS WITH NITS
(byte-vs-char ellipsis fixed; the rest by-design or v2). Live gate: drive the
tools from a real MCP client / katana session (queued).
