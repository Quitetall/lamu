# ADR 0002: Keep LAMU a lean Rust workspace, not a batteries-included framework

## Status
Accepted 2026-05-31

## Context
LAMU is a port of an earlier Python stack. That stack carried a heavy
dependency footprint: a Chainlit web frontend (`/home/brianklam/local-llm/web/app.py`),
a Flask-style serving layer, and assorted ML glue. A separate scratch clone,
Odysseus (gitignored at `/home/brianklam/local-llm/odysseus/`, "mined for ideas,
not vendored" per `.gitignore`), pushes the batteries-included shape further:
Python/Flask + opencode + llmfit + ChromaDB + a JS frontend + a bundled model.
The thing LAMU actually does is narrow — manage GGUF subprocess backends,
schedule VRAM, route requests, speak MCP and an OpenAI-compatible HTTP shim,
and call cloud providers. None of that needs a UI framework, an ORM, a vector
database, or a plugin runtime. The port was scoped as mechanical
(`PORT_PLAN.md:3` "Port mechanically from Python — no redesign"), and each
dependency in a long-lived daemon is a maintenance + supply-chain liability:
the current lockfile already pulls 462 crates transitively, almost entirely
from the unavoidable core (tokio, reqwest, axum, rusqlite, tree-sitter). Every
direct dependency added on top compounds that. A later 90-agent polish audit
re-examined the dependency set and proposed library swaps; the proposals were
treated as bloat and rejected, leaving the original lean set intact.

## Decision
LAMU is a five-crate Rust workspace (`lamu-core`, `lamu-api`, `lamu-cli`,
`lamu-mcp`, `lamu-providers`; `Cargo.toml:3-9`) with a deliberately minimal
direct-dependency set and no batteries-included framework. The shared
foundation is tokio, serde/serde_json/serde_yaml, thiserror/anyhow, tracing,
and walkdir (`Cargo.toml:23-41`). HTTP serving uses axum and only in
`lamu-api` (`lamu-api/Cargo.toml:18`); the rest is reqwest for outbound calls,
nvml-wrapper for GPU queries (`lamu-core` only, `lamu-core/Cargo.toml:22`),
parking_lot for locks, nix/libc for signals, which/dirs for path resolution.
There is no web framework beyond axum, no ORM, no plugin system, and no
embedded vector DB. The MCP transport is hand-rolled JSON-RPC 2.0 over stdio
(`lamu-mcp/src/server.rs:1-4`) rather than the `rmcp` crate that
`PORT_PLAN.md:43` floated. Persistence is plain `rusqlite` with the bundled
SQLite (`lamu-mcp/Cargo.toml:27`), not a query builder. The optional compressed
vector backend (`turbovec`) is gated behind a cargo feature that is off by
default (`lamu-mcp/Cargo.toml:39-45`), so the default build never pulls it.
Where a dependency would buy only convenience, a small hand-rolled helper is
written instead.

## Rationale
- The problem domain is narrow. A subprocess+VRAM+routing daemon does not need
  the surface area a general AI application framework provides; the Python
  predecessor's Chainlit frontend (`web/app.py`) has no analogue in the Rust
  port because the chosen interface is MCP (see ADR 0001), not a browser UI.
- Each direct dependency is a recurring cost: version churn, CVE triage, and
  build-time. With 462 transitive crates already in the lockfile from the
  irreducible core, every avoidable top-level dep makes the tree worse, not
  marginally so.
- Hand-rolling JSON-RPC over stdio is ~cheap and removes a dependency on
  `rmcp`'s protocol-version assumptions and release cadence; the protocol is
  line-delimited JSON-RPC 2.0 (`lamu-mcp/src/server.rs:1-4`), which serde_json
  already handles. The PORT_PLAN's tentative "rmcp recommended" was not adopted
  — `rmcp` is absent from the lockfile.
- An ORM would add a code-generation/macro dependency and a migration runtime
  for what is a handful of tables; `rusqlite` with bundled SQLite covers the
  memory/RAG persistence directly (`lamu-mcp/Cargo.toml:27`).
- Feature-gating the heavyweight, system-linking pieces keeps the default build
  clean: `turbovec` links a system OpenBLAS and is opt-in
  (`lamu-mcp/Cargo.toml:37-45`); media backends are similarly gated. The common
  path stays lean.
- The 90-agent polish audit's rejection of library swaps confirmed the set is
  at a local minimum — the proposed replacements added surface without removing
  a real pain point.

## Alternatives Considered
- **Python web stack (the predecessor / Odysseus shape)** — Flask/Chainlit +
  ChromaDB + a JS frontend + bundled tooling. Rejected: it is exactly what the
  port moved away from. It couples the daemon to a browser UI and a Python
  runtime, and the vector-DB + frontend dependencies dwarf the actual serving
  logic. LAMU's interface is MCP (ADR 0001), so a web UI is dead weight.
- **`rmcp` for the MCP transport** — use the upstream MCP crate instead of
  hand-rolling. Floated in `PORT_PLAN.md:43`. Rejected: the protocol surface
  LAMU uses (initialize, ping, tools/list, tools/call over line-delimited
  JSON-RPC) is small enough to implement directly in `server.rs`, and doing so
  removes a dependency whose protocol-version and API churn LAMU would have to
  track. `rmcp` is not in the lockfile.
- **An ORM / query builder (diesel, sea-orm, sqlx)** — typed schema + migration
  runtime over SQLite. Rejected: none appear in the lockfile; the persistence
  need is a few tables, served by `rusqlite` with bundled SQLite
  (`lamu-mcp/Cargo.toml:27`). An ORM adds proc-macro build cost and a migration
  framework for no proportional benefit.
- **A plugin system (libloading/extism dynamic loading of caller code)** —
  Rejected: tools are compiled in (`lamu-mcp/src/tools.rs`); no LAMU crate
  depends on `libloading` directly (it appears only transitively, via
  `nvml-wrapper` dynamically loading `libnvidia-ml`). A dlopen plugin ABI would
  add a stability contract and a sandbox-escape surface for a single-operator
  daemon that does not need third-party plugins.
- **Heavier framework deps generally** (a larger metrics/telemetry stack, a
  fuller HTTP framework) — Rejected: `prometheus` is pulled with
  `default-features = false` (`lamu-api/Cargo.toml:23`) and axum is the only web
  framework, scoped to the one crate that serves HTTP. The 90-agent audit's
  swap suggestions in this category were rejected as bloat.

## Consequences
- Commits us to maintaining hand-rolled code where a library was declined: the
  JSON-RPC dispatch loop in `lamu-mcp/src/server.rs` is now load-bearing and
  must track MCP spec changes ourselves. If the MCP spec grows materially, this
  becomes a real maintenance item and the `rmcp` decision should be revisited.
- The HTTP surface is intentionally thin (axum in `lamu-api` only). Adding
  rich web features later would mean either expanding that crate or
  reintroducing a frontend — a decision ADR 0008 (headless council, no compare
  UI) and ADR 0001 (MCP-first) already lean against.
- The default build stays small and fast (release profile uses thin LTO,
  single codegen unit, strip; `Cargo.toml:47-50`), but feature-gated paths
  (`turbovec`, media) carry their own system-dependency burden (OpenBLAS,
  gfortran) that users must satisfy out-of-band when they opt in.
- The TUI deps (`ratatui`, `crossterm`, `rustyline`, `termimad`;
  `lamu-cli/Cargo.toml:29-34`) live entirely in `lamu-cli` and are terminal-UI,
  not web — they do not leak into `lamu-core`/`lamu-api`/`lamu-mcp`, preserving
  the leanness of the serving crates. This is a constraint future CLI work must
  honor: UI deps stay confined to `lamu-cli`.
- We forgo the velocity a batteries-included framework gives. Net-new
  capabilities (a vector DB, a richer persistence layer, a plugin API) cost
  more up front because they are not free imports. The bet is that a daemon's
  lifetime maintenance cost dominates its initial feature-add cost.

## Related Decisions
ADR 0001 (MCP-first orchestration; HTTP serve as a thin compat shim) — the
reason no UI framework is needed. ADR 0008 (headless multi-model council
instead of a compare UI) — a direct application of the no-UI stance. ADR 0007
(unified cloud routing) — implemented with the lean reqwest client rather than
per-provider SDKs.

## Validation
- Right if: `cargo tree -p lamu-mcp` on the default build continues to list no
  `turbovec` and no `rmcp`; the direct-dependency count per crate stays in the
  same order of magnitude; CVE/version-bump churn remains tractable for one
  maintainer.
- Wrong if: the hand-rolled JSON-RPC layer starts failing against real MCP
  clients due to spec drift, or accumulates enough special-casing that
  adopting `rmcp` would be a net reduction in code — that triggers a superseding
  ADR. Likewise if the no-UI/no-DB stance forces repeated, awkward
  reimplementation of something a library would give cleanly (e.g. a real
  vector store), revisit the relevant gate.
- Signal to watch: feature-gated deps (`turbovec`, media/OpenBLAS) generating
  user-reported build breakage frequently enough that "opt-in" is effectively
  "required" — that would mean the lean default no longer matches real usage.

