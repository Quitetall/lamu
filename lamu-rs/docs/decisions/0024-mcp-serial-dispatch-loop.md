# ADR 0024: MCP dispatch loop stays serial; concurrency lives in tools and the per-model queue

## Status

Accepted 2026-06-11

## Context

The MCP stdio JSON-RPC loop (`lamu-mcp/src/server.rs:145-204`) processes one
request at a time: read a line, `self.handle(...).await` inline, write the
response, loop. A 30-second `query` therefore blocks every subsequent request
on the same pipe until it completes. The 2026-06 audit flagged this as a
decision to make explicitly: embrace serial dispatch, or spawn a task per
`tools/call` and buffer responses.

Three existing mechanisms already shape the answer:

- **Per-model `RequestQueue`** (`lamu-core/src/queue.rs`) serializes backend
  access at concurrency = 1 per model (`LAMU_QUEUE_CONCURRENCY`, default 1).
  Even with a concurrent dispatch loop, at most one request per model reaches
  the backend; the rest queue.
- **Concurrency already exists inside tools**: `parallel_query`
  (`handlers.rs:872-1014`) and `council` (`council.rs:67-89`) fan out with
  `join_all` within a single tool call, throttled by per-model semaphores.
  A caller who needs concurrent generation has a supported path today.
- **Dispatch-boundary policies** (ADR 0001): the routing-mode gate, the
  GPU-training-lock check, and error journaling are enforced at the loop
  boundary where `&LamuMcpServer` is in scope, guarded by simple `Mutex`
  state with no concurrent-dispatch hazards.

## Decision

The MCP dispatch loop remains serial by design. One request is read, fully
handled, and answered before the next is read. Concurrency is provided at
exactly two layers: *within* a tool call (`parallel_query`, `council`) and
*below* the tools at the per-model `RequestQueue`. We will not spawn
per-request tasks at the dispatch boundary.

## Rationale

- The per-model queue already serializes the expensive resource (the
  backend / GPU). Loop-level concurrency would add a second buffering layer
  *above* a queue that admits one request per model anyway — complexity with
  no added backend throughput on a single-GPU host (ADR 0014/0017).
- Each Claude Code instance spawns its own MCP subprocess over stdio
  (documented multi-agent model). Cross-client concurrency is achieved by
  process isolation, not by concurrent dispatch within one process.
- Dispatch-layer policies (routing gate, lock check, registry mutation in
  `scan_models`) currently rely on the loop being the single mutator between
  reads. Spawning would force every policy to become re-entrant and would
  reopen the registry read-modify-write race that audit fix A2 just closed.
- The Multi-user priority work (ADR 0018) and the inflight gauge (scale P0)
  both model queue depth per model. A second dispatch-side buffer would split
  that signal across two queues and complicate fairness accounting.
- An unbounded spawn loop converts a slow client into unbounded memory growth
  (buffered pending requests). Serial dispatch is implicitly back-pressured
  by the pipe.

## Alternatives Considered

- **Spawn per `tools/call`, serialize responses via a writer task** — gives
  pipelining for cheap metadata calls (`list_models` behind a long `query`).
  Rejected: the dominant calls are model-bound and serialize at the per-model
  queue anyway; the win is limited to metadata calls, while every dispatch
  policy must become concurrency-safe and responses need an ordering/buffer
  layer. Cost exceeds the benefit at current scale (single user, single GPU).
- **One MCP instance per model** — sidesteps shared state entirely.
  Rejected: multiplies llama-server supervision, breaks the single scheduler
  view of VRAM, and the client (Claude Code) expects one server per session.

## Consequences

- A long-running tool call delays subsequent calls on the same session,
  including cheap ones. Callers needing parallelism must use
  `parallel_query`/`council`, or run additional Claude Code sessions (each
  gets its own MCP process).
- Dispatch-boundary code may continue to assume single-flight semantics
  (no re-entrancy) — this is now a documented invariant, not an accident.
- If a future workload demands pipelined metadata calls, the revisit path is
  a bounded spawn for *read-only* methods only (`tools/list`, `list_models`),
  keeping mutating calls serial. That would need its own ADR.

## Related Decisions

ADR 0001 (MCP-first orchestration), ADR 0018 (multi-user), ADR 0020 (HTTP has
no request queue; single-flight is load-only), ADR 0023 (module seam).

## Validation

- `queue.rs` unit tests pin per-model fairness and concurrency limits.
- Future work: an integration test driving two overlapping `tools/call`
  requests through the stdio loop, asserting strict response ordering — a
  regression tripwire against accidental loop-level spawning.
- Revisit trigger: a real workload where metadata-call latency behind long
  queries measurably hurts (multi-second p95 on `list_models`).
