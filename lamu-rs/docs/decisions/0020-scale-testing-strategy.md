# ADR 0020: Scale-testing strategy — ignore-gated harness tiers + HTTP has no request queue

## Status

Accepted 2026-06-01

## Context

The multi-GPU (ADR 0017) and multi-user (ADR 0018) tracks both raise the same
question the unit suite cannot answer: *does the system hold up under concurrent
load, and where is the bottleneck when it doesn't?* `cargo test --workspace`
runs deterministic, in-process, no-GPU unit + fixture tests; it deliberately
never spawns a real `lamu serve`, never loads a GGUF, and never drives sustained
concurrency. That is correct for the default suite (it must stay fast and
hermetic on a laptop / GitHub-hosted runner) but it leaves scale uncovered.

Three facts about the current system shape this decision:

- **HTTP has no request queue; single-flight is load-only.** The HTTP inference
  surface (lamu-api) does **not** throttle or admission-queue concurrent
  requests. Concurrent requests for the *same* model coalesce on **one** backend
  spawn via the per-name single-flight load gate — a process-global
  `tokio::sync::Mutex` taken at `lamu-core/src/loader.rs:148`
  (`spawn_gate().lock().await`), with a double-checked `get_loaded` before and
  after the gate (`loader.rs:143` and `:149`) so the second-and-later callers
  wait for the loader and then reuse the loaded backend rather than
  double-spawning. **That gate serializes loading, not inference.** Once the
  model is `ModelState::Loaded`, every request proxies straight through to the
  llama-server subprocess (`openai_compat.rs` `resolve_and_ensure_loaded:354`)
  with no LAMU-side serialization. The only place HTTP can gain an admission
  queue is the *optional, flag-gated* per-model `Strategy::Priority` wrap
  described in ADR 0018 §4 — which is a fairness/quota feature, not the
  same-model coalescing gate, and is off by default. So: **same-model concurrency
  coalesces on a single load; it is otherwise unthrottled at the HTTP layer.**
  Any load test must be read against this invariant — a "queue" tail at the HTTP
  layer would be a *regression*, not expected behavior.

- **The interesting contention is the shared `parking_lot::Mutex`** around the
  scheduler / router / health registry (`AppState` in `openai_compat.rs`), held
  briefly on the read path (`/health`, `/v1/models`, `/metrics`) and on every
  `ensure_loaded` bookkeeping step. That lock — not a request queue — is what a
  scale test stresses, and it is exactly what the multi-user track (ADR 0018)
  will hammer with per-principal quota lookups.

- **Real load needs a GPU and a real model.** Measuring tokens/s, prefill, and
  decode tail latency requires a loaded 27B on the 4090. CI runners have no GPU,
  so any test that needs one must skip cleanly (return Ok) rather than hang or
  fail — the contract `spec_e2e.rs` and `frontend_matrix.rs` already implement
  (skip when `lamu` is off PATH, the registry is missing, or no model fits VRAM).

## Decision

Adopt a **four-tier, ignore-gated scale-test strategy**, none of which run in the
default `cargo test --workspace` suite, plus a **deferred live HTTP load baseline**
driven by an external tool (`oha`). The tiers, in increasing cost/fidelity:

1. **`spec_e2e`** (`lamu-api/tests/spec_e2e.rs`, `#[ignore]`) — spawns the real
   `lamu` binary on an ephemeral port and asserts the spawn / status / pidfile +
   three-surface (OpenAI / Anthropic / Ollama) contract end-to-end. The
   integration backstop: catches wire-up regressions a passing unit suite would
   miss. Skips when `lamu` is off PATH / registry missing / no model fits VRAM.

2. **`frontend_matrix`** (`lamu-api/tests/frontend_matrix.rs`, `#[ignore]`) — for
   a running serve, exercises the exact call shape each documented frontend uses
   (docs/API.md "Point your frontend at LAMU": Open WebUI/Continue
   `/v1/chat/completions` + `/v1/models`; Claude Code `/v1/messages` stream +
   non-stream; AnythingLLM/Ollama `/api/chat` + `/api/tags`; RAG `/v1/embeddings`)
   and asserts the response shape that client library reads. Same skip contract;
   embeddings skips when no embedding-capability entry exists.

3. **`perf_bench`** (`lamu-api/tests/perf_bench.rs`, `#[ignore]`) — the
   **no-GPU regression tripwire**. In-process axum `oneshot` over the read-path
   endpoints (`/health`, `/v1/models`, `/metrics`), single-threaded
   (`bench_read_path_oneshot`) and under N concurrent tasks
   (`bench_concurrent_read_path`, default concurrency 64, mirrors
   `http.rs` `concurrent_requests_no_deadlock`). Emits req/s + p50/p99 and
   asserts a **release-only throughput floor of `rps > 1000.0`**
   (`perf_bench.rs:164` and `:225`); in debug builds the floor is skipped with a
   warning because debug overhead dominates and the number is meaningless.
   Tunable via `LAMU_BENCH_ITERS` (default 2000) / `LAMU_BENCH_CONCURRENCY`
   (default 64). This is the one tier that runs without a GPU and is the
   front-line guard against a lock-contention / serialization regression on the
   shared `parking_lot::Mutex`.

4. **`load_e2e` (the live HTTP load baseline, driven by `oha`)** — sustained,
   ramped-concurrency load against a *running* `lamu serve` on the 4090,
   measuring p50/p95/p99 + tokens/s per surface. Implemented as a thin bash
   wrapper (`lamu-rs/loadtest/oha.sh`) around `oha`
   (https://github.com/hatoo/oha), **not** as a `cargo test` — it needs a live
   GPU, a warmed model, and an external HTTP load generator, none of which belong
   in the Rust test binary. Run via `just bench-http`. **Its committed
   `baseline.json` is deliberately deferred** until a live 4090 run produces real
   numbers; we do not fabricate or commit placeholder numbers (ADR-0014-style
   honesty about not recording hardware results we haven't measured).

## Rationale

- **Gate the expensive tiers, keep the default suite hermetic.** `cargo test
  --workspace` must stay green and fast on a no-GPU runner. `#[ignore]` is the
  right tool: the tests live in the tree, compile in CI (so they can't bit-rot),
  and run only when explicitly asked (`-- --ignored`).
- **A no-GPU perf tripwire is worth more than a GPU-only one for CI.**
  `perf_bench` runs anywhere and catches the regressions most likely to slip in
  — routing/lock/serialization overhead on the read path — without needing
  hardware. The `rps > 1000.0` floor is a coarse but real tripwire.
- **The single-flight invariant must be recorded, because it is load-bearing for
  interpreting results.** Anyone reading a load report needs to know that
  same-model concurrency coalesces on one spawn and is otherwise unthrottled at
  HTTP; a visible HTTP-layer queue tail means a regression introduced
  serialization that shouldn't exist. Writing it down here prevents a future
  reader from "fixing" the absence of a queue.
- **`oha` over a bespoke Rust load generator.** `oha` is a mature, single-binary
  HTTP load tool with built-in p50/p95/p99, ramped concurrency, and JSON output.
  Reimplementing it as a `cargo test` would re-solve a solved problem and couple
  the load generator to the workspace's test runner.
- **Defer the baseline rather than fake it.** A `baseline.json` with invented
  tokens/s is worse than none — it would silently become the regression
  reference. It lands only after a real 4090 run.

## Alternatives Considered

- **Put load tests in the default `cargo test` suite.** Rejected: they need a
  GPU + a real model + sustained wall-clock; running them by default would make
  the suite slow, flaky, and impossible on a hosted runner. `#[ignore]` keeps
  them in-tree and compiling without running.
- **criterion for `perf_bench`.** Rejected (per the SPEC decision note already
  referenced in `perf_bench.rs:18`): a plain tokio timer emitting req/s + p50/p99
  is enough for a tripwire, and criterion's statistical machinery + harness
  override is overkill for an ignore-gated read-path bench.
- **A bespoke Rust HTTP load generator for the live tier.** Rejected: `oha`
  already does ramped concurrency + percentile reporting + JSON out as one
  binary; a custom generator is maintenance with no upside.
- **Commit a baseline.json now with estimated numbers.** Rejected: fabricated
  hardware numbers become a false regression reference. Defer to a live run.
- **Add an HTTP-layer request queue so load is "smooth".** Rejected as a
  non-goal: the single-flight gate is intentionally load-only; throttling
  inference at HTTP is a fairness/quota concern (ADR 0018 §4, opt-in), not a
  default. The load test asserts the *absence* of HTTP-layer serialization.

## Consequences

- **A new directory `lamu-rs/loadtest/` and a `just bench-http` target** enter
  the tree; `oha` becomes an optional external dependency for the live tier
  (documented, not vendored — the operator installs it).
- **`baseline.json` is intentionally absent** until a 4090 run; until then there
  is no committed regression reference for the live tier — the in-process
  `perf_bench` floor is the only automated guard. This is a known, recorded gap.
- **The single-flight-is-load-only invariant is now a documented contract.** A
  future change that adds HTTP-layer per-request serialization must update this
  ADR (and is presumed a regression until justified). The existing
  `loader.rs` `ensure_loaded_single_flight` test (`loader.rs:355`) pins the
  coalescing behavior; this ADR records *why* nothing further throttles HTTP.
- **No CI job runs the GPU tiers.** The repo has **no self-hosted runner
  configured**; `spec_e2e` / `frontend_matrix` / `load_e2e` run only on the
  operator's 4090 on demand. A self-hosted nightly is sketched as a documented
  template (`docs/decisions/` references it), not wired up.
- **Single-GPU + StaticToken/Off paths are untouched.** These artifacts are
  additive (docs + a bash script + a justfile target); no Rust source changes, so
  `cargo test --workspace` stays byte-identical and green.

## Related Decisions

ADR 0001 (the MCP/HTTP split — this scale strategy targets the HTTP inference
surface; MCP queues are tested separately). ADR 0006 (HTTP path never
auto-evicts — the load gate refusing to evict, `loader.rs:168`, is part of why
HTTP has no queue: it fails fast instead of serializing). ADR 0017 (multi-GPU —
the device pool is what a scaled load test will eventually exercise across
devices). ADR 0018 (multi-user — its optional per-model `Strategy::Priority`
wrap is the *only* sanctioned HTTP admission queue, and its quota path is the
contention this strategy stresses). ADR 0014 (don't record hardware results we
haven't measured — the deferred baseline follows this).

## Validation

- `perf_bench` runs green with its release-only `rps > 1000.0` floor on the
  operator's machine (`cargo test --release --test perf_bench -- --ignored
  --nocapture`); a drop below the floor flags a read-path/lock regression.
- `spec_e2e` + `frontend_matrix` pass against a live `lamu serve` on the 4090
  (`-- --ignored --nocapture`); they skip cleanly (Ok) on no-GPU runners.
- `just bench-http` produces p50/p95/p99 + tokens/s per surface against a running
  serve, with **no HTTP-layer queue tail** under ramped concurrency 1/4/16/64 —
  confirming the single-flight-is-load-only invariant empirically.
- We know this is right when a load run shows same-model concurrency coalescing
  on one spawn (one backend, no double-load) and otherwise scaling with backend
  throughput, not with an HTTP-side queue. We'd know it was wrong if the load
  generator surfaces a serialization tail at the HTTP layer, or if `perf_bench`'s
  floor starts failing on unchanged hardware. The deferred `baseline.json` lands
  after the first real 4090 run and becomes the live-tier regression reference.
