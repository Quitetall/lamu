# LAMU Roadmap — Multi-GPU, Multi-User, Scale-Tested + Double-Down Hardening

Status: active arc, drafted 2026-05-31. Authoritative ADRs: **0017** (multi-GPU,
supersedes 0014), **0018** (multi-user, supersedes 0012). This roadmap sequences
three feature tracks plus the double-down hardening backlog so that **every phase
compiles, keeps `cargo test --workspace` green, and ships independently.**

## Landed so far (updated 2026-06-01)

All three tracks' P1 + the buildable P2 work is merged on `main`; 547 tests
pass, 0 warnings. Where the implementation deviates from the phase plan below,
it's called out so this doc stays honest:

- **Multi-GPU P1** (per-device scheduler + aggregate facades) — ✅ shipped.
- **Multi-GPU P2** (placement + device-aware load) — ✅ shipped (`61eaa97`,
  `c2e65b1`). `DevicePlacement`, `placement_of`, `Backend::set_device`, the five
  `CUDA_VISIBLE_DEVICES` literals now driven by the placed NVML index. Best-fit +
  per-device eviction landed in P1. Single-GPU byte-identical. Physical
  cross-card correctness is hardware-gated (no 2nd card yet).
- **Multi-user P1** (key store + AuthMode + Principal) — ✅ shipped.
- **Multi-user P2/P3** — ✅ shipped *together* (`717c3a4`, `0e1d66a`, `e33b6db`):
  the per-request structured audit event (P2) **and** `quota.rs` token-bucket →
  429 (P3, ahead of the plan's ordering). **Deferred from P2:** the bounded
  Prometheus `user` label on `requests_total`/`tokens_generated_total` — the
  tracing event already carries per-user attribution; the label ripples to every
  counter call-site for low marginal value (tracked as P2b). **Not done from
  P3:** the optional flagged priority-queue wrap of the forward path.
- **Scale** — ✅ frontend integration matrix (P3-equivalent) + an in-process perf
  benchmark shipped as ignore-gated test binaries (`b7569b7`, `d07833b`).
  **Deviations:** the matrix is `lamu-api/tests/frontend_matrix.rs` (a Rust test
  binary), not `loadtest/frontends.sh`; the perf harness is a plain in-process
  tokio timer (`perf_bench.rs`), not `oha` against a real port, and there is no
  committed `loadtest/baseline.json` yet. **Not done:** real-port `load_e2e.rs`
  (P1), the `oha` external tool + baseline (P2 letter), CI hooks + scale ADR (P4).

Remaining buildable-today work: the deferred Prometheus `user` label (MU P2b),
the priority-queue wrap (MU P3), and the real-port/oha/CI scale phases (Scale
P1/P2-letter/P4). Multi-GPU P3 (sharding) + P4 (cookbook multi-device) and
Multi-user P4 (memory owner-scope) remain hardware/condition-gated as below.

## Honest sizing first

This arc is **three tracks of different sizes**, and saying so up front prevents
scope creep:

- **Multi-GPU (ADR 0017)** is the largest and the only one **blocked on hardware
  the dev rig doesn't have**. Phases 1-2 are testable today with synthetic device
  pools; Phase 3 (tensor-parallel sharding) is *merged-but-hardware-validation-
  pending* until a second card lands. This is the same honesty ADR 0014 used.
- **Multi-user (ADR 0018)** is **medium** — a few hundred LOC concentrated in
  `lamu-api` + a CLI verb + one new SQLite. It is genuinely smaller than a browser
  auth stack because LAMU correctly omits passwords/2FA/sessions/CSP. Fully
  testable today (no GPU needed for auth).
- **Scale-testing** is **small-to-medium and mostly test/harness code**, no
  product behavior change except one metrics gauge. Its Phase 0 runs in CI today
  with no GPU; the real-load phases need the 4090 and so become a nightly/release
  GPU job, not a PR gate.
- The **double-downs** (from task #157 / the double-down plan Part 2) are
  *extensions of existing structures, not new subsystems*, and several are direct
  prerequisites or beneficiaries of the tracks above.

## Cross-track dependency graph

```
Scale Phase 0 (unit concurrency + inflight gauge)  ── runs in CI today, no deps
        │
        ├──► Scale Phase 1 (real-port load harness) ──► Scale Phase 2 (oha baseline) ──► Scale Phase 3 (frontend matrix) ──► Scale Phase 4 (CI hooks + ADR)
        │
Multi-GPU P0 (LAMU_GPU_INDEX seam) ─► MGPU P1 (per-device scheduler/facades) ─► MGPU P2 (placement + device-aware load) ─► MGPU P3 (sharding, HW-pending) ─► MGPU P4 (cookbook + flip 0014)
        │                                   │                                        │
        │                                   └── feeds ──► Double-down #1 (ctx-aware eviction) is per-device after MGPU P1
        │                                   └── feeds ──► Double-down #2 (PDEATHSIG reconciliation) is per-device after MGPU P1
        │
Multi-User P0 (ADR 0018, no code) ─► MU P1 (key store + AuthMode) ─► MU P2 (audit/user label) ─► MU P3 (quotas + priority queue) ─► MU P4 (memory owner-scope, DEFERRED)
                                          │                                              │
                                          └──────── MU P3's HTTP priority queue is the natural consumer of Scale's inflight gauge + the existing queue.rs
```

Hard ordering constraints:

1. **Multi-GPU P1 (per-device scheduler) must precede** the per-device variants
   of double-downs #1 and #2 — both touch the scheduler's budget/eviction, which
   changes shape in P1. Doing the double-downs against the scalar pool first would
   mean redoing them.
2. **Scale P0 should land first overall** — it is GPU-free, runs in today's CI,
   and the concurrency-correctness tests it adds (queue fairness, single-flight
   rollback) are the safety net for everything the other tracks do under load.
3. **Multi-user P3's optional HTTP priority queue depends on Scale P0's inflight
   gauge** (or the documented decision to use a new `lamu_inflight` gauge) so the
   contention it manages is observable.
4. **Multi-user P4 (memory owner-scoping) is deferred** and only activates if a
   shared/HTTP memory service is built — it is recorded, not scheduled.
5. Within multi-GPU, **P2 (placement) precedes P3 (sharding)**: cross-card
   placement of distinct models is common, testable with synthetic pools, and
   high-value; sharding is rarer and hardware-blocked.

## Recommended sequence

**Wave A — land the safety net (GPU-free, CI today):**
Scale P0, then Multi-GPU P0 (the `LAMU_GPU_INDEX` seam — smallest possible step,
unblocks "use card 1" immediately), then Multi-user P0 (write ADR 0018, no code).

**Wave B — the two big internal refactors that everything else builds on:**
Multi-GPU P1 (per-device scheduler with scalar facades) and Multi-user P1 (key
store + AuthMode). Independent of each other; can proceed in parallel since one is
`lamu-core/scheduler` and the other is `lamu-api/auth`.

**Wave C — make the new pools/identities actually do work:**
Multi-GPU P2 (best-fit placement + device-aware load), Multi-user P2 (audit/user
label), Scale P1 (real-port load harness on the 4090). Multi-GPU P2 is the first
phase that needs a second card to *exercise* (synthetic tests pass without one).

**Wave D — close out + the hard/rare cases:**
Multi-user P3 (quotas + optional priority queue), Multi-GPU P3 (sharding, marked
HW-pending), Scale P2 (oha baseline). Double-downs #1 and #2 land here as
per-device work now that the scheduler is reshaped.

**Wave E — documentation, validation, and the supersedes:**
Multi-GPU P4 (cookbook multi-device + flip ADR 0014 Status + README index),
Scale P3 (frontend matrix) + P4 (CI hooks + scale ADR), remaining double-downs
(#3 sandbox, #4 council hardware-sizing, #5 temporal-memory fact path).

---

## Track 1 — Multi-GPU (ADR 0017)

Invariant for all phases: `device_count() <= 1` or `LAMU_GPU_INDEX` set → behavior
byte-identical to ADR 0014. A 1-element device Vec reproduces today's numbers.

- **P0 — Seam, no behavior change.** Read `LAMU_GPU_INDEX` (default 0) in the
  scheduler ctor via `config.rs::parse_env_or`. Pure ADR-0014 fulfillment; lets
  the operator pin a non-zero card today. Tests unchanged. *Smallest shippable
  step in the whole arc.*
- **P1 — Per-device scheduler, internal only.** `DeviceBudget` +
  `VramScheduler { devices: Vec<_> }`, enumerate via `device_count()`. **All
  existing scalar methods become aggregate facades.** `LoadedModel.device`
  defaults `Single(0)`; add `set_devices_for_tests`. *Ships:* multi-GPU rigs
  report correct aggregate + per-device VRAM in `vram_status`/`budget`, still load
  single-card. Existing single-GPU tests pass byte-identical.
- **P2 — Placement + device-aware load.** `DevicePlacement`, `place()` best-fit,
  `plan_load_placed`; thread device via `set_device` no-op trait method +
  `make_backend(entry, device)`; replace the five `CUDA_VISIBLE_DEVICES="0"`
  literals with the assigned single index; per-device eviction; VRAM read-back
  from the placed device. *Ships:* distinct models land on distinct cards by
  best-fit (the heavy-LLM-on-0 + S2-Pro/ComfyUI-on-1 scenario ADR 0003 flagged).
- **P3 — Tensor-parallel sharding (llama.cpp only), HW-validation-pending.**
  `shardable` (or derived from `vram_mb > max single-device total`),
  `Sharded(Vec)` placement, `--split-mode layer` (default; `row` via
  `LAMU_SPLIT_MODE`) + `--tensor-split` + multi-card mask; Python backends reject
  `Sharded`. *Ships:* a 70B that fits 2x24 but not 1x24 loads — **merged, real
  masking/`--tensor-split` validated only when a second card lands.**
- **P4 — Cookbook multi-device (closes ADR 0015 Phase 4).** `Hardware.devices`,
  fit against largest-single vs summed-for-shardable, `_group_gpus` helper, update
  both call sites. **Flip ADR 0014 Status → "Superseded by 0017", add 0017 to the
  README index, add Related pointers from 0003 and 0015.** *Ships:* `lamu cookbook`
  ranks correctly on multi-GPU; the feature is documented + superseded.

## Track 2 — Multi-user (ADR 0018)

Invariant: `AuthMode::StaticToken` is the default; ADR-0012 deployments unchanged.

- **P0 — ADR + decision (no code).** Write ADR 0018 superseding 0012. Contract:
  per-token identity + quotas + audit on HTTP; MCP stays single-principal-per-
  process; no sessions/2FA/passwords.
- **P1 — Key store + multi-token auth (the core).** `keys.rs` / `keys.db`
  (hashed tokens, issue/revoke/list/verify); `AuthMode` in `AppState`; branch in
  `require_bearer`, insert `Principal` into request extensions. CLI: `auth
  issue/list/revoke`. *Ships:* multiple users with distinct revocable keys — this
  alone is genuine multi-user for an API. Off-loopback gate must treat empty key
  store as "no auth configured" → hard-fail.
- **P2 — Audit + usage attribution.** Bounded `user` label on
  `requests_total`/`tokens_generated_total`; per-request structured tracing event
  `{user, key_prefix, model, route, status, tokens, ts}`. *Ships:* who-did-what.
- **P3 — Quotas + fairness.** `quota.rs` token-bucket per user (from
  `daily_token_quota`) → 429; then optionally (flagged) wrap the HTTP forward path
  with the existing `RequestQueue` `Strategy::Priority` at `principal.priority`.
  *Ships:* per-user rate limits + priority preemption under contention. Consumes
  Scale P0's inflight gauge for observability.
- **P4 — Memory owner-scoping. DEFERRED — recorded, not scheduled.** Idempotent
  `owner` ALTER on `lifetime_memory.memories` + thread through
  `remember`/`recall_memory`/`forget`; conversation memory via owner-prefixed
  `conversation_id`. **Activates only if a shared/HTTP memory service is built**;
  today per-process MCP + OS-user isolation already separates users' memory.

## Track 3 — Scale-testing

Invariant: `#[ignore]`'d real-model tests auto-skip when binary/registry/VRAM
absent, so GPU-less CI is a clean skip, never a hang. Default CI stays fast.

- **P0 — Pure-unit concurrency hardening (CI today, no GPU).**
  `lamu-core/tests/test_queue_load.rs` (1000 enqueues × Fifo/Lifo/Priority:
  ordering invariants, no permit leak, FIFO fairness, per-enqueue
  `tokio::time::timeout` so a missed-wake regression fails loudly not hangs); a
  single-flight rollback-under-concurrency test (N concurrent calls where spawn
  fails → exactly one attempt, scheduler returns empty, no leaked
  `mark_loading`); optional new `lamu_inflight_requests{model}` gauge + its unit
  test. *Exit:* `just test-rust` green, new tests in normal CI.
- **P1 — Real-port load harness (gated, opt-in).** Factor `spec_e2e.rs` helpers
  into `tests/common/mod.rs`; add `lamu-api/tests/load_e2e.rs`:
  `concurrent_chat_same_model_single_flight`, `mixed_surface_concurrent`,
  `streaming_under_load` (per-format terminators), `eviction_refused_under_
  parallel_load` (asserts the ADR-0006 `won't auto-evict` error, no leaked
  state). Add `just load`. *Exit:* `just load` passes on the 4090, GPU-less CI
  skips cleanly. **One serve per test, fan concurrency inside it** (avoid the
  `ephemeral_port` TOCTOU under many servers).
- **P2 — External load tool + regression baseline.** `loadtest/oha.sh` +
  per-surface profiles (temp 0, small `max_tokens`); ramped concurrency
  (1/4/16/64) against `just serve`; record p50/p95/p99 + tokens/s + 5xx into
  committed `loadtest/baseline.json`; `just bench-http`. *Exit:* one command
  reproduces a load run with pass/fail vs baseline (>X% p95 regression fails).
- **P3 — Frontend integration matrix.** `loadtest/frontends.sh`, one assertion
  per `docs/API.md:580` row: Claude Code (Anthropic stream + tool_use round-trip,
  `model:"lamu"` alias), Open WebUI (OpenAI `/v1/models` + stream/non-stream
  reasoning_content handling; Ollama `/api/tags` + `stream`-defaults-true NDJSON +
  predictable 404s on `/api/version`/`/api/show`/`/api/generate`), AnythingLLM,
  RAG `/v1/embeddings`, the per-surface 401 envelope matrix + CORS preflight.
  `just smoke-frontends`. *Exit:* one script certifies every documented frontend
  pairing live.
- **P4 — Observability + CI hooks + ADR.** Document the under-load watch-list
  (`request_duration` p95/p99, `requests_total` status distribution,
  `backend_health_state` degraded/quarantined flips, `backend_restarts_total`
  thrash, `vram_used_mb` headroom, the new inflight gauge). Keep oneshot
  `http.rs` + P0 units in the always-on PR gate; add a **self-hosted-4090 nightly/
  release job** running `just load` + `bench-http` + `smoke-frontends`. Write the
  scale-test ADR recording the **"HTTP has no request queue; single-flight is
  load-only"** invariant so it can't regress silently. *Exit:* fast PR gate +
  nightly GPU job; strategy documented.

## Double-down hardening (task #157, from the double-down plan Part 2)

These extend existing structures; none add a subsystem; all hold the loopback-
default calibration. Sequenced relative to the tracks:

1. **Cross-modal VRAM scheduler — ctx-aware reload-cost eviction.** Feed the
   cookbook's ctx-aware estimate + roofline into `plan_eviction`/`plan_load`
   (reload-cost-aware sort; ctx-parameterized fit to stop over-reserving; "I can
   load this if I drop you to 16k ctx" fallback). **Do after Multi-GPU P1** — it
   becomes per-device.
2. **PDEATHSIG reconciliation loop.** Diff `query_gpu_pids()` vs registered PIDs,
   surface orphan holders in `budget()`, add guarded `lamu reclaim`. **Do after
   Multi-GPU P1** — orphans are now per-device (an orphan on card 1 must not
   shrink card 0's budget). Also directly supports Scale-test correctness.
3. **Bubblewrap sandbox for the review preflight.** Extend the `lamu agent`
   harness to the `review_commit` cargo preflight + git reads (hostile `build.rs`
   runs un-sandboxed during `cargo test` today). Independent of the three tracks;
   land in Wave E.
4. **Judged council hardware-sizing.** Sequential load/unload on one GPU via
   `plan_eviction`; envelope the blind-answer + judge concatenation (the #154
   highest-value site); log cross-vendor disagreement to temporal memory.
   Benefits from Multi-GPU P2 (a multi-card rig can run council members in
   parallel) but doesn't require it; land in Wave E.
5. **Temporal-memory fact-TEXT path.** Envelope recalled fact bodies into the
   contradiction judge + recall return; add a provenance field (user-stated >
   tool-ingested on conflict); surface superseded-fact history. Independent; pairs
   naturally with Multi-user P4 if/when owner-scoping lands. Land in Wave E.

## Definition of done per phase

Every phase: compiles, `cargo test --workspace` green, `just lint` clean, one
commit → `mcp__local-llm__review_commit`. Hardware-blocked phases (Multi-GPU P3,
all real-model Scale phases) ship behind `#[ignore]`/synthetic pools and are
labelled **merged, hardware-validation-pending** in their ADR Validation section,
mirroring ADR 0014's honesty.

