# ADR 0022: First HTTP serving micro-baseline (warm, short, single-model)

## Status

Accepted 2026-06-05

(ADR 0021 is reserved for the context-occupancy work on `feat/context-occupancy`;
this benchmark ADR is independent and lands via `fix/oha-output-flag`.)

## Context

ADR 0020 deferred committing a `baseline.json` until a real GPU run existed and
forbade fabricating numbers. `loadtest/oha.sh` also used oha's old `-j` flag,
broken on `oha >= 1.14` (fixed in this branch → `--output-format json`).

A live run is now possible: `lamu serve` up on one RTX 4090,
`qwen3.6-27b-uncensored-heretic-v2-q4_k_m` resident (~20 GB), ~1.0–1.2 GB free.
That headroom is too small to spawn a second serve, so `load_e2e` / `just load`
self-skip on VRAM contention; `bench-http` against the running server is the
correct tool at this headroom. The deployed binary predates the ADR 0021 work.

This ADR commits a **deliberately narrow micro-baseline** and is explicit about
what it does and does not measure — because an adversarial audit of the first
draft caught it overclaiming (details under "What this does NOT measure").

## Decision

Commit, under `lamu-rs/loadtest/baseline/`, the raw oha JSON for every run plus
a distilled `baseline.json` with full provenance. Report **only the
2xx-verified surfaces (OpenAI, Ollama)**; record the Anthropic result as a
**failure finding**, not a latency number. Methodology: `oha 1.14.0` via
`loadtest/oha.sh`, identical 6-word prompt, temperature 0, `stream:false`,
single run per cell (no repeats), concurrency 1/4/16/64 (read path and 128-tok
run cover fewer rungs — see Coverage). Every cell is live-measured.

## Measured (OpenAI + Ollama, HTTP 200 verified, warm)

`max_tokens=24` (every request hit `finish_reason=length`, i.e. generated
exactly 24 tokens — see the reasoning-only caveat below). req/s · p50 · p95 · p99 (s):

| conc | openai (200) | ollama (200) |
|---|---|---|
| 1  | 2.50 · 0.39 · 0.46 · 0.62 | 2.46 · 0.40 · 0.46 · 0.48 |
| 4  | 2.56 · 1.54 · 1.74 · 1.87 | 2.45 · 1.64 · 1.79 · 1.85 |
| 16 | 2.52 · 6.28 · 6.44 · 6.47 | 2.73 · 5.74 · 6.12 · 6.20 |
| 64 | 2.47 · 12.6 · 24.5 · 25.9 | 2.64 · 13.2 · 23.2 · 24.2 |

`max_tokens=128`, openai (200), coverage c1/c4/c16 only: 0.76 · 0.76 · 0.82 req/s;
p95 1.41 / 6.65 / 20.2 s.

Decode rate, **derived from committed c1 latency ÷ the token cap** (valid because
every request hit `finish_reason=length`, generating exactly the cap): 24 tok /
0.391 s ≈ **61 tok/s**; 128 tok / 1.304 s ≈ **98 tok/s**. The gap is prefill/HTTP
fixed-cost amortization over more decoded tokens. (The earlier "18.8 tok/s"
draft figure was a single cold curl probe and is retracted — it divided 24 tok
by an unrelated 1.3 s wall and contradicted the 0.39 s measured latency.)

Read path (GET, conc 16 only, decode-free): /health 16.3k req/s, /v1/models
17.0k, /metrics 25.0k; p95 ≤ 1.74 ms but **p99 5–11 ms** (disclosed: the tail is
not sub-2 ms); HTTP 200.

## Findings

- **Anthropic `/v1/messages` 502s for a reasoning-only response.** All four
  anthropic runs were HTTP 502 (`backend_returned_empty`), reproduced live. Root
  cause: a thinking model given a small `max_tokens` spends the whole budget in
  `<think>` (`finish_reason=length`, `content:""`); the OpenAI path returns that
  as an empty-content 200, but the Anthropic bridge builds zero content_blocks
  and 502s its empty-gate. A real surface divergence — follow-up: have the
  bridge surface reasoning-only / empty completions instead of 502, or document
  the contract. (Note: oha's `successRate:1.0` counts "got an HTTP response",
  NOT 2xx — the 502s were hidden until status codes were inspected.)
- **Flat req/s under concurrency — consistent with single-flight, but this run
  cannot prove it.** req/s holds ~2.5 (openai) / ~2.6 (ollama) from conc 1→64
  while latency rises linearly. That matches single-flight serialization, but is
  equally consistent with a single 27B decode stream already saturating the 4090
  (parallel slots wouldn't raise aggregate req/s then) or with `n=64` being one
  request deep per worker (no pipeline). Distinguishing them needs a control
  (smaller model, or `--parallel` slots) not run here. Stated as hypothesis.
- **Runs measure raw DECODE latency, not answer latency.** Every completion was
  reasoning-only/empty-content (`finish=length`), so these are "time to emit N
  tokens", not "time to a useful answer".
- **Read path is not the bottleneck:** decode-free GETs sustain 16–25k req/s,
  ~4 orders of magnitude above the ~2.5 req/s decode ceiling.

## Coverage (what actually ran)

OpenAI + Ollama: full 1/4/16/64 @ 24 tok. OpenAI 128-tok: c1/c4/c16 (no c64).
Read path: conc 16 only (not the full ramp). Anthropic: 1/4/16/64, all 502.

## What this does NOT measure

Streaming / time-to-first-token (`stream:false` → oha's first-byte == total in
every JSON, so token-level behavior is invisible); real or long prompts (one
fixed 6-word string → prefill pinned at a trivial floor, ADR-0021
context-occupancy path untested); cold start / model load; sustained / steady
state (runs are 26–39 s, one batch deep — no backlog, no thermal drift); genuine
overload (`n=64` never builds a real queue, so "no OOM" only means it survived 64
trivially-short requests); multi-model / model-switch contention; run-to-run
variance (single run per cell — no repeats).

## Alternatives Considered

- **Commit the first (overclaiming) draft** — rejected: an adversarial audit
  showed it stated single-flight as fact, called the surfaces equivalent while
  Anthropic was 100% 502, and carried a self-contradictory decode figure.
- **Defer until the #166 rebuild** — rejected: the OpenAI/Ollama decode numbers
  are real and worth a pre-feature reference point.
- **`load_e2e`** — self-skips at ~1 GB free (needs ~20 GB for a 2nd serve).

## Consequences

- A narrow, honest regression reference for OpenAI/Ollama short-gen decode
  latency on this model+GPU. NOT a throughput SLA or a streaming baseline.
- The Anthropic 502-on-reasoning-only divergence is a tracked follow-up.
- Throughput headroom (parallel decode slots / continuous batching) is the
  named lever to lift the flat ~2.5 req/s ceiling — and the experiment that
  would also settle the single-flight-vs-saturation question.
- No `±N%` regression gate is set: single runs give no variance estimate (the
  Anthropic-adjacent ollama/openai spread already shows per-cell noise). A real
  gate needs repeated runs.

## Related Decisions

ADR 0020 (policy this fulfills), ADR 0016 (orchestrator), ADR 0021
(context-occupancy — the next re-run target, and owner of the Anthropic-bridge
follow-up).

## Validation

Right if a repeat run on the same model/GPU reproduces the OpenAI/Ollama req/s
and p50 directionally, and if the Anthropic 502 reproduces (confirming it's a
real contract gap, not a one-off). Wrong / superseded the day decode runs with
adequate `max_tokens` and `stream:true` give answer-latency + TTFT, or `--parallel`
slots make req/s scale with concurrency — both of which this baseline explicitly
does not capture.
