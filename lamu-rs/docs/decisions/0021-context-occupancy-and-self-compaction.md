# ADR 0021: Un-fakeable context-occupancy signal and self-compaction tools

## Status

Accepted 2026-06-04

## Context

Agents driving lamu (Claude Code, Odysseus, any OpenAI/Anthropic client) have
no trustworthy way to know how close a request is to the served model's context
limit. The model's own reply text ("I'm about 80% full") is unfounded — a
sampled token stream the model can fabricate, and not in the data path of any
real measurement. The requirement: surface a context-full signal tied to
something the generating model **cannot** fake, plus tools for an agent to
compact its own conversation when near the limit.

A six-investigator survey of the codebase plus a three-lens adversarial review
established the constraints:

- lamu talks to llama-server over HTTP and already probes `/health` out-of-band
  in the reconcile loop (`reconcile.rs:62`), treating the engine's own answer +
  NVML as ground truth — never the model's self-report. The same pattern can
  carry a real occupancy number.
- lamu already parses GGUF headers (`registry.rs:55 parse_gguf_meta`), so the
  model's real trained context length is readable without a new dependency.
- Two distinct "context" substrates exist and were being conflated: (a) the
  llama-server **KV cache** (`/slots` `n_past`), driven by whatever external
  client is decoding on a slot, and (b) the lamu-side **`cloud_query` SQLite
  conversation log** (`memory.rs`, schema `turns(conversation_id, idx, role,
  content, ts, metadata)`), which is re-sent statelessly each call.
- `/slots` is server-wide: lamu sends no `id_slot`/`conversation_id`, so there
  is zero conversation→slot affinity; under concurrency (`LAMU_QUEUE_CONCURRENCY
  > 1`, streaming unserialized) the "max-occupied slot" reflects some other
  request, not the caller's. `--cache-reuse 256` further decouples `n_past`
  (resident KV, possibly shared/partial) from a single conversation's logical
  token count.
- `registry.rs:328` hardcodes `context_max = 131072` for every discovered
  model regardless of its real trained context, so any ratio using it as the
  denominator under-reports fill by up to ~4× — a "you're fine" false negative
  at the exact moment of overflow.

The reviews proved that sourcing a *per-conversation* signal from `/slots` and
then "compacting" the SQLite log would alarm on one subsystem and act on an
unrelated one, while dividing by a fabricated denominator.

## Decision

The per-conversation context-occupancy metric is **the engine tokenizer's exact
count of the assembled prompt divided by the model's real trained context
length**: POST the rendered prompt to llama-server `POST /tokenize`, count the
returned ids, divide by `n_ctx_train` read from GGUF metadata (extending
`parse_gguf_meta`; the launched `--ctx-size` is the secondary cap). This number
is un-fakeable (it is computed by the engine's tokenizer, never emitted by the
generating model) and per-request (it measures *this* prompt). It is surfaced
additively only: a `context_window` sibling inside the `usage` object on
OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` responses, and a
pull-based `context_status` MCP tool. When no live engine can tokenize, the
metric is reported as `source: "cold"` / `null` — never a fabricated number.
llama-server `/slots` `kv_cache_usage_ratio` is retained **only** as a
separate, explicitly server-scoped Prometheus gauge for ops, never presented as
a caller's conversation fill. `compact_context` is provided in **both** forms:
by default stateless — it takes the caller's message list, preserves the system
turn + latest user turn + last K turns verbatim, summarizes only the stale
middle via mimo-v2.5, and **re-tokenizes the result through the same `/tokenize`
path to prove the token drop is real**; with `persist: true` it additionally
rewrites the stored `cloud_query` conversation using append-only supersede
markers in the `metadata` column, guarded by a per-`conversation_id` lock, a
confirm-time re-read (TOCTOU guard), and a matching reader-side filter in
`recall`/`recall_ranked`.

## Rationale

- **Un-fakeable means out-of-band, not just "from the engine".** The model's
  sampler can write tokens into its reply but cannot write the engine
  tokenizer's id count. `/tokenize` is computed by the same out-of-band HTTP
  surface the reconcile loop already trusts for `/health`.
- **Per-request beats per-slot for "this conversation".** `/tokenize` measures
  the exact prompt that will be decoded; `/slots` measures whatever a
  (possibly different) request left resident in a shared KV slot. With no slot
  affinity, only `/tokenize` answers the caller's question.
- **The denominator must be the model's real context.** GGUF
  `{arch}.context_length` is the trained context; the hardcoded 131072 is an
  intent placeholder that is wrong for most models. Dividing by truth is the
  difference between a real card and a prettier fake one.
- **Additive surfacing keeps every existing client working.** Unknown keys
  inside `usage` are ignored by OpenAI/Anthropic SDKs (the spec already ships
  beta `cache_*` siblings there); a new MCP tool is auto-registered. HTTP
  headers were rejected — no handler sets custom headers today and SDKs hide
  them.
- **Close the loop with the same metric that fired the alarm.** A compaction is
  only proven by re-measuring with `/tokenize`, not by a `char/4` estimate of
  SQLite rows (lamu has no internal tokenizer; `char/4` is the very estimate
  this ADR refuses as authoritative).
- **Stateless-by-default dissolves the dangerous failure modes.** Returning a
  compacted message list (rather than mutating shared state) sidesteps the
  supersede-reader, TOCTOU, and concurrent-write blockers entirely; the
  stateful path is opt-in and pays for those guards explicitly.
- **Preserve-first, never blind-trust a lossy summary.** System + latest-user +
  recent turns are kept verbatim; only the middle is summarized, with a
  compaction-specific prompt that retains in-flight working state (current
  task, files, open threads) — not the cross-session `EXTRACTION_PROMPT`, which
  is lossy by design and would discard exactly that.

## Alternatives Considered

- **`/slots` `n_past` as the per-conversation metric** — rejected: server-wide
  with no conversation→slot affinity (lamu sends no `id_slot`), so under
  concurrency it reports another request's fill; `--cache-reuse` decouples
  `n_past` from a conversation's logical tokens. Retained only as an ops gauge.
- **`registry.context_max` (131072) as the denominator** — rejected: fabricated
  per-model, under-reports fill up to ~4×, produces false "context is fine".
  Replaced by GGUF `n_ctx_train`; `--ctx-size` boot value is the secondary cap.
- **Model self-reported occupancy** — rejected: this is the fakeable card the
  requirement exists to eliminate; not in any measurement's data path.
- **`char/4` token estimate as the compaction proof** — rejected: not engine
  truth; only acceptable as an explicitly-labelled client-side hint.
- **Destructive compaction (delete/overwrite middle turns)** — rejected: breaks
  the append-only audit log and makes undo impossible. Stateful path uses
  additive supersede markers instead.
- **HTTP response headers / top-level response fields** — rejected: no handler
  sets headers, SDKs hide them; strict validators are likelier to reject
  unknown top-level keys than unknown keys inside `usage`.

## Consequences

- New load-bearing path: `parse_gguf_meta` gains an `n_ctx_train` read that the
  denominator depends on; a missing/odd GGUF key must degrade to `unknown`, not
  to 131072.
- `usage` objects on OpenAI + Anthropic responses gain a `context_window`
  sibling; downstream consumers may start depending on it.
- The stateful compaction path makes `recall`/`recall_ranked` supersede-aware —
  every future reader of conversation turns must honor the filter or it will
  re-inflate compacted conversations.
- A per-`conversation_id` async lock is introduced around the stateful
  read-summarize-write; future memory writers must respect it.
- Live verification is gated on task #166 (GPU is training; no local
  llama-server up): the actual `/tokenize` response shape, the real
  `n_ctx_train` per installed model, and occupancy climbing across a growing
  conversation can only be confirmed against a running engine. Until then the
  feature ships behind default-safe paths (no live engine → `cold`) with unit
  tests against captured fixtures.

## Related Decisions

ADR 0016 (backend orchestrator / BYO frontend), ADR 0018 (multi-user per-token
identity — future `id_slot` affinity would build on this), ADR 0019 (cloud
model catalog — where `n_ctx_train` for cloud models would be sourced).

## Validation

Right if, on a live server (post-#166): (a) `context_window.tokens` from
`/tokenize` matches an independent token count of the same prompt within
rounding; (b) `occupancy_ratio` uses each model's real `n_ctx_train`, not
131072; (c) `compact_context` shows a real `/tokenize` token drop in its
before/after, and the stateless path never mutates stored state; (d) the
stateful path's superseded ranges never reappear in `recall_ranked` output and
two concurrent compactions of one id cannot double-summarize. Wrong if any
surface ever emits a non-null occupancy when no engine tokenized the prompt, or
if the denominator falls back to 131072 silently.
