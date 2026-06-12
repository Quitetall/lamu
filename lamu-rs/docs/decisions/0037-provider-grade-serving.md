# ADR 0037: Provider-grade serving — structured reasoning, prefix-cache exposure, caps discovery

## Status

Accepted 2026-06-12

## Context

An external agent harness (katana, harness-spec §5.4) embeds `lamu serve`
as its local Provider. Its compiler needs three things the surface didn't
give: thinking blocks as STRUCTURED data (streams stripped reasoning
entirely; the Anthropic bridge dropped it even non-streaming, or passed it
off as text), prompt-cache behavior it can budget against (llama-server's
native `cache_prompt` prefix cache was unexposed, and reuse was never
reported), and `Provider::capabilities()` discoverability (callers had to
hardcode what each model supports). Per-call usage was already engine-true
(ADR 0021).

## Decision

One `ReasoningSplitter` (the M5 stripper's tag state machine, routing
instead of dropping) feeds all three bridges; reasoning surfaces in each
protocol's NATIVE shape: OpenAI `delta.reasoning_content` chunks (and the
existing non-stream `message.reasoning_content`), real Anthropic
`thinking` blocks with a lazy block lifecycle (thinking opens on first
reasoning delta, text on first visible delta, exactly one open at a time;
non-stream leads with a thinking block), Ollama `message.thinking`. An
unclosed think block flushes to the reasoning side — it provably was
reasoning; it never leaks into content and is never silently lost.

`cache_prompt: Option<bool>` is a lamu extension on the OpenAI and
Anthropic request shapes, forwarded to llama-server only when set. Reuse
is reported ONLY when the engine reports it (`prompt_tokens_details.
cached_tokens`, else `prompt_tokens − timings.prompt_n`), surfaced as
OpenAI `usage.prompt_tokens_details.cached_tokens` and Anthropic
`usage.cache_read_input_tokens` — omitted when silent, never fabricated.

Each `/v1/models` entry carries a `caps` object — `{thinking, tools,
cache_prompt, embeddings, context_max}` derived from the registry entry —
so a harness implements `capabilities()` by reading LAMU.

## Rationale

- Reasoning is data the paying caller produced; dropping it on streams was
  lossy and surface-inconsistent. Native shapes (DeepSeek's
  `reasoning_content` convention, Anthropic `thinking` blocks, Ollama
  `thinking`) mean existing clients either render it or ignore it — no
  custom envelope to teach.
- The reasoning-only-completion 502 fix is preserved structurally: a lone
  thinking block is non-empty, so the empty-gate stays quiet — now
  spec-shaped instead of mislabeled as text.
- Cache truth must come from the engine. A fabricated cached-token count
  would poison a harness's hit-rate metric, which is exactly the metric it
  CIs on (spec §5.3); omission is honest, fabrication is not.
- Caps in `/v1/models` keeps discovery on the surface the harness already
  polls — no second endpoint, additive JSON only.

## Alternatives Considered

- **Custom reasoning envelope (one lamu shape on all surfaces)** — one
  parser for us, but every client needs teaching; native shapes are
  already in clients' vocabularies. Rejected.
- **Always-on cache_prompt** — changes engine behavior for clients that
  never asked and can alter sampling determinism across requests; opt-in
  preserves the default contract. Rejected.
- **Separate /v1/capabilities endpoint** — second poll, second cache;
  `/v1/models` is already the model-discovery surface. Rejected.

## Consequences

- Wire deltas (documented, additive): streams now carry reasoning fields;
  whitespace-only pre-think text streams (the old scanner trimmed it);
  Anthropic blocks are lazy (no pre-opened empty text block; tool_use
  indexes continue dynamically); OpenAI transport errors share the typed
  error shape.
- The Anthropic bridge emits `thinking` blocks lamu didn't author —
  clients treating thinking blocks as Claude-only must key on `model`.
- `caps.cache_prompt` keys on the llama_cpp backend; module backends
  (ONNX/candle, ADR 0033+) must extend the mapping as they land.
- Ollama's request surface deliberately gets no cache knob (its options
  vocabulary is upstream-defined).

## Related Decisions

ADR 0021 (engine-true usage — the foundation), ADR 0016 (BYO frontend),
D9 stream-core (the shared parser this builds on), harness spec §5.3/§5.4.

## Validation

- Unit: splitter routing suite (split-tag byte-exact reassembly, unclosed
  block → reasoning, pre-think ordering); cached_tokens_of preference +
  no-fabrication + underflow; parser Usage-with-cached; anthropic block
  tests on the thinking contract. 82 lamu-api tests green.
- Live (training-quiet window): curl all three streams with a Qwen
  reasoning model — reasoning in the new fields, never in content; same
  prompt twice with cache_prompt:true → second call reports reuse;
  /v1/models shows caps. (Queued behind the BLUT run.)
- Katana acceptance: a Provider impl written against documented fields
  only — see docs/API.md § Embedding LAMU as a provider.
