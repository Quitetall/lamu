# ADR 0035: HF candle runtime — safetensors models in-process

## Status

Accepted 2026-06-12

## Context

Safetensors checkpoints without GGUF conversions (new architectures,
unconverted finetunes, models straight off the Hub via `lamu pull`'s
future growth) had no runtime: scan labeled them and assigned llama.cpp,
which can't load them. The user decision: candle (HuggingFace's Rust ML
framework) first — no Python in the serving path — with a `hf_py`
python-server escape hatch recorded as the follow-up for architectures
candle lacks.

## Decision

`lamu-hf` (backend_kind "hf_candle", ADR 0023/0026 pattern) runs
Llama/Mistral/Qwen2-family safetensors models in-process through the
ADR 0033 shim, whose ChatEngine half this wave adds to lamu-inproc:
`/v1/chat/completions` (non-stream + SSE, llama-server wire shapes the
lamu-api bridges already parse; raw `<think>` flows as content — the
ADR 0037 splitter owns reasoning downstream), `/tokenize` (count is the
contract — ADR 0021 engine-truth occupancy via the engine's real
tokenizer), `/health`. Error seam: the response is held until the first
fragment, so pre-output failures are real non-2xx envelopes
(`send_upstream` decodes them) and mid-stream failures emit one in-band
error line and close WITHOUT `[DONE]`.

Engine: config.json `model_type` dispatch; ALL shards mmapped
(`VarBuilder::from_mmaped_safetensors`); minijinja over
`tokenizer_config.json`'s chat_template (pycompat + `raise_exception`;
render failure → ChatML fallback with a warning, never a failed
request); eos ids collected across config/generation_config/tokenizer;
KV-cached forward + `LogitsProcessor` (temp/top_p/top_k; min_p and
repeat_penalty are documented no-ops — candle has no processor for
them); incremental detokenization via `DecodeStream` (split-UTF-8 safe);
`Mutex` serializes generation, matching RequestQueue concurrency=1.
`set_device` stores the `DevicePlacement`; the `cuda` feature maps it to
`Device::new_cuda(primary)` — never CUDA_VISIBLE_DEVICES (in-process).
Feature ladder: lamu-hf default = CPU candle; lamu-cli `hf-candle`
(CPU), `hf-candle-cuda`, and `full` stays CPU-candle so it builds
without nvcc (this box needs `CUDAHOSTCXX=g++-14`).

Scan: safetensors + supported model_type → HfCandle with real arch,
context_max from max_position_embeddings, and
`vram_mb = 1.1 × SUM(all shards) + KV headroom` — fixing the
first-shard-only sizing bug for EVERY safetensors entry; unsupported/
absent config keeps pre-0035 behavior exactly.

Unlike the embeddings-only ONNX backend, `Backend::generate/
generate_with_opts/stream` are implemented directly — lamu-mcp holds the
Box and calls `generate_with_opts` (handlers.rs), so an erroring impl
would have broken MCP `query` for hf models.

## Rationale

- candle keeps the single-static-binary story and the PDEATHSIG-free
  in-process model (ADR 0033's tradeoffs apply; the resident CUDA
  context ~300-400 MB is recorded as scheduler margin once the first
  cuda load happens).
- Engine-truth `/tokenize` extends ADR 0021's un-fakeable occupancy to
  safetensors models — llama.cpp was previously the only source.
- Hold-until-first-fragment error framing means a model that fails to
  prompt-process produces a diagnosable HTTP error, not an empty stream
  killed by the empty-backend gate.

## Alternatives Considered

- **Python transformers/vLLM server first** — max architecture coverage,
  but a venv + python process in the serving path and slow loads; kept
  as the recorded `hf_py` follow-up for exotic archs.
- **GGUF-convert-on-scan** — lossy, slow, and silently changes the
  artifact the operator placed; conversion is the operator's call.
- **Erroring generate like ONNX** — would break MCP query (the one
  consumer that drives the Box directly). Rejected on verified call-site
  evidence.

## Consequences

- `/tokenize` returns sequential placeholder ids (the count is the
  contract; the only in-tree consumer measures length) — widening
  ChatEngine for real ids is a recorded follow-up.
- Clock-seeded sampling (no per-request seed plumbing exists yet).
- `HF_CANDLE_MODEL_TYPES` is mirrored in lamu-core (scan can't depend on
  the feature-gated crate); drift is benign (entry assigned, load fails
  loudly) and documented.
- CUDA path is wiring-verified but compile-untested here (nvcc minutes;
  needs CUDAHOSTCXX) — first cuda build is a live-gate item.

## Related Decisions

ADR 0033 (shim + feature policy), ADR 0021 (tokenize truth), ADR 0037
(reasoning split downstream), ADR 0026 (kind dispatch), ADR 0017
(DevicePlacement).

## Validation

20 lamu-hf tests (templates incl. pycompat/raise_exception + ChatML
fallback, config normalization, eos collection, DecodeStream split-UTF-8
proof on an inline BPE tokenizer, backend lifecycle) + 11 new
lamu-inproc chat-server tests (SSE framing replayed through
parse_upstream_line's exact extraction; mid-stream error-no-DONE) + 6
scan fixtures (multi-shard sum, headroom, ctx-cap). Engine e2e RUN
against a local Llama-3.2-1B checkpoint on CPU: loaded, 16 greedy
tokens, finish_reason length. Live CUDA gate queued (training +
nvcc setup).
