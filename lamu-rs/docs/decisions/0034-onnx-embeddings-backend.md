# ADR 0034: ONNX embeddings-first backend (`lamu-onnx`, ort, CPU EP v1)

## Status

Accepted 2026-06-12

## Context

Memory recall, RAG, and `/v1/embeddings` all depend on an embedding
model; the only local option was a GGUF embedding model through
llama.cpp, and the memory stores still fell back to OpenAI keys. The
richest open embedding models (bge, MiniLM, gte) ship as ONNX exports
with a `tokenizer.json` sidecar. Local-first memory (ADR 0030, next)
needs a keyless embedder; ONNX text-GENERATION is not worth the
complexity on a box where llama.cpp already serves generation.

## Decision

`lamu-onnx`, embeddings-only in v1, behind the `onnx` cargo feature:
ort (pinned `=2.0.0-rc.12` — pre-release semver can't float; wraps ONNX
Runtime 1.24, default features = CPU-only EPs + binary download) +
HuggingFace `tokenizers`. Engine: session inputs introspected by name
(`input_ids`/`attention_mask` required, `token_type_ids` fed only if
declared), truncation at the tokenizer's declared max else 512, output =
`sentence_embedding` if the export has one else attention-masked
mean-pool of the first output, L2-normalized; dims discovered by probe.
Serves through the ADR 0033 shim. CPU EP only → `vram_mb 0`, no
scheduler interplay; CUDA EP is an explicit follow-up. Scan: `.onnx`
files → `BackendType::Onnx`, `Capability::Embedding`, name from parent
dir (stem when the dir is generic), note flags a missing tokenizer.json.

## Rationale

- Embeddings-first is the highest-value slice: it unblocks keyless
  memory/RAG (ADR 0030) with small models where CPU latency is fine and
  zero VRAM means zero contention with the 4090's LLM/training load.
- Mean-pool + L2 matches the sentence-transformers convention the target
  models were trained with; honoring a `sentence_embedding` output skips
  double-pooling on exports that bake it in.
- Name-based input introspection fails loudly (listing the session's
  actual inputs) instead of feeding tensors positionally into unknown
  graphs.

## Alternatives Considered

- **ONNX text generation v1** — KV-cache plumbing + sampling for models
  llama.cpp already serves better locally. Deferred indefinitely.
- **CUDA EP v1** — pulls CUDA into ort's build/runtime for models that
  run in milliseconds on CPU. Follow-up only if corpus latency demands.
- **fastembed-style vendored runtime** — another abstraction over ort
  with less control of input introspection. Rejected.

## Related Decisions

ADR 0033 (the shim it serves through), ADR 0030 (local-first embedder —
consumes this), ADR 0026 (backend_kind "onnx"), ADR 0010 (Embedding
capability routing).

## Validation

8 lamu-onnx tests (engine fixture-gated on `$LAMU_TEST_ONNX_MODEL` with
a clean SKIP + setup hint; embeddings-only error; lifecycle sanity) +
3 scan tests + serde-name agreement + composition-root both feature
states. Live gate (queued behind training + fixture download): scan a
bge-small ONNX export, `curl /v1/embeddings`, semantic sanity, then the
ADR 0030 keyless recall e2e.
