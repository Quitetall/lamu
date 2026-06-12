# ADR 0030: Local-first embedding with per-store embedder identity

## Status

Accepted 2026-06-12

## Context

Every semantic surface (lifetime-fact recall, conversation ranking, RAG)
embedded exclusively through OpenAI: no `OPENAI_API_KEY` meant recall
degraded to recency-only, and a supposedly local-first stack had a cloud
key on its memory path. Local embedding execution already existed
(registry `Capability::Embedding` model via llama.cpp, and now ONNX per
ADR 0034) but was unreachable from the memory code: the memory MCP tools
dispatch as `HandlerKind::Free` (no server reference) and the detached
autocapture/reconcile tasks have no context either — `ToolCtx` threading
cannot reach them. Mixing embedding models silently corrupts cosine
ranking (vectors from different models share no space), so going local
also forces identity tracking.

## Decision

An async `Embedder` trait in lamu-memory with a PROCESS-GLOBAL
registration (`set_global`/`resolve`), resolution order:
`LAMU_EMBED_PROVIDER=openai` pin (warn-once + None when keyless — an
explicit pin never silently falls through) → registered local embedder →
keyed `OpenAiEmbedder` (the old code, relocated) → None. Composition
roots register local adapters only when the registry has an
embedding-capable entry at startup (restart-to-pick-up): lamu-mcp wraps
its `ToolCtx::embed` path (model pick made deterministic — min-by-name),
lamu-api wraps the factored `resolve_embedding_backend()`. The CLI gets
its own chain (env → a RUNNING `lamu serve` via `HttpServeEmbedder` →
OpenAI) for `lamu memory reembed`.

Identity is enforced per row and per store: writes stamp
`embedding_model`; `embedding_stores` upserts when absent/matching and
warns-once + keeps the old pin on mismatch; EVERY vector leg (recall,
novelty dedup, chunk search) filters `embedding_model = current` —
cross-model cosine is structurally impossible. Recall is hybrid: FTS5
BM25 leg + model-filtered vector leg merged by reciprocal rank fusion
(constant 60, pure fn), hydrated in merge order; the wire shape is
frozen and `score` remains the raw cosine for vector hits only, because
lamu-mcp's contradiction guard compares it against `NOVELTY_THRESHOLD`
(an RRF score there would have silently disabled the guard). No
embedder → FTS + recency (strictly better than the old recency-only).
Convergence after a model switch = `lamu memory reembed` (dry-run
default, batched, per-batch commits so reruns resume, store identity
flipped only after rows converge).

## Rationale

- Process-global registration is the only seam that reaches Free
  handlers AND detached tasks; ToolCtx threading reaches neither.
- The trait is async because the storage fns it serves already are, and
  the MCP adapter must await `ensure_loaded` — a sync trait would force
  `block_on` inside tokio workers.
- Identity-filtering at the SQL layer makes mixed-dim corruption
  unrepresentable rather than policed; the FTS leg keeps mismatched-era
  rows reachable until reembedded.
- RRF over score normalization: rank-only fusion needs no calibration
  between BM25 and cosine scales and is trivially testable.

## Alternatives Considered

- **Thread ToolCtx into memory tools** — flips five tools to stateful
  dispatch and still misses the detached tasks. Rejected.
- **Re-embed on switch automatically at startup** — surprise API/compute
  cost at boot; explicit `reembed` with a dry-run default matches the
  `lamu clean` safety convention. Rejected.
- **Score-normalized fusion** — needs per-corpus calibration; RRF is
  parameter-light and robust. Rejected.

## Consequences

- Keyless installs get semantic recall via any local embedding model
  (GGUF nomic or ONNX bge) and keyword recall with none at all.
- `index_repo`'s mtime skip means unchanged files keep old-model vectors
  after a switch until `reembed --store chunks` (documented).
- Registering adapters at startup only: adding the first embedding model
  to the registry needs a restart to activate local embedding.
- lamu-api now depends on lamu-memory (frontend → module crate,
  ADR 0029-conformant) ahead of the ADR 0032 routes.

## Related Decisions

ADR 0028 (schema columns this fills), ADR 0029 (crate), ADR 0034 (ONNX
embedder), ADR 0010 (Embedding capability routing), ADR 0032 (memory
HTTP surface — next consumer).

## Validation

~25 new tests: chain resolution incl. pin-never-falls-through; identity
write/filter/mismatch-warn/upsert; RRF purity (overlap boost, tie
stability, k-cap) + FTS sanitization; hybrid e2e (lexical + semantic
both surface; embedder=None still finds lexical); reembed plan/run.
Workspace 742 green. Live gate queued: keyless recall against a real
ONNX export once a fixture lands + training quiets.
