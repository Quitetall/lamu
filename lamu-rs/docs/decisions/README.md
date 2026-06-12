# Architecture Decision Records (ADRs)

Each significant architectural decision in LAMU is captured as a numbered
Markdown file in this directory. ADRs answer the question "**why** is the
system this way?" — the kind of context that's obvious in the moment, easy
to forget six months later, and impossible to recover after the maintainers
turn over.

## When to write an ADR

Write one when the decision:

- Affects multiple components (architecture, data flow, API surface)
- Closes off a future direction ("we considered X, here's why we don't do X")
- Has a non-obvious rationale (the implementation alone doesn't explain *why*)
- Would prompt a reviewer to ask "did you consider...?"

Don't write one for:

- Implementation details (those go in code comments)
- Bug fixes (those go in commit messages)
- Reversible operational tweaks (preset values, log paths)

## How to write an ADR

Copy [`TEMPLATE.md`](./TEMPLATE.md) to `NNNN-short-title.md` where `NNNN` is
the next available number, fill it in, commit. ADRs are immutable once
accepted — to revisit a decision, write a new ADR that supersedes the old
one and update the old one's `## Status` line to point at it.

## How to find an ADR

```bash
# Browse chronologically (latest first)
ls decisions/ | sort -r | head

# Search for a topic
grep -l 'eviction' decisions/

# Find what superseded a deprecated decision
grep -l 'Supersedes 0005' decisions/
```

## Index

| #    | Title                                                                            | Status   |
| ---- | -------------------------------------------------------------------------------- | -------- |
| [0001](./0001-mcp-first-orchestration-http-thin-compat-shim.md) | MCP-first orchestration; HTTP serve as a thin compat shim | Accepted |
| [0002](./0002-lean-rust-backend.md) | Keep LAMU a lean Rust workspace, not a batteries-included framework | Accepted |
| [0003](./0003-vram-scheduler-with-modality-tiered-eviction.md) | Single-GPU NVML scheduler with modality-tiered eviction | Accepted |
| [0004](./0004-managed-subprocess-backends-with-unconditional-pdeathsig.md) | Manage model backends as hardened child processes with unconditional PDEATHSIG | Accepted |
| [0005](./0005-bind-loopback-by-default.md) | Bind 127.0.0.1 by default | Accepted |
| [0006](./0006-http-path-never-auto-evicts.md) | HTTP serving path never auto-evicts; eviction is an MCP-only operation | Accepted |
| [0007](./0007-unified-cloud-routing-anthropic-vs-openai-compat.md) | Unified cloud routing — Anthropic wire format vs OpenAI-compat for everything else | Accepted |
| [0008](./0008-headless-council-instead-of-compare-ui.md) | Provide multi-model comparison as a headless judged council, not a compare UI | Accepted |
| [0009](./0009-confined-media-output-paths.md) | Confine media output paths to a per-modality dir | Accepted |
| [0010](./0010-capability-modality-routing-embedding-never-chat-routed.md) | Route by capability subset AND modality; never chat-route embedding or non-LLM models | Accepted |
| [0011](./0011-structural-untrusted-content-envelope.md) | Structural prompt-injection boundary — one untrusted-content envelope | Accepted |
| [0012](./0012-minimal-bearer-auth.md) | Minimal single-token HTTP bearer auth for lamu-api | Superseded by 0018 |
| [0013](./0013-at-rest-key-encryption-deferred.md) | At-rest encryption of cloud keys / API token deferred | Accepted |
| [0014](./0014-single-gpu-and-root-paths-intentional.md) | Single-GPU (superseded by 0017) + ~/local-llm paths (in force) | Partly superseded |
| [0015](./0015-cookbook-roofline-scoring-engine.md) | Cookbook roofline + composite scoring engine (ported from hwfit) | Accepted |
| [0016](./0016-backend-orchestrator-byo-frontend.md) | LAMU is a backend orchestrator; bring your own frontend via broad API compat | Accepted |
| [0017](./0017-multi-gpu-device-pool.md) | Multi-GPU device pool + best-fit placement + opt-in tensor-parallel sharding | Accepted |
| [0018](./0018-multi-user-per-token-identity.md) | Multi-user — per-token identity, per-user memory namespacing, quotas | Accepted |
| [0019](./0019-cloud-model-catalog-sync.md) | Cloud-model catalog auto-sync from OpenRouter + per-provider /v1/models | Accepted |
| [0020](./0020-scale-testing-strategy.md) | Scale-testing strategy — ignore-gated harness tiers; HTTP has no request queue (single-flight is load-only) | Accepted |
| [0021](./0021-context-occupancy-and-self-compaction.md) | Un-fakeable context-occupancy signal and self-compaction tools | Accepted |
| [0022](./0022-http-serving-baseline.md) | First HTTP serving micro-baseline (warm, short, single-model) | Accepted |
| [0023](./0023-module-architecture.md) | Backend / module / frontend architecture | Accepted |
| [0024](./0024-mcp-serial-dispatch-loop.md) | MCP dispatch loop stays serial; concurrency lives in tools and the per-model queue | Accepted |
| [0025](./0025-registry-out-of-work-tree.md) | Live model registry moves to the user data dir, out of the git work tree | Accepted |
| [0026](./0026-backend-kind-string-dispatch.md) | `backend_kind` string dispatch key + composition-root drift test | Accepted |
| [0027](./0027-typed-toolctx-error-seam.md) | Typed `ToolCtxError` at the ToolCtx seam; "error:" strings stay wire-only | Accepted |
| [0037](./0037-provider-grade-serving.md) | Provider-grade serving: structured reasoning on all bridges, prefix-cache exposure, caps discovery | Accepted |

The newest ADR is the most authoritative for a given topic; check
`Status: Superseded by NNNN` on older entries.
