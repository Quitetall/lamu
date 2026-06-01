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
| [0012](./0012-minimal-bearer-auth.md) | Minimal single-token HTTP bearer auth for lamu-api | Accepted |
| [0013](./0013-at-rest-key-encryption-deferred.md) | At-rest encryption of cloud keys / API token deferred | Accepted |
| [0014](./0014-single-gpu-and-root-paths-intentional.md) | Single-GPU + ~/local-llm path assumptions are intentional | Accepted |
| [0015](./0015-cookbook-roofline-scoring-engine.md) | Cookbook roofline + composite scoring engine (ported from hwfit) | Accepted |
| [0016](./0016-backend-orchestrator-byo-frontend.md) | LAMU is a backend orchestrator; bring your own frontend via broad API compat | Accepted |

The newest ADR is the most authoritative for a given topic; check
`Status: Superseded by NNNN` on older entries.
