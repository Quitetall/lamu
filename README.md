# LAMU

**Local Agent Model Utility** — a single-process MCP-first daemon that auto-discovers your GGUF models, schedules them on a budgeted GPU, and serves them over MCP and OpenAI-compatible HTTP. Three speed tiers up to **106 t/s** on one RTX 4090. Python prototype, Rust drop-in.

```
                     ┌──────── lamu ────────┐
  Claude Code ─MCP─▶ │  router · scheduler  │ ─▶ llama.cpp / megakernel / DFlash
       agents        │  queue · reasoning   │      (per-backend spawn)
                     └────────┬─────────────┘
                              ▼
                          OpenAI HTTP
                          for everyone else
```

| Tier | Speed | Engine | Use |
|------|-------|--------|-----|
| **DFlash** (Lucebox DDTree) | **106 t/s** | matched-3.6 draft, 5.12 tok/step | one-shot, full GPU |
| **ngram-mod** (warm) | 49.5 t/s | hash-based speculation, no draft | always-on, 131K ctx |
| **megakernel** | 494 t/s | hand-written CUDA, Qwen3.5-0.8B | routing, agent tools |

Started from 2021 InferKit / GPT-2 nostalgia — the GPT-2 proxy is still in the registry, not dead code. Two upstream merges along the way ([Lucebox #89](https://github.com/Luce-Org/lucebox-hub/pull/89), [#94](https://github.com/Luce-Org/lucebox-hub/pull/94)).

---

## Quick Start

```bash
git clone https://github.com/Quitetall/lamu ~/local-llm
cd ~/local-llm
python3.12 -m venv .venv && uv pip install -e . --python .venv/bin/python

python -m lamu scan                   # discover GGUFs in ~/models
python -m lamu start                  # MCP daemon on stdio
python -m lamu serve [port=8020]      # OpenAI-compat HTTP
python -m lamu repl  [api_url]        # chat REPL
```

Rust drop-in (same CLI surface):

```bash
cd lamu-rs && cargo build --release
./target/release/lamu start            # same MCP behaviour, lower overhead
./target/release/lamu serve --port 8020
```

For the **106 t/s** DFlash one-shot:

```bash
just serve-fast "Write quicksort in Python"
```

---

## v2 Architecture

```
┌────────────────────────────────────────────────────────────┐
│  lamu daemon (single process)                              │
│                                                            │
│  ┌───────────┐  ┌────────────┐  ┌─────────────────────┐    │
│  │ MCP stdio │  │ OpenAI :*  │  │ CLI REPL (lamu repl)│    │
│  │ (primary) │  │ (compat)   │  │  → talks to daemon  │    │
│  └─────┬─────┘  └─────┬──────┘  └──────────┬──────────┘    │
│        │              │                    │               │
│  ┌─────▼──────────────▼────────────────────▼──────────┐    │
│  │  Router  — capability routing (chat/code/...)      │    │
│  │  Queue   — FIFO/LIFO/Priority, bounded concurrency │    │
│  │  Reasoning extractor — per-family <think> handling │    │
│  └────────────────────────┬───────────────────────────┘    │
│                           │                                │
│  ┌────────────────────────▼───────────────────────────┐    │
│  │  VRAM scheduler — bin-packing + LRU eviction       │    │
│  │  pinned models honoured · NVML-driven              │    │
│  └─┬──────────────┬──────────────┬─────────────┬──────┘    │
│    │              │              │             │           │
│  ┌─▼─────┐  ┌─────▼──────┐  ┌────▼────┐  ┌─────▼────┐      │
│  │llama  │  │megakernel  │  │ DFlash  │  │HF / ONNX │      │
│  │.cpp   │  │  (PyTorch) │  │ lucebox │  │ (future) │      │
│  └───────┘  └────────────┘  └─────────┘  └──────────┘      │
└────────────────────────────────────────────────────────────┘
```

Key v2 invariants:

- **MCP first.** OpenAI HTTP is a compat shim. The CLI also targets the daemon, not the backends directly.
- **Capabilities are requirements, not preferences.** `capabilities=["code"]` will load a code model, evicting LRU if needed. The router never silently downgrades.
- **`plan_query` dry-run.** Returns `{would_route_to, reason, loaded, would_evict}` for debugging agent loops.
- **Per-model request queue.** Concurrent agents calling the same model serialise on a configurable strategy (FIFO default). Set `LAMU_QUEUE_STRATEGY=priority` and pass `priority`/`origin` per request when ordering matters.
- **Reasoning extractor lives in the model entry.** `<think>...</think>` is buffered and stripped (or annotated) per family — Qwen3.5/3.6, DeepSeek, o1.

---

## MCP — Claude Code Integration

```jsonc
// ~/.claude.json
{
  "mcpServers": {
    "local-llm": {
      "type": "stdio",
      "command": "/home/YOU/local-llm/.venv/bin/python",
      "args": ["-m", "lamu", "start"],
      "cwd": "/home/YOU/local-llm"
    }
    // — or, drop-in Rust binary —
    // "command": "/home/YOU/local-llm/lamu-rs/target/release/lamu",
    // "args": ["start"]
  }
}
```

**Tools exposed:**

| Tool | Purpose |
|------|---------|
| `query` | Send prompt. `model=` overrides routing; `capabilities=[…]` enforces requirements; `priority`/`origin` flow through the queue; `include_reasoning=true` returns `<think>` as a structured field. |
| `plan_query` | Dry-run routing decision — no generation. |
| `list_models` | Registry + load status + capabilities. |
| `load_model` / `unload_model` | Manual VRAM control. |
| `vram_status` | Snapshot of allocation. |
| `scan_models` | Re-discover GGUFs on disk. |
| `queue_status` | Per-model queue depth + scheduling strategy. |

---

## OpenAI HTTP

Drop-in compat:

```bash
curl http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":1000,"stream":true}'
```

Streaming, models list, health — all standard. Reasoning is stripped from `content` and exposed in `reasoning_content` (Qwen extension). Both Python (FastAPI) and Rust (axum) implementations validated for end-to-end SSE: identical chunk format, identical `[DONE]` terminator.

---

## Model Registry

`python -m lamu scan` walks `~/models/`, parses GGUF headers, and emits `config/models.yaml`:

```yaml
- name: qwen3.6-27b-uncensored-heretic-v2-q4_k_m
  path: ~/models/qwen3.6-27b-heretic/Qwen3.6-27B-uncensored-heretic-v2-Q4_K_M.gguf
  format: gguf
  backend: llama_cpp
  arch: qwen35
  params_b: 27.6
  quant: Q4_K_M
  vram_mb: 17358
  context_max: 262144
  capabilities: [chat, code, reasoning, long_context]
  reasoning_marker: { open_tag: "<think>", close_tag: "</think>", family: qwen35 }
  speculative:
    draft_path: ~/models/qwen3.6-dflash-gguf/dflash-3.6-q4km.gguf
    method: dflash
    draft_max: 8
```

Backends: `llama_cpp`, `megakernel`, `dflash`, `dflash_lucebox` — chosen per-entry. Adding a new backend is a single file in `lamu/backends/` (or `lamu-rs/lamu-core/src/backends/`) plus one `make_backend` arm.

---

## Three Speed Tiers (v1 perf legacy, still authoritative)

RTX 4090, 24 GB VRAM, Qwen3.6-27B-uncensored-heretic-v2 Q4_K_M:

| Method | Speed | Acceptance | Notes |
|--------|-------|-----------|-------|
| **Lucebox DFlash + DDTree** | **106 t/s** | 32%, 5.12 tok/step | Matched 3.6 draft; PR [#94](https://github.com/Luce-Org/lucebox-hub/pull/94) |
| llama.cpp DFlash PR | 82 t/s | 77.9%, draft-max=8 | GGUF Q4_K_M draft |
| ngram-mod (warm) | 49.5 t/s | pattern matching | no draft model |
| ngram-mod (cold) | 9.8 t/s | first request | |
| 0.8B megakernel | 494 t/s | n/a | hand-written CUDA |

Q4_K_M draft outperforms F16 (77.6 vs 72.7 t/s) — bandwidth beats accuracy on the draft path.

---

## Testing

`pytest` with stubbed heavy deps so the unit layer runs CPU-only (no torch/transformers/llama_cpp imported):

```bash
pytest tests/ -q
# → 264 passed, 15 deselected (GPU-marked)

cargo test --workspace
# → 50 passed across 9 crates
```

Layout:

```
tests/
├── unit/        — 250 tests, modules stubbed at conftest level
│   ├── core/    — registry, scheduler, router, reasoning, types, health, supervisor
│   ├── backends/, mcp/, api/, daemon/, cli/
│   └── server/, agents/, scripts/, web/
└── integration/ — 14 tests, real subprocesses
    ├── test_backend_death.py        — process kill, scheduler reconciles
    ├── test_oom_quarantine.py       — VRAM exhausted → quarantine path
    ├── test_bad_registry.py         — corrupt YAML, missing files
    ├── test_no_hang.py              — load/unload/query never blocks indefinitely
    └── test_concurrent_health.py    — N agents probing health at once
```

Heavy modules (`torch`, `transformers`, `llama_cpp`, `langchain*`, `chainlit`, …) are replaced with `_StubModule` instances at conftest import — the runtime never touches them. Real subprocesses are guarded by `no_real_subprocess`. `nvidia-smi` is intercepted by the `mock_nvidia_smi` fixture which simulates VRAM state, PIDs, timeouts, and failures.

GGUF tests use a synthesised binary from `make_gguf_bytes(arch, file_type, truncate=, bad_magic=)` — covers happy path + corruption.

---

## Build Requirements

- **GPU:** NVIDIA RTX 4090 (24 GB) or larger
- **OS:** Linux (Arch / CachyOS tested)
- **CUDA:** 13.2 with **gcc-14** as host compiler (`CUDAHOSTCXX=g++-14`). gcc-16 + nvcc 13.2 do not link.
- **Python:** 3.12+
- **Rust:** 1.85+ (edition 2024)
- **Tools:** `just`, `cmake`, `git`, `uv`

---

## Commands

```bash
# v2 daemon
python -m lamu scan|start|status|serve|repl
lamu  scan|start|status|serve|repl                  # rust binary, same surface

# v1 servers (still wired through justfile)
just swap 3.6 | 3.5 | dflash                        # rotate :8020 / :8000
just serve-fast ["prompt"]                          # 106 t/s DFlash, optional one-shot
just serve-megakernel                               # 494 t/s on :8001
just status                                         # all endpoints

# Chat
llm                                                 # legacy direct REPL
python -m lamu repl                                 # v2 daemon-routed REPL

# Agent swarm + training
just swarm "task" /path/to/repo
just bench-swarm
just train-status | train

# Tests
pytest tests/ -q
cargo test --workspace
```

---

## Wiki

13 pages of hard-won optimization knowledge in `wiki/pages/`:

`dflash-speculative.md` · `build-requirements.md` · `262k-context.md` · `ngram-speculation.md` · `vram-budget.md` · `eagle-training.md` · `eagle-cpp-integration.md` · `mcp-setup.md` · `model-selection.md` · `serving-engine.md` · `token-efficiency.md` · `training-loop.md` · `vllm-limitations.md`.

Knowledge graph (~1,000 nodes) in `graphify-out/graph.html`. Query with `/graphify query "<question>"`.

---

## Open Source Contributions

| PR | Repo | Status | Impact |
|----|------|--------|--------|
| [#89](https://github.com/Luce-Org/lucebox-hub/pull/89) | Luce-Org/lucebox-hub | **Merged** | Fixed `conv_input_cache` crash on all 24 GB GPUs |
| [#94](https://github.com/Luce-Org/lucebox-hub/pull/94) | Luce-Org/lucebox-hub | Submitted | Qwen3.6 SWA draft support → 57% speedup |

---

## Philosophy

Wanted GPT-2 running locally like InferKit in 2021. Now running a 27B uncensored model at 106 t/s with speculative decoding, agent swarms, MCP integration, and a Rust port — all on one consumer GPU.

The config is opinionated. The architecture isn't. Every layer is swappable: model, backend, transport, framework. When something better lands, change one path.

---

## License

MIT
