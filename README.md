# LAMU

**Local Agent Model Utility** — single-process MCP-first daemon. Auto-discovers GGUF models, schedules them on a budgeted GPU, serves them over MCP and OpenAI-compatible HTTP. Three speed tiers up to **106 t/s** on one RTX 4090.

```
                     ┌──────── lamu ────────┐
 Claude Code ─MCP─▶ │  router · scheduler  │ ─▶ llama.cpp / megakernel / DFlash
       agents        │  queue · reasoning   │      (per-backend spawn)
                     └────────┬─────────────┘
                              ▼
                          OpenAI HTTP
                          for everything else
```

| Tier | Speed | Engine | Use |
|------|-------|--------|-----|
| **DFlash** (Lucebox DDTree) | **106 t/s** | matched-3.6 draft, 5.12 tok/step | one-shot, full GPU |
| **ngram-mod** (warm) | 49.5 t/s | hash-based speculation, no draft | always-on, 131K ctx |
| **megakernel** | 494 t/s | hand-written CUDA, Qwen3.5-0.8B | routing, agent tools |

Started from 2021 InferKit / GPT-2 nostalgia — the GPT-2 proxy is still in the registry, not dead code. Two upstream merges along the way ([Lucebox #89](https://github.com/Luce-Org/lucebox-hub/pull/89), [#94](https://github.com/Luce-Org/lucebox-hub/pull/94)).

---

## Quick Start

LAMU ships a single canonical binary, `lamu` (Rust). Install once and use it from anywhere on `$PATH`.

```bash
git clone https://github.com/Quitetall/lamu ~/local-llm
cd ~/local-llm

just install          # cargo install --path lamu-rs/lamu-cli --locked
lamu pull qwen36-27b  # ~16 GB GGUF from HuggingFace → ~/models/
lamu scan             # discover GGUFs → config/models.yaml
lamu serve &          # OpenAI HTTP on :8020 (background)
lamu run heretic      # one-shot: resolve, drop into chat
```

Or just type `lamu` — opens a ratatui dashboard with model list (j/k navigate, Enter chats), live VRAM gauge, MCP/HTTP/Bifrost status. First-run-aware: empty registry triggers a `[Y/n]` to download Qwen3.6-27B; if `LAMU_GATEWAY_URL` set + Bifrost down, prompts to `just serve-bifrost`.

Full Ollama-shaped surface:

| Command | Effect |
|---------|--------|
| `lamu` | TUI dashboard |
| `lamu run <model>` | Drop into chat with a model (substring match) |
| `lamu pull <id> [--quant Q4_K_M]` | Download a GGUF from HuggingFace; auto re-scan |
| `lamu show <model>` | Print full registry entry as YAML |
| `lamu rm <model>` | Remove from registry + delete file on disk |
| `lamu list` / `lamu scan` | Re-discover models |
| `lamu status` | VRAM + which port answers |
| `lamu start` | MCP daemon on stdio (Claude Code) |
| `lamu serve [port]` | OpenAI HTTP (default :8020) |
| `lamu repl [url]` | Chat REPL against a running serve |

That's the whole onboarding. `lamu` is your interface; everything else is plumbing.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  lamu daemon (single process, single binary)               │
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
│  │  Health + Supervisor — restart-with-backoff        │    │
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

Invariants:
- **MCP first.** OpenAI HTTP is a compat shim. The CLI also targets the daemon, not the backends directly.
- **Capabilities are requirements, not preferences.** `capabilities=["code"]` will load a code model, evicting LRU if needed. The router never silently downgrades.
- **`plan_query` dry-run.** Returns `{would_route_to, reason, loaded, would_evict}` for debugging agent loops.
- **Per-model request queue.** Concurrent agents calling the same model serialise on a configurable strategy (FIFO default). Set `LAMU_QUEUE_STRATEGY=priority` and pass `priority` / `origin` per request when ordering matters.
- **Reasoning extractor lives in the model entry.** `<think>...</think>` is buffered and stripped (or annotated) per family — Qwen3.5/3.6, DeepSeek, o1.
- **Backend death is loud, not silent.** Health state machine (HEALTHY → DEGRADED → DEAD → QUARANTINED) plus Supervisor with 1s/2s/4s backoff, structured JSON events to stderr or `$LAMU_EVENT_LOG`.

---

## MCP — Claude Code Integration

```jsonc
// ~/.claude.json
{
  "mcpServers": {
    "local-llm": {
      "type": "stdio",
      "command": "lamu",
      "args": ["start"]
    }
  }
}
```

Reload Claude Code, then `/mcp` should show `local-llm` connected. Tools exposed:

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

`lamu serve` boots the FastAPI/axum compat layer. Drop-in for any OpenAI client.

**Bifrost passthrough (optional):** Set `LAMU_GATEWAY_URL=http://localhost:8080/v1` and `lamu serve` forwards every chat completion through Bifrost (`just serve-bifrost`) instead of hitting the backend directly. Bifrost dispatches by `provider/model` id (e.g. `qwen/qwen3.6-27b-uncensored` → `:8020`, `dflash/luce-dflash` → `:8000`, `anthropic/claude-opus-4-7` → cloud). 1.67% latency cost, gain a unified cloud + local OpenAI surface plus Bifrost's logging/key-rotation. Default off; opt in when you want it.

Drop-in for any OpenAI client:

```bash
curl http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":1000,"stream":true}'
```

Streaming, models list, health — all standard. Reasoning is stripped from `content` and exposed in `reasoning_content` (Qwen extension).

Observability:

- `GET /metrics` — Prometheus text. `lamu_requests_total`, `lamu_request_duration_seconds`, `lamu_tokens_generated_total`, `lamu_vram_used_mb`, `lamu_queue_depth`, `lamu_backend_health_state`, `lamu_backend_restarts_total`.
- `GET /health` — `{"status":"ok","models_loaded":N}` for liveness.
- W3C `traceparent` on requests gets propagated through `lamu`'s structured event stream (mid-16 hex of the traceid as the internal trace_id).
- `LAMU_EVENT_LOG=/path/to/jsonl` appends every event to a file alongside stderr.

---

## Harnesses — point any client at lamu

Lamu speaks three API flavors on the same port:

| Flavor | Routes | Clients |
|--------|--------|---------|
| OpenAI | `/v1/chat/completions`, `/v1/models` | Codex, Cursor, Aider, Continue, pi |
| Anthropic | `/v1/messages` (SSE + `tool_use`) | Claude Code, Crush, Hermes |
| Ollama | `/api/chat`, `/api/tags` (NDJSON) | AnythingLLM, Open WebUI |

The default model is whichever `config/models.yaml` entry has `main: true`. Aliases `default` / `main` / `lamu` all resolve there, so harnesses don't need a model name configured.

Registered harnesses live in `config/harnesses.yaml`. Launch one with the right env wired up:

```bash
just open                # default (claude-code)
just open codex          # named entry
just open list           # show all configured harnesses
```

Per-harness API flavor + cmd + extra env is yaml-only — no rebuild to add a new one. Full reference: [`wiki/pages/harness-setup.md`](wiki/pages/harness-setup.md).

Per-request `enable_thinking: false` turns off Qwen3.6 reasoning (works on all three surfaces + MCP `query` tool).

---

## Model Registry

`lamu scan` walks `~/models/`, parses GGUF headers, writes `config/models.yaml`:

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

Backends: `llama_cpp`, `megakernel`, `dflash`, `dflash_lucebox` — chosen per-entry. Adding a new backend is one file in `lamu-rs/lamu-core/src/backends/` (mirrored in `lamu/backends/`) plus one `make_backend` arm.

---

## Three Speed Tiers (perf legacy, still authoritative)

RTX 4090, 24 GB VRAM, Qwen3.6-27B-uncensored-heretic-v2 Q4_K_M:

| Method | Speed | Acceptance | Notes |
|--------|-------|-----------|-------|
| **Lucebox DFlash + DDTree** | **106 t/s** | 32%, 5.12 tok/step | Matched 3.6 draft; PR [#94](https://github.com/Luce-Org/lucebox-hub/pull/94) |
| llama.cpp DFlash PR | 82 t/s | 77.9%, draft-max=8 | GGUF Q4_K_M draft |
| ngram-mod (warm) | 49.5 t/s | pattern matching | no draft model |
| ngram-mod (cold) | 9.8 t/s | first request | |
| 0.8B megakernel | 494 t/s | n/a | hand-written CUDA |

Q4_K_M draft outperforms F16 (77.6 vs 72.7 t/s) — bandwidth beats accuracy on the draft path.

For the **106 t/s DFlash one-shot** run via the legacy stack: `just serve-fast "Write quicksort in Python"` (requires the custom DFlash llama.cpp branch built — see [`wiki/pages/dflash-speculative.md`](wiki/pages/dflash-speculative.md)).

---

## Hacking on LAMU — the Python prototype

The Python package at `lamu/` is the iteration surface. Every Rust module in `lamu-rs/` started as a Python prototype that got translated mechanically once the design stabilised. The two run in lock-step — cross-language MCP contract tests in `tests/contract/` lock the wire format.

Use Python when:
- You're sketching a new module.
- You want a stack trace.
- You're debugging health / scheduler / queue behaviour interactively.

Use Rust (`lamu`) for everything else. Mirror surface is identical:

```bash
python -m lamu scan|status|start|serve|repl     # prototype
lamu               scan|status|start|serve|repl  # canonical
```

---

## Build Requirements

- **GPU:** NVIDIA RTX 4090 (24 GB) or larger
- **OS:** Linux (Arch / CachyOS tested)
- **CUDA:** 13.2 with **gcc-14** as host compiler (`CUDAHOSTCXX=g++-14`). gcc-16 + nvcc 13.2 do not link.
- **Rust:** 1.85+ (edition 2024) — `cargo install` lands `lamu` at `~/.cargo/bin/lamu`.
- **Python:** 3.12+ (only needed for the prototype + agents)
- **Tools:** `just`, `cmake`, `git`, `uv`

---

## Testing

```bash
pytest tests/ -q          # 288 unit + 14 integration, heavy deps stubbed
cargo test --workspace    # 56 Rust tests across 9 crates
just test-contract        # Python ↔ Rust MCP wire-format parity
ruff check lamu           # strict on lamu/, soft on legacy paths
```

CI gates on coverage (`fail_under = 70`), strict ruff over `lamu/`, full Python + Rust suites, and the cross-language contract diff.

`tests/conftest.py` stubs `torch`, `transformers`, `llama_cpp`, `langchain*`, `chainlit`, etc. with `_StubModule` instances at import time — the unit layer never touches a GPU. `mock_nvidia_smi` simulates VRAM/PIDs/failures; `no_real_subprocess` is autouse-guarded so a stray Popen can't escape a test.

---

## Legacy v1 stack

The script-driven v1 workflow (swap a model into `:8020`, sidecar a small one into `:8001`, run Bifrost on `:8080`) lives under `legacy/`. See [`legacy/README.md`](legacy/README.md) for the full inventory and what each script does.

`just` exposes both:

```bash
just install            # v3 — lamu binary
just start              # v3 — MCP daemon
just serve              # v3 — OpenAI HTTP
just repl               # v3 — chat REPL

just start-v1           # v1 — full Qwen3.6 + megakernel + Bifrost stack
just swap 3.6 | 3.5     # v1 — rotate model on :8020
just sidecar fast|lobo  # v1 — small sidecar on :8001
just serve-fast "..."   # v1 — DFlash 106 t/s one-shot
```

Bifrost (`:8080`) is dead on the v3 request path — kept under `scripts/serve-bifrost.sh` only because [`wiki/pages/bifrost-bench.md`](wiki/pages/bifrost-bench.md) hasn't been run yet to settle whether it's worth keeping. Run `bash scripts/bench-bifrost.sh` to decide.

---

## Wiki

13 pages in `wiki/pages/`:

`dflash-speculative.md` · `build-requirements.md` · `262k-context.md` · `ngram-speculation.md` · `vram-budget.md` · `eagle-training.md` · `eagle-cpp-integration.md` · `mcp-setup.md` · `harness-setup.md` · `model-selection.md` · `serving-engine.md` · `token-efficiency.md` · `training-loop.md` · `vllm-limitations.md` · `bifrost-bench.md`.

Knowledge graph (~1,600 nodes, 162 communities) in `graphify-out/graph.html`. Query with `/graphify query "<question>"`.

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
