# LAMU

**Local Agent Model Utility** — 106 tokens/second on a single RTX 4090. No cloud, no API keys, no censorship.

Three models running simultaneously on 24 GB:

| Model | Speed | Use |
|-------|-------|-----|
| **Qwen3.6-27B** uncensored | **106 t/s** (DFlash) · 49 t/s (ngram) | Complex reasoning, 131K context |
| **Qwen3.5-0.8B** megakernel | **494 t/s** | Instant routing, agent tools |
| **Qwen3.5-27B** | 49 t/s (swap) | Alternative reasoning model |

Built from wanting to recreate the 2021 InferKit experience. Evolved into a full inference stack with two merged upstream contributions.

---

## Quick Start

```bash
# Clone
git clone https://github.com/Quitetall/lamu ~/local-llm
cd ~/local-llm

# Create environment
python3.12 -m venv .venv

# Download Qwen3.6-27B uncensored (~16 GB)
just setup-qwen36

# Start production server (49 t/s ngram-mod, always-on)
just swap 3.6

# Chat
llm
```

For **106 t/s** DFlash speculative decoding:
```bash
# One-time: build llama.cpp DFlash branch + download draft model
cd ~/llama.cpp && git checkout dflash-pr
cd build && cmake --build . --target llama-speculative-simple -j$(nproc)

# Run (one-shot, uses full GPU)
just serve-fast "Write quicksort in Python"
```

---

## Three Speed Tiers

### Tier 1: Megakernel — 494 t/s
Qwen3.5-0.8B with hand-written CUDA kernels from [Lucebox](https://github.com/Luce-Org/lucebox-hub). Runs alongside the 27B model (2.7 GB VRAM).

```bash
just serve-fast    # starts on :8001
```

### Tier 2: ngram-mod — 49 t/s (warm)
Qwen3.6-27B with hash-based speculative decoding. Zero overhead, no draft model. Production server with OpenAI API.

```bash
just swap 3.6      # starts on :8020
```

### Tier 3: DFlash — 106 t/s
Block-diffusion speculative decoding with matched Qwen3.6-27B-DFlash draft. 5.12 tokens committed per step.

```bash
just serve-fast "your prompt here"    # one-shot (full GPU)
```

---

## Model Switching

```bash
just swap 3.6       # Qwen3.6-27B heretic uncensored
just swap 3.5       # Qwen3.5-27B
just swap status    # show what's running
```

In the chat REPL:
```
llm
> /model fast       # switch to 0.8B (494 t/s)
> /model smart      # switch to 27B
> /model dflash     # switch to DFlash 27B
```

---

## Claude Code Integration (MCP)

LAMU exposes local models as tools for Claude Code.

**Setup** (add to `~/.claude.json`):
```json
{
  "mcpServers": {
    "local-llm": {
      "type": "stdio",
      "command": "/home/YOUR_USER/local-llm/.venv/bin/python",
      "args": ["/home/YOUR_USER/local-llm/server/mcp_qwen.py"]
    }
  }
}
```

**Tools:**
- `query_local_llm` — send prompts to local model (default 27B, `model="fast"` for 0.8B)
- `list_local_models` — discover running models

**Routing:**
```
model="fast"     → 0.8B megakernel (494 t/s, simple tasks)
model="dflash"   → DFlash 27B (when running)
default          → Qwen3.6-27B (complex reasoning)
```

---

## API Reference

OpenAI-compatible on every model:

```bash
# 27B (production)
curl http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":1000}'

# 0.8B (instant)
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":200}'
```

**Streaming:** Add `"stream": true` to any request.

**Python SDK:**
```python
from server.client import LocalLLM

llm = LocalLLM()
response = llm.chat("explain quicksort")
for token in llm.stream("write a long function"):
    print(token, end="", flush=True)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  SURFACES                                            │
│  llm (REPL) │ MCP (Claude Code) │ OpenAI API │ SDK  │
└──────────────────────┬───────────────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         │             │              │
  ┌──────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
  │ Qwen3.6 27B │ │ Qwen3.5  │ │  DFlash    │
  │  :8020      │ │  0.8B    │ │  106 t/s   │
  │ ngram-mod   │ │  :8001   │ │  one-shot  │
  │ 131K ctx    │ │ 494 t/s  │ │  DDTree    │
  └─────────────┘ └──────────┘ └────────────┘
         │
  ┌──────▼──────────┐
  │  Agent Swarm    │
  │  plan → work    │
  │  → test → review│
  └─────────────────┘
```

---

## Benchmarks

RTX 4090, 24 GB VRAM, Qwen3.6-27B-uncensored-heretic-v2 Q4_K_M:

| Method | Speed | Acceptance | Notes |
|--------|-------|-----------|-------|
| **Lucebox DFlash+DDTree** | **106 t/s** | 32%, 5.12 tok/step | Matched 3.6 draft |
| llama.cpp DFlash PR | 82 t/s | 77.9%, draft-max=8 | GGUF Q4_K_M draft |
| ngram-mod (warm) | 49.5 t/s | Pattern matching | No draft model |
| ngram-mod (cold) | 9.8 t/s | First request | |
| 0.8B megakernel | 494 t/s | N/A | Different model |

Q4_K_M draft outperforms F16 draft (77.6 > 72.7 t/s). Bandwidth beats accuracy.

---

## Full Manual

### Build Requirements

- **GPU:** NVIDIA RTX 4090 (24 GB) or similar
- **OS:** Linux (Arch/CachyOS tested)
- **CUDA:** 13.2 with **gcc-14** as host compiler (`CUDAHOSTCXX=g++-14`)
- **Python:** 3.12+
- **Tools:** `just`, `cmake`, `git`

GCC 16 + NVCC 13.2 are incompatible. Always use gcc-14 for CUDA builds.

### All Commands

```bash
# Server management
just swap 3.6           # Qwen3.6-27B ngram-mod on :8020
just swap 3.5           # Qwen3.5-27B on :8020
just swap dflash        # DFlash lucebox on :8000
just swap status        # show what's running
just serve-fast         # DFlash one-shot (106 t/s)
just status             # all endpoints

# Chat
llm                     # interactive REPL
llm "your question"     # one-shot

# Agent swarm
just swarm "task" /path/to/repo
just bench-swarm        # run benchmarks

# Training
just train-status       # check collected data
just train              # QLoRA fine-tuning

# Model setup
just setup-qwen36       # download Qwen3.6-27B
```

### Wiki

13 pages of hard-won optimization knowledge at `wiki/pages/`:

- `dflash-speculative.md` — DFlash setup, benchmarks, both implementations
- `build-requirements.md` — gcc-14 requirement, clang status
- `262k-context.md` — how to achieve 262K on 24 GB
- `ngram-speculation.md` — ngram-mod tuning
- `vram-budget.md` — what fits where
- `eagle-training.md` — EAGLE v3 experiments (archived)

### Knowledge Graph

Graphify builds a navigable graph of the codebase (321 nodes, 424 edges, 25 communities):

```bash
# View in browser
open graphify-out/graph.html

# Query
/graphify query "how does DFlash connect to MCP"
```

---

## Open Source Contributions

| PR | Repo | Status | Impact |
|---|------|--------|--------|
| [#89](https://github.com/Luce-Org/lucebox-hub/pull/89) | Luce-Org/lucebox-hub | **Merged** | Fixed conv_input_cache crash for all 24 GB GPUs |
| [#94](https://github.com/Luce-Org/lucebox-hub/pull/94) | Luce-Org/lucebox-hub | Submitted | Qwen3.6 SWA draft support → 57% speedup |

---

## Philosophy

Started from wanting to run GPT-2 locally like InferKit in 2021. Now running a 27B uncensored model at 106 tokens per second with speculative decoding, agent swarms, and MCP integration — all on a single consumer GPU.

The config is opinionated. The architecture isn't. Every layer is swappable: model, engine, gateway, framework integration. When something better comes along, change one path.

---

## License

MIT
