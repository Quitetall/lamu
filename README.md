# LAMU

**Local Agent Model Utility** — a framework for running AI models as local agents on consumer hardware. Plug into any tool, any framework, zero cloud dependency.

Currently configured for Qwen3.6-27B Uncensored on RTX 4090 with 262K context, but designed to be model-agnostic and futureproof. Swap the model, keep the framework.

## What it does

Type `llm` and talk to a 27B parameter model running on your GPU. No internet, no API keys, no rate limits, no censorship.

```
$ llm "write a fibonacci function in C"
```

That same model is simultaneously available to:
- **Claude Code** via MCP (`query_local_llm` tool)
- **Any HTTP client** via OpenAI-compatible API (`:8080/v1/chat/completions`)
- **Python** via `from server.client import chat`
- **Web browser** via Chainlit UI (`:7860`)
- **Any framework** that speaks OpenAI protocol

---

## Table of Contents

- [Quick Start](#quick-start)
- [Full Manual](#full-manual)
  - [Installation](#installation)
  - [Daily Usage](#daily-usage)
  - [Claude Code Integration (MCP)](#claude-code-integration-mcp)
  - [Python SDK](#python-sdk)
  - [API Reference](#api-reference)
  - [Agent Swarm](#agent-swarm)
  - [Training Pipeline](#training-pipeline)
  - [Image & Video Generation](#image--video-generation)
  - [Model Management](#model-management)
  - [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [Philosophy](#philosophy)
- [Comparison](#comparison)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Quitetall/lamu ~/lamu
cd ~/lamu

# 2. Download model (~16 GB, takes a few minutes)
just setup-qwen36

# 3. Start
just start

# 4. Chat
llm "what is quicksort?"
```

That's it. You now have a local AI running with 262K context.

---

## Full Manual

### Installation

**Prerequisites:**
- NVIDIA GPU with 24GB+ VRAM (RTX 4090 recommended)
- Linux (tested on Arch/CachyOS)
- Python 3.12+
- `just` command runner (`pacman -S just` or `cargo install just`)
- ~50GB free disk space

**Step 1: Clone and enter the repo**
```bash
git clone https://github.com/Quitetall/lamu ~/lamu
cd ~/lamu
```

**Step 2: Create the Python environment**
```bash
python3.12 -m venv .venv
uv pip install llama-cpp-python[server] rich --python .venv/bin/python
```
(If you want CUDA-accelerated inference, install with `CMAKE_ARGS="-DGGML_CUDA=on"`)

**Step 3: Download a model**
```bash
just setup-qwen36    # Qwen3.6-27B Uncensored (~16 GB)
```

**Step 4: Start the stack**
```bash
just start
```

**Step 5: Add shell aliases** (add to `~/.zshrc` or `~/.bashrc`):
```bash
alias j='cd ~/lamu && just'
alias llm='bash ~/lamu/scripts/chat.sh'
alias llm-start='bash ~/lamu/scripts/start.sh'
alias llm-stop='bash ~/lamu/scripts/stop.sh'
alias llm-status='cd ~/lamu && just status'
```

**Step 6: Enable auto-start on boot**
```bash
systemctl --user enable local-llm.service
```

---

### Daily Usage

**Interactive chat:**
```bash
llm
```
Opens a REPL. Type messages, get responses with markdown rendering. Commands inside the REPL:
- `/model <name>` — switch model
- `/models` — list available models
- `/status` — show what's running
- `/clear` — clear conversation history
- `/quit` — exit

**One-shot questions:**
```bash
llm "explain the difference between TCP and UDP"
llm "write a rust function that parses JSON"
llm -m dflash/luce-dflash "hello"    # pick a specific model
```

**Check status:**
```bash
just status
```
Shows which services are up/down with checkmarks.

**Stop everything:**
```bash
just stop
```

---

### Claude Code Integration (MCP)

LAMU ships an MCP server that gives Claude Code (or any MCP client) direct access to your local model.

**Setup** (one-time, already done if you followed install):

Add to `~/.claude.json`:
```json
{
  "mcpServers": {
    "lamu": {
      "type": "stdio",
      "command": "/home/YOUR_USER/lamu/.venv/bin/python",
      "args": ["/home/YOUR_USER/lamu/server/mcp_qwen.py"]
    }
  }
}
```

Restart Claude Code. You now have two tools:
- **`query_local_llm`** — send any prompt to your local model
- **`list_local_models`** — see what's running

**How Claude Code uses it:**
- Offloads bulk code generation to the free local model
- Gets second opinions on implementation approaches
- Drafts code locally, Claude reviews — 70-80% of tokens free

**Parameters for `query_local_llm`:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| prompt | string | required | The prompt to send |
| model | string | qwen/qwen3.6-27b-uncensored | Which model |
| system | string | "" | System prompt |
| max_tokens | int | 4096 | Max response length |
| temperature | float | 0.3 | Randomness (0=deterministic, 2=creative) |

---

### Python SDK

Use LAMU from any Python script:

```python
from server.client import LocalLLM

llm = LocalLLM()

# Simple chat
response = llm.chat("explain quicksort")
print(response)

# With system prompt
response = llm.chat(
    "refactor this function",
    system="You are a senior Python developer. Be concise.",
)

# Stream tokens
for token in llm.stream("write a long essay about AI"):
    print(token, end="", flush=True)

# Check what's available
print(llm.models())   # ['qwen/qwen3.6-27b-uncensored']
print(llm.health())   # {'bifrost': 'up', 'qwen36': 'up', ...}

# Multi-turn conversation
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "A programming language."},
    {"role": "user", "content": "Show me hello world in it."},
]
response = llm.chat_multi(messages)
```

**One-liner convenience:**
```python
from server.client import chat
print(chat("what is 2+2"))
```

**Environment variables:**
| Var | Default | Description |
|-----|---------|-------------|
| LLM_BASE_URL | http://localhost:8080/v1 | API endpoint |
| LLM_API_KEY | sk-local | Auth key |
| LLM_MODEL | qwen/qwen3.6-27b-uncensored | Default model |

---

### API Reference

LAMU exposes a standard OpenAI-compatible API. Any tool that works with OpenAI works with LAMU.

**Base URL:** `http://localhost:8080/v1` (Bifrost gateway)  
**Direct URL:** `http://localhost:8020/v1` (model server, no routing)  
**Auth:** `Authorization: Bearer sk-local`

**Chat completion:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{
    "model": "qwen/qwen3.6-27b-uncensored",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": false
  }'
```

**List models:**
```bash
curl http://localhost:8020/v1/models
```

**Health check:**
```bash
curl http://localhost:8020/health
# {"status": "ok", "model": "qwen3.6-27b-uncensored", "context": 262144}
```

**Streaming:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{
    "model": "qwen/qwen3.6-27b-uncensored",
    "messages": [{"role": "user", "content": "write a poem"}],
    "stream": true
  }'
```

**Use with OpenAI Python SDK:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")
response = client.chat.completions.create(
    model="qwen/qwen3.6-27b-uncensored",
    messages=[{"role": "user", "content": "hello"}],
)
print(response.choices[0].message.content)
```

---

### Agent Swarm

The swarm is a LangGraph state machine that plans, implements, tests, and reviews code — all locally.

**Run the swarm on a task:**
```bash
just swarm "Fix the authentication bug in login.py" /path/to/repo
```

**How it works:**
```
PLANNER (thinks about the task, produces subtasks)
    ↓
WORKERS (implement each subtask in parallel — but sequential on single GPU)
    ↓
INTEGRATOR (merges parallel outputs if needed)
    ���
TEST RUNNER (runs pytest — deterministic, no LLM)
    ↓ PASS → CRITIC (reviews code quality)
    ↓ FAIL → back to WORKERS with error context (up to 3 retries)
    ↓
CRITIC APPROVES → saves training data → DONE
CRITIC REJECTS → back to PLANNER with feedback (up to 2 loops)
```

**Override models** (use cloud for planner, local for workers):
```bash
PLANNER_MODEL=anthropic/claude-opus-4-7 \
WORKER_MODEL=qwen/qwen3.6-27b-uncensored \
CRITIC_MODEL=anthropic/claude-sonnet-4-6 \
just swarm "Add pagination" /path/to/repo
```

**Run the benchmark:**
```bash
just bench-swarm        # run all 5 builtin tasks
just bench-list         # see available tasks
```

**Custom benchmark tasks** (`tasks.json`):
```json
[
  {
    "id": "my-task",
    "description": "Add retry logic to the HTTP client",
    "repo": "/path/to/project",
    "test_cmd": "python -m pytest tests/test_http.py -v"
  }
]
```

---

### Training Pipeline

Every successful swarm run automatically saves a training tuple. Over time, fine-tune the model on your codebase.

**Check collected data:**
```bash
just train-status
```

**Prepare dataset:**
```bash
just train-prepare
```

**Run QLoRA fine-tuning:**
```bash
just train                          # defaults: Qwen3.5-27B, 3 epochs
just train "Qwen/Qwen3.6-27B" 5    # custom model + epochs
```

**Export to GGUF (for serving with LAMU):**
```bash
just train-export-gguf
```

**Export to HuggingFace format (for vLLM/SGLang):**
```bash
just train-export-hf
```

The loop:
```
swarm succeeds → data saved → accumulate → fine-tune → export → serve → swarm is better
```

---

### Image & Video Generation

LAMU supports GPU-swapping to ComfyUI for image/video generation.

**Setup (one-time):**
```bash
just setup-comfyui
```

**Swap GPU to ComfyUI:**
```bash
just swap-comfyui    # kills LLM, starts ComfyUI on :8188
```

**Swap back to LLM:**
```bash
just swap-llm        # kills ComfyUI, starts Qwen3.6
```

**Supported models:**
- **FLUX.1/2 Dev** — current image quality leader (FP8, ~17GB)
- **SDXL** — fast, massive LoRA ecosystem (~8GB)
- **Wan 2.2** — video generation (480p-720p)
- **LTX-2.3** — 4K video on 24GB

Download models through ComfyUI's built-in model manager at `http://localhost:8188`.

---

### Model Management

**Currently serving:**
```bash
just status
curl http://localhost:8020/health
```

**Available models on disk:**
```bash
ls ~/models/
```

**Download models:**
```bash
just setup-qwen36           # Qwen3.6-27B dense (recommended)
just setup-qwen36-moe       # Qwen3.6-35B-A3B MoE (doesn't fit on single 4090)
```

**Swap context profiles:**
```bash
just serve-qwen36              # auto: best quant, optimal context
just serve-qwen36 262144       # force 262K context (uses Q4 KV)
just serve-qwen36 108000       # 108K with Q8 KV (better quality)
```

**Context vs quality tradeoffs:**
| Quant | KV Cache | Max Context | Quality |
|-------|----------|-------------|---------|
| Q5_K_S | Q8_0 | 108K | Best |
| Q4_K_M | Q8_0 | 173K | Good |
| Q4_K_M | Q4_0 | 262K | Acceptable |

---

### Troubleshooting

**"command not found: just"**
```bash
sudo pacman -S just    # Arch
cargo install just     # Any Linux
```

**"No models detected" when running `llm`**
```bash
just start    # starts the stack
```

**Model won't start (CUDA OOM)**
- Close browsers, Discord, other GPU-using apps
- Check: `nvidia-smi` — anything else using the GPU?
- Reduce context: `just serve-qwen36 32768`

**MCP not working in Claude Code**
- Restart Claude Code (picks up ~/.claude.json changes)
- Check model is running: `curl http://localhost:8020/health`
- Check MCP config path matches your actual home directory

**Bifrost returns 401/400**
- Cloud provider keys are placeholders — edit `deps/bifrost/config.json`
- For local-only use, just hit `:8020` directly instead of `:8080`

**"Failed to create llama_context"**
- Context too large for available VRAM
- Fix: reduce context or kill other GPU processes

**Server crashed — how to restart:**
```bash
just stop && just start
```

**Check logs:**
```bash
tail -f /tmp/qwen36-server.log     # model server
cat ~/lamu/deps/bifrost/bifrost.log # gateway
```

---

## Architecture

```
┌─────────────────────���───────────────────────────────────────┐
│  SURFACES                                                    │
│  llm (terminal) │ Chainlit (web) │ MCP (Claude Code) │ API  │
└────────────────────────────┬─────────────���──────────────────┘
                             │
                     ┌───────▼───���───┐
                     │   Bifrost     │  Gateway (:8080)
                     │   Routes by   │  provider/model
                     └───────┬───────┘
                             │
              ┌──────────────┼───────────��──┐
              │              │              │
     ┌────────▼──┐  ┌───────▼───┐  ┌──────▼──────┐
     │ Qwen3.6   │  │  DFlash   │  │  ComfyUI    │
     │ 27B Dense │  │ Qwen3.5   │  │ FLUX/Wan    │
     │ :8020     │  │ :8000     │  │ :8188       │
     │ 262K ctx  │  ��� 8K ctx    │  │ img/video   │
     └───────────┘  └───────────┘  └─────────────┘
              ▲
              │
     ┌────────┴───────���┐
     │  Agent Swarm    │
     │  plan→work→test │
     │  →review→train  │
     └─────────────────��
```

## Philosophy

LAMU is a **framework**, not just a model runner. The current cutting-edge config ships ready to go, but every layer is swappable:

- **Model**: Currently Qwen3.6-27B Heretic. Tomorrow it could be Llama 4, Gemma 5, or whatever beats it. Change one GGUF path.
- **Engine**: Currently llama-cpp-python. When SGLang/vLLM properly support your model + hardware, swap the serve script.
- **Gateway**: Currently Bifrost. Could be LiteLLM, any OpenAI-compatible router.
- **Framework integration**: MCP for Claude Code today. Tomorrow it's whatever protocol wins.

The config is opinionated. The architecture isn't.

## Comparison

| Feature | Ollama | LM Studio | Open WebUI | text-gen-webui | **LAMU** |
|---------|--------|-----------|------------|----------------|----------|
| OpenAI-compatible API | Yes | Yes | Via Ollama | Yes | **Yes** |
| MCP server (Claude Code) | Community | No | No | No | **Built-in** |
| Agent swarm (plan→implement→test→review) | No | No | No | No | **Yes** |
| Training from swarm outputs | No | No | No | No | **Yes** |
| 262K context on 24GB | No | Maybe | N/A | Maybe | **Yes** |
| Uncensored by default | Depends | Depends | Depends | Depends | **Yes** |
| Test-driven validation loop | No | No | No | No | **Yes** |
| Image/video generation swap | No | No | No | Yes | **Yes** |
| Single CLI (`just`) | No | N/A | docker-compose | No | **43 commands** |
| Auto-starts on boot | No | Yes | docker | No | **systemd** |

## All Commands

```
STACK
  just start              Start full stack (Qwen3.6 + Bifrost)
  just stop               Stop everything
  just status             Show what's up/down

CHAT
  just chat               Interactive REPL
  just ask "prompt"       One-shot answer
  just chat-with <model>  Chat with specific model

MODELS
  just serve-qwen36              Start Qwen3.6 (auto-picks best config)
  just serve-qwen36 262144       Force 262K context
  just serve-qwen36-dense        Force dense 27B model
  just serve-dflash              Start DFlash (Qwen3.5-27B)
  just serve-bifrost             Start gateway
  just serve-web                 Start Chainlit web UI
  just serve-comfyui             Start ComfyUI (image/video)
  just serve-timemachine         GPT-2 XL nostalgia presets

SWAP
  just swap-comfyui       GPU → ComfyUI (kills LLM)
  just swap-llm           GPU → Qwen3.6 (kills ComfyUI)
  just swap-qwen          Hot-swap to Qwen3.5 DFlash
  just swap-gpt2          Hot-swap to GPT-2 XL

SWARM
  just swarm "task" repo         Run agent swarm
  just bench-list                List benchmark tasks
  just bench-swarm               Run benchmark (full swarm)
  just bench-opus                Run benchmark (cloud-only baseline)
  just bench-compare a b         Compare two runs

TRAINING
  just train-status              Show collected data
  just train-prepare             Build dataset from swarm runs
  just train                     Run QLoRA fine-tuning
  just train-export-gguf         Export to GGUF
  just train-export-hf           Export to HuggingFace format

SETUP
  just setup-qwen36              Download Qwen3.6-27B (~16 GB)
  just setup-qwen36-moe          Download Qwen3.6-35B MoE (~21 GB)
  just setup-comfyui             Install ComfyUI + video nodes
  just setup-web                 Install Chainlit web UI
  just setup-agents              Install LangGraph agents

TEST
  just test-qwen36               Smoke test Qwen3.6
  just test-dflash               Smoke test DFlash
  just mcp-test                  Test MCP server
```

## Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| RAM | 32GB | 64GB |
| Disk | 50GB free | 100GB free |
| OS | Linux (NVIDIA drivers) | Arch/CachyOS |

## File Structure

```
server/          Production server + MCP + Python client
  serve.py       Main server (think-block middleware, health, auto-config)
  mcp_qwen.py    MCP server for Claude Code
  client.py      Python SDK
  dflash.py      DFlash OpenAI server (tool calling)
cli/             Terminal chat REPL
  chat_repl.py   Rich markdown rendering, auto-start, model discovery
web/             Chainlit web frontend
  app.py         Agentic chat + streaming
  data_layer.py  SQLite persistence for chat history
agents/          Swarm + training + benchmarks
  swarm.py       LangGraph state machine
  trainer.py     QLoRA fine-tuning pipeline
  bench.py       Benchmark runner
config/          Model registry (models.yaml)
scripts/         All serve/stop/setup/swap shell scripts
deps/            Third-party (Bifrost, Langfuse, lucebox-hub)
justfile         Single command surface (43 commands)
```

## Why "LAMU"

**L**ocal **A**gent **M**odel **U**tility. Also a nod to keeping things personal and close to home — your models, your hardware, your data, your rules.

## License

Apache-2.0 (model weights), MIT (this repo code).
