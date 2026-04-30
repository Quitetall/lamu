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

## Philosophy

LAMU is a **framework**, not just a model runner. The current cutting-edge config ships ready to go, but every layer is swappable:

- **Model**: Currently Qwen3.6-27B Heretic. Tomorrow it could be Llama 4, Gemma 5, or whatever beats it. Change one GGUF path.
- **Engine**: Currently llama-cpp-python. When SGLang/vLLM properly support your model + hardware, swap the serve script.
- **Gateway**: Currently Bifrost. Could be LiteLLM, any OpenAI-compatible router.
- **Framework integration**: MCP for Claude Code today. Tomorrow it's whatever protocol wins.

The config is opinionated. The architecture isn't.

## What makes this different

No other project combines all of these in one stack:

| Feature | Ollama | LM Studio | Open WebUI | text-gen-webui | **LAMU** |
|---------|--------|-----------|------------|----------------|---------------|
| OpenAI-compatible API | Yes | Yes | Via Ollama | Yes | **Yes** |
| MCP server (Claude Code) | Community | No | No | No | **Built-in** |
| Agent swarm (plan→implement→test→review) | No | No | No | No | **Yes** |
| Training from swarm outputs | No | No | No | No | **Yes** |
| 262K context on 24GB | No | Maybe | N/A | Maybe | **Yes** |
| Uncensored by default | Depends | Depends | Depends | Depends | **Yes** |
| Test-driven validation loop | No | No | No | No | **Yes** |
| Image/video generation swap | No | No | No | Yes | **Yes** |
| Single `justfile` CLI | No | N/A | docker-compose | No | **43 commands** |
| Auto-starts on boot | No | Yes | docker | No | **systemd** |

### The novel parts:

1. **Agentic swarm with local-only execution.** Planner → parallel workers → integrator → pytest → critic, all on one GPU. 80% pass rate on coding tasks at $0 cost.

2. **Self-improving training loop.** Every successful swarm run saves (task, implementation, test_result) triples. Accumulate enough → QLoRA fine-tune → export GGUF → local model gets better at your codebase over time.

3. **262K context on 24GB via Q4_0 KV + flash attention.** Qwen3.6's DeltaNet hybrid architecture only needs KV cache for 16/64 layers. Combined with quantized KV and `logits_all=False` (fixes a 242 GiB RAM bug in llama-cpp-python), full native context fits on a consumer card.

4. **MCP bridge to Claude Code.** Cloud AI plans, local AI executes. 70-80% of tokens generated for free. The cloud model reviews local output — you get cloud quality at local cost.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  SURFACES                                                    │
│  llm (terminal) │ Chainlit (web) │ MCP (Claude Code) │ API  │
└────────────────────────────┬────────────────────────────────┘
                             │
                     ┌───────▼───────┐
                     │   Bifrost     │  Gateway (:8080)
                     │   Routes by   │  provider/model
                     └───────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──┐  ┌───────▼───┐  ┌──────▼──────┐
     │ Qwen3.6   │  │  DFlash   │  │  ComfyUI    │
     │ 27B Dense │  │ Qwen3.5   │  │ FLUX/Wan    │
     │ :8020     │  │ :8000     │  │ :8188       │
     │ 262K ctx  │  │ 8K ctx    │  │ img/video   │
     └───────────┘  └───────────┘  └─────────────┘
              ▲
              │
     ┌────────┴────────┐
     │  Agent Swarm    │
     │  plan→work→test │
     │  →review→train  │
     └─────────────────┘
```

## Quick start

```bash
# Clone
git clone https://github.com/Quitetall/lamu ~/lamu
cd ~/lamu

# Download model (~16 GB)
just setup-qwen36

# Start (auto-starts on boot after first run)
just start

# Chat
llm "explain quicksort"

# Or use the justfile
just chat
just status
just --list
```

## Commands

```
just start              Start full stack
just stop               Stop everything  
just status             What's running
just chat               Interactive REPL
just ask "prompt"       One-shot answer
just serve-qwen36       Start Qwen3.6 (auto-picks best quant)
just swap-comfyui       GPU → image/video generation
just swap-llm           GPU → LLM (back)
just swarm "task" repo  Run agent swarm on a codebase
just bench-swarm        Run coding benchmark
just train              QLoRA fine-tune from collected data
just train-status       Show training data stats
just setup-comfyui      Install ComfyUI + video nodes
just test-qwen36        Smoke test through Bifrost
```

## Model

**Qwen3.6-27B Dense Uncensored (Heretic v2)**
- 94% fewer refusals than the censored model (6/100 vs 92/100)
- 0.0021 KL divergence from original (quality preserved)
- SWE-bench Verified: 77.2 | Terminal-Bench: 59.3 | AIME 2026: 94.1
- GGUF Q4_K_M, served via llama-cpp-python with flash attention
- 262K native context via Q4_0 KV cache (only 16/64 layers need KV)

## Hardware requirements

- **GPU**: RTX 4090 (24GB) — or any 24GB+ NVIDIA card
- **RAM**: 32GB minimum (62GB recommended)
- **Disk**: ~50GB (model + stack)

## File structure

```
server/          Custom server code (DFlash, GPT-2 proxy, MCP, client lib)
cli/             Terminal chat REPL
web/             Chainlit web frontend
agents/          Swarm graph + trainer + benchmark
config/          Model registry
scripts/         All serve/stop/setup/swap scripts
deps/            Third-party (Bifrost gateway, Langfuse, lucebox-hub)
justfile         Single command surface
```

## Training loop

```
swarm runs → successful pairs saved → prepare dataset → QLoRA fine-tune → export GGUF → serve
     ▲                                                                              │
     └──────────────────── model gets better at your code ──────────────────────────┘
```

Every successful swarm run (tests pass + critic approves) auto-saves a training tuple. Over time, fine-tune the local model on your codebase conventions. The model improves at your specific domain.

## Why "LAMU"

**L**ocal **A**gent **M**odel **U**tility. Also a nod to keeping things personal and close to home — your models, your hardware, your data, your rules.

## License

Apache-2.0 (model weights), MIT (this repo code).
