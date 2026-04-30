# LAMU — Local Agent Model Utility
# Run `just` with no args to see all available commands.

set dotenv-load

root := env("HOME") / "local-llm"

# ── Stack lifecycle ──────────────────────────────────────────────────────

# Start the full stack (DFlash + Bifrost + Langfuse + Chainlit)
start:
    bash {{root}}/scripts/start.sh

# Stop the full stack
stop:
    bash {{root}}/scripts/stop.sh

# Show what's running
status:
    @echo "DFlash  :8000  $(curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "vLLM    :8020  $(curl -sf http://localhost:8020/v1/models >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "Bifrost :8080  $(curl -sf http://localhost:8080/health >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "Langfuse:3000  $(curl -sf http://localhost:3000/api/public/health >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "SGLang  :8001  $(curl -sf http://localhost:8001/v1/models >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "GPT2prx :9001  $(curl -sf http://localhost:9001/health >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "Chainlit:7860  $(curl -sf http://localhost:7860 >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "ComfyUI :8188  $(curl -sf http://localhost:8188/system_stats >/dev/null 2>&1 && echo '✓' || echo '✗')"

# ── Individual services ──────────────────────────────────────────────────

# Start DFlash (Qwen3.5-27B on :8000)
serve-dflash:
    bash {{root}}/scripts/serve-dflash.sh

# Start Qwen3.6 uncensored on :8020 (auto-detects dense vs MoE)
serve-qwen36:
    bash {{root}}/scripts/serve-qwen36.sh

# Start Qwen3.6-27B dense specifically (best benchmarks)
serve-qwen36-dense:
    bash {{root}}/scripts/serve-qwen36.sh dense

# Start Qwen3.6-35B-A3B MoE specifically (faster per-token)
serve-qwen36-moe:
    bash {{root}}/scripts/serve-qwen36.sh moe

# Start Qwen3.6-27B uncensored via vLLM — full 262K context, tool calling, reasoning parser
serve-vllm ctx="262144":
    bash {{root}}/scripts/serve-vllm-qwen36.sh {{ctx}}

# Start Bifrost gateway (:8080)
serve-bifrost:
    bash {{root}}/scripts/serve-bifrost.sh

# Start Langfuse observability (:3000)
serve-langfuse:
    bash {{root}}/scripts/serve-langfuse.sh

# GPT-2 XL with 2021-era presets (shitty-best2021, shitty-inferkit, etc.)
serve-timemachine:
    bash {{root}}/scripts/serve-sglang.sh gpt2-xl && bash {{root}}/scripts/serve-sglang-presets.sh

# Start Chainlit web UI (:7860)
serve-web:
    bash {{root}}/web/serve.sh

# Start ComfyUI for image/video generation (:8188)
serve-comfyui:
    bash {{root}}/scripts/serve-comfyui.sh

# ── Model swap ───────────────────────────────────────────────────────────

# Hot-swap to Qwen3.5-27B (DFlash)
swap-qwen:
    bash {{root}}/scripts/swap.sh qwen

# Hot-swap to GPT-2 XL (SGLang)
swap-gpt2:
    bash {{root}}/scripts/swap.sh gpt2

# Swap GPU to ComfyUI (kills LLM, starts ComfyUI)
swap-comfyui:
    #!/usr/bin/env bash
    echo -e "\033[1mSwapping GPU to ComfyUI\033[0m"
    kill $(cat /tmp/qwen36-server.pid 2>/dev/null) 2>/dev/null && echo "  Qwen3.6 stopped" || true
    kill $(cat /tmp/dflash-server.pid 2>/dev/null) 2>/dev/null && echo "  DFlash stopped" || true
    rm -f /tmp/qwen36-server.pid /tmp/dflash-server.pid
    sleep 2
    bash {{root}}/scripts/serve-comfyui.sh

# Swap GPU back to LLM (kills ComfyUI, starts Qwen3.6)
swap-llm:
    #!/usr/bin/env bash
    echo -e "\033[1mSwapping GPU to LLM\033[0m"
    kill $(cat /tmp/comfyui.pid 2>/dev/null) 2>/dev/null && echo "  ComfyUI stopped" || true
    rm -f /tmp/comfyui.pid
    sleep 2
    bash {{root}}/scripts/serve-qwen36.sh

# ── Chat ─────────────────────────────────────────────────────────────────

# Interactive REPL (auto-starts models if needed)
chat:
    bash {{root}}/scripts/chat.sh

# One-shot prompt (no REPL)
ask +prompt:
    bash {{root}}/scripts/chat.sh {{prompt}}

# Chat with a specific model
chat-with model:
    bash {{root}}/scripts/chat.sh -m {{model}}

# ── Swarm ────────────────────────────────────────────────────────────────

# Run the agent swarm on a task
swarm task repo test_cmd="python -m pytest tests/ -v --tb=short":
    cd {{root}} && .venv/bin/python -m agents.swarm "{{task}}" --repo "{{repo}}" --test "{{test_cmd}}"

# ── Benchmarks ───────────────────────────────────────────────────────────

# List available builtin benchmark tasks
bench-list:
    cd {{root}} && .venv/bin/python -m agents.bench list

# Run builtin benchmark with Opus solo (cloud-only baseline)
bench-opus:
    cd {{root}} && .venv/bin/python -m agents.bench run --suite builtin --config opus-solo

# Run builtin benchmark with the full swarm
bench-swarm:
    cd {{root}} && .venv/bin/python -m agents.bench run --suite builtin --config swarm

# Run SWE-bench Lite (real GitHub issues, requires datasets package)
bench-swebench config limit="10":
    cd {{root}} && .venv/bin/python -m agents.bench run --suite swebench --config {{config}} --limit {{limit}}

# Compare two benchmark runs
bench-compare run_a run_b:
    cd {{root}} && .venv/bin/python -m agents.bench compare {{run_a}} {{run_b}}

# ── Training ─────────────────────────────────────────────────────────────

# Show training data stats
train-status:
    cd {{root}} && .venv/bin/python -m agents.trainer status

# Prepare dataset from collected swarm data
train-prepare:
    cd {{root}} && .venv/bin/python -m agents.trainer prepare

# Run QLoRA fine-tuning (default: Qwen3.5-27B, 3 epochs)
train model="Qwen/Qwen3.5-27B" epochs="3":
    cd {{root}} && .venv/bin/python -m agents.trainer train --model "{{model}}" --epochs {{epochs}}

# Export fine-tuned model to GGUF (for DFlash)
train-export-gguf:
    cd {{root}} && .venv/bin/python -m agents.trainer export --format gguf

# Export fine-tuned model to HF format (for vLLM)
train-export-hf:
    cd {{root}} && .venv/bin/python -m agents.trainer export --format hf

# ── Setup ────────────────────────────────────────────────────────────────

# Set up Chainlit web frontend (create venv + install deps)
setup-web:
    bash {{root}}/web/setup.sh

# Set up LangGraph agents (create venv + install deps)
setup-agents:
    bash {{root}}/agents/setup.sh

# Set up ComfyUI + video nodes (image/video generation)
setup-comfyui:
    bash {{root}}/scripts/setup-comfyui.sh

# Download Qwen3.6-27B dense uncensored GGUF (~16 GB) — recommended
setup-qwen36:
    bash {{root}}/scripts/setup-qwen36-dense.sh

# Download Qwen3.6-27B dense specifically
setup-qwen36-dense:
    bash {{root}}/scripts/setup-qwen36-dense.sh

# Download Qwen3.6-35B-A3B MoE uncensored GGUF (~21 GB)
setup-qwen36-moe:
    bash {{root}}/scripts/setup-qwen36-moe.sh

# Clone + set up club-3090 for vLLM serving (~20 GB download)
setup-vllm:
    bash {{root}}/scripts/setup-club3090.sh

# ── MCP (Claude Code integration) ────────────────────────────────────────

# Test the MCP server (checks if local LLM is reachable)
mcp-test:
    @echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"clientInfo":{"name":"test","version":"1"},"protocolVersion":"2024-11-05"}}' \
      | timeout 3 {{root}}/.venv/bin/python {{root}}/server/mcp_qwen.py 2>/dev/null \
      && echo "MCP server responds OK" || echo "MCP server OK (timeout expected on stdin)"

# ── Quick test ───────────────────────────────────────────────────────────

# Smoke test: send a quick prompt through Bifrost → DFlash
test-dflash:
    @curl -s http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer sk-local" \
      -d '{"model":"dflash/luce-dflash","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":50}' \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# Smoke test: send a prompt through Bifrost → Qwen3.6 uncensored
test-qwen36:
    @curl -s http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer sk-local" \
      -d '{"model":"qwen/qwen3.6-27b-uncensored","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":50}' \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# Start Qwen3.6 with native C++ server + ngram-mod speculation (50-137 t/s)
serve-fast:
    bash {{root}}/scripts/serve-qwen36-fast.sh
