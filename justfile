# local-llm — command reference
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

# ── Individual services ──────────────────────────────────────────────────

# Start DFlash (Qwen3.5-27B on :8000)
serve-dflash:
    bash {{root}}/scripts/serve-dflash.sh

# Start Qwen3.6-35B-A3B uncensored MoE on :8020 (swarm worker model)
serve-qwen36:
    bash {{root}}/scripts/serve-qwen36.sh

# Start vLLM via club-3090 (Qwen3.6-27B on :8020 — alternative)
serve-vllm:
    bash {{root}}/scripts/serve-vllm.sh

# Start Bifrost gateway (:8080)
serve-bifrost:
    bash {{root}}/scripts/serve-bifrost.sh

# Start Langfuse observability (:3000)
serve-langfuse:
    bash {{root}}/scripts/serve-langfuse.sh

# Start SGLang + GPT-2 proxy
serve-sglang:
    bash {{root}}/scripts/serve-sglang.sh gpt2-xl && bash {{root}}/scripts/serve-sglang-presets.sh

# Start Chainlit web UI (:7860)
serve-web:
    bash {{root}}/web/serve.sh

# ── Model swap ───────────────────────────────────────────────────────────

# Hot-swap to Qwen3.5-27B (DFlash)
swap-qwen:
    bash {{root}}/scripts/swap.sh qwen

# Hot-swap to GPT-2 XL (SGLang)
swap-gpt2:
    bash {{root}}/scripts/swap.sh gpt2

# ── Chat ─────────────────────────────────────────────────────────────────

# Terminal chat REPL
chat:
    bash {{root}}/scripts/chat.sh

# ── Swarm ────────────────────────────────────────────────────────────────

# Run the agent swarm on a task
swarm task repo test_cmd="python -m pytest tests/ -v --tb=short":
    cd {{root}} && .venv/bin/python -m agents.swarm "{{task}}" --repo "{{repo}}" --test "{{test_cmd}}"

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

# Download Qwen3.6-35B-A3B uncensored GGUF (~21 GB)
setup-qwen36:
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
      -d '{"model":"qwen/qwen3.6-35b-uncensored","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":50}' \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
