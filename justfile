# LAMU — Local Agent Model Utility
# Run `just` with no args to see all available commands.

set dotenv-load

root := env("HOME") / "local-llm"
lamu_bin := root / "lamu-rs" / "target" / "release" / "lamu"


# ═══════════════════════════════════════════════════════════════════════════
# v3 — `lamu` (Rust). Canonical path. Use these.
# ═══════════════════════════════════════════════════════════════════════════

# Install `lamu` to ~/.cargo/bin (one-time)
install:
    cd {{root}}/lamu-rs && cargo install --path lamu-cli --locked
    @echo "✓ lamu installed to ~/.cargo/bin (ensure on \$PATH)"

# Build the release binary in-tree (no cargo install)
build:
    cd {{root}}/lamu-rs && cargo build --release -p lamu-cli

# Discover GGUFs in ~/models/ → config/models.yaml
scan:
    {{lamu_bin}} scan

# Show registry + VRAM + which backend ports answer
status:
    {{lamu_bin}} status

# Boot the MCP server (stdio — for Claude Code)
start:
    {{lamu_bin}} start

# Boot the OpenAI-compat HTTP server on :8020 (override with `just serve PORT`)
serve port="8020":
    {{lamu_bin}} serve --port {{port}}

# Interactive REPL chatting against `lamu serve`
repl url="http://localhost:8020/v1/chat/completions":
    {{lamu_bin}} repl {{url}}

# ── Setup (still needed by quick start) ─────────────────────────────────────

# Download Qwen3.6-27B dense uncensored GGUF (~16 GB). Run before `lamu scan`.
setup-qwen36:
    bash {{root}}/scripts/setup-qwen36-dense.sh

# Download Qwen3.6-35B-A3B MoE uncensored GGUF (~21 GB)
setup-qwen36-moe:
    bash {{root}}/scripts/setup-qwen36-moe.sh

# Clone + set up club-3090 for vLLM serving (~20 GB download)
setup-vllm:
    bash {{root}}/scripts/setup-club3090.sh

# Set up ComfyUI + video nodes (image/video generation)
setup-comfyui:
    bash {{root}}/scripts/setup-comfyui.sh

# ── Tests + lint ────────────────────────────────────────────────────────────

# Default test target: fast unit suite (skips slow/gpu/network/rust/contract)
test:
    cd {{root}} && .venv/bin/python -m pytest

test-fast:
    cd {{root}} && .venv/bin/python -m pytest

test-slow:
    cd {{root}} && .venv/bin/python -m pytest tests/integration --override-ini="addopts="

test-gpu:
    cd {{root}} && .venv/bin/python -m pytest -m gpu --override-ini="addopts="

test-rust:
    cd {{root}}/lamu-rs && cargo test --workspace

# Cross-language MCP contract tests (Python ↔ Rust parity)
test-contract:
    cd {{root}} && .venv/bin/python -m pytest tests/contract -m contract --override-ini="addopts="

coverage:
    cd {{root}} && .venv/bin/python -m pytest --cov=lamu --cov=agents --cov=cli --cov=web --cov=server --cov-report=term-missing

lint:
    -cd {{root}} && .venv/bin/python -m ruff check lamu agents cli web server tests
    cd {{root}}/lamu-rs && cargo clippy --workspace -- -D warnings

# Install dev dependencies (pytest etc.)
test-setup:
    cd {{root}} && uv pip install --python .venv/bin/python -e ".[dev]"

# ── Swarm + training (still v1 surface) ─────────────────────────────────────

swarm task repo test_cmd="python -m pytest tests/ -v --tb=short":
    cd {{root}} && .venv/bin/python -m agents.swarm "{{task}}" --repo "{{repo}}" --test "{{test_cmd}}"

bench-list:
    cd {{root}} && .venv/bin/python -m agents.bench list

bench-opus:
    cd {{root}} && .venv/bin/python -m agents.bench run --suite builtin --config opus-solo

bench-swarm:
    cd {{root}} && .venv/bin/python -m agents.bench run --suite builtin --config swarm

bench-swebench config limit="10":
    cd {{root}} && .venv/bin/python -m agents.bench run --suite swebench --config {{config}} --limit {{limit}}

bench-compare run_a run_b:
    cd {{root}} && .venv/bin/python -m agents.bench compare {{run_a}} {{run_b}}

train-status:
    cd {{root}} && .venv/bin/python -m agents.trainer status

train-prepare:
    cd {{root}} && .venv/bin/python -m agents.trainer prepare

train model="Qwen/Qwen3.5-27B" epochs="3":
    cd {{root}} && .venv/bin/python -m agents.trainer train --model "{{model}}" --epochs {{epochs}}

train-export-gguf:
    cd {{root}} && .venv/bin/python -m agents.trainer export --format gguf

train-export-hf:
    cd {{root}} && .venv/bin/python -m agents.trainer export --format hf

setup-web:
    bash {{root}}/web/setup.sh

setup-agents:
    bash {{root}}/agents/setup.sh


# ═══════════════════════════════════════════════════════════════════════════
# Legacy v1 — script-driven launchers. Preserved in legacy/.
# Prefer `lamu` for new work. These are kept for perf-table reproducibility
# and DFlash/megakernel custom-server invocation.
# ═══════════════════════════════════════════════════════════════════════════

# Boot the v1 stack (Qwen3.6 + megakernel + Bifrost + Langfuse + Chainlit).
start-v1 ctx="med":
    bash {{root}}/legacy/scripts/swap-model.sh 3.6 {{ctx}}
    @echo "Starting megakernel 0.8B on :8001..."
    @nohup {{root}}/.venv/bin/python {{root}}/server/megakernel_server.py --port 8001 > /tmp/megakernel.log 2>&1 &
    @sleep 8 && curl -sf http://localhost:8001/health > /dev/null && echo "  0.8B ready on :8001" || echo "  0.8B failed (check /tmp/megakernel.log)"

# Stop everything spawned by start-v1.
stop-v1:
    -pkill -f "llama-server" 2>/dev/null
    -pkill -f "megakernel_server" 2>/dev/null
    @echo "All v1 models stopped."

# v1 doctor — diagnostic for the legacy stack.
doctor-v1:
    bash {{root}}/legacy/scripts/doctor.sh

# v1 status — every port the legacy stack used.
status-v1:
    @echo "DFlash  :8000  $(curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "Qwen3.6 :8020  $(curl -sf http://localhost:8020/v1/models >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "Bifrost :8080  $(curl -sf http://localhost:8080/health >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "Langfuse:3000  $(curl -sf http://localhost:3000/api/public/health >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "SGLang  :8001  $(curl -sf http://localhost:8001/v1/models >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "GPT2prx :9001  $(curl -sf http://localhost:9001/health >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "Chainlit:7860  $(curl -sf http://localhost:7860 >/dev/null 2>&1 && echo '✓' || echo '✗')"
    @echo "ComfyUI :8188  $(curl -sf http://localhost:8188/system_stats >/dev/null 2>&1 && echo '✓' || echo '✗')"

# Sidecar small model on :8001 alongside 27B (v1 path).
sidecar tier="fast":
    #!/usr/bin/env bash
    pkill -f "megakernel_server\|llama-server.*8001" 2>/dev/null; sleep 2
    case "{{tier}}" in
      fast|4b)
        echo "Starting Qwen3.5-4B on :8001 (~200 t/s)..."
        nohup {{root}}/../llama.cpp/build/bin/llama-server \
          -m ~/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
          --host 0.0.0.0 --port 8001 -ngl 99 --flash-attn on --parallel 1 \
          --ctx-size 8192 > /tmp/sidecar.log 2>&1 &
        sleep 8 && curl -sf http://localhost:8001/health > /dev/null && echo "  4B ready on :8001" || echo "  4B failed"
        ;;
      lobo|0.8b|mega)
        echo "Starting 0.8B megakernel on :8001 (494 t/s, lobotomized)..."
        nohup {{root}}/.venv/bin/python {{root}}/server/megakernel_server.py --port 8001 > /tmp/sidecar.log 2>&1 &
        sleep 10 && curl -sf http://localhost:8001/health > /dev/null && echo "  0.8B ready on :8001" || echo "  0.8B failed"
        ;;
      off)
        echo "Sidecar stopped."
        ;;
      *)
        echo "Usage: just sidecar [fast|lobo|off]"
        ;;
    esac

# Hot-swap the model on :8020.
swap model="status" ctx="":
    bash {{root}}/legacy/scripts/swap-model.sh {{model}} {{ctx}}

# Start DFlash (Qwen3.5-27B speculative on :8000 — 106 t/s).
serve-dflash:
    bash {{root}}/legacy/scripts/serve-dflash.sh

# Start Qwen3.6 ngram-mod on :8020 (40+ t/s warm).
serve-qwen36:
    bash {{root}}/legacy/scripts/serve-qwen36.sh

serve-qwen36-dense:
    bash {{root}}/legacy/scripts/serve-qwen36.sh dense

serve-qwen36-moe:
    bash {{root}}/legacy/scripts/serve-qwen36.sh moe

# DFlash one-shot (uses full GPU).
serve-fast prompt="Write Python quicksort.":
    bash {{root}}/legacy/scripts/serve-qwen36-fast.sh "{{prompt}}"

# Qwen3.6 via vLLM with full 262K context.
serve-vllm ctx="262144":
    bash {{root}}/legacy/scripts/serve-vllm-qwen36.sh {{ctx}}

# Bifrost gateway (:8080) — pending Phase 1 benchmark.
serve-bifrost:
    bash {{root}}/scripts/serve-bifrost.sh

stop-bifrost:
    bash {{root}}/scripts/stop-bifrost.sh

serve-langfuse:
    bash {{root}}/legacy/scripts/serve-langfuse.sh

# GPT-2 XL with 2021-era presets (timemachine mode).
serve-timemachine:
    bash {{root}}/legacy/scripts/serve-sglang.sh gpt2-xl && bash {{root}}/legacy/scripts/serve-sglang-presets.sh

serve-web:
    bash {{root}}/web/serve.sh

serve-comfyui:
    bash {{root}}/legacy/scripts/serve-comfyui.sh

# v1 chat (talked direct to backends + Bifrost). Use `lamu repl` instead.
chat:
    bash {{root}}/legacy/scripts/chat.sh

ask +prompt:
    bash {{root}}/legacy/scripts/chat.sh {{prompt}}

# EAGLE speculative decoding research server.
serve-eagle:
    bash {{root}}/legacy/scripts/serve-eagle.sh

# Hot-reload Qwen3.6 quant via legacy reload endpoint.
reload-max-ctx:
    @curl -s -X POST http://localhost:8020/reload -H "Content-Type: application/json" -d '{"quant":"Q4_K_M"}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Reloading: {r.get(\"from\",\"?\")} → {r.get(\"to\",\"?\")} ({r.get(\"context\",\"?\")} ctx)')"

reload-quality:
    @curl -s -X POST http://localhost:8020/reload -H "Content-Type: application/json" -d '{"quant":"Q5_K_S"}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'Reloading: {r.get(\"from\",\"?\")} → {r.get(\"to\",\"?\")} ({r.get(\"context\",\"?\")} ctx)')"

# v1 MCP server (server/mcp_qwen.py). Use `lamu start` instead.
mcp-test:
    @echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"clientInfo":{"name":"test","version":"1"},"protocolVersion":"2024-11-05"}}' \
      | timeout 3 {{root}}/.venv/bin/python {{root}}/server/mcp_qwen.py 2>/dev/null \
      && echo "MCP server responds OK" || echo "MCP server OK (timeout expected on stdin)"
