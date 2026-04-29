#!/usr/bin/env bash
# scripts/serve-vllm-qwen36.sh — serve Qwen3.6-27B uncensored via vLLM
#
# Uses bitsandbytes 4-bit quantization (loads BF16 safetensors, quantizes on GPU).
# FP8 KV cache to maximize context length.
# Qwen3.6-27B has 48 DeltaNet layers (O(1) memory) + 16 attention layers (need KV),
# so 262K context may fit on a single 4090 with FP8 KV.
#
# First run downloads ~54GB of safetensors from HuggingFace.
set -euo pipefail

ROOT="$HOME/local-llm"
VENV="$ROOT/.venv"
PORT=8020
MODEL="llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
PID_FILE="/tmp/qwen36-server.pid"
LOG="/tmp/qwen36-vllm.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

# Context length: try full 262K, fall back to 131K or 65K if OOM
CTX="${1:-262144}"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  Qwen3.6  ${GRY}already running on :$PORT${R}"
  exit 0
fi

echo -e "  Starting Qwen3.6-27B via vLLM ${GRY}(log: $LOG)${R}"
echo -e "  ${GRY}Model: $MODEL${R}"
echo -e "  ${GRY}Context: $CTX tokens${R}"
echo -e "  ${GRY}First run downloads ~54 GB from HuggingFace${R}"

nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "qwen3.6-35b-uncensored" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --max-model-len "$CTX" \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --kv-cache-dtype fp8_e5m2 \
  --gpu-memory-utilization 0.95 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --dtype half \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for vLLM"
# vLLM takes longer to start (downloads model on first run)
for _ in $(seq 1 300); do
  if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    exit 0
  fi
  # Check for crash
  if ! kill -0 "$(cat $PID_FILE 2>/dev/null)" 2>/dev/null; then
    echo -e " ${YEL}crashed — check $LOG${R}"
    # If OOM, suggest lower context
    if grep -q "OutOfMemory\|CUDA out of memory\|OOM" "$LOG" 2>/dev/null; then
      echo -e "  ${YEL}OOM detected. Try lower context:${R}"
      echo -e "  ${GRY}bash $0 131072${R}  (131K)"
      echo -e "  ${GRY}bash $0 65536${R}   (65K)"
    fi
    exit 1
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
