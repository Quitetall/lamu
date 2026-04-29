#!/usr/bin/env bash
# scripts/serve-qwen36.sh — serve Qwen3.6-35B-A3B uncensored via llama-cpp-python
# MoE model: 35B total / 3B active. Uncensored heretic variant.
# Serves OpenAI-compatible API on :8020.
set -euo pipefail

ROOT="$HOME/local-llm"
VENV="$ROOT/.venv"
PORT=8020
MODEL_DIR="$HOME/models/qwen3.6-35b-a3b-heretic"
PID_FILE="/tmp/qwen36-server.pid"
LOG="/tmp/qwen36-server.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  Qwen3.6  ${GRY}already running on :$PORT${R}"
  exit 0
fi

# Find the GGUF file
GGUF=$(find "$MODEL_DIR" -name "*Q4_K_M*.gguf" -print -quit 2>/dev/null)
if [[ -z "$GGUF" ]]; then
  echo -e "${YEL}Model not found.${R} Run: just setup-qwen36"
  exit 1
fi

echo -e "  Starting Qwen3.6-35B-A3B ${GRY}(log: $LOG)${R}"
echo -e "  ${GRY}Model: $GGUF${R}"

# llama-cpp-python server with CUDA offload
# n_gpu_layers=-1 offloads all layers to GPU
# n_ctx=32768 for reasonable context on 24GB
nohup "$VENV/bin/python" -m llama_cpp.server \
  --model "$GGUF" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --n_gpu_layers -1 \
  --n_ctx 32768 \
  --chat_format chatml \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for Qwen3.6"
for _ in $(seq 1 90); do
  if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    exit 0
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
