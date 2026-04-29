#!/usr/bin/env bash
# scripts/serve-qwen36.sh — serve Qwen3.6 uncensored via llama-cpp-python
# Auto-detects which model is downloaded. Prefers dense 27B over MoE 35B-A3B.
# Usage: serve-qwen36.sh [dense|moe]
set -euo pipefail

ROOT="$HOME/local-llm"
VENV="$ROOT/.venv"
PORT=8020
PID_FILE="/tmp/qwen36-server.pid"
LOG="/tmp/qwen36-server.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

DENSE_DIR="$HOME/models/qwen3.6-27b-heretic"
MOE_DIR="$HOME/models/qwen3.6-35b-a3b-heretic"
VARIANT="${1:-auto}"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  Qwen3.6  ${GRY}already running on :$PORT${R}"
  exit 0
fi

# Resolve model file
GGUF=""
MODEL_LABEL=""
CTX=32768

case "$VARIANT" in
  dense)
    GGUF=$(find "$DENSE_DIR" -name "*Q4_K_M*.gguf" -print -quit 2>/dev/null)
    MODEL_LABEL="Qwen3.6-27B Dense"
    CTX=49152  # 16GB model leaves ~8GB for KV cache
    ;;
  moe)
    GGUF=$(find "$MOE_DIR" -name "*Q4_K_M*.gguf" -print -quit 2>/dev/null)
    MODEL_LABEL="Qwen3.6-35B-A3B MoE"
    CTX=32768  # 21GB model, less room
    ;;
  auto)
    # Prefer dense (better benchmarks, smaller VRAM)
    GGUF=$(find "$DENSE_DIR" -name "*Q4_K_M*.gguf" -print -quit 2>/dev/null)
    if [[ -n "$GGUF" ]]; then
      MODEL_LABEL="Qwen3.6-27B Dense"
      CTX=49152
    else
      GGUF=$(find "$MOE_DIR" -name "*Q4_K_M*.gguf" -print -quit 2>/dev/null)
      MODEL_LABEL="Qwen3.6-35B-A3B MoE"
      CTX=32768
    fi
    ;;
esac

if [[ -z "$GGUF" ]]; then
  echo -e "${YEL}No model found.${R} Download one first:"
  echo -e "  ${GRY}just setup-qwen36-dense   (recommended — 27B, ~16 GB)${R}"
  echo -e "  ${GRY}just setup-qwen36-moe     (35B MoE, ~21 GB)${R}"
  exit 1
fi

echo -e "  Starting ${MODEL_LABEL} Uncensored ${GRY}(log: $LOG)${R}"
echo -e "  ${GRY}Model: $GGUF${R}"
echo -e "  ${GRY}Context: $CTX${R}"

nohup "$VENV/bin/python" -m llama_cpp.server \
  --model "$GGUF" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --n_gpu_layers -1 \
  --n_ctx "$CTX" \
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
