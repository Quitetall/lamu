#!/usr/bin/env bash
# scripts/serve-qwen36.sh — serve Qwen3.6 uncensored via llama-cpp-python
# Full 262K context on single 4090 with Q4_0 KV cache + flash attention.
# Usage: serve-qwen36.sh [dense|moe] [context_length]
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
CTX="${2:-262144}"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  Qwen3.6  ${GRY}already running on :$PORT${R}"
  exit 0
fi

# Resolve model file
GGUF=""
MODEL_LABEL=""

case "$VARIANT" in
  dense)
    GGUF=$(find "$DENSE_DIR" -name "*Q4_K_M*.gguf" -print -quit 2>/dev/null)
    MODEL_LABEL="Qwen3.6-27B Dense"
    ;;
  moe)
    GGUF=$(find "$MOE_DIR" -name "*Q4_K_M*.gguf" -print -quit 2>/dev/null)
    MODEL_LABEL="Qwen3.6-35B-A3B MoE"
    ;;
  auto)
    GGUF=$(find "$DENSE_DIR" -name "*Q4_K_M*.gguf" -print -quit 2>/dev/null)
    if [[ -n "$GGUF" ]]; then
      MODEL_LABEL="Qwen3.6-27B Dense"
    else
      GGUF=$(find "$MOE_DIR" -name "*Q4_K_M*.gguf" -print -quit 2>/dev/null)
      MODEL_LABEL="Qwen3.6-35B-A3B MoE"
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
echo -e "  ${GRY}Context: $CTX | KV: Q4_0 | Flash Attention${R}"

# Use Python API directly to set logits_all=False
# (server CLI defaults logits_all=True which OOMs on large ctx × 248K vocab)
nohup "$VENV/bin/python" -c "
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings
import uvicorn

model = ModelSettings(
    model='$GGUF',
    model_alias='qwen3.6-35b-uncensored',
    n_gpu_layers=-1,
    n_ctx=$CTX,
    type_k=2,
    type_v=2,
    flash_attn=True,
    logits_all=False,
    chat_format='chatml',
)

server = ServerSettings(host='0.0.0.0', port=$PORT)
app = create_app(server_settings=server, model_settings=[model])
uvicorn.run(app, host='0.0.0.0', port=$PORT, log_level='info')
" >"$LOG" 2>&1 &
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
