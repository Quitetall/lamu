#!/usr/bin/env bash
# inference/sglang/serve.sh — start SGLang server for a model from models.yaml
# Usage: ./serve.sh <model-id>
#   ./serve.sh gpt2-xl
set -euo pipefail

MODEL_ID="${1:-gpt2-xl}"
VENV="$HOME/local-llm/.venv"
PID_FILE="/tmp/sglang-${MODEL_ID}.pid"
LOG="$HOME/local-llm/inference/sglang/${MODEL_ID}.log"

GRY="\033[90m"; R="\033[0m"; GREEN="\033[32m"; YEL="\033[33m"

# ── Model registry (must match models.yaml) ───────────────────────────────
declare -A HF_MODEL=( [gpt2-xl]="gpt2-xl" )
declare -A PORT=(     [gpt2-xl]="8001" )
declare -A CTX=(      [gpt2-xl]="1024" )

if [[ -z "${HF_MODEL[$MODEL_ID]+x}" ]]; then
  echo "Unknown model: $MODEL_ID"
  echo "Known: ${!HF_MODEL[*]}"
  exit 1
fi

HF="${HF_MODEL[$MODEL_ID]}"
PORT_N="${PORT[$MODEL_ID]}"
CTX_N="${CTX[$MODEL_ID]}"

# ── Check if already running ──────────────────────────────────────────────
if curl -sf "http://localhost:$PORT_N/v1/models" &>/dev/null; then
  echo -e "  SGLang $MODEL_ID  ${GRY}already running on :$PORT_N${R}"
  exit 0
fi

# ── Launch ────────────────────────────────────────────────────────────────
echo -e "  Starting SGLang: ${GRY}$HF on :$PORT_N (ctx $CTX_N)${R}"
nohup "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$HF" \
  --port "$PORT_N" \
  --host 0.0.0.0 \
  --context-length "$CTX_N" \
  --dtype float16 \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for SGLang"
for _ in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT_N/v1/models" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    exit 0
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
