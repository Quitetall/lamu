#!/usr/bin/env bash
# inference/dflash/serve.sh — start DFlash server (Qwen3.5-27B)
set -euo pipefail

DFLASH_DIR="$HOME/local-llm/lucebox-hub/dflash"
PORT=8000
PID_FILE="/tmp/dflash-server.pid"
LOG="$HOME/local-llm/inference/dflash/server.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  DFlash  ${GRY}already running on :$PORT${R}"
  exit 0
fi

echo -e "  Starting DFlash ${GRY}(log: $LOG)${R}"
cd "$DFLASH_DIR"
# Use the local-llm repo's server.py (has tool-calling support) instead of
# the upstream lucebox-hub copy.
SERVER_PY="$HOME/local-llm/inference/dflash/server.py"
# budget=22 tuned for RTX 3090; on 4090, CUDA fragmentation prevents the 1.85 GB
# rollback cache at budget=22 — reduce to 10 (~0.84 GB) so it fits contiguously.
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 nohup .venv/bin/python "$SERVER_PY" --port "$PORT" --max-ctx 8192 >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for DFlash"
for _ in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    exit 0
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
