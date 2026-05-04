#!/usr/bin/env bash
# scripts/serve-dflash.sh — start DFlash server (Qwen3.5-27B)
set -euo pipefail

DFLASH_DIR="$HOME/local-llm/lucebox-hub/dflash"
PORT=8000
PID_FILE="/tmp/dflash-server.pid"
LOG="/tmp/dflash-server.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  DFlash  ${GRY}already running on :$PORT${R}"
  exit 0
fi

echo -e "  Starting DFlash ${GRY}(log: $LOG)${R}"
cd "$DFLASH_DIR"
# Custom server with tool-calling support (server/dflash.py)
SERVER_PY="$HOME/local-llm/server/dflash.py"
# budget=22 tuned for RTX 3090; on 4090, CUDA fragmentation prevents the 1.85 GB
# rollback cache at budget=22 — reduce to 10 (~0.84 GB) so it fits contiguously.
SERVER_24GB="$HOME/local-llm/server/dflash_24gb.py"
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 nohup "$HOME/local-llm/.venv/bin/python" "$SERVER_24GB" \
  --port "$PORT" --max-ctx 8192 --budget 6 \
  --bin "$DFLASH_DIR/build/test_dflash" \
  --target "$HOME/models/qwen3.6-27b-heretic/Qwen3.6-27B-uncensored-heretic-v2-Q4_K_M.gguf" \
  --draft "$HOME/models/draft-35" \
  >"$LOG" 2>&1 &
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
