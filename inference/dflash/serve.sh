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
nohup .venv/bin/python scripts/server.py --port "$PORT" --max-ctx 8192 >"$LOG" 2>&1 &
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
