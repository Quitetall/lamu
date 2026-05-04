#!/usr/bin/env bash
# scripts/serve-megakernel.sh — Qwen3.5-0.8B megakernel server on :8001
# 462+ t/s on RTX 4090. Runs alongside 27B model (~1.5 GB VRAM).
set -euo pipefail

PORT=8001
LOG="/tmp/megakernel-server.log"
MEGA_DIR="$HOME/local-llm/lucebox-hub/megakernel"
GRY="\033[90m"; GREEN="\033[32m"; R="\033[0m"

if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
  echo -e "  Megakernel ${GRY}already running on :$PORT${R}"
  exit 0
fi

echo -e "  Starting Megakernel 0.8B ${GRY}(log: $LOG)${R}"
cd "$MEGA_DIR"

nohup "$HOME/local-llm/.venv/bin/python" "$HOME/local-llm/server/megakernel_server.py" \
  --port "$PORT" > "$LOG" 2>&1 &
echo $! > /tmp/megakernel.pid

echo -n "  waiting for megakernel"
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
    echo -e " ${GREEN}ready (462+ t/s)${R}"
    exit 0
  fi
  echo -n "."; sleep 1
done
echo -e " timeout — check $LOG"
