#!/usr/bin/env bash
# scripts/serve-eagle.sh — start EAGLE speculative decoding server
# Loads main model (4-bit) + trained EAGLE head on same GPU.
# Uses hidden states from the main model for draft predictions.
set -euo pipefail

ROOT="$HOME/local-llm"
PORT=8020
PID_FILE="/tmp/qwen36-server.pid"
LOG="/tmp/qwen36-eagle.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
  echo -e "  EAGLE  ${GRY}already running on :$PORT${R}"
  exit 0
fi

if [[ ! -f "$HOME/models/qwen3.6-27b-heretic-eagle/eagle_head/eagle_head_best.pt" ]]; then
  echo -e "${YEL}EAGLE head not found.${R} Train it first: bash scripts/train-eagle-head.sh"
  exit 1
fi

echo -e "  Starting EAGLE speculative server ${GRY}(log: $LOG)${R}"
nohup "$ROOT/.venv/bin/python" -m server.eagle_server --port "$PORT" >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for EAGLE"
for _ in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    curl -s "http://localhost:$PORT/health" | python3 -c "import sys,json; h=json.load(sys.stdin); print(f'  \033[90m{h[\"engine\"]} | acceptance: {h[\"acceptance_rate\"]}\033[0m')"
    exit 0
  fi
  if ! kill -0 "$(cat $PID_FILE 2>/dev/null)" 2>/dev/null; then
    echo -e " ${YEL}crashed${R}"; tail -5 "$LOG"; exit 1
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
