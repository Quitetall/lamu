#!/usr/bin/env bash
# scripts/serve-qwen36.sh — start Qwen3.6 uncensored (production server)
# Full 262K context, think-block stripping, health endpoint.
# Usage: serve-qwen36.sh [context_override]
set -euo pipefail

ROOT="$HOME/local-llm"
VENV="$ROOT/.venv"
PORT=8020
PID_FILE="/tmp/qwen36-server.pid"
LOG="/tmp/qwen36-server.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

CTX="${1:-}"

if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
  echo -e "  Qwen3.6  ${GRY}already running on :$PORT${R}"
  exit 0
fi

echo -e "  Starting Qwen3.6 ${GRY}(log: $LOG)${R}"

LLM_CTX="${CTX}" nohup "$VENV/bin/python" "$ROOT/server/serve.py" >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for Qwen3.6"
for _ in $(seq 1 90); do
  if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    # Show what loaded
    curl -s "http://localhost:$PORT/health" | python3 -c "import sys,json; h=json.load(sys.stdin); print(f'  \033[90m{h[\"model\"]} | {h[\"context\"]:,} ctx\033[0m')"
    exit 0
  fi
  if ! kill -0 "$(cat $PID_FILE 2>/dev/null)" 2>/dev/null; then
    echo -e " ${YEL}crashed${R}"
    tail -5 "$LOG"
    exit 1
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
