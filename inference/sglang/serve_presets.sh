#!/usr/bin/env bash
# inference/sglang/serve_presets.sh — start GPT-2 preset proxy on :9001
set -euo pipefail

PORT=9001
VENV="$HOME/local-llm/.venv"
PID_FILE="/tmp/gpt2-proxy.pid"
LOG="$HOME/local-llm/inference/sglang/gpt2_proxy.log"
SCRIPT="$HOME/local-llm/inference/sglang/gpt2_proxy.py"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
  echo -e "  GPT-2 proxy  ${GRY}already running on :$PORT${R}"
  exit 0
fi

echo -e "  Starting GPT-2 preset proxy ${GRY}(log: $LOG)${R}"
nohup "$VENV/bin/python" "$SCRIPT" >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for proxy"
for _ in $(seq 1 15); do
  if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    exit 0
  fi
  echo -n "."; sleep 1
done
echo -e " ${YEL}timeout — check $LOG${R}"
