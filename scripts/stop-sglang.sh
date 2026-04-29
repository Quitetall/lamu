#!/usr/bin/env bash
# Stop a running SGLang server
# Usage: ./stop.sh <model-id>
MODEL_ID="${1:-gpt2-xl}"
PID_FILE="/tmp/sglang-${MODEL_ID}.pid"
GRY="\033[90m"; R="\033[0m"

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  kill "$pid" 2>/dev/null && echo -e "  SGLang $MODEL_ID ${GRY}stopped (pid $pid)${R}" || echo -e "  SGLang $MODEL_ID ${GRY}already stopped${R}"
  rm -f "$PID_FILE"
else
  echo -e "  SGLang $MODEL_ID ${GRY}not running${R}"
fi
