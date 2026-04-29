#!/usr/bin/env bash
# gateway/bifrost/serve.sh — start Bifrost AI gateway
set -euo pipefail

PORT=8080
PID_FILE="/tmp/bifrost.pid"
LOG="$HOME/local-llm/gateway/bifrost/bifrost.log"
DATA_DIR="$HOME/local-llm/gateway/bifrost/data"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

mkdir -p "$DATA_DIR"

if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
  echo -e "  Bifrost  ${GRY}already running on :$PORT${R}"
  exit 0
fi

echo -e "  Starting Bifrost ${GRY}(log: $LOG)${R}"
podman run -d \
  --network=host \
  --name bifrost \
  -v "$DATA_DIR":/app/data \
  docker.io/maximhq/bifrost \
  >/dev/null 2>&1 \
|| podman start bifrost >/dev/null 2>&1

echo -n "  waiting for Bifrost"
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    exit 0
  fi
  echo -n "."; sleep 1
done
echo -e " ${YEL}timeout — check $LOG${R}"
