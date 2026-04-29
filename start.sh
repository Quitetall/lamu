#!/usr/bin/env bash
# llm start — DFlash server + LibreChat
set -euo pipefail

DFLASH_DIR="$HOME/local-llm/lucebox-hub/dflash"
LIBRECHAT_DIR="$HOME/local-llm/librechat"
SERVER_PORT=8000
UI_PORT=3080
PID_FILE="/tmp/dflash-server.pid"
LOG="$HOME/local-llm/server.log"

GREEN="\033[32m"; YEL="\033[33m"; GRY="\033[90m"; R="\033[0m"; BOLD="\033[1m"

wait_for() {
  local url=$1 label=$2 retries=${3:-30} delay=${4:-1}
  echo -n "  waiting for $label"
  for _ in $(seq 1 "$retries"); do
    if curl -sf "$url" &>/dev/null; then echo -e " ${GREEN}ready${R}"; return 0; fi
    echo -n "."; sleep "$delay"
  done
  echo -e " ${YEL}timeout — check logs${R}"
}

echo -e "\n${BOLD}Luce DFlash + LibreChat${R}"

# ── 1. DFlash API server ───────────────────────────────────────────────────
if curl -sf "http://localhost:$SERVER_PORT/v1/models" &>/dev/null; then
  echo -e "  DFlash server  ${GRY}already running on :$SERVER_PORT${R}"
else
  echo -e "  Starting DFlash server ${GRY}(log: $LOG)${R}"
  cd "$DFLASH_DIR"
  nohup .venv/bin/python scripts/server.py --port "$SERVER_PORT" \
    >"$LOG" 2>&1 &
  echo $! >"$PID_FILE"
  wait_for "http://localhost:$SERVER_PORT/v1/models" "server" 60 2
fi

# ── 2. LibreChat (podman-compose) ──────────────────────────────────────────
cd "$LIBRECHAT_DIR"
if podman ps --format '{{.Names}}' 2>/dev/null | grep -q '^librechat$'; then
  echo -e "  LibreChat      ${GRY}already running on :$UI_PORT${R}"
else
  echo "  Starting LibreChat..."
  podman-compose up -d 2>/dev/null
  wait_for "http://localhost:$UI_PORT" "LibreChat" 90 3
fi

# ── 3. Open browser ────────────────────────────────────────────────────────
xdg-open "http://localhost:$UI_PORT" &>/dev/null &

echo -e "\n${GREEN}  http://localhost:$UI_PORT${R}  ${GRY}(server log: llm log)${R}\n"
