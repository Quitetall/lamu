#!/usr/bin/env bash
# scripts/serve-comfyui.sh — start ComfyUI for image/video generation
# Serves node editor on :8188. FLUX needs ~17GB, SDXL ~8GB.
# Can't run simultaneously with LLMs on one 4090 — use just swap-comfyui.
set -euo pipefail

COMFY_DIR="$HOME/ComfyUI"
PORT=8188
PID_FILE="/tmp/comfyui.pid"
LOG="/tmp/comfyui.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

if curl -sf "http://localhost:$PORT/system_stats" &>/dev/null; then
  echo -e "  ComfyUI  ${GRY}already running on :$PORT${R}"
  exit 0
fi

if [[ ! -d "$COMFY_DIR" ]]; then
  echo -e "${YEL}ComfyUI not found.${R} Run: just setup-comfyui"
  exit 1
fi

echo -e "  Starting ComfyUI ${GRY}(log: $LOG)${R}"
cd "$COMFY_DIR"
nohup .venv/bin/python main.py --port "$PORT" --listen >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for ComfyUI"
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/system_stats" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    echo -e "  ${GRY}http://localhost:$PORT${R}"
    exit 0
  fi
  echo -n "."; sleep 1
done
echo -e " ${YEL}timeout — check $LOG${R}"
