#!/usr/bin/env bash
# llm-chat — terminal REPL, standalone only (can't share GPU with server)
DFLASH_DIR="$HOME/local-llm/lucebox-hub/dflash"
GRY="\033[90m"; YEL="\033[33m"; R="\033[0m"

if curl -sf http://localhost:8000/v1/models &>/dev/null; then
  echo -e "${YEL}warning:${R} DFlash server is already running and holds the GPU."
  echo -e "${GRY}llm-chat spawns a second model instance — no VRAM left."
  echo -e "Use the browser (http://localhost:3000) or stop the server first with: llm-stop${R}\n"
  read -r -p "Start anyway? (y/N) " ans
  [[ "$ans" =~ ^[Yy]$ ]] || exit 0
fi

cd "$DFLASH_DIR"
.venv/bin/python examples/chat_ux.py
