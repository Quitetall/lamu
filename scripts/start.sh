#!/usr/bin/env bash
# llm start — full local AI stack
set -euo pipefail

ROOT="$HOME/local-llm"
GREEN="\033[32m"; YEL="\033[33m"; GRY="\033[90m"; R="\033[0m"; BOLD="\033[1m"

echo -e "\n${BOLD}Local AI Stack${R}"

bash "$ROOT/scripts/serve-qwen36.sh"
bash "$ROOT/scripts/serve-dflash.sh"
bash "$ROOT/scripts/serve-bifrost.sh"
bash "$ROOT/scripts/serve-langfuse.sh"
bash "$ROOT/web/serve.sh"

echo -e "\n${GREEN}  Chainlit:    http://localhost:7860${R}"
echo -e "${GRY}  Bifrost UI:  http://localhost:8080${R}"
echo -e "${GRY}  Langfuse:    http://localhost:3000${R}"
echo -e "${GRY}  Qwen3.6:     http://localhost:8020/v1  (uncensored worker)${R}"
echo -e "${GRY}  DFlash:      http://localhost:8000/v1  (Qwen3.5-27B)${R}\n"

xdg-open "http://localhost:7860" &>/dev/null &
