#!/usr/bin/env bash
# llm start — full local AI stack
set -euo pipefail

ROOT="$HOME/local-llm"
GREEN="\033[32m"; YEL="\033[33m"; GRY="\033[90m"; R="\033[0m"; BOLD="\033[1m"

echo -e "\n${BOLD}Local AI Stack${R}"

bash "$ROOT/inference/dflash/serve.sh"
# Add SGLang models here as needed:
# bash "$ROOT/inference/sglang/serve.sh" gpt2-xl
bash "$ROOT/gateway/bifrost/serve.sh"
# bash "$ROOT/observability/langfuse/serve.sh"   # coming soon
# bash "$ROOT/frontend/chainlit/serve.sh"         # coming soon

echo -e "\n${GREEN}  Bifrost UI:  http://localhost:8080${R}"
echo -e "${GRY}  DFlash:      http://localhost:8000/v1${R}"
echo -e "${GRY}  SGLang:      http://localhost:8001/v1  (start with: llm-sglang gpt2-xl)${R}\n"

xdg-open "http://localhost:8080" &>/dev/null &
