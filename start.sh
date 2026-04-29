#!/usr/bin/env bash
# llm start — full local AI stack
set -euo pipefail

ROOT="$HOME/local-llm"
GREEN="\033[32m"; YEL="\033[33m"; GRY="\033[90m"; R="\033[0m"; BOLD="\033[1m"

echo -e "\n${BOLD}Local AI Stack${R}"

bash "$ROOT/inference/dflash/serve.sh"
bash "$ROOT/gateway/bifrost/serve.sh"
bash "$ROOT/observability/langfuse/serve.sh"
bash "$ROOT/frontend/chainlit/serve.sh"

echo -e "\n${GREEN}  Chainlit:    http://localhost:7860${R}"
echo -e "${GRY}  Bifrost UI:  http://localhost:8080${R}"
echo -e "${GRY}  Langfuse:    http://localhost:3000${R}"
echo -e "${GRY}  DFlash:      http://localhost:8000/v1${R}"
echo -e "${GRY}  SGLang:      http://localhost:8001/v1  (llm-sglang gpt2-xl)${R}\n"

xdg-open "http://localhost:7860" &>/dev/null &
