#!/usr/bin/env bash
# llm-chat — terminal REPL via Bifrost
GRY="\033[90m"; YEL="\033[33m"; R="\033[0m"

if ! curl -sf http://localhost:8080/health &>/dev/null; then
  echo -e "${YEL}Bifrost is not running.${R} ${GRY}Start the stack first: llm${R}"
  exit 1
fi

exec python3 "$HOME/local-llm/cli/chat_repl.py"
