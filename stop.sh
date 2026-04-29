#!/usr/bin/env bash
# llm stop — shut down full local AI stack
ROOT="$HOME/local-llm"
BOLD="\033[1m"; R="\033[0m"

echo -e "\n${BOLD}Stopping Local AI Stack${R}"

bash "$ROOT/gateway/bifrost/stop.sh"
bash "$ROOT/inference/sglang/stop.sh" gpt2-xl 2>/dev/null || true

if [[ -f /tmp/dflash-server.pid ]]; then
  pid=$(cat /tmp/dflash-server.pid)
  kill "$pid" 2>/dev/null && echo -e "  DFlash   \033[90mstopped (pid $pid)\033[0m" || true
  rm -f /tmp/dflash-server.pid
fi

echo
