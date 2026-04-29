#!/usr/bin/env bash
# llm stop — shut down full local AI stack
ROOT="$HOME/local-llm"
BOLD="\033[1m"; R="\033[0m"

echo -e "\n${BOLD}Stopping Local AI Stack${R}"

bash "$ROOT/scripts/stop-bifrost.sh"
bash "$ROOT/scripts/stop-sglang.sh" gpt2-xl 2>/dev/null || true

if [[ -f /tmp/gpt2-proxy.pid ]]; then
  pid=$(cat /tmp/gpt2-proxy.pid)
  kill "$pid" 2>/dev/null && echo -e "  GPT-2 proxy  \033[90mstopped (pid $pid)\033[0m" || true
  rm -f /tmp/gpt2-proxy.pid
fi

if [[ -f /tmp/dflash-server.pid ]]; then
  pid=$(cat /tmp/dflash-server.pid)
  kill "$pid" 2>/dev/null && echo -e "  DFlash   \033[90mstopped (pid $pid)\033[0m" || true
  rm -f /tmp/dflash-server.pid
fi

if [[ -f /tmp/qwen36-server.pid ]]; then
  pid=$(cat /tmp/qwen36-server.pid)
  kill "$pid" 2>/dev/null && echo -e "  Qwen3.6  \033[90mstopped (pid $pid)\033[0m" || true
  rm -f /tmp/qwen36-server.pid
fi

echo
