#!/usr/bin/env bash
# llm stop — kill DFlash server + LibreChat
GRY="\033[90m"; R="\033[0m"; BOLD="\033[1m"

echo -e "\n${BOLD}Stopping Luce DFlash + LibreChat${R}"

if [[ -f /tmp/dflash-server.pid ]]; then
  pid=$(cat /tmp/dflash-server.pid)
  if kill "$pid" 2>/dev/null; then
    echo -e "  DFlash server  ${GRY}stopped (pid $pid)${R}"
  else
    echo -e "  DFlash server  ${GRY}already stopped${R}"
  fi
  rm -f /tmp/dflash-server.pid
fi

cd "$HOME/local-llm/librechat"
if podman ps --format '{{.Names}}' 2>/dev/null | grep -q '^librechat$'; then
  podman-compose down 2>/dev/null
  echo -e "  LibreChat      ${GRY}stopped${R}"
else
  echo -e "  LibreChat      ${GRY}not running${R}"
fi

echo
