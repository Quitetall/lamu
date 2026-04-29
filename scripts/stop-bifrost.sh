#!/usr/bin/env bash
GRY="\033[90m"; R="\033[0m"
if podman ps --format '{{.Names}}' 2>/dev/null | grep -q '^bifrost$'; then
  podman stop bifrost >/dev/null
  echo -e "  Bifrost  ${GRY}stopped${R}"
else
  echo -e "  Bifrost  ${GRY}not running${R}"
fi
