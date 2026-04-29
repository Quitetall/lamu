#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YEL='\033[0;33m'
NC='\033[0m'

LANGFUSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEALTH_URL="http://127.0.0.1:3000/api/public/health"

if curl -sf "$HEALTH_URL" &>/dev/null; then
  echo -e "${GREEN}Langfuse already running at http://localhost:3000${NC}"
  exit 0
fi

cd "$LANGFUSE_DIR"
podman-compose up -d

echo "Waiting for Langfuse to become ready..."
ELAPSED=0
INTERVAL=2
TIMEOUT=120

while true; do
  if curl -sf "$HEALTH_URL" &>/dev/null; then
    echo -e "${GREEN}Langfuse ready at http://localhost:3000${NC}"
    exit 0
  fi

  ELAPSED=$((ELAPSED + INTERVAL))
  if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    echo -e "${YEL}Timed out waiting for Langfuse after ${TIMEOUT}s. Check logs: podman-compose logs -f${NC}"
    exit 1
  fi

  sleep "$INTERVAL"
done
