#!/usr/bin/env bash
# scripts/serve-vllm.sh — start vLLM via club-3090 for Qwen3.6-27B
# Serves on :8020 (OpenAI-compatible). Bifrost routes vllm/* models here.
set -euo pipefail

ROOT="$HOME/local-llm"
CLUB_DIR="$ROOT/deps/club-3090"
PORT=8020
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  vLLM  ${GRY}already running on :$PORT${R}"
  exit 0
fi

if [[ ! -d "$CLUB_DIR" ]]; then
  echo -e "${YEL}club-3090 not set up.${R} Run: bash $ROOT/scripts/setup-club3090.sh"
  exit 1
fi

cd "$CLUB_DIR"

# Use the club-3090 launch script with the default single-card config.
# Override port to 8020 so it doesn't conflict with DFlash on 8000.
# The launch script handles docker/podman, model loading, and health checks.
echo -e "  Starting vLLM via club-3090 ${GRY}(port $PORT)${R}"

# club-3090's launch.sh uses docker compose. Set the port via env.
export VLLM_PORT="$PORT"

# Try the non-interactive variant launch
if bash scripts/switch.sh vllm/default 2>/dev/null; then
  echo -e "  vLLM  ${GREEN}ready on :$PORT${R}"
else
  # Fallback: launch directly
  echo -e "  ${GRY}Falling back to direct launch...${R}"
  bash scripts/launch.sh --variant vllm/default
fi

echo -n "  waiting for vLLM"
for _ in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    exit 0
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check: podman logs (or docker logs)${R}"
