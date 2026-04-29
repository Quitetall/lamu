#!/usr/bin/env bash
# scripts/setup-club3090.sh — clone + set up club-3090 vLLM serving
# Adds Qwen3.6-27B via vLLM as a second inference backend alongside DFlash.
set -euo pipefail

ROOT="$HOME/local-llm"
CLUB_DIR="$ROOT/deps/club-3090"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"; BOLD="\033[1m"

echo -e "\n${BOLD}Setting up club-3090 (vLLM serving for Qwen3.6-27B)${R}\n"

# ── Clone ────────────────────────────────────────────────────────────────
if [[ -d "$CLUB_DIR" ]]; then
  echo -e "  club-3090  ${GRY}already cloned at $CLUB_DIR${R}"
  cd "$CLUB_DIR" && git pull --ff-only 2>/dev/null || true
else
  echo -e "  Cloning noonghunna/club-3090..."
  git clone https://github.com/noonghunna/club-3090.git "$CLUB_DIR"
fi

cd "$CLUB_DIR"

# ── Download model + patches ─────────────────────────────────────────────
if [[ ! -f "$CLUB_DIR/.setup_done" ]]; then
  echo -e "\n  Running setup (downloads model ~20 GB)..."
  bash scripts/setup.sh qwen3.6-27b
  touch "$CLUB_DIR/.setup_done"
else
  echo -e "  Model setup  ${GRY}already complete${R}"
fi

echo -e "\n${GREEN}club-3090 ready.${R}"
echo -e "  Start vLLM:  ${GRY}bash $ROOT/scripts/serve-vllm.sh${R}"
echo -e "  Or use:      ${GRY}bash $CLUB_DIR/scripts/launch.sh${R}"
