#!/usr/bin/env bash
# scripts/setup-qwen36-moe.sh — download Qwen3.6-35B-A3B uncensored GGUF
# MoE model: 35B total / 3B active. Q4_K_M fits in 24GB VRAM (~21.2 GB).
set -euo pipefail

MODEL_DIR="$HOME/models/qwen3.6-35b-a3b-heretic"
REPO="llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF"
QUANT="${1:-Q4_K_M}"
BOLD="\033[1m"; GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

echo -e "\n${BOLD}Downloading Qwen3.6-35B-A3B Uncensored (Heretic) — ${QUANT}${R}"
echo -e "  ${GRY}Repo: $REPO${R}"
echo -e "  ${GRY}Dest: $MODEL_DIR${R}\n"

mkdir -p "$MODEL_DIR"

FILENAME="Qwen3.6-35B-A3B-uncensored-heretic-${QUANT}.gguf"
TARGET="$MODEL_DIR/$FILENAME"

if [[ -f "$TARGET" ]]; then
  SIZE=$(du -h "$TARGET" | cut -f1)
  echo -e "  ${GREEN}Already downloaded${R} ($SIZE): $TARGET"
  exit 0
fi

echo -e "  Downloading ${QUANT} (~21 GB)...\n"
hf download "$REPO" "$FILENAME" \
  --local-dir "$MODEL_DIR"

FOUND=$(find "$MODEL_DIR" -name "*${QUANT}*.gguf" -print -quit)
if [[ -n "$FOUND" ]]; then
  SIZE=$(du -h "$FOUND" | cut -f1)
  echo -e "\n  ${GREEN}Done${R} ($SIZE): $FOUND"
else
  echo -e "\n  ${YEL}Download may have failed. Check $MODEL_DIR${R}"
  exit 1
fi
