#!/usr/bin/env bash
# scripts/setup-qwen36-dense.sh — download Qwen3.6-27B dense uncensored GGUF
# Dense 27B model. Q4_K_M ~16 GB — fits easily on 4090 with room for KV cache.
# Higher benchmarks than the MoE variant across the board.
set -euo pipefail

MODEL_DIR="$HOME/models/qwen3.6-27b-heretic"
REPO="llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF"
QUANT="${1:-Q4_K_M}"
BOLD="\033[1m"; GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

echo -e "\n${BOLD}Downloading Qwen3.6-27B Dense Uncensored (Heretic v2) — ${QUANT}${R}"
echo -e "  ${GRY}Repo: $REPO${R}"
echo -e "  ${GRY}Dest: $MODEL_DIR${R}\n"

mkdir -p "$MODEL_DIR"

FILENAME="Qwen3.6-27B-uncensored-heretic-v2-${QUANT}.gguf"
TARGET="$MODEL_DIR/$FILENAME"

if [[ -f "$TARGET" ]]; then
  SIZE=$(du -h "$TARGET" | cut -f1)
  echo -e "  ${GREEN}Already downloaded${R} ($SIZE): $TARGET"
  exit 0
fi

echo -e "  Downloading ${QUANT} (~16 GB for Q4_K_M)...\n"
huggingface-cli download "$REPO" "$FILENAME" \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False

FOUND=$(find "$MODEL_DIR" -name "*${QUANT}*.gguf" -print -quit)
if [[ -n "$FOUND" ]]; then
  SIZE=$(du -h "$FOUND" | cut -f1)
  echo -e "\n  ${GREEN}Done${R} ($SIZE): $FOUND"
else
  echo -e "\n  ${YEL}Download may have failed. Check $MODEL_DIR${R}"
  exit 1
fi
