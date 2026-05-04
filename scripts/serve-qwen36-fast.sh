#!/usr/bin/env bash
# scripts/serve-qwen36-fast.sh — Maximum speed: DFlash speculative decoding
#
# Qwen3.6-27B heretic @ 80+ t/s on RTX 4090
# Uses llama.cpp DFlash PR branch (~/llama.cpp on dflash-pr branch)
# Draft: z-lab/Qwen3.6-27B-DFlash Q4_K_M (974 MB)
#
# NOTE: DFlash in llama-server crashes on 2nd request (PR bug).
#       This script uses llama-speculative-simple (one-shot mode).
#       For persistent server, use `just swap 3.6` (ngram-mod only, 49.5 t/s).
set -euo pipefail

LLAMA_DIR="$HOME/llama.cpp"
TARGET=$(ls "$HOME/models/qwen3.6-27b-heretic/"*Q4_K_M*.gguf | head -1)
DRAFT="$HOME/models/qwen3.6-dflash-gguf/dflash-3.6-q4km.gguf"
BIN="$LLAMA_DIR/build/bin/llama-speculative-simple"

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

if [ ! -f "$BIN" ]; then
  echo -e "  ${YEL}DFlash binary not found. Build with:${R}"
  echo "  cd ~/llama.cpp && git checkout dflash-pr"
  echo "  cd build && cmake --build . --target llama-speculative-simple -j\$(nproc)"
  exit 1
fi

if [ ! -f "$DRAFT" ]; then
  echo -e "  ${YEL}DFlash draft not found at $DRAFT${R}"
  echo "  Convert: python convert_hf_to_gguf.py ~/models/qwen3.6-27b-dflash-draft/ --outtype f16 ..."
  exit 1
fi

PROMPT="${1:-Write a Python implementation of quicksort with comments.}"
N_GEN="${2:-256}"

echo -e "  ${GREEN}DFlash${R} spec decode ${GRY}(draft-max=8, Q4_K_M draft)${R}"
echo -e "  ${GRY}prompt: ${PROMPT:0:60}...${R}"

"$BIN" \
  -m "$TARGET" \
  -md "$DRAFT" \
  --dflash --draft-max 8 \
  -p "$PROMPT" \
  -n "$N_GEN" \
  -cd 512 -c 4096 \
  --temp 0 --top-k 1 --seed 42 \
  -ngl 999 -ngld 99 -fa on \
  -ctk q4_0 -ctv q4_0 \
  -ctkd q8_0 -ctvd q8_0 \
  -t 8
