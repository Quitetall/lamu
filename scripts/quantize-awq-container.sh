#!/usr/bin/env bash
# scripts/quantize-awq-container.sh — GPTQ quantize heretic model in a container
# Uses transformers' built-in GPTQConfig in a clean container.
# Mounts HF cache (model already downloaded) and output dir.
set -euo pipefail

OUTPUT_DIR="$HOME/models/qwen3.6-27b-heretic-gptq"
HF_CACHE="$HOME/.cache/huggingface"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOLD="\033[1m"; GRY="\033[90m"; GREEN="\033[32m"; R="\033[0m"

echo -e "\n${BOLD}GPTQ 4-bit Quantization (containerized)${R}"
echo -e "  ${GRY}Source: llmfan46/Qwen3.6-27B-uncensored-heretic-v2 (from HF cache)${R}"
echo -e "  ${GRY}Output: $OUTPUT_DIR${R}\n"

mkdir -p "$OUTPUT_DIR"

podman run --rm \
  --device nvidia.com/gpu=all \
  --shm-size=16g \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$OUTPUT_DIR:/output" \
  -v "$SCRIPT_DIR/quantize_inner.py:/quantize.py:ro" \
  quantize-local \
  python3 /quantize.py

if [[ -f "$OUTPUT_DIR/config.json" ]]; then
  SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
  echo -e "\n${GREEN}Quantization complete!${R} ($SIZE)"
  echo -e "  ${GRY}Serve with: just serve-vllm${R}"
else
  echo -e "\n\033[33mQuantization may have failed. Check output above.\033[0m"
fi
