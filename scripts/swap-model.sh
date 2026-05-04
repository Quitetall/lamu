#!/usr/bin/env bash
# scripts/swap-model.sh — swap between qwen3.6 (ngram-mod) and qwen3.5 (dflash)
# Usage: swap-model.sh [qwen36|dflash]
set -euo pipefail

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; RED="\033[31m"; R="\033[0m"

kill_all_models() {
  pkill -9 -f "llama-server.*8020" 2>/dev/null || true
  pkill -9 -f "dflash.*8000\|server.py.*8000" 2>/dev/null || true
  sleep 2
}

case "${1:-}" in
  dflash|qwen35|fast)
    echo -e "  ${YEL}Swapping to DFlash (Qwen3.5-27B, 130+ t/s)${R}"
    kill_all_models
    bash ~/local-llm/scripts/serve-dflash.sh
    echo -e "  ${GREEN}DFlash ready on :8000${R} — 200+ t/s speculative decoding"
    ;;
  qwen36|prod|ngram)
    echo -e "  ${YEL}Swapping to Qwen3.6-27B (ngram-mod, 40+ t/s warm)${R}"
    kill_all_models
    GGUF=$(ls ~/models/qwen3.6-27b-heretic/*Q4_K_M*.gguf | head -1)
    nohup ~/llama.cpp/build/bin/llama-server \
      --model "$GGUF" \
      --host 0.0.0.0 --port 8020 \
      --ctx-size 131072 --n-gpu-layers 99 --flash-attn on \
      --cache-type-k q4_0 --cache-type-v q4_0 --parallel 1 \
      --spec-type ngram-mod --spec-ngram-mod-n-match 24 \
      --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 48 \
      > /tmp/llama-prod.log 2>&1 &
    echo -n "  waiting for qwen3.6"
    for _ in $(seq 1 30); do
      curl -sf http://localhost:8020/health &>/dev/null && break
      echo -n "."; sleep 2
    done
    echo -e " ${GREEN}ready on :8020${R}"
    ;;
  status)
    echo -e "  Qwen3.6 :8020  $(curl -sf http://localhost:8020/health &>/dev/null && echo -e "${GREEN}UP${R}" || echo -e "${GRY}down${R}")"
    echo -e "  DFlash  :8000  $(curl -sf http://localhost:8000/v1/models &>/dev/null && echo -e "${GREEN}UP${R}" || echo -e "${GRY}down${R}")"
    ;;
  *)
    echo "Usage: swap-model.sh [qwen36|dflash|status]"
    echo "  qwen36/prod/ngram  — Qwen3.6-27B uncensored, 131K ctx, ngram-mod (40+ t/s warm)"
    echo "  dflash/qwen35/fast — Qwen3.5-27B, DFlash speculative (130-200+ t/s)"
    echo "  status             — show what's running"
    ;;
esac
