#!/usr/bin/env bash
# scripts/swap-model.sh — swap between models (shared GPU, only one 27B at a time)
# Usage: swap-model.sh [qwen36|qwen35|dflash|status]
set -euo pipefail

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"
LLAMA="$HOME/llama.cpp/build/bin/llama-server"
PORT=8020

kill_27b() {
  pkill -9 -f "llama-server.*$PORT" 2>/dev/null || true
  pkill -9 -f "dflash.*8000\|server.py.*8000" 2>/dev/null || true
  sleep 2
}

serve_llama() {
  local gguf="$1" label="$2" ctx="${3:-131072}"
  nohup "$LLAMA" \
    --model "$gguf" \
    --host 0.0.0.0 --port $PORT \
    --ctx-size "$ctx" --n-gpu-layers 99 --flash-attn on \
    --cache-type-k q4_0 --cache-type-v q4_0 --parallel 1 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 24 \
    --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 48 \
    > /tmp/llama-prod.log 2>&1 &
  echo -n "  waiting for $label"
  for _ in $(seq 1 30); do
    curl -sf http://localhost:$PORT/health &>/dev/null && break
    echo -n "."; sleep 2
  done
  echo -e " ${GREEN}ready on :$PORT${R}"
}

case "${1:-}" in
  qwen36|3.6|prod|smart)
    echo -e "  ${YEL}Swapping to Qwen3.6-27B uncensored (ngram-mod, 40+ t/s)${R}"
    kill_27b
    serve_llama "$(ls ~/models/qwen3.6-27b-heretic/*Q4_K_M*.gguf | head -1)" "qwen3.6"
    ;;
  qwen35|3.5)
    echo -e "  ${YEL}Swapping to Qwen3.5-27B (ngram-mod, 40+ t/s)${R}"
    kill_27b
    serve_llama "$HOME/models/qwen3.5-27b-gguf/Qwen3.5-27B-Q4_K_M.gguf" "qwen3.5"
    ;;
  dflash)
    echo -e "  ${YEL}Swapping to DFlash (Qwen3.5-27B speculative, 130+ t/s)${R}"
    kill_27b
    bash ~/local-llm/scripts/serve-dflash.sh
    echo -e "  ${GREEN}DFlash ready on :8000${R}"
    ;;
  status)
    echo -e "  27B     :$PORT  $(curl -sf http://localhost:$PORT/health &>/dev/null && echo -e "${GREEN}UP${R}" || echo -e "${GRY}down${R}")"
    echo -e "  DFlash  :8000  $(curl -sf http://localhost:8000/v1/models &>/dev/null && echo -e "${GREEN}UP${R}" || echo -e "${GRY}down${R}")"
    echo -e "  0.8B    :8001  $(curl -sf http://localhost:8001/health &>/dev/null && echo -e "${GREEN}UP${R}" || echo -e "${GRY}down${R}")"
    ;;
  *)
    echo "Usage: swap-model.sh [qwen36|qwen35|dflash|status]"
    echo ""
    echo "  qwen36/3.6/prod  — Qwen3.6-27B uncensored heretic, 131K ctx (40+ t/s)"
    echo "  qwen35/3.5       — Qwen3.5-27B, 131K ctx (40+ t/s)"
    echo "  dflash            — Qwen3.5-27B DFlash speculative (130-200+ t/s, WIP)"
    echo "  status            — show what's running"
    echo ""
    echo "  Note: 0.8B megakernel (:8001) runs alongside any 27B model"
    ;;
esac
