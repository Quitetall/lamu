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

# Context presets
resolve_ctx() {
  case "${1:-}" in
    lightning|32k)  echo 32768 ;;
    fast|64k)       echo 65536 ;;
    med|med-ctx|131k) echo 131072 ;;
    big|big-ctx|262k) echo 262144 ;;
    "")             echo 131072 ;;  # default
    *)              echo "$1" ;;     # raw number
  esac
}

CTX=$(resolve_ctx "${2:-}")

case "${1:-}" in
  qwen36|3.6|prod|smart)
    echo -e "  ${YEL}Swapping to Qwen3.6-27B uncensored (ctx=$CTX)${R}"
    kill_27b
    serve_llama "$(ls ~/models/qwen3.6-27b-heretic/*Q4_K_M*.gguf | head -1)" "qwen3.6" "$CTX"
    ;;
  qwen35|3.5)
    echo -e "  ${YEL}Swapping to Qwen3.5-27B (ctx=$CTX)${R}"
    kill_27b
    serve_llama "$HOME/models/qwen3.5-27b-gguf/Qwen3.5-27B-Q4_K_M.gguf" "qwen3.5" "$CTX"
    ;;
  dflash)
    echo -e "  ${YEL}Swapping to DFlash (Qwen3.5-27B speculative, 130+ t/s)${R}"
    kill_27b
    bash ~/local-llm/scripts/serve-dflash.sh
    echo -e "  ${GREEN}DFlash ready on :8000${R}"
    ;;
  gpt2|2021|retro)
    echo -e "  ${YEL}Swapping to GPT-2 XL (the 2021 experience)${R}"
    kill_27b
    nohup "$LLAMA" \
      --model "$HOME/models/gpt2-xl-gguf/gpt2-xl.Q4_K_M.gguf" \
      --host 0.0.0.0 --port $PORT \
      --ctx-size 1024 -ngl 99 --parallel 1 \
      > /tmp/llama-prod.log 2>&1 &
    echo -n "  waiting for gpt2"
    for _ in $(seq 1 15); do
      curl -sf http://localhost:$PORT/health &>/dev/null && break
      echo -n "."; sleep 1
    done
    echo -e " ${GREEN}ready on :$PORT${R} (shitty presets: shitty-inferkit, shitty-terrible, etc.)"
    ;;
  status)
    echo -e "  Main    :$PORT  $(curl -sf http://localhost:$PORT/health &>/dev/null && echo -e "${GREEN}UP${R}" || echo -e "${GRY}down${R}")"
    echo -e "  DFlash  :8000  $(curl -sf http://localhost:8000/v1/models &>/dev/null && echo -e "${GREEN}UP${R}" || echo -e "${GRY}down${R}")"
    echo -e "  0.8B    :8001  $(curl -sf http://localhost:8001/health &>/dev/null && echo -e "${GREEN}UP${R}" || echo -e "${GRY}down${R}")"
    ;;
  *)
    echo "Usage: swap-model.sh [qwen36|qwen35|dflash|gpt2|status]"
    echo ""
    echo "  qwen36/3.6 [ctx]  — Qwen3.6-27B uncensored heretic (49+ t/s)"
    echo "  qwen35/3.5 [ctx]  — Qwen3.5-27B (49+ t/s)"
    echo "  dflash             — DFlash speculative decode (106 t/s)"
    echo "  gpt2/2021/retro    — GPT-2 XL 1.5B (the 2021 InferKit experience)"
    echo "  status             — show what's running"
    echo ""
    echo "Context presets (2nd arg):"
    echo "  lightning / 32k   — 32K ctx (fastest inference)"
    echo "  fast / 64k        — 64K ctx"
    echo "  med / 131k        — 131K ctx (default)"
    echo "  big / 262k        — 262K ctx (max, tight VRAM)"
    echo ""
    echo "Examples:"
    echo "  swap-model.sh 3.6 lightning   # Qwen3.6 @ 32K (fastest)"
    echo "  swap-model.sh 3.6 big         # Qwen3.6 @ 262K (max context)"
    echo "  swap-model.sh 3.6 65536       # raw number also works"
    echo ""
    echo ""
    echo "Sidecar (runs alongside 27B on :8001):"
    echo "  Use 'just sidecar fast' for 4B (200 t/s, smart)"
    echo "  Use 'just sidecar lobo' for 0.8B megakernel (494 t/s, lobotomized)"
    ;;
esac
