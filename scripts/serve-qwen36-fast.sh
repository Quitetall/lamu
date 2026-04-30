#!/usr/bin/env bash
# scripts/serve-qwen36-fast.sh — native llama-server with ngram-mod speculation
#
# Uses the optimized C++ llama-server binary (no Python overhead) with
# ngram-mod speculative decoding for 50-137 t/s on RTX 4090.
#
# ngram-mod: hash-based pattern matching from conversation history.
# No draft model needed. Gets faster as the conversation grows.
# Especially good for code generation (repetitive patterns).
#
# Usage: serve-qwen36-fast.sh [context]
set -euo pipefail

LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
MODELS_DIR="$HOME/models/qwen3.6-27b-heretic"
PORT=8020
PID_FILE="/tmp/qwen36-server.pid"
LOG="/tmp/qwen36-server.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

CTX="${1:-262144}"

if [[ ! -f "$LLAMA_SERVER" ]]; then
  echo -e "${YEL}llama-server not built.${R} Run:"
  echo -e "  ${GRY}cd ~/llama.cpp && cmake --build build --config Release -j4 --target llama-server${R}"
  exit 1
fi

if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
  echo -e "  Qwen3.6  ${GRY}already running on :$PORT${R}"
  exit 0
fi

# Find best GGUF (prefer Q5_K_S for quality)
GGUF=""
for q in Q5_K_S Q5_K_M Q4_K_M Q4_K_S; do
  GGUF=$(find "$MODELS_DIR" -name "*${q}*.gguf" -print -quit 2>/dev/null)
  [[ -n "$GGUF" ]] && break
done

if [[ -z "$GGUF" ]]; then
  echo -e "${YEL}No model found in $MODELS_DIR${R}"
  exit 1
fi

# KV cache type based on quant
KV_TYPE="q4_0"
if [[ "$GGUF" == *"Q5_K"* ]] && [[ "$CTX" -le 108000 ]]; then
  KV_TYPE="q8_0"
fi

echo -e "  Starting Qwen3.6 (native, ngram-mod) ${GRY}(log: $LOG)${R}"
echo -e "  ${GRY}Model: $(basename "$GGUF")${R}"
echo -e "  ${GRY}Context: $CTX | KV: $KV_TYPE | Speculation: ngram-mod${R}"

nohup "$LLAMA_SERVER" \
  -m "$GGUF" \
  --alias "qwen3.6-27b-uncensored" \
  --host 0.0.0.0 \
  --port "$PORT" \
  -ngl 99 \
  --ctx-size "$CTX" \
  --cache-type-k "$KV_TYPE" \
  --cache-type-v "$KV_TYPE" \
  --flash-attn on \
  --spec-type ngram-mod \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 12 \
  --spec-ngram-mod-n-max 48 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for Qwen3.6"
for _ in $(seq 1 90); do
  if curl -sf "http://localhost:$PORT/health" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    echo -e "  ${GRY}ngram-mod: gets faster as conversation grows (50→137 t/s)${R}"
    exit 0
  fi
  if ! kill -0 "$(cat $PID_FILE 2>/dev/null)" 2>/dev/null; then
    echo -e " ${YEL}crashed${R}"; tail -5 "$LOG"; exit 1
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
