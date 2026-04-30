#!/usr/bin/env bash
# scripts/serve-vllm-qwen36.sh — serve Qwen3.6-27B uncensored via vLLM
#
# Uses a pre-quantized AWQ INT4 model (~14GB VRAM).
# FP8 KV cache + DeltaNet hybrid (only 16/64 layers need KV) = large context.
# Native tool calling + reasoning parser.
#
# First run downloads ~14GB from HuggingFace.
set -euo pipefail

ROOT="$HOME/local-llm"
VENV="$ROOT/.venv"
PORT=8020
# Prefer local heretic GPTQ/AWQ if available, fall back to HuggingFace
if [[ -d "$HOME/models/qwen3.6-27b-heretic-gptq" ]]; then
  MODEL="$HOME/models/qwen3.6-27b-heretic-gptq"
  QUANT="gptq"
elif [[ -d "$HOME/models/qwen3.6-27b-heretic-awq" ]]; then
  MODEL="$HOME/models/qwen3.6-27b-heretic-awq"
  QUANT="awq"
else
  MODEL="zhiqing/Huihui-Qwen3.6-27B-abliterated-AWQ"
  QUANT="awq"
fi
PID_FILE="/tmp/qwen36-server.pid"
LOG="/tmp/qwen36-vllm.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

CTX="${1:-262144}"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  Qwen3.6  ${GRY}already running on :$PORT${R}"
  exit 0
fi

echo -e "  Starting Qwen3.6-27B via vLLM ${GRY}(log: $LOG)${R}"
echo -e "  ${GRY}Model: $MODEL (AWQ INT4)${R}"
echo -e "  ${GRY}Context: $CTX tokens${R}"

nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "qwen3.6-35b-uncensored" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --max-model-len "$CTX" \
  --quantization "$QUANT" \
  --kv-cache-dtype fp8_e5m2 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --dtype half \
  --enforce-eager \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for vLLM"
for _ in $(seq 1 300); do
  if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    exit 0
  fi
  if ! kill -0 "$(cat $PID_FILE 2>/dev/null)" 2>/dev/null; then
    echo -e " ${YEL}crashed — check $LOG${R}"
    if grep -q "OutOfMemory\|CUDA out of memory\|OOM" "$LOG" 2>/dev/null; then
      echo -e "  ${YEL}OOM. Try lower context:${R}"
      echo -e "  ${GRY}bash $0 131072${R}"
      echo -e "  ${GRY}bash $0 65536${R}"
    fi
    exit 1
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
