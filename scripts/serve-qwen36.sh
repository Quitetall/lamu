#!/usr/bin/env bash
# scripts/serve-qwen36.sh — serve Qwen3.6 uncensored via llama-cpp-python
# Full context on single 4090 with quantized KV cache + flash attention.
#
# Auto-selects best quant available. Supports context override.
#
# Usage:
#   serve-qwen36.sh                    # auto: best quant, optimal context
#   serve-qwen36.sh dense              # force dense model
#   serve-qwen36.sh dense 262144       # force 262K context (uses Q4_K_M + Q4 KV)
#   serve-qwen36.sh dense 108000       # ~108K (uses Q5_K_S + Q8 KV if available)
set -euo pipefail

ROOT="$HOME/local-llm"
VENV="$ROOT/.venv"
PORT=8020
PID_FILE="/tmp/qwen36-server.pid"
LOG="/tmp/qwen36-server.log"
GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

DENSE_DIR="$HOME/models/qwen3.6-27b-heretic"
MOE_DIR="$HOME/models/qwen3.6-35b-a3b-heretic"
VARIANT="${1:-auto}"
CTX_OVERRIDE="${2:-}"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  Qwen3.6  ${GRY}already running on :$PORT${R}"
  exit 0
fi

# ── Resolve model file ──────────────────────────────────────────────────
GGUF=""
MODEL_LABEL=""

find_gguf() {
  local dir="$1"
  # Prefer Q5_K_S > Q5_K_M > Q4_K_M (best quality that fits)
  for q in Q5_K_S Q5_K_M Q4_K_M Q4_K_S Q3_K_L; do
    local f=$(find "$dir" -name "*${q}*.gguf" -print -quit 2>/dev/null)
    if [[ -n "$f" ]]; then echo "$f"; return; fi
  done
}

case "$VARIANT" in
  dense) GGUF=$(find_gguf "$DENSE_DIR"); MODEL_LABEL="Qwen3.6-27B Dense" ;;
  moe)   GGUF=$(find_gguf "$MOE_DIR");   MODEL_LABEL="Qwen3.6-35B-A3B MoE" ;;
  auto)
    GGUF=$(find_gguf "$DENSE_DIR")
    if [[ -n "$GGUF" ]]; then MODEL_LABEL="Qwen3.6-27B Dense"
    else GGUF=$(find_gguf "$MOE_DIR"); MODEL_LABEL="Qwen3.6-35B-A3B MoE"; fi
    ;;
esac

if [[ -z "$GGUF" ]]; then
  echo -e "${YEL}No model found.${R} Run: just setup-qwen36"
  exit 1
fi

# ── Choose KV cache type and context based on quant ─────────────────────
# Q5_K_S/Q5_K_M (~18-19 GB): use Q8_0 KV for quality, 108K context
# Q4_K_M (~16 GB): use Q4_0 KV, fits full 262K context
QUANT_NAME=$(basename "$GGUF" | grep -oP 'Q[0-9]+_K_[A-Z]+')
case "$QUANT_NAME" in
  Q5_K_S|Q5_K_M)
    TYPE_K=8   # Q8_0
    TYPE_V=8   # Q8_0
    CTX="${CTX_OVERRIDE:-108000}"
    KV_LABEL="Q8_0"
    ;;
  Q4_K_M|Q4_K_S|Q3_K_L|Q3_K_M|*)
    TYPE_K=2   # Q4_0
    TYPE_V=2   # Q4_0
    CTX="${CTX_OVERRIDE:-262144}"
    KV_LABEL="Q4_0"
    ;;
esac

# If user forces 262K with Q5, downgrade KV to Q4_0 to fit
if [[ "$CTX" -gt 200000 ]] && [[ "$TYPE_K" -eq 8 ]]; then
  TYPE_K=2; TYPE_V=2; KV_LABEL="Q4_0 (forced for 262K)"
fi

echo -e "  Starting ${MODEL_LABEL} Uncensored ${GRY}(log: $LOG)${R}"
echo -e "  ${GRY}Model: $(basename "$GGUF") | Ctx: $CTX | KV: $KV_LABEL | Flash Attn${R}"

# Use Python API to set logits_all=False
# (server CLI defaults True — allocates ctx × 248K × 4B = OOM on large context)
nohup "$VENV/bin/python" -c "
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings
import uvicorn

model = ModelSettings(
    model='$GGUF',
    model_alias='qwen3.6-27b-uncensored',
    n_gpu_layers=-1,
    n_ctx=$CTX,
    type_k=$TYPE_K,
    type_v=$TYPE_V,
    flash_attn=True,
    logits_all=False,
    chat_format='chatml',
)

server = ServerSettings(host='0.0.0.0', port=$PORT)
app = create_app(server_settings=server, model_settings=[model])
uvicorn.run(app, host='0.0.0.0', port=$PORT, log_level='info')
" >"$LOG" 2>&1 &
echo $! >"$PID_FILE"

echo -n "  waiting for Qwen3.6"
for _ in $(seq 1 90); do
  if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    exit 0
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
