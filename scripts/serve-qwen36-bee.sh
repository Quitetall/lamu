#!/usr/bin/env bash
# scripts/serve-qwen36-bee.sh — BeeLlama.cpp Qwen3.6-27B + DFlash + turbo3_tcq KV
#
# Validated 2026-05-11: 82 t/s @ 4k ctx (1.84x vanilla 44.6),
# 101.7 t/s @ 262k ctx (lamu target). 4090 / 24GB.
# Bench artifacts: memory/project_beellama_bench.md
#
# Side-by-side with `lamu serve` on :8020 and legacy DFlash on :8000.

set -euo pipefail

PORT="${BEE_PORT:-8021}"
# 131k ctx default (after VRAM clamp from requested 262k on 4090).
CTX="${BEE_CTX:-262144}"
TARGET="${BEE_TARGET:-$HOME/models/qwen3.6-official-gguf/Qwen3.6-27B-Q4_K_M.gguf}"
DRAFT="${BEE_DRAFT:-$HOME/models/qwen3.6-27b-dflash-spiritbuun/dflash-draft-3.6-q4_k_m.gguf}"
KV_K="${BEE_KV_K:-turbo3_tcq}"
KV_V="${BEE_KV_V:-turbo3_tcq}"
BEE_BIN="$HOME/local-llm/beellama.cpp/build/bin/llama-server"
PID_FILE="/tmp/bee-server.pid"
LOG="/tmp/bee-server.log"

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; RED="\033[31m"; R="\033[0m"

if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
  echo -e "  BeeLlama ${GRY}already running on :$PORT${R}"
  exit 0
fi

# Sanity: binary + GGUFs exist before nohup'ing into the void
for f in "$BEE_BIN" "$TARGET" "$DRAFT"; do
  if [[ ! -f "$f" ]]; then
    echo -e "  ${RED}missing:${R} $f"
    exit 1
  fi
done

echo -e "  Starting BeeLlama Qwen3.6-27B + DFlash ${GRY}(:$PORT, $CTX ctx, kv=$KV_K/$KV_V, log: $LOG)${R}"

nohup "$BEE_BIN" \
  -m "$TARGET" \
  -md "$DRAFT" \
  --spec-dflash-default \
  --cache-type-k "$KV_K" \
  --cache-type-v "$KV_V" \
  -ngl 99 -ngld 99 \
  -c "$CTX" \
  --host 0.0.0.0 --port "$PORT" \
  --flash-attn on \
  --parallel 1 \
  --metrics \
  --cache-ram 16384 \
  >"$LOG" 2>&1 &
# Note: requested $CTX may auto-clamp to fit VRAM. Server logs final n_ctx.
# Observed 2026-05-11: BEE_CTX=262144 clamps to 131072 with turbo3_tcq KV on 4090.
echo $! >"$PID_FILE"

echo -n "  waiting for BeeLlama"
for _ in $(seq 1 90); do
  if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
    echo -e " ${GREEN}ready${R}"
    echo -e "  ${GRY}smoke: curl -s localhost:$PORT/v1/chat/completions -d '{\"model\":\"any\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":20}'${R}"
    echo -e "  ${GRY}slots: curl -s localhost:$PORT/slots | jq${R}"
    exit 0
  fi
  echo -n "."; sleep 2
done
echo -e " ${YEL}timeout — check $LOG${R}"
exit 1
