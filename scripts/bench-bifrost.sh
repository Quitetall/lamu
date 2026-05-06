#!/usr/bin/env bash
# scripts/bench-bifrost.sh — measure Bifrost gateway overhead
#
# Compares same prompt sent direct to the backend at :8020 vs proxied
# through Bifrost at :8080. Reports TTFT, total wall-clock, and
# tokens/s for each path.
#
# Decision rule: if Bifrost adds >3% to total latency, strip it from
# the v3 runtime path (Phase 1b in the v3 path-consolidation plan).
# Otherwise keep it as the v3 daemon's optional gateway.
#
# Pre-reqs:
#   1. Heretic (or any chat model) running on :8020.
#   2. Bifrost running on :8080 with the backend mapped to that model.
#   3. `jq` installed for JSON parsing.
#   4. `hyperfine` installed for wall-clock measurement.
#
# Output: appends a markdown table to wiki/pages/bifrost-bench.md.

set -euo pipefail

ROOT="${LAMU_ROOT:-$HOME/local-llm}"
N="${BIFROST_BENCH_N:-20}"
PROMPT="${BIFROST_BENCH_PROMPT:-Write a single sentence about quicksort.}"
MAX_TOKENS="${BIFROST_BENCH_MAX_TOKENS:-128}"
DIRECT_URL="http://localhost:8020/v1/chat/completions"
GATEWAY_URL="http://localhost:8080/v1/chat/completions"
OUT="$ROOT/wiki/pages/bifrost-bench.md"

GREEN="\033[32m"; YEL="\033[33m"; RED="\033[31m"; GRY="\033[90m"; R="\033[0m"

# ── Pre-flight ──────────────────────────────────────────────────────────────

require() {
  command -v "$1" >/dev/null 2>&1 || { echo -e "${RED}missing dep: $1${R}"; exit 1; }
}
require curl
require jq
require hyperfine

probe() {
  curl -sf "$1/v1/models" >/dev/null 2>&1
}

if ! probe "http://localhost:8020"; then
  echo -e "${RED}backend not on :8020. Start a model first (e.g. just serve-qwen36).${R}"
  exit 1
fi
if ! probe "http://localhost:8080"; then
  echo -e "${RED}Bifrost not on :8080. Run \`just serve-bifrost\` first.${R}"
  exit 1
fi

MODEL_ID="$(curl -sf http://localhost:8020/v1/models | jq -r '.data[0].id')"
echo -e "${GRY}Backend model: $MODEL_ID${R}"
echo -e "${GRY}Bench: N=$N runs, max_tokens=$MAX_TOKENS${R}\n"

# ── Build payloads ──────────────────────────────────────────────────────────

DIRECT_PAYLOAD=$(jq -nc \
  --arg model "$MODEL_ID" \
  --arg prompt "$PROMPT" \
  --argjson max "$MAX_TOKENS" \
  '{model:$model, messages:[{role:"user",content:$prompt}], max_tokens:$max, temperature:0.0}')

GATEWAY_PAYLOAD="$DIRECT_PAYLOAD"

# ── Single-shot warm-up + token accounting ──────────────────────────────────

warmup_and_count() {
  local url="$1" payload="$2"
  curl -sf "$url" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-local" \
    -d "$payload" \
    | jq -r '.usage.completion_tokens // 0'
}

echo -e "Warming up direct path..."
DIRECT_TOKS=$(warmup_and_count "$DIRECT_URL" "$DIRECT_PAYLOAD")
echo -e "  generated $DIRECT_TOKS tokens"

echo -e "Warming up gateway path..."
GATEWAY_TOKS=$(warmup_and_count "$GATEWAY_URL" "$GATEWAY_PAYLOAD")
echo -e "  generated $GATEWAY_TOKS tokens\n"

# ── Wall-clock benchmark ────────────────────────────────────────────────────

run_one() {
  local url="$1" payload="$2"
  curl -sf "$url" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-local" \
    -d "$payload" >/dev/null
}
export -f run_one

echo -e "${GREEN}Direct (:8020)${R}"
DIRECT_RESULT=$(hyperfine --warmup 1 --runs "$N" --export-json /tmp/bifrost-direct.json \
  "bash -c 'run_one $DIRECT_URL '\''$DIRECT_PAYLOAD'\'''" 2>&1 | tail -3)
echo "$DIRECT_RESULT"

echo -e "\n${GREEN}Through Bifrost (:8080)${R}"
GATEWAY_RESULT=$(hyperfine --warmup 1 --runs "$N" --export-json /tmp/bifrost-gateway.json \
  "bash -c 'run_one $GATEWAY_URL '\''$GATEWAY_PAYLOAD'\'''" 2>&1 | tail -3)
echo "$GATEWAY_RESULT"

# ── Summarise ───────────────────────────────────────────────────────────────

DIRECT_MEAN=$(jq -r '.results[0].mean'   /tmp/bifrost-direct.json)
DIRECT_STDDEV=$(jq -r '.results[0].stddev' /tmp/bifrost-direct.json)
GATEWAY_MEAN=$(jq -r '.results[0].mean'   /tmp/bifrost-gateway.json)
GATEWAY_STDDEV=$(jq -r '.results[0].stddev' /tmp/bifrost-gateway.json)

OVERHEAD_PCT=$(awk -v d="$DIRECT_MEAN" -v g="$GATEWAY_MEAN" 'BEGIN { printf "%.2f", (g-d)/d*100 }')
DIRECT_TPS=$(awk -v t="$DIRECT_TOKS"   -v s="$DIRECT_MEAN"  'BEGIN { printf "%.1f", t/s }')
GATEWAY_TPS=$(awk -v t="$GATEWAY_TOKS" -v s="$GATEWAY_MEAN" 'BEGIN { printf "%.1f", t/s }')

VERDICT="keep"
if awk "BEGIN { exit !($OVERHEAD_PCT > 3) }"; then
  VERDICT="strip"
fi

# ── Append to wiki ──────────────────────────────────────────────────────────

mkdir -p "$(dirname "$OUT")"
{
  echo
  echo "## Run on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "| Path | Mean (s) | Stddev | Tokens | Tokens/s |"
  echo "|------|---------:|-------:|-------:|---------:|"
  echo "| Direct (:8020)        | $DIRECT_MEAN  | $DIRECT_STDDEV  | $DIRECT_TOKS  | $DIRECT_TPS |"
  echo "| Through Bifrost (:8080) | $GATEWAY_MEAN | $GATEWAY_STDDEV | $GATEWAY_TOKS | $GATEWAY_TPS |"
  echo
  echo "Bifrost overhead: **${OVERHEAD_PCT}%**. Verdict: **${VERDICT}**."
  echo
  echo "- Model: \`$MODEL_ID\`"
  echo "- N=$N runs · max_tokens=$MAX_TOKENS · temperature=0.0"
  echo "- Prompt: \`$PROMPT\`"
} >> "$OUT"

echo -e "\n${GREEN}Wrote summary to $OUT${R}"
echo -e "Bifrost overhead: ${OVERHEAD_PCT}% — verdict: ${VERDICT}"
case "$VERDICT" in
  keep)  echo -e "${GREEN}Bifrost stays. Wire LAMU_GATEWAY_URL=$GATEWAY_URL to route through it.${R}" ;;
  strip) echo -e "${YEL}Bifrost loses. Run Phase 1b: move serve-bifrost.sh + drop fallbacks.${R}" ;;
esac
