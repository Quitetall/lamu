#!/usr/bin/env bash
# lamu-rs/loadtest/oha.sh — live HTTP load baseline for `lamu serve` (ADR 0020, tier 4).
#
# Thin wrapper around `oha` (https://github.com/hatoo/oha). Drives a *running*
# `lamu serve` with per-surface profiles at ramped concurrency 1/4/16/64 and
# reports p50/p95/p99 latency + tokens/s.
#
# This is NOT a `cargo test` — it needs a live GPU, a warmed model, and an
# external load generator (see the ADR for why). Run via `just bench-http`.
#
#   Install oha:  cargo install oha   (or: pacman -S oha / brew install oha)
#
# Usage:
#   loadtest/oha.sh [BASE_URL] [SURFACE] [MODEL]
#     BASE_URL  default http://127.0.0.1:8020   (lamu serve default port)
#     SURFACE   one of: openai | anthropic | ollama | read | all   (default all)
#     MODEL     model name sent in the body      (default: empty → server picks)
#
# Env knobs:
#   LAMU_LOAD_CONCURRENCY="1 4 16 64"   ramp steps (override to taste)
#   LAMU_LOAD_REQUESTS=64               total requests per (surface, concurrency)
#   LAMU_LOAD_MAXTOK=32                 max_tokens per request (small → fast)
#   LAMU_LOAD_TIMEOUT=120               per-request timeout (s)
#   LAMU_API_TOKEN=...                  bearer, if `lamu serve` has auth on
#   LAMU_LOAD_OUT=                      if set, write per-run oha JSON here (dir)
#
# IMPORTANT: committing a baseline.json requires a live 4090 run. Do NOT
# fabricate numbers — this script only *produces* them; the operator commits the
# real output (ADR 0020 defers the baseline until a real run exists).

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8020}"
SURFACE="${2:-all}"
MODEL="${3:-}"

CONCURRENCY="${LAMU_LOAD_CONCURRENCY:-1 4 16 64}"
REQUESTS="${LAMU_LOAD_REQUESTS:-64}"
MAXTOK="${LAMU_LOAD_MAXTOK:-32}"
TIMEOUT="${LAMU_LOAD_TIMEOUT:-120}"
OUTDIR="${LAMU_LOAD_OUT:-}"

# ── preflight ───────────────────────────────────────────────────────────────
if ! command -v oha >/dev/null 2>&1; then
  echo "error: oha not found. Install: cargo install oha" >&2
  exit 127
fi
if ! curl -sf "${BASE_URL}/v1/models" >/dev/null 2>&1; then
  echo "error: no lamu serve answering at ${BASE_URL} (try: just serve)" >&2
  exit 1
fi

AUTH_ARGS=()
if [[ -n "${LAMU_API_TOKEN:-}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${LAMU_API_TOKEN}")
fi

mkdir -p "${OUTDIR:-/dev/null}" 2>/dev/null || true

# Per-surface request body. temp 0 + tiny max_tokens for repeatable, fast load.
# All bodies use a fixed short prompt so prefill is constant across runs.
body_for() {
  local surface="$1"
  case "$surface" in
    openai)
      cat <<JSON
{"model":"${MODEL}","messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":${MAXTOK},"temperature":0,"stream":false}
JSON
      ;;
    anthropic)
      cat <<JSON
{"model":"${MODEL}","max_tokens":${MAXTOK},"temperature":0,"messages":[{"role":"user","content":"Say hi in one short sentence."}]}
JSON
      ;;
    ollama)
      cat <<JSON
{"model":"${MODEL}","messages":[{"role":"user","content":"Say hi in one short sentence."}],"stream":false,"options":{"temperature":0,"num_predict":${MAXTOK}}}
JSON
      ;;
  esac
}

path_for() {
  case "$1" in
    openai)    echo "/v1/chat/completions" ;;
    anthropic) echo "/v1/messages" ;;
    ollama)    echo "/api/chat" ;;
  esac
}

# ── one (surface, concurrency) run ──────────────────────────────────────────
run_one() {
  local surface="$1" conc="$2"
  local path body json_arg=()
  path="$(path_for "$surface")"
  body="$(body_for "$surface")"

  if [[ -n "$OUTDIR" ]]; then
    json_arg=(-j)  # oha emits a JSON summary to stdout when -j is set
  fi

  echo "── ${surface}  POST ${path}  conc=${conc} n=${REQUESTS} max_tokens=${MAXTOK} ──"
  # oha flags:
  #   -n total requests, -c concurrency, -m POST, -d body, -t per-req timeout,
  #   -p p50/p90/p95/p99 are in the default latency-distribution table.
  local out
  out="$(oha \
    -n "${REQUESTS}" \
    -c "${conc}" \
    -m POST \
    -H "Content-Type: application/json" \
    "${AUTH_ARGS[@]}" \
    -d "${body}" \
    -t "${TIMEOUT}s" \
    --no-tui \
    "${json_arg[@]}" \
    "${BASE_URL}${path}")"

  if [[ -n "$OUTDIR" ]]; then
    local f="${OUTDIR}/${surface}-c${conc}.json"
    printf '%s\n' "$out" > "$f"
    # tokens/s: derive from completion_tokens across n requests / wall time when
    # the surface returns usage. oha measures HTTP latency, not tokens; the
    # tokens/s line below is computed from a single probe request's usage block.
    echo "  → $f"
  else
    printf '%s\n' "$out"
  fi
}

# tokens/s probe: one request, read usage.completion_tokens / elapsed. This is a
# rough single-shot decode-rate number to annotate the latency run; the
# authoritative t/s is in the served model's own /metrics
# (lamu_tokens_generated_total). Never fabricated — measured live here.
tokens_per_sec_probe() {
  local surface="$1" path body t0 t1 resp ctok dt
  path="$(path_for "$surface")"
  body="$(body_for "$surface")"
  t0="$(date +%s.%N)"
  resp="$(curl -sf "${AUTH_ARGS[@]}" -H "Content-Type: application/json" \
            -d "${body}" "${BASE_URL}${path}" 2>/dev/null || true)"
  t1="$(date +%s.%N)"
  dt="$(awk "BEGIN{print ${t1}-${t0}}")"
  # completion_tokens lives at different paths per surface; grep the common ones.
  ctok="$(printf '%s' "$resp" | grep -oE '"(completion_tokens|output_tokens|eval_count)":[0-9]+' | head -1 | grep -oE '[0-9]+$' || true)"
  if [[ -n "$ctok" && -n "$dt" ]]; then
    awk "BEGIN{printf \"  tokens/s (single-shot probe): %.1f  (%s tok / %.3fs)\n\", ${ctok}/${dt}, ${ctok}, ${dt}}"
  else
    echo "  tokens/s probe: usage not present in response (surface may not report it)"
  fi
}

run_surface() {
  local surface="$1"
  tokens_per_sec_probe "$surface"
  for c in $CONCURRENCY; do
    run_one "$surface" "$c"
  done
}

# read path needs no body / model — just a GET ramp to exercise the shared lock.
run_read_path() {
  for ep in /health /v1/models /metrics; do
    for c in $CONCURRENCY; do
      echo "── read GET ${ep}  conc=${c} n=${REQUESTS} ──"
      oha -n "${REQUESTS}" -c "${c}" -m GET "${AUTH_ARGS[@]}" \
        -t "${TIMEOUT}s" --no-tui "${BASE_URL}${ep}"
    done
  done
}

echo "lamu HTTP load baseline → ${BASE_URL}  surface=${SURFACE}  model='${MODEL:-<server-default>}'"
echo "concurrency ramp: ${CONCURRENCY}   (ADR 0020: HTTP has no request queue — single-flight is load-only)"
echo

case "$SURFACE" in
  openai|anthropic|ollama) run_surface "$SURFACE" ;;
  read)                    run_read_path ;;
  all)
    run_read_path
    for s in openai anthropic ollama; do run_surface "$s"; done
    ;;
  *)
    echo "error: unknown surface '${SURFACE}' (openai|anthropic|ollama|read|all)" >&2
    exit 2
    ;;
esac

echo
echo "done. Authoritative tokens/s: served model /metrics (lamu_tokens_generated_total)."
echo "To commit a baseline, run on the live 4090 with LAMU_LOAD_OUT set, then"
echo "review the per-run JSON — do NOT commit fabricated numbers (ADR 0020)."
