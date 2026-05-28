#!/usr/bin/env bash
# scripts/setup-omp-mimo.sh — point OMP's xiaomi provider at the
# token-plan-sgp host so `tp-` keys (MIMO_API_KEY) work end-to-end.
#
# OMP (oh-my-pi) ships with a bundled xiaomi provider whose baseUrl
# is `https://api.xiaomimimo.com/anthropic` — the canonical
# pay-as-you-go gateway. That endpoint rejects token-plan `tp-`
# keys with 401. This script writes a provider override to
# ~/.omp/agent/models.json that retargets the xiaomi provider to
# `token-plan-sgp.xiaomimimo.com/anthropic`.
#
# Idempotent: rewrites the xiaomi entry, preserves any others.
#
# Requires MIMO_API_KEY in ~/.config/lamu/api-keys.env (or in env).
# OMP reads $XIAOMI_API_KEY at runtime.

set -euo pipefail

OMP_CFG_DIR="${PI_CODING_AGENT_DIR:-$HOME/.omp/agent}"
OMP_MODELS="$OMP_CFG_DIR/models.json"
OMP_CACHE_DB="$OMP_CFG_DIR/models.db"

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; RED="\033[31m"; R="\033[0m"

KEYS_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/lamu/api-keys.env"
if [[ -f "$KEYS_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$KEYS_FILE"
  set +a
fi

if [[ -z "${MIMO_API_KEY:-}" ]]; then
  echo -e "${RED}MIMO_API_KEY not set. Add it to $KEYS_FILE first.${R}"
  exit 1
fi

if ! command -v omp >/dev/null 2>&1; then
  echo -e "${RED}'omp' not found on PATH. Install with: bun install -g @oh-my-pi/pi-coding-agent${R}"
  exit 1
fi

mkdir -p "$OMP_CFG_DIR"

# Region defaults to sgp; override with LAMU_MIMO_REGION=ams or cn.
REGION="${LAMU_MIMO_REGION:-sgp}"
case "$REGION" in
  sgp|ams|cn) ;;
  *) echo -e "${RED}LAMU_MIMO_REGION must be sgp|ams|cn (got '$REGION')${R}"; exit 1 ;;
esac
BASE_URL="https://token-plan-${REGION}.xiaomimimo.com/anthropic"

# Merge xiaomi override into models.json. Preserves other providers
# the user may have configured.
if [[ -f "$OMP_MODELS" ]]; then
  TMP=$(mktemp)
  jq --arg url "$BASE_URL" '
    .providers //= {} |
    .providers.xiaomi = (.providers.xiaomi // {}) |
    .providers.xiaomi.baseUrl = $url
  ' "$OMP_MODELS" > "$TMP"
  mv "$TMP" "$OMP_MODELS"
else
  cat > "$OMP_MODELS" <<EOF
{
  "providers": {
    "xiaomi": {
      "baseUrl": "$BASE_URL"
    }
  }
}
EOF
fi

# Bust the cached provider catalog so the new baseUrl takes effect on
# the next omp invocation. Cache is just sqlite; nuking the xiaomi
# row is enough.
if [[ -f "$OMP_CACHE_DB" ]] && command -v sqlite3 >/dev/null 2>&1; then
  sqlite3 "$OMP_CACHE_DB" "DELETE FROM model_cache WHERE provider_id='xiaomi';" 2>/dev/null || true
fi

echo -e "${GREEN}✓${R} xiaomi.baseUrl set to $BASE_URL in $OMP_MODELS"
echo -e "${GREEN}✓${R} provider cache invalidated"
echo -e "${GRY}use it:${R} XIAOMI_API_KEY=\$MIMO_API_KEY omp --model xiaomi/mimo-v2.5-pro \"hello\""
echo -e "${GRY}or via lamu:${R} just open omp-mimo"
echo
echo -e "${YEL}note:${R} OMP requires bun >= 1.3.14 — check with: bun --version"
