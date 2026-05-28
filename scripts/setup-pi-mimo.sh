#!/usr/bin/env bash
# scripts/setup-pi-mimo.sh — register Xiaomi MiMo as a `pi` custom provider.
#
# pi (Earendil pi-coding-agent) doesn't read OPENAI_BASE_URL. It reads
# ~/.pi/agent/models.json — a JSON registry of OpenAI-compatible
# providers. This script merges a `mimo` provider entry into that file
# so `pi --provider mimo --model mimo-v2.5-pro` routes through
# Xiaomi's OpenAI-compat endpoint at token-plan-sgp.xiaomimimo.com/v1.
#
# Idempotent: re-running updates the mimo entry without touching others.
#
# Usage:
#   bash scripts/setup-pi-mimo.sh
#
# Requires MIMO_API_KEY in ~/.config/lamu/api-keys.env (or in env).

set -euo pipefail

MIMO_BASE="https://token-plan-sgp.xiaomimimo.com/v1"
PI_CFG="$HOME/.pi/agent/models.json"

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; RED="\033[31m"; R="\033[0m"

# Source api-keys.env for MIMO_API_KEY availability check.
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

if [[ ! -d "$HOME/.pi/agent" ]]; then
  echo -e "${RED}~/.pi/agent does not exist — run 'pi' once to initialize it, then re-run.${R}"
  exit 1
fi

for dep in jq curl; do
  if ! command -v "$dep" >/dev/null 2>&1; then
    echo -e "${RED}'$dep' is required. Install with your distro's package manager.${R}"
    exit 1
  fi
done

# Discover available models from MiMo's /v1/models endpoint. Falls back
# to the known V2.5/V2 list if reachability fails so the setup still
# completes offline-ish.
MODELS_JSON=$(curl -sf -m 5 "$MIMO_BASE/models" \
  -H "Authorization: Bearer $MIMO_API_KEY" 2>/dev/null \
  || echo '{"data":[{"id":"mimo-v2.5-pro"},{"id":"mimo-v2.5"},{"id":"mimo-v2-pro"},{"id":"mimo-v2-omni"}]}')
MODEL_IDS=$(echo "$MODELS_JSON" | jq -r '.data[].id' | jq -R . | jq -s .)

MIMO_PROVIDER=$(jq -n \
  --arg url "$MIMO_BASE" \
  --arg key "$MIMO_API_KEY" \
  --argjson models "$MODEL_IDS" \
  '{
    api: "openai-completions",
    apiKey: $key,
    baseUrl: $url,
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false
    },
    models: ($models | map({id: .}))
  }')

if [[ ! -f "$PI_CFG" ]]; then
  echo '{"providers":{}}' > "$PI_CFG"
fi

TMP=$(mktemp)
jq --argjson p "$MIMO_PROVIDER" '.providers.mimo = $p' "$PI_CFG" > "$TMP"
mv "$TMP" "$PI_CFG"

echo -e "${GREEN}✓${R} mimo provider registered in $PI_CFG"
echo -e "${GRY}registered:${R} $(echo "$MODELS_JSON" | jq -r '.data | length') models from MiMo registry"
echo -e "${GRY}use it:${R} pi --provider mimo --model mimo-v2.5-pro \"hello\""
echo -e "${GRY}or via lamu:${R} just open pi-mimo"
echo
echo -e "${YEL}note:${R} this does NOT flip pi's default provider away from lamu."
echo "      To make MiMo the bare-invocation default, edit $HOME/.pi/agent/settings.json:"
echo '        "defaultProvider": "mimo"'
echo '        "defaultModel": "mimo-v2.5-pro"'
