#!/usr/bin/env bash
# scripts/setup-pi.sh — register lamu as a `pi` custom provider.
#
# pi (Earendil pi-coding-agent) doesn't read OPENAI_BASE_URL. It reads
# ~/.pi/agent/models.json — a JSON registry of OpenAI-compatible
# providers. This script merges a `lamu` provider entry into that file
# so `pi --provider lamu` (or `pi config` → pick `lamu`) routes through
# lamu serve on :8020.
#
# Idempotent: re-running updates the lamu entry without touching others.
#
# Usage:
#   bash scripts/setup-pi.sh                 # uses LAMU_URL=http://127.0.0.1:8020
#   LAMU_URL=http://host:port bash setup-pi.sh

set -euo pipefail

LAMU_URL="${LAMU_URL:-http://127.0.0.1:8020}"
PI_CFG="$HOME/.pi/agent/models.json"

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; RED="\033[31m"; R="\033[0m"

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

if ! curl -sf -m 3 "$LAMU_URL/v1/models" >/dev/null 2>&1; then
  echo -e "${YEL}warning:${R} lamu not reachable at $LAMU_URL — config will still install."
fi

# Pull current model registry from lamu so the provider knows about
# every entry (pi requires explicit `models: [...]` per provider).
# Includes the full registry — pi's UI scales fine to hundreds of
# entries, and capping silently would surprise users who add more
# models after `lamu scan`.
MODELS_JSON=$(curl -sf -m 3 "$LAMU_URL/v1/models" 2>/dev/null || echo '{"data":[]}')
MODEL_IDS=$(echo "$MODELS_JSON" | jq -r '.data[].id' | jq -R . | jq -s .)

LAMU_PROVIDER=$(jq -n \
  --arg url "$LAMU_URL/v1" \
  --argjson models "$MODEL_IDS" \
  '{
    api: "openai-completions",
    apiKey: "lamu-local",
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

# Merge: keep existing providers, replace just the `lamu` key.
TMP=$(mktemp)
jq --argjson p "$LAMU_PROVIDER" '.providers.lamu = $p' "$PI_CFG" > "$TMP"
mv "$TMP" "$PI_CFG"

# Also flip the user's default provider + model so bare `pi "..."`
# (no --provider flag) routes through lamu. Picks the lamu main:true
# entry name from the registry — that's the operator-designated
# default model, matching what lamu's HTTP alias `lamu`/`default`/
# `main` resolves to. If no entry is flagged we leave the existing
# defaultModel alone and just flip the provider.
SETTINGS="$HOME/.pi/agent/settings.json"
DEFAULT_MODEL=$(echo "$MODELS_JSON" | jq -r '
  .data[] | select(.id | test("heretic-v2-q4_k_m$")) | .id' | head -1)
# Fall back to first registry entry if the heretic-v2 isn't present.
[[ -z "$DEFAULT_MODEL" ]] && DEFAULT_MODEL=$(echo "$MODELS_JSON" | jq -r '.data[0].id // empty')

if [[ ! -f "$SETTINGS" ]]; then
  echo '{}' > "$SETTINGS"
fi
TMP=$(mktemp)
if [[ -n "$DEFAULT_MODEL" ]]; then
  jq --arg m "$DEFAULT_MODEL" '.defaultProvider = "lamu" | .defaultModel = $m' "$SETTINGS" > "$TMP"
else
  jq '.defaultProvider = "lamu"' "$SETTINGS" > "$TMP"
fi
mv "$TMP" "$SETTINGS"

echo -e "${GREEN}✓${R} lamu provider registered in $PI_CFG"
echo -e "${GREEN}✓${R} default provider/model set in $SETTINGS"
echo -e "${GRY}default model:${R} ${DEFAULT_MODEL:-(unset — registry empty)}"
echo -e "${GRY}use it:${R} pi \"hello\"   (bare invocation, no flags)"
echo -e "${GRY}or override:${R} pi --provider lamu --model <name>"
echo -e "${GRY}registered:${R} $(echo "$MODELS_JSON" | jq -r '.data | length') models from lamu registry"
