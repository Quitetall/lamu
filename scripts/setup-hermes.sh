#!/usr/bin/env bash
# scripts/setup-hermes.sh — point `hermes` at lamu as its inference provider.
#
# hermes (Nous Research agent) reads ~/.hermes/config.yaml. To use a
# local OpenAI-compatible endpoint, set `provider: custom` + `base_url:
# <url>` under the `inference:` (or `llm:`) top-level section. This
# script edits that file in place — backed up to .yaml.bak first.
#
# Idempotent: re-running just overwrites the lamu-related keys.
#
# Usage:
#   bash scripts/setup-hermes.sh             # uses LAMU_URL=http://127.0.0.1:8020
#   LAMU_URL=http://host:port bash setup-hermes.sh

set -euo pipefail

LAMU_URL="${LAMU_URL:-http://127.0.0.1:8020}"
HERMES_CFG="$HOME/.hermes/config.yaml"

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; RED="\033[31m"; R="\033[0m"

if [[ ! -f "$HERMES_CFG" ]]; then
  echo -e "${RED}$HERMES_CFG missing — run 'hermes setup' once to create it.${R}"
  exit 1
fi

for dep in python3 curl; do
  if ! command -v "$dep" >/dev/null 2>&1; then
    echo -e "${RED}'$dep' is required. Install with your distro's package manager.${R}"
    exit 1
  fi
done

if ! curl -sf -m 3 "$LAMU_URL/v1/models" >/dev/null 2>&1; then
  echo -e "${YEL}warning:${R} lamu not reachable at $LAMU_URL — config will still install."
fi

cp -f "$HERMES_CFG" "$HERMES_CFG.bak"

python3 - "$HERMES_CFG" "$LAMU_URL" <<'PYEOF'
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
lamu_url = sys.argv[2]

text = cfg_path.read_text()

# Hermes config is a yaml file with extensive comments — preserve them
# by doing a line-level edit on the two keys we care about. They live
# under the top-level `model:` section (Hermes calls it that even
# though the keys configure the inference provider). Tolerant of
# `model:` / `inference:` / `llm:` section names since wording has
# shifted across hermes versions.
SECTION_KEYS = {"model:", "inference:", "llm:"}

lines = text.splitlines(keepends=True)
out = []
in_section = False
patched_provider = False
patched_base_url = False
patched_default = False

for line in lines:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    # Top-level keys: no leading whitespace, end with `:`. Track
    # whether we're inside the provider-config section.
    if not line.startswith((' ', '\t', '#', '\n')) and line.rstrip().endswith(':'):
        in_section = line.rstrip() in SECTION_KEYS

    # Only patch the FIRST uncommented occurrence of each key.
    if in_section and stripped.startswith('default:') and not patched_default:
        out.append(f'{indent}default: "lamu"  # lamu alias → main: true entry — set by setup-hermes.sh\n')
        patched_default = True
        continue
    if in_section and stripped.startswith('provider:') and not patched_provider:
        out.append(f'{indent}provider: "custom"  # lamu — set by setup-hermes.sh\n')
        patched_provider = True
        continue
    if in_section and stripped.startswith('base_url:') and not patched_base_url:
        out.append(f'{indent}base_url: "{lamu_url}/v1"  # lamu — set by setup-hermes.sh\n')
        patched_base_url = True
        continue
    out.append(line)

# Only append a fresh stanza if NEITHER key existed in any known
# section — avoids the double-config bug where the existing keys
# remained under `model:` while a duplicate `inference:` block was
# appended at EOF.
if not patched_provider and not patched_base_url:
    out.append('\n# Added by setup-hermes.sh — point hermes at lamu.\n')
    out.append('model:\n')
    out.append('  default: "lamu"\n')
    out.append('  provider: "custom"\n')
    out.append(f'  base_url: "{lamu_url}/v1"\n')

cfg_path.write_text(''.join(out))
print(f'  default model patched: {patched_default}')
print(f'  provider patched: {patched_provider}')
print(f'  base_url patched: {patched_base_url}')
PYEOF

echo -e "${GREEN}✓${R} hermes config patched in $HERMES_CFG"
echo -e "${GRY}backup at:${R} $HERMES_CFG.bak"
echo -e "${GRY}use it:${R} hermes chat   (or 'hermes model' to pick a specific lamu model)"
