#!/usr/bin/env bash
# Launch a configured harness with env wired to talk to lamu.
#
# Usage:
#   open-harness.sh          # launches the default harness
#   open-harness.sh codex    # launches the named entry
#   open-harness.sh list     # show configured harnesses
#
# Config: config/harnesses.yaml.
# Lamu base URL: $LAMU_URL or default http://localhost:8020

set -euo pipefail

ROOT="$HOME/local-llm"
CFG="$ROOT/config/harnesses.yaml"
LAMU_URL="${LAMU_URL:-http://localhost:8020}"

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; RED="\033[31m"; R="\033[0m"

if [[ ! -f "$CFG" ]]; then
  echo -e "${RED}config not found:${R} $CFG"
  exit 1
fi

PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

read_yaml() {
  "$PY" - "$CFG" "$1" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
name = sys.argv[2] if len(sys.argv) > 2 else ""
harnesses = cfg.get("harnesses", {}) or {}
if name == "__list__":
    for k, v in harnesses.items():
        flag = " (default)" if v.get("default") else ""
        print(f"{k}\t{v.get('flavor','?')}\t{v.get('cmd','')}{flag}")
    sys.exit(0)
if name == "__default__":
    for k, v in harnesses.items():
        if v.get("default"):
            print(k)
            sys.exit(0)
    print("", end="")
    sys.exit(0)
entry = harnesses.get(name)
if not entry:
    sys.exit(2)
extra_env = entry.get("extra_env") or {}
print(entry.get("flavor", "openai"))
print(entry.get("cmd", ""))
for k, v in extra_env.items():
    print(f"{k}={v}")
PYEOF
}

if [[ "${1:-}" == "list" ]]; then
  echo -e "${GRY}configured harnesses (config/harnesses.yaml):${R}"
  read_yaml __list__ | column -t -s $'\t'
  exit 0
fi

NAME="${1:-}"
if [[ -z "$NAME" ]]; then
  NAME=$(read_yaml __default__)
  if [[ -z "$NAME" ]]; then
    echo -e "${RED}no default harness set${R} — add 'default: true' to one entry in $CFG"
    exit 1
  fi
fi

INFO=$(read_yaml "$NAME" || true)
if [[ -z "$INFO" ]]; then
  echo -e "${RED}unknown harness:${R} $NAME"
  echo -e "${GRY}run 'just open list' to see configured ones${R}"
  exit 1
fi

FLAVOR=$(echo "$INFO" | sed -n '1p')
CMD=$(echo "$INFO" | sed -n '2p')
EXTRA=$(echo "$INFO" | tail -n +3)

case "$FLAVOR" in
  anthropic)
    export ANTHROPIC_BASE_URL="$LAMU_URL"
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-lamu-local}"
    EnvNote="ANTHROPIC_BASE_URL=$LAMU_URL"
    ;;
  openai)
    export OPENAI_BASE_URL="$LAMU_URL/v1"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-lamu-local}"
    EnvNote="OPENAI_BASE_URL=$LAMU_URL/v1"
    ;;
  ollama)
    export OLLAMA_BASE_URL="$LAMU_URL"
    export OLLAMA_HOST="${LAMU_URL#http://}"
    EnvNote="OLLAMA_BASE_URL=$LAMU_URL"
    ;;
  *)
    echo -e "${RED}unknown flavor '$FLAVOR' for $NAME${R}"
    exit 1
    ;;
esac

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  export "$line"
done <<< "$EXTRA"

# Pre-check lamu is reachable.
if ! curl -sf "$LAMU_URL/v1/models" >/dev/null 2>&1; then
  echo -e "${YEL}warning:${R} lamu not reachable at $LAMU_URL — start it with 'just serve' first"
fi

shift || true  # drop harness name; rest of argv passes to harness
echo -e "${GREEN}→${R} $NAME ${GRY}($FLAVOR, $EnvNote)${R}"
echo -e "${GRY}\$${R} $CMD $*"
exec $CMD "$@"
