#!/usr/bin/env bash
# Launch a configured harness with env wired to talk to lamu.
#
# Usage:
#   open-harness.sh                         # default harness, default model
#   open-harness.sh codex                   # named harness
#   open-harness.sh list                    # show configured harnesses
#   LAMU_MODEL=name open-harness.sh codex   # pick model for the harness
#
# Knobs:
#   LAMU_URL          base URL of lamu serve (default http://localhost:8020)
#   LAMU_MODEL        model id from `lamu list` (default: lamu alias →
#                     registry main: true entry; the alias resolves
#                     server-side, no client-side lookup needed).
#   LAMU_SANDBOX      0 = exec the harness directly (no sandbox)
#                     1 = wrap in `lamu agent --` (cwd-rw, no $HOME,
#                         no network unless LAMU_SANDBOX_NET=1).
#                     default 0. Most harnesses (pi, hermes,
#                     claude-code) need $HOME-resident config files
#                     to start, which the sandbox masks; opt-in only
#                     when running a config-less command or when
#                     `lamu agent` grows config-dir bind-mounts.
#   LAMU_SANDBOX_NET  1 = pass --net to `lamu agent`. Required when
#                     the harness needs to reach lamu :8020 OR any
#                     external HTTP. default 1 (because lamu itself
#                     lives at :8020 which is a separate process,
#                     not in the sandbox).
#   LAMU_GIT_SNAP     1 = if cwd is a git repo, record current HEAD +
#                         dirty state to `.lamu-harness-snap` file so
#                         the user can `git reset --hard` to the
#                         recorded SHA after the session.
#                     0 = skip.
#                     default 1.
#
# Config: $HOME/local-llm/config/harnesses.yaml.

set -euo pipefail

ROOT="$HOME/local-llm"
CFG="$ROOT/config/harnesses.yaml"
LAMU_URL="${LAMU_URL:-http://localhost:8020}"
LAMU_MODEL="${LAMU_MODEL:-}"
LAMU_SANDBOX="${LAMU_SANDBOX:-0}"
LAMU_SANDBOX_NET="${LAMU_SANDBOX_NET:-1}"
LAMU_GIT_SNAP="${LAMU_GIT_SNAP:-1}"

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

# ── Per-harness MODEL injection ─────────────────────────────────────
#
# Each harness has its own way to pin the model. Translate LAMU_MODEL
# (or default "lamu" alias, resolved server-side) into the harness's
# native flag/env. Harnesses without a known model knob get the env
# var anyway in case extensions/wrappers respect it.

MODEL_ARGS=()
SAFE_TOOLS_NOTE=""
if [[ -n "$LAMU_MODEL" ]]; then
  MODEL_NOTE=", model=$LAMU_MODEL"
else
  MODEL_NOTE=""
fi

case "$NAME" in
  pi)
    [[ -n "$LAMU_MODEL" ]] && MODEL_ARGS+=(--provider lamu --model "$LAMU_MODEL")
    # Safe-default tool allowlist for pi. Default is read-only — pi
    # otherwise enables `bash`, `edit`, `write` automatically, which
    # turn a prompt-injection into arbitrary file/cmd execution
    # against the user's working directory. To escalate:
    #     LAMU_PI_TOOLS=read,grep,find,ls,bash,edit,write just open pi
    # To opt out of the safe-default entirely:
    #     LAMU_PI_TOOLS=  just open pi
    if [[ -z "${LAMU_PI_TOOLS+x}" ]]; then
      LAMU_PI_TOOLS="read,grep,find,ls"
    fi
    if [[ -n "$LAMU_PI_TOOLS" ]]; then
      MODEL_ARGS+=(--tools "$LAMU_PI_TOOLS")
      SAFE_TOOLS_NOTE=", tools=$LAMU_PI_TOOLS"
    fi
    ;;
  hermes)
    [[ -n "$LAMU_MODEL" ]] && MODEL_ARGS+=(-m "$LAMU_MODEL")
    ;;
  claude-code)
    [[ -n "$LAMU_MODEL" ]] && export ANTHROPIC_MODEL="$LAMU_MODEL"
    ;;
  crush)
    [[ -n "$LAMU_MODEL" ]] && export ANTHROPIC_MODEL="$LAMU_MODEL"
    ;;
  codex)
    [[ -n "$LAMU_MODEL" ]] && export OPENAI_MODEL="$LAMU_MODEL"
    ;;
  aider)
    [[ -n "$LAMU_MODEL" ]] && MODEL_ARGS+=(--model "openai/$LAMU_MODEL")
    ;;
  *)
    if [[ -n "$LAMU_MODEL" ]]; then
      export LAMU_MODEL OPENAI_MODEL="$LAMU_MODEL" ANTHROPIC_MODEL="$LAMU_MODEL"
    fi
    ;;
esac

# ── Git snapshot (rollback aid) ─────────────────────────────────────
#
# `lamu sessions` only captures chat sessions, not harness launches.
# Write our own breadcrumb so the user can `git reset --hard <sha>`
# after the session if the harness damages tracked files. The dirty
# state isn't preserved — use `git stash` manually for that.

SNAP_NOTE=""
if [[ "$LAMU_GIT_SNAP" == "1" ]] && command -v git >/dev/null 2>&1; then
  if SHA=$(git rev-parse --verify HEAD 2>/dev/null); then
    BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    DIRTY=$(git status --porcelain 2>/dev/null | wc -l)
    # Verify cwd is writable before redirecting — read-only cwd
    # (e.g. /usr/share or a mounted snapshot) would silently swallow
    # the snap with `set -e` not catching redirect failures.
    if touch .lamu-harness-snap 2>/dev/null; then
      {
        echo "harness=$NAME"
        echo "model=${LAMU_MODEL:-(default)}"
        echo "cwd=$(pwd)"
        echo "git_branch=$BR"
        echo "git_head=$SHA"
        echo "dirty_paths=$DIRTY"
        # date -Is is GNU-only; the +"%Y-..." form works on BSD/macOS too.
        echo "launched_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
      } > .lamu-harness-snap
      SNAP_NOTE=" + snap(.lamu-harness-snap)"
    else
      SNAP_NOTE=" + snap(skipped: cwd not writable)"
    fi
  fi
fi

# Build the exec line. Sandbox wrap goes outermost so env from above
# is inherited into the bubblewrap namespace.
shift || true   # drop harness name; remaining argv passes to harness

CMD_LINE=()
SANDBOX_NOTE=""
if [[ "$LAMU_SANDBOX" == "1" ]]; then
  if command -v lamu >/dev/null 2>&1; then
    CMD_LINE=(lamu agent)
    if [[ "$LAMU_SANDBOX_NET" == "1" ]]; then
      CMD_LINE+=(--net)
      SANDBOX_NOTE=" + sandbox+net"
    else
      SANDBOX_NOTE=" + sandbox"
    fi
    CMD_LINE+=(--)
  else
    echo -e "${YEL}warning:${R} LAMU_SANDBOX=1 but \`lamu\` not on PATH — running unsandboxed" >&2
  fi
fi
# `$CMD` from yaml may be multi-token (e.g. `cmd: "open-webui serve"`).
# Split it ONCE into an array so each token becomes a separate argv
# slot — avoids re-splitting any embedded subsequent expansions.
read -ra CMD_PARTS <<< "$CMD"
CMD_LINE+=("${CMD_PARTS[@]}" "${MODEL_ARGS[@]}" "$@")

echo -e "${GREEN}→${R} $NAME ${GRY}($FLAVOR, $EnvNote${MODEL_NOTE}${SAFE_TOOLS_NOTE}${SANDBOX_NOTE}${SNAP_NOTE})${R}"
echo -e "${GRY}\$${R} ${CMD_LINE[*]}"
exec "${CMD_LINE[@]}"
