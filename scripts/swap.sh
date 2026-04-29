#!/usr/bin/env bash
# llm-swap <model> — hot-swap the active inference backend
#   llm-swap qwen    → unload gpt2, load Qwen3.5-27B (DFlash)
#   llm-swap gpt2    → unload Qwen, load gpt2-xl (SGLang GPU)
set -euo pipefail

ROOT="$HOME/local-llm"
BOLD="\033[1m"; GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; R="\033[0m"

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo -e "Usage: llm-swap <qwen|gpt2>"
  echo -e "  ${GRY}qwen${R}  — Qwen3.5-27B via DFlash  (:8000)"
  echo -e "  ${GRY}gpt2${R}  — gpt2-xl via SGLang      (:8001)"
  exit 1
fi

stop_dflash() {
  if [[ -f /tmp/dflash-server.pid ]]; then
    pid=$(cat /tmp/dflash-server.pid)
    kill "$pid" 2>/dev/null && echo -e "  DFlash   ${GRY}unloaded${R}" || true
    rm -f /tmp/dflash-server.pid
  fi
}

stop_sglang() {
  if [[ -f /tmp/sglang-gpt2-xl.pid ]]; then
    pid=$(cat /tmp/sglang-gpt2-xl.pid)
    kill "$pid" 2>/dev/null && echo -e "  SGLang   ${GRY}unloaded${R}" || true
    rm -f /tmp/sglang-gpt2-xl.pid
  fi
  if [[ -f /tmp/gpt2-proxy.pid ]]; then
    pid=$(cat /tmp/gpt2-proxy.pid)
    kill "$pid" 2>/dev/null && echo -e "  GPT-2 proxy ${GRY}stopped${R}" || true
    rm -f /tmp/gpt2-proxy.pid
  fi
}

case "$TARGET" in
  qwen)
    echo -e "\n${BOLD}Swapping to Qwen3.5-27B${R}"
    stop_sglang
    bash "$ROOT/scripts/serve-dflash.sh"
    echo -e "\n${GREEN}Active: dflash/luce-dflash${R}\n"
    ;;
  gpt2)
    echo -e "\n${BOLD}Swapping to gpt2-xl${R}"
    stop_dflash
    bash "$ROOT/scripts/serve-sglang.sh" gpt2-xl
    bash "$ROOT/scripts/serve-sglang-presets.sh"
    echo -e "\n${GREEN}Active: gpt2/shitty-{best2021,inferkit,coherent,terrible,…}${R}\n"
    ;;
  *)
    echo -e "${YEL}Unknown model: $TARGET${R}  (use: qwen | gpt2)"
    exit 1
    ;;
esac
