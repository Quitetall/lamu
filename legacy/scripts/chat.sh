#!/usr/bin/env bash
# llm — local LLM chat. Works as REPL or one-shot.
#   llm                          interactive REPL
#   llm "what is quicksort"      one-shot answer
#   llm -m dflash/luce-dflash    pick model
#   llm --direct 8020            bypass Bifrost
exec python3 "$HOME/local-llm/cli/chat_repl.py" "$@"
