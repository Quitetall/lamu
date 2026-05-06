#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/local-llm/deps/langfuse"
podman-compose down
