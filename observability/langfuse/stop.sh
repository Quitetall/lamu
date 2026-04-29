#!/usr/bin/env bash
set -euo pipefail

LANGFUSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$LANGFUSE_DIR"
podman-compose down
