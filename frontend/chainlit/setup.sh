#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtualenv with python3.12..."
    python3.12 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install -r requirements.txt -q

echo ""
echo "Setup complete."
echo "  1. Copy .env.example to .env and fill in your Langfuse keys:"
echo "       cp .env.example .env"
echo "  2. Start the server:"
echo "       ./serve.sh"
echo "  3. Open http://localhost:7860"
