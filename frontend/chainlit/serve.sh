#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=7860

if lsof -i :"$PORT" -sTCP:LISTEN -t &>/dev/null; then
    echo "Port $PORT is already in use. Chainlit may already be running."
    echo "  Kill existing process: kill \$(lsof -ti :$PORT)"
    exit 1
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo ".venv not found. Run ./setup.sh first."
    exit 1
fi

source .venv/bin/activate

echo "Starting Chainlit on http://0.0.0.0:$PORT ..."
echo "Local URL: http://localhost:$PORT"
echo ""

chainlit run app.py --port "$PORT" --host 0.0.0.0
