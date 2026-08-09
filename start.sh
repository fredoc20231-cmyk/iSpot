#!/bin/bash
# iSpot server startup script
# Usage: ./start.sh [--host 0.0.0.0] [--port 8100]

set -e

HOST="${ISPOT_HOST:-0.0.0.0}"
PORT="${ISPOT_PORT:-8100}"

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure job storage directory exists
mkdir -p ispot_jobs

# Clear stale bytecode
find ispot -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Starting iSpot server on ${HOST}:${PORT}..."
echo "API docs at: http://${HOST}:${PORT}/api/docs"
echo "Frontend at: http://${HOST}:${PORT}/"

exec python -m uvicorn ispot.api:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info
