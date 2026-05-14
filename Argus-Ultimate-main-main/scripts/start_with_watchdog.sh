#!/usr/bin/env bash
# ARGUS Ultimate — Start with Self-Healing Watchdog (Linux/macOS)
#
# Usage:
#   ./scripts/start_with_watchdog.sh                     # paper mode (default)
#   ./scripts/start_with_watchdog.sh --mode live         # live mode
#   ./scripts/start_with_watchdog.sh --check-interval 60 # custom check interval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================"
echo "  ARGUS Ultimate — Self-Healing Watchdog"
echo "============================================"
echo ""

# Ensure data directory
mkdir -p data

# Detect Python command
if command -v py &>/dev/null; then
    PY=py
elif command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: Python not found. Install Python 3.11+."
    exit 1
fi

echo "  Python: $($PY --version 2>&1)"
echo "  Project: $PROJECT_ROOT"
echo ""
echo "Starting watchdog..."
echo "Press Ctrl+C to stop."
echo ""

# Pass all args through
exec "$PY" -B -m ops.watchdog --project-root "$PROJECT_ROOT" "$@"
