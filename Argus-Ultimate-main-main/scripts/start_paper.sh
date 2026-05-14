#!/usr/bin/env bash
# ARGUS Ultimate — One-command paper trading start (Linux / R740)
#
# Usage:
#   bash scripts/start_paper.sh
#   bash scripts/start_paper.sh --capital 500
#   bash scripts/start_paper.sh --skip-clean
#   bash scripts/start_paper.sh --skip-validation
#
# Prerequisites:
#   - Python 3.10+ accessible as `python3` or `py`
#   - unified_config.yaml present at repo root
#   - .env file with API keys

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Detect python command
if command -v py &>/dev/null; then
    PY=py
elif command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "FATAL: No python interpreter found (tried py, python3, python)"
    exit 1
fi

# Parse arguments
CAPITAL=1000
SKIP_CLEAN=false
SKIP_VALIDATION=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --capital)
            CAPITAL="$2"
            shift 2
            ;;
        --skip-clean)
            SKIP_CLEAN=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  ARGUS ULTIMATE -- Paper Trading Startup"
echo "============================================"
echo ""

# Step 1: Clean paper state
if [ "$SKIP_CLEAN" = false ]; then
    echo "[1/3] Cleaning paper state..."
    $PY -B scripts/clean_paper_state.py || echo "WARNING: Clean step had issues (non-fatal)"
    echo ""
else
    echo "[1/3] Skipping clean (--skip-clean)"
fi

# Step 2: Validate config
if [ "$SKIP_VALIDATION" = false ]; then
    echo "[2/3] Validating config..."
    $PY -B scripts/validate_config.py
    if [ $? -ne 0 ]; then
        echo "FATAL: Config validation failed. Fix errors above."
        exit 1
    fi
    echo ""
else
    echo "[2/3] Skipping validation (--skip-validation)"
fi

# Step 3: Start paper trading
echo "[3/3] Starting paper trading (capital: $CAPITAL AUD)..."
echo ""
exec $PY -B main.py paper --capital "$CAPITAL"
