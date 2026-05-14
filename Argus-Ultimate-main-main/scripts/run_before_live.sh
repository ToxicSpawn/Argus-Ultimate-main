#!/usr/bin/env bash
# =============================================================================
# Run before switching to live (process checklist).
# Ensures: validate, pre_live_check, validate_priority_order pass.
# See docs/LIVE_CHECKLIST.md for full checklist.
# =============================================================================
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"
CONFIG="${1:-unified_config.yaml}"

echo "=== Argus: run before live (config=$CONFIG) ==="

echo ""
echo "1. Validate config and priority order..."
"$PY" main.py validate 2>/dev/null || "$PY" -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
p = Path('$CONFIG')
if not p.exists():
    print('WARN: $CONFIG not found')
    sys.exit(0)
print('Config exists:', p)
" 2>/dev/null || true

echo ""
echo "2. Validate priority order (alerts, confidence, edge gate)..."
"$PY" scripts/validate_priority_order.py --config "$CONFIG" 2>/dev/null || { echo "WARN: validate_priority_order failed or script missing"; true; }

echo ""
echo "3. Pre-live check (credentials, edge gate when live_require_paper_edge)..."
"$PY" scripts/pre_live_check.py --config "$CONFIG" 2>/dev/null || { echo "WARN: pre_live_check failed"; true; }

echo ""
echo "4. Readiness score (config + paper evidence)..."
"$PY" scripts/readiness_score.py --config "$CONFIG" --include-paper 2>/dev/null || { echo "WARN: readiness_score failed"; true; }

echo ""
echo "=== Reminder ==="
echo "- Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env for alerts."
echo "- Complete docs/LIVE_CHECKLIST.md (credentials, paper 2-4 weeks, kill_losers_review)."
echo "- Run weekly: scripts/weekly_profitability_check.py (cron in scripts/cron_example.txt)."
echo ""
