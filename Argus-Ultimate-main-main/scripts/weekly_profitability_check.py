#!/usr/bin/env python3
"""
Weekly profitability discipline (best way #12).

Runs:
  1. kill_losers_review – suggest strategies to remove from whitelist
  2. readiness_score --include-paper – config + paper checklist
  3. Optional: tca_summary if ledger exists

Usage: python scripts/weekly_profitability_check.py [--no-tca]
Cron: 0 4 * * 0 cd /opt/argus/repo && python scripts/weekly_profitability_check.py >> logs/weekly_profitability.log 2>&1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly profitability check: kill_losers + readiness + optional TCA")
    parser.add_argument("--no-tca", action="store_true", help="Skip TCA summary (no ledger)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os_env = None  # use current env

    print("=== Weekly profitability check ===\n")

    # 1. Kill losers
    print("1. Kill losers review (trim whitelist to positive PnL strategies)")
    try:
        r = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "kill_losers_review.py"), "--config", "unified_config.yaml"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            env=os_env,
        )
        print(r.stdout or "")
        if r.stderr:
            print(r.stderr, file=sys.stderr)
    except Exception as e:
        print("WARN: kill_losers_review failed:", e)

    # 2. Readiness score
    print("\n2. Readiness score (config + paper)")
    try:
        r = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "readiness_score.py"), "--include-paper"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            env=os_env,
        )
        print(r.stdout or "")
        if r.stderr:
            print(r.stderr, file=sys.stderr)
    except Exception as e:
        print("WARN: readiness_score failed:", e)

    # 3. TCA summary (if ledger exists)
    if not args.no_tca and (repo_root / "data" / "unified_trades.db").exists():
        print("\n3. TCA summary (avg slippage by strategy/symbol)")
        try:
            r = subprocess.run(
                [sys.executable, str(repo_root / "scripts" / "tca_summary.py")],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=30,
                env=os_env,
            )
            print(r.stdout or "")
            if r.stderr:
                print(r.stderr, file=sys.stderr)
        except Exception as e:
            print("WARN: tca_summary failed:", e)
    else:
        print("\n3. TCA summary skipped (no ledger or --no-tca)")

    print("\n=== Reminder: update strategies.strategy_whitelist from kill_losers output; re-run paper 2–4 weeks before live. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
