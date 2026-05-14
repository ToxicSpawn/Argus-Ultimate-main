#!/usr/bin/env python3
"""
Live vs backtest consistency: compare realized slippage/IS to backtest assumptions.

Usage: python scripts/live_vs_backtest_consistency.py [--config unified_config.yaml]
Reads data/unified_trades.db and optional execution IS tracker; compares avg slippage/IS
to backtest.slippage_bps and execution_engine.max_avg_is_bps. Exits 0 if live is within
tolerance of backtest (or no live data), 1 if live is meaningfully worse (warn).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare live realized cost to backtest assumptions")
    parser.add_argument("--config", default="unified_config.yaml", help="Config path")
    parser.add_argument("--tolerance-worse-pct", type=float, default=50.0, help="Allow live slippage up to this % worse than backtest (default 50)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    db_path = repo_root / "data" / "unified_trades.db"
    if not db_path.exists():
        print("No trade ledger found; skip consistency check.")
        return 0

    # Load config backtest assumptions
    backtest_slippage_bps = 8.0
    try:
        import yaml
        cfg = yaml.safe_load((repo_root / args.config).read_text(encoding="utf-8")) or {}
        bt = (cfg.get("backtest") or {})
        backtest_slippage_bps = float(bt.get("slippage_bps", 8.0))
    except Exception:
        pass

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT slippage, price, size, side FROM trades WHERE status = 'filled' AND slippage IS NOT NULL AND slippage != 0 LIMIT 500")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No filled trades with slippage; skip consistency check.")
        return 0

    # Realized slippage bps (slippage/price * 1e4)
    slippage_bps_list = []
    for r in rows:
        price = float(r["price"] or 0)
        if price <= 0:
            continue
        slip = float(r["slippage"] or 0)
        bps = abs(slip / price) * 1e4
        slippage_bps_list.append(bps)
    if not slippage_bps_list:
        print("Could not compute realized slippage bps.")
        return 0
    avg_live_bps = sum(slippage_bps_list) / len(slippage_bps_list)
    threshold = backtest_slippage_bps * (1 + args.tolerance_worse_pct / 100.0)
    if avg_live_bps > threshold:
        print("WARN: Live avg slippage %.1f bps > backtest assumption %.1f bps (tolerance %.0f%% = %.1f bps). Consider raising backtest.slippage_bps or improving execution."
              % (avg_live_bps, backtest_slippage_bps, args.tolerance_worse_pct, threshold))
        return 1
    print("PASS: Live avg slippage %.1f bps within tolerance of backtest %.1f bps (n=%s)." % (avg_live_bps, backtest_slippage_bps, len(slippage_bps_list)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
