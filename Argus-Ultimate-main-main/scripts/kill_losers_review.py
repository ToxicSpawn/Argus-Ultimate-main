#!/usr/bin/env python3
"""
Weekly review: suggest strategies to remove from whitelist based on paper/allocator PnL.
Usage: python scripts/kill_losers_review.py [--config unified_config.yaml] [--min-trades 10]
Reads data/paper_results.json and/or data/strategy_allocator_stats.json; prints strategies
with negative PnL (or negative EMA) and suggests removing from strategy_whitelist.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Suggest strategies to remove from whitelist (kill losers)")
    parser.add_argument("--config", default="unified_config.yaml", help="Config file path")
    parser.add_argument("--min-trades", type=int, default=5, help="Min trades before suggesting removal")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    paper_path = repo_root / "data" / "paper_results.json"
    allocator_path = repo_root / "data" / "strategy_allocator_stats.json"

    suggested_remove = []
    reasons = []

    # Paper results: per-strategy PnL
    if paper_path.exists():
        try:
            with open(paper_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            by_strategy = data.get("by_strategy") or data.get("strategy_pnl") or {}
            for strat, v in by_strategy.items():
                if not isinstance(v, dict):
                    continue
                trades = int(v.get("trades", v.get("count", 0)) or 0)
                pnl = float(v.get("pnl", v.get("total_pnl", 0)) or 0)
                if trades >= args.min_trades and pnl < 0:
                    suggested_remove.append(strat)
                    reasons.append(f"{strat}: trades={trades} pnl={pnl:.2f}")
        except Exception as e:
            print("WARN: Could not read paper_results.json:", e)

    # Allocator stats: strategy performance
    if allocator_path.exists():
        try:
            with open(allocator_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            strategies = data.get("strategies") or data.get("strategy_stats") or {}
            for strat, v in strategies.items():
                if not isinstance(v, dict):
                    continue
                trades = int(v.get("trades", v.get("n_trades", 0)) or 0)
                pnl_ema = v.get("pnl_ema") or v.get("ema_pnl") or v.get("total_pnl")
                if pnl_ema is None:
                    continue
                pnl_ema = float(pnl_ema)
                if trades >= args.min_trades and pnl_ema < 0 and strat not in suggested_remove:
                    suggested_remove.append(strat)
                    reasons.append(f"{strat}: trades={trades} pnl_ema={pnl_ema:.2f}")
        except Exception as e:
            print("WARN: Could not read strategy_allocator_stats.json:", e)

    if not suggested_remove:
        print("PASS: No strategies with negative PnL (min_trades=%s). Whitelist OK." % args.min_trades)
        return 0

    print("Suggested to REMOVE from strategy_whitelist (negative PnL / pnl_ema, min_trades=%s):" % args.min_trades)
    for r in reasons:
        print("  -", r)
    print("\nIn unified_config.yaml under strategies.strategy_whitelist, remove the above names.")
    print("Re-add only after a dedicated backtest or paper run shows positive edge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
