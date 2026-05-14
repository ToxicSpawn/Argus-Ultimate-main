#!/usr/bin/env python3
"""
Export performance time series from paper_results, trade ledger, and allocator stats.

Usage:
  python scripts/export_performance_series.py [--output data/performance_series.csv] [--format csv|json]
Reads data/paper_results.json, data/unified_trades.db, data/strategy_allocator_stats.json
and writes a time series (date, equity_or_cumulative_pnl, trades, etc.) for analysis/plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_paper_results(repo_root: Path) -> list[dict]:
    """Load paper_results.json and return list of daily/snapshot records if present."""
    path = repo_root / "data" / "paper_results.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    out = []
    # Common shapes: { "equity_curve": [...], "daily_pnl": {...}, "trades": N, "return_pct": X }
    equity_curve = data.get("equity_curve") or data.get("equity") or []
    if isinstance(equity_curve, list):
        for i, v in enumerate(equity_curve):
            if isinstance(v, (int, float)):
                out.append({"ts_idx": i, "equity": float(v)})
            elif isinstance(v, dict):
                out.append({"ts_idx": v.get("i", i), "equity": float(v.get("equity", v.get("value", 0))), **v})
    daily = data.get("daily_pnl") or data.get("daily_pnl_aud") or {}
    for date_str, pnl in daily.items():
        out.append({"date": date_str, "daily_pnl": float(pnl) if isinstance(pnl, (int, float)) else 0})
    if not out and data:
        out.append({
            "return_pct": float(data.get("return_pct", 0) or 0),
            "trades": int(data.get("trades", 0) or 0),
            "max_drawdown_pct": float(data.get("max_drawdown_pct", 0) or 0),
        })
    return out


def load_ledger_series(repo_root: Path) -> list[dict]:
    """Build time series from unified_trades.db (trades table): timestamp, cumulative_pnl, trade_count."""
    db_path = repo_root / "data" / "unified_trades.db"
    if not db_path.exists():
        return []
    out = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp, pnl, value, symbol, side FROM trades WHERE status = 'filled' OR status = 'closed' ORDER BY timestamp"
        )
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return []
    cumulative_pnl = 0.0
    for r in rows:
        ts = float(r["timestamp"])
        pnl = float(r["pnl"] or 0)
        cumulative_pnl += pnl
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        out.append({
            "timestamp": ts,
            "date": dt.strftime("%Y-%m-%d"),
            "datetime_utc": dt.isoformat(),
            "cumulative_pnl": cumulative_pnl,
            "trade_pnl": pnl,
            "symbol": r["symbol"],
            "side": r["side"],
        })
    return out


def load_allocator_series(repo_root: Path) -> list[dict]:
    """Load strategy_allocator_stats.json and return strategy-level stats as a single snapshot (or time series if present)."""
    path = repo_root / "data" / "strategy_allocator_stats.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    out = []
    buckets = data.get("buckets") or data.get("strategies") or {}
    for strategy, v in buckets.items():
        if not isinstance(v, dict):
            continue
        out.append({
            "strategy": strategy,
            "trades": int(v.get("trades", v.get("n_trades", 0)) or 0),
            "pnl_ema": float(v.get("pnl_ema", v.get("ema_pnl", 0)) or 0),
            "wins": int(v.get("wins", 0) or 0),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export performance time series from paper_results, ledger, allocator")
    parser.add_argument("--output", default="data/performance_series.csv", help="Output file path")
    parser.add_argument("--format", choices=("csv", "json"), default="csv", help="Output format")
    parser.add_argument("--source", choices=("ledger", "paper", "allocator", "all"), default="all", help="Data source")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    if args.source in ("paper", "all"):
        paper = load_paper_results(repo_root)
        for r in paper:
            r["_source"] = "paper"
        all_rows.extend(paper)
    if args.source in ("ledger", "all"):
        ledger = load_ledger_series(repo_root)
        for r in ledger:
            r["_source"] = "ledger"
        all_rows.extend(ledger)
    if args.source in ("allocator", "all") and (args.format == "json" or not all_rows):
        alloc = load_allocator_series(repo_root)
        for r in alloc:
            r["_source"] = "allocator"
        all_rows.extend(alloc)

    if not all_rows and args.source == "all":
        # If no ledger/paper series, write a single summary from paper_results if present
        paper = load_paper_results(repo_root)
        if paper:
            all_rows = [{"_source": "paper", **p} for p in paper]

    if args.format == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, default=str)
    else:
        if not all_rows:
            # Empty CSV with headers
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                f.write("date,source,cumulative_pnl,trades\n")
        else:
            key_set: set[str] = set()
            for r in all_rows:
                key_set.update(k for k in r if not k.startswith("_"))
            fieldnames = sorted(key_set - {"_source"}) + ["_source"]
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_rows)

    print("Wrote %s (%s rows) to %s" % (args.format.upper(), len(all_rows), out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
