"""CLI P&L report generator — Push 54.

Loads a trade CSV and prints a SessionStats summary.

CSV format (header row required)::

    symbol,side,entry_price,exit_price,qty,entry_time,exit_time,fee_bps

Usage::

    python scripts/pnl_report.py trades.csv
    python scripts/pnl_report.py trades.csv --out report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pnl.trade_record import TradeRecord
from core.pnl.session_stats import SessionStats


def load_trades_csv(path: Path) -> list:
    trades = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(TradeRecord(
                symbol=row["symbol"],
                side=row["side"],
                entry_price=float(row["entry_price"]),
                exit_price=float(row["exit_price"]),
                qty=float(row["qty"]),
                entry_time=datetime.fromisoformat(row["entry_time"]).replace(tzinfo=timezone.utc),
                exit_time=datetime.fromisoformat(row["exit_time"]).replace(tzinfo=timezone.utc),
                fee_bps=float(row.get("fee_bps", 2.0)),
            ))
    return trades


def main() -> None:
    p = argparse.ArgumentParser(description="Argus P&L report")
    p.add_argument("csv", type=Path, help="Trade CSV file")
    p.add_argument("--out", type=Path, default=None, help="Write JSON report to file")
    args = p.parse_args()

    trades = load_trades_csv(args.csv)
    stats = SessionStats.from_trades(trades)
    print(stats.pretty_str())

    if args.out:
        args.out.write_text(json.dumps(stats.to_dict(), indent=2))
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
