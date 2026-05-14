#!/usr/bin/env python3
"""
TCA (transaction cost analysis) summary: avg slippage bps, avg implementation shortfall by strategy/symbol.

Usage: python scripts/tca_summary.py [--output data/tca_summary.json]
Reads data/unified_trades.db and optionally execution IS tracker; writes summary for tuning.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="TCA summary from trade ledger")
    parser.add_argument("--output", default="data/tca_summary.json", help="Output JSON path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    db_path = repo_root / "data" / "unified_trades.db"
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {"avg_slippage_bps": None, "trades_with_slippage": 0, "by_symbol": {}, "by_strategy": {}, "by_venue": {}}

    if not db_path.exists():
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("No ledger; wrote empty TCA summary to", out_path)
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT symbol, side, price, size, slippage, raw_json FROM trades WHERE status = 'filled'")
    rows = cur.fetchall()
    conn.close()

    bps_list = []
    by_sym = {}
    by_strat = {}
    by_venue = {}
    for r in rows:
        price = float(r["price"] or 0)
        if price <= 0:
            continue
        slip = float(r["slippage"] or 0)
        bps = abs(slip / price) * 1e4
        bps_list.append(bps)
        sym = str(r["symbol"] or "?")
        by_sym.setdefault(sym, []).append(bps)
        raw = r["raw_json"]
        strat = "unknown"
        venue = "default"
        if raw:
            try:
                d = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
                strat = str(d.get("source_strategy", d.get("strategy", "unknown")))
                venue = str(d.get("venue", d.get("exchange", "default")))
            except Exception:
                pass
        by_strat.setdefault(strat, []).append(bps)
        by_venue.setdefault(venue, []).append(bps)

    if bps_list:
        summary["avg_slippage_bps"] = round(sum(bps_list) / len(bps_list), 2)
        summary["trades_with_slippage"] = len(bps_list)
        summary["by_symbol"] = {k: round(sum(v) / len(v), 2) for k, v in by_sym.items()}
        summary["by_strategy"] = {k: round(sum(v) / len(v), 2) for k, v in by_strat.items()}
        summary["by_venue"] = {k: round(sum(v) / len(v), 2) for k, v in by_venue.items()}

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote TCA summary to", out_path, "| avg_slippage_bps:", summary.get("avg_slippage_bps"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
