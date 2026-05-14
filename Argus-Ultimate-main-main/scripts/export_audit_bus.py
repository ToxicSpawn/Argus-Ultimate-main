#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (name,)).fetchone()
    return row is not None


def _rows(cur: sqlite3.Cursor, table: str, limit: int) -> List[Dict[str, Any]]:
    if table == "decision_events":
        q = "SELECT timestamp, id, stage, correlation_id, payload_json FROM decision_events ORDER BY timestamp DESC LIMIT ?"
    elif table == "decision_snapshots":
        q = "SELECT timestamp, id, run_id, trace_id, allowed, reason_code, details_json, cost_json, execution_plan_json FROM decision_snapshots ORDER BY timestamp DESC LIMIT ?"
    elif table == "trades":
        q = "SELECT timestamp, id, order_id, symbol, side, exchange, price, size, status, commission, slippage, pnl, value FROM trades ORDER BY timestamp DESC LIMIT ?"
    else:
        return []
    return [dict(r) for r in cur.execute(q, (int(limit),)).fetchall()]


def export(db_path: str, limit: int) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    events: List[Dict[str, Any]] = []
    for table in ("decision_events", "decision_snapshots", "trades"):
        if not _table_exists(cur, table):
            continue
        for row in _rows(cur, table, limit):
            events.append(
                {
                    "timestamp": float(row.get("timestamp") or 0.0),
                    "event_type": table,
                    "payload": row,
                }
            )
    conn.close()
    events.sort(key=lambda x: (float(x.get("timestamp") or 0.0), str(x.get("event_type") or "")))
    return events


def main() -> int:
    p = argparse.ArgumentParser(description="Export replayable audit bus from SQLite tables into JSONL")
    p.add_argument("--db", default="data/unified_trades.db")
    p.add_argument("--output", default="logs/audit_bus_latest.jsonl")
    p.add_argument("--limit", type=int, default=5000)
    args = p.parse_args()

    events = export(args.db, int(args.limit))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=True, default=str) + "\n")
    print(str(out.resolve()))
    print(f"events={len(events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
