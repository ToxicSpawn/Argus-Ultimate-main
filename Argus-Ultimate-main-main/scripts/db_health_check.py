#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _table_names(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [str(r[0]) for r in cur.fetchall()]


def _table_row_counts(conn: sqlite3.Connection, tables: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    cur = conn.cursor()
    for name in tables:
        if name.startswith("sqlite_"):
            continue
        try:
            cur.execute(f"SELECT COUNT(*) FROM {name}")
            out[name] = int(cur.fetchone()[0] or 0)
        except Exception:
            out[name] = -1
    return out


def inspect_db(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": bool(path.exists()),
        "file_size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "quick_check": "missing",
        "ok": False,
        "table_growth": {},
        "page_count": 0,
        "page_size": 0,
        "estimated_size_bytes": 0,
        "error": "",
    }
    if not path.exists():
        return info

    try:
        conn = sqlite3.connect(str(path))
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA quick_check")
            qc_row = cur.fetchone()
            qc = str(qc_row[0] if qc_row else "")
            info["quick_check"] = qc
            info["ok"] = qc.lower() == "ok"
            cur.execute("PRAGMA page_count")
            page_count = int(cur.fetchone()[0] or 0)
            cur.execute("PRAGMA page_size")
            page_size = int(cur.fetchone()[0] or 0)
            info["page_count"] = page_count
            info["page_size"] = page_size
            info["estimated_size_bytes"] = int(page_count * page_size)
            tables = _table_names(conn)
            info["table_growth"] = _table_row_counts(conn, tables)
        finally:
            conn.close()
    except Exception as e:
        info["ok"] = False
        info["quick_check"] = "error"
        info["error"] = str(e)
    return info


def build_health_report(db_paths: List[Path]) -> Dict[str, Any]:
    db_reports = [inspect_db(p) for p in db_paths]
    overall_ok = all(bool(r.get("ok")) for r in db_reports if bool(r.get("exists")))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_ok": bool(overall_ok),
        "databases": db_reports,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="ARGUS SQLite health check")
    p.add_argument(
        "--db",
        action="append",
        dest="dbs",
        help="Database path (repeatable). Defaults to core ARGUS DBs.",
    )
    p.add_argument("--output", default="", help="Optional JSON output path")
    args = p.parse_args()

    dbs = args.dbs or ["data/unified_trades.db", "data/unified_state.db", "data/meta_weights.db"]
    report = build_health_report([Path(d) for d in dbs])
    payload = json.dumps(report, indent=2, ensure_ascii=True)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(str(out_path))
    else:
        print(payload)
    return 0 if bool(report.get("overall_ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
