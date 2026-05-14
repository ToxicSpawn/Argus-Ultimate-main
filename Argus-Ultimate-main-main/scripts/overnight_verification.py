#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


def _safe_json(raw: Any) -> Dict[str, Any]:
    try:
        out = json.loads(str(raw or "{}"))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _query_one(db_path: Path, sql: str, params: Tuple[Any, ...] = ()) -> Any:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _table_exists(db_path: Path, table: str) -> bool:
    if not db_path.exists():
        return False
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (str(table),),
        ).fetchone()
        return bool(row)
    finally:
        conn.close()


def build_overnight_verification(
    *,
    trades_db: Path,
    state_db: Path,
    meta_db: Path,
    lookback_hours: float = 12.0,
    min_decisions: int = 1,
    max_recon_required: int = 200,
    max_emergency_stops: int = 5,
    max_meta_total_abs_change: float = 2.0,
    max_liquidity_clamp_ratio: float = 0.95,
) -> Dict[str, Any]:
    now = float(time.time())
    cutoff = now - max(0.0, float(lookback_hours or 0.0)) * 3600.0

    decision_count = int(
        _query_one(
            trades_db,
            "SELECT COUNT(*) FROM decision_snapshots WHERE timestamp >= ?",
            (cutoff,),
        )
        or 0
    )

    liquidity_clamp_count = 0
    microstructure_rows = 0
    execution_plan_rows = 0
    if _table_exists(trades_db, "decision_snapshots"):
        conn = sqlite3.connect(str(trades_db))
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT details_json, execution_plan_json
                FROM decision_snapshots
                WHERE timestamp >= ?
                """,
                (cutoff,),
            )
            for details_raw, plan_raw in cur.fetchall():
                details = _safe_json(details_raw)
                if bool(details.get("liquidity_clamp_flag", False)):
                    liquidity_clamp_count += 1
                if any(k in details for k in ("microprice", "order_book_imbalance", "trade_velocity")):
                    microstructure_rows += 1
                plan = _safe_json(plan_raw)
                if bool(plan):
                    execution_plan_rows += 1
        finally:
            conn.close()

    recon_required_intents = int(
        _query_one(
            state_db,
            "SELECT COUNT(*) FROM order_intents WHERE state='RECON_REQUIRED'",
        )
        or 0
    ) if _table_exists(state_db, "order_intents") else 0

    emergency_stop_count = 0
    if _table_exists(trades_db, "decision_events"):
        emergency_stop_count = int(
            _query_one(
                trades_db,
                """
                SELECT COUNT(*)
                FROM decision_events
                WHERE timestamp >= ?
                  AND (stage LIKE 'emergency%' OR stage IN ('reconciliation_halt_transition', 'safety_halt'))
                """,
                (cutoff,),
            )
            or 0
        )

    meta_total_abs_change = 0.0
    if _table_exists(meta_db, "meta_weight_snapshots"):
        conn = sqlite3.connect(str(meta_db))
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT weights_json
                FROM meta_weight_snapshots
                WHERE ts >= ?
                ORDER BY ts ASC
                """,
                (cutoff,),
            )
            rows = [(_safe_json(r[0])) for r in cur.fetchall()]
            if len(rows) >= 2:
                prev, curr = rows[-2], rows[-1]
                names = sorted(set(prev.keys()) | set(curr.keys()))
                meta_total_abs_change = float(
                    sum(abs(float(curr.get(n, 0.0) or 0.0) - float(prev.get(n, 0.0) or 0.0)) for n in names)
                )
        finally:
            conn.close()

    liquidity_clamp_ratio = float(liquidity_clamp_count / decision_count) if decision_count > 0 else 0.0
    microstructure_presence_ratio = float(microstructure_rows / decision_count) if decision_count > 0 else 0.0
    execution_plan_presence_ratio = float(execution_plan_rows / decision_count) if decision_count > 0 else 0.0

    checks = {
        "decision_snapshot_growth": bool(decision_count >= int(min_decisions)),
        "recon_required_within_limit": bool(recon_required_intents <= int(max_recon_required)),
        "emergency_stop_within_limit": bool(emergency_stop_count <= int(max_emergency_stops)),
        "meta_weight_stability": bool(meta_total_abs_change <= float(max_meta_total_abs_change)),
        "liquidity_clamp_ratio_ok": bool(liquidity_clamp_ratio <= float(max_liquidity_clamp_ratio)),
        "microstructure_fields_present": bool(microstructure_presence_ratio > 0.0),
        "execution_plan_fields_present": bool(execution_plan_presence_ratio > 0.0),
    }
    overall_pass = all(bool(v) for v in checks.values())

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lookback_hours": float(lookback_hours),
        "overall_pass": bool(overall_pass),
        "checks": checks,
        "metrics": {
            "decision_count": int(decision_count),
            "recon_required_intents": int(recon_required_intents),
            "emergency_stop_count": int(emergency_stop_count),
            "meta_total_abs_change": float(meta_total_abs_change),
            "liquidity_clamp_ratio": float(liquidity_clamp_ratio),
            "microstructure_presence_ratio": float(microstructure_presence_ratio),
            "execution_plan_presence_ratio": float(execution_plan_presence_ratio),
            "trades_db_size_mb": float((trades_db.stat().st_size / (1024.0 * 1024.0)) if trades_db.exists() else 0.0),
        },
        "sources": {
            "trades_db": str(trades_db),
            "state_db": str(state_db),
            "meta_db": str(meta_db),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="ARGUS overnight verification harness")
    p.add_argument("--trades-db", default="data/unified_trades.db")
    p.add_argument("--state-db", default="data/unified_state.db")
    p.add_argument("--meta-db", default="data/meta_weights.db")
    p.add_argument("--lookback-hours", type=float, default=12.0)
    p.add_argument("--min-decisions", type=int, default=1)
    p.add_argument("--max-recon-required", type=int, default=200)
    p.add_argument("--max-emergency-stops", type=int, default=5)
    p.add_argument("--max-meta-total-abs-change", type=float, default=2.0)
    p.add_argument("--max-liquidity-clamp-ratio", type=float, default=0.95)
    p.add_argument("--output", default="reports/overnight_verification_summary.json")
    args = p.parse_args()

    report = build_overnight_verification(
        trades_db=Path(args.trades_db),
        state_db=Path(args.state_db),
        meta_db=Path(args.meta_db),
        lookback_hours=float(args.lookback_hours),
        min_decisions=int(args.min_decisions),
        max_recon_required=int(args.max_recon_required),
        max_emergency_stops=int(args.max_emergency_stops),
        max_meta_total_abs_change=float(args.max_meta_total_abs_change),
        max_liquidity_clamp_ratio=float(args.max_liquidity_clamp_ratio),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
