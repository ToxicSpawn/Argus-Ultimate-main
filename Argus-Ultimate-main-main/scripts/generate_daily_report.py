#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_json(raw: Any) -> Dict[str, Any]:
    try:
        out = json.loads(str(raw or "{}"))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _query_rows(db_path: Path, sql: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        conn.close()


def _table_exists(db_path: Path, table_name: str) -> bool:
    if not db_path.exists():
        return False
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
            (str(table_name),),
        ).fetchone()
        return bool(row)
    finally:
        conn.close()


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(float(v) for v in values)
    idx = int(round(max(0.0, min(1.0, q)) * (len(vs) - 1)))
    return float(vs[idx])


def build_daily_runtime_summary(
    *,
    trades_db: Path,
    state_db: Path,
    meta_db: Path,
    lookback_hours: float = 24.0,
) -> Dict[str, Any]:
    now_ts = float(time.time())
    cutoff = now_ts - max(0.0, float(lookback_hours or 0.0)) * 3600.0

    snapshot_rows = _query_rows(
        trades_db,
        """
        SELECT timestamp, cycle_id, strategy, reason_code, details_json, execution_plan_json
        FROM decision_snapshots
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (cutoff,),
    )
    trade_rows = _query_rows(
        trades_db,
        """
        SELECT timestamp, slippage
        FROM trades
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (cutoff,),
    )
    cycle_latency_rows: List[sqlite3.Row] = []
    if _table_exists(trades_db, "decision_events"):
        cycle_latency_rows = _query_rows(
            trades_db,
            """
            SELECT payload_json
            FROM decision_events
            WHERE timestamp >= ?
              AND stage IN ('cycle_complete', 'cycle_metrics', 'cycle_timing')
            ORDER BY timestamp ASC
            """,
            (cutoff,),
        )

    emergency_rows: List[sqlite3.Row] = []
    if _table_exists(trades_db, "decision_events"):
        emergency_rows = _query_rows(
            trades_db,
            """
            SELECT stage, payload_json
            FROM decision_events
            WHERE timestamp >= ?
              AND (
                   stage LIKE 'emergency%'
                   OR stage IN ('reconciliation_halt_transition', 'safety_halt')
              )
            ORDER BY timestamp ASC
            """,
            (cutoff,),
        )
    meta_rows = _query_rows(
        meta_db,
        """
        SELECT ts, weights_json
        FROM meta_weight_snapshots
        WHERE ts >= ?
        ORDER BY ts ASC
        """,
        (cutoff,),
    )

    cycles = {int(r["cycle_id"]) for r in snapshot_rows if r["cycle_id"] is not None}
    decisions_generated = len(snapshot_rows)
    liquidity_clamp_count = 0
    strategy_counter: Counter[str] = Counter()
    recon_reason_count = 0

    for row in snapshot_rows:
        strategy = str(row["strategy"] or "").strip()
        if strategy:
            strategy_counter[strategy] += 1
        if str(row["reason_code"] or "") == "RECON_REQUIRED_LOCK":
            recon_reason_count += 1
        details = _safe_json(row["details_json"])
        if bool(details.get("liquidity_clamp_flag", False)):
            liquidity_clamp_count += 1

    slippage_values: List[float] = []
    for row in trade_rows:
        try:
            slippage_values.append(float(row["slippage"] or 0.0) * 10000.0)
        except Exception:
            continue
    avg_slippage_bps = float(sum(slippage_values) / len(slippage_values)) if slippage_values else 0.0
    p90_slippage_bps = _pct(slippage_values, 0.90)

    cycle_latency_values: List[float] = []
    for row in cycle_latency_rows:
        payload = _safe_json(row["payload_json"])
        ms = payload.get("cycle_latency_ms")
        if ms is None:
            ms = payload.get("latency_ms")
        if ms is None:
            dur = payload.get("cycle_duration_seconds")
            if isinstance(dur, (int, float)):
                ms = float(dur) * 1000.0
        if isinstance(ms, (int, float)):
            cycle_latency_values.append(float(ms))
    avg_cycle_latency_ms = float(sum(cycle_latency_values) / len(cycle_latency_values)) if cycle_latency_values else 0.0
    p90_cycle_latency_ms = _pct(cycle_latency_values, 0.90)

    state_recon_required = 0
    state_recovery_halted = 0
    intent_count_by_state: Dict[str, int] = {}
    if state_db.exists():
        conn = sqlite3.connect(str(state_db))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='order_intents'"
            )
            if int(cur.fetchone()[0]) > 0:
                cur.execute("SELECT COUNT(*) FROM order_intents WHERE state='RECON_REQUIRED'")
                state_recon_required = int(cur.fetchone()[0] or 0)
                cur.execute("SELECT state, COUNT(*) FROM order_intents GROUP BY state")
                intent_count_by_state = {str(s): int(n) for s, n in cur.fetchall()}
            cur.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='recon_recovery_state'"
            )
            if int(cur.fetchone()[0]) > 0:
                cur.execute("SELECT COUNT(*) FROM recon_recovery_state WHERE recovery_status='halted'")
                state_recovery_halted = int(cur.fetchone()[0] or 0)
        finally:
            conn.close()

    meta_weight_changes = {
        "snapshot_count": len(meta_rows),
        "changed_strategies": 0,
        "total_abs_change": 0.0,
    }
    if len(meta_rows) >= 2:
        prev = _safe_json(meta_rows[-2]["weights_json"])
        curr = _safe_json(meta_rows[-1]["weights_json"])
        names = sorted(set(prev.keys()) | set(curr.keys()))
        changed = 0
        total_abs = 0.0
        for name in names:
            a = float(prev.get(name, 0.0) or 0.0)
            b = float(curr.get(name, 0.0) or 0.0)
            d = abs(b - a)
            if d > 1e-12:
                changed += 1
            total_abs += d
        meta_weight_changes["changed_strategies"] = int(changed)
        meta_weight_changes["total_abs_change"] = float(total_abs)

    top_strategy_activity = [
        {"strategy": str(name), "count": int(count)}
        for name, count in strategy_counter.most_common(5)
    ]
    emergency_stop_count = int(len(emergency_rows))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lookback_hours": float(lookback_hours),
        "data_sources": {
            "trades_db": str(trades_db),
            "state_db": str(state_db),
            "meta_db": str(meta_db),
        },
        "cycles_executed": int(len(cycles)),
        "decisions_generated": int(decisions_generated),
        "intent_count_by_state": intent_count_by_state,
        "liquidity_clamp_count": int(liquidity_clamp_count),
        "slippage_summary": {
            "avg_bps": float(avg_slippage_bps),
            "p90_bps": float(p90_slippage_bps),
        },
        "cycle_latency_ms": {
            "avg": float(avg_cycle_latency_ms),
            "p90": float(p90_cycle_latency_ms),
        },
        "strategy_signal_counts": dict(strategy_counter),
        "top_strategies_by_activity": top_strategy_activity,
        "recon_required_counts": {
            "decision_reason_code_count": int(recon_reason_count),
            "state_recon_required_intents": int(state_recon_required),
            "state_recovery_halted": int(state_recovery_halted),
        },
        "meta_weight_changes": meta_weight_changes,
        "emergency_stop_count": emergency_stop_count,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate ARGUS daily runtime summary")
    p.add_argument("--trades-db", default="data/unified_trades.db")
    p.add_argument("--state-db", default="data/unified_state.db")
    p.add_argument("--meta-db", default="data/meta_weights.db")
    p.add_argument("--lookback-hours", type=float, default=24.0)
    p.add_argument("--output", default="reports/daily_runtime_summary.json")
    args = p.parse_args()

    report = build_daily_runtime_summary(
        trades_db=Path(args.trades_db),
        state_db=Path(args.state_db),
        meta_db=Path(args.meta_db),
        lookback_hours=float(args.lookback_hours),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
