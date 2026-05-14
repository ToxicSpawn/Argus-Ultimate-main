#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _table_exists(db_path: Path, table_name: str) -> bool:
    if not db_path.exists():
        return False
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (str(table_name),),
        ).fetchone()
        return bool(row)
    finally:
        conn.close()


def _safe_json(raw: Any) -> Dict[str, Any]:
    try:
        out = json.loads(str(raw or "{}"))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _fetch_rows(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> List[sqlite3.Row]:
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


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(float(v) for v in values)
    idx = int(round(max(0.0, min(1.0, q)) * (len(vs) - 1)))
    return float(vs[idx])


def build_dashboard_views(
    *,
    omega_db: Path,
    trades_db: Path,
    state_db: Path,
    meta_db: Path,
) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1) System health view.
    latest_health = {}
    if _table_exists(omega_db, "system_health_snapshots"):
        rows = _fetch_rows(
            omega_db,
            """
            SELECT timestamp, cycles_completed, avg_latency_ms, errors_last_hour, warnings_last_hour, event_loop_delay_ms
            FROM system_health_snapshots
            ORDER BY id DESC
            LIMIT 1
            """,
        )
        if rows:
            r = rows[0]
            latest_health = {
                "timestamp": str(r["timestamp"]),
                "cycles_completed": int(r["cycles_completed"] or 0),
                "avg_cycle_latency_ms": float(r["avg_latency_ms"] or 0.0),
                "errors_last_hour": int(r["errors_last_hour"] or 0),
                "warnings_last_hour": int(r["warnings_last_hour"] or 0),
                "event_loop_delay_ms": float(r["event_loop_delay_ms"] or 0.0),
            }

    # 2) Trading intelligence view.
    strategy_activity: Dict[str, int] = {}
    regime_state = ""
    target_exposures: Dict[str, Dict[str, float]] = {}
    microstructure_state: Dict[str, Dict[str, Any]] = {}
    latest_decisions = _fetch_rows(
        trades_db,
        """
        SELECT strategy, details_json, execution_plan_json
        FROM decision_snapshots
        ORDER BY timestamp DESC
        LIMIT 500
        """,
    ) if _table_exists(trades_db, "decision_snapshots") else []
    for row in latest_decisions:
        strategy = str(row["strategy"] or "").strip()
        if strategy:
            strategy_activity[strategy] = int(strategy_activity.get(strategy, 0) + 1)
        details = _safe_json(row["details_json"])
        symbol = str(details.get("symbol") or "")
        if not regime_state:
            regime_state = str(details.get("regime_label") or "")
        if symbol:
            if details.get("target_exposure_pct") is not None:
                target_exposures[symbol] = {
                    "target_exposure_pct": float(details.get("target_exposure_pct") or 0.0),
                    "current_exposure_pct": float(details.get("current_exposure_pct") or 0.0),
                    "delta_exposure_pct": float(details.get("delta_exposure_pct") or 0.0),
                }
            if details.get("spread_bps") is not None:
                microstructure_state[symbol] = {
                    "spread_bps": float(details.get("spread_bps") or 0.0),
                    "order_book_imbalance": float(details.get("order_book_imbalance") or 0.0),
                    "microprice": float(details.get("microprice") or 0.0),
                    "trade_velocity": float(details.get("trade_velocity") or 0.0),
                    "liquidity_vacuum_flag": bool(details.get("liquidity_vacuum_flag", False)),
                    "adverse_selection_risk": float(details.get("adverse_selection_risk") or 0.0),
                    "microstructure_bias": str(details.get("microstructure_bias") or "neutral"),
                }

    strategy_weights = {}
    if _table_exists(meta_db, "meta_weight_snapshots"):
        rows = _fetch_rows(
            meta_db,
            """
            SELECT weights_json
            FROM meta_weight_snapshots
            ORDER BY id DESC
            LIMIT 1
            """,
        )
        if rows:
            strategy_weights = _safe_json(rows[0]["weights_json"])

    # 3) Execution + risk view.
    liquidity_scores: List[float] = []
    clamp_count = 0
    execution_plan_counts: Dict[str, int] = {}
    if latest_decisions:
        for row in latest_decisions:
            details = _safe_json(row["details_json"])
            if details.get("liquidity_score") is not None:
                liquidity_scores.append(float(details.get("liquidity_score") or 0.0))
            if bool(details.get("liquidity_clamp_flag", False)):
                clamp_count += 1
            plan = _safe_json(row["execution_plan_json"])
            order_type = str(plan.get("order_type") or "").strip().lower()
            if order_type:
                execution_plan_counts[order_type] = int(execution_plan_counts.get(order_type, 0) + 1)

    slippage_bps: List[float] = []
    if _table_exists(trades_db, "trades"):
        rows = _fetch_rows(trades_db, "SELECT slippage FROM trades ORDER BY timestamp DESC LIMIT 500")
        for row in rows:
            if row["slippage"] is None:
                continue
            slippage_bps.append(float(row["slippage"] or 0.0) * 10000.0)

    intent_state_counts: Dict[str, int] = {}
    recon_required_count = 0
    position_exposure = 0.0
    if _table_exists(state_db, "order_intents"):
        rows = _fetch_rows(state_db, "SELECT state, COUNT(*) AS n FROM order_intents GROUP BY state")
        for row in rows:
            state = str(row["state"] or "")
            intent_state_counts[state] = int(row["n"] or 0)
        recon_required_count = int(intent_state_counts.get("RECON_REQUIRED", 0))
    if _table_exists(state_db, "positions"):
        rows = _fetch_rows(state_db, "SELECT quantity, current_price FROM positions")
        for row in rows:
            position_exposure += abs(float(row["quantity"] or 0.0) * float(row["current_price"] or 0.0))

    return {
        "generated_at_utc": now_iso,
        "system_health": {
            "uptime_seconds": 0.0,
            "engine_state": "unknown",
            "emergency_state": False,
            **latest_health,
        },
        "trading_intelligence": {
            "strategy_activity": strategy_activity,
            "strategy_weights": strategy_weights,
            "regime_state": regime_state,
            "target_exposures": target_exposures,
            "microstructure_state": microstructure_state,
        },
        "execution_risk": {
            "liquidity_score_avg": float(sum(liquidity_scores) / len(liquidity_scores)) if liquidity_scores else 0.0,
            "liquidity_score_p10": _pct(liquidity_scores, 0.10),
            "liquidity_clamp_count": int(clamp_count),
            "execution_plan_type_counts": execution_plan_counts,
            "slippage_summary_bps": {
                "avg": float(sum(slippage_bps) / len(slippage_bps)) if slippage_bps else 0.0,
                "p90": _pct(slippage_bps, 0.90),
            },
            "intent_state_counts": intent_state_counts,
            "recon_required_count": int(recon_required_count),
            "position_exposure_notional": float(position_exposure),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Export ARGUS runtime monitoring views")
    p.add_argument("--omega-db", default="data/unified_omega.db")
    p.add_argument("--trades-db", default="data/unified_trades.db")
    p.add_argument("--state-db", default="data/unified_state.db")
    p.add_argument("--meta-db", default="data/meta_weights.db")
    p.add_argument("--output", default="reports/runtime_dashboard_latest.json")
    args = p.parse_args()

    payload = build_dashboard_views(
        omega_db=Path(args.omega_db),
        trades_db=Path(args.trades_db),
        state_db=Path(args.state_db),
        meta_db=Path(args.meta_db),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
