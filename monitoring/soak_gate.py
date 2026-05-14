from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_TIMEOUT_REASON_TOKENS = ("TIMEOUT",)
_ERROR_REASON_TOKENS = ("ERROR", "FAILED", "EXCEPTION")
# Duplicate-intent guardrail should count execution idempotency failures and
# duplicate-signal outcomes that indicate de-duplication pressure in runtime.
_DUPLICATE_REASON_CODES = {"DUPLICATE_INTENT", "IDEMPOTENT_REPLAY", "DUPLICATE_SIGNAL"}
_RECON_HALT_REASON = "RECONCILIATION_HALT"


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return float(xs[0])
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


@dataclass
class SoakGateThresholds:
    min_duration_seconds: float = 1800.0
    min_decision_count: int = 50
    min_trade_count: Optional[int] = None
    max_error_rate: float = 0.05
    max_timeout_rate: float = 0.05
    max_reconciliation_halts: int = 0
    max_duplicate_intents: int = 0
    max_drawdown_pct: Optional[float] = None
    max_cycle_latency_p90_ms: Optional[float] = None
    max_age_hours: float = 24.0
    report_path: str = "reports/soak_gate_latest.json"


def load_thresholds_from_runtime(runtime_cfg: Dict[str, Any]) -> SoakGateThresholds:
    soak_cfg = dict((runtime_cfg or {}).get("soak_gate") or {})
    def _num(key: str, default: float) -> float:
        v = soak_cfg.get(key)
        if v is None:
            return float(default)
        return float(v)

    def _int(key: str, default: int) -> int:
        v = soak_cfg.get(key)
        if v is None:
            return int(default)
        return int(v)

    def _optional_num(key: str) -> Optional[float]:
        if key not in soak_cfg or soak_cfg.get(key) is None:
            return None
        return float(soak_cfg.get(key))

    def _optional_int(key: str) -> Optional[int]:
        if key not in soak_cfg or soak_cfg.get(key) is None:
            return None
        return int(soak_cfg.get(key))

    min_duration_seconds = soak_cfg.get("min_duration_seconds")
    if min_duration_seconds is None:
        min_window_hours = soak_cfg.get("min_paper_window_hours")
        if min_window_hours is not None:
            min_duration_seconds = float(min_window_hours) * 3600.0
        else:
            min_duration_seconds = 1800.0

    return SoakGateThresholds(
        min_duration_seconds=float(min_duration_seconds),
        min_decision_count=_int("min_decision_count", 50),
        min_trade_count=_optional_int("min_trade_count"),
        max_error_rate=_num("max_error_rate", 0.05),
        max_timeout_rate=_num("max_timeout_rate", 0.05),
        max_reconciliation_halts=_int("max_reconciliation_halts", 0),
        max_duplicate_intents=_int("max_duplicate_intents", 0),
        max_drawdown_pct=_optional_num("max_drawdown_pct"),
        max_cycle_latency_p90_ms=_optional_num("max_cycle_latency_p90_ms"),
        max_age_hours=_num("max_age_hours", 24.0),
        report_path=str(soak_cfg.get("report_path", "reports/soak_gate_latest.json") or "reports/soak_gate_latest.json"),
    )


def _query_metrics(db_path: str, start_ts: float) -> Dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {
            "trade_count": 0,
            "decision_count": 0,
            "reconciliation_halts": 0,
            "duplicate_intent_count": 0,
            "timeout_events": 0,
            "error_events": 0,
            "max_drawdown_pct": 0.0,
            "cycle_latency_p90_ms": 0.0,
            "cycle_latency_sample_count": 0,
            "portfolio_point_count": 0,
            "max_ts": start_ts,
        }
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    trades = cur.execute("SELECT COUNT(*) AS n, COALESCE(MAX(timestamp), ?) AS max_ts FROM trades WHERE timestamp >= ?", (start_ts, start_ts)).fetchone()
    snaps = cur.execute(
        "SELECT reason_code, timestamp FROM decision_snapshots WHERE timestamp >= ?",
        (start_ts,),
    ).fetchall()
    try:
        events = cur.execute(
            "SELECT stage, timestamp, payload_json FROM decision_events WHERE timestamp >= ? AND stage = 'portfolio_update' ORDER BY timestamp ASC",
            (start_ts,),
        ).fetchall()
    except sqlite3.OperationalError:
        events = []
    conn.close()

    decision_count = len(snaps)
    reconciliation_halts = 0
    duplicate_intent_count = 0
    timeout_events = 0
    error_events = 0
    max_snap_ts = start_ts

    for r in snaps:
        reason = str(r["reason_code"] or "").upper()
        ts = float(r["timestamp"] or start_ts)
        if ts > max_snap_ts:
            max_snap_ts = ts
        if reason == _RECON_HALT_REASON:
            reconciliation_halts += 1
            error_events += 1
        if reason in _DUPLICATE_REASON_CODES:
            duplicate_intent_count += 1
        if any(tok in reason for tok in _TIMEOUT_REASON_TOKENS):
            timeout_events += 1
        if any(tok in reason for tok in _ERROR_REASON_TOKENS):
            error_events += 1

    portfolio_values: List[float] = []
    portfolio_timestamps: List[float] = []
    for e in events:
        try:
            payload = json.loads(str(e["payload_json"] or ""))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            continue
        pv = payload.get("portfolio_value_aud")
        if pv is None:
            continue
        try:
            pv_f = float(pv)
            ts_f = float(e["timestamp"] or start_ts)
        except Exception:
            continue
        if pv_f <= 0:
            continue
        portfolio_values.append(pv_f)
        portfolio_timestamps.append(ts_f)

    max_drawdown_pct = 0.0
    if portfolio_values:
        peak = float(portfolio_values[0])
        for value in portfolio_values:
            peak = max(peak, float(value))
            if peak > 0:
                max_drawdown_pct = max(max_drawdown_pct, (peak - float(value)) / peak)

    cycle_latencies_ms: List[float] = []
    for idx in range(1, len(portfolio_timestamps)):
        dt_s = float(portfolio_timestamps[idx] - portfolio_timestamps[idx - 1])
        if dt_s > 0:
            cycle_latencies_ms.append(dt_s * 1000.0)

    return {
        "trade_count": int(trades["n"] or 0) if trades else 0,
        "decision_count": int(decision_count),
        "reconciliation_halts": int(reconciliation_halts),
        "duplicate_intent_count": int(duplicate_intent_count),
        "timeout_events": int(timeout_events),
        "error_events": int(error_events),
        "max_drawdown_pct": float(max_drawdown_pct),
        "cycle_latency_p90_ms": float(_percentile(cycle_latencies_ms, 0.90)),
        "cycle_latency_sample_count": int(len(cycle_latencies_ms)),
        "portfolio_point_count": int(len(portfolio_values)),
        "max_ts": float(max(float(trades["max_ts"] or start_ts), max_snap_ts)) if trades else max_snap_ts,
    }


def evaluate_soak_gate(
    *,
    thresholds: SoakGateThresholds,
    db_path: str,
    start_ts: float,
    end_ts: Optional[float] = None,
) -> Dict[str, Any]:
    metrics = _query_metrics(db_path=db_path, start_ts=float(start_ts))
    now_ts = datetime.now(timezone.utc).timestamp()
    final_end_ts = float(end_ts if end_ts is not None else max(metrics["max_ts"], now_ts))
    duration = max(0.0, final_end_ts - float(start_ts))

    decision_count = int(metrics["decision_count"])
    denom = max(1, decision_count)
    error_rate = float(metrics["error_events"]) / float(denom)
    timeout_rate = float(metrics["timeout_events"]) / float(denom)

    failures: List[str] = []
    if duration < float(thresholds.min_duration_seconds):
        failures.append(
            f"duration_seconds={duration:.2f} below min_duration_seconds={thresholds.min_duration_seconds:.2f}"
        )
    if decision_count < int(thresholds.min_decision_count):
        failures.append(
            f"decision_count={decision_count} below min_decision_count={int(thresholds.min_decision_count)}"
        )
    if thresholds.min_trade_count is not None and int(metrics["trade_count"]) < int(thresholds.min_trade_count):
        failures.append(
            f"trade_count={int(metrics['trade_count'])} below min_trade_count={int(thresholds.min_trade_count)}"
        )
    if error_rate > float(thresholds.max_error_rate):
        failures.append(f"error_rate={error_rate:.6f} above max_error_rate={thresholds.max_error_rate:.6f}")
    if timeout_rate > float(thresholds.max_timeout_rate):
        failures.append(f"timeout_rate={timeout_rate:.6f} above max_timeout_rate={thresholds.max_timeout_rate:.6f}")
    if int(metrics["reconciliation_halts"]) > int(thresholds.max_reconciliation_halts):
        failures.append(
            "reconciliation_halts="
            f"{int(metrics['reconciliation_halts'])} above max_reconciliation_halts={int(thresholds.max_reconciliation_halts)}"
        )
    if int(metrics["duplicate_intent_count"]) > int(thresholds.max_duplicate_intents):
        failures.append(
            "duplicate_intent_count="
            f"{int(metrics['duplicate_intent_count'])} above max_duplicate_intents={int(thresholds.max_duplicate_intents)}"
        )
    if thresholds.max_drawdown_pct is not None:
        if int(metrics["portfolio_point_count"]) <= 0:
            failures.append("max_drawdown_pct unavailable (no portfolio_update events)")
        elif float(metrics["max_drawdown_pct"]) > float(thresholds.max_drawdown_pct):
            failures.append(
                f"max_drawdown_pct={float(metrics['max_drawdown_pct']):.6f} above max_drawdown_pct={float(thresholds.max_drawdown_pct):.6f}"
            )
    if thresholds.max_cycle_latency_p90_ms is not None:
        if int(metrics["cycle_latency_sample_count"]) <= 0:
            failures.append("cycle_latency_p90_ms unavailable (need >=2 portfolio_update events)")
        elif float(metrics["cycle_latency_p90_ms"]) > float(thresholds.max_cycle_latency_p90_ms):
            failures.append(
                "cycle_latency_p90_ms="
                f"{float(metrics['cycle_latency_p90_ms']):.3f} above max_cycle_latency_p90_ms={float(thresholds.max_cycle_latency_p90_ms):.3f}"
            )

    return {
        "status": "PASS" if not failures else "FAIL",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "start_ts": float(start_ts),
        "end_ts": float(final_end_ts),
        "duration_seconds": float(duration),
        "metrics": {
            "trade_count": int(metrics["trade_count"]),
            "decision_count": int(metrics["decision_count"]),
            "error_events": int(metrics["error_events"]),
            "timeout_events": int(metrics["timeout_events"]),
            "reconciliation_halts": int(metrics["reconciliation_halts"]),
            "duplicate_intent_count": int(metrics["duplicate_intent_count"]),
            "error_rate": float(error_rate),
            "timeout_rate": float(timeout_rate),
            "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
            "cycle_latency_p90_ms": float(metrics["cycle_latency_p90_ms"]),
            "cycle_latency_sample_count": int(metrics["cycle_latency_sample_count"]),
            "portfolio_point_count": int(metrics["portfolio_point_count"]),
        },
        "thresholds": asdict(thresholds),
        "fail_reasons": failures,
    }


def write_soak_gate_report(report: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path = out_dir / f"soak_gate_{stamp}.json"
    latest_path = out_dir / "soak_gate_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=True)
    ts_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return {"timestamped": str(ts_path), "latest": str(latest_path)}
