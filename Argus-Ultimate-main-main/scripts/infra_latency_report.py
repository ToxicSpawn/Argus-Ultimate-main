#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    pos = max(0.0, min(1.0, float(q))) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (name,)).fetchone()
    return row is not None


def _to_bps(slippage: Any) -> float:
    try:
        v = float(slippage or 0.0)
    except Exception:
        return 0.0
    # Handle fractional representation (e.g. 0.0007 == 7bps).
    return v * 10000.0 if abs(v) < 1.0 else v


def _safe_json(raw: Any) -> Dict[str, Any]:
    txt = str(raw or "").strip()
    if not txt:
        return {}
    try:
        out = json.loads(txt)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def build_report(db_path: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    slippage_bps: List[float] = []
    latency_ms: List[float] = []
    net_edge_bps: List[float] = []
    total_fees = 0.0
    total_notional = 0.0
    reject_hist = Counter()
    route_hist = Counter()
    timeout_count = 0
    error_count = 0
    decision_count = 0

    trade_count = 0
    if _table_exists(cur, "trades"):
        for row in cur.execute("SELECT slippage, commission, value, price, size FROM trades"):
            trade_count += 1
            slippage_bps.append(_to_bps(row["slippage"]))
            try:
                total_fees += float(row["commission"] or 0.0)
            except Exception:
                pass
            try:
                notional = float(row["value"] or 0.0)
                if notional <= 0:
                    notional = float(row["price"] or 0.0) * float(row["size"] or 0.0)
                total_notional += notional
            except Exception:
                pass

    if _table_exists(cur, "decision_snapshots"):
        query = """
            SELECT allowed, reason_code, details_json, cost_json, execution_plan_json
            FROM decision_snapshots
        """
        for row in cur.execute(query):
            decision_count += 1
            allowed = int(row["allowed"] or 0) == 1
            reason = str(row["reason_code"] or "").strip()
            if not allowed:
                reject_hist[reason or "UNKNOWN"] += 1
            r_up = reason.upper()
            if "TIMEOUT" in r_up:
                timeout_count += 1
            if "ERROR" in r_up or "EXCEPTION" in r_up or "FAIL" in r_up:
                error_count += 1

            details = _safe_json(row["details_json"])
            costs = _safe_json(row["cost_json"])
            plan = _safe_json(row["execution_plan_json"])

            v = costs.get("net_edge_bps")
            if isinstance(v, (int, float)):
                net_edge_bps.append(float(v))

            for key in (
                "latency_ms",
                "cycle_latency_ms",
                "decision_latency_ms",
                "execution_latency_ms",
                "route_latency_ms",
                "total_latency_ms",
            ):
                val = details.get(key)
                if isinstance(val, (int, float)):
                    latency_ms.append(float(val))
            route = plan.get("route") or details.get("route") or ""
            route = str(route).strip().lower()
            if route:
                route_hist[route] += 1

    conn.close()
    fee_churn_ratio = (total_fees / total_notional) if total_notional > 0 else 0.0
    timeout_rate = (timeout_count / decision_count) if decision_count > 0 else 0.0
    error_rate = (error_count / decision_count) if decision_count > 0 else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(db_path),
        "trade_count": int(trade_count),
        "decision_count": int(decision_count),
        "reject_histogram": dict(reject_hist),
        "route_histogram": dict(route_hist),
        "net_edge_p50_bps": _percentile(net_edge_bps, 0.50),
        "net_edge_p90_bps": _percentile(net_edge_bps, 0.90),
        "net_edge_p99_bps": _percentile(net_edge_bps, 0.99),
        "slippage_p50_bps": _percentile(slippage_bps, 0.50),
        "slippage_p90_bps": _percentile(slippage_bps, 0.90),
        "slippage_p99_bps": _percentile(slippage_bps, 0.99),
        "latency_p50_ms": _percentile(latency_ms, 0.50),
        "latency_p90_ms": _percentile(latency_ms, 0.90),
        "latency_p99_ms": _percentile(latency_ms, 0.99),
        "timeout_rate": float(timeout_rate),
        "error_rate": float(error_rate),
        "fee_churn_ratio": float(fee_churn_ratio),
        "total_fees": float(total_fees),
        "total_notional": float(total_notional),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate infra latency/jitter report from SQLite ledger")
    p.add_argument("--db", default="data/unified_trades.db")
    p.add_argument("--output-dir", default="reports/infra")
    args = p.parse_args()

    report = build_report(args.db)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    out_json = out_dir / f"infra_latency_report_{stamp}.json"
    out_txt = out_dir / f"infra_latency_report_{stamp}.txt"
    out_latest = out_dir / "infra_latency_report_latest.json"
    out_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    out_latest.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    out_txt.write_text(
        "\n".join(
            [
                f"generated_at={report['generated_at']}",
                f"trade_count={report['trade_count']}",
                f"decision_count={report['decision_count']}",
                f"latency_p50_ms={report['latency_p50_ms']:.4f}",
                f"latency_p90_ms={report['latency_p90_ms']:.4f}",
                f"latency_p99_ms={report['latency_p99_ms']:.4f}",
                f"slippage_p50_bps={report['slippage_p50_bps']:.4f}",
                f"slippage_p90_bps={report['slippage_p90_bps']:.4f}",
                f"slippage_p99_bps={report['slippage_p99_bps']:.4f}",
                f"net_edge_p50_bps={report['net_edge_p50_bps']:.4f}",
                f"net_edge_p90_bps={report['net_edge_p90_bps']:.4f}",
                f"net_edge_p99_bps={report['net_edge_p99_bps']:.4f}",
                f"timeout_rate={report['timeout_rate']:.8f}",
                f"error_rate={report['error_rate']:.8f}",
                f"fee_churn_ratio={report['fee_churn_ratio']:.8f}",
                f"reject_histogram={json.dumps(report['reject_histogram'], ensure_ascii=True)}",
                f"route_histogram={json.dumps(report['route_histogram'], ensure_ascii=True)}",
            ]
        ),
        encoding="utf-8",
    )
    print(str(out_json.resolve()))
    print(str(out_txt.resolve()))
    print(str(out_latest.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
