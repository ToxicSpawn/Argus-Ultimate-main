#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def build_report(db_path: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    trades = cur.execute("SELECT price, size, slippage, commission, value FROM trades").fetchall()
    snapshots = cur.execute("SELECT allowed, reason_code, cost_json FROM decision_snapshots").fetchall()

    reject_hist = Counter()
    net_edges: List[float] = []
    slippage_bps: List[float] = []
    total_fees = 0.0
    total_notional = 0.0

    for t in trades:
        total_fees += float(t["commission"] or 0.0)
        total_notional += float(t["value"] or (float(t["price"] or 0.0) * float(t["size"] or 0.0)))
        slippage_bps.append(float(t["slippage"] or 0.0) * 10000.0)

    for s in snapshots:
        if int(s["allowed"] or 0) == 0:
            reject_hist[str(s["reason_code"] or "UNKNOWN")] += 1
        raw = str(s["cost_json"] or "")
        if raw:
            try:
                c = json.loads(raw)
                v = c.get("net_edge_bps")
                if v is not None:
                    net_edges.append(float(v))
            except Exception:
                pass

    conn.close()
    fee_churn_ratio = total_fees / total_notional if total_notional > 0 else 0.0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_db": db_path,
        "trade_count": len(trades),
        "decision_count": len(snapshots),
        "reject_histogram": dict(reject_hist),
        "slippage_p50_bps": _percentile(slippage_bps, 0.5),
        "slippage_p90_bps": _percentile(slippage_bps, 0.9),
        "net_edge_p50_bps": _percentile(net_edges, 0.5),
        "net_edge_p90_bps": _percentile(net_edges, 0.9),
        "fee_churn_ratio": float(fee_churn_ratio),
        "total_fees": float(total_fees),
        "total_notional": float(total_notional),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate daily ops report from ledger DB")
    p.add_argument("--db", default="data/unified_trades.db")
    p.add_argument("--output-dir", default="reports")
    args = p.parse_args()

    report = build_report(args.db)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    out_json = out_dir / f"daily_report_{stamp}.json"
    out_txt = out_dir / f"daily_report_{stamp}.txt"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    out_txt.write_text(
        "\n".join(
            [
                f"generated_at={report['generated_at']}",
                f"trade_count={report['trade_count']}",
                f"decision_count={report['decision_count']}",
                f"slippage_p50_bps={report['slippage_p50_bps']:.4f}",
                f"slippage_p90_bps={report['slippage_p90_bps']:.4f}",
                f"net_edge_p50_bps={report['net_edge_p50_bps']:.4f}",
                f"net_edge_p90_bps={report['net_edge_p90_bps']:.4f}",
                f"fee_churn_ratio={report['fee_churn_ratio']:.8f}",
                f"reject_histogram={json.dumps(report['reject_histogram'], ensure_ascii=True)}",
            ]
        ),
        encoding="utf-8",
    )
    print(str(out_json))
    print(str(out_txt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
