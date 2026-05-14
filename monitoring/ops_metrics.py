from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


@dataclass
class OpsMetrics:
    reject_histogram: Counter = field(default_factory=Counter)
    route_histogram: Counter = field(default_factory=Counter)
    net_edge_bps: List[float] = field(default_factory=list)
    slippage_bps: List[float] = field(default_factory=list)
    maker_slippage_bps: List[float] = field(default_factory=list)
    taker_slippage_bps: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)
    total_fees: float = 0.0
    total_notional: float = 0.0
    adverse_selection_count: int = 0
    error_count: int = 0
    event_count: int = 0

    def observe_decision(
        self,
        *,
        allowed: bool,
        reason_code: str,
        net_edge_bps: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        self.event_count += 1
        if not allowed:
            self.reject_histogram[str(reason_code or "UNKNOWN")] += 1
        if net_edge_bps is not None:
            self.net_edge_bps.append(float(net_edge_bps))
        if latency_ms is not None:
            self.latency_ms.append(float(latency_ms))

    def observe_trade(
        self,
        *,
        slippage_bps: float,
        fee: float,
        notional: float,
        route: Optional[str] = None,
        adverse_selection: bool = False,
    ) -> None:
        slip = float(slippage_bps)
        self.slippage_bps.append(slip)
        self.total_fees += float(fee or 0.0)
        self.total_notional += float(notional or 0.0)
        route_s = str(route or "").strip().lower()
        if route_s:
            self.route_histogram[route_s] += 1
            if "maker" in route_s:
                self.maker_slippage_bps.append(slip)
            elif "taker" in route_s:
                self.taker_slippage_bps.append(slip)
        if bool(adverse_selection):
            self.adverse_selection_count += 1

    def observe_error(self) -> None:
        self.error_count += 1
        self.event_count += 1

    def summary(self) -> Dict[str, float | Dict[str, int]]:
        fee_churn_ratio = self.total_fees / self.total_notional if self.total_notional > 0 else 0.0
        err_rate = self.error_count / self.event_count if self.event_count > 0 else 0.0
        trade_count = len(self.slippage_bps)
        adverse_rate = self.adverse_selection_count / trade_count if trade_count > 0 else 0.0
        return {
            "reject_histogram": dict(self.reject_histogram),
            "route_histogram": dict(self.route_histogram),
            "net_edge_p50_bps": _percentile(self.net_edge_bps, 0.5),
            "net_edge_p90_bps": _percentile(self.net_edge_bps, 0.9),
            "slippage_p50_bps": _percentile(self.slippage_bps, 0.5),
            "slippage_p90_bps": _percentile(self.slippage_bps, 0.9),
            "maker_slippage_p50_bps": _percentile(self.maker_slippage_bps, 0.5),
            "maker_slippage_p90_bps": _percentile(self.maker_slippage_bps, 0.9),
            "taker_slippage_p50_bps": _percentile(self.taker_slippage_bps, 0.5),
            "taker_slippage_p90_bps": _percentile(self.taker_slippage_bps, 0.9),
            "adverse_selection_rate": float(adverse_rate),
            "fee_churn_ratio": float(fee_churn_ratio),
            "latency_p50_ms": _percentile(self.latency_ms, 0.5),
            "latency_p90_ms": _percentile(self.latency_ms, 0.9),
            "error_rate": float(err_rate),
        }
