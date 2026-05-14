from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class VenueHealth:
    venue_id: str
    latency_ms: float
    reject_rate: float
    stale_data: bool
    healthy: bool
    reason: str
    score: float = 100.0
    avg_execution_alpha_bps: float = 0.0
    avg_slippage_bps: float = 0.0
    adverse_selection_bps: float = 0.0


@dataclass(frozen=True)
class VenueHealthSnapshot:
    venue_id: str
    score: float
    healthy: bool
    preferred: bool
    reason: str
    latency_ms: float
    reject_rate: float
    avg_execution_alpha_bps: float
    avg_slippage_bps: float
    adverse_selection_bps: float


def assess_venue_health(
    venue: str,
    median_latency_ms: float,
    reject_rate: float,
    stale_data: bool,
    max_latency_ms: float = 500.0,
    max_reject_rate: float = 0.10,
    avg_execution_alpha_bps: float = 0.0,
    avg_slippage_bps: float = 0.0,
    adverse_selection_bps: float = 0.0,
) -> VenueHealth:
    reasons: list[str] = []
    score = 100.0
    if median_latency_ms >= max_latency_ms:
        reasons.append(f"latency breach: {median_latency_ms:.1f} ms >= {max_latency_ms:.1f} ms")
        score -= min(35.0, (median_latency_ms - max_latency_ms) / max(max_latency_ms, 1.0) * 50.0)
    else:
        score -= min(15.0, median_latency_ms / max(max_latency_ms, 1.0) * 10.0)

    if reject_rate > max_reject_rate:
        reasons.append(f"reject rate {reject_rate:.2%} > {max_reject_rate:.2%}")
        score -= min(35.0, (reject_rate - max_reject_rate) * 200.0)
    else:
        score -= reject_rate * 20.0

    if stale_data:
        reasons.append("stale data feed")
        score -= 30.0

    if avg_execution_alpha_bps < 0:
        reasons.append(f"negative execution alpha {avg_execution_alpha_bps:.2f}bps")
        score -= min(20.0, abs(avg_execution_alpha_bps) * 2.0)

    if avg_slippage_bps > 0:
        score -= min(15.0, avg_slippage_bps * 0.5)
        if avg_slippage_bps > 20.0:
            reasons.append(f"high slippage {avg_slippage_bps:.2f}bps")

    if adverse_selection_bps > 0:
        score -= min(20.0, adverse_selection_bps)
        if adverse_selection_bps > 5.0:
            reasons.append(f"adverse selection {adverse_selection_bps:.2f}bps")

    healthy = score >= 50.0 and not stale_data and reject_rate <= max(max_reject_rate * 2, 0.20)
    reason = "; ".join(reasons) if reasons else "all checks passed"
    return VenueHealth(
        venue_id=venue,
        latency_ms=median_latency_ms,
        reject_rate=reject_rate,
        stale_data=stale_data,
        healthy=healthy,
        reason=reason,
        score=max(0.0, min(100.0, score)),
        avg_execution_alpha_bps=avg_execution_alpha_bps,
        avg_slippage_bps=avg_slippage_bps,
        adverse_selection_bps=adverse_selection_bps,
    )


class VenueHealthModel:
    def __init__(self, max_latency_ms: float = 500.0, max_reject_rate: float = 0.10) -> None:
        self.max_latency_ms = max_latency_ms
        self.max_reject_rate = max_reject_rate

    def snapshot(self, *, venue: str, fills: Iterable[object], stale_data: bool = False) -> VenueHealthSnapshot:
        fills = list(fills)
        if not fills:
            health = assess_venue_health(
                venue=venue,
                median_latency_ms=0.0,
                reject_rate=0.0,
                stale_data=stale_data,
                max_latency_ms=self.max_latency_ms,
                max_reject_rate=self.max_reject_rate,
            )
        else:
            latency_ms = mean(float(getattr(f, 'latency_ms', 0.0)) for f in fills)
            reject_rate = sum(int(getattr(f, 'reject_flag', 0)) for f in fills) / max(len(fills), 1)
            avg_execution_alpha_bps = mean(float(getattr(f, 'execution_alpha_bps', 0.0)) for f in fills)
            avg_slippage_bps = mean(abs(float(getattr(f, 'slippage_bps', 0.0))) for f in fills)
            adverse_selection_bps = mean(max(0.0, float(getattr(f, 'adverse_price_move_bps', 0.0))) for f in fills)
            health = assess_venue_health(
                venue=venue,
                median_latency_ms=latency_ms,
                reject_rate=reject_rate,
                stale_data=stale_data,
                max_latency_ms=self.max_latency_ms,
                max_reject_rate=self.max_reject_rate,
                avg_execution_alpha_bps=avg_execution_alpha_bps,
                avg_slippage_bps=avg_slippage_bps,
                adverse_selection_bps=adverse_selection_bps,
            )
        return VenueHealthSnapshot(
            venue_id=health.venue_id,
            score=health.score,
            healthy=health.healthy,
            preferred=health.healthy and health.score >= 70.0,
            reason=health.reason,
            latency_ms=health.latency_ms,
            reject_rate=health.reject_rate,
            avg_execution_alpha_bps=health.avg_execution_alpha_bps,
            avg_slippage_bps=health.avg_slippage_bps,
            adverse_selection_bps=health.adverse_selection_bps,
        )
