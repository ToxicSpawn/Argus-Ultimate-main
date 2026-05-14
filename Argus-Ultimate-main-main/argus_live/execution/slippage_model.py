from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageEstimate:
    expected_bps: float
    worst_case_bps: float
    reason: str


def estimate_slippage(*, spread_bps: float, volatility_bps: float, participation_ratio: float) -> SlippageEstimate:
    expected = (spread_bps * 0.5) + (volatility_bps * 0.2) + (participation_ratio * 50.0)
    return SlippageEstimate(expected, expected * 2.0, "spread+volatility+participation")
