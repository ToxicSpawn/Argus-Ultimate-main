from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillProbabilityEstimate:
    maker_fill_probability: float
    expected_wait_seconds: float
    reason: str


def estimate_fill_probability(
    queue_ahead_notional: float,
    recent_trade_flow_notional_per_second: float,
    cancellation_relief_ratio: float = 0.0,
) -> FillProbabilityEstimate:
    """Estimate the probability of a maker fill given queue depth and trade flow.

    effective_flow = flow * max(0.1, 1 + cancellation_relief_ratio)
    wait = queue_ahead / effective_flow
    prob = 1 / (1 + wait / 10)
    """
    if recent_trade_flow_notional_per_second <= 0:
        return FillProbabilityEstimate(
            maker_fill_probability=0.0,
            expected_wait_seconds=float("inf"),
            reason="no_trade_flow",
        )

    effective_flow = recent_trade_flow_notional_per_second * max(
        0.1, 1.0 + cancellation_relief_ratio
    )
    wait = queue_ahead_notional / effective_flow if effective_flow > 0 else float("inf")
    prob = 1.0 / (1.0 + wait / 10.0)

    return FillProbabilityEstimate(
        maker_fill_probability=round(prob, 6),
        expected_wait_seconds=round(wait, 4),
        reason="estimated",
    )
