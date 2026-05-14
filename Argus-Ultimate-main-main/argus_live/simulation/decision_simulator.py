"""Decision simulation for pre-trade approval."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionSimulationResult:
    expected_value_bps: float
    downside_bps: float
    upside_bps: float
    variance_proxy: float
    approve: bool
    reason: str


def simulate_decision(
    edge_bps: float,
    slippage_bps: float,
    fee_bps: float,
    confidence: float,
    volatility_bps: float = 10.0,
) -> DecisionSimulationResult:
    """Simulate a trade decision and determine approval.

    expected_value = (edge - slippage - fee) * confidence
    approve if expected_value > 0 AND downside > -max(1, fee)
    """
    expected_value = (edge_bps - slippage_bps - fee_bps) * confidence

    downside = -(slippage_bps + fee_bps + volatility_bps * 0.5)
    upside = edge_bps * confidence - fee_bps
    variance_proxy = volatility_bps * (1.0 + abs(edge_bps - slippage_bps) * 0.1)

    floor = -max(1.0, fee_bps)
    approve = expected_value > 0.0 and downside > floor

    reason = (
        f"EV={expected_value:.2f}bps (edge={edge_bps}-slip={slippage_bps}-fee={fee_bps})"
        f"*conf={confidence:.2f}; downside={downside:.2f} vs floor={floor:.2f}"
    )

    logger.debug("decision_sim: approve=%s %s", approve, reason)

    return DecisionSimulationResult(
        expected_value_bps=expected_value,
        downside_bps=downside,
        upside_bps=upside,
        variance_proxy=variance_proxy,
        approve=approve,
        reason=reason,
    )
