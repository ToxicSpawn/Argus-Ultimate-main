from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionAlphaAttribution:
    expected_slippage_bps: float
    realized_slippage_bps: float
    execution_alpha_bps: float
    reason: str


def attribute_execution_alpha(
    expected_slippage_bps: float,
    realized_slippage_bps: float,
) -> ExecutionAlphaAttribution:
    """Attribute execution alpha as expected - realized slippage.

    Positive alpha means we beat expectations (less slippage than expected).
    """
    alpha = expected_slippage_bps - realized_slippage_bps

    if alpha > 0:
        reason = "positive_alpha"
    elif alpha < 0:
        reason = "negative_alpha"
    else:
        reason = "neutral"

    return ExecutionAlphaAttribution(
        expected_slippage_bps=round(expected_slippage_bps, 4),
        realized_slippage_bps=round(realized_slippage_bps, 4),
        execution_alpha_bps=round(alpha, 4),
        reason=reason,
    )
