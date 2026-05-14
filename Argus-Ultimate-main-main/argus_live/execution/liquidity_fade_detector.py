from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiquidityFadeSignal:
    fade_risk: float
    suspicious: bool
    reason: str


def detect_liquidity_fade(
    displayed_notional: float,
    executed_notional: float,
    cancel_rate: float,
) -> LiquidityFadeSignal:
    """Detect potential liquidity fade / spoofing.

    fade_risk = 1 - execution_ratio + cancel_rate
    suspicious if fade_risk > 0.7.
    """
    if displayed_notional <= 0:
        return LiquidityFadeSignal(
            fade_risk=0.0,
            suspicious=False,
            reason="no_displayed_liquidity",
        )

    execution_ratio = min(executed_notional / displayed_notional, 1.0)
    fade_risk = 1.0 - execution_ratio + cancel_rate
    fade_risk = max(0.0, min(fade_risk, 2.0))  # clamp

    suspicious = fade_risk > 0.7

    if suspicious:
        reason = "high_fade_risk"
    else:
        reason = "normal"

    return LiquidityFadeSignal(
        fade_risk=round(fade_risk, 6),
        suspicious=suspicious,
        reason=reason,
    )
