"""Regime transition prediction engine."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeTransitionSignal:
    next_regime: str
    probability: float
    horizon_cycles: int
    confidence: float
    reason: str


def predict_regime_transition(
    volatility_slope: float,
    spread_widening_bps_per_cycle: float,
    imbalance_acceleration: float,
    correlation_tightening: float,
) -> RegimeTransitionSignal:
    """Predict the next regime transition from microstructure signals.

    Computes a stress_score and trend_score from the inputs, then determines
    whether the market is transitioning to STRESSED or TRENDING.

    Returns a frozen RegimeTransitionSignal.
    """
    stress_score = (
        volatility_slope * 1.0
        + spread_widening_bps_per_cycle * 0.8
        + abs(imbalance_acceleration) * 0.6
        + correlation_tightening * 1.2
    )

    trend_score = (
        abs(imbalance_acceleration) * 1.0
        + volatility_slope * 0.3
    )

    threshold = max(0.5, trend_score)

    if stress_score >= threshold:
        next_regime = "STRESSED"
        raw_probability = stress_score / 5.0
        reason = (
            f"stress_score={stress_score:.3f} >= threshold={threshold:.3f}; "
            f"vol_slope={volatility_slope}, spread_widen={spread_widening_bps_per_cycle}"
        )
    else:
        next_regime = "TRENDING"
        raw_probability = trend_score / 5.0
        reason = (
            f"trend_score={trend_score:.3f} > stress_score={stress_score:.3f}; "
            f"imbalance_accel={imbalance_acceleration}"
        )

    probability = max(0.0, min(1.0, raw_probability))
    confidence = probability * 0.9
    horizon_cycles = max(1, int(10 - stress_score * 2))

    logger.debug(
        "regime_transition: next=%s prob=%.3f horizon=%d",
        next_regime, probability, horizon_cycles,
    )

    return RegimeTransitionSignal(
        next_regime=next_regime,
        probability=probability,
        horizon_cycles=horizon_cycles,
        confidence=confidence,
        reason=reason,
    )
