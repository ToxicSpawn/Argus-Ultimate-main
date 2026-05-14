"""Short-horizon forecasting model for microstructure signals."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShortHorizonForecast:
    expected_drift_bps: float
    expected_spread_change_bps: float
    expected_volatility_bps: float
    confidence: float
    reason: str


def forecast_short_horizon(
    imbalance: float,
    volatility_bps: float,
    spread_bps: float,
    momentum_bps: float,
) -> ShortHorizonForecast:
    """Produce a short-horizon forecast from current microstructure state.

    drift = imbalance * 5 + momentum * 0.5
    spread_change is clamped to [-spread_bps, spread_bps]
    expected_volatility = volatility_bps + |imbalance| * 2
    """
    drift = imbalance * 5.0 + momentum_bps * 0.5

    raw_spread_change = imbalance * 2.0
    spread_change = max(-spread_bps, min(spread_bps, raw_spread_change))

    expected_volatility = volatility_bps + abs(imbalance) * 2.0

    # Confidence decreases with higher volatility and extreme imbalance
    confidence = max(0.1, min(1.0, 1.0 - abs(imbalance) * 0.3 - volatility_bps / 200.0))

    reason = (
        f"drift={drift:.2f}bps (imb={imbalance:.3f}, mom={momentum_bps:.2f}); "
        f"spread_chg={spread_change:.2f}bps; vol={expected_volatility:.2f}bps"
    )

    logger.debug("short_horizon_forecast: %s", reason)

    return ShortHorizonForecast(
        expected_drift_bps=drift,
        expected_spread_change_bps=spread_change,
        expected_volatility_bps=expected_volatility,
        confidence=confidence,
        reason=reason,
    )
