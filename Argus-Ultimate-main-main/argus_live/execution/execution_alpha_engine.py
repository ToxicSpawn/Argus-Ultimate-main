from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from argus_live.execution.fill_probability_model import FillProbabilityEstimate
from argus_live.execution.liquidity_fade_detector import LiquidityFadeSignal
from argus_live.execution.microstructure_imbalance import ImbalanceSignal, Pressure

logger = logging.getLogger(__name__)


class Aggression(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class ExecutionAlphaDecision:
    aggression: Aggression
    maker_preferred: bool
    should_slice: bool
    wait_preferred: bool
    reason: str


def build_execution_alpha_decision(
    side: str,
    imbalance: ImbalanceSignal,
    fill_prob: FillProbabilityEstimate,
    fade: LiquidityFadeSignal,
    volatility_bps: float,
) -> ExecutionAlphaDecision:
    """Build an execution alpha decision from microstructure signals.

    Logic:
    - fade suspicious -> LOW aggression, slice, wait
    - buy + BUY_PRESSURE + low fill prob -> HIGH aggression
    - buy + SELL_PRESSURE -> LOW aggression, maker preferred, wait
    - sell + SELL_PRESSURE + low fill prob -> HIGH aggression
    - else -> MEDIUM
    """
    side_upper = side.upper()
    low_fill = fill_prob.maker_fill_probability < 0.5

    # Fade detection overrides everything
    if fade.suspicious:
        return ExecutionAlphaDecision(
            aggression=Aggression.LOW,
            maker_preferred=False,
            should_slice=True,
            wait_preferred=True,
            reason="liquidity_fade_detected",
        )

    # Buy-side logic
    if side_upper == "BUY":
        if imbalance.pressure == Pressure.BUY_PRESSURE and low_fill:
            return ExecutionAlphaDecision(
                aggression=Aggression.HIGH,
                maker_preferred=False,
                should_slice=volatility_bps > 50,
                wait_preferred=False,
                reason="buy_pressure_low_fill",
            )
        if imbalance.pressure == Pressure.SELL_PRESSURE:
            return ExecutionAlphaDecision(
                aggression=Aggression.LOW,
                maker_preferred=True,
                should_slice=False,
                wait_preferred=True,
                reason="buy_into_sell_pressure",
            )

    # Sell-side logic
    if side_upper == "SELL":
        if imbalance.pressure == Pressure.SELL_PRESSURE and low_fill:
            return ExecutionAlphaDecision(
                aggression=Aggression.HIGH,
                maker_preferred=False,
                should_slice=volatility_bps > 50,
                wait_preferred=False,
                reason="sell_pressure_low_fill",
            )
        if imbalance.pressure == Pressure.BUY_PRESSURE:
            return ExecutionAlphaDecision(
                aggression=Aggression.LOW,
                maker_preferred=True,
                should_slice=False,
                wait_preferred=True,
                reason="sell_into_buy_pressure",
            )

    # Default
    return ExecutionAlphaDecision(
        aggression=Aggression.MEDIUM,
        maker_preferred=True,
        should_slice=volatility_bps > 100,
        wait_preferred=False,
        reason="default",
    )
