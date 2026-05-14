from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Pressure(Enum):
    BUY_PRESSURE = "BUY_PRESSURE"
    SELL_PRESSURE = "SELL_PRESSURE"
    BALANCED = "BALANCED"


@dataclass(frozen=True)
class OrderBookSnapshot:
    bid_size: float
    ask_size: float
    bid_price: float
    ask_price: float


@dataclass(frozen=True)
class ImbalanceSignal:
    imbalance: float
    spread_bps: float
    pressure: Pressure
    reason: str


def compute_imbalance(snapshot: OrderBookSnapshot) -> ImbalanceSignal:
    """Compute order-book imbalance from a snapshot.

    imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    spread_bps = (ask_price - bid_price) / mid * 10_000
    pressure: BUY_PRESSURE if imbalance > 0.2, SELL_PRESSURE if < -0.2, else BALANCED.
    """
    total = snapshot.bid_size + snapshot.ask_size
    if total <= 0:
        return ImbalanceSignal(
            imbalance=0.0,
            spread_bps=0.0,
            pressure=Pressure.BALANCED,
            reason="no_liquidity",
        )

    imbalance = (snapshot.bid_size - snapshot.ask_size) / total

    mid = (snapshot.bid_price + snapshot.ask_price) / 2.0
    if mid <= 0:
        spread_bps = 0.0
    else:
        spread_bps = (snapshot.ask_price - snapshot.bid_price) / mid * 10_000

    if imbalance > 0.2:
        pressure = Pressure.BUY_PRESSURE
        reason = "bid_dominant"
    elif imbalance < -0.2:
        pressure = Pressure.SELL_PRESSURE
        reason = "ask_dominant"
    else:
        pressure = Pressure.BALANCED
        reason = "balanced"

    return ImbalanceSignal(
        imbalance=round(imbalance, 6),
        spread_bps=round(spread_bps, 4),
        pressure=pressure,
        reason=reason,
    )
