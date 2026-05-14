"""Push 66 — Order Book Imbalance (OBI) alpha signal.

The most proven HFT microstructure signal (IC 0.12-0.18 at 1s horizon
vs IC ~0.02 for RSI-based momentum).

Formula:
    OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    fair_price = mid_price + c1 * normalised_OBI
    reservation_price = fair_price - skew * inventory

Reference: Avellaneda-Stoikov (2008), Cartea (2015),
           Order Book Filtration paper arXiv 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class OBILevel:
    price: float
    qty: float


@dataclass
class OBISignal:
    obi: float                  # [-1, +1]
    mid_price: float
    fair_price: float
    reservation_price: float
    spread: float
    bid_depth: float
    ask_depth: float
    direction: str              # "BUY" | "SELL" | "NEUTRAL"
    strength: float             # abs(obi)


class OBICalculator:
    """Computes OBI and derived signals from a limit order book snapshot."""

    def __init__(
        self,
        depth: int = 5,
        c1: float = 0.5,          # fair price adjustment coefficient
        skew: float = 0.1,        # inventory skew coefficient
        neutral_band: float = 0.1, # |OBI| below this = NEUTRAL
    ):
        self.depth = depth
        self.c1 = c1
        self.skew = skew
        self.neutral_band = neutral_band

    def compute(
        self,
        bids: List[Tuple[float, float]],  # [(price, qty), ...] best first
        asks: List[Tuple[float, float]],  # [(price, qty), ...] best first
        inventory: float = 0.0,
    ) -> OBISignal:
        """Compute OBI signal from top-N order book levels."""
        bid_levels = bids[:self.depth]
        ask_levels = asks[:self.depth]

        bid_qty = sum(q for _, q in bid_levels)
        ask_qty = sum(q for _, q in ask_levels)
        total = bid_qty + ask_qty

        obi = (bid_qty - ask_qty) / total if total > 0 else 0.0

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        mid_price = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0
        spread = best_ask - best_bid if best_bid and best_ask else 0.0

        # Avellaneda-Stoikov fair price
        fair_price = mid_price + self.c1 * obi * spread
        reservation_price = fair_price - self.skew * inventory * spread

        if abs(obi) < self.neutral_band:
            direction = "NEUTRAL"
        elif obi > 0:
            direction = "BUY"
        else:
            direction = "SELL"

        return OBISignal(
            obi=obi,
            mid_price=mid_price,
            fair_price=fair_price,
            reservation_price=reservation_price,
            spread=spread,
            bid_depth=bid_qty,
            ask_depth=ask_qty,
            direction=direction,
            strength=abs(obi),
        )

    def rolling_obi(
        self,
        history: List[float],
        window: int = 20,
    ) -> float:
        """Smoothed OBI using EMA over rolling window."""
        if not history:
            return 0.0
        arr = np.array(history[-window:])
        weights = np.exp(np.linspace(-1.0, 0.0, len(arr)))
        weights /= weights.sum()
        return float(np.dot(weights, arr))
