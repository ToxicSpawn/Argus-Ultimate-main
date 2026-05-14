"""
Market Making strategy for Argus-Ultimate v5.0.0.
Quote placement scaffold based on spread capture and inventory skew.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketMakingQuote:
    bid_offset_bps: float
    ask_offset_bps: float
    size_pct: float
    skew: float


class MarketMakingStrategy:
    def __init__(self, base_offset_bps: float = 1.0, inventory_skew_coef: float = 0.5, max_size_pct: float = 0.08):
        self.base_offset_bps = base_offset_bps
        self.inventory_skew_coef = inventory_skew_coef
        self.max_size_pct = max_size_pct

    def generate(self, spread_bps: float, inventory: float, volatility: float) -> MarketMakingQuote:
        skew = inventory * self.inventory_skew_coef
        width = max(self.base_offset_bps, spread_bps * 0.4) * (1.0 + volatility)
        bid_offset = max(width + skew, 0.1)
        ask_offset = max(width - skew, 0.1)
        size_pct = min(0.02 + spread_bps / 200.0, self.max_size_pct)
        return MarketMakingQuote(bid_offset, ask_offset, size_pct, skew)
