"""Liquidation heatmap data for market analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LiquidationLevel:
    """Liquidation level on the order book.
    
    Attributes
    ----------
    price : float
        Price level
    volume : float
        Cumulative liquidation volume at this level
    side : str
        "long" or "short"
    timestamp : float
        Last update timestamp
    """
    price: float = 0.0
    volume: float = 0.0
    side: str = "long"
    timestamp: float = field(default_factory=time.time)


class LiquidationHeatmap:
    """Liquidation heatmap generator.
    
    Tracks estimated liquidation levels based on leverage distribution
    and order book data.
    """
    
    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._levels: List[LiquidationLevel] = []
    
    def update(self, price: float, funding_rate: float = 0.0) -> None:
        """Update heatmap with current price."""
        pass
    
    def get_levels(self, side: Optional[str] = None) -> List[LiquidationLevel]:
        """Get liquidation levels, optionally filtered by side."""
        if side:
            return [l for l in self._levels if l.side == side]
        return self._levels.copy()
    
    def get_total_liquidation_volume(self, side: Optional[str] = None) -> float:
        """Get total liquidation volume."""
        levels = self.get_levels(side)
        return sum(l.volume for l in levels)
