"""Push 78 — PositionSizer: three unified sizing methods.

Methods:
  KELLY         — fractional Kelly: equity * kelly_frac * strength / price
  FIXED_FRAC    — risk% per trade / (atr_mult * ATR): units risked
  VOL_ADJUSTED  — target_vol / realised_vol * equity / price

All methods enforce:
  - max_position_pct cap (% of equity)
  - min_qty floor
  - Returns 0 if price or equity <= 0

Usage:
    sizer = PositionSizer(method=SizingMethod.KELLY)
    qty = sizer.size(equity=10000, price=50000, strength=0.7)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SizingMethod(str, Enum):
    KELLY        = "kelly"
    FIXED_FRAC   = "fixed_frac"
    VOL_ADJUSTED = "vol_adjusted"


@dataclass
class SizerConfig:
    method:            SizingMethod = SizingMethod.KELLY
    kelly_fraction:    float = 0.25       # fractional Kelly multiplier
    risk_per_trade_pct: float = 1.0       # % equity risked per trade (FIXED_FRAC)
    atr_mult:          float = 1.5        # stop distance = atr_mult * ATR
    target_vol_pct:    float = 15.0       # annualised target volatility %
    max_position_pct:  float = 20.0       # max % of equity in single position
    min_qty:           float = 1e-6
    periods_per_year:  int   = 252


class PositionSizer:
    """Unified position sizing calculator.

    Args:
        config: SizerConfig
    """

    def __init__(self, config: Optional[SizerConfig] = None):
        self.config = config or SizerConfig()
        self._audit: List[dict] = []

    def size(
        self,
        equity:   float,
        price:    float,
        strength: float = 0.5,
        atr:      float = 0.0,
        realised_vol_pct: float = 0.0,
        method:   Optional[SizingMethod] = None,
    ) -> float:
        """Calculate position size in base units.

        Args:
            equity:           Current portfolio equity
            price:            Current asset price
            strength:         Signal strength [0,1] (used by Kelly)
            atr:              ATR value (used by FIXED_FRAC)
            realised_vol_pct: Annualised realised vol % (used by VOL_ADJUSTED)
            method:           Override config method for this call

        Returns:
            Position size in base units (e.g. BTC).
        """
        if price <= 0 or equity <= 0:
            return 0.0

        m = method or self.config.method
        if m == SizingMethod.KELLY:
            qty = self._kelly(equity, price, strength)
        elif m == SizingMethod.FIXED_FRAC:
            qty = self._fixed_frac(equity, price, atr)
        else:
            qty = self._vol_adjusted(equity, price, realised_vol_pct)

        # Cap at max_position_pct
        max_qty = (equity * self.config.max_position_pct / 100) / price
        qty     = min(qty, max_qty)

        if qty < self.config.min_qty:
            return 0.0

        self._audit.append({
            "method": m.value, "equity": equity, "price": price,
            "strength": strength, "qty": round(qty, 8),
        })
        return round(qty, 8)

    # ------------------------------------------------------------------
    # Sizing methods
    # ------------------------------------------------------------------

    def _kelly(self, equity: float, price: float, strength: float) -> float:
        """Fractional Kelly: equity * fraction * strength / price."""
        raw = equity * self.config.kelly_fraction * max(0.0, min(strength, 1.0))
        return raw / price

    def _fixed_frac(
        self,
        equity: float,
        price:  float,
        atr:    float,
    ) -> float:
        """Fixed fractional: risk$ / stop_distance.

        risk$          = equity * risk_per_trade_pct / 100
        stop_distance  = atr_mult * ATR (in price units)
        qty            = risk$ / stop_distance
        """
        risk_usd       = equity * self.config.risk_per_trade_pct / 100
        stop_distance  = self.config.atr_mult * atr if atr > 0 else price * 0.01
        return risk_usd / stop_distance

    def _vol_adjusted(
        self,
        equity:          float,
        price:           float,
        realised_vol_pct: float,
    ) -> float:
        """Volatility-targeting: allocate so portfolio vol = target_vol.

        weight = target_vol / realised_vol
        qty    = weight * equity / price
        """
        if realised_vol_pct <= 0:
            realised_vol_pct = self.config.target_vol_pct  # neutral: weight=1
        weight = self.config.target_vol_pct / realised_vol_pct
        weight = min(weight, 2.0)   # cap at 2x leverage
        return weight * equity / price

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def realised_vol(
        self,
        prices: List[float],
        annualise: bool = True,
    ) -> float:
        """Compute annualised realised volatility from price series."""
        if len(prices) < 2:
            return 0.0
        returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
            if prices[i - 1] > 0
        ]
        if not returns:
            return 0.0
        n    = len(returns)
        mean = sum(returns) / n
        std  = math.sqrt(sum((r - mean) ** 2 for r in returns) / n)
        return std * math.sqrt(self.config.periods_per_year) * 100 if annualise else std * 100

    @property
    def audit(self) -> List[dict]:
        return list(self._audit)

    def clear_audit(self) -> None:
        self._audit.clear()
