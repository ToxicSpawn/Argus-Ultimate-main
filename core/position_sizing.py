"""
core/position_sizing.py

ATR-volatility targeting + Fractional Kelly + portfolio heat gate.

Usage
-----
    from core.position_sizing import PositionSizer

    sizer = PositionSizer(config)          # pass argus config object
    result = sizer.size(
        equity_aud   = 1_500.0,
        atr          = 0.0042,             # ATR as fraction of price  (ATR / mid_price)
        win_rate     = 0.54,
        avg_win      = 0.018,
        avg_loss     = 0.011,
        open_heat    = 0.12,               # sum of current open risk fractions
        kelly_fraction = None,             # override; computed internally if None
    )
    # result.size_pct   – fraction of equity to risk
    # result.size_usd   – notional in USD (equity * aud_to_usd * size_pct)
    # result.method     – audit string
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from core.capital_tier import CapitalTier, classify_tier

logger = logging.getLogger(__name__)

# ── Per-tier ATR multipliers (vol-target = ATR_MULT * atr => risk per trade) ──
_ATR_RISK_MULT: dict[CapitalTier, float] = {
    CapitalTier.NANO:  0.50,
    CapitalTier.MICRO: 0.75,
    CapitalTier.SMALL: 1.00,
    CapitalTier.MID:   1.25,
    CapitalTier.LARGE: 1.50,
}

# ── Per-tier Kelly fraction caps ───────────────────────────────────────────────
_KELLY_CAP: dict[CapitalTier, float] = {
    CapitalTier.NANO:  0.04,
    CapitalTier.MICRO: 0.06,
    CapitalTier.SMALL: 0.08,
    CapitalTier.MID:   0.12,
    CapitalTier.LARGE: 0.20,
}

# ── Per-tier portfolio heat limits ─────────────────────────────────────────────
_HEAT_LIMIT: dict[CapitalTier, float] = {
    CapitalTier.NANO:  0.06,
    CapitalTier.MICRO: 0.10,
    CapitalTier.SMALL: 0.15,
    CapitalTier.MID:   0.20,
    CapitalTier.LARGE: 0.25,
}

# Fractional Kelly multiplier (half-Kelly is standard for crypto)
_KELLY_FRACTION: float = 0.5


@dataclass
class SizeResult:
    size_pct:  float
    size_usd:  float
    method:    str
    tier:      CapitalTier
    blocked:   bool = False
    block_reason: str = ""


class PositionSizer:
    """
    ATR-vol targeting + fractional Kelly + portfolio heat gate,
    tier-aware for all five CapitalTier levels.
    """

    def __init__(self, config=None, aud_to_usd: float = 0.65) -> None:
        self.config      = config
        self.aud_to_usd  = aud_to_usd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def size(
        self,
        equity_aud:      float,
        atr:             float,           # ATR / mid_price  (dimensionless)
        win_rate:        float    = 0.50,
        avg_win:         float    = 0.0,
        avg_loss:        float    = 0.0,
        open_heat:       float    = 0.0,
        kelly_fraction:  Optional[float] = None,
        min_notional_usd: float   = 1.0,
    ) -> SizeResult:
        tier = classify_tier(equity_aud)
        equity_usd = equity_aud * self.aud_to_usd

        # ── 1. ATR vol-target sizing ───────────────────────────────────
        atr_mult  = _ATR_RISK_MULT[tier]
        risk_frac = atr_mult * max(atr, 1e-6)   # risk fraction of equity
        atr_size  = min(risk_frac, _KELLY_CAP[tier])

        # ── 2. Fractional Kelly ────────────────────────────────────────
        if kelly_fraction is not None:
            raw_kelly = float(kelly_fraction)
        else:
            raw_kelly = self._kelly(win_rate, avg_win, avg_loss)

        frac_kelly = raw_kelly * _KELLY_FRACTION
        kelly_cap  = _KELLY_CAP[tier]
        kelly_size = min(frac_kelly, kelly_cap)

        # ── 3. Blend: geometric mean of ATR & Kelly (both must agree) ─
        if atr_size > 0 and kelly_size > 0:
            size_pct = math.sqrt(atr_size * kelly_size)
            method   = (f"atr({atr_size:.4f})*kelly({kelly_size:.4f})=>blend"
                        f"({size_pct:.4f}) tier={tier.value}")
        elif atr_size > 0:
            size_pct = atr_size
            method   = f"atr_only({atr_size:.4f}) tier={tier.value}"
        else:
            size_pct = kelly_size
            method   = f"kelly_only({kelly_size:.4f}) tier={tier.value}"

        size_pct = max(0.0, size_pct)

        # ── 4. Portfolio heat gate ─────────────────────────────────────
        heat_limit = _HEAT_LIMIT[tier]
        remaining_heat = max(0.0, heat_limit - open_heat)
        if size_pct > remaining_heat:
            size_pct = remaining_heat
            method  += f"+heat_capped({remaining_heat:.4f})"

        size_usd = equity_usd * size_pct

        # ── 5. Min-notional gate ───────────────────────────────────────
        if size_usd < min_notional_usd and min_notional_usd > 0:
            return SizeResult(
                size_pct=0.0, size_usd=0.0,
                method=method + "+BLOCKED:min_notional",
                tier=tier, blocked=True,
                block_reason=f"notional ${size_usd:.2f} < min ${min_notional_usd:.2f}",
            )

        return SizeResult(size_pct=size_pct, size_usd=size_usd,
                          method=method, tier=tier)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Full Kelly fraction; returns 0 if edge is non-positive."""
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - p
        k = (b * p - q) / b
        return max(0.0, k)
