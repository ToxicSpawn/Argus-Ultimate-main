"""Push 66 — Kelly Criterion dynamic position sizer.

Implements fractional Kelly for mathematically optimal geometric
growth rate maximisation. Default fraction=0.25 (quarter-Kelly)
for conservative crypto risk management.

Reference: Kelly (1956), López de Prado (2018 AFML)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class KellyResult:
    kelly_fraction: float      # raw Kelly fraction
    safe_fraction: float       # kelly_fraction * safety_multiplier
    position_usd: float        # recommended position size in USD
    edge: float                # expected edge per trade
    odds: float                # avg_win / avg_loss ratio


class KellySizer:
    """Fractional Kelly position sizer.

    Args:
        safety_multiplier: Fraction of full Kelly to use (0.25 = quarter-Kelly)
        max_fraction: Hard cap on fraction of equity per trade
        min_fraction: Minimum fraction (no sub-threshold trades)
    """

    def __init__(
        self,
        safety_multiplier: float = 0.25,
        max_fraction: float = 0.20,
        min_fraction: float = 0.01,
    ):
        assert 0.0 < safety_multiplier <= 1.0
        assert 0.0 < max_fraction <= 1.0
        self.safety_multiplier = safety_multiplier
        self.max_fraction = max_fraction
        self.min_fraction = min_fraction

    def size(
        self,
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> KellyResult:
        """Compute Kelly-optimal position size.

        Args:
            equity:    Current account equity in USD
            win_rate:  Historical win rate [0, 1]
            avg_win:   Average winning trade return (positive, e.g. 0.02 = 2%)
            avg_loss:  Average losing trade return (positive magnitude, e.g. 0.01)

        Returns:
            KellyResult with recommended position_usd
        """
        if avg_loss <= 0 or avg_win <= 0:
            return KellyResult(0.0, 0.0, 0.0, 0.0, 0.0)

        b = avg_win / avg_loss        # odds ratio
        q = 1.0 - win_rate
        edge = b * win_rate - q

        if edge <= 0:
            return KellyResult(0.0, 0.0, 0.0, edge, b)

        raw_kelly = edge / b
        safe_kelly = raw_kelly * self.safety_multiplier
        safe_kelly = max(self.min_fraction, min(self.max_fraction, safe_kelly))
        position_usd = equity * safe_kelly

        return KellyResult(
            kelly_fraction=raw_kelly,
            safe_fraction=safe_kelly,
            position_usd=position_usd,
            edge=edge,
            odds=b,
        )

    def size_from_trades(
        self,
        equity: float,
        trade_returns: Sequence[float],
    ) -> KellyResult:
        """Compute Kelly from a sequence of trade returns."""
        if len(trade_returns) < 10:
            return KellyResult(0.0, 0.0, equity * self.min_fraction, 0.0, 0.0)

        wins = [r for r in trade_returns if r > 0]
        losses = [abs(r) for r in trade_returns if r < 0]

        if not wins or not losses:
            return KellyResult(0.0, 0.0, equity * self.min_fraction, 0.0, 0.0)

        win_rate = len(wins) / len(trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        return self.size(equity, win_rate, avg_win, avg_loss)

    @staticmethod
    def atr_size(
        equity: float,
        atr: float,
        price: float,
        risk_pct: float = 0.01,
    ) -> float:
        """Volatility-normalised sizing: risk fixed % of equity per ATR move."""
        if atr <= 0 or price <= 0:
            return 0.0
        atr_pct = atr / price
        return (equity * risk_pct) / atr_pct
