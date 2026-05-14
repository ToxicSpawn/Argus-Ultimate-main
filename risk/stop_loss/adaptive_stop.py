"""Adaptive Stop Loss - adjusts based on recent volatility and trade performance."""
from __future__ import annotations
import numpy as np
from collections import deque


class AdaptiveStop:
    """Stop that adapts to current volatility regime and recent win rate."""
    def __init__(self, base_atr_mult: float = 2.0, vol_scale_factor: float = 1.5, win_rate_window: int = 20, min_mult: float = 1.0, max_mult: float = 4.0):
        self.base_atr_mult = base_atr_mult
        self.vol_scale_factor = vol_scale_factor
        self.min_mult = min_mult
        self.max_mult = max_mult
        self._trade_results = deque(maxlen=win_rate_window)
        self._vol_history = deque(maxlen=100)

    def record_trade(self, pnl: float) -> None:
        self._trade_results.append(1.0 if pnl > 0 else 0.0)

    def _current_win_rate(self) -> float:
        if not self._trade_results:
            return 0.5
        return float(np.mean(list(self._trade_results)))

    def _vol_regime_multiplier(self, current_vol: float) -> float:
        self._vol_history.append(current_vol)
        if len(self._vol_history) < 10:
            return 1.0
        median_vol = float(np.median(list(self._vol_history)))
        if median_vol <= 0:
            return 1.0
        ratio = current_vol / median_vol
        return max(0.5, min(2.0, ratio ** self.vol_scale_factor))

    def calculate(self, entry_price: float, side: str, current_atr: float, current_vol: float = 0.0, regime_widen: float = 1.0) -> dict:
        win_rate = self._current_win_rate()
        wr_adj = 1.0 + (0.5 - win_rate) * 0.5  # Widen stops when losing more
        vol_adj = self._vol_regime_multiplier(current_vol) if current_vol > 0 else 1.0
        effective_mult = max(self.min_mult, min(self.max_mult, self.base_atr_mult * wr_adj * vol_adj * regime_widen))
        stop_dist = effective_mult * current_atr
        if side == "long":
            stop_price = entry_price - stop_dist
        else:
            stop_price = entry_price + stop_dist
        return {"stop_price": stop_price, "effective_multiplier": effective_mult, "win_rate": win_rate, "vol_adjustment": vol_adj, "method": "adaptive_stop"}
