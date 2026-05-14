"""
Kelly Criterion Position Sizing - Real Implementation
Tracks win rate, average win/loss, and computes optimal fraction with edge decay.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Any

import numpy as np


class KellyCriterionSizer:
    """Full Kelly criterion with Bayesian edge estimation and fractional safety."""

    def __init__(
        self,
        fraction: float = 0.25,
        min_trades: int = 10,
        lookback: int = 100,
        max_position_pct: float = 0.15,
        edge_decay_halflife: int = 50,
    ):
        self.fraction = float(fraction)
        self.min_trades = int(min_trades)
        self.max_position_pct = float(max_position_pct)
        self.edge_decay_halflife = int(edge_decay_halflife)
        self._wins: deque = deque(maxlen=lookback)
        self._losses: deque = deque(maxlen=lookback)
        self._trade_pnls: deque = deque(maxlen=lookback)

    def record_trade(self, pnl: float) -> None:
        self._trade_pnls.append(float(pnl))
        if pnl > 0:
            self._wins.append(float(pnl))
        elif pnl < 0:
            self._losses.append(abs(float(pnl)))

    def _compute_kelly_fraction(self) -> float:
        if len(self._wins) < 3 or len(self._losses) < 3:
            return 0.0
        total = len(self._wins) + len(self._losses)
        if total < self.min_trades:
            return 0.0
        win_rate = len(self._wins) / total
        avg_win = float(np.mean(list(self._wins)))
        avg_loss = float(np.mean(list(self._losses)))
        if avg_loss <= 0:
            return 0.0
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / max(win_loss_ratio, 1e-9)
        if len(self._trade_pnls) >= 10:
            decay = 0.5 ** (1.0 / max(self.edge_decay_halflife, 1))
            weights = np.array([decay ** i for i in range(len(self._trade_pnls) - 1, -1, -1)])
            weighted_pnls = np.array(list(self._trade_pnls))
            weighted_edge = float(np.sum(weights * weighted_pnls) / np.sum(weights))
            if weighted_edge < 0:
                kelly *= 0.5
        return max(0.0, kelly * self.fraction)

    def calculate(self, capital: float, risk_per_trade: float, confidence: float = 1.0) -> Dict[str, Any]:
        cap = max(float(capital), 1.0)
        kelly_f = self._compute_kelly_fraction()
        if kelly_f > 0:
            position_size = cap * kelly_f * float(confidence)
        else:
            position_size = cap * float(risk_per_trade) * float(confidence)
        max_size = cap * self.max_position_pct
        position_size = min(position_size, max_size)
        total_trades = len(self._wins) + len(self._losses)
        win_rate = len(self._wins) / total_trades if total_trades > 0 else 0.0
        return {
            "position_size": position_size,
            "pct_of_capital": (position_size / cap) * 100,
            "kelly_fraction": kelly_f,
            "full_kelly": kelly_f / max(self.fraction, 1e-9),
            "win_rate": win_rate,
            "total_trades": total_trades,
            "method": "kelly_criterion",
        }
