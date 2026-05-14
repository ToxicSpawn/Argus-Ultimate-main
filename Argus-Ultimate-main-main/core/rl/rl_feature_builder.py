"""Push 66 — 7-dimensional state vector builder for ArgusRLEnv.

Dimensions:
  0  price_return     — bar close/open - 1
  1  rsi_norm         — RSI(14) / 100
  2  funding_rate     — 8h funding rate (normalised)
  3  inventory        — current position [-1, +1]
  4  pnl_norm         — unrealised PnL / initial_equity
  5  regime           — HMM regime label {-1, 0, +1} / normalised
  6  vol_proxy        — (high-low)/open — intrabar volatility
"""
from __future__ import annotations
import numpy as np
from typing import Protocol, runtime_checkable


@runtime_checkable
class BarLike(Protocol):
    open: float
    high: float
    low: float
    close: float


class FeatureBuilder:
    """Stateful feature builder — call update() each bar, read obs property."""

    def __init__(self, rsi_period: int = 14):
        self.rsi_period = rsi_period
        self._closes: list[float] = []
        self.funding_rate: float = 0.0
        self.regime: float = 0.0      # set externally by HMM

    def update(self, bar: BarLike) -> None:
        self._closes.append(bar.close)
        if len(self._closes) > self.rsi_period + 2:
            self._closes = self._closes[-(self.rsi_period + 2):]

    def build(self, bar: BarLike, inventory: float,
              pnl_norm: float) -> np.ndarray:
        price_return = bar.close / bar.open - 1.0
        rsi = self._rsi()
        vol_proxy = (bar.high - bar.low) / bar.open if bar.open > 0 else 0.0
        return np.array([
            float(np.clip(price_return, -0.1, 0.1)),
            rsi / 100.0,
            float(np.clip(self.funding_rate, -0.01, 0.01)),
            float(np.clip(inventory, -1.0, 1.0)),
            float(np.clip(pnl_norm, -1.0, 1.0)),
            float(self.regime),
            float(np.clip(vol_proxy, 0.0, 0.1)),
        ], dtype=np.float32)

    def _rsi(self) -> float:
        if len(self._closes) < self.rsi_period + 1:
            return 50.0
        closes = np.array(self._closes[-(self.rsi_period + 1):])
        deltas = np.diff(closes)
        gains = np.maximum(deltas, 0.0)
        losses = np.maximum(-deltas, 0.0)
        avg_gain = gains.mean() + 1e-9
        avg_loss = losses.mean() + 1e-9
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))
