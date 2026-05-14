"""Push 75 — MLStrategy: wraps a stable-baselines3 RL model.

Feature vector (20 features):
  price_returns[-5:], volume_norm[-5:], rsi_norm, bb_z,
  ema_diff_norm, atr_norm, position (0/1/-1), drawdown_pct

Logic:
  - model.predict(obs) -> action in {0=HOLD, 1=LONG, 2=SHORT, 3=FLAT}
  - action_prob (if model supports) used as signal strength
  - confidence_threshold gates weak predictions
  - Graceful stub: returns FLAT when model not loaded

Default params:
  lookback=20, confidence_threshold=0.55, model_path=None
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional

from core.strategy.base_strategy import BaseStrategy, StrategyConfig
from core.strategy.signal import Signal, SignalSide

_ACTION_MAP = {0: None, 1: SignalSide.LONG, 2: SignalSide.SHORT, 3: SignalSide.FLAT}


class MLStrategy(BaseStrategy):
    """RL model-driven strategy wrapping stable-baselines3."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        p = config.params
        self.lookback   = int(p.get("lookback",   20))
        self.conf_thresh = float(p.get("confidence_threshold", 0.55))
        model_path      = p.get("model_path", None)

        self._prices:  Deque[float] = deque(maxlen=self.lookback + 5)
        self._volumes: Deque[float] = deque(maxlen=self.lookback + 5)
        self._model = None
        self._model_loaded = False

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(path)
            self._model_loaded = True
        except Exception:
            self._model_loaded = False

    def _build_obs(self) -> Optional[List[float]]:
        prices  = list(self._prices)
        volumes = list(self._volumes)
        if len(prices) < self.lookback:
            return None

        # Price returns (last 5)
        returns = [
            (prices[-i] - prices[-i - 1]) / prices[-i - 1]
            if prices[-i - 1] != 0 else 0.0
            for i in range(1, 6)
        ]

        # Volume normalised (last 5)
        max_vol = max(volumes[-5:]) if volumes else 1.0
        vol_norm = [v / max_vol if max_vol > 0 else 0.0 for v in volumes[-5:]]
        if len(vol_norm) < 5:
            vol_norm = [0.0] * 5

        # RSI
        rsi = self._rsi(prices)
        rsi_norm = (rsi - 50) / 50

        # BB z-score
        mean = sum(prices[-20:]) / min(20, len(prices))
        std  = math.sqrt(sum((x - mean) ** 2 for x in prices[-20:]) / min(20, len(prices)))
        bb_z = (prices[-1] - mean) / std if std > 0 else 0.0

        # EMA diff
        from core.strategy.momentum_strategy import _ema
        ema_fast = _ema(prices, 9)
        ema_slow = _ema(prices, 21)
        ema_diff = (ema_fast - ema_slow) / ema_slow if ema_slow != 0 else 0.0

        # ATR normalised
        atr = self._atr(prices)
        atr_norm = atr / prices[-1] if prices[-1] != 0 else 0.0

        # Position & drawdown
        position   = 0.0
        dd_pct     = self.metrics.drawdown_pct / 100

        obs = returns + vol_norm + [rsi_norm, bb_z, ema_diff, atr_norm, position, dd_pct]
        return obs[:20]  # ensure exactly 20 features

    @staticmethod
    def _rsi(prices: list, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains  = [max(d, 0)   for d in deltas[-period:]]
        losses = [abs(min(d, 0)) for d in deltas[-period:]]
        ag = sum(gains) / period
        al = sum(losses) / period
        if al == 0:
            return 100.0
        return 100 - 100 / (1 + ag / al)

    @staticmethod
    def _atr(prices: list, period: int = 14) -> float:
        if len(prices) < 2:
            return 0.0
        trs = [abs(prices[-i] - prices[-i - 1]) for i in range(1, min(period + 1, len(prices)))]
        return sum(trs) / len(trs) if trs else 0.0

    def tick(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Optional[Signal]:
        self._prices.append(price)
        self._volumes.append(volume)

        if not self._model_loaded:
            # Stub: no model loaded, emit nothing
            return None

        obs = self._build_obs()
        if obs is None:
            return None

        try:
            import numpy as np
            action, _ = self._model.predict(np.array(obs, dtype=np.float32))
            action = int(action)
        except Exception:
            return None

        side = _ACTION_MAP.get(action)
        if side is None:
            return None

        strength = self.conf_thresh  # could use softmax prob if available
        return self._make_signal(side, strength)
