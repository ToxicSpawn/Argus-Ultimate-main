"""Real-time volatility estimation for Avellaneda-Stoikov quoting."""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

logger = logging.getLogger(__name__)


class VolatilityRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass(slots=True)
class VolatilitySnapshot:
    realized_volatility: float
    ewma_volatility: float
    selected_volatility: float
    regime: VolatilityRegime
    spread_multiplier: float
    sample_count: int
    timestamp: float


class VolatilityEstimator:
    """Tracks realized and EWMA volatility from tick-level mid prices."""

    def __init__(self, window_size: int = 120, ewma_lambda: float = 0.94):
        if window_size < 3:
            raise ValueError("window_size must be >= 3")
        if not 0.0 < ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be in (0, 1)")

        self.window_size = window_size
        self.ewma_lambda = ewma_lambda
        self._prices: Deque[float] = deque(maxlen=window_size)
        self._timestamps: Deque[float] = deque(maxlen=window_size)
        self._ewma_variance: float = 0.0

    def update(self, price: float, timestamp: Optional[float] = None) -> VolatilitySnapshot:
        if price <= 0:
            raise ValueError("price must be positive")

        timestamp = float(timestamp if timestamp is not None else time.time())
        if self._prices:
            last_price = self._prices[-1]
            ret = math.log(price / last_price)
            realized_step_var = ret * ret
            self._ewma_variance = (
                self.ewma_lambda * self._ewma_variance
                + (1.0 - self.ewma_lambda) * realized_step_var
            )

        self._prices.append(price)
        self._timestamps.append(timestamp)
        return self.snapshot()

    def _log_returns(self) -> list[float]:
        prices = list(self._prices)
        if len(prices) < 2:
            return []
        returns: list[float] = []
        for previous, current in zip(prices[:-1], prices[1:]):
            if previous > 0 and current > 0:
                returns.append(math.log(current / previous))
        return returns

    def realized_volatility(self) -> float:
        returns = self._log_returns()
        if len(returns) < 2:
            return 0.0
        return statistics.pstdev(returns)

    def ewma_volatility(self) -> float:
        return math.sqrt(max(self._ewma_variance, 0.0))

    def detect_regime(self, selected_volatility: float) -> VolatilityRegime:
        if selected_volatility < 0.0005:
            return VolatilityRegime.LOW
        if selected_volatility < 0.0020:
            return VolatilityRegime.NORMAL
        if selected_volatility < 0.0050:
            return VolatilityRegime.HIGH
        return VolatilityRegime.EXTREME

    def spread_multiplier(self, regime: VolatilityRegime) -> float:
        return {
            VolatilityRegime.LOW: 0.85,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 1.25,
            VolatilityRegime.EXTREME: 1.6,
        }[regime]

    def snapshot(self) -> VolatilitySnapshot:
        realized = self.realized_volatility()
        ewma = self.ewma_volatility()
        selected = max(realized, ewma)
        regime = self.detect_regime(selected)
        return VolatilitySnapshot(
            realized_volatility=realized,
            ewma_volatility=ewma,
            selected_volatility=selected,
            regime=regime,
            spread_multiplier=self.spread_multiplier(regime),
            sample_count=len(self._prices),
            timestamp=self._timestamps[-1] if self._timestamps else time.time(),
        )
