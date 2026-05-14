"""
base_tentacle.py — Tentacle base class + global registry.

Inspired by OctoBot's Tentacle architecture.
Every evaluator, trading mode, or signal source subclasses BaseTentacle
and self-registers in TENTACLE_REGISTRY on import.

Tentacle types
--------------
  TA_EVALUATOR    — technical analysis signal  (-1..1)
  SOCIAL_EVALUATOR — sentiment / social signal (-1..1)
  REAL_TIME_EVALUATOR — live microstructure    (-1..1)
  TRADING_MODE    — position sizing / order logic
  SCRIPTED        — user-defined script tentacle

Evaluation contract
-------------------
  evaluate(candles, **kwargs) -> float in [-1.0, 1.0]
    +1.0 = strong buy
    -1.0 = strong sell
     0.0 = neutral
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)


class TentacleType(str, Enum):
    TA_EVALUATOR       = "TA_EVALUATOR"
    SOCIAL_EVALUATOR   = "SOCIAL_EVALUATOR"
    REAL_TIME_EVALUATOR= "REAL_TIME_EVALUATOR"
    TRADING_MODE       = "TRADING_MODE"
    SCRIPTED           = "SCRIPTED"


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

TENTACLE_REGISTRY: Dict[str, Type["BaseTentacle"]] = {}


def register_tentacle(cls: Type["BaseTentacle"]) -> Type["BaseTentacle"]:
    """Class decorator — auto-registers a tentacle in TENTACLE_REGISTRY."""
    TENTACLE_REGISTRY[cls.name] = cls
    logger.debug("Registered tentacle: %s (%s)", cls.name, cls.tentacle_type.value)
    return cls


def get_tentacle(name: str) -> Optional[Type["BaseTentacle"]]:
    return TENTACLE_REGISTRY.get(name)


def list_tentacles(tentacle_type: Optional[TentacleType] = None) -> List[str]:
    if tentacle_type is None:
        return list(TENTACLE_REGISTRY.keys())
    return [
        n for n, cls in TENTACLE_REGISTRY.items()
        if cls.tentacle_type == tentacle_type
    ]


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    tentacle_name: str
    signal: float                  # [-1.0, 1.0]
    confidence: float = 1.0        # [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)

    @property
    def weighted_signal(self) -> float:
        return self.signal * self.confidence


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseTentacle(ABC):
    """
    Abstract base for all Argus tentacles.

    Class attributes (must be defined on subclass)
    -----------------------------------------------
    name           : str  — unique tentacle identifier
    tentacle_type  : TentacleType
    version        : str  — semver string
    weight         : float — contribution weight in matrix evaluator (default 1.0)
    """

    name: str = "base_tentacle"
    tentacle_type: TentacleType = TentacleType.TA_EVALUATOR
    version: str = "1.0.0"
    weight: float = 1.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._last_result: Optional[EvalResult] = None
        self._call_count: int = 0

    @abstractmethod
    def evaluate(self, candles: np.ndarray, **kwargs: Any) -> EvalResult:
        """
        Evaluate the tentacle on a candle array.

        Parameters
        ----------
        candles : np.ndarray shape (N, 6)
                  columns: [timestamp, open, high, low, close, volume]

        Returns
        -------
        EvalResult with signal in [-1.0, 1.0]
        """

    def safe_evaluate(self, candles: np.ndarray, **kwargs: Any) -> EvalResult:
        """Wrapper that catches exceptions and returns neutral on failure."""
        try:
            result = self.evaluate(candles, **kwargs)
            self._last_result = result
            self._call_count += 1
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] evaluate() error: %s", self.name, exc)
            return EvalResult(
                tentacle_name=self.name,
                signal=0.0,
                confidence=0.0,
                metadata={"error": str(exc)},
            )

    @property
    def last_result(self) -> Optional[EvalResult]:
        return self._last_result

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self) -> None:
        """Reset internal state between backtests or symbol changes."""
        self._last_result = None
        self._call_count = 0

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} v{self.version}>"


# ---------------------------------------------------------------------------
# Candle helpers (shared utilities for all TA tentacles)
# ---------------------------------------------------------------------------

def candles_close(candles: np.ndarray) -> np.ndarray:
    """Extract close prices from candle array (col index 4)."""
    return candles[:, 4].astype(np.float64)


def candles_volume(candles: np.ndarray) -> np.ndarray:
    return candles[:, 5].astype(np.float64)


def candles_high(candles: np.ndarray) -> np.ndarray:
    return candles[:, 2].astype(np.float64)


def candles_low(candles: np.ndarray) -> np.ndarray:
    return candles[:, 3].astype(np.float64)


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    out = np.zeros_like(prices, dtype=np.float64)
    if len(prices) == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = prices[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.convolve(gain, np.ones(period) / period, mode="same")
    avg_l = np.convolve(loss, np.ones(period) / period, mode="same")
    rs = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-10))
    return 100.0 - (100.0 / (1 + rs))
