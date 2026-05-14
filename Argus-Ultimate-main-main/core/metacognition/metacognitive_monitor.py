"""
Singleton metacognitive monitor.

Collects competence + capability bounds + recent calibration error into
a single snapshot for the trading loop and dashboards.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from .competence_estimator import CompetenceEstimator
from .capability_bounds import CapabilityBounds


# ═════════════════════════════════════════════════════════════════════════════
# MetacognitiveMonitor
# ═════════════════════════════════════════════════════════════════════════════


class MetacognitiveMonitor:
    """
    Thread-safe singleton that aggregates metacognition signals.

    The trading loop records every fill via ``record_fill()``. The
    ``pre_order_check()`` hook reads the current competence score via
    ``current_competence()``.
    """

    def __init__(self, window: int = 100, min_fills: int = 10) -> None:
        self.competence = CompetenceEstimator(window=window, min_fills=min_fills)
        self.bounds = CapabilityBounds(window=window)
        self._lock = threading.Lock()

    def record_fill(
        self,
        *,
        symbol: str,
        regime: str,
        strategy: str,
        pnl: float,
        confidence: float = 0.5,
    ) -> None:
        """Record a completed trade. Called on every fill."""
        with self._lock:
            self.competence.record_fill(
                symbol=symbol,
                regime=regime,
                strategy=strategy,
                pnl=pnl,
                confidence=confidence,
            )
            self.bounds.record(
                regime=regime,
                symbol=symbol,
                strategy=strategy,
                pnl=pnl,
            )

    def current_competence(
        self,
        *,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> float:
        """
        Current competence score in [0, 1] for the given filter.

        Used by the trading loop: multiply position size by this when
        below 0.5 to reduce exposure when the model is uncertain of its
        own competence.
        """
        with self._lock:
            return self.competence.competence(
                symbol=symbol, regime=regime, strategy=strategy,
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "competence": self.competence.snapshot(),
                "capability_bounds": self.bounds.snapshot(),
            }


# ═════════════════════════════════════════════════════════════════════════════
# Singleton
# ═════════════════════════════════════════════════════════════════════════════


_INSTANCE: Optional[MetacognitiveMonitor] = None


def get_metacognitive_monitor() -> MetacognitiveMonitor:
    """Return the global metacognitive monitor singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MetacognitiveMonitor()
    return _INSTANCE
