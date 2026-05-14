"""
Competence estimator: "Am I good at this right now?"

Aggregates:
- Rolling per-regime Sharpe ratio (from trade ledger)
- Confidence calibration error (are my confidences well-calibrated?)
- Decision journal accuracy (did past decisions pan out?)
- Regime-specific win rate

Into a single 0-1 competence score that the trading loop uses to gate
position sizing. Low score → size down (we don't trust our own model).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CompetenceEstimator
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class _FillRecord:
    timestamp: float
    symbol: str
    regime: str
    strategy: str
    pnl: float
    confidence: float
    won: bool


class CompetenceEstimator:
    """
    Tracks recent performance across (regime, symbol, strategy) dimensions
    and produces a competence score in [0, 1].

    Parameters
    ----------
    window : int
        Rolling window size (number of recent fills to consider).
    min_fills : int
        Minimum number of fills before a confident competence score is
        produced; below this, the score defaults to 0.5 (neutral).
    """

    def __init__(self, window: int = 100, min_fills: int = 10) -> None:
        self.window = int(window)
        self.min_fills = int(min_fills)
        self._fills: Deque[_FillRecord] = deque(maxlen=window)

    def record_fill(
        self,
        *,
        symbol: str,
        regime: str,
        strategy: str,
        pnl: float,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a completed trade's outcome."""
        import time as _time
        self._fills.append(_FillRecord(
            timestamp=timestamp or _time.time(),
            symbol=str(symbol),
            regime=str(regime),
            strategy=str(strategy),
            pnl=float(pnl),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            won=pnl > 0,
        ))

    def competence(
        self,
        *,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> float:
        """
        Return a competence score in [0, 1] for the given filter.

        Score components (each in [0, 1]):
        - win_rate: fraction of winning trades
        - sharpe_score: mapped Sharpe ratio (1.0 at Sharpe=2.0, 0.5 at 0)
        - calibration: 1 - mean absolute calibration error
        - consistency: 1 - normalized std of P&L

        Averaged with equal weight, clamped to [0, 1].
        """
        relevant = [
            f for f in self._fills
            if (symbol is None or f.symbol == symbol)
            and (regime is None or f.regime == regime)
            and (strategy is None or f.strategy == strategy)
        ]

        if len(relevant) < self.min_fills:
            return 0.5  # neutral

        pnls = np.array([f.pnl for f in relevant], dtype=float)
        confidences = np.array([f.confidence for f in relevant], dtype=float)
        won = np.array([f.won for f in relevant], dtype=float)

        # 1. Win rate [0, 1]
        win_rate = float(np.mean(won))

        # 2. Sharpe score: mapped to [0, 1] with sigmoid(sharpe - 1)
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls)) or 1e-6
        sharpe = mean_pnl / std_pnl
        sharpe_score = float(1.0 / (1.0 + math.exp(-(sharpe - 1.0))))

        # 3. Calibration: 1 - mean |confidence - won|
        calibration = float(1.0 - np.mean(np.abs(confidences - won)))
        calibration = max(0.0, min(1.0, calibration))

        # 4. Consistency: inverse of normalized std
        normalized_std = float(std_pnl / (abs(mean_pnl) + 1e-6))
        consistency = float(1.0 / (1.0 + normalized_std))

        # Weighted average
        score = 0.30 * win_rate + 0.35 * sharpe_score + 0.20 * calibration + 0.15 * consistency
        return float(np.clip(score, 0.0, 1.0))

    def n_fills(self) -> int:
        return len(self._fills)

    def snapshot(self) -> Dict[str, Any]:
        overall = self.competence()
        by_regime: Dict[str, float] = {}
        regimes = set(f.regime for f in self._fills)
        for r in regimes:
            by_regime[r] = self.competence(regime=r)
        return {
            "overall_competence": overall,
            "n_fills": self.n_fills(),
            "by_regime": by_regime,
            "window": self.window,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Stateless helper for one-shot scoring
# ═════════════════════════════════════════════════════════════════════════════


def compute_competence_score(
    win_rate: float,
    sharpe: float,
    calibration_error: float,
    pnl_std_normalized: float,
) -> float:
    """
    Stateless version that takes pre-computed stats.

    Useful for external callers that already have their own rolling stats.
    """
    win_rate = float(np.clip(win_rate, 0.0, 1.0))
    sharpe_score = float(1.0 / (1.0 + math.exp(-(sharpe - 1.0))))
    calibration = float(np.clip(1.0 - calibration_error, 0.0, 1.0))
    consistency = float(1.0 / (1.0 + max(pnl_std_normalized, 0.0)))
    return float(np.clip(
        0.30 * win_rate + 0.35 * sharpe_score + 0.20 * calibration + 0.15 * consistency,
        0.0, 1.0,
    ))
