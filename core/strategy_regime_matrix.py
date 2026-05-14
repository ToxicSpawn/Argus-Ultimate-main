"""
StrategyRegimeMatrix — Regime-adaptive strategy fitness multipliers.

Builds a rolling exponential-averaged matrix:
  fitness[regime][strategy] = float (0.25–1.5; neutral = 1.0)

Updated from PerformanceAttribution.compute_by_regime() data.
The EMA decay (default 0.99) means old data fades over ~100 updates
so the matrix reflects recent conditions without abrupt jumps.

Fitness formula:
  sharpe_to_fitness(s) = clamp(1.0 + tanh(s), cap_min, cap_max)
  Examples:
    s = 0.0  → 1.0  (neutral)
    s = 1.0  → 1.76 → capped at 1.5
    s = -1.0 → 0.24 → capped at 0.25

Output: advisory["strategy_regime_matrix"]
Updated every 50 cycles in on_cycle() to amortise SQLite I/O.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _clamp(v: float, lo: float = 0.25, hi: float = 1.5) -> float:
    return max(lo, min(hi, v))


def _sharpe_to_fitness(sharpe: float, cap_min: float, cap_max: float) -> float:
    """Map sharpe-like score to fitness multiplier via tanh."""
    return _clamp(1.0 + math.tanh(float(sharpe or 0.0)), cap_min, cap_max)


class StrategyRegimeMatrix:
    """
    EMA-decayed regime×strategy fitness matrix.

    Parameters
    ----------
    decay                : EMA decay per update (0 = instant, 0.99 = slow fade)
    min_trades_for_fitness: below this trade count, return neutral 1.0
    fitness_cap          : (min_multiplier, max_multiplier) for returned fitness
    config               : optional config object for threshold overrides
    """

    def __init__(
        self,
        decay: float = 0.99,
        min_trades_for_fitness: int = 5,
        fitness_cap: Tuple[float, float] = (0.25, 1.5),
        config: Optional[Any] = None,
    ) -> None:
        self.decay = float(max(0.0, min(0.9999, decay)))
        self.min_trades_for_fitness = max(1, int(min_trades_for_fitness))
        self.cap_min, self.cap_max = float(fitness_cap[0]), float(fitness_cap[1])

        # _matrix[regime][strategy] = fitness float
        self._matrix: Dict[str, Dict[str, float]] = {}
        # _trade_counts[regime][strategy] = int
        self._trade_counts: Dict[str, Dict[str, int]] = {}
        self._last_update_ts: float = 0.0
        self._update_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, perf_attribution: Any) -> None:
        """
        Update matrix from a PerformanceAttribution instance.

        Calls perf_attribution.compute_by_regime() which returns a dict like:
          {regime: {strategy_name: AttributionBucket or float}}
        """
        try:
            by_regime = perf_attribution.compute_by_regime()
        except Exception as exc:
            logger.debug("StrategyRegimeMatrix.update: compute_by_regime failed: %s", exc)
            return

        if not by_regime or not isinstance(by_regime, dict):
            return

        # Extract sharpe-like scores per regime × strategy
        raw: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, Dict[str, int]] = {}
        for regime, strategies in by_regime.items():
            if not isinstance(strategies, dict):
                continue
            raw[regime] = {}
            counts[regime] = {}
            for strat, bucket in strategies.items():
                if bucket is None:
                    continue
                # AttributionBucket has .sharpe_like; raw dicts may have "sharpe" key
                sharpe = 0.0
                n = 0
                if hasattr(bucket, "sharpe_like"):
                    sharpe = float(bucket.sharpe_like or 0.0)
                    n = int(getattr(bucket, "count", 0) or 0)
                elif isinstance(bucket, (int, float)):
                    sharpe = float(bucket)
                    n = self.min_trades_for_fitness  # assume enough trades
                elif isinstance(bucket, dict):
                    sharpe = float(bucket.get("sharpe_like", bucket.get("sharpe", 0.0)) or 0.0)
                    n = int(bucket.get("count", bucket.get("n_trades", 0)) or 0)
                raw[regime][strat] = sharpe
                counts[regime][strat] = n

        self.update_from_raw(raw, counts)

    def update_from_dict(
        self,
        regime_strategy_pnl: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Update from a plain {regime: {strategy: sharpe_like}} dict.
        Used for unit testing without a real PerformanceAttribution instance.
        """
        if not regime_strategy_pnl:
            return
        # Treat all as having sufficient trades
        counts: Dict[str, Dict[str, int]] = {
            regime: {s: self.min_trades_for_fitness for s in strategies}
            for regime, strategies in regime_strategy_pnl.items()
        }
        self.update_from_raw(regime_strategy_pnl, counts)

    def update_from_raw(
        self,
        raw: Dict[str, Dict[str, float]],
        counts: Dict[str, Dict[str, int]],
    ) -> None:
        """
        Apply EMA update from raw sharpe scores.

        Internal helper used by both update() and update_from_dict().
        """
        for regime, strategies in raw.items():
            if regime not in self._matrix:
                self._matrix[regime] = {}
                self._trade_counts[regime] = {}

            for strat, sharpe in strategies.items():
                n = counts.get(regime, {}).get(strat, 0)
                if n < self.min_trades_for_fitness:
                    # Not enough data — don't update, keep neutral
                    if strat not in self._matrix[regime]:
                        self._matrix[regime][strat] = 1.0
                    continue

                new_fitness = _sharpe_to_fitness(sharpe, self.cap_min, self.cap_max)
                old_fitness = self._matrix[regime].get(strat, 1.0)
                # EMA update
                updated = self.decay * old_fitness + (1.0 - self.decay) * new_fitness
                self._matrix[regime][strat] = _clamp(updated, self.cap_min, self.cap_max)
                self._trade_counts[regime][strat] = n

        self._last_update_ts = time.time()
        self._update_count += 1

    def get_fitness(self, strategy_name: str, regime: str) -> float:
        """
        Return fitness multiplier for a strategy in a given regime.
        Returns 1.0 (neutral) if no data available.
        """
        return float(
            self._matrix.get(str(regime), {}).get(str(strategy_name), 1.0)
        )

    def get_regime_weights(self, current_regime: str) -> Dict[str, float]:
        """
        Return {strategy_name: multiplier} for all tracked strategies in current_regime.
        Returns {} if the regime is unknown.
        """
        regime_data = self._matrix.get(str(current_regime))
        if not regime_data:
            return {}
        return {strat: round(fitness, 4) for strat, fitness in regime_data.items()}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "matrix": {
                regime: dict(strategies)
                for regime, strategies in self._matrix.items()
            },
            "update_count": self._update_count,
            "last_update_ts": self._last_update_ts,
            "decay": self.decay,
            "fitness_cap": (self.cap_min, self.cap_max),
        }
