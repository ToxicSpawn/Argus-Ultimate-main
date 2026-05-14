"""
adaptive/auto_capital_allocator.py --- Automatic Capital Allocation.

Distributes available AUD capital across active strategies using a
Kelly-weighted, regime-aware, correlation-penalised optimiser.

Constraints
-----------
- Minimum allocation per strategy: $50 AUD (below this, fees dominate).
- Maximum concentration: 40% in any one strategy.
- Rebalance only when drift from optimal exceeds a configurable threshold.

Usage::

    alloc = AutoCapitalAllocator(config=cfg_section)
    result = alloc.optimize(strategies, capital_aud=1000.0)
    # result = {"momentum_eth": 300.0, "pairs_btc_eth": 250.0, ...}
    if alloc.rebalance_check(current_allocations, result):
        apply(result)

Standalone --- no hard imports on the rest of the ARGUS tree at module load.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_ALLOC_AUD = 50.0
_MAX_CONCENTRATION_PCT = 0.40
_REBALANCE_DRIFT_THRESHOLD = 0.10   # 10 ppt drift triggers rebalance


# ---------------------------------------------------------------------------
# AutoCapitalAllocator
# ---------------------------------------------------------------------------

class AutoCapitalAllocator:
    """Kelly-weighted, regime-aware capital allocator.

    Parameters
    ----------
    config : dict, optional
        ``auto_capital_allocator`` section from unified config.
    """

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._min_alloc_aud: float = float(cfg.get("min_allocation_aud", _MIN_ALLOC_AUD))
        self._max_concentration: float = float(cfg.get("max_concentration_pct", _MAX_CONCENTRATION_PCT))
        self._rebalance_threshold: float = float(cfg.get("rebalance_drift_threshold", _REBALANCE_DRIFT_THRESHOLD))
        self._kelly_fraction_cap: float = float(cfg.get("kelly_fraction_cap", 0.25))
        self._regime_boost: float = float(cfg.get("regime_boost", 1.3))
        self._regime_penalty: float = float(cfg.get("regime_penalty", 0.5))
        self._last_allocation: Dict[str, float] = {}

        logger.info(
            "AutoCapitalAllocator initialised (min=$%.0f, max_conc=%.0f%%, rebal_thresh=%.0f%%)",
            self._min_alloc_aud, self._max_concentration * 100, self._rebalance_threshold * 100,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        strategies: Dict[str, Dict[str, Any]],
        capital_aud: float,
    ) -> Dict[str, float]:
        """Compute optimal capital allocation.

        Parameters
        ----------
        strategies : dict
            strategy_name -> {
                sharpe, win_rate, avg_win_pct, avg_loss_pct,
                regime_suitable (bool), correlation_group (str, optional),
                is_active (bool, default True)
            }
        capital_aud : float
            Total deployable capital in AUD.

        Returns
        -------
        dict
            strategy_name -> allocation_aud (only strategies meeting minimum).
        """
        if not self._enabled or capital_aud <= 0:
            return {}

        active = {
            name: m for name, m in strategies.items()
            if m.get("is_active", True)
        }
        if not active:
            return {}

        # Step 1: Raw Kelly fraction per strategy
        raw_scores: Dict[str, float] = {}
        for name, m in active.items():
            kelly = self._compute_kelly(m)

            # Regime suitability adjustment
            if m.get("regime_suitable", True):
                kelly *= self._regime_boost
            else:
                kelly *= self._regime_penalty

            # Recent performance scaling (Sharpe as confidence proxy)
            sharpe = float(m.get("sharpe", 0.0))
            perf_mult = max(0.1, min(2.0, 0.5 + sharpe * 0.3))
            kelly *= perf_mult

            raw_scores[name] = max(0.0, kelly)

        # Step 2: Correlation penalty -- reduce overlapping groups
        raw_scores = self._apply_correlation_penalty(raw_scores, active)

        # Step 3: Normalise to sum=1, then apply concentration cap
        total_score = sum(raw_scores.values())
        if total_score <= 0:
            # Equal weight fallback
            n = len(active)
            equal = 1.0 / max(n, 1)
            weights = {name: equal for name in active}
        else:
            weights = {name: s / total_score for name, s in raw_scores.items()}

        # Cap concentration
        weights = self._cap_concentration(weights)

        # Step 4: Convert to AUD and enforce minimum
        allocation: Dict[str, float] = {}
        for name, w in weights.items():
            aud = round(capital_aud * w, 2)
            if aud >= self._min_alloc_aud:
                allocation[name] = aud

        # Redistribute unallocated capital proportionally among survivors
        allocated_total = sum(allocation.values())
        if allocation and allocated_total < capital_aud * 0.95:
            surplus = capital_aud - allocated_total
            alloc_sum = sum(allocation.values())
            if alloc_sum > 0:
                for name in allocation:
                    allocation[name] += round(surplus * (allocation[name] / alloc_sum), 2)

        self._last_allocation = dict(allocation)

        logger.info(
            "AutoCapitalAllocator: $%.2f across %d strategies (of %d active)",
            sum(allocation.values()), len(allocation), len(active),
        )
        return allocation

    def rebalance_check(
        self,
        current: Optional[Dict[str, float]] = None,
        optimal: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Return True if the current allocation has drifted enough to warrant rebalancing.

        Parameters
        ----------
        current : dict, optional
            Current allocations (strategy -> AUD). Falls back to last_allocation.
        optimal : dict, optional
            Optimal allocations to compare against.
        """
        curr = current or self._last_allocation
        opt = optimal or {}
        if not curr or not opt:
            return False

        drift = self.get_allocation_drift(curr, opt)
        max_drift = max(abs(v) for v in drift.values()) if drift else 0.0
        should = max_drift > self._rebalance_threshold

        if should:
            logger.info(
                "Rebalance recommended: max drift %.1f%% > threshold %.1f%%",
                max_drift * 100, self._rebalance_threshold * 100,
            )
        return should

    def get_allocation_drift(
        self,
        current: Optional[Dict[str, float]] = None,
        optimal: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Return per-strategy drift from optimal as a fraction.

        Returns
        -------
        dict
            strategy_name -> drift_fraction (positive = overweight, negative = underweight).
        """
        curr = current or self._last_allocation
        opt = optimal or self._last_allocation
        if not curr and not opt:
            return {}

        all_names = set(curr) | set(opt)
        total_curr = sum(curr.values()) or 1.0
        total_opt = sum(opt.values()) or 1.0

        drift: Dict[str, float] = {}
        for name in all_names:
            c_pct = curr.get(name, 0.0) / total_curr
            o_pct = opt.get(name, 0.0) / total_opt
            drift[name] = round(c_pct - o_pct, 4)

        return drift

    @property
    def last_allocation(self) -> Dict[str, float]:
        return dict(self._last_allocation)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_kelly(self, m: Dict[str, Any]) -> float:
        """Compute Kelly fraction for a strategy."""
        win_rate = float(m.get("win_rate", 0.5))
        avg_win = float(m.get("avg_win_pct", 1.0))
        avg_loss = float(m.get("avg_loss_pct", 1.0))

        if avg_loss <= 0 or win_rate <= 0:
            return 0.0

        b = avg_win / max(avg_loss, 0.001)
        kelly = (win_rate * b - (1.0 - win_rate)) / max(b, 0.001)
        return max(0.0, min(kelly, self._kelly_fraction_cap))

    def _apply_correlation_penalty(
        self,
        scores: Dict[str, float],
        active: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Reduce scores for strategies in the same correlation group."""
        groups: Dict[str, List[str]] = {}
        for name, m in active.items():
            grp = str(m.get("correlation_group", name))
            groups.setdefault(grp, []).append(name)

        adjusted = dict(scores)
        for grp, members in groups.items():
            if len(members) <= 1:
                continue
            # Penalise all but the strongest in the group
            ranked = sorted(members, key=lambda n: scores.get(n, 0), reverse=True)
            for i, name in enumerate(ranked):
                if i > 0:
                    adjusted[name] = adjusted.get(name, 0) * (0.5 ** i)

        return adjusted

    def _cap_concentration(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Iteratively cap any weight above max_concentration and redistribute."""
        result = dict(weights)
        for _ in range(10):  # max iterations
            excess = 0.0
            uncapped = []
            for name, w in result.items():
                if w > self._max_concentration:
                    excess += w - self._max_concentration
                    result[name] = self._max_concentration
                else:
                    uncapped.append(name)

            if excess <= 0.001 or not uncapped:
                break

            uncapped_total = sum(result[n] for n in uncapped)
            if uncapped_total > 0:
                for n in uncapped:
                    result[n] += excess * (result[n] / uncapped_total)

        # Final normalisation
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result
