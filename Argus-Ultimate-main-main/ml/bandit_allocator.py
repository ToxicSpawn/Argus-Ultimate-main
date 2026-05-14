"""
Multi-Armed Bandit Strategy Allocator for ARGUS.

Dynamically allocates capital to strategies using Thompson Sampling.
Tracks win/loss outcomes per strategy as Beta distribution parameters
and samples to produce capital allocation weights. Includes an
exploration floor (5% minimum per strategy) and automatic disabling
of persistently losing strategies.

Usage:
    allocator = BanditStrategyAllocator(["momentum", "mean_revert", "breakout"])
    allocator.record_outcome("momentum", pnl=12.5)
    allocator.record_outcome("mean_revert", pnl=-3.0)
    allocations = allocator.get_allocations(total_capital=1000.0)
    # {'momentum': 450.0, 'mean_revert': 250.0, 'breakout': 300.0}
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BanditStrategyAllocator
# ---------------------------------------------------------------------------


class BanditStrategyAllocator:
    """
    Thompson Sampling allocator for strategy capital distribution.

    Each strategy is modelled as a Bernoulli bandit with a Beta(alpha, beta)
    posterior.  Positive PnL trades increment alpha (successes); negative PnL
    trades increment beta (failures).  Capital is allocated proportionally to
    samples drawn from each strategy's Beta distribution.

    Parameters
    ----------
    strategy_names : list[str]
        Names of the strategies to track.
    exploration_rate : float
        Minimum allocation fraction per strategy (exploration floor).
        Default 0.05 = 5%.
    """

    def __init__(
        self,
        strategy_names: List[str],
        exploration_rate: float = 0.05,
    ) -> None:
        if not strategy_names:
            raise ValueError("strategy_names must not be empty")
        self._strategy_names = list(strategy_names)
        self._exploration_rate = max(0.0, min(0.5, float(exploration_rate)))

        # Beta distribution parameters (uninformative prior: Beta(1,1))
        self._alpha: Dict[str, float] = {name: 1.0 for name in strategy_names}
        self._beta: Dict[str, float] = {name: 1.0 for name in strategy_names}

        # Cumulative tracking
        self._cumulative_pnl: Dict[str, float] = {name: 0.0 for name in strategy_names}
        self._trade_count: Dict[str, int] = {name: 0 for name in strategy_names}
        self._win_count: Dict[str, int] = {name: 0 for name in strategy_names}
        self._loss_count: Dict[str, int] = {name: 0 for name in strategy_names}

        # History for analysis
        self._outcome_history: List[Dict[str, Any]] = []

    # ── Recording ─────────────────────────────────────────────────────────

    def record_outcome(self, strategy: str, pnl: float) -> None:
        """
        Record a trade outcome.

        Positive PnL increments alpha (success); negative increments beta (failure).
        The magnitude of PnL is used to scale the update — larger wins/losses
        produce stronger posterior updates.
        """
        if strategy not in self._alpha:
            # Auto-register unknown strategies
            self._alpha[strategy] = 1.0
            self._beta[strategy] = 1.0
            self._cumulative_pnl[strategy] = 0.0
            self._trade_count[strategy] = 0
            self._win_count[strategy] = 0
            self._loss_count[strategy] = 0
            if strategy not in self._strategy_names:
                self._strategy_names.append(strategy)

        self._trade_count[strategy] += 1
        self._cumulative_pnl[strategy] += float(pnl)

        if pnl > 0:
            # Scale update: small wins get ~1.0, big wins get up to 2.0
            update = min(2.0, 1.0 + abs(pnl) / 100.0)
            self._alpha[strategy] += update
            self._win_count[strategy] += 1
        elif pnl < 0:
            update = min(2.0, 1.0 + abs(pnl) / 100.0)
            self._beta[strategy] += update
            self._loss_count[strategy] += 1
        # pnl == 0: no update (scratch trade)

        self._outcome_history.append({
            "strategy": strategy,
            "pnl": float(pnl),
            "timestamp": time.time(),
        })

        # Trim history
        if len(self._outcome_history) > 10000:
            self._outcome_history = self._outcome_history[-5000:]

    # ── Allocation ────────────────────────────────────────────────────────

    def get_allocations(self, total_capital: float) -> Dict[str, float]:
        """
        Thompson Sampling: sample from each strategy's Beta distribution,
        normalize to allocate capital proportionally to sampled values.
        Minimum exploration_rate allocation to each strategy.
        """
        if total_capital <= 0:
            return {name: 0.0 for name in self._strategy_names}

        n = len(self._strategy_names)
        if n == 0:
            return {}

        # Sample from Beta distributions
        samples: Dict[str, float] = {}
        for name in self._strategy_names:
            a = max(self._alpha[name], 0.01)
            b = max(self._beta[name], 0.01)
            samples[name] = float(np.random.beta(a, b))

        # Normalize samples to sum to 1.0
        total_sample = sum(samples.values())
        if total_sample < 1e-12:
            # Uniform if all samples are zero
            raw_weights = {name: 1.0 / n for name in self._strategy_names}
        else:
            raw_weights = {name: s / total_sample for name, s in samples.items()}

        # Apply exploration floor
        floor = self._exploration_rate
        remaining = 1.0 - floor * n
        if remaining < 0:
            # If floor * n > 1, just use uniform
            final_weights = {name: 1.0 / n for name in self._strategy_names}
        else:
            final_weights = {}
            for name in self._strategy_names:
                final_weights[name] = floor + remaining * raw_weights[name]

        # Convert to capital amounts
        allocations = {
            name: round(w * total_capital, 2)
            for name, w in final_weights.items()
        }
        return allocations

    # ── Rankings ──────────────────────────────────────────────────────────

    def get_rankings(self) -> List[dict]:
        """
        Return strategies ranked by expected win rate with confidence intervals.

        Each entry: {strategy, expected_win_rate, ci_lower, ci_upper,
                     trades, cumulative_pnl}
        """
        rankings = []
        for name in self._strategy_names:
            a = self._alpha[name]
            b = self._beta[name]
            expected = a / (a + b)

            # 95% confidence interval approximation for Beta distribution
            n_total = a + b
            if n_total > 2:
                std = math.sqrt(a * b / (n_total ** 2 * (n_total + 1)))
                ci_lower = max(0.0, expected - 1.96 * std)
                ci_upper = min(1.0, expected + 1.96 * std)
            else:
                ci_lower = 0.0
                ci_upper = 1.0

            rankings.append({
                "strategy": name,
                "expected_win_rate": round(expected, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "trades": self._trade_count[name],
                "cumulative_pnl": round(self._cumulative_pnl[name], 4),
            })

        rankings.sort(key=lambda x: x["expected_win_rate"], reverse=True)
        return rankings

    # ── Disable detection ─────────────────────────────────────────────────

    def should_disable(
        self,
        strategy: str,
        min_trades: int = 30,
        max_loss_rate: float = 0.7,
    ) -> bool:
        """
        True if strategy has min_trades+ trades and > max_loss_rate loss rate.
        """
        trades = self._trade_count.get(strategy, 0)
        if trades < min_trades:
            return False

        losses = self._loss_count.get(strategy, 0)
        loss_rate = losses / max(trades, 1)
        return loss_rate > max_loss_rate

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for dashboard/logging."""
        return {
            "strategy_count": len(self._strategy_names),
            "total_outcomes": sum(self._trade_count.values()),
            "rankings": self.get_rankings()[:5],
            "cumulative_pnl": dict(self._cumulative_pnl),
        }
