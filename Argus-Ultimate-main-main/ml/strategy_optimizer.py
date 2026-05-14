"""
Autonomous Strategy Optimizer for ARGUS.

Automatically tunes strategy parameters based on live trade performance.
Uses exponential-decay-weighted parameter-outcome correlations to nudge
parameters toward configurations that produced positive PnL.

Stability: parameters never change by more than 10% per optimization round.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TradeOutcome:
    """Single trade observation used for parameter optimization."""
    strategy_name: str
    params: Dict[str, float]
    pnl: float
    slippage_bps: float
    hold_time_seconds: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationResult:
    """Result of a single optimization pass."""
    strategy_name: str
    old_params: Dict[str, float]
    new_params: Dict[str, float]
    trades_used: int
    avg_pnl_before: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# StrategyOptimizer
# ---------------------------------------------------------------------------


class StrategyOptimizer:
    """
    Autonomously tunes strategy parameters based on live performance.

    Maintains a rolling window of trade outcomes per strategy and uses
    exponential-decay-weighted correlations between parameter values and
    PnL to nudge parameters toward profitable configurations.

    Parameters
    ----------
    lookback_trades : int
        Maximum trades to keep in history per strategy (FIFO).
    optimization_interval_hours : float
        Minimum hours between consecutive optimizations per strategy.
    max_param_change_pct : float
        Maximum relative change per parameter per optimization (default 0.10 = 10%).
    decay_half_life : int
        Number of trades for exponential decay half-life (recent trades weighted more).
    """

    def __init__(
        self,
        lookback_trades: int = 100,
        optimization_interval_hours: float = 24.0,
        max_param_change_pct: float = 0.10,
        decay_half_life: int = 30,
    ) -> None:
        if lookback_trades < 5:
            raise ValueError("lookback_trades must be >= 5")
        self.lookback_trades = lookback_trades
        self.optimization_interval_hours = optimization_interval_hours
        self.max_param_change_pct = max_param_change_pct
        self.decay_half_life = decay_half_life

        # Per-strategy state
        self._trades: Dict[str, List[TradeOutcome]] = defaultdict(list)
        self._last_optimized: Dict[str, float] = {}
        self._current_params: Dict[str, Dict[str, float]] = {}
        self._optimization_history: Dict[str, List[OptimizationResult]] = defaultdict(list)

    # ── Recording ─────────────────────────────────────────────────────────

    def record_trade_outcome(
        self,
        strategy_name: str,
        params: Dict[str, float],
        pnl: float,
        slippage_bps: float = 0.0,
        hold_time: float = 0.0,
    ) -> None:
        """Record a completed trade's outcome for the strategy."""
        outcome = TradeOutcome(
            strategy_name=strategy_name,
            params=dict(params),
            pnl=float(pnl),
            slippage_bps=float(slippage_bps),
            hold_time_seconds=float(hold_time),
        )
        trades = self._trades[strategy_name]
        trades.append(outcome)
        # FIFO: keep only the most recent lookback_trades
        if len(trades) > self.lookback_trades:
            self._trades[strategy_name] = trades[-self.lookback_trades:]

        # Track the latest params as "current" if we have none yet
        if strategy_name not in self._current_params and params:
            self._current_params[strategy_name] = dict(params)

    # ── Optimization trigger ──────────────────────────────────────────────

    def should_optimize(self, strategy_name: str) -> bool:
        """True if enough trades have accumulated since last optimization."""
        trades = self._trades.get(strategy_name, [])
        if len(trades) < 10:
            return False

        last = self._last_optimized.get(strategy_name, 0.0)
        hours_since = (time.time() - last) / 3600.0
        if hours_since < self.optimization_interval_hours:
            return False

        # Require at least 10 new trades since last optimization
        new_trades = sum(
            1 for t in trades if t.timestamp > last
        )
        return new_trades >= 10

    # ── Core optimization ─────────────────────────────────────────────────

    def optimize(self, strategy_name: str) -> Dict[str, float]:
        """
        Return optimized parameters using exponential-decay-weighted
        parameter-outcome correlations.

        - Track parameter -> outcome correlations
        - Nudge parameters toward configurations that produced positive PnL
        - Use exponential decay to weight recent trades more heavily
        - Never change parameters by more than max_param_change_pct per round
        - Return {param_name: new_value} dict
        """
        trades = self._trades.get(strategy_name, [])
        if len(trades) < 5:
            return self._current_params.get(strategy_name, {})

        # Determine parameter names from the first trade that has params
        param_names: List[str] = []
        for t in trades:
            if t.params:
                param_names = list(t.params.keys())
                break
        if not param_names:
            return self._current_params.get(strategy_name, {})

        # Build arrays: rows = trades, columns = params, target = pnl
        n = len(trades)
        param_matrix = np.zeros((n, len(param_names)))
        pnl_array = np.zeros(n)
        weights = np.zeros(n)

        # Exponential decay: most recent trade gets weight 1.0
        decay_rate = math.log(2.0) / max(self.decay_half_life, 1)
        for i, t in enumerate(trades):
            age = n - 1 - i  # 0 for newest, n-1 for oldest
            weights[i] = math.exp(-decay_rate * age)
            pnl_array[i] = t.pnl
            for j, pname in enumerate(param_names):
                param_matrix[i, j] = t.params.get(pname, 0.0)

        # Normalize weights
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum

        # Weighted mean PnL
        weighted_mean_pnl = float(np.dot(weights, pnl_array))

        # Current params (or weighted-average of all observed params)
        current = self._current_params.get(strategy_name, {})
        if not current:
            # Use weighted average of observed params as baseline
            for j, pname in enumerate(param_names):
                current[pname] = float(np.dot(weights, param_matrix[:, j]))

        # For each parameter, compute weighted correlation with PnL
        # Then nudge in the direction of the gradient
        new_params: Dict[str, float] = {}

        for j, pname in enumerate(param_names):
            col = param_matrix[:, j]
            cur_val = current.get(pname, float(np.dot(weights, col)))

            # Weighted correlation between param values and PnL
            param_mean = float(np.dot(weights, col))
            pnl_mean = weighted_mean_pnl

            # Weighted covariance
            diffs_param = col - param_mean
            diffs_pnl = pnl_array - pnl_mean
            cov = float(np.dot(weights, diffs_param * diffs_pnl))

            # Weighted variance of param
            var_param = float(np.dot(weights, diffs_param ** 2))
            var_pnl = float(np.dot(weights, diffs_pnl ** 2))

            # Correlation coefficient
            denom = math.sqrt(max(var_param, 1e-12) * max(var_pnl, 1e-12))
            corr = cov / denom if denom > 1e-12 else 0.0

            # Gradient direction: if positive correlation, increase param
            # Scale the nudge by the correlation strength
            # Use param's standard deviation to normalize the step size
            param_std = math.sqrt(max(var_param, 1e-12))

            # Step = correlation * fraction of param std
            step = corr * param_std * 0.3  # conservative factor

            # Clamp to max_param_change_pct of current value
            abs_cur = abs(cur_val) if abs(cur_val) > 1e-9 else 1e-3
            max_change = abs_cur * self.max_param_change_pct
            step = max(-max_change, min(max_change, step))

            new_params[pname] = cur_val + step

        # Record optimization
        old_params = dict(current)
        result = OptimizationResult(
            strategy_name=strategy_name,
            old_params=old_params,
            new_params=dict(new_params),
            trades_used=n,
            avg_pnl_before=weighted_mean_pnl,
        )
        self._optimization_history[strategy_name].append(result)
        self._current_params[strategy_name] = dict(new_params)
        self._last_optimized[strategy_name] = time.time()

        logger.info(
            "StrategyOptimizer: optimized '%s' with %d trades — "
            "avg_pnl=%.4f, params=%d adjusted",
            strategy_name, n, weighted_mean_pnl, len(new_params),
        )

        return new_params

    # ── Reporting ─────────────────────────────────────────────────────────

    def get_strategy_report(self, strategy_name: str) -> Dict[str, Any]:
        """
        Return performance report for a single strategy.

        Returns:
            {trades, win_rate, avg_pnl, sharpe, best_params, worst_params, last_optimized}
        """
        trades = self._trades.get(strategy_name, [])
        if not trades:
            return {
                "strategy": strategy_name,
                "trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "sharpe": 0.0,
                "best_params": {},
                "worst_params": {},
                "last_optimized": None,
            }

        pnls = [t.pnl for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        n = len(pnls)
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls)) if n > 1 else 1.0

        sharpe = avg_pnl / std_pnl if std_pnl > 1e-9 else 0.0

        # Best/worst params by PnL
        sorted_trades = sorted(trades, key=lambda t: t.pnl, reverse=True)
        best_params = sorted_trades[0].params if sorted_trades else {}
        worst_params = sorted_trades[-1].params if sorted_trades else {}

        last_opt = self._last_optimized.get(strategy_name)

        return {
            "strategy": strategy_name,
            "trades": n,
            "win_rate": wins / n if n > 0 else 0.0,
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "best_params": best_params,
            "worst_params": worst_params,
            "last_optimized": last_opt,
        }

    def get_all_reports(self) -> Dict[str, Any]:
        """Aggregate performance report across all strategies."""
        reports = {}
        for name in self._trades:
            reports[name] = self.get_strategy_report(name)
        return reports

    def get_current_params(self, strategy_name: str) -> Dict[str, float]:
        """Return the current optimized parameters for a strategy."""
        return dict(self._current_params.get(strategy_name, {}))
