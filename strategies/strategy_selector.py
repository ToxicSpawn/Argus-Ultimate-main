"""
Smart Strategy Selector — dynamically enables/disables strategies based on
current market regime and recent performance.

The selector maps market regimes to strategy sets and adjusts capital
allocation based on both regime suitability and rolling performance.

Usage:
    selector = StrategySelector()
    active = selector.select("RANGE", performance_data)
    allocations = selector.get_allocation("RANGE", capital=1000.0)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Performance tracking
# ---------------------------------------------------------------------------

@dataclass
class StrategyPerformance:
    """Rolling performance metrics for a single strategy."""

    name: str
    recent_pnl: List[float] = field(default_factory=list)  # list of per-trade PnL %
    max_recent: int = 50  # rolling window size

    @property
    def n_trades(self) -> int:
        return len(self.recent_pnl)

    @property
    def total_pnl(self) -> float:
        return sum(self.recent_pnl) if self.recent_pnl else 0.0

    @property
    def mean_pnl(self) -> float:
        return self.total_pnl / self.n_trades if self.n_trades > 0 else 0.0

    @property
    def win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return sum(1 for p in self.recent_pnl if p > 0) / self.n_trades

    @property
    def sharpe(self) -> float:
        """Annualized Sharpe-like ratio from recent trades."""
        if self.n_trades < 2:
            return 0.0
        mean = self.mean_pnl
        var = sum((p - mean) ** 2 for p in self.recent_pnl) / (self.n_trades - 1)
        std = math.sqrt(var) if var > 0 else 1e-9
        # Rough annualization (assume ~1 trade/day, 252 trading days)
        return (mean / std) * math.sqrt(min(252, self.n_trades))

    def record(self, pnl_pct: float) -> None:
        """Record a trade result."""
        self.recent_pnl.append(pnl_pct)
        if len(self.recent_pnl) > self.max_recent:
            self.recent_pnl.pop(0)


# ---------------------------------------------------------------------------
# Regime to strategy mapping
# ---------------------------------------------------------------------------

# Default regime -> strategy mapping.  Each list is ordered by priority.
REGIME_MAP: Dict[str, List[str]] = {
    "TRENDING_UP":   ["momentum", "breakout"],
    "TRENDING_DOWN": ["momentum"],               # momentum captures downtrends too
    "TREND_UP":      ["momentum", "breakout"],    # alias
    "TREND_DOWN":    ["momentum"],                # alias
    "RANGE":         ["mean_reversion", "kalman_pairs", "stat_arb"],
    "HIGH_VOL":      ["breakout", "kalman_pairs"],
    "LOW_VOL":       ["mean_reversion", "stat_arb"],
    "CRISIS":        [],                          # stay flat — capital preservation
    "UNKNOWN":       ["mean_reversion", "momentum"],  # balanced fallback
    # Hurst exponent regimes (v8.5.0)
    "HURST_MEAN_REVERSION": ["mean_reversion", "kalman_pairs", "stat_arb"],
    "HURST_MOMENTUM":       ["momentum", "breakout"],
    "HURST_AVOID":          [],                    # random walk — avoid trading
}

# Base weight per strategy when it's eligible for the regime
BASE_WEIGHTS: Dict[str, float] = {
    "momentum":       1.0,
    "breakout":       0.8,
    "mean_reversion": 1.0,
    "kalman_pairs":   1.2,   # highest risk-adjusted, gets a premium
    "stat_arb":       1.0,
    "scalping":       0.5,   # lower weight — sensitive to fees
}

# Minimum Sharpe to keep a strategy active (over last 50 trades)
MIN_SHARPE_THRESHOLD: float = 0.0  # disable strategies with negative Sharpe


class StrategySelector:
    """
    Dynamically selects which strategies should be active based on market
    regime and recent performance.

    Thread-safe for single-writer / multi-reader (the typical async loop).
    """

    def __init__(
        self,
        regime_map: Optional[Dict[str, List[str]]] = None,
        base_weights: Optional[Dict[str, float]] = None,
        min_sharpe: float = MIN_SHARPE_THRESHOLD,
        min_trades_for_eval: int = 10,
    ) -> None:
        self._regime_map = regime_map or dict(REGIME_MAP)
        self._base_weights = base_weights or dict(BASE_WEIGHTS)
        self._min_sharpe = min_sharpe
        self._min_trades_for_eval = min_trades_for_eval

        # Strategy name -> StrategyPerformance
        self._performance: Dict[str, StrategyPerformance] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        regime: str,
        strategy_performance: Optional[Dict[str, StrategyPerformance]] = None,
    ) -> List[str]:
        """
        Return list of strategy names to activate for the given regime.

        Also disables any strategy with negative Sharpe over the last
        ``min_trades_for_eval`` trades (if enough data is available).

        Parameters
        ----------
        regime:
            Current market regime string (e.g. "RANGE", "TRENDING_UP").
        strategy_performance:
            Optional override of the internal performance tracker.
            If None, uses ``self._performance``.

        Returns
        -------
        List of strategy name strings that should be active.
        """
        perf = strategy_performance or self._performance
        regime_upper = regime.upper().replace(" ", "_")

        # Get candidate strategies for this regime
        candidates = list(self._regime_map.get(regime_upper, []))

        if not candidates:
            logger.info(
                "StrategySelector: regime=%s maps to no strategies (crisis / unknown)",
                regime_upper,
            )
            return []

        # Filter out strategies with negative Sharpe (if we have enough data)
        active: List[str] = []
        for name in candidates:
            sp = perf.get(name)
            if sp is not None and sp.n_trades >= self._min_trades_for_eval:
                if sp.sharpe < self._min_sharpe:
                    logger.info(
                        "StrategySelector: disabling %s — Sharpe=%.3f < %.3f "
                        "over %d trades",
                        name, sp.sharpe, self._min_sharpe, sp.n_trades,
                    )
                    continue
            active.append(name)

        logger.debug(
            "StrategySelector: regime=%s active=%s (from candidates=%s)",
            regime_upper, active, candidates,
        )
        return active

    def get_allocation(
        self,
        regime: str,
        capital: float,
        strategy_performance: Optional[Dict[str, StrategyPerformance]] = None,
    ) -> Dict[str, float]:
        """
        Return ``{strategy_name: capital_allocation}`` based on regime +
        performance-weighted allocation.

        Strategies with higher Sharpe ratios get proportionally more capital.
        Minimum allocation per strategy is 10% of an equal-weight share.

        Parameters
        ----------
        regime:
            Market regime string.
        capital:
            Total capital available (AUD).
        strategy_performance:
            Optional override.

        Returns
        -------
        Dict mapping active strategy names to AUD allocation.
        """
        active = self.select(regime, strategy_performance)
        if not active:
            return {}

        perf = strategy_performance or self._performance

        # Compute raw weights: base_weight * (1 + sharpe_bonus)
        raw_weights: Dict[str, float] = {}
        for name in active:
            base_w = self._base_weights.get(name, 1.0)
            sp = perf.get(name)
            sharpe_bonus = 0.0
            if sp is not None and sp.n_trades >= self._min_trades_for_eval:
                # Clamp Sharpe bonus to [-0.5, 1.0] to avoid extreme swings
                sharpe_bonus = max(-0.5, min(1.0, sp.sharpe * 0.2))
            raw_weights[name] = max(0.1, base_w * (1.0 + sharpe_bonus))

        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            return {}

        allocations = {
            name: capital * (w / total_weight)
            for name, w in raw_weights.items()
        }

        logger.debug(
            "StrategySelector: regime=%s capital=%.2f allocations=%s",
            regime, capital, {k: f"${v:.2f}" for k, v in allocations.items()},
        )
        return allocations

    def record_trade(self, strategy_name: str, pnl_pct: float) -> None:
        """Record a completed trade for performance tracking."""
        if strategy_name not in self._performance:
            self._performance[strategy_name] = StrategyPerformance(name=strategy_name)
        self._performance[strategy_name].record(pnl_pct)

    def get_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance tracker for a strategy."""
        return self._performance.get(strategy_name)

    def get_all_performance(self) -> Dict[str, StrategyPerformance]:
        """Get all performance trackers."""
        return dict(self._performance)

    def get_regime_strategies(self, regime: str) -> List[str]:
        """Get the raw (unfiltered) strategy list for a regime."""
        return list(self._regime_map.get(regime.upper().replace(" ", "_"), []))

    def set_regime_strategies(self, regime: str, strategies: List[str]) -> None:
        """Override the strategy list for a regime."""
        self._regime_map[regime.upper().replace(" ", "_")] = list(strategies)

    def select_with_hurst(
        self,
        regime: str,
        hurst_regime: Optional[str] = None,
        strategy_performance: Optional[Dict[str, StrategyPerformance]] = None,
    ) -> List[str]:
        """
        Select strategies considering both market regime and Hurst exponent regime.
        
        The Hurst regime provides additional signal:
        - HURST_MEAN_REVERSION: Boost mean reversion strategies
        - HURST_MOMENTUM: Boost momentum strategies
        - HURST_AVOID: Reduce or eliminate trading
        
        Parameters
        ----------
        regime:
            Market regime string (e.g. "RANGE", "TRENDING_UP").
        hurst_regime:
            Hurst regime string (e.g. "mean_reversion", "momentum", "avoid").
            If None, uses standard regime selection.
        strategy_performance:
            Optional performance data.
            
        Returns
        -------
        List of strategy names to activate.
        """
        if hurst_regime is None:
            return self.select(regime, strategy_performance)
        
        hurst_regime_upper = f"HURST_{hurst_regime.upper()}"
        hurst_strategies = self._regime_map.get(hurst_regime_upper, [])
        
        # If Hurst says avoid, return empty (no trading)
        if hurst_regime == "avoid":
            logger.info(
                "StrategySelector: Hurst regime=avoid — skipping trading"
            )
            return []
        
        # Get base strategies from market regime
        base_strategies = self.select(regime, strategy_performance)
        
        # Combine strategies, prioritizing Hurst-aligned ones
        if hurst_regime == "mean_reversion":
            # Boost mean reversion strategies
            combined = []
            for s in base_strategies:
                if s in ["mean_reversion", "kalman_pairs", "stat_arb"]:
                    combined.insert(0, s)  # Move to front
                else:
                    combined.append(s)
            # Add any additional Hurst-recommended strategies
            for s in hurst_strategies:
                if s not in combined:
                    combined.append(s)
            return combined[:5]  # Limit to 5 strategies
        
        elif hurst_regime == "momentum":
            # Boost momentum strategies
            combined = []
            for s in base_strategies:
                if s in ["momentum", "breakout"]:
                    combined.insert(0, s)  # Move to front
                else:
                    combined.append(s)
            # Add any additional Hurst-recommended strategies
            for s in hurst_strategies:
                if s not in combined:
                    combined.append(s)
            return combined[:5]  # Limit to 5 strategies
        
        return base_strategies

    def get_allocation_with_hurst(
        self,
        regime: str,
        capital: float,
        hurst_regime: Optional[str] = None,
        strategy_performance: Optional[Dict[str, StrategyPerformance]] = None,
    ) -> Dict[str, float]:
        """
        Get capital allocation considering both market regime and Hurst regime.
        
        When Hurst regime is specified:
        - mean_reversion: Full allocation to mean reversion strategies
        - momentum: Full allocation to momentum strategies
        - avoid: Zero allocation (no trading)
        
        Parameters
        ----------
        regime:
            Market regime string.
        capital:
            Total capital available.
        hurst_regime:
            Hurst regime string (optional).
        strategy_performance:
            Optional performance data.
            
        Returns
        -------
        Dict mapping strategy names to capital allocations.
        """
        if hurst_regime is None:
            return self.get_allocation(regime, capital, strategy_performance)
        
        # If Hurst says avoid, return empty allocation
        if hurst_regime == "avoid":
            logger.info(
                "StrategySelector: Hurst regime=avoid — zero allocation"
            )
            return {}
        
        # Get strategies considering Hurst
        active = self.select_with_hurst(regime, hurst_regime, strategy_performance)
        if not active:
            return {}
        
        perf = strategy_performance or self._performance
        
        # Compute weights with Hurst boost
        raw_weights: Dict[str, float] = {}
        hurst_boost = 1.5 if hurst_regime in ["mean_reversion", "momentum"] else 1.0
        
        for name in active:
            base_w = self._base_weights.get(name, 1.0)
            sp = perf.get(name)
            sharpe_bonus = 0.0
            if sp is not None and sp.n_trades >= self._min_trades_for_eval:
                sharpe_bonus = max(-0.5, min(1.0, sp.sharpe * 0.2))
            
            # Apply Hurst boost to aligned strategies
            is_hurst_aligned = False
            if hurst_regime == "mean_reversion" and name in ["mean_reversion", "kalman_pairs", "stat_arb"]:
                is_hurst_aligned = True
            elif hurst_regime == "momentum" and name in ["momentum", "breakout"]:
                is_hurst_aligned = True
            
            boost = hurst_boost if is_hurst_aligned else 1.0
            raw_weights[name] = max(0.1, base_w * (1.0 + sharpe_bonus) * boost)
        
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            return {}
        
        allocations = {
            name: capital * (w / total_weight)
            for name, w in raw_weights.items()
        }
        
        logger.debug(
            "StrategySelector: regime=%s hurst=%s capital=%.2f allocations=%s",
            regime, hurst_regime, capital, {k: f"${v:.2f}" for k, v in allocations.items()},
        )
        return allocations
