"""
Adaptive Hyperparameter Optimizer — self-tuning strategy parameters.

Automatically adjusts strategy parameters based on recent performance,
market regime, and volatility. Uses Bayesian optimization principles
with online learning.

Features:
- Automatic parameter tuning based on performance feedback
- Regime-specific parameter sets
- Bounded exploration to prevent catastrophic changes
- Performance tracking per parameter configuration
- Multi-armed bandit for strategy selection

Example::

    optimizer = AdaptiveHyperparameterOptimizer()
    optimizer.register_param("momentum_threshold", default=0.02, min_val=0.005, max_val=0.05)
    optimizer.register_param("stop_loss_pct", default=0.02, min_val=0.01, max_val=0.05)
    
    # After trades complete
    optimizer.update_performance(params={"momentum_threshold": 0.02, "stop_loss_pct": 0.02}, 
                                  pnl=150.0, sharpe=1.5)
    
    # Get next recommended parameters
    new_params = optimizer.get_best_params()
"""

from __future__ import annotations

import logging
import time
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParamConfig:
    """Parameter configuration."""
    name: str
    default: float
    min_val: float
    max_val: float
    step_size: Optional[float] = None  # Discrete step size (None = continuous)
    log_scale: bool = False  # Use logarithmic scale


@dataclass
class ParamSnapshot:
    """Parameter configuration with performance."""
    params: Dict[str, float]
    trades: int
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    timestamp: float
    regime: str = "unknown"


@dataclass
class OptimizationResult:
    """Optimization result."""
    best_params: Dict[str, float]
    expected_improvement: float
    exploration_needed: bool
    confidence: float
    param_history_count: int


class AdaptiveHyperparameterOptimizer:
    """
    Self-tuning hyperparameter optimizer using Bayesian optimization principles.

    Parameters
    ----------
    exploration_rate : float
        Probability of exploring new parameters (default 0.1).
    min_trades_for_update : int
        Minimum trades before updating parameters (default 10).
    performance_decay : float
        Decay factor for older performance (default 0.95).
    bounds_margin : float
        Margin from bounds for exploration (default 0.1).
    """

    def __init__(
        self,
        exploration_rate: float = 0.1,
        min_trades_for_update: int = 10,
        performance_decay: float = 0.95,
        bounds_margin: float = 0.1,
    ) -> None:
        self._exploration_rate = exploration_rate
        self._min_trades = min_trades_for_update
        self._decay = performance_decay
        self._bounds_margin = bounds_margin
        
        self._params: Dict[str, ParamConfig] = {}
        self._current_params: Dict[str, float] = {}
        self._param_history: List[ParamSnapshot] = []
        self._regime_params: Dict[str, Dict[str, float]] = {}  # regime -> best params
        
        logger.info(
            "AdaptiveHyperparameterOptimizer initialized: exploration=%.1f%% min_trades=%d",
            exploration_rate * 100, min_trades_for_update,
        )

    def register_param(
        self,
        name: str,
        default: float,
        min_val: float,
        max_val: float,
        step_size: Optional[float] = None,
        log_scale: bool = False,
    ) -> None:
        """Register a parameter for optimization."""
        self._params[name] = ParamConfig(
            name=name,
            default=default,
            min_val=min_val,
            max_val=max_val,
            step_size=step_size,
            log_scale=log_scale,
        )
        self._current_params[name] = default
        logger.info("Registered param: %s (default=%.4f, range=[%.4f, %.4f])",
                    name, default, min_val, max_val)

    def update_performance(
        self,
        params: Dict[str, float],
        pnl: float,
        sharpe: float = 0.0,
        win_rate: float = 0.5,
        max_drawdown: float = 0.0,
        regime: str = "unknown",
    ) -> None:
        """Record performance for a parameter configuration."""
        # Find or create snapshot for these params
        param_key = self._params_to_key(params)
        
        # Update existing or create new
        snapshot = ParamSnapshot(
            params=dict(params),
            trades=1,
            total_pnl=pnl,
            total_return_pct=pnl / 10000 * 100,  # Assuming $10k base
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            timestamp=time.time(),
            regime=regime,
        )
        
        # Check if similar params exist
        existing = self._find_similar_snapshot(params, regime)
        if existing:
            # Update existing snapshot with decay
            existing.trades += 1
            existing.total_pnl = existing.total_pnl * self._decay + pnl
            existing.sharpe_ratio = existing.sharpe_ratio * 0.8 + sharpe * 0.2
            existing.win_rate = existing.win_rate * 0.9 + win_rate * 0.1
            existing.max_drawdown = max(existing.max_drawdown, max_drawdown)
            existing.timestamp = time.time()
        else:
            self._param_history.append(snapshot)

        # Keep history bounded
        if len(self._param_history) > 1000:
            self._param_history = self._param_history[-500:]

    def _params_to_key(self, params: Dict[str, float]) -> str:
        """Convert params dict to hashable key."""
        return str(sorted(params.items()))

    def _find_similar_snapshot(
        self,
        params: Dict[str, float],
        regime: str,
        tolerance: float = 0.1,
    ) -> Optional[ParamSnapshot]:
        """Find similar parameter snapshot."""
        for snapshot in reversed(self._param_history):
            if snapshot.regime != regime:
                continue
            
            similar = True
            for name, value in params.items():
                if name in snapshot.params:
                    param_config = self._params.get(name)
                    if param_config:
                        range_size = param_config.max_val - param_config.min_val
                        diff = abs(snapshot.params[name] - value) / range_size
                        if diff > tolerance:
                            similar = False
                            break
            
            if similar:
                return snapshot
        
        return None

    def get_best_params(self, regime: str = "unknown") -> Dict[str, float]:
        """Get best parameters for current regime."""
        import random
        
        # Exploration vs exploitation
        if random.random() < self._exploration_rate:
            return self._explore_params(regime)
        
        # Get regime-specific best params
        regime_best = self._regime_params.get(regime)
        if regime_best:
            return regime_best
        
        # Find best from history
        best_snapshot = self._get_best_snapshot(regime)
        if best_snapshot and best_snapshot.trades >= self._min_trades:
            self._regime_params[regime] = dict(best_snapshot.params)
            return dict(best_snapshot.params)
        
        # Default params
        return dict(self._current_params)

    def _explore_params(self, regime: str) -> Dict[str, float]:
        """Generate exploration parameters."""
        import random
        
        explored = {}
        for name, config in self._params.items():
            current = self._current_params.get(name, config.default)
            
            # Add random perturbation
            range_size = config.max_val - config.min_val
            perturbation = random.uniform(-0.2, 0.2) * range_size
            
            new_val = current + perturbation
            
            # Apply bounds
            new_val = max(config.min_val, min(config.max_val, new_val))
            
            # Apply step size if discrete
            if config.step_size:
                new_val = round(new_val / config.step_size) * config.step_size
            
            explored[name] = new_val
        
        logger.debug("Exploring params: %s", explored)
        return explored

    def _get_best_snapshot(self, regime: str) -> Optional[ParamSnapshot]:
        """Get best performing parameter snapshot."""
        regime_snapshots = [
            s for s in self._param_history
            if s.regime == regime and s.trades >= self._min_trades
        ]
        
        if not regime_snapshots:
            # Try any regime
            regime_snapshots = [
                s for s in self._param_history
                if s.trades >= self._min_trades
            ]
        
        if not regime_snapshots:
            return None
        
        # Score by Sharpe * win_rate / drawdown
        def score(s: ParamSnapshot) -> float:
            drawdown_factor = 1.0 / (1.0 + s.max_drawdown)
            return s.sharpe_ratio * s.win_rate * drawdown_factor
        
        return max(regime_snapshots, key=score)

    def get_param_importance(self) -> Dict[str, float]:
        """Calculate parameter importance based on performance variance."""
        if len(self._param_history) < 5:
            return {name: 0.0 for name in self._params}
        
        importance = {}
        for name in self._params:
            # Calculate correlation between param value and performance
            values = []
            scores = []
            for snapshot in self._param_history:
                if name in snapshot.params and snapshot.trades >= 3:
                    values.append(snapshot.params[name])
                    scores.append(snapshot.sharpe_ratio)
            
            if len(values) >= 3:
                correlation = np.corrcoef(values, scores)[0, 1]
                importance[name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importance[name] = 0.0
        
        # Normalize
        total = sum(importance.values()) or 1.0
        return {k: v / total for k, v in importance.items()}

    def get_optimization_result(self, regime: str = "unknown") -> OptimizationResult:
        """Get comprehensive optimization result."""
        best_params = self.get_best_params(regime)
        best_snapshot = self._get_best_snapshot(regime)
        
        # Calculate expected improvement
        current_snapshot = self._find_similar_snapshot(best_params, regime)
        if best_snapshot and current_snapshot:
            expected_improvement = best_snapshot.sharpe_ratio - current_snapshot.sharpe_ratio
        else:
            expected_improvement = 0.0
        
        # Determine if more exploration needed
        exploration_needed = len(self._param_history) < 50 or self._exploration_rate > 0.15
        
        # Confidence based on data
        regime_data = [s for s in self._param_history if s.regime == regime]
        confidence = min(1.0, len(regime_data) / 50)
        
        return OptimizationResult(
            best_params=best_params,
            expected_improvement=expected_improvement,
            exploration_needed=exploration_needed,
            confidence=confidence,
            param_history_count=len(self._param_history),
        )

    def reset_regime(self, regime: str) -> None:
        """Reset learned parameters for a regime."""
        if regime in self._regime_params:
            del self._regime_params[regime]
        self._param_history = [
            s for s in self._param_history if s.regime != regime
        ]
        logger.info("Reset parameters for regime: %s", regime)

    def set_exploration_rate(self, rate: float) -> None:
        """Update exploration rate."""
        self._exploration_rate = max(0.0, min(1.0, rate))
        logger.info("Exploration rate set to: %.1f%%", self._exploration_rate * 100)

    def get_current_params(self) -> Dict[str, float]:
        """Get current parameter values."""
        return dict(self._current_params)

    def set_params(self, params: Dict[str, float]) -> None:
        """Manually set parameters."""
        for name, value in params.items():
            if name in self._params:
                config = self._params[name]
                value = max(config.min_val, min(config.max_val, value))
                self._current_params[name] = value

    def get_all_regimes(self) -> List[str]:
        """Get all regimes with learned parameters."""
        return list(self._regime_params.keys())


__all__ = ["AdaptiveHyperparameterOptimizer", "ParamConfig", "OptimizationResult"]
