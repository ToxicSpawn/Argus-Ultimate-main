"""
adaptive/unified_strategy_orchestrator.py — Unified Adaptive Strategy Orchestrator

Integrates all adaptive systems with trading strategies for maximum returns.

This orchestrator coordinates:
1. Regime detection → Strategy routing
2. Meta-learning → Weight optimization
3. Risk adaptation → Position sizing
4. Decay detection → Strategy lifecycle
5. Parameter tuning → Strategy optimization

Usage::

    from adaptive.unified_strategy_orchestrator import UnifiedStrategyOrchestrator
    
    orchestrator = UnifiedStrategyOrchestrator()
    
    # Register strategies
    orchestrator.register_strategy("momentum", MomentumStrategy())
    orchestrator.register_strategy("mean_reversion", MeanReversionStrategy())
    orchestrator.register_strategy("mev_sandwich", MEVSandwichStrategy())
    orchestrator.register_strategy("triangular_arb", TriangularArbitrageStrategy())
    orchestrator.register_strategy("options_vol_arb", OptionsVolArbitrageStrategy())
    orchestrator.register_strategy("grid_mean_reversion", GridMeanReversionStrategy())
    
    # Run adaptation cycle
    result = await orchestrator.run_adaptation_cycle(market_data)
    
    # Get strategy weights for current regime
    weights = orchestrator.get_strategy_weights()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive.regime import MarketRegime
from adaptive.regime_strategy_router import RegimeStrategyRouter
from adaptive.auto_strategy_manager import AutoStrategyManager, StrategyAction
from adaptive.self_optimizing_meta_engine import SelfOptimizingMetaEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""
    strategy_name: str
    sharpe: float = 0.0
    sharpe_14d: float = 0.0
    sharpe_trend: float = 0.0  # +1 = improving, -1 = declining
    win_rate: float = 0.5
    profit_factor: float = 1.0
    expectancy: float = 0.0
    current_weight: float = 0.1
    trades_7d: int = 0
    last_trade_ts: float = 0.0
    is_active: bool = True
    strategy_type: str = "unknown"
    backtest_passed: bool = True
    regime_label: Optional[str] = None
    drawdown_pct: float = 0.0
    avg_realized_slippage_bps: float = 0.0
    fee_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for adaptive system compatibility."""
        return {
            "sharpe": self.sharpe,
            "sharpe_14d": self.sharpe_14d,
            "sharpe_trend": self.sharpe_trend,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "current_weight": self.current_weight,
            "trades_7d": self.trades_7d,
            "last_trade_ts": self.last_trade_ts,
            "is_active": self.is_active,
            "strategy_type": self.strategy_type,
            "backtest_passed": self.backtest_passed,
            "regime_label": self.regime_label,
            "drawdown_pct": self.drawdown_pct,
            "avg_realized_slippage_bps": self.avg_realized_slippage_bps,
            "fee_ratio": self.fee_ratio,
        }


@dataclass
class StrategyState:
    """Current state of a strategy."""
    strategy_name: str
    is_enabled: bool = True
    current_weight: float = 0.1
    adjusted_weight: float = 0.1
    regime_boost: float = 1.0
    meta_weight: float = 0.1
    risk_multiplier: float = 1.0
    final_allocation: float = 0.1
    last_signal_ts: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: float = 0.0


@dataclass
class AdaptationCycleResult:
    """Result of a complete adaptation cycle."""
    timestamp: datetime
    regime: str
    regime_confidence: float
    strategy_weights: Dict[str, float]
    strategy_actions: List[StrategyAction]
    risk_adjustments: Dict[str, float]
    total_adaptation_time_ms: float
    alerts: List[str]


# ============================================================================
# Strategy Type Classification
# ============================================================================

# Classify strategies by their optimal market conditions
STRATEGY_REGIME_AFFINITY: Dict[str, Dict[str, float]] = {
    # High-frequency / execution strategies
    "mev_sandwich": {
        "trending": 1.2, "bull": 1.1, "mean_revert": 0.8, "range": 0.8,
        "crisis": 0.5, "bear": 0.6, "volatile": 0.7, "calm": 1.0, "recovery": 1.0,
    },
    "triangular_arb": {
        "trending": 1.0, "bull": 1.0, "mean_revert": 1.2, "range": 1.2,
        "crisis": 0.6, "bear": 0.8, "volatile": 0.8, "calm": 1.3, "recovery": 1.0,
    },
    "cross_chain_arb": {
        "trending": 1.1, "bull": 1.1, "mean_revert": 1.0, "range": 1.0,
        "crisis": 0.4, "bear": 0.6, "volatile": 0.6, "calm": 1.2, "recovery": 1.0,
    },
    "oracle_deviation": {
        "trending": 0.8, "bull": 0.9, "mean_revert": 1.3, "range": 1.3,
        "crisis": 0.7, "bear": 0.9, "volatile": 1.1, "calm": 1.2, "recovery": 1.0,
    },
    
    # Volatility strategies
    "options_vol_arb": {
        "trending": 0.7, "bull": 0.8, "mean_revert": 1.0, "range": 1.1,
        "crisis": 1.5, "bear": 1.3, "volatile": 1.4, "calm": 0.6, "recovery": 0.9,
    },
    
    # Grid / mean reversion
    "grid_mean_reversion": {
        "trending": 0.5, "bull": 0.6, "mean_revert": 1.5, "range": 1.5,
        "crisis": 0.4, "bear": 0.7, "volatile": 0.8, "calm": 1.4, "recovery": 1.1,
    },
    
    # Traditional strategies
    "momentum": {
        "trending": 1.5, "bull": 1.4, "mean_revert": 0.4, "range": 0.4,
        "crisis": 0.3, "bear": 0.3, "volatile": 0.5, "calm": 0.6, "recovery": 1.2,
    },
    "mean_reversion": {
        "trending": 0.4, "bull": 0.5, "mean_revert": 1.5, "range": 1.4,
        "crisis": 0.5, "bear": 0.8, "volatile": 0.7, "calm": 1.3, "recovery": 1.0,
    },
    "trend_following": {
        "trending": 1.5, "bull": 1.4, "mean_revert": 0.3, "range": 0.3,
        "crisis": 0.2, "bear": 0.3, "volatile": 0.4, "calm": 0.5, "recovery": 1.1,
    },
}


# ============================================================================
# Unified Strategy Orchestrator
# ============================================================================

class UnifiedStrategyOrchestrator:
    """
    Unified orchestrator that integrates all adaptive systems with strategies.
    
    Coordinates:
    - Regime detection → Strategy routing
    - Meta-learning → Weight optimization  
    - Risk adaptation → Position sizing
    - Decay detection → Strategy lifecycle
    - Parameter tuning → Strategy optimization
    """
    
    def __init__(
        self,
        *,
        enable_regime_routing: bool = True,
        enable_meta_learning: bool = True,
        enable_risk_adaptation: bool = True,
        enable_decay_detection: bool = True,
        min_regime_confidence: float = 0.6,
        max_strategy_allocation: float = 0.4,
        min_strategy_allocation: float = 0.02,
    ):
        """Initialize the unified orchestrator."""
        self.enable_regime_routing = enable_regime_routing
        self.enable_meta_learning = enable_meta_learning
        self.enable_risk_adaptation = enable_risk_adaptation
        self.enable_decay_detection = enable_decay_detection
        self.min_regime_confidence = min_regime_confidence
        self.max_strategy_allocation = max_strategy_allocation
        self.min_strategy_allocation = min_strategy_allocation
        
        # Adaptive components
        self._regime_router = RegimeStrategyRouter()
        self._strategy_manager = AutoStrategyManager()
        self._meta_engine = SelfOptimizingMetaEngine(
            min_trades_for_reweighting=5,
            meta_alpha=0.2,
            max_weight_change_per_update=0.15,
            min_weight_per_strategy=min_strategy_allocation,
            max_weight_per_strategy=max_strategy_allocation,
        )
        
        # Strategy state
        self._strategies: Dict[str, Any] = {}
        self._strategy_states: Dict[str, StrategyState] = {}
        self._strategy_metrics: Dict[str, StrategyMetrics] = {}
        
        # Current state
        self._current_regime: str = "unknown"
        self._regime_confidence: float = 0.0
        self._last_cycle_time: float = 0.0
        self._adaptation_count: int = 0
        
        logger.info(
            "UnifiedStrategyOrchestrator initialized (regime=%s, meta=%s, risk=%s)",
            enable_regime_routing, enable_meta_learning, enable_risk_adaptation,
        )
    
    def register_strategy(
        self,
        name: str,
        strategy: Any,
        *,
        strategy_type: str = "unknown",
        initial_weight: float = 0.1,
    ) -> None:
        """Register a strategy with the orchestrator."""
        self._strategies[name] = strategy
        self._strategy_states[name] = StrategyState(
            strategy_name=name,
            current_weight=initial_weight,
            adjusted_weight=initial_weight,
            meta_weight=initial_weight,
            final_allocation=initial_weight,
        )
        self._strategy_metrics[name] = StrategyMetrics(
            strategy_name=name,
            strategy_type=strategy_type,
            current_weight=initial_weight,
        )
        logger.info("Registered strategy: %s (type=%s, weight=%.2f)", name, strategy_type, initial_weight)
    
    def unregister_strategy(self, name: str) -> None:
        """Unregister a strategy."""
        self._strategies.pop(name, None)
        self._strategy_states.pop(name, None)
        self._strategy_metrics.pop(name, None)
        logger.info("Unregistered strategy: %s", name)
    
    def update_strategy_metrics(self, name: str, metrics: Dict[str, Any]) -> None:
        """Update performance metrics for a strategy."""
        if name not in self._strategy_metrics:
            logger.warning("Unknown strategy: %s", name)
            return
        
        m = self._strategy_metrics[name]
        for key, value in metrics.items():
            if hasattr(m, key):
                setattr(m, key, value)
    
    def update_regime(self, regime: str, confidence: float) -> None:
        """Update the current market regime."""
        self._current_regime = regime.lower()
        self._regime_confidence = confidence
        logger.debug("Regime updated: %s (confidence=%.2f)", regime, confidence)
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights after all adaptations."""
        return {
            name: state.final_allocation
            for name, state in self._strategy_states.items()
        }
    
    def get_strategy_state(self, name: str) -> Optional[StrategyState]:
        """Get the current state of a strategy."""
        return self._strategy_states.get(name)
    
    def should_skip_strategy(self, name: str) -> bool:
        """Check if a strategy should be skipped in current regime."""
        if not self.enable_regime_routing:
            return False
        
        state = self._strategy_states.get(name)
        if state and not state.is_enabled:
            return True
        
        return self._regime_router.should_skip_strategy(name, self._current_regime)
    
    def run_adaptation_cycle(
        self,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> AdaptationCycleResult:
        """
        Run a complete adaptation cycle.
        
        This coordinates all adaptive systems:
        1. Regime-based weight adjustment
        2. Meta-learning weight optimization
        3. Risk-based position sizing
        4. Decay detection and lifecycle management
        """
        start_time = time.monotonic()
        alerts: List[str] = []
        strategy_actions: List[StrategyAction] = []
        risk_adjustments: Dict[str, float] = {}
        
        # Step 1: Get base weights
        base_weights = {
            name: state.current_weight
            for name, state in self._strategy_states.items()
        }
        
        # Step 2: Apply regime-based routing
        regime_weights = base_weights
        if self.enable_regime_routing and self._regime_confidence >= self.min_regime_confidence:
            regime_weights = self._regime_router.get_weights(
                self._current_regime,
                base_weights,
            )
            for name, state in self._strategy_states.items():
                state.regime_boost = regime_weights.get(name, 1.0) / max(base_weights.get(name, 0.1), 0.01)
            
            if self._regime_confidence < 0.7:
                alerts.append(f"Low regime confidence: {self._regime_confidence:.2f}")
        
        # Step 3: Apply meta-learning weights
        meta_weights = regime_weights
        if self.enable_meta_learning:
            # Update meta engine with current metrics
            for name, metrics in self._strategy_metrics.items():
                if metrics.trades_7d > 0:
                    self._meta_engine.record_trade({
                        "source_strategy": name,
                        "strategy": name,
                        "pnl": metrics.expectancy * metrics.trades_7d,
                        "regime_label": self._current_regime,
                    })
            
            # Get meta-optimized weights
            try:
                meta_result = self._meta_engine.compute_weights(
                    list(self._strategies.keys()),
                    regime_label=self._current_regime,
                )
                if meta_result and "weights" in meta_result:
                    meta_weights = meta_result["weights"]
                    for name, state in self._strategy_states.items():
                        state.meta_weight = meta_weights.get(name, state.current_weight)
            except Exception as e:
                logger.warning("Meta-learning failed: %s", e)
                alerts.append(f"Meta-learning error: {str(e)[:50]}")
        
        # Step 4: Apply risk adjustments
        final_weights = meta_weights
        if self.enable_risk_adaptation:
            for name, state in self._strategy_states.items():
                metrics = self._strategy_metrics.get(name)
                if metrics:
                    # Reduce weight if drawdown is high
                    if metrics.drawdown_pct > 10:
                        risk_mult = 0.5
                    elif metrics.drawdown_pct > 5:
                        risk_mult = 0.75
                    elif metrics.drawdown_pct > 2:
                        risk_mult = 0.9
                    else:
                        risk_mult = 1.0
                    
                    # Reduce weight if too many consecutive losses
                    if state.consecutive_losses > 5:
                        risk_mult *= 0.5
                    elif state.consecutive_losses > 3:
                        risk_mult *= 0.75
                    
                    state.risk_multiplier = risk_mult
                    risk_adjustments[name] = risk_mult
                    final_weights[name] = final_weights.get(name, state.current_weight) * risk_mult
        
        # Step 5: Apply lifecycle management
        if self.enable_decay_detection:
            strategy_metrics_dict = {
                name: metrics.to_dict()
                for name, metrics in self._strategy_metrics.items()
            }
            actions = self._strategy_manager.evaluate_all_strategies(
                strategy_metrics_dict,
                regime=self._current_regime,
            )
            strategy_actions = actions
            
            for action in actions:
                if action.action == "disable":
                    state = self._strategy_states.get(action.strategy_name)
                    if state:
                        state.is_enabled = False
                        final_weights[action.strategy_name] = 0.0
                        alerts.append(f"Strategy disabled: {action.strategy_name} ({action.reason})")
                elif action.action == "reduce":
                    final_weights[action.strategy_name] = action.new_weight
                elif action.action == "enable":
                    state = self._strategy_states.get(action.strategy_name)
                    if state:
                        state.is_enabled = True
                        final_weights[action.strategy_name] = action.new_weight
        
        # Step 6: Normalize and apply final weights
        total_weight = sum(max(0, w) for w in final_weights.values())
        if total_weight > 0:
            for name in self._strategy_states:
                raw_weight = max(0, final_weights.get(name, 0))
                normalized = raw_weight / total_weight
                
                # Clamp to min/max
                clamped = max(
                    self.min_strategy_allocation,
                    min(self.max_strategy_allocation, normalized),
                )
                
                state = self._strategy_states[name]
                state.adjusted_weight = regime_weights.get(name, state.current_weight)
                state.final_allocation = clamped
        
        # Calculate cycle duration
        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._last_cycle_time = time.monotonic()
        self._adaptation_count += 1
        
        return AdaptationCycleResult(
            timestamp=datetime.utcnow(),
            regime=self._current_regime,
            regime_confidence=self._regime_confidence,
            strategy_weights=self.get_strategy_weights(),
            strategy_actions=strategy_actions,
            risk_adjustments=risk_adjustments,
            total_adaptation_time_ms=duration_ms,
            alerts=alerts,
        )
    
    def get_regime_affinity(self, strategy_name: str) -> float:
        """Get the regime affinity multiplier for a strategy."""
        affinity_map = STRATEGY_REGIME_AFFINITY.get(strategy_name, {})
        return affinity_map.get(self._current_regime, 1.0)
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        return {
            "total_strategies": len(self._strategies),
            "enabled_strategies": sum(1 for s in self._strategy_states.values() if s.is_enabled),
            "current_regime": self._current_regime,
            "regime_confidence": self._regime_confidence,
            "adaptation_count": self._adaptation_count,
            "last_cycle_time": self._last_cycle_time,
            "total_allocation": sum(s.final_allocation for s in self._strategy_states.values()),
        }
    
    def reset_strategy_state(self, name: str) -> None:
        """Reset a strategy's adaptive state (e.g., after manual intervention)."""
        state = self._strategy_states.get(name)
        if state:
            state.is_enabled = True
            state.consecutive_losses = 0
            state.cooldown_until = 0.0
            state.risk_multiplier = 1.0
            logger.info("Reset adaptive state for strategy: %s", name)


# ============================================================================
# Factory Function
# ============================================================================

def create_unified_orchestrator(
    strategies: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> UnifiedStrategyOrchestrator:
    """Create a unified orchestrator with optional initial strategies."""
    orchestrator = UnifiedStrategyOrchestrator(**kwargs)
    
    if strategies:
        for name, strategy in strategies.items():
            orchestrator.register_strategy(name, strategy)
    
    return orchestrator
