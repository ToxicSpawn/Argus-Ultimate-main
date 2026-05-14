"""
Advanced Learning Orchestrator
===============================
Integrates all advanced learning systems into a unified interface:

1. Market Adaptive Parameters (0.5s learning)
2. Meta-Learning (learns how to learn)
3. Predictive Regime Detection (predicts changes before they happen)
4. Ensemble Learning (multiple strategies competing)
5. Causal Learning (understands WHY things work)

This is the brain of Argus's learning capability.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import all learning systems
from learning.market_adaptive_parameters import (
    MarketAdaptiveParameters,
    LearningConfig,
)
from learning.meta_learning_engine import MetaLearningEngine, MetaLearningConfig
from learning.predictive_regime import (
    RegimePredictor,
    PredictiveParameterAdjuster,
    create_enhanced_market_features,
)
from learning.ensemble_learning import EnsembleLearningSystem
from learning.causal_learning import CausalLearningSystem


@dataclass
class AdvancedLearningConfig:
    """Configuration for the advanced learning orchestrator."""
    # Learning intervals
    main_learning_interval: float = 0.5  # 0.5 seconds
    
    # Feature flags
    enable_meta_learning: bool = True
    enable_predictive_regime: bool = True
    enable_ensemble: bool = True
    enable_causal_learning: bool = True
    
    # Performance targets
    target_win_rate: float = 0.55
    target_profit_factor: float = 1.5
    target_trades_per_hour: float = 30


class AdvancedLearningOrchestrator:
    """
    Orchestrates all learning systems for maximum effectiveness.
    
    Learning Hierarchy:
    ┌─────────────────────────────────────────────────────┐
    │          Advanced Learning Orchestrator              │
    │  (Coordinates all learning systems at 0.5s)         │
    ├─────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
    │  │   Meta-     │  │ Predictive  │  │   Ensemble  │ │
    │  │  Learning   │  │   Regime    │  │  Learning   │ │
    │  │  (How to    │  │ (What's     │  │  (Who's     │ │
    │  │   learn)    │  │  coming)    │  │   best)     │ │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
    │         │                │                 │         │
    │  ┌──────▼──────────────────────────────────────────┐│
    │  │           Market Adaptive Parameters             ││
    │  │  (Filter thresholds, confidence, strategies)     ││
    │  └──────────────────────────────────────────────────┘│
    │         │                │                 │         │
    │  ┌──────▼──────────────────────────────────────────┐│
    │  │              Causal Learning                     ││
    │  │  (Understands WHY things work)                   ││
    │  └──────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Optional[AdvancedLearningConfig] = None):
        self.config = config or AdvancedLearningConfig()
        
        # Initialize all learning systems
        self.market_params = MarketAdaptiveParameters(LearningConfig(
            learning_interval=self.config.main_learning_interval,
            target_win_rate=self.config.target_win_rate,
            target_trades_per_hour=self.config.target_trades_per_hour,
        ))
        
        self.meta_learning = MetaLearningEngine() if self.config.enable_meta_learning else None
        self.regime_predictor = RegimePredictor() if self.config.enable_predictive_regime else None
        self.ensemble = EnsembleLearningSystem() if self.config.enable_ensemble else None
        self.causal = CausalLearningSystem() if self.config.enable_causal_learning else None
        
        # Predictive adjuster (uses regime predictor)
        self.predictive_adjuster = None
        if self.regime_predictor:
            self.predictive_adjuster = PredictiveParameterAdjuster(self.regime_predictor)
        
        # State
        self.current_regime: str = "ranging"
        self.current_parameters: Dict[str, float] = {}
        self.learning_cycles: int = 0
        self.total_learning_time_ms: float = 0.0
        
        # Performance tracking
        self.trade_history: Deque[Dict] = deque(maxlen=100)
        self.learning_evolution: Deque[Dict] = deque(maxlen=500)
        
        logger.info(
            f"AdvancedLearningOrchestrator initialized: "
            f"meta_learning={self.config.enable_meta_learning}, "
            f"predictive={self.config.enable_predictive_regime}, "
            f"ensemble={self.config.enable_ensemble}, "
            f"causal={self.config.enable_causal_learning}"
        )
    
    def update_market_state(self, 
                            prices: List[float],
                            regime: str,
                            additional_features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Update all learning systems with new market state.
        
        This is called every 0.5 seconds.
        """
        start_time = time.time()
        self.current_regime = regime
        
        results = {
            "learning_cycle": self.learning_cycles + 1,
            "regime": regime,
            "systems_updated": [],
            "pre_adjustments": {},
        }
        
        # 1. Update regime predictor
        if self.regime_predictor:
            market_features = create_enhanced_market_features(prices)
            if additional_features:
                market_features.update(additional_features)
            
            self.regime_predictor.update(regime, market_features)
            
            # Get prediction
            prediction = self.regime_predictor.get_prediction()
            if prediction:
                results["regime_prediction"] = prediction
            
            results["systems_updated"].append("regime_predictor")
        
        # 2. Get predictive pre-adjustments
        if self.predictive_adjuster:
            pre_adjustments = self.predictive_adjuster.get_pre_adjustments(
                self.current_parameters
            )
            if pre_adjustments:
                results["pre_adjustments"] = pre_adjustments
                results["systems_updated"].append("predictive_adjuster")
        
        # 3. Run main market adaptive learning
        market_learning_result = self.market_params.learn()
        if market_learning_result.get("learned"):
            results["market_learning"] = market_learning_result
            results["systems_updated"].append("market_params")
        
        # 4. Adapt meta-learning rates
        if self.meta_learning:
            lr_changes = self.meta_learning.adapt_all_learning_rates()
            results["meta_learning_rates"] = lr_changes
            results["systems_updated"].append("meta_learning")
        
        # 5. Evolve ensemble strategies (every 100 cycles)
        if self.ensemble and self.learning_cycles % 100 == 0:
            replaced = self.ensemble.evolve_strategies()
            results["ensemble_evolution"] = {
                "replaced_strategies": replaced,
                "current_strategies": len(self.ensemble.strategies),
            }
            results["systems_updated"].append("ensemble")
        
        # Update timing
        elapsed_ms = (time.time() - start_time) * 1000
        self.learning_cycles += 1
        self.total_learning_time_ms += elapsed_ms
        
        results["learning_time_ms"] = elapsed_ms
        
        # Store evolution data
        self.learning_evolution.append({
            "cycle": self.learning_cycles,
            "regime": regime,
            "time_ms": elapsed_ms,
            "systems": results["systems_updated"],
        })
        
        return results
    
    def get_final_parameters(self, 
                              base_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Get final learned parameters after all learning systems have contributed.
        
        Combines:
        1. Market adaptive parameters
        2. Predictive pre-adjustments
        3. Ensemble suggestions
        4. Causal recommendations
        """
        final_params = dict(base_params)
        adjustments_log = []
        
        # 1. Get market adaptive parameters
        filter_threshold = self.market_params.get_filter_threshold(self.current_regime)
        final_params["filter_threshold"] = filter_threshold
        
        # Confidence floors
        for signal_type in ["trend", "momentum", "mean_reversion", "breakout"]:
            floor = self.market_params.get_confidence_floor(signal_type)
            final_params[f"confidence_floor_{signal_type}"] = floor
        
        # Strategy thresholds
        for signal_type in ["trend", "momentum", "reversion", "breakout"]:
            threshold = self.market_params.get_strategy_threshold(signal_type)
            final_params[f"strategy_threshold_{signal_type}"] = threshold
        
        # Filter settings
        max_trades, min_time = self.market_params.get_filter_settings()
        final_params["max_trades_per_hour"] = max_trades
        final_params["min_time_between_trades"] = min_time
        
        # 2. Apply ensemble adjustment
        if self.ensemble:
            metrics = self.market_params.performance.get_win_rate()
            ensemble_adj, details = self.ensemble.get_ensemble_adjustment(
                filter_threshold,
                {"win_rate": metrics, "profit_factor": 1.0},
                self.current_regime
            )
            final_params["ensemble_adjustment"] = ensemble_adj
            final_params["ensemble_consensus"] = details.get("consensus", 1.0)
        
        # 3. Apply causal recommendations
        if self.causal:
            causal_suggestions = self.causal.suggest_parameter_change(final_params)
            if causal_suggestions:
                adjustments_log.append({
                    "source": "causal",
                    "adjustments": causal_suggestions,
                })
                final_params["causal_suggestions"] = causal_suggestions
        
        # Store current parameters
        self.current_parameters = final_params
        
        return {
            "parameters": final_params,
            "adjustments_log": adjustments_log,
            "regime": self.current_regime,
            "learning_cycles": self.learning_cycles,
        }
    
    def record_trade_outcome(self,
                              trade: Dict[str, Any],
                              parameters_used: Dict[str, float],
                              market_features: Dict[str, float]) -> None:
        """Record a trade outcome for all learning systems."""
        pnl = trade.get("pnl", 0.0)
        won = pnl > 0
        
        # 1. Record in market adaptive params
        self.market_params.record_trade(trade)
        
        # 2. Record in meta-learning
        if self.meta_learning:
            for param_name, param_value in parameters_used.items():
                self.meta_learning.record_parameter_outcome(
                    param_name, 0.0, pnl  # Simplified
                )
        
        # 3. Record in ensemble
        if self.ensemble:
            self.ensemble.record_trade_outcome(
                pnl, won, ["conservative", "momentum"]  # Simplified
            )
        
        # 4. Record in causal learning
        if self.causal:
            self.causal.record_trade(parameters_used, market_features, pnl)
        
        # 5. Record regime performance for meta-learning memory
        if self.meta_learning:
            self.meta_learning.record_regime_performance(
                self.current_regime, parameters_used, {"pnl": pnl, "profit_factor": 1.0 if won else 0.5}
            )
        
        self.trade_history.append(trade)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all learning systems."""
        stats = {
            "orchestrator": {
                "learning_cycles": self.learning_cycles,
                "avg_cycle_time_ms": self.total_learning_time_ms / max(self.learning_cycles, 1),
                "current_regime": self.current_regime,
                "trades_recorded": len(self.trade_history),
            },
            "market_adaptive": self.market_params.get_stats(),
        }
        
        if self.meta_learning:
            stats["meta_learning"] = self.meta_learning.get_learning_stats()
        
        if self.regime_predictor:
            stats["regime_prediction"] = self.regime_predictor.get_regime_statistics()
        
        if self.ensemble:
            stats["ensemble"] = self.ensemble.get_stats()
        
        if self.causal:
            stats["causal"] = self.causal.get_causal_insights()
        
        return stats


# Singleton
_orchestrator: Optional[AdvancedLearningOrchestrator] = None


def get_orchestrator() -> AdvancedLearningOrchestrator:
    """Get or create singleton orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AdvancedLearningOrchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset singleton (for testing)."""
    global _orchestrator
    _orchestrator = None
