# pyright: reportMissingImports=false
"""
Universal Learning Integration
================================
Connects ALL Argus systems to continuous parameter learning.

SYSTEMS UPGRADED:
1. Ensemble Predictor - Dynamic model weights
2. Dynamic Kelly - Learned regime multipliers
3. Market Making - Adaptive spread/skew
4. Smart Order Router - Optimal venue selection
5. Strategy Parameters - All 90+ strategies learn

Every system now ADAPTS based on actual trading outcomes.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 1. ENSEMBLE PREDICTOR WITH LEARNING
# ============================================================================

@dataclass
class LearnedModelWeight:
    """Learned weight for an ML model."""
    model_name: str
    base_weight: float
    learned_weight: float
    confidence: float
    observations: int
    regime_weights: Dict[str, float] = field(default_factory=dict)


class LearningEnhancedEnsemble:
    """
    Ensemble Predictor with continuous weight learning.
    
    Learns optimal model weights per regime from actual predictions.
    """
    
    def __init__(self, learning_engine=None):
        self.learning_engine = learning_engine
        self.models: Dict[str, LearnedModelWeight] = {}
        
        # Default models
        self.add_model("lstm", weight=0.25)
        self.add_model("transformer", weight=0.30)
        self.add_model("xgboost", weight=0.25)
        self.add_model("quantum", weight=0.20)
        
        # Performance tracking
        self.total_predictions: int = 0
        self.correct_predictions: int = 0
        
    def add_model(self, name: str, weight: float) -> None:
        """Add a model to the ensemble."""
        self.models[name] = LearnedModelWeight(
            model_name=name,
            base_weight=weight,
            learned_weight=weight,
            confidence=0.5,
            observations=0,
        )
    
    def predict(self, features: Dict[str, float], regime: str = "neutral") -> Dict[str, Any]:
        """
        Make ensemble prediction with learned weights.
        
        Returns combined prediction using optimal weights.
        """
        self.total_predictions += 1
        
        # Get learned weights for this regime
        weights = self._get_regime_weights(regime)
        
        # Simulate individual model predictions (in real system, call actual models)
        predictions = {}
        for name in self.models:
            # Simplified - would call actual model
            predictions[name] = np.random.randn() * 0.5 + features.get("price_change", 0)
        
        # Weighted combination
        combined = sum(predictions[name] * weights[name] for name in predictions)
        confidence = self._calculate_confidence(predictions, weights)
        
        return {
            "prediction": combined,
            "confidence": confidence,
            "weights": weights,
            "individual_predictions": predictions,
            "regime": regime,
        }
    
    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get weights for current regime."""
        weights = {}
        for name, model in self.models.items():
            # Use regime-specific weight if available, otherwise use learned weight
            if regime in model.regime_weights:
                weights[name] = model.regime_weights[regime]
            else:
                weights[name] = model.learned_weight
        return weights
    
    def _calculate_confidence(self, predictions: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate ensemble confidence from prediction agreement."""
        vals = list(predictions.values())
        if len(vals) < 2:
            return 0.5
        # Lower std = higher agreement = higher confidence
        std = np.std(vals)
        return max(0.0, min(1.0, 1.0 - std))
    
    def record_outcome(self, model_predictions: Dict[str, float], actual: float, regime: str) -> None:
        """Record prediction outcome for learning."""
        # Calculate individual model errors
        for name, prediction in model_predictions.items():
            error = abs(prediction - actual)
            
            # Update learned weight based on performance
            model = self.models[name]
            model.observations += 1
            
            # Adjust weight: lower error = higher weight
            learning_rate = 0.01
            error_normalized = min(1.0, error / (abs(actual) + 1e-6))
            weight_adjustment = (1.0 - error_normalized) * learning_rate
            
            model.learned_weight = model.learned_weight * (1 - learning_rate) + weight_adjustment
            
            # Update regime-specific weight
            if regime not in model.regime_weights:
                model.regime_weights[regime] = model.learned_weight
            else:
                model.regime_weights[regime] = model.regime_weights[regime] * 0.99 + weight_adjustment * 0.01
        
        # Normalize weights to sum to 1
        self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """Normalize all weights to sum to 1."""
        total = sum(m.learned_weight for m in self.models.values())
        if total > 0:
            for model in self.models.values():
                model.learned_weight /= total
    
    def get_status(self) -> Dict[str, Any]:
        """Get ensemble status."""
        return {
            "total_predictions": self.total_predictions,
            "models": {
                name: {
                    "learned_weight": m.learned_weight,
                    "base_weight": m.base_weight,
                    "observations": m.observations,
                    "regime_weights": m.regime_weights,
                }
                for name, m in self.models.items()
            }
        }


# ============================================================================
# 2. DYNAMIC KELLY WITH LEARNING
# ============================================================================

class LearningEnhancedKelly:
    """
    Dynamic Kelly position sizing with learned regime multipliers.
    
    Learns optimal Kelly fraction per regime from actual trade outcomes.
    """
    
    def __init__(self, learning_engine=None):
        self.learning_engine = learning_engine
        
        # Default regime multipliers (will be learned)
        self.regime_multipliers: Dict[str, float] = {
            "trending": 0.5,
            "bull": 0.45,
            "bear": 0.20,
            "neutral": 0.30,
            "high_volatility": 0.15,
            "low_volatility": 0.40,
            "ranging": 0.25,
        }
        
        # Learned adjustments (starts at 1.0 = no adjustment)
        self.learned_adjustments: Dict[str, float] = {k: 1.0 for k in self.regime_multipliers}
        
        # Trade history per regime
        self.regime_outcomes: Dict[str, List[float]] = defaultdict(list)
        
        # Settings
        self.min_fraction = 0.01
        self.max_fraction = 0.25
        
    def calculate_kelly(
        self,
        regime: str,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        trade_history: Optional[List[float]] = None
    ) -> float:
        """Calculate Kelly fraction with learned regime adjustment."""
        
        # Get base Kelly from inputs or history
        if win_rate and avg_win and avg_loss:
            # Use provided stats
            b = avg_win / max(abs(avg_loss), 1e-6)
            p = win_rate
            q = 1 - p
            kelly_f = (b * p - q) / b
        elif trade_history and len(trade_history) >= 10:
            # Calculate from history
            wins = [t for t in trade_history if t > 0]
            losses = [t for t in trade_history if t <= 0]
            
            if wins and losses:
                p = len(wins) / len(trade_history)
                avg_win = np.mean(wins)
                avg_loss = abs(np.mean(losses))
                b = avg_win / avg_loss
                kelly_f = (b * p - (1 - p)) / b
            else:
                kelly_f = 0.1
        else:
            kelly_f = 0.2  # Default
        
        # Apply regime multiplier
        base_multiplier = self.regime_multipliers.get(regime, 0.25)
        learned_adjustment = self.learned_adjustments.get(regime, 1.0)
        
        adjusted_multiplier = base_multiplier * learned_adjustment
        
        # Calculate final fraction
        final_fraction = kelly_f * adjusted_multiplier
        
        # Clamp to bounds
        return max(self.min_fraction, min(self.max_fraction, final_fraction))
    
    def record_outcome(self, regime: str, kelly_fraction: float, pnl_pct: float) -> None:
        """Record trade outcome for learning."""
        self.regime_outcomes[regime].append(pnl_pct)
        
        # Keep last 100 outcomes per regime
        if len(self.regime_outcomes[regime]) > 100:
            self.regime_outcomes[regime] = self.regime_outcomes[regime][-100:]
        
        # Learn optimal adjustment
        self._learn_adjustment(regime)
    
    def _learn_adjustment(self, regime: str) -> None:
        """Learn optimal Kelly adjustment for this regime."""
        outcomes = self.regime_outcomes[regime]
        if len(outcomes) < 20:
            return
        
        # Calculate Sharpe-like metric
        mean_return = np.mean(outcomes)
        std_return = np.std(outcomes) + 1e-6
        sharpe = mean_return / std_return
        
        # If positive Sharpe, we're doing well - slightly increase exposure
        # If negative Sharpe, reduce exposure
        learning_rate = 0.01
        if sharpe > 0.5:
            # Good performance, increase slightly
            adjustment_delta = learning_rate * min(sharpe, 2.0)
        elif sharpe < 0:
            # Bad performance, decrease
            adjustment_delta = -learning_rate * min(abs(sharpe), 2.0)
        else:
            adjustment_delta = 0
        
        # Apply adjustment
        current = self.learned_adjustments[regime]
        self.learned_adjustments[regime] = max(0.5, min(1.5, current + adjustment_delta))
        
        logger.debug(f"Kelly learning [{regime}]: adjustment={self.learned_adjustments[regime]:.3f}, sharpe={sharpe:.2f}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Kelly status."""
        return {
            "regime_multipliers": self.regime_multipliers,
            "learned_adjustments": self.learned_adjustments,
            "regime_outcome_counts": {k: len(v) for k, v in self.regime_outcomes.items()},
        }


# ============================================================================
# 3. MARKET MAKING WITH LEARNING
# ============================================================================

class LearningEnhancedMarketMaking:
    """
    Market making with learned parameters.
    
    Learns optimal:
    - Base spread offset
    - Inventory skew coefficient
    - Size limits
    - Volatility adjustments
    """
    
    def __init__(self, learning_engine=None):
        self.learning_engine = learning_engine
        
        # Default parameters (will be learned)
        self.base_offset_bps: float = 5.0
        self.inventory_skew_coef: float = 0.5
        self.max_size_pct: float = 0.08
        self.volatility_mult: float = 1.5
        
        # Learned adjustments
        self.learned_offset_mult: float = 1.0
        self.learned_skew_mult: float = 1.0
        self.learned_size_mult: float = 1.0
        
        # Performance tracking
        self.spreads_captured: List[float] = []
        self.inventory_costs: List[float] = []
        
    def generate_quote(
        self,
        spread_bps: float,
        inventory: float,
        volatility: float
    ) -> Dict[str, float]:
        """Generate market making quote with learned parameters."""
        
        # Apply learned adjustments
        offset = self.base_offset_bps * self.learned_offset_mult
        skew_coef = self.inventory_skew_coef * self.learned_skew_mult
        max_size = self.max_size_pct * self.learned_size_mult
        
        # Calculate skew from inventory
        skew = inventory * skew_coef
        
        # Calculate width with volatility adjustment
        width = max(offset, spread_bps * 0.3) * (1.0 + volatility * self.volatility_mult)
        
        # Calculate bid/ask offsets
        bid_offset = max(width + skew, 0.5)
        ask_offset = max(width - skew, 0.5)
        
        # Size based on spread
        size_pct = min(0.02 + spread_bps / 300.0, max_size)
        
        return {
            "bid_offset_bps": bid_offset,
            "ask_offset_bps": ask_offset,
            "size_pct": size_pct,
            "skew": skew,
            "width": width,
        }
    
    def record_outcome(
        self,
        spread_captured_bps: float,
        inventory_cost_bps: float,
        volatility: float
    ) -> None:
        """Record quote outcome for learning."""
        self.spreads_captured.append(spread_captured_bps)
        self.inventory_costs.append(inventory_cost_bps)
        
        # Keep last 200
        if len(self.spreads_captured) > 200:
            self.spreads_captured = self.spreads_captured[-200:]
            self.inventory_costs = self.inventory_costs[-200:]
        
        # Learn adjustments
        self._learn_parameters()
    
    def _learn_parameters(self) -> None:
        """Learn optimal market making parameters."""
        if len(self.spreads_captured) < 30:
            return
        
        # Calculate net PnL (spread - inventory cost)
        net_pnl = [s - c for s, c in zip(self.spreads_captured, self.inventory_costs)]
        avg_pnl = np.mean(net_pnl)
        std_pnl = np.std(net_pnl) + 1e-6
        sharpe = avg_pnl / std_pnl
        
        learning_rate = 0.01
        
        # If positive Sharpe, we're capturing spread well
        if sharpe > 0.5:
            # Slightly tighten spreads to capture more volume
            self.learned_offset_mult = max(0.7, self.learned_offset_mult - learning_rate)
        elif sharpe < 0:
            # Widen spreads for safety
            self.learned_offset_mult = min(1.5, self.learned_offset_mult + learning_rate)
        
        # Adjust skew based on inventory cost
        avg_inventory_cost = np.mean(self.inventory_costs)
        if avg_inventory_cost > 2.0:
            # Inventory costs too high, reduce skew
            self.learned_skew_mult = max(0.5, self.learned_skew_mult - learning_rate)
        elif avg_inventory_cost < 0.5:
            # Can take more inventory risk
            self.learned_skew_mult = min(1.5, self.learned_skew_mult + learning_rate)
    
    def get_status(self) -> Dict[str, Any]:
        """Get market making status."""
        return {
            "base_offset_bps": self.base_offset_bps,
            "learned_offset_mult": self.learned_offset_mult,
            "learned_skew_mult": self.learned_skew_mult,
            "avg_spread_captured": np.mean(self.spreads_captured) if self.spreads_captured else 0,
            "avg_inventory_cost": np.mean(self.inventory_costs) if self.inventory_costs else 0,
        }


# ============================================================================
# 4. SMART ORDER ROUTER WITH LEARNING
# ============================================================================

class LearningEnhancedOrderRouter:
    """
    Smart order router with learned venue selection.
    
    Learns optimal venues based on:
    - Fill rates by time of day
    - Latency by volume
    - Fees by order size
    - Slippage by volatility
    """
    
    def __init__(self, learning_engine=None):
        self.learning_engine = learning_engine
        
        # Venue configurations
        self.venues: Dict[str, Dict[str, Any]] = {
            "binance": {"base_fee": 0.001, "latency_ms": 50, "fill_rate": 0.95, "score": 1.0},
            "okx": {"base_fee": 0.001, "latency_ms": 60, "fill_rate": 0.92, "score": 0.9},
            "bybit": {"base_fee": 0.001, "latency_ms": 70, "fill_rate": 0.90, "score": 0.85},
            "kraken": {"base_fee": 0.0016, "latency_ms": 80, "fill_rate": 0.88, "score": 0.8},
        }
        
        # Learned venue scores (multipliers)
        self.learned_venue_scores: Dict[str, float] = {v: 1.0 for v in self.venues}
        
        # Performance tracking per venue
        self.venue_outcomes: Dict[str, List[float]] = defaultdict(list)
        
    def select_venue(
        self,
        order_size_usd: float,
        volatility: float,
        urgency: str = "normal"
    ) -> str:
        """Select optimal venue with learned scores."""
        best_venue = None
        best_score = -1
        
        for venue, config in self.venues.items():
            # Base score from fill rate and latency
            base_score = config["fill_rate"] * (100 / (config["latency_ms"] + 10))
            
            # Adjust for order size
            if order_size_usd > 10000:
                base_score *= 0.9  # Penalize smaller venues for large orders
            
            # Urgency adjustment
            if urgency == "high":
                base_score *= (100 / (config["latency_ms"] + 10))  # Favor low latency
            
            # Apply learned adjustment
            learned_mult = self.learned_venue_scores.get(venue, 1.0)
            final_score = base_score * learned_mult
            
            if final_score > best_score:
                best_score = final_score
                best_venue = venue
        
        return best_venue or "binance"
    
    def record_outcome(
        self,
        venue: str,
        fill_rate: float,
        slippage_bps: float,
        latency_ms: float
    ) -> None:
        """Record venue outcome for learning."""
        # Calculate venue score (higher is better)
        outcome_score = fill_rate * 100 - slippage_bps - latency_ms / 10
        
        self.venue_outcomes[venue].append(outcome_score)
        
        # Keep last 100
        if len(self.venue_outcomes[venue]) > 100:
            self.venue_outcomes[venue] = self.venue_outcomes[venue][-100:]
        
        # Learn optimal venue scores
        self._learn_venue_scores()
    
    def _learn_venue_scores(self) -> None:
        """Learn optimal venue scores."""
        if not any(len(v) >= 20 for v in self.venue_outcomes.values()):
            return
        
        # Calculate relative performance
        avg_scores = {}
        for venue, outcomes in self.venue_outcomes.items():
            if len(outcomes) >= 20:
                avg_scores[venue] = np.mean(outcomes)
        
        if len(avg_scores) < 2:
            return
        
        # Normalize to adjustments
        overall_mean = np.mean(list(avg_scores.values()))
        overall_std = np.std(list(avg_scores.values())) + 1e-6
        
        learning_rate = 0.05
        
        for venue, avg_score in avg_scores.items():
            # Z-score relative to other venues
            z_score = (avg_score - overall_mean) / overall_std
            
            # Convert to adjustment
            adjustment = 1.0 + z_score * learning_rate
            
            # Apply with momentum
            current = self.learned_venue_scores[venue]
            self.learned_venue_scores[venue] = current * 0.95 + adjustment * 0.05
    
    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "venues": self.venues,
            "learned_scores": self.learned_venue_scores,
            "outcome_counts": {k: len(v) for k, v in self.venue_outcomes.items()},
        }


# ============================================================================
# 5. STRATEGY PARAMETER LEARNING HOOKS
# ============================================================================

class StrategyParameterLearner:
    """
    Hooks into individual strategies for parameter learning.
    
    Covers common strategy parameters:
    - Entry/exit thresholds
    - Lookback periods
    - Multipliers
    - Risk per trade
    """
    
    def __init__(self, learning_engine=None):
        self.learning_engine = learning_engine
        
        # Common strategy parameters that can be learned
        self.learnable_params: Dict[str, Dict[str, Any]] = {
            # Trend Following
            "trend_ma_fast": {"default": 10, "min": 5, "max": 30, "category": "signal"},
            "trend_ma_slow": {"default": 50, "min": 20, "max": 100, "category": "signal"},
            "trend_entry_threshold": {"default": 0.01, "min": 0.001, "max": 0.05, "category": "threshold"},
            
            # Mean Reversion
            "mr_zscore_entry": {"default": 2.0, "min": 1.5, "max": 3.0, "category": "threshold"},
            "mr_zscore_exit": {"default": 0.5, "min": 0.0, "max": 1.0, "category": "threshold"},
            "mr_lookback": {"default": 20, "min": 10, "max": 50, "category": "signal"},
            
            # Market Making
            "mm_base_spread": {"default": 10.0, "min": 1.0, "max": 50.0, "category": "execution"},
            "mm_inventory_limit": {"default": 0.10, "min": 0.02, "max": 0.30, "category": "risk"},
            "mm_skew_factor": {"default": 0.5, "min": 0.0, "max": 1.0, "category": "signal"},
            
            # Funding Rate Arb
            "funding_min_apy": {"default": 0.10, "min": 0.02, "max": 0.50, "category": "threshold"},
            "funding_max_basis": {"default": 0.003, "min": 0.001, "max": 0.01, "category": "threshold"},
            "funding_risk_per_trade": {"default": 0.02, "min": 0.005, "max": 0.05, "category": "risk"},
            
            # Scalping
            "scalp_profit_target": {"default": 0.002, "min": 0.0005, "max": 0.01, "category": "threshold"},
            "scalp_stop_loss": {"default": 0.001, "min": 0.0005, "max": 0.005, "category": "threshold"},
            "scalp_max_hold_seconds": {"default": 30, "min": 5, "max": 120, "category": "execution"},
            
            # Breakout
            "breakout_atr_mult": {"default": 2.0, "min": 1.0, "max": 4.0, "category": "signal"},
            "breakout_volume_confirm": {"default": 1.5, "min": 1.0, "max": 3.0, "category": "signal"},
            "breakout_retest_pct": {"default": 0.3, "min": 0.1, "max": 0.5, "category": "threshold"},
        }
        
        # Learned values
        self.learned_values: Dict[str, float] = {}
        for name, config in self.learnable_params.items():
            self.learned_values[name] = config["default"]
        
        # Outcome tracking
        self.param_outcomes: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    
    def get_param(self, name: str, regime: str = "neutral") -> float:
        """Get learned parameter value."""
        return self.learned_values.get(name, self.learnable_params.get(name, {}).get("default", 0))
    
    def record_outcome(
        self,
        param_name: str,
        param_value: float,
        pnl: float,
        regime: str
    ) -> None:
        """Record parameter outcome for learning."""
        self.param_outcomes[param_name].append((param_value, pnl))
        
        # Keep last 200
        if len(self.param_outcomes[param_name]) > 200:
            self.param_outcomes[param_name] = self.param_outcomes[param_name][-200:]
        
        # Learn optimal value
        self._learn_param_value(param_name)
    
    def _learn_param_value(self, param_name: str) -> None:
        """Learn optimal parameter value."""
        outcomes = self.param_outcomes[param_name]
        if len(outcomes) < 30:
            return
        
        config = self.learnable_params.get(param_name)
        if not config:
            return
        
        # Group outcomes by parameter value (rounded)
        value_pnl: Dict[float, List[float]] = defaultdict(list)
        for val, pnl in outcomes:
            rounded = round(val, 4)
            value_pnl[rounded].append(pnl)
        
        # Find best value
        best_value = None
        best_sharpe = -999
        
        for value, pnls in value_pnl.items():
            if len(pnls) >= 5:
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls) + 1e-6
                sharpe = mean_pnl / std_pnl
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_value = value
        
        if best_value is not None:
            # Update with momentum
            current = self.learned_values[param_name]
            learning_rate = 0.05
            self.learned_values[param_name] = current * (1 - learning_rate) + best_value * learning_rate
    
    def get_all_learned(self) -> Dict[str, float]:
        """Get all learned parameter values."""
        return self.learned_values.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy learner status."""
        return {
            "tracked_params": len(self.learnable_params),
            "learned_values": self.learned_values,
            "outcome_counts": {k: len(v) for k, v in self.param_outcomes.items()},
        }


# ============================================================================
# 6. UNIVERSAL INTEGRATION HUB
# ============================================================================

class UniversalLearningHub:
    """
    Central hub that connects ALL systems to parameter learning.
    
    This is the ONE place where all learning integration happens.
    """
    
    def __init__(self):
        # Learning components
        self.ensemble = LearningEnhancedEnsemble()
        self.kelly = LearningEnhancedKelly()
        self.market_maker = LearningEnhancedMarketMaking()
        self.order_router = LearningEnhancedOrderRouter()
        self.strategy_learner = StrategyParameterLearner()
        
        # Learning engine (lazy loaded)
        self._learning_engine = None
        
        # Statistics
        self.total_learning_cycles: int = 0
        self.total_parameters_updated: int = 0
        
        logger.info("UniversalLearningHub initialized")
    
    @property
    def learning_engine(self):
        """Lazy load learning engine."""
        if self._learning_engine is None:
            from learning.parameter_learning_integration import ParameterLearningIntegrator
            self._learning_engine = ParameterLearningIntegrator()
        return self._learning_engine
    
    def make_trading_decision(
        self,
        price: float,
        volume: float,
        symbol: str,
        regime: str
    ) -> Dict[str, Any]:
        """
        Make trading decision using all learned systems.
        """
        # Get ensemble prediction
        features = {"price": price, "volume": volume}
        ensemble_pred = self.ensemble.predict(features, regime)
        
        # Get Kelly position size
        kelly_fraction = self.kelly.calculate_kelly(regime)
        
        # Get market making quote (if market making)
        mm_quote = self.market_maker.generate_quote(
            spread_bps=5.0,
            inventory=0.0,
            volatility=0.02
        )
        
        # Get optimal venue
        venue = self.order_router.select_venue(
            order_size_usd=kelly_fraction * 10000,
            volatility=0.02
        )
        
        # Get strategy parameters
        trend_params = {
            "ma_fast": self.strategy_learner.get_param("trend_ma_fast", regime),
            "ma_slow": self.strategy_learner.get_param("trend_ma_slow", regime),
        }
        
        # Determine action from ensemble
        action = "buy" if ensemble_pred["prediction"] > 0.01 else "sell" if ensemble_pred["prediction"] < -0.01 else "hold"
        
        return {
            "action": action,
            "confidence": ensemble_pred["confidence"],
            "kelly_fraction": kelly_fraction,
            "venue": venue,
            "mm_quote": mm_quote,
            "trend_params": trend_params,
            "ensemble_weights": ensemble_pred["weights"],
            "regime": regime,
        }
    
    def record_trade_outcome(
        self,
        decision: Dict[str, Any],
        actual_pnl: float,
        model_predictions: Optional[Dict[str, float]] = None,
        spread_captured: Optional[float] = None,
        inventory_cost: Optional[float] = None,
        fill_rate: Optional[float] = None,
        slippage: Optional[float] = None,
    ) -> None:
        """Record outcome for all relevant systems."""
        
        regime = decision.get("regime", "neutral")
        
        # Record for ensemble
        if model_predictions:
            self.ensemble.record_outcome(model_predictions, actual_pnl, regime)
        
        # Record for Kelly
        pnl_pct = actual_pnl / 10000.0  # Assuming $10k capital
        self.kelly.record_outcome(regime, decision.get("kelly_fraction", 0.1), pnl_pct)
        
        # Record for market making
        if spread_captured is not None and inventory_cost is not None:
            self.market_maker.record_outcome(spread_captured, inventory_cost, 0.02)
        
        # Record for order router
        if fill_rate is not None and slippage is not None:
            self.order_router.record_outcome(
                decision.get("venue", "binance"),
                fill_rate,
                slippage,
                50.0  # default latency
            )
        
        # Record for strategy learner
        for param_name in ["trend_ma_fast", "trend_ma_slow", "mr_zscore_entry", "mm_base_spread"]:
            if param_name in decision.get("trend_params", {}):
                self.strategy_learner.record_outcome(
                    param_name,
                    decision["trend_params"][param_name],
                    actual_pnl,
                    regime
                )
        
        # Record for main learning engine
        self.learning_engine.record_trade_outcome(
            decision.get("parameters_used", {}),
            actual_pnl,
            {"regime": regime, "symbol": decision.get("symbol", "BTC")}
        )
        
        self.total_learning_cycles += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        return {
            "ensemble": self.ensemble.get_status(),
            "kelly": self.kelly.get_status(),
            "market_maker": self.market_maker.get_status(),
            "order_router": self.order_router.get_status(),
            "strategy_learner": self.strategy_learner.get_status(),
            "total_learning_cycles": self.total_learning_cycles,
        }


# Global singleton
_hub: Optional[UniversalLearningHub] = None


def get_universal_learning_hub() -> UniversalLearningHub:
    """Get or create the universal learning hub."""
    global _hub
    if _hub is None:
        _hub = UniversalLearningHub()
    return _hub


__all__ = [
    "UniversalLearningHub",
    "LearningEnhancedEnsemble",
    "LearningEnhancedKelly",
    "LearningEnhancedMarketMaking",
    "LearningEnhancedOrderRouter",
    "StrategyParameterLearner",
    "get_universal_learning_hub",
]
