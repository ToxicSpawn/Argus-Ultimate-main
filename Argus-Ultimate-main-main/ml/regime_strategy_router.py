"""
Regime-Strategy Router — maps market regimes to optimal strategy allocations.

This module connects regime detection to strategy selection:
  - Maps regime predictions to strategy weight allocations
  - Adapts weights based on regime confidence
  - Tracks regime-specific strategy performance
  - Provides regime-aware position sizing

Usage:
    router = RegimeStrategyRouter()
    
    # Get strategy weights for current regime
    regime = regime_classifier.predict(prices)
    weights = router.get_strategy_weights(regime)
    # weights = {"trend_following": 0.6, "momentum": 0.4}
    
    # Update with actual performance
    router.update_performance("trend_following", regime, pnl=0.02)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.trading_decision_controls import ConfidenceTradeGate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

REGIMES = ["TREND_UP", "TREND_DOWN", "RANGING", "VOLATILE", "CRISIS"]

# Default strategy allocations per regime
DEFAULT_REGIME_STRATEGY_MAP: Dict[str, Dict[str, float]] = {
    "TREND_UP": {
        "trend_following": 0.40,
        "momentum": 0.30,
        "breakout": 0.20,
        "mean_reversion": 0.10,
    },
    "TREND_DOWN": {
        "trend_following": 0.30,
        "momentum": 0.20,
        "mean_reversion": 0.20,
        "volatility_arb": 0.30,
    },
    "RANGING": {
        "mean_reversion": 0.40,
        "grid_trader": 0.30,
        "market_maker": 0.20,
        "scalping": 0.10,
    },
    "VOLATILE": {
        "volatility_arb": 0.35,
        "options_vol_arb": 0.25,
        "mean_reversion": 0.20,
        "cash": 0.20,
    },
    "CRISIS": {
        "cash": 0.60,
        "volatility_arb": 0.20,
        "mean_reversion": 0.20,
    },
}


@dataclass
class RegimeStrategyWeights:
    """Strategy weights for a given regime."""
    regime: str
    confidence: float
    weights: Dict[str, float]
    timestamp: datetime
    method: str  # "default", "learned", "adaptive"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            "weights": self.weights,
            "timestamp": self.timestamp.isoformat(),
            "method": self.method,
            "top_strategies": sorted(
                self.weights.items(), key=lambda x: x[1], reverse=True
            )[:3],
        }
    
    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on regime confidence."""
        # Lower confidence = smaller positions
        # Crisis regime = smaller positions regardless of confidence
        base = self.confidence
        
        if self.regime == "CRISIS":
            base *= 0.5
        elif self.regime == "VOLATILE":
            base *= 0.75
        
        return float(np.clip(base, 0.1, 1.0))


@dataclass
class StrategyPerformance:
    """Track strategy performance per regime."""
    strategy_name: str
    regime: str
    n_trades: int = 0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.5  # Default neutral
        return self.wins / self.n_trades
    
    @property
    def avg_pnl(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return self.total_pnl / self.n_trades
    
    def update(self, pnl: float) -> None:
        """Update with new trade result."""
        self.n_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.last_updated = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "regime": self.regime,
            "n_trades": self.n_trades,
            "total_pnl": round(self.total_pnl, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_pnl": round(self.avg_pnl, 6),
        }


class RegimeStrategyRouter:
    """
    Maps market regimes to optimal strategy allocations.
    
    Features:
    - Default regime→strategy mappings
    - Adaptive weight adjustment based on historical performance
    - Confidence-adjusted position sizing
    - Performance tracking per regime/strategy pair
    
    Usage:
        router = RegimeStrategyRouter()
        
        # Get weights for current regime
        weights = router.get_strategy_weights("TREND_UP", confidence=0.8)
        
        # Update with trade results
        router.update_performance("trend_following", "TREND_UP", pnl=0.02)
        
        # Get position multiplier
        multiplier = router.get_position_multiplier("TREND_UP", confidence=0.8)
    """
    
    def __init__(
        self,
        regime_map: Optional[Dict[str, Dict[str, float]]] = None,
        adaptive: bool = True,
        learning_rate: float = 0.1,
        min_trades_for_adaptation: int = 10,
    ):
        self.regime_map = regime_map or DEFAULT_REGIME_STRATEGY_MAP.copy()
        self.adaptive = adaptive
        self.learning_rate = learning_rate
        self.min_trades_for_adaptation = min_trades_for_adaptation
        
        # Performance tracking: {(strategy, regime): StrategyPerformance}
        self._performance: Dict[Tuple[str, str], StrategyPerformance] = {}
        
        # History
        self._weight_history: List[RegimeStrategyWeights] = []
        
        logger.info("RegimeStrategyRouter initialized: %d regimes, adaptive=%s",
                    len(self.regime_map), adaptive)
    
    def get_strategy_weights(
        self,
        regime: str,
        confidence: float = 1.0,
        available_strategies: Optional[List[str]] = None,
    ) -> RegimeStrategyWeights:
        """
        Get strategy weights for a given regime.
        
        Args:
            regime: Market regime (TREND_UP, TREND_DOWN, RANGING, VOLATILE, CRISIS)
            confidence: Regime prediction confidence [0, 1]
            available_strategies: Optional filter to only include available strategies
            
        Returns:
            RegimeStrategyWeights with normalized weights
        """
        # Get base weights
        if regime in self.regime_map:
            base_weights = self.regime_map[regime].copy()
            method = "default"
        else:
            # Unknown regime - use RANGING as fallback
            logger.warning("Unknown regime: %s, using RANGING fallback", regime)
            base_weights = self.regime_map.get("RANGING", {"cash": 1.0}).copy()
            method = "fallback"
        
        # Apply adaptive adjustments if enabled
        if self.adaptive and self._has_enough_data(regime):
            base_weights = self._apply_adaptive_weights(base_weights, regime)
            method = "adaptive"
        
        # Filter to available strategies
        if available_strategies:
            base_weights = {
                k: v for k, v in base_weights.items()
                if k in available_strategies
            }
        
        # Normalize weights
        total = sum(base_weights.values())
        if total > 0:
            normalized = {k: v / total for k, v in base_weights.items()}
        else:
            normalized = {"cash": 1.0}
        
        result = RegimeStrategyWeights(
            regime=regime,
            confidence=confidence,
            weights=normalized,
            timestamp=datetime.now(timezone.utc),
            method=method,
        )
        
        self._weight_history.append(result)
        return result
    
    def get_position_multiplier(
        self,
        regime: str,
        confidence: float,
    ) -> float:
        """
        Get position size multiplier based on regime and confidence.
        
        Returns:
            Float in [0.1, 1.0] to scale position sizes
        """
        weights = self.get_strategy_weights(regime, confidence)
        return weights.get_position_multiplier()
    
    def update_performance(
        self,
        strategy_name: str,
        regime: str,
        pnl: float,
    ) -> None:
        """
        Update strategy performance for a regime.
        
        Args:
            strategy_name: Name of the strategy
            regime: Regime during the trade
            pnl: Profit/loss (positive = win, negative = loss)
        """
        key = (strategy_name, regime)
        
        if key not in self._performance:
            self._performance[key] = StrategyPerformance(
                strategy_name=strategy_name,
                regime=regime,
            )
        
        self._performance[key].update(pnl)
        
        logger.debug("Performance update: %s in %s: pnl=%.4f (n=%d)",
                     strategy_name, regime, pnl, self._performance[key].n_trades)
    
    def get_regime_performance(
        self,
        regime: Optional[str] = None,
    ) -> List[StrategyPerformance]:
        """Get performance data, optionally filtered by regime."""
        results = list(self._performance.values())
        
        if regime:
            results = [p for p in results if p.regime == regime]
        
        return results
    
    def get_best_strategy(self, regime: str) -> Optional[str]:
        """Get the best performing strategy for a regime."""
        regime_perf = self.get_regime_performance(regime)
        
        if not regime_perf:
            return None
        
        # Filter to strategies with enough data
        valid = [p for p in regime_perf if p.n_trades >= self.min_trades_for_adaptation]
        
        if not valid:
            return None
        
        best = max(valid, key=lambda p: p.avg_pnl)
        return best.strategy_name
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "n_regimes": len(self.regime_map),
            "regimes": list(self.regime_map.keys()),
            "adaptive": self.adaptive,
            "n_performance_records": len(self._performance),
            "n_weight_history": len(self._weight_history),
            "strategies_tracked": list(set(
                k[0] for k in self._performance.keys()
            )),
        }
    
    def _has_enough_data(self, regime: str) -> bool:
        """Check if we have enough performance data for adaptation."""
        regime_perf = self.get_regime_performance(regime)
        total_trades = sum(p.n_trades for p in regime_perf)
        return total_trades >= self.min_trades_for_adaptation
    
    def _apply_adaptive_weights(
        self,
        base_weights: Dict[str, float],
        regime: str,
    ) -> Dict[str, float]:
        """Adjust weights based on historical performance."""
        adapted = base_weights.copy()
        
        for strategy_name in base_weights.keys():
            key = (strategy_name, regime)
            
            if key in self._performance:
                perf = self._performance[key]
                
                if perf.n_trades >= self.min_trades_for_adaptation:
                    # Adjust weight based on average PnL
                    adjustment = perf.avg_pnl * self.learning_rate
                    
                    # Apply adjustment (clip to reasonable range)
                    adapted[strategy_name] = max(
                        0.01,  # Minimum 1% weight
                        adapted[strategy_name] + adjustment
                    )
        
        # Re-normalize
        total = sum(adapted.values())
        if total > 0:
            adapted = {k: v / total for k, v in adapted.items()}
        
        return adapted


# ---------------------------------------------------------------------------
# ML Strategy Bridge — connects ML predictions to strategy execution
# ---------------------------------------------------------------------------

class MLStrategyBridge:
    """
    Bridge between ML predictions and strategy execution.
    
    Provides a unified interface for:
    - Getting regime-based strategy allocations
    - Adjusting position sizes based on ML confidence
    - Fusing multiple ML signals into trading decisions
    
    Usage:
        bridge = MLStrategyBridge()
        
        # Get trading decision from ML signals
        decision = bridge.make_decision(
            regime="TREND_UP",
            regime_confidence=0.85,
            price_prediction={"direction": "up", "confidence": 0.7},
            sentiment_score=0.6,
        )
        
        # decision = {
        #     "action": "buy",
        #     "strategies": {"trend_following": 0.6, "momentum": 0.4},
        #     "position_multiplier": 0.85,
        #     "confidence": 0.75,
        # }
    """
    
    def __init__(self, min_trade_confidence: float = 0.0):
        self.router = RegimeStrategyRouter()
        self.min_trade_confidence = float(np.clip(min_trade_confidence, 0.0, 1.0))
        self._trade_gate = ConfidenceTradeGate(min_confidence=self.min_trade_confidence)
        
        # Signal weights for fusion
        self._signal_weights = {
            "regime": 0.35,
            "price_prediction": 0.35,
            "sentiment": 0.15,
            "orderbook": 0.15,
        }
    
    def make_decision(
        self,
        regime: str,
        regime_confidence: float,
        price_prediction: Optional[Dict[str, Any]] = None,
        sentiment_score: Optional[float] = None,
        orderbook_imbalance: Optional[float] = None,
        available_strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a unified trading decision from multiple ML signals.
        
        Args:
            regime: Current market regime
            regime_confidence: Regime prediction confidence
            price_prediction: Dict with "direction" and "confidence"
            sentiment_score: Sentiment in [-1, 1]
            orderbook_imbalance: Orderbook imbalance in [-1, 1]
            available_strategies: Optional list of available strategies
            
        Returns:
            Dict with action, strategies, position_multiplier, confidence
        """
        # Get regime-based strategy weights
        regime_weights = self.router.get_strategy_weights(
            regime, regime_confidence, available_strategies
        )
        
        # Compute signal confidences
        signals = []
        
        # Regime signal
        signals.append(("regime", regime_confidence if regime != "CRISIS" else 0.0))
        
        # Price prediction signal
        if price_prediction:
            price_conf = price_prediction.get("confidence", 0.5)
            signals.append(("price_prediction", price_conf))
        
        # Sentiment signal
        if sentiment_score is not None:
            sentiment_conf = (sentiment_score + 1) / 2  # Map [-1,1] to [0,1]
            signals.append(("sentiment", sentiment_conf))
        
        # Orderbook signal
        if orderbook_imbalance is not None:
            orderbook_conf = abs(orderbook_imbalance)
            signals.append(("orderbook", orderbook_conf))
        
        # Weighted average confidence
        total_weight = sum(
            self._signal_weights.get(name, 0.1) for name, _ in signals
        )
        weighted_confidence = sum(
            conf * self._signal_weights.get(name, 0.1)
            for name, conf in signals
        ) / max(total_weight, 1e-9)
        
        # Determine action
        action = self._determine_action(
            regime, price_prediction, sentiment_score, orderbook_imbalance
        )
        
        # Position multiplier
        position_multiplier = regime_weights.get_position_multiplier()
        gate_decision = self._trade_gate.evaluate(action, float(weighted_confidence))
        if self.min_trade_confidence > 0.0 and not gate_decision.should_trade:
            action = gate_decision.action
            position_multiplier = 0.0
        elif self.min_trade_confidence > 0.0:
            position_multiplier *= gate_decision.size_multiplier

        return {
            "action": action,
            "regime": regime,
            "strategies": regime_weights.weights,
            "position_multiplier": round(position_multiplier, 3),
            "confidence": round(weighted_confidence, 3),
            "gate": gate_decision.to_dict(),
            "signals_used": len(signals),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _determine_action(
        self,
        regime: str,
        price_prediction: Optional[Dict],
        sentiment: Optional[float],
        orderbook: Optional[float],
    ) -> str:
        """Determine overall trading action from signals."""
        # Crisis = always defensive
        if regime == "CRISIS":
            return "reduce"
        
        # Count bullish/bearish signals
        bullish = 0
        bearish = 0
        
        # Regime signal
        if regime in ("TREND_UP",):
            bullish += 1
        elif regime in ("TREND_DOWN",):
            bearish += 1
        
        # Price prediction
        if price_prediction:
            direction = price_prediction.get("direction", "")
            if direction == "up":
                bullish += 1
            elif direction == "down":
                bearish += 1
        
        # Sentiment
        if sentiment is not None:
            if sentiment > 0.2:
                bullish += 1
            elif sentiment < -0.2:
                bearish += 1
        
        # Orderbook
        if orderbook is not None:
            if orderbook > 0.2:
                bullish += 1
            elif orderbook < -0.2:
                bearish += 1
        
        # Determine action
        if bullish > bearish:
            return "buy"
        elif bearish > bullish:
            return "sell"
        else:
            return "hold"
