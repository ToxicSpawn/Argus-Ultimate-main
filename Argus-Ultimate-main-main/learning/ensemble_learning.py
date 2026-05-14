"""
Ensemble Learning System
========================
Combines multiple learning strategies and lets them compete.
The best-performing strategies get more weight over time.

Key Features:
1. Multiple Learning Agents - Each uses different learning strategies
2. Weighted Voting - Better performers get more influence
3. Strategy Evolution - Poor performers are replaced with new ones
4. Diversity Bonus - Strategies that disagree get a small bonus
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LearningStrategy(ABC):
    """Base class for learning strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.weight: float = 1.0
        self.total_pnl: float = 0.0
        self.trade_count: int = 0
        self.wins: int = 0
        self.recent_pnl: Deque[float] = deque(maxlen=50)
    
    @abstractmethod
    def suggest_adjustment(self, 
                           current_value: float,
                           metrics: Dict[str, float],
                           regime: str) -> float:
        """Suggest a parameter adjustment."""
        pass
    
    def record_outcome(self, pnl: float, trade_won: bool) -> None:
        """Record outcome of a trade using this strategy's suggestion."""
        self.total_pnl += pnl
        self.trade_count += 1
        self.recent_pnl.append(pnl)
        if trade_won:
            self.wins += 1
    
    def get_win_rate(self) -> float:
        """Get win rate."""
        if self.trade_count == 0:
            return 0.5
        return self.wins / self.trade_count
    
    def get_sharpe(self) -> float:
        """Get Sharpe ratio of recent PnL."""
        if len(self.recent_pnl) < 5:
            return 0.0
        
        pnl_list = list(self.recent_pnl)
        mean_pnl = np.mean(pnl_list)
        std_pnl = np.std(pnl_list)
        
        if std_pnl == 0:
            return 0.0
        
        return mean_pnl / std_pnl
    
    def get_score(self) -> float:
        """Get overall score for weighting."""
        if self.trade_count < 10:
            return 0.5  # Default for new strategies
        
        # Combined score: profit factor + sharpe + win rate
        win_rate = self.get_win_rate()
        sharpe = self.get_sharpe()
        
        # Profit factor
        profits = [p for p in self.recent_pnl if p > 0]
        losses = [abs(p) for p in self.recent_pnl if p < 0]
        
        if losses:
            profit_factor = sum(profits) / sum(losses) if profits else 0.0
        else:
            profit_factor = 2.0 if profits else 1.0
        
        score = (profit_factor * 0.4 + sharpe * 0.3 + win_rate * 0.3)
        return max(0.1, score)


class ConservativeStrategy(LearningStrategy):
    """Makes small, cautious adjustments."""
    
    def __init__(self):
        super().__init__("conservative")
    
    def suggest_adjustment(self, current_value: float, 
                           metrics: Dict[str, float], regime: str) -> float:
        win_rate = metrics.get("win_rate", 0.5)
        
        if win_rate < 0.4:
            return current_value * 0.02  # 2% increase
        elif win_rate > 0.6:
            return current_value * -0.01  # 1% decrease
        return 0.0


class AggressiveStrategy(LearningStrategy):
    """Makes larger, faster adjustments."""
    
    def __init__(self):
        super().__init__("aggressive")
    
    def suggest_adjustment(self, current_value: float,
                           metrics: Dict[str, float], regime: str) -> float:
        win_rate = metrics.get("win_rate", 0.5)
        profit_factor = metrics.get("profit_factor", 1.0)
        
        if win_rate < 0.35 or profit_factor < 0.8:
            return current_value * 0.10  # 10% increase
        elif win_rate > 0.65 and profit_factor > 1.5:
            return current_value * -0.05  # 5% decrease
        return 0.0


class MomentumStrategy(LearningStrategy):
    """Follows the momentum of recent performance."""
    
    def __init__(self):
        super().__init__("momentum")
        self.last_direction = 0
    
    def suggest_adjustment(self, current_value: float,
                           metrics: Dict[str, float], regime: str) -> float:
        recent_pnl = list(self.recent_pnl)[-10:]
        if len(recent_pnl) < 3:
            return 0.0
        
        # Calculate momentum
        momentum = np.mean(recent_pnl[-3:]) - np.mean(recent_pnl[-6:-3])
        
        if momentum > 0 and self.last_direction >= 0:
            # Continuing good performance, keep direction
            self.last_direction = 1
            return current_value * -0.02  # Lower thresholds (more trades)
        elif momentum < 0 and self.last_direction <= 0:
            # Continuing bad performance, reverse
            self.last_direction = -1
            return current_value * 0.03  # Raise thresholds
        else:
            self.last_direction = 0
            return 0.0


class RegimeAdaptiveStrategy(LearningStrategy):
    """Adapts based on regime-specific performance."""
    
    def __init__(self):
        super().__init__("regime_adaptive")
        self.regime_performance: Dict[str, List[float]] = {}
    
    def suggest_adjustment(self, current_value: float,
                           metrics: Dict[str, float], regime: str) -> float:
        if regime not in self.regime_performance:
            return 0.0
        
        perf = self.regime_performance[regime]
        if len(perf) < 5:
            return 0.0
        
        avg_pnl = np.mean(perf[-10:])
        
        if avg_pnl < -0.5:
            return current_value * 0.08  # Raise threshold in bad regime
        elif avg_pnl > 0.5:
            return current_value * -0.04  # Lower threshold in good regime
        return 0.0
    
    def record_regime_outcome(self, regime: str, pnl: float) -> None:
        """Record outcome for specific regime."""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = []
        self.regime_performance[regime].append(pnl)


class MeanReversionStrategy(LearningStrategy):
    """Reverts toward mean when extreme."""
    
    def __init__(self):
        super().__init__("mean_reversion")
        self.adjustment_history: Deque[float] = deque(maxlen=20)
    
    def suggest_adjustment(self, current_value: float,
                           metrics: Dict[str, float], regime: str) -> float:
        if len(self.adjustment_history) < 5:
            return 0.0
        
        # Calculate if we've drifted from mean
        recent = list(self.adjustment_history)
        mean_adj = np.mean(recent)
        
        if abs(mean_adj) > current_value * 0.1:
            # We've drifted, pull back toward mean
            return -mean_adj * 0.3
        
        return 0.0


class EnsembleLearningSystem:
    """
    Combines multiple learning strategies with weighted voting.
    """
    
    def __init__(self):
        # Initialize strategies
        self.strategies: List[LearningStrategy] = [
            ConservativeStrategy(),
            AggressiveStrategy(),
            MomentumStrategy(),
            RegimeAdaptiveStrategy(),
            MeanReversionStrategy(),
        ]
        
        # Tracking
        self.ensemble_decisions: Deque[Dict] = deque(maxlen=100)
        self.consensus_history: Deque[float] = deque(maxlen=50)
        
        # Evolution settings
        self.min_strategies = 3
        self.max_strategies = 7
        self.replacement_threshold = 0.3  # Replace if score below this
        
        logger.info(f"EnsembleLearningSystem initialized with {len(self.strategies)} strategies")
    
    def get_ensemble_adjustment(self,
                                 current_value: float,
                                 metrics: Dict[str, float],
                                 regime: str) -> Tuple[float, Dict[str, Any]]:
        """
        Get weighted ensemble adjustment.
        
        Returns:
            (adjustment, details_dict)
        """
        suggestions = {}
        weights = {}
        
        # Get suggestions from all strategies
        for strategy in self.strategies:
            adj = strategy.suggest_adjustment(current_value, metrics, regime)
            score = strategy.get_score()
            
            suggestions[strategy.name] = adj
            weights[strategy.name] = score * strategy.weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {
                name: w / total_weight 
                for name, w in weights.items()
            }
        else:
            normalized_weights = {
                name: 1.0 / len(self.strategies)
                for name in weights.keys()
            }
        
        # Calculate weighted adjustment
        weighted_adjustment = sum(
            suggestions[name] * normalized_weights[name]
            for name in suggestions.keys()
        )
        
        # Calculate consensus (how much strategies agree)
        adj_values = list(suggestions.values())
        if len(adj_values) > 1:
            consensus = 1.0 - (np.std(adj_values) / max(abs(np.mean(adj_values)), 0.001))
            consensus = np.clip(consensus, 0.0, 1.0)
        else:
            consensus = 1.0
        
        self.consensus_history.append(consensus)
        
        details = {
            "suggestions": suggestions,
            "weights": normalized_weights,
            "consensus": consensus,
            "num_strategies": len(self.strategies),
        }
        
        # Record decision
        self.ensemble_decisions.append({
            "adjustment": weighted_adjustment,
            "consensus": consensus,
            "regime": regime,
            "timestamp": time.time(),
        })
        
        return weighted_adjustment, details
    
    def record_trade_outcome(self, pnl: float, trade_won: bool,
                              contributing_strategies: List[str]) -> None:
        """Record trade outcome for strategy learning."""
        for strategy in self.strategies:
            if strategy.name in contributing_strategies:
                strategy.record_outcome(pnl, trade_won)
    
    def evolve_strategies(self) -> List[str]:
        """
        Evolve strategies - replace poor performers.
        
        Returns:
            List of strategy names that were replaced.
        """
        replaced = []
        
        # Calculate scores
        scores = [(s.name, s.get_score(), s.trade_count) for s in self.strategies]
        
        # Find underperformers (only if they have enough trades)
        underperformers = [
            (name, score) 
            for name, score, count in scores 
            if count > 20 and score < self.replacement_threshold
        ]
        
        # Replace if we have room for new strategies
        if underperformers and len(self.strategies) < self.max_strategies:
            for name, _ in underperformers[:1]:
                # Remove old strategy
                self.strategies = [s for s in self.strategies if s.name != name]
                
                # Create new strategy variant
                new_strategy = self._create_new_strategy()
                self.strategies.append(new_strategy)
                
                replaced.append(name)
                logger.info(f"Replaced {name} with {new_strategy.name}")
        
        # Update weights based on performance
        for strategy in self.strategies:
            strategy.weight = max(0.5, strategy.get_score())
        
        return replaced
    
    def _create_new_strategy(self) -> LearningStrategy:
        """Create a new strategy variant."""
        # Mix elements from existing strategies
        return ConservativeStrategy()  # Simplified
    
    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Get current strategy rankings."""
        rankings = []
        
        for strategy in self.strategies:
            rankings.append({
                "name": strategy.name,
                "score": strategy.get_score(),
                "weight": strategy.weight,
                "trades": strategy.trade_count,
                "win_rate": strategy.get_win_rate(),
                "sharpe": strategy.get_sharpe(),
                "total_pnl": strategy.total_pnl,
            })
        
        # Sort by score
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        return rankings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        avg_consensus = np.mean(list(self.consensus_history)) if self.consensus_history else 1.0
        
        return {
            "num_strategies": len(self.strategies),
            "avg_consensus": avg_consensus,
            "decisions_made": len(self.ensemble_decisions),
            "strategy_rankings": self.get_strategy_rankings(),
        }


# Singleton
_ensemble: Optional[EnsembleLearningSystem] = None


def get_ensemble() -> EnsembleLearningSystem:
    """Get or create singleton ensemble system."""
    global _ensemble
    if _ensemble is None:
        _ensemble = EnsembleLearningSystem()
    return _ensemble


def reset_ensemble() -> None:
    """Reset singleton (for testing)."""
    global _ensemble
    _ensemble = None
