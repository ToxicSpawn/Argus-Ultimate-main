"""
Meta-Learning Engine
====================
Learns HOW to learn - optimizes learning rates, exploration strategies,
and adaptation timing based on what's working.

Key Concepts:
1. Learning Rate Adaptation - Each parameter has its own optimal learning speed
2. Exploration vs Exploitation - Know when to try new values vs use known good ones
3. Learning Velocity - Track how fast parameters should change
4. Confidence in Learning - Know when learning is reliable vs noisy
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""
    # Learning rate bounds for different parameter types
    filter_lr_bounds: Tuple[float, float] = (0.05, 0.40)
    confidence_lr_bounds: Tuple[float, float] = (0.05, 0.30)
    threshold_lr_bounds: Tuple[float, float] = (0.10, 0.50)
    
    # Exploration settings
    exploration_rate: float = 0.10  # 10% of the time, explore
    exploration_decay: float = 0.99  # Decay exploration over time
    min_exploration_rate: float = 0.02
    
    # Learning velocity tracking
    velocity_window: int = 20  # How many adjustments to track
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.3


class ParameterLearningState:
    """Tracks learning state for a single parameter."""
    
    def __init__(self, name: str, initial_lr: float, bounds: Tuple[float, float]):
        self.name = name
        self.current_lr = initial_lr
        self.lr_bounds = bounds
        self.current_value = 0.0
        self.adjustments: Deque[float] = deque(maxlen=20)
        self.outcomes: Deque[float] = deque(maxlen=50)  # PnL after each adjustment
        self.exploration_count = 0
        self.exploitation_count = 0
        
    def record_adjustment(self, adjustment: float, outcome_pnl: float) -> None:
        """Record an adjustment and its outcome."""
        self.adjustments.append(adjustment)
        self.outcomes.append(outcome_pnl)
    
    def get_learning_velocity(self) -> float:
        """Get how fast this parameter is learning (improvement per adjustment)."""
        if len(self.outcomes) < 5:
            return 0.0
        
        recent = list(self.outcomes)[-10:]
        if len(recent) < 2:
            return 0.0
        
        # Calculate trend in outcomes
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)
        return slope
    
    def get_learning_confidence(self) -> float:
        """Get confidence in current learning (0-1)."""
        if len(self.outcomes) < 10:
            return 0.3  # Low confidence with few data points
        
        # Higher confidence if outcomes are consistent and positive
        recent = list(self.outcomes)[-20:]
        mean_pnl = np.mean(recent)
        std_pnl = np.std(recent)
        
        if std_pnl == 0:
            return 0.5
        
        # Sharpe-like metric for learning
        learning_sharpe = mean_pnl / std_pnl
        
        # Map to 0-1
        confidence = 0.5 + np.tanh(learning_sharpe) * 0.5
        return np.clip(confidence, 0.1, 0.95)
    
    def adapt_learning_rate(self) -> float:
        """Adapt learning rate based on learning velocity and confidence."""
        velocity = self.get_learning_velocity()
        confidence = self.get_learning_confidence()
        
        # High velocity + high confidence → increase LR (learn faster)
        # Low velocity + high confidence → decrease LR (fine-tuning)
        # Low confidence → moderate LR (uncertain)
        
        if confidence > 0.7:
            if velocity > 0:
                # Learning well, speed up
                self.current_lr *= 1.1
            else:
                # Learning poorly, slow down
                self.current_lr *= 0.9
        elif confidence < 0.3:
            # Uncertain, moderate LR
            self.current_lr = (self.lr_bounds[0] + self.lr_bounds[1]) / 2
        
        # Apply bounds
        self.current_lr = np.clip(self.current_lr, self.lr_bounds[0], self.lr_bounds[1])
        
        return self.current_lr
    
    def should_explore(self, exploration_rate: float) -> bool:
        """Determine if we should explore or exploit."""
        confidence = self.get_learning_confidence()
        
        # Low confidence → explore more
        adjusted_rate = exploration_rate * (1.5 - confidence)
        
        return np.random.random() < adjusted_rate


class RegimeLearningMemory:
    """Remembers what learned well in each regime."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories: Dict[str, Deque[Dict]] = {}
        
    def record(self, regime: str, parameters: Dict[str, float], 
               performance: Dict[str, float]) -> None:
        """Record a successful parameter set for a regime."""
        if regime not in self.memories:
            self.memories[regime] = deque(maxlen=self.capacity)
        
        memory = {
            "parameters": dict(parameters),
            "performance": dict(performance),
            "timestamp": time.time(),
        }
        self.memories[regime].append(memory)
    
    def get_best_parameters(self, regime: str, 
                            metric: str = "profit_factor") -> Optional[Dict[str, float]]:
        """Get the best-performing parameters for a regime."""
        if regime not in self.memories or not self.memories[regime]:
            return None
        
        memories = list(self.memories[regime])
        
        # Find best by metric
        best = max(memories, key=lambda m: m["performance"].get(metric, 0))
        return best.get("parameters")
    
    def get_regime_similarity(self, regime1: str, regime2: str) -> float:
        """Calculate similarity between two regimes based on learned parameters."""
        if regime1 not in self.memories or regime2 not in self.memories:
            return 0.0
        
        mem1 = list(self.memories[regime1])[-5:]  # Recent memories
        mem2 = list(self.memories[regime2])[-5:]
        
        if not mem1 or not mem2:
            return 0.0
        
        # Average parameter sets
        avg1 = {}
        avg2 = {}
        
        for key in set(
            list(mem1[0]["parameters"].keys()) + 
            list(mem2[0]["parameters"].keys())
        ):
            vals1 = [m["parameters"].get(key, 0) for m in mem1]
            vals2 = [m["parameters"].get(key, 0) for m in mem2]
            avg1[key] = np.mean(vals1)
            avg2[key] = np.mean(vals2)
        
        # Calculate cosine similarity
        if not avg1 or not avg2:
            return 0.0
        
        keys = set(avg1.keys()) & set(avg2.keys())
        if not keys:
            return 0.0
        
        vec1 = np.array([avg1[k] for k in keys])
        vec2 = np.array([avg2[k] for k in keys])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)


class MetaLearningEngine:
    """
    Advanced meta-learning engine that optimizes the learning process itself.
    
    Features:
    1. Adaptive learning rates per parameter
    2. Exploration vs exploitation balance
    3. Regime memory (remembers what worked before)
    4. Transfer learning between similar regimes
    5. Learning confidence tracking
    """
    
    def __init__(self, config: Optional[MetaLearningConfig] = None):
        self.config = config or MetaLearningConfig()
        
        # Parameter learning states
        self.parameters: Dict[str, ParameterLearningState] = {
            "filter_threshold": ParameterLearningState(
                "filter_threshold", 0.2, self.config.filter_lr_bounds
            ),
            "confidence_floor_trend": ParameterLearningState(
                "confidence_floor_trend", 0.15, self.config.confidence_lr_bounds
            ),
            "confidence_floor_momentum": ParameterLearningState(
                "confidence_floor_momentum", 0.15, self.config.confidence_lr_bounds
            ),
            "confidence_floor_reversion": ParameterLearningState(
                "confidence_floor_reversion", 0.15, self.config.confidence_lr_bounds
            ),
            "strategy_threshold_trend": ParameterLearningState(
                "strategy_threshold_trend", 0.25, self.config.threshold_lr_bounds
            ),
            "strategy_threshold_momentum": ParameterLearningState(
                "strategy_threshold_momentum", 0.25, self.config.threshold_lr_bounds
            ),
        }
        
        # Regime memory
        self.regime_memory = RegimeLearningMemory()
        
        # Global state
        self.exploration_rate = self.config.exploration_rate
        self.meta_learning_cycles = 0
        self.total_improvement = 0.0
        
        logger.info("MetaLearningEngine initialized")
    
    def get_parameter_lr(self, parameter_name: str) -> float:
        """Get adaptive learning rate for a parameter."""
        if parameter_name in self.parameters:
            return self.parameters[parameter_name].current_lr
        return 0.2  # Default
    
    def should_explore(self, parameter_name: str) -> bool:
        """Determine if we should explore for this parameter."""
        if parameter_name in self.parameters:
            return self.parameters[parameter_name].should_explore(self.exploration_rate)
        return np.random.random() < self.exploration_rate
    
    def record_parameter_outcome(self, parameter_name: str,
                                  adjustment: float, outcome_pnl: float) -> None:
        """Record outcome of a parameter adjustment."""
        if parameter_name in self.parameters:
            self.parameters[parameter_name].record_adjustment(adjustment, outcome_pnl)
    
    def adapt_all_learning_rates(self) -> Dict[str, float]:
        """Adapt learning rates for all parameters."""
        results = {}
        
        for name, state in self.parameters.items():
            new_lr = state.adapt_learning_rate()
            results[name] = new_lr
        
        # Decay exploration rate
        self.exploration_rate *= self.config.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.config.min_exploration_rate)
        
        return results
    
    def get_best_regime_parameters(self, regime: str) -> Optional[Dict[str, float]]:
        """Get best-known parameters for a regime (transfer learning)."""
        return self.regime_memory.get_best_parameters(regime)
    
    def record_regime_performance(self, regime: str, parameters: Dict[str, float],
                                   performance: Dict[str, float]) -> None:
        """Record successful parameters for a regime."""
        if performance.get("profit_factor", 0) > 1.2:
            self.regime_memory.record(regime, parameters, performance)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        stats = {
            "meta_learning_cycles": self.meta_learning_cycles,
            "exploration_rate": self.exploration_rate,
            "total_improvement": self.total_improvement,
            "parameters": {},
        }
        
        for name, state in self.parameters.items():
            stats["parameters"][name] = {
                "learning_rate": state.current_lr,
                "learning_velocity": state.get_learning_velocity(),
                "learning_confidence": state.get_learning_confidence(),
                "adjustments": len(state.adjustments),
                "exploration_count": state.exploration_count,
                "exploitation_count": state.exploitation_count,
            }
        
        return stats


# Singleton
_meta_engine: Optional[MetaLearningEngine] = None


def get_meta_engine() -> MetaLearningEngine:
    """Get or create singleton meta-learning engine."""
    global _meta_engine
    if _meta_engine is None:
        _meta_engine = MetaLearningEngine()
    return _meta_engine


def reset_meta_engine() -> None:
    """Reset singleton (for testing)."""
    global _meta_engine
    _meta_engine = None
