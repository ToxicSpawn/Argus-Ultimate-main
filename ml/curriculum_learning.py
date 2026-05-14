# pyright: reportMissingImports=false
"""
Curriculum Learning System for Argus Trading.

This module implements curriculum learning to train models progressively
from simple to complex market scenarios for better generalization.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    EASY = auto()      # Trending markets, low volatility
    MEDIUM = auto()    # Ranging markets, moderate volatility
    HARD = auto()      # Volatile markets, high uncertainty
    EXTREME = auto()   # Crashes, flash crashes, black swan events


@dataclass
class MarketScenario:
    """A market scenario for training."""
    scenario_id: str
    difficulty: DifficultyLevel
    market_state: NDArray[np.float64]
    optimal_action: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    initial_level: DifficultyLevel = DifficultyLevel.EASY
    advance_threshold: float = 0.8  # Performance threshold to advance
    regression_threshold: float = 0.6  # Performance threshold to regress
    min_samples_per_level: int = 100
    max_samples_per_level: int = 1000
    enable_auto_advance: bool = True


class DifficultyEstimator:
    """Estimates the difficulty of market scenarios."""

    def __init__(self):
        self.feature_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

    def estimate_difficulty(self, market_state: NDArray[np.float64]) -> float:
        """Estimate difficulty score (0-1) for a market scenario."""
        if len(market_state) < 5:
            return 0.5

        # Feature 1: Volatility (std of recent prices)
        volatility = np.std(market_state)
        vol_score = min(volatility / 2.0, 1.0)

        # Feature 2: Trend strength (slope)
        if len(market_state) > 1:
            slope = np.polyfit(range(len(market_state)), market_state, 1)[0]
            trend_score = min(abs(slope) / 0.5, 1.0)
        else:
            trend_score = 0.5

        # Feature 3: Regime stability
        regime_changes = self._count_regime_changes(market_state)
        regime_score = min(regime_changes / 5.0, 1.0)

        # Feature 4: Extreme values
        extreme_count = np.sum(np.abs(market_state) > 2.0)
        extreme_score = min(extreme_count / len(market_state), 1.0)

        # Feature 5: Non-linearity
        nonlinearity = self._measure_nonlinearity(market_state)
        nonlin_score = min(nonlinearity, 1.0)

        # Combined difficulty score
        difficulty = (
            vol_score * self.feature_weights[0] +
            trend_score * self.feature_weights[1] +
            regime_score * self.feature_weights[2] +
            extreme_score * self.feature_weights[3] +
            nonlin_score * self.feature_weights[4]
        )

        return min(max(difficulty, 0.0), 1.0)

    def _count_regime_changes(self, data: NDArray[np.float64]) -> int:
        """Count number of regime changes in data."""
        if len(data) < 3:
            return 0

        changes = 0
        window = min(5, len(data) // 3)

        for i in range(window, len(data) - window):
            before_mean = np.mean(data[i-window:i])
            after_mean = np.mean(data[i:i+window])
            if abs(after_mean - before_mean) > np.std(data) * 0.5:
                changes += 1

        return changes

    def _measure_nonlinearity(self, data: NDArray[np.float64]) -> float:
        """Measure non-linearity of data."""
        if len(data) < 4:
            return 0.0

        # Compare linear vs quadratic fit
        x = np.arange(len(data))
        linear_fit = np.polyfit(x, data, 1)
        quadratic_fit = np.polyfit(x, data, 2)

        linear_pred = np.polyval(linear_fit, x)
        quadratic_pred = np.polyval(quadratic_fit, x)

        linear_error = np.mean((data - linear_pred) ** 2)
        quadratic_error = np.mean((data - quadratic_pred) ** 2)

        if linear_error < 1e-8:
            return 0.0

        improvement = (linear_error - quadratic_error) / linear_error
        return max(0.0, improvement)


class CurriculumGenerator:
    """Generates training scenarios for each difficulty level."""

    def __init__(self):
        self.estimator = DifficultyEstimator()

    def generate_scenarios(self, 
                          level: DifficultyLevel,
                          num_scenarios: int = 100) -> List[MarketScenario]:
        """Generate market scenarios for a difficulty level."""
        scenarios = []

        for i in range(num_scenarios):
            if level == DifficultyLevel.EASY:
                market_state, optimal_action = self._generate_easy_scenario()
            elif level == DifficultyLevel.MEDIUM:
                market_state, optimal_action = self._generate_medium_scenario()
            elif level == DifficultyLevel.HARD:
                market_state, optimal_action = self._generate_hard_scenario()
            else:  # EXTREME
                market_state, optimal_action = self._generate_extreme_scenario()

            scenario = MarketScenario(
                scenario_id=f"{level.name}_{i}",
                difficulty=level,
                market_state=market_state,
                optimal_action=optimal_action,
                metadata={"generated": True, "level": level.name}
            )
            scenarios.append(scenario)

        logger.info(f"Generated {num_scenarios} {level.name} scenarios")
        return scenarios

    def _generate_easy_scenario(self) -> Tuple[NDArray[np.float64], int]:
        """Generate an easy market scenario (strong trend, low noise)."""
        # Strong upward trend with low noise
        trend = 0.1
        noise = 0.05
        length = 20

        base = np.cumsum(np.random.randn(length) * noise + trend)
        base = (base - np.mean(base)) / (np.std(base) + 1e-8)

        # Optimal action: buy (1) if trending up, sell (2) if trending down
        optimal = 1 if trend > 0 else 2

        return base, optimal

    def _generate_medium_scenario(self) -> Tuple[NDArray[np.float64], int]:
        """Generate a medium market scenario (ranging with some noise)."""
        # Mean-reverting with moderate noise
        mean = 0.0
        reversion_strength = 0.1
        noise = 0.15
        length = 20

        base = np.zeros(length)
        for i in range(1, length):
            base[i] = base[i-1] + reversion_strength * (mean - base[i-1]) + np.random.randn() * noise

        base = (base - np.mean(base)) / (np.std(base) + 1e-8)

        # Optimal action: hold (0) or mean-reversion
        current = base[-1]
        if current > 0.5:
            optimal = 2  # Sell (overbought)
        elif current < -0.5:
            optimal = 1  # Buy (oversold)
        else:
            optimal = 0  # Hold

        return base, optimal

    def _generate_hard_scenario(self) -> Tuple[NDArray[np.float64], int]:
        """Generate a hard market scenario (high volatility, regime changes)."""
        # High volatility with regime changes
        volatility = 0.4
        length = 20

        # Create regime changes
        regime_length = length // 3
        base = np.zeros(length)

        for i in range(length):
            regime = i // regime_length
            if regime == 0:
                base[i] = np.random.randn() * volatility
            elif regime == 1:
                base[i] = np.random.randn() * volatility + 1.0
            else:
                base[i] = np.random.randn() * volatility - 0.5

        base = (base - np.mean(base)) / (np.std(base) + 1e-8)

        # Optimal action: hedge (3) in volatile conditions
        optimal = 3 if np.std(base[-5:]) > 0.3 else random.choice([0, 1, 2])

        return base, optimal

    def _generate_extreme_scenario(self) -> Tuple[NDArray[np.float64], int]:
        """Generate an extreme market scenario (crash, flash crash)."""
        # Crash or flash crash scenario
        length = 20

        base = np.random.randn(length) * 0.2

        # Insert crash
        crash_start = length // 2
        crash_magnitude = random.uniform(-3.0, -1.5)
        for i in range(crash_start, min(crash_start + 5, length)):
            base[i] += crash_magnitude * (1 - (i - crash_start) / 5)

        base = (base - np.mean(base)) / (np.std(base) + 1e-8)

        # Optimal action: hedge or sell in crash
        optimal = 3  # Hedge

        return base, optimal


class CurriculumLearner:
    """Curriculum learning system for progressive training."""

    def __init__(self, config: Optional[CurriculumConfig] = None):
        """Initialize the curriculum learner."""
        self.config = config or CurriculumConfig()
        self.generator = CurriculumGenerator()
        self.current_level = self.config.initial_level
        self.performance_history: Dict[DifficultyLevel, List[float]] = {
            level: [] for level in DifficultyLevel
        }
        self.training_history: List[Dict[str, Any]] = []
        self.samples_generated: Dict[DifficultyLevel, int] = {
            level: 0 for level in DifficultyLevel
        }

    def get_training_batch(self, batch_size: int = 32) -> List[MarketScenario]:
        """Get a training batch at the current difficulty level."""
        needed = batch_size - len(self._current_buffer())
        
        if needed > 0:
            new_scenarios = self.generator.generate_scenarios(
                self.current_level, needed
            )
            self._add_to_buffer(new_scenarios)
            self.samples_generated[self.current_level] += needed

        return self._sample_buffer(batch_size)

    def _current_buffer(self) -> List[MarketScenario]:
        """Get current buffer for the level."""
        return getattr(self, f'_buffer_{self.current_level.name}', [])

    def _add_to_buffer(self, scenarios: List[MarketScenario]) -> None:
        """Add scenarios to buffer."""
        buffer_name = f'_buffer_{self.current_level.name}'
        if not hasattr(self, buffer_name):
            setattr(self, buffer_name, [])
        getattr(self, buffer_name).extend(scenarios)

    def _sample_buffer(self, batch_size: int) -> List[MarketScenario]:
        """Sample from buffer."""
        buffer = self._current_buffer()
        if len(buffer) <= batch_size:
            return buffer
        return random.sample(buffer, batch_size)

    def update_performance(self, performance: float) -> None:
        """Update performance for current level."""
        self.performance_history[self.current_level].append(performance)

        # Check if we should advance
        if self.config.enable_auto_advance:
            self._check_level_progression()

    def _check_level_progression(self) -> None:
        """Check if we should advance or regress in difficulty."""
        history = self.performance_history[self.current_level]
        
        if len(history) < self.config.min_samples_per_level:
            return

        recent_performance = np.mean(history[-self.config.min_samples_per_level:])

        # Check for advancement
        if recent_performance >= self.config.advance_threshold:
            levels = list(DifficultyLevel)
            current_idx = levels.index(self.current_level)
            
            if current_idx < len(levels) - 1:
                self.current_level = levels[current_idx + 1]
                logger.info(f"Advancing to {self.current_level.name} "
                          f"(performance: {recent_performance:.2%})")
                
                self.training_history.append({
                    "action": "advance",
                    "from": levels[current_idx].name,
                    "to": self.current_level.name,
                    "performance": recent_performance
                })

        # Check for regression
        elif recent_performance < self.config.regression_threshold:
            levels = list(DifficultyLevel)
            current_idx = levels.index(self.current_level)
            
            if current_idx > 0:
                self.current_level = levels[current_idx - 1]
                logger.info(f"Regressing to {self.current_level.name} "
                          f"(performance: {recent_performance:.2%})")
                
                self.training_history.append({
                    "action": "regress",
                    "from": levels[current_idx].name,
                    "to": self.current_level.name,
                    "performance": recent_performance
                })

    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum status."""
        return {
            "current_level": self.current_level.name,
            "performance_by_level": {
                level.name: {
                    "samples": len(history),
                    "avg_performance": np.mean(history) if history else 0.0,
                    "recent_performance": np.mean(history[-10:]) if len(history) >= 10 else 0.0
                }
                for level, history in self.performance_history.items()
            },
            "samples_generated": {
                level.name: count for level, count in self.samples_generated.items()
            },
            "training_history": self.training_history[-10:]  # Last 10 events
        }


class AdaptiveCurriculumLearner(CurriculumLearner):
    """Enhanced curriculum learner with adaptive difficulty adjustment."""

    def __init__(self, config: Optional[CurriculumConfig] = None):
        super().__init__(config)
        self.difficulty_adjustment_rate = 0.1
        self.adaptive_thresholds = {
            DifficultyLevel.EASY: 0.85,
            DifficultyLevel.MEDIUM: 0.75,
            DifficultyLevel.HARD: 0.65,
            DifficultyLevel.EXTREME: 0.55
        }

    def _check_level_progression(self) -> None:
        """Adaptive level progression with level-specific thresholds."""
        history = self.performance_history[self.current_level]
        
        if len(history) < 20:
            return

        recent_performance = np.mean(history[-20:])
        threshold = self.adaptive_thresholds[self.current_level]

        # Adaptive threshold based on learning rate
        if len(history) > 50:
            early_perf = np.mean(history[:20])
            recent_perf = np.mean(history[-20:])
            learning_rate = recent_perf - early_perf
            
            # Adjust threshold based on learning progress
            if learning_rate > 0.1:
                threshold += 0.05  # Raise bar if learning fast
            elif learning_rate < 0.01:
                threshold -= 0.05  # Lower bar if stuck

        # Check advancement
        if recent_performance >= threshold:
            levels = list(DifficultyLevel)
            current_idx = levels.index(self.current_level)
            
            if current_idx < len(levels) - 1:
                self.current_level = levels[current_idx + 1]
                logger.info(f"Adaptive advancement to {self.current_level.name}")


__all__ = [
    "CurriculumLearner",
    "AdaptiveCurriculumLearner",
    "CurriculumConfig",
    "CurriculumGenerator",
    "DifficultyLevel",
    "DifficultyEstimator",
    "MarketScenario"
]