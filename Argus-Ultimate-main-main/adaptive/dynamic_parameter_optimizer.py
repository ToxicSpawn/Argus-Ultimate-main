"""Dynamic Parameter Optimization System.

Features:
- Real-time parameter tuning
- Bayesian optimization
- Multi-armed bandit selection
- Performance-based adaptation
- Parameter drift detection
- A/B testing framework
- Ensemble parameter selection
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from collections import deque, defaultdict
import random

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    GRADIENT_DESCENT = "gradient_descent"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    ADAPTIVE = "adaptive"


@dataclass
class ParameterSet:
    name: str
    params: Dict[str, float]
    score: float = 0.0
    sample_count: int = 0
    last_updated: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    best_params: Dict[str, float]
    best_score: float
    iterations: int
    convergence: float
    method: OptimizationMethod


class ParameterSpace:
    def __init__(self):
        self._spaces: Dict[str, Tuple[float, float]] = {}
        self._defaults: Dict[str, float] = {}

    def add_param(
        self,
        name: str,
        min_value: float,
        max_value: float,
        default: Optional[float] = None,
    ) -> None:
        self._spaces[name] = (min_value, max_value)
        if default is not None:
            self._defaults[name] = default
        elif min_value <= 0 <= max_value:
            self._defaults[name] = 0.0
        else:
            self._defaults[name] = min_value

    def sample(self, method: str = "random") -> Dict[str, float]:
        params = {}
        for name, (min_val, max_val) in self._spaces.items():
            if method == "random":
                params[name] = random.uniform(min_val, max_val)
            elif method == "center":
                params[name] = (min_val + max_val) / 2
            elif method == "edges":
                params[name] = random.choice([min_val, max_val])
        return params

    def get_default(self) -> Dict[str, float]:
        return self._defaults.copy()

    def clip(self, params: Dict[str, float]) -> Dict[str, float]:
        clipped = {}
        for name, value in params.items():
            if name in self._spaces:
                min_val, max_val = self._spaces[name]
                clipped[name] = max(min_val, min(max_val, value))
            else:
                clipped[name] = value
        return clipped


class BayesianOptimizer:
    def __init__(self, parameter_space: ParameterSpace):
        self._space = parameter_space
        self._observations: List[Tuple[Dict[str, float], float]] = []
        self._acquisition_alpha = 0.1

    def suggest(self) -> Dict[str, float]:
        if len(self._observations) < 3:
            return self._space.sample("random")

        best_score = max(score for _, score in self._observations)
        
        candidates = []
        for _ in range(50):
            candidate = self._space.sample("random")
            acquisition = self._calculate_acquisition(candidate, best_score)
            candidates.append((candidate, acquisition))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _calculate_acquisition(
        self,
        params: Dict[str, float],
        best_score: float,
    ) -> float:
        distances = []
        for obs_params, _ in self._observations:
            dist = sum(
                (params.get(k, 0) - obs_params.get(k, 0)) ** 2
                for k in set(params.keys()) | set(obs_params.keys())
            ) ** 0.5
            distances.append(dist)

        avg_distance = np.mean(distances) if distances else 1.0
        exploration = avg_distance / (avg_distance + 1.0)

        similarity = 1.0 / (1.0 + min(distances) if distances else 1.0)

        return exploration * 0.3 + similarity * 0.7

    def observe(self, params: Dict[str, float], score: float) -> None:
        self._observations.append((params.copy(), score))

    def get_best(self) -> Tuple[Dict[str, float], float]:
        if not self._observations:
            return self._space.get_default(), 0.0

        best = max(self._observations, key=lambda x: x[1])
        return best[0], best[1]


class MultiArmedBandit:
    def __init__(self, epsilon: float = 0.1):
        self._epsilon = epsilon
        self._arms: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "value": 0.0, "wins": 0}
        )

    def select_arm(self) -> str:
        if random.random() < self._epsilon:
            return random.choice(list(self._arms.keys()))

        best_arm = max(
            self._arms.keys(),
            key=lambda k: self._arms[k]["value"]
        )
        return best_arm

    def add_arm(self, arm_id: str) -> None:
        if arm_id not in self._arms:
            self._arms[arm_id] = {"count": 0, "value": 0.0, "wins": 0}

    def update(self, arm_id: str, reward: float) -> None:
        arm = self._arms[arm_id]
        arm["count"] += 1
        arm["wins"] += 1 if reward > 0 else 0

        n = arm["count"]
        old_value = arm["value"]
        arm["value"] = old_value + (reward - old_value) / n

    def get_best_arm(self) -> str:
        return max(self._arms.keys(), key=lambda k: self._arms[k]["value"])


class GradientDescentOptimizer:
    def __init__(self, parameter_space: ParameterSpace, learning_rate: float = 0.01):
        self._space = parameter_space
        self._lr = learning_rate
        self._gradients: Dict[str, float] = {}

    def compute_gradient(
        self,
        params: Dict[str, float],
        score: float,
        observations: List[Tuple[Dict[str, float], float]],
    ) -> Dict[str, float]:
        if len(observations) < 2:
            return {k: random.uniform(-0.1, 0.1) for k in params.keys()}

        gradients = {}
        for param_name in params.keys():
            param_scores = [
                (obs[0].get(param_name, 0), obs[1])
                for obs in observations
                if param_name in obs[0]
            ]

            if len(param_scores) >= 2:
                scores = [s for _, s in param_scores]
                values = [v for v, _ in param_scores]
                if len(set(scores)) > 1:
                    corr = np.corrcoef(values, scores)[0, 1]
                    gradients[param_name] = corr if not np.isnan(corr) else 0.0
                else:
                    gradients[param_name] = 0.0
            else:
                gradients[param_name] = 0.0

        return gradients

    def step(
        self,
        params: Dict[str, float],
        score: float,
        observations: List[Tuple[Dict[str, float], float]],
    ) -> Dict[str, float]:
        gradients = self.compute_gradient(params, score, observations)

        new_params = {}
        for name, value in params.items():
            grad = gradients.get(name, 0.0)
            new_params[name] = value + self._lr * grad

        return self._space.clip(new_params)


class ParameterDriftDetector:
    def __init__(self, window: int = 50, threshold: float = 2.0):
        self._window = window
        self._threshold = threshold
        self._history: deque = deque(maxlen=window)

    def add_observation(self, params: Dict[str, float], score: float) -> None:
        self._history.append({"params": params.copy(), "score": score})

    def detect_drift(self) -> Dict[str, bool]:
        if len(self._history) < self._window:
            return {}

        scores = [obs["score"] for obs in self._history]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        drift_detected = {}

        recent = list(self._history)[-10:]
        recent_mean = np.mean([obs["score"] for obs in recent])

        drift_detected["performance"] = abs(recent_mean - mean_score) > self._threshold * std_score

        return drift_detected


class DynamicParameterOptimizer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self._parameter_space = ParameterSpace()
        self._current_params: Dict[str, float] = {}
        
        self._method = OptimizationMethod(
            self.config.get("method", "adaptive")
        )
        
        self._bayesian = BayesianOptimizer(self._parameter_space)
        self._bandit = MultiArmedBandit(epsilon=self.config.get("epsilon", 0.1))
        self._gradient = GradientDescentOptimizer(
            self._parameter_space,
            learning_rate=self.config.get("learning_rate", 0.01)
        )
        
        self._drift_detector = ParameterDriftDetector(
            window=self.config.get("drift_window", 50)
        )
        
        self._observations: deque = deque(maxlen=1000)
        self._iteration = 0
        self._best_score = float("-inf")
        self._best_params: Dict[str, float] = {}

    def add_parameter(
        self,
        name: str,
        min_value: float,
        max_value: float,
        default: Optional[float] = None,
    ) -> None:
        self._parameter_space.add_param(name, min_value, max_value, default)
        self._bandit.add_arm(name)

        if default is not None:
            self._current_params[name] = default

    def get_current_params(self) -> Dict[str, float]:
        return self._current_params.copy()

    def suggest_params(self) -> Dict[str, float]:
        if not self._current_params:
            self._current_params = self._parameter_space.get_default()

        if self._method == OptimizationMethod.BAYESIAN:
            return self._bayesian.suggest()
        elif self._method == OptimizationMethod.MULTI_ARMED_BANDIT:
            arm = self._bandit.select_arm()
            return {arm: self._current_params.get(arm, 0.5)}
        elif self._method == OptimizationMethod.GRADIENT_DESCENT:
            observations = [(obs["params"], obs["score"]) for obs in self._observations]
            return self._gradient.step(self._current_params, 0.0, observations)
        elif self._method == OptimizationMethod.ADAPTIVE:
            return self._adaptive_suggest()
        else:
            return self._parameter_space.sample()

    def _adaptive_suggest(self) -> Dict[str, float]:
        drift = self._drift_detector.detect_drift()

        if drift.get("performance", False):
            logger.info("Performance drift detected, re-exploring parameters")
            return self._parameter_space.sample("random")

        if len(self._observations) < 10:
            return self._parameter_space.sample("random")

        exploration_rate = max(0.1, 1.0 - (len(self._observations) / 500))

        if random.random() < exploration_rate:
            return self._parameter_space.sample("random")

        return self._bayesian.suggest()

    def update(
        self,
        params: Dict[str, float],
        score: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        self._iteration += 1

        params = self._parameter_space.clip(params)
        
        self._observations.append({
            "params": params.copy(),
            "score": score,
            "iteration": self._iteration,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })

        self._drift_detector.add_observation(params, score)

        if self._method == OptimizationMethod.BAYESIAN:
            self._bayesian.observe(params, score)
        elif self._method == OptimizationMethod.MULTI_ARMED_BANDIT:
            for param_name in params.keys():
                self._bandit.update(param_name, score)

        if score > self._best_score:
            self._best_score = score
            self._best_params = params.copy()

        self._current_params = params

    def get_best_params(self) -> Tuple[Dict[str, float], float]:
        return self._best_params.copy(), self._best_score

    def get_optimization_result(self) -> OptimizationResult:
        scores = [obs["score"] for obs in self._observations]
        
        if len(scores) >= 2:
            recent_scores = list(scores)[-20:]
            convergence = abs(recent_scores[-1] - recent_scores[0]) / (abs(recent_scores[0]) + 1e-8)
        else:
            convergence = 0.0

        return OptimizationResult(
            best_params=self._best_params.copy(),
            best_score=self._best_score,
            iterations=self._iteration,
            convergence=convergence,
            method=self._method,
        )

    def reset(self) -> None:
        self._observations.clear()
        self._iteration = 0
        self._best_score = float("-inf")
        self._best_params.clear()
        self._current_params = self._parameter_space.get_default()


class StrategyParameterTuner:
    def __init__(self):
        self._optimizers: Dict[str, DynamicParameterOptimizer] = {}
        self._strategy_params: Dict[str, Dict[str, float]] = {}

    def create_optimizer(
        self,
        strategy_name: str,
        parameters: Dict[str, Tuple[float, float, float]],
    ) -> DynamicParameterOptimizer:
        optimizer = DynamicParameterOptimizer()
        
        for param_name, (min_val, max_val, default) in parameters.items():
            optimizer.add_parameter(param_name, min_val, max_val, default)
        
        self._optimizers[strategy_name] = optimizer
        return optimizer

    def get_params(self, strategy_name: str) -> Dict[str, float]:
        if strategy_name in self._optimizers:
            return self._optimizers[strategy_name].get_current_params()
        return self._strategy_params.get(strategy_name, {})

    def update(
        self,
        strategy_name: str,
        score: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        if strategy_name not in self._optimizers:
            return

        optimizer = self._optimizers[strategy_name]
        params = optimizer.get_current_params()
        optimizer.update(params, score, metadata)

    def suggest_new_params(self, strategy_name: str) -> Dict[str, float]:
        if strategy_name not in self._optimizers:
            return {}

        return self._optimizers[strategy_name].suggest_params()

    def get_best_for_strategy(
        self,
        strategy_name: str,
    ) -> Tuple[Dict[str, float], float]:
        if strategy_name not in self._optimizers:
            return {}, 0.0

        return self._optimizers[strategy_name].get_best_params()

    def get_all_results(self) -> Dict[str, OptimizationResult]:
        return {
            name: opt.get_optimization_result()
            for name, opt in self._optimizers.items()
        }
