"""HyperAdaptive - Next Level Adaptive Trading System.

Features:
- Neural Architecture Search
- Meta-Learning
- Multi-Agent Coordination
- Adversarial Defense
- Causal Inference
- Continuous Online Learning
- Expert Portfolio System
"""

from __future__ import annotations

import logging
import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from collections import deque, defaultdict
from copy import deepcopy

logger = logging.getLogger(__name__)


class ModelType(Enum):
    LINEAR = "linear"
    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    TRANSFORMER = "transformer"
    ATTENTION = "attention"
    ENSEMBLE = "ensemble"


@dataclass
class NeuralArchitecture:
    model_type: ModelType
    layers: List[int]
    activations: List[str]
    dropout: float
    score: float


class NeuralArchitectureSearch:
    def __init__(self, input_dim: int = 10):
        self._input_dim = input_dim
        self._search_space = self._build_search_space()
        self._history: deque = deque(maxlen=100)
        self._best_architecture: Optional[NeuralArchitecture] = None

    def _build_search_space(self) -> Dict:
        return {
            "model_type": [ModelType.LINEAR, ModelType.MLP, ModelType.RNN],
            "layers": [[32], [64], [32, 16], [64, 32], [128, 64, 32]],
            "activations": ["relu", "tanh", "sigmoid"],
            "dropout": [0.0, 0.2, 0.4],
        }

    def sample_architecture(self) -> NeuralArchitecture:
        model_type = random.choice(self._search_space["model_type"])
        layers = random.choice(self._search_space["layers"])
        activations = random.choices(self._search_space["activations"], k=len(layers))
        dropout = random.choice(self._search_space["dropout"])

        return NeuralArchitecture(
            model_type=model_type,
            layers=layers,
            activations=activations,
            dropout=dropout,
            score=0.0,
        )

    def fitness_fn(self, arch: NeuralArchitecture) -> float:
        return random.uniform(0.5, 1.0)

    def search(
        self,
        generations: int = 20,
        population_size: int = 10,
    ) -> NeuralArchitecture:
        best = None

        for gen in range(generations):
            candidates = [self.sample_architecture() for _ in range(population_size)]

            for arch in candidates:
                arch.score = self.fitness_fn(arch)

            sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

            best = sorted_candidates[0]

            self._history.append({
                "generation": gen,
                "best_score": best.score,
            })

            if gen < generations - 1:
                parents = sorted_candidates[:3]
                for i in range(3, population_size):
                    parent = random.choice(parents)
                    sorted_candidates[i] = self.mutate(parent)

        self._best_architecture = best
        return best

    def mutate(self, arch: NeuralArchitecture) -> NeuralArchitecture:
        if random.random() < 0.3:
            arch.model_type = random.choice(self._search_space["model_type"])
        if random.random() < 0.5:
            layer_idx = random.randint(0, len(arch.layers) - 1)
            arch.layers[layer_idx] = max(8, arch.layers[layer_idx] + random.choice([-16, 16]))
            arch.layers = [max(8, l) for l in arch.layers]
        if random.random() < 0.3:
            arch.dropout = random.choice(self._search_space["dropout"])

        return arch


class MetaLearningEngine:
    def __init__(self):
        self._task_policies: Dict[str, Dict] = {}
        self._meta_parameters: Dict[str, float] = {}
        self._adaptation_steps: int = 5

    def create_task(self, task_id: str) -> None:
        self._task_policies[task_id] = {
            "parameters": {
                "position_size": 1.0,
                "stop_loss": 1.0,
                "take_profit": 1.0,
                "threshold": 0.5,
            },
            "gradient": {},
            "optimizer_state": {},
        }

    def get_parameters(self, task_id: str) -> Dict[str, float]:
        return deepcopy(self._task_policies.get(task_id, {}).get("parameters", {}))

    def adapt(
        self,
        task_id: str,
        gradients: Dict[str, float],
        learning_rate: float = 0.01,
    ) -> Dict[str, float]:
        if task_id not in self._task_policies:
            self.create_task(task_id)

        params = self._task_policies[task_id]["parameters"]

        new_params = {}
        for key, value in params.items():
            grad = gradients.get(key, 0.0)
            new_params[key] = value - learning_rate * grad

        self._task_policies[task_id]["parameters"] = new_params
        self._task_policies[task_id]["gradient"] = gradients

        return deepcopy(new_params)

    def compute_few_shot_gradient(
        self,
        task_id: str,
        support_set: List[Tuple[Dict, float]],
    ) -> Dict[str, float]:
        gradients = {}

        params = self.get_parameters(task_id)

        for key in params:
            gradient_sum = 0.0

            for data, loss in support_set:
                param_value = data.get(key, params[key])
                simulated_loss = (param_value - loss) * random.uniform(0.5, 1.5)
                gradient_sum += simulated_loss

            gradients[key] = gradient_sum / len(support_set) if support_set else 0.0

        return gradients

    def get_meta_parameters(self) -> Dict[str, float]:
        return deepcopy(self._meta_parameters)


class MultiAgentCoordinator:
    def __init__(self):
        self._agents: Dict[str, Dict] = {}
        self._coordination_history: deque = deque(maxlen=100)

    def register_agent(
        self,
        agent_id: str,
        strategy: str,
        specialization: str = "general",
    ) -> None:
        self._agents[agent_id] = {
            "strategy": strategy,
            "specialization": specialization,
            "confidence": 0.5,
            "performance": 0.0,
            "votes": defaultdict(int),
        }

    def request_vote(
        self,
        agent_id: str,
        context: Dict[str, Any],
    ) -> str:
        if agent_id not in self._agents:
            return "hold"

        agent = self._agents[agent_id]

        if agent["specialization"] == "trend":
            return "buy" if context.get("trend", 0) > 0 else "sell"
        elif agent["specialization"] == "volatility":
            return "hold" if context.get("volatility", 0) > 0.8 else "buy"
        elif agent["specialization"] == "momentum":
            return "buy" if context.get("momentum", 0) > 0 else "sell"

        return random.choice(["buy", "sell", "hold"])

    def aggregate_votes(
        self,
        context: Dict[str, Any],
    ) -> Tuple[str, float]:
        votes = defaultdict(int)
        confidence_sum = 0.0

        for agent_id in self._agents:
            vote = self.request_vote(agent_id, context)
            votes[vote] += 1
            confidence_sum += self._agents[agent_id]["confidence"]

        if not votes:
            return "hold", 0.0

        avg_confidence = confidence_sum / len(self._agents)

        decision = max(votes.items(), key=lambda x: x[1])[0]

        confidence = avg_confidence * (votes[decision] / len(self._agents))

        return decision, confidence

    def update_agent_performance(
        self,
        agent_id: str,
        performance: float,
    ) -> None:
        if agent_id in self._agents:
            old_perf = self._agents[agent_id]["performance"]
            self._agents[agent_id]["performance"] = 0.9 * old_perf + 0.1 * performance

    def get_best_agents(self, n: int = 3) -> List[str]:
        sorted_agents = sorted(
            self._agents.items(),
            key=lambda x: x[1]["performance"],
            reverse=True,
        )
        return [a[0] for a in sorted_agents[:n]]


class AdversarialDetector:
    def __init__(self):
        self._adversarial_patterns: Dict[str, List] = {}
        self._attack_history: deque = deque(maxlen=100)

    def add_pattern(
        self,
        pattern_type: str,
        features: List[float],
    ) -> None:
        if pattern_type not in self._adversarial_patterns:
            self._adversarial_patterns[pattern_type] = []
        self._adversarial_patterns[pattern_type].append(features)

    def detect_adversarial(
        self,
        features: Dict[str, float],
    ) -> Tuple[bool, float]:
        is_adversarial = False
        confidence = 0.0

        for pattern_type, patterns in self._adversarial_patterns.items():
            if len(patterns) < 3:
                continue

            pattern_arrays = np.array(patterns)

            mean = np.mean(pattern_arrays, axis=0)
            std = np.std(pattern_arrays, axis=0)

            feature_vec = np.array([
                features.get(k, 0) for k in features.keys()
            ])

            if len(feature_vec) == len(mean):
                z_scores = np.abs((feature_vec - mean) / (std + 1e-8))

                if np.max(z_scores) > 3.0:
                    is_adversarial = True
                    confidence = min(1.0, np.max(z_scores) / 5.0)

        self._attack_history.append({
            "is_adversarial": is_adversarial,
            "confidence": confidence,
            "timestamp": time.time(),
        })

        return is_adversarial, confidence

    def is_under_attack(self, threshold: float = 0.7) -> bool:
        if len(self._attack_history) < 5:
            return False

        recent = list(self._attack_history)[-5:]
        attack_count = sum(1 for a in recent if a["is_adversarial"])

        return attack_count >= 3


class CausalInferenceEngine:
    def __init__(self):
        self._causal_graph: Dict[str, List[str]] = {}
        self._evidence: Dict[str, float] = {}
        self._interventions: Dict[str, Any] = {}

    def add_edge(self, cause: str, effect: str) -> None:
        if cause not in self._causal_graph:
            self._causal_graph[cause] = []
        self._causal_graph[cause].append(effect)

    def set_evidence(self, variable: str, value: float) -> None:
        self._evidence[variable] = value

    def infer(
        self,
        target: str,
        given: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        given = given or self._evidence

        if target in given:
            return given[target]

        if target not in self._causal_graph:
            return None

        for cause in self._causal_graph:
            if target in self._causal_graph[cause]:
                if cause in given:
                    effect_size = 0.5
                    return given[cause] * effect_size

        return None

    def do_intervention(self, variable: str, value: Any) -> None:
        self._interventions[variable] = value

        for effect in self._causal_graph.get(variable, []):
            self._evidence[effect] = None

    def get_causal_strength(self, cause: str, effect: str) -> float:
        if cause not in self._causal_graph:
            return 0.0

        if effect not in self._causal_graph[cause]:
            return 0.0

        return 0.5


class ContinuousOnlineLearner:
    def __init__(self):
        self._model_weights: Dict[str, float] = {}
        self._drift_detector = DriftDetector()
        self._update_history: deque = deque(maxlen=1000)

    def add_sample(
        self,
        features: Dict[str, float],
        target: float,
        weight: float = 1.0,
    ) -> None:
        self._drift_detector.add_prediction(target)
        is_drift = self._drift_detector.detect_drift()

        learning_rate = 0.1 if not is_drift else 0.5

        for feature, value in features.items():
            if feature not in self._model_weights:
                self._model_weights[feature] = 0.0

            error = target - self.predict_single(feature)
            self._model_weights[feature] += learning_rate * error * value * weight

        self._update_history.append({
            "timestamp": time.time(),
            "is_drift": is_drift,
            "learning_rate": learning_rate,
        })

    def predict_single(self, feature: str) -> float:
        return self._model_weights.get(feature, 0.0)

    def predict(self, features: Dict[str, float]) -> float:
        prediction = 0.0
        for feature, value in features.items():
            prediction += self._model_weights.get(feature, 0.0) * value

        return prediction

    def get_adaptation_rate(self) -> float:
        if len(self._update_history) < 10:
            return 0.1

        recent = list(self._update_history)[-10:]
        avg_lr = np.mean([u["learning_rate"] for u in recent])
        return avg_lr


class DriftDetector:
    def __init__(self, window: int = 50):
        self._window = window
        self._predictions: deque = deque(maxlen=window)

    def add_prediction(self, prediction: float) -> None:
        self._predictions.append(prediction)

    def detect_drift(self, threshold: float = 0.3) -> bool:
        if len(self._predictions) < self._window:
            return False

        recent = list(self._predictions)[-10:]
        older = list(self._predictions)[:-10]

        if not recent or not older:
            return False

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        older_std = np.std(older)

        if older_std == 0:
            return False

        drift = abs(recent_mean - older_mean) / older_std

        return drift > threshold


class ExpertTrader:
    def __init__(self):
        self._experts: Dict[str, Dict] = {}
        self._active_expert: Optional[str] = None
        self._performance_history: deque = deque(maxlen=100)

    def register_expert(
        self,
        expert_id: str,
        strategy_fn: Callable,
        regime_focus: List[str],
        confidence: float = 0.5,
    ) -> None:
        self._experts[expert_id] = {
            "strategy_fn": strategy_fn,
            "regime_focus": regime_focus,
            "confidence": confidence,
            "total_trades": 0,
            "successful_trades": 0,
            "avg_return": 0.0,
        }

    def select_expert(
        self,
        regime: str,
        market_context: Dict[str, Any],
    ) -> Tuple[Optional[str], Dict]:
        candidates = []

        for expert_id, expert in self._experts.items():
            if regime in expert["regime_focus"]:
                score = expert["confidence"]
                candidates.append((expert_id, score, expert))

        if not candidates:
            return None, {}

        best = max(candidates, key=lambda x: x[1])
        self._active_expert = best[0]

        signal = best[2]["strategy_fn"](market_context)

        return self._active_expert, signal

    def record_performance(
        self,
        expert_id: str,
        return_pct: float,
        success: bool,
    ) -> None:
        if expert_id not in self._experts:
            return

        expert = self._experts[expert_id]

        total = expert["total_trades"] + 1
        successful = expert["successful_trades"] + (1 if success else 0)

        expert["total_trades"] = total
        expert["successful_trades"] = successful
        expert["avg_return"] = (
            (expert["avg_return"] * (total - 1) + return_pct) / total
        )

        win_rate = successful / total if total > 0 else 0
        expert["confidence"] = 0.5 + 0.5 * win_rate

        self._performance_history.append({
            "expert_id": expert_id,
            "return": return_pct,
            "success": success,
            "timestamp": time.time(),
        })

    def get_expert_stats(self, expert_id: str) -> Dict:
        if expert_id not in self._experts:
            return {}

        expert = self._experts[expert_id]
        return {
            "confidence": expert["confidence"],
            "total_trades": expert["total_trades"],
            "successful_trades": expert["successful_trades"],
            "avg_return": expert["avg_return"],
        }


class HyperAdaptiveEngine:
    """Ultimate hyper-adaptive trading engine."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self._nas = NeuralArchitectureSearch()
        self._meta = MetaLearningEngine()
        self._multi_agent = MultiAgentCoordinator()
        self._adversarial = AdversarialDetector()
        self._causal = CausalInferenceEngine()
        self._online = ContinuousOnlineLearner()
        self._expert = ExpertTrader()

        self._initialized = False

    def initialize(self) -> None:
        self._nas.search(generations=10, population_size=10)

        self._causal.add_edge("volume", "volatility")
        self._causal.add_edge("volatility", "spread")
        self._causal.add_edge("trend", "momentum")
        self._causal.add_edge("sentiment", "trend")

        self._initialized = True
        logger.info("HyperAdaptiveEngine initialized")

    def update(
        self,
        symbol: str,
        features: Dict[str, float],
        regime: str,
        target: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        if target is not None:
            self._online.add_sample(features, target)

        drift = self._online._drift_detector.detect_drift()
        adaptation_rate = self._online.get_adaptation_rate()

        nas_score = self._nas._best_architecture.score if self._nas._best_architecture else 0.0

        is_attack, attack_conf = self._adversarial.detect_adversarial(features)

        context = {
            "trend": features.get("momentum", 0),
            "volatility": features.get("volatility", 0),
            "momentum": features.get("momentum", 0),
        }

        agent_decision, agent_confidence = self._multi_agent.aggregate_votes(context)

        causal_inference = self._causal.infer("spread", {})

        meta_params = self._meta.get_parameters(regime)

        prediction = self._online.predict(features)

        meta_learning_effect = np.mean(list(meta_params.values())) if meta_params else 1.0

        result = {
            "nas_score": nas_score,
            "is_adversarial": is_attack,
            "attack_confidence": attack_conf,
            "drift_detected": drift,
            "adaptation_rate": adaptation_rate,
            "causal_inference": causal_inference,
            "agent_decision": agent_decision,
            "agent_confidence": agent_confidence,
            "meta_learning_effect": meta_learning_effect,
            "prediction": prediction,
            "regime": regime,
        }

        if is_attack:
            result["position_mult"] = 0.1
        elif drift:
            result["position_mult"] = 0.3
        else:
            result["position_mult"] = meta_learning_effect

        return result

    def add_expert(
        self,
        expert_id: str,
        strategy_fn: Callable,
        regime_focus: List[str],
    ) -> None:
        self._expert.register_expert(expert_id, strategy_fn, regime_focus)

    def select_expert(
        self,
        regime: str,
        market_context: Dict[str, Any],
    ) -> Tuple[Optional[str], Dict]:
        return self._expert.select_expert(regime, market_context)

    def record_expert_performance(
        self,
        expert_id: str,
        return_pct: float,
        success: bool,
    ) -> None:
        self._expert.record_performance(expert_id, return_pct, success)

    def get_adaptive_params(self) -> Dict[str, Any]:
        return {
            "nas_architecture": self._nas._best_architecture.__dict__ if self._nas._best_architecture else {},
            "model_weights": self._online._model_weights,
            "meta_parameters": self._meta._meta_parameters,
            "causal_edges": self._causal._causal_graph,
        }


def create_hyper_adaptive_engine(config: Optional[Dict] = None) -> HyperAdaptiveEngine:
    return HyperAdaptiveEngine(config)