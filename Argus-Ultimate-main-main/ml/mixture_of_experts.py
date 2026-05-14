# pyright: reportMissingImports=false
"""
Mixture of Experts (MoE) System for Argus Trading.

This module implements a mixture of experts architecture where multiple
specialized models are combined with intelligent gating for better decisions.
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


class ExpertType(Enum):
    """Types of specialized experts."""
    MOMENTUM = auto()
    MEAN_REVERSION = auto()
    VOLATILITY = auto()
    BREAKOUT = auto()
    SENTIMENT = auto()
    TECHNICAL = auto()
    FUNDAMENTAL = auto()
    REGIME_AWARE = auto()


@dataclass
class ExpertOutput:
    """Output from an expert."""
    expert_type: ExpertType
    action: int  # 0: hold, 1: buy, 2: sell, 3: hedge
    confidence: float
    features: NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts."""
    num_experts: int = 8
    top_k: int = 3  # Number of experts to use
    gating_hidden_dim: int = 64
    load_balance_weight: float = 0.01  # Weight for load balancing loss
    noise_epsilon: float = 0.1  # Noise for exploration


class Expert:
    """Base class for specialized experts."""

    def __init__(self, expert_type: ExpertType, input_dim: int = 8):
        self.expert_type = expert_type
        self.input_dim = input_dim
        self.parameters = np.random.randn(input_dim * 4) * 0.1
        self.performance_history: List[float] = []
        self.call_count = 0

    def predict(self, state: NDArray[np.float64]) -> ExpertOutput:
        """Make a prediction based on specialized knowledge."""
        # Each expert has different specialization logic
        action, confidence = self._specialized_predict(state)
        
        self.call_count += 1
        
        return ExpertOutput(
            expert_type=self.expert_type,
            action=action,
            confidence=confidence,
            features=state[:min(8, len(state))],
            metadata={"call_count": self.call_count}
        )

    def _specialized_predict(self, state: NDArray[np.float64]) -> Tuple[int, float]:
        """Specialized prediction logic."""
        if self.expert_type == ExpertType.MOMENTUM:
            # Momentum expert looks for trending markets
            if len(state) >= 3:
                trend = np.mean(np.diff(state[-3:]))
                if trend > 0.1:
                    return 1, 0.8  # Buy
                elif trend < -0.1:
                    return 2, 0.8  # Sell
            return 0, 0.5  # Hold

        elif self.expert_type == ExpertType.MEAN_REVERSION:
            # Mean reversion expert looks for overbought/oversold
            mean = np.mean(state)
            current = state[-1]
            if current > mean * 1.2:
                return 2, 0.7  # Sell (overbought)
            elif current < mean * 0.8:
                return 1, 0.7  # Buy (oversold)
            return 0, 0.6  # Hold

        elif self.expert_type == ExpertType.VOLATILITY:
            # Volatility expert responds to high volatility
            volatility = np.std(state)
            if volatility > 1.0:
                return 3, 0.8  # Hedge
            return 0, 0.5

        elif self.expert_type == ExpertType.BREAKOUT:
            # Breakout expert looks for breakouts
            if len(state) >= 5:
                recent_high = np.max(state[-5:])
                recent_low = np.min(state[-5:])
                current = state[-1]
                
                if current > recent_high * 0.99:
                    return 1, 0.75  # Buy breakout
                elif current < recent_low * 1.01:
                    return 2, 0.75  # Sell breakdown
            return 0, 0.5

        elif self.expert_type == ExpertType.REGIME_AWARE:
            # Regime-aware expert adapts to market regime
            volatility = np.std(state)
            trend = np.mean(np.diff(state)) if len(state) > 1 else 0
            
            if volatility > 0.5 and abs(trend) < 0.1:
                return 3, 0.7  # Hedge in volatile, non-trending
            elif trend > 0.1:
                return 1, 0.7  # Buy in uptrend
            elif trend < -0.1:
                return 2, 0.7  # Sell in downtrend
            return 0, 0.6

        else:
            # Default expert behavior
            return random.randint(0, 3), 0.5

    def update_performance(self, reward: float) -> None:
        """Update expert performance tracking."""
        self.performance_history.append(reward)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)


class GatingNetwork:
    """Gating network that selects which experts to use."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_experts: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Network weights
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.w2 = np.random.randn(hidden_dim, num_experts) * 0.1

    def forward(self, state: NDArray[np.float64], noise: float = 0.0) -> NDArray[np.float64]:
        """Compute gating weights for each expert."""
        # Ensure state is correct dimension
        if len(state) < self.input_dim:
            padded = np.zeros(self.input_dim)
            padded[:len(state)] = state
            state = padded
        elif len(state) > self.input_dim:
            state = state[:self.input_dim]

        # Forward pass
        hidden = np.tanh(state @ self.w1)
        logits = hidden @ self.w2

        # Add noise for exploration
        if noise > 0:
            logits += np.random.randn(*logits.shape) * noise

        # Softmax
        weights = np.exp(logits - np.max(logits))
        weights = weights / (np.sum(weights) + 1e-8)

        return weights


class MixtureOfExperts:
    """Mixture of Experts system for trading."""

    def __init__(self, config: Optional[MoEConfig] = None):
        """Initialize the MoE system."""
        self.config = config or MoEConfig()

        # Initialize experts
        expert_types = list(ExpertType)[:self.config.num_experts]
        self.experts = [Expert(et) for et in expert_types]

        # Initialize gating network
        self.gating_network = GatingNetwork(num_experts=self.config.num_experts)

        self.decision_history: List[Dict[str, Any]] = []
        self.expert_usage: Dict[str, int] = {et.name: 0 for et in expert_types}

    def predict(self, state: NDArray[np.float64]) -> Tuple[int, Dict[str, Any]]:
        """Make a prediction using the mixture of experts."""
        # Get gating weights
        gating_weights = self.gating_network.forward(state, self.config.noise_epsilon)

        # Select top-k experts
        top_k_indices = np.argsort(gating_weights)[-self.config.top_k:][::-1]
        top_k_weights = gating_weights[top_k_indices]
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / (np.sum(top_k_weights) + 1e-8)

        # Get expert predictions
        expert_outputs = []
        for idx in top_k_indices:
            expert = self.experts[idx]
            output = expert.predict(state)
            expert_outputs.append((output, top_k_weights[len(expert_outputs)]))

        # Weighted combination of expert actions
        action_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for output, weight in expert_outputs:
            action_scores[output.action] += weight * output.confidence

            # Track expert usage
            self.expert_usage[output.expert_type.name] += 1

        # Final decision
        final_action = max(action_scores, key=action_scores.get)
        total_score = sum(action_scores.values())
        final_confidence = action_scores[final_action] / total_score if total_score > 0 else 0.5

        # Build metadata
        metadata = {
            "gating_weights": gating_weights.tolist(),
            "top_k_experts": [oe[0].expert_type.name for oe in expert_outputs],
            "expert_decisions": [
                {
                    "expert": oe[0].expert_type.name,
                    "action": oe[0].action,
                    "confidence": oe[0].confidence,
                    "weight": oe[1]
                }
                for oe in expert_outputs
            ],
            "action_scores": action_scores,
            "final_confidence": final_confidence
        }

        # Record decision
        self.decision_history.append({
            "action": final_action,
            "confidence": final_confidence,
            "experts_used": len(expert_outputs)
        })

        return final_action, metadata

    def update_experts(self, reward: float) -> None:
        """Update all experts with reward signal."""
        for expert in self.experts:
            expert.update_performance(reward)

    def get_expert_performance(self) -> Dict[str, Any]:
        """Get performance summary for all experts."""
        summary = {}
        
        for expert in self.experts:
            perf = expert.performance_history
            summary[expert.expert_type.name] = {
                "call_count": expert.call_count,
                "avg_reward": np.mean(perf) if perf else 0.0,
                "recent_reward": np.mean(perf[-10:]) if len(perf) >= 10 else 0.0,
                "usage_count": self.expert_usage.get(expert.expert_type.name, 0)
            }

        return summary

    def load_balance_loss(self, gating_weights: NDArray[np.float64]) -> float:
        """Compute load balancing loss to encourage expert utilization."""
        # Mean of gating weights across batch (simulated as single sample)
        expert_utilization = gating_weights
        
        # Target uniform distribution
        target = np.ones(self.config.num_experts) / self.config.num_experts
        
        # KL divergence
        kl_div = np.sum(target * np.log(target / (expert_utilization + 1e-8) + 1e-8))
        
        return kl_div * self.config.load_balance_weight


class AdaptiveMoE(MixtureOfExperts):
    """Adaptive Mixture of Experts that learns expert selection."""

    def __init__(self, config: Optional[MoEConfig] = None):
        super().__init__(config)
        self.expert_rewards: Dict[str, List[float]] = {
            et.name: [] for et in ExpertType
        }

    def predict_with_adaptation(self, 
                               state: NDArray[np.float64]) -> Tuple[int, Dict[str, Any]]:
        """Predict with adaptive expert selection."""
        # Adjust gating based on recent expert performance
        self._update_gating_weights()

        # Make prediction
        return self.predict(state)

    def _update_gating_weights(self) -> None:
        """Update gating network based on expert performance."""
        # Simple update: boost weights of high-performing experts
        for i, expert in enumerate(self.experts):
            if expert.performance_history:
                recent_perf = np.mean(expert.performance_history[-10:])
                
                # Adjust gating network weights
                if recent_perf > 0:
                    # Positive reinforcement
                    self.gating_network.w2[:, i] *= 1.01
                else:
                    # Negative reinforcement
                    self.gating_network.w2[:, i] *= 0.99

    def record_expert_reward(self, expert_type: ExpertType, reward: float) -> None:
        """Record reward for a specific expert."""
        self.expert_rewards[expert_type.name].append(reward)
        
        # Find and update the expert
        for expert in self.experts:
            if expert.expert_type == expert_type:
                expert.update_performance(reward)
                break


__all__ = [
    "MixtureOfExperts",
    "AdaptiveMoE",
    "MoEConfig",
    "Expert",
    "ExpertType",
    "ExpertOutput",
    "GatingNetwork"
]