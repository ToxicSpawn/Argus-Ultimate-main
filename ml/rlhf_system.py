# pyright: reportMissingImports=false
"""
Reinforcement Learning from Human Feedback (RLHF) System for Argus Trading.

This module implements RLHF to learn from expert trader decisions and feedback,
improving model performance through human guidance.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of human feedback."""
    BINARY_RATING = auto()  # Good/Bad
    RANKING = auto()  # Rank multiple decisions
    PREFERENCE = auto()  # A vs B preference
    DEMONSTRATION = auto()  # Expert demonstration
    CORRECTION = auto()  # Correct the model's decision


@dataclass
class HumanFeedback:
    """A piece of human feedback."""
    feedback_id: str
    feedback_type: FeedbackType
    timestamp: float
    expert_id: str
    market_state: NDArray[np.float64]
    model_decision: int
    human_decision: Optional[int] = None
    rating: Optional[float] = None  # 0-1 for binary, or rank
    preferences: Optional[Dict[int, float]] = None  # decision -> preference score
    comments: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertProfile:
    """Profile of an expert trader."""
    expert_id: str
    expertise_level: float  # 0-1
    specializations: List[str]
    win_rate: float
    total_trades: int
    feedback_count: int = 0
    reliability_score: float = 1.0


@dataclass
class RewardModelTrainingResult:
    """Result of reward model training."""
    accuracy: float
    loss: float
    preference_accuracy: float
    examples_processed: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RewardModel:
    """Reward model trained on human preferences."""

    def __init__(self, state_dim: int = 8):
        self.state_dim = state_dim
        self.parameters = np.random.randn(state_dim * 10) * 0.1
        self.training_history: List[RewardModelTrainingResult] = []
        
    def predict_reward(self, state: NDArray[np.float64], action: int) -> float:
        """Predict reward for a state-action pair."""
        # Encode state and action
        features = np.concatenate([state, np.array([action])])
        features = features[:self.state_dim]
        
        # Simple linear model simulation
        reward = np.dot(features, self.parameters[:len(features)])
        return float(np.tanh(reward))  # Bound between -1 and 1
    
    def predict_preference(self, state: NDArray[np.float64], 
                          action_a: int, action_b: int) -> float:
        """Predict preference probability for action_a over action_b."""
        reward_a = self.predict_reward(state, action_a)
        reward_b = self.predict_reward(state, action_b)
        
        # Softmax preference
        diff = reward_a - reward_b
        pref_prob = 1.0 / (1.0 + np.exp(-diff * 5))  # Temperature = 0.2
        return pref_prob
    
    def update(self, gradients: NDArray[np.float64], learning_rate: float = 0.001) -> None:
        """Update reward model parameters."""
        self.parameters += gradients * learning_rate


class RLHFSystem:
    """Reinforcement Learning from Human Feedback system."""

    def __init__(self, reward_model: Optional[RewardModel] = None):
        """Initialize the RLHF system."""
        self.reward_model = reward_model or RewardModel()
        self.feedback_buffer: List[HumanFeedback] = []
        self.experts: Dict[str, ExpertProfile] = {}
        self.training_count = 0
        
        # Initialize some default experts
        self._initialize_experts()
        
    def _initialize_experts(self) -> None:
        """Initialize expert profiles."""
        default_experts = [
            ExpertProfile(
                expert_id="expert_1",
                expertise_level=0.9,
                specializations=["trend", "momentum"],
                win_rate=0.65,
                total_trades=1000
            ),
            ExpertProfile(
                expert_id="expert_2",
                expertise_level=0.85,
                specializations=["mean_reversion", "range"],
                win_rate=0.62,
                total_trades=800
            ),
            ExpertProfile(
                expert_id="expert_3",
                expertise_level=0.88,
                specializations=["breakout", "volatility"],
                win_rate=0.58,
                total_trades=1200
            ),
        ]
        
        for expert in default_experts:
            self.experts[expert.expert_id] = expert
    
    def collect_feedback(self, 
                        market_state: NDArray[np.float64],
                        model_decision: int,
                        feedback_type: FeedbackType,
                        expert_id: str = "expert_1",
                        **kwargs) -> HumanFeedback:
        """Collect human feedback on a decision."""
        feedback = HumanFeedback(
            feedback_id=f"feedback_{len(self.feedback_buffer) + 1}",
            feedback_type=feedback_type,
            timestamp=np.random.uniform(0, 1000),
            expert_id=expert_id,
            market_state=market_state.copy(),
            model_decision=model_decision,
            **kwargs
        )
        
        self.feedback_buffer.append(feedback)
        
        # Update expert profile
        if expert_id in self.experts:
            self.experts[expert_id].feedback_count += 1
        
        logger.info(f"Collected {feedback_type.name} feedback from {expert_id}")
        return feedback
    
    def collect_rating(self, market_state: NDArray[np.float64],
                      model_decision: int, rating: float,
                      expert_id: str = "expert_1") -> HumanFeedback:
        """Collect binary rating (good/bad)."""
        return self.collect_feedback(
            market_state=market_state,
            model_decision=model_decision,
            feedback_type=FeedbackType.BINARY_RATING,
            expert_id=expert_id,
            rating=rating
        )
    
    def collect_preference(self, market_state: NDArray[np.float64],
                          action_a: int, action_b: int, preferred: int,
                          expert_id: str = "expert_1") -> HumanFeedback:
        """Collect preference feedback (A vs B)."""
        preferences = {action_a: 1.0 if preferred == action_a else 0.0,
                      action_b: 1.0 if preferred == action_b else 0.0}
        
        return self.collect_feedback(
            market_state=market_state,
            model_decision=action_a,
            feedback_type=FeedbackType.PREFERENCE,
            expert_id=expert_id,
            human_decision=preferred,
            preferences=preferences
        )
    
    def collect_demonstration(self, market_state: NDArray[np.float64],
                             expert_decision: int,
                             expert_id: str = "expert_1") -> HumanFeedback:
        """Collect expert demonstration."""
        return self.collect_feedback(
            market_state=market_state,
            model_decision=expert_decision,
            feedback_type=FeedbackType.DEMONSTRATION,
            expert_id=expert_id,
            human_decision=expert_decision
        )
    
    def train_reward_model(self, epochs: int = 10) -> RewardModelTrainingResult:
        """Train the reward model on collected feedback."""
        if len(self.feedback_buffer) < 10:
            logger.warning("Not enough feedback for training")
            return RewardModelTrainingResult(accuracy=0.0, loss=0.0, 
                                           preference_accuracy=0.0, examples_processed=0)
        
        logger.info(f"Training reward model on {len(self.feedback_buffer)} feedback examples")
        
        total_loss = 0.0
        correct_predictions = 0
        preference_correct = 0
        preference_total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for feedback in self.feedback_buffer:
                if feedback.feedback_type == FeedbackType.BINARY_RATING:
                    # Train on binary rating
                    predicted_reward = self.reward_model.predict_reward(
                        feedback.market_state, feedback.model_decision
                    )
                    
                    # Rating is 0 or 1, predicted_reward is -1 to 1
                    target = feedback.rating * 2 - 1  # Convert to -1, 1
                    loss = (predicted_reward - target) ** 2
                    epoch_loss += loss
                    
                    # Check if prediction is correct
                    predicted_rating = 1 if predicted_reward > 0 else 0
                    if predicted_rating == feedback.rating:
                        correct_predictions += 1
                
                elif feedback.feedback_type == FeedbackType.PREFERENCE:
                    # Train on preference
                    if feedback.preferences and feedback.human_decision is not None:
                        # Get the non-preferred action
                        actions = list(feedback.preferences.keys())
                        non_preferred = actions[0] if actions[1] == feedback.human_decision else actions[1]
                        
                        pref_prob = self.reward_model.predict_preference(
                            feedback.market_state, feedback.human_decision, non_preferred
                        )
                        
                        loss = -(np.log(pref_prob + 1e-8))
                        epoch_loss += loss
                        
                        preference_total += 1
                        if pref_prob > 0.5:
                            preference_correct += 1
            
            total_loss += epoch_loss / len(self.feedback_buffer)
            
            # Simulate gradient update
            gradients = np.random.randn(len(self.reward_model.parameters)) * 0.01
            self.reward_model.update(gradients)
        
        self.training_count += 1
        
        result = RewardModelTrainingResult(
            accuracy=correct_predictions / len(self.feedback_buffer) if self.feedback_buffer else 0,
            loss=total_loss / epochs,
            preference_accuracy=preference_correct / preference_total if preference_total > 0 else 0,
            examples_processed=len(self.feedback_buffer),
            metadata={
                "epochs": epochs,
                "training_count": self.training_count,
                "feedback_types": {
                    ft.name: sum(1 for f in self.feedback_buffer if f.feedback_type == ft)
                    for ft in FeedbackType
                }
            }
        )
        
        self.reward_model.training_history.append(result)
        
        logger.info(f"Reward model trained: accuracy={result.accuracy:.2%}, "
                   f"preference_accuracy={result.preference_accuracy:.2%}")
        
        return result
    
    def get_feedback_quality_score(self) -> float:
        """Calculate quality score of collected feedback."""
        if not self.feedback_buffer:
            return 0.0
        
        # Factors: amount of feedback, expert reliability, diversity
        amount_score = min(len(self.feedback_buffer) / 100, 1.0)
        
        # Expert reliability
        expert_reliability = []
        for feedback in self.feedback_buffer:
            if feedback.expert_id in self.experts:
                expert_reliability.append(self.experts[feedback.expert_id].reliability_score)
        reliability_score = np.mean(expert_reliability) if expert_reliability else 0.5
        
        # Feedback type diversity
        feedback_types = set(f.feedback_type for f in self.feedback_buffer)
        diversity_score = len(feedback_types) / len(FeedbackType)
        
        return (amount_score * 0.3 + reliability_score * 0.4 + diversity_score * 0.3)
    
    def get_expert_performance(self) -> Dict[str, Any]:
        """Get expert performance summary."""
        summary = {}
        for expert_id, expert in self.experts.items():
            summary[expert_id] = {
                "expertise_level": expert.expertise_level,
                "specializations": expert.specializations,
                "win_rate": expert.win_rate,
                "feedback_count": expert.feedback_count,
                "reliability": expert.reliability_score
            }
        return summary


class OnlineRLHF(RLHFSystem):
    """Online RLHF that continuously learns from feedback."""

    def __init__(self, reward_model: Optional[RewardModel] = None, 
                 update_frequency: int = 50):
        super().__init__(reward_model)
        self.update_frequency = update_frequency
        self.feedback_since_last_update = 0
        
    def collect_and_update(self, market_state: NDArray[np.float64],
                          model_decision: int, rating: float,
                          expert_id: str = "expert_1") -> Optional[RewardModelTrainingResult]:
        """Collect feedback and update if needed."""
        self.collect_rating(market_state, model_decision, rating, expert_id)
        self.feedback_since_last_update += 1
        
        if self.feedback_since_last_update >= self.update_frequency:
            result = self.train_reward_model(epochs=5)
            self.feedback_since_last_update = 0
            return result
        
        return None


__all__ = [
    "RLHFSystem",
    "OnlineRLHF",
    "RewardModel",
    "HumanFeedback",
    "ExpertProfile",
    "FeedbackType",
    "RewardModelTrainingResult"
]