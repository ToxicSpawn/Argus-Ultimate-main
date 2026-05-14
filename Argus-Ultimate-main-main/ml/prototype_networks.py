# pyright: reportMissingImports=false
"""
Prototype Networks for Few-Shot Learning in Argus Trading.

This module implements prototype-based few-shot learning to quickly
adapt to new patterns and market conditions with minimal examples.
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


@dataclass
class Prototype:
    """A prototype representing a pattern class."""
    pattern_type: str
    embedding: NDArray[np.float64]
    examples_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrototypeConfig:
    """Configuration for prototype networks."""
    embedding_dim: int = 64
    distance_metric: str = "euclidean"  # euclidean, cosine, manhattan
    temperature: float = 1.0
    adaptation_rate: float = 0.1


class PrototypicalNetwork:
    """Prototypical network for few-shot learning."""

    def __init__(self, config: Optional[PrototypeConfig] = None):
        """Initialize the prototypical network."""
        self.config = config or PrototypeConfig()
        self.prototypes: Dict[str, Prototype] = {}
        self.embedding_network = self._create_embedding_network()
        self.training_history: List[Dict[str, Any]] = []

    def _create_embedding_network(self) -> Dict[str, NDArray[np.float64]]:
        """Create embedding network weights."""
        return {
            "w1": np.random.randn(8, 32) * 0.1,
            "w2": np.random.randn(32, self.config.embedding_dim) * 0.1
        }

    def embed(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Embed input data into prototype space."""
        # Ensure correct input dimension
        if len(data) < 8:
            padded = np.zeros(8)
            padded[:len(data)] = data
            data = padded
        elif len(data) > 8:
            data = data[:8]

        # Forward pass through embedding network
        hidden = np.tanh(data @ self.embedding_network["w1"])
        embedding = np.tanh(hidden @ self.embedding_network["w2"])

        # Normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def compute_distance(self, 
                        embedding1: NDArray[np.float64],
                        embedding2: NDArray[np.float64]) -> float:
        """Compute distance between embeddings."""
        if self.config.distance_metric == "euclidean":
            return float(np.linalg.norm(embedding1 - embedding2))
        elif self.config.distance_metric == "cosine":
            return 1.0 - float(np.dot(embedding1, embedding2))
        elif self.config.distance_metric == "manhattan":
            return float(np.sum(np.abs(embedding1 - embedding2)))
        else:
            return float(np.linalg.norm(embedding1 - embedding2))

    def learn_prototype(self, 
                       pattern_type: str,
                       examples: List[NDArray[np.float64]]) -> Prototype:
        """Learn a prototype from examples."""
        embeddings = [self.embed(example) for example in examples]
        prototype_embedding = np.mean(embeddings, axis=0)

        # Normalize prototype
        norm = np.linalg.norm(prototype_embedding) + 1e-8
        prototype_embedding = prototype_embedding / norm

        prototype = Prototype(
            pattern_type=pattern_type,
            embedding=prototype_embedding,
            examples_count=len(examples),
            metadata={
                "std": float(np.std([np.linalg.norm(e - prototype_embedding) for e in embeddings])),
                "examples": len(examples)
            }
        )

        self.prototypes[pattern_type] = prototype
        logger.info(f"Learned prototype for '{pattern_type}' with {len(examples)} examples")
        
        return prototype

    def update_prototype(self,
                        pattern_type: str,
                        new_example: NDArray[np.float64]) -> None:
        """Update existing prototype with new example."""
        if pattern_type not in self.prototypes:
            self.learn_prototype(pattern_type, [new_example])
            return

        prototype = self.prototypes[pattern_type]
        new_embedding = self.embed(new_example)

        # Exponential moving average update
        updated_embedding = (
            prototype.embedding * (1 - self.config.adaptation_rate) +
            new_embedding * self.config.adaptation_rate
        )

        # Normalize
        norm = np.linalg.norm(updated_embedding) + 1e-8
        updated_embedding = updated_embedding / norm

        # Update prototype
        prototype.embedding = updated_embedding
        prototype.examples_count += 1

        logger.debug(f"Updated prototype for '{pattern_type}' (total examples: {prototype.examples_count})")

    def classify(self, 
                data: NDArray[np.float64]) -> Tuple[str, float, Dict[str, float]]:
        """Classify data by distance to prototypes."""
        if not self.prototypes:
            return "unknown", 0.0, {}

        embedding = self.embed(data)

        # Compute distances to all prototypes
        distances = {}
        for pattern_type, prototype in self.prototypes.items():
            distance = self.compute_distance(embedding, prototype.embedding)
            distances[pattern_type] = distance

        # Convert distances to similarities (softmax)
        distances_array = np.array(list(distances.values()))
        similarities = np.exp(-distances_array / self.config.temperature)
        similarities = similarities / (np.sum(similarities) + 1e-8)

        # Create similarity dict
        similarity_dict = dict(zip(distances.keys(), similarities))

        # Find best match
        best_pattern = max(similarity_dict, key=similarity_dict.get)
        confidence = similarity_dict[best_pattern]

        return best_pattern, confidence, similarity_dict

    def few_shot_learn(self,
                      support_set: Dict[str, List[NDArray[np.float64]]]) -> None:
        """Learn prototypes from a few-shot support set."""
        for pattern_type, examples in support_set.items():
            self.learn_prototype(pattern_type, examples)

    def get_prototype_info(self) -> Dict[str, Any]:
        """Get information about all prototypes."""
        return {
            "num_prototypes": len(self.prototypes),
            "prototypes": {
                name: {
                    "examples_count": proto.examples_count,
                    "metadata": proto.metadata
                }
                for name, proto in self.prototypes.items()
            },
            "embedding_dim": self.config.embedding_dim,
            "distance_metric": self.config.distance_metric
        }


class FewShotTradingClassifier:
    """Few-shot trading classifier using prototypical networks."""

    def __init__(self, prototype_network: Optional[PrototypicalNetwork] = None):
        """Initialize the few-shot classifier."""
        self.prototype_network = prototype_network or PrototypicalNetwork()
        self.action_map = {
            "trending_up": 1,  # Buy
            "trending_down": 2,  # Sell
            "ranging": 0,  # Hold
            "volatile": 3,  # Hedge
            "breakout_up": 1,
            "breakout_down": 2,
            "overbought": 2,
            "oversold": 1
        }
        self.classification_history: List[Dict[str, Any]] = []

    def train_on_patterns(self, 
                         patterns: Dict[str, List[NDArray[np.float64]]]) -> None:
        """Train on known market patterns."""
        self.prototype_network.few_shot_learn(patterns)
        logger.info(f"Trained on {len(patterns)} pattern types")

    def classify_and_decide(self, 
                           market_state: NDArray[np.float64]) -> Tuple[int, str, float]:
        """Classify market state and return trading decision."""
        pattern_type, confidence, all_similarities = self.prototype_network.classify(market_state)

        # Map pattern to action
        action = self.action_map.get(pattern_type, 0)

        # Record classification
        self.classification_history.append({
            "pattern": pattern_type,
            "action": action,
            "confidence": confidence,
            "similarities": all_similarities
        })

        return action, pattern_type, confidence

    def learn_from_feedback(self,
                           market_state: NDArray[np.float64],
                           correct_pattern: str) -> None:
        """Learn from feedback by updating prototype."""
        self.prototype_network.update_prototype(correct_pattern, market_state)
        logger.debug(f"Learned from feedback: {correct_pattern}")


__all__ = [
    "PrototypicalNetwork",
    "FewShotTradingClassifier",
    "PrototypeConfig",
    "Prototype"
]