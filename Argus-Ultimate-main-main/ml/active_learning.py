# pyright: reportMissingImports=false
"""
Active Learning System for Argus Trading.

This module implements active learning to intelligently select the most informative
examples for labeling, reducing the amount of data needed for training.
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


class AcquisitionFunction(Enum):
    """Functions for selecting samples to label."""
    UNCERTAINTY_SAMPLING = auto()  # Most uncertain predictions
    DENSITY_WEIGHTED = auto()  # Uncertain + representative
    QUERY_BY_COMMITTEE = auto()  # Max disagreement between models
    EXPECTED_ERROR_REDUCTION = auto()  # Minimize expected future error
    BALANCE = auto()  # Balance exploration and exploitation


@dataclass
class LabeledSample:
    """A labeled data sample."""
    sample_id: str
    features: NDArray[np.float64]
    label: int
    confidence: float
    acquisition_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    acquisition_function: AcquisitionFunction = AcquisitionFunction.UNCERTAINTY_SAMPLING
    batch_size: int = 10
    initial_pool_size: int = 100
    max_samples: int = 1000
    uncertainty_threshold: float = 0.3
    diversity_weight: float = 0.3


class ActiveLearner:
    """Active learning system for intelligent sample selection."""

    def __init__(self, config: Optional[ActiveLearningConfig] = None):
        """Initialize the active learner."""
        self.config = config or ActiveLearningConfig()
        self.unlabeled_pool: List[NDArray[np.float64]] = []
        self.labeled_samples: List[LabeledSample] = []
        self.acquisition_history: List[Dict[str, Any]] = []
        self.model_uncertainties: Dict[str, float] = {}  # sample_id -> uncertainty
        
    def initialize_pool(self, samples: List[NDArray[np.float64]]) -> None:
        """Initialize the unlabeled pool."""
        self.unlabeled_pool = [s.copy() for s in samples]
        logger.info(f"Initialized pool with {len(samples)} unlabeled samples")
    
    def compute_acquisition_scores(self, 
                                   model_predictions: List[Tuple[NDArray[np.float64], float]],
                                   sample_indices: Optional[List[int]] = None) -> NDArray[np.float64]:
        """Compute acquisition scores for samples."""
        if sample_indices is None:
            sample_indices = list(range(len(self.unlabeled_pool)))
        
        scores = np.zeros(len(sample_indices))
        
        if self.config.acquisition_function == AcquisitionFunction.UNCERTAINTY_SAMPLING:
            scores = self._uncertainty_sampling(model_predictions, sample_indices)
        elif self.config.acquisition_function == AcquisitionFunction.DENSITY_WEIGHTED:
            scores = self._density_weighted(model_predictions, sample_indices)
        elif self.config.acquisition_function == AcquisitionFunction.QUERY_BY_COMMITTEE:
            scores = self._query_by_committee(model_predictions, sample_indices)
        elif self.config.acquisition_function == AcquisitionFunction.BALANCE:
            scores = self._balanced_acquisition(model_predictions, sample_indices)
        else:
            scores = self._uncertainty_sampling(model_predictions, sample_indices)
        
        return scores
    
    def _uncertainty_sampling(self, 
                             predictions: List[Tuple[NDArray[np.float64], float]],
                             indices: List[int]) -> NDArray[np.float64]:
        """Uncertainty-based acquisition."""
        scores = np.zeros(len(indices))
        
        for i, idx in enumerate(indices):
            if i < len(predictions):
                logits, uncertainty = predictions[i]
                # Higher uncertainty = higher score
                scores[i] = uncertainty
        
        return scores
    
    def _density_weighted(self,
                         predictions: List[Tuple[NDArray[np.float64], float]],
                         indices: List[int]) -> NDArray[np.float64]:
        """Density-weighted acquisition."""
        scores = np.zeros(len(indices))
        
        for i, idx in enumerate(indices):
            if i < len(predictions):
                logits, uncertainty = predictions[i]
                
                # Density estimation (simplified)
                sample = self.unlabeled_pool[idx]
                density = self._estimate_density(sample)
                
                # Combine uncertainty and density
                scores[i] = uncertainty * (1 - self.config.diversity_weight) + \
                           density * self.config.diversity_weight
        
        return scores
    
    def _estimate_density(self, sample: NDArray[np.float64]) -> float:
        """Estimate local density around a sample."""
        if not self.labeled_samples:
            return 0.5
        
        # Calculate average distance to labeled samples
        distances = []
        for labeled in self.labeled_samples[:20]:  # Sample for efficiency
            dist = np.linalg.norm(sample - labeled.features)
            distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 1.0
        # Convert distance to density (closer = denser)
        density = 1.0 / (1.0 + avg_distance)
        
        return density
    
    def _query_by_committee(self,
                           predictions: List[Tuple[NDArray[np.float64], float]],
                           indices: List[int]) -> NDArray[np.float64]:
        """Query by committee acquisition."""
        scores = np.zeros(len(indices))
        
        # Simulate committee disagreement
        for i, idx in enumerate(indices):
            if i < len(predictions):
                logits, _ = predictions[i]
                
                # Simulate multiple committee member predictions
                committee_predictions = []
                for _ in range(5):
                    noise = np.random.randn(len(logits)) * 0.1
                    committee_pred = np.argmax(logits + noise)
                    committee_predictions.append(committee_pred)
                
                # Disagreement = entropy of committee predictions
                unique, counts = np.unique(committee_predictions, return_counts=True)
                probs = counts / len(committee_predictions)
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                
                scores[i] = entropy
        
        return scores
    
    def _balanced_acquisition(self,
                             predictions: List[Tuple[NDArray[np.float64], float]],
                             indices: List[int]) -> NDArray[np.float64]:
        """Balanced acquisition (exploration + exploitation)."""
        scores = np.zeros(len(indices))
        
        for i, idx in enumerate(indices):
            if i < len(predictions):
                logits, uncertainty = predictions[i]
                
                # Uncertainty component
                uncertainty_score = uncertainty
                
                # Novelty component (how different from labeled data)
                sample = self.unlabeled_pool[idx]
                novelty = self._compute_novelty(sample)
                
                # Balance uncertainty and novelty
                scores[i] = uncertainty_score * 0.5 + novelty * 0.5
        
        return scores
    
    def _compute_novelty(self, sample: NDArray[np.float64]) -> float:
        """Compute novelty of a sample compared to labeled data."""
        if not self.labeled_samples:
            return 1.0
        
        # Calculate minimum distance to labeled samples
        min_distance = float('inf')
        for labeled in self.labeled_samples:
            dist = np.linalg.norm(sample - labeled.features)
            min_distance = min(min_distance, dist)
        
        # Normalize novelty
        novelty = min(min_distance / 2.0, 1.0)
        return novelty
    
    def select_samples(self, 
                      model_predictions: List[Tuple[NDArray[np.float64], float]],
                      batch_size: Optional[int] = None) -> List[int]:
        """Select the most informative samples to label."""
        batch_size = batch_size or self.config.batch_size
        
        # Compute acquisition scores
        scores = self.compute_acquisition_scores(model_predictions)
        
        # Select top-k samples
        top_indices = np.argsort(scores)[-batch_size:][::-1]
        
        selected_indices = [int(i) for i in top_indices]
        
        # Record selection
        self.acquisition_history.append({
            "batch_size": batch_size,
            "selected_indices": selected_indices,
            "scores": scores[selected_indices].tolist(),
            "function": self.config.acquisition_function.name
        })
        
        logger.info(f"Selected {batch_size} samples for labeling")
        return selected_indices
    
    def add_label(self, 
                  sample_index: int,
                  label: int,
                  confidence: float = 1.0,
                  acquisition_score: float = 0.0) -> None:
        """Add a labeled sample."""
        if sample_index >= len(self.unlabeled_pool):
            logger.warning(f"Invalid sample index: {sample_index}")
            return
        
        features = self.unlabeled_pool[sample_index]
        
        labeled_sample = LabeledSample(
            sample_id=f"labeled_{len(self.labeled_samples) + 1}",
            features=features,
            label=label,
            confidence=confidence,
            acquisition_score=acquisition_score,
            metadata={"original_index": sample_index}
        )
        
        self.labeled_samples.append(labeled_sample)
        
        # Remove from unlabeled pool (swap with last and pop)
        self.unlabeled_pool[sample_index] = self.unlabeled_pool[-1]
        self.unlabeled_pool.pop()
        
        logger.info(f"Added label to sample {sample_index}, "
                   f"pool size: {len(self.unlabeled_pool)}")
    
    def get_labeled_data(self) -> Tuple[List[NDArray[np.float64]], List[int]]:
        """Get all labeled data."""
        features = [s.features for s in self.labeled_samples]
        labels = [s.label for s in self.labeled_samples]
        return features, labels
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get statistics about the active learning pool."""
        labeled_features, labeled_labels = self.get_labeled_data()
        
        label_distribution = {}
        for label in labeled_labels:
            label_distribution[label] = label_distribution.get(label, 0) + 1
        
        return {
            "unlabeled_pool_size": len(self.unlabeled_pool),
            "labeled_samples": len(self.labeled_samples),
            "label_distribution": label_distribution,
            "acquisition_history_size": len(self.acquisition_history),
            "average_acquisition_score": np.mean([
                np.mean(h["scores"]) for h in self.acquisition_history
            ]) if self.acquisition_history else 0.0
        }


class ActiveLearningOrchestrator:
    """Orchestrates the active learning process."""

    def __init__(self, learner: Optional[ActiveLearner] = None):
        self.learner = learner or ActiveLearner()
        self.iteration_count = 0
        self.performance_history: List[float] = []
        
    def run_iteration(self,
                     model: Any,
                     unlabeled_pool: List[NDArray[np.float64]]) -> Dict[str, Any]:
        """Run one iteration of active learning."""
        self.iteration_count += 1
        
        # Initialize pool if needed
        if not self.learner.unlabeled_pool:
            self.learner.initialize_pool(unlabeled_pool)
        
        # Get model predictions (simulated)
        predictions = []
        for sample in self.learner.unlabeled_pool[:100]:  # Sample for efficiency
            logits = np.random.randn(4)
            uncertainty = random.uniform(0.1, 0.5)
            predictions.append((logits, uncertainty))
        
        # Select samples to label
        selected_indices = self.learner.select_samples(predictions)
        
        # Simulate labeling (in reality, this would be done by experts)
        for idx in selected_indices:
            # Simulate expert label
            label = random.randint(0, 3)
            confidence = random.uniform(0.7, 1.0)
            score = random.uniform(0.3, 0.8)
            self.learner.add_label(idx, label, confidence, score)
        
        # Simulate model training and evaluation
        model_performance = random.uniform(0.6, 0.9)
        self.performance_history.append(model_performance)
        
        result = {
            "iteration": self.iteration_count,
            "samples_labeled": len(selected_indices),
            "model_performance": model_performance,
            "pool_stats": self.learner.get_pool_statistics()
        }
        
        logger.info(f"Iteration {self.iteration_count}: "
                   f"labeled {len(selected_indices)} samples, "
                   f"performance: {model_performance:.2%}")
        
        return result


__all__ = [
    "ActiveLearner",
    "ActiveLearningOrchestrator",
    "ActiveLearningConfig",
    "AcquisitionFunction",
    "LabeledSample"
]