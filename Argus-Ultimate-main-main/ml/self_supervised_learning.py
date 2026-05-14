# pyright: reportMissingImports=false
"""
Self-Supervised Learning System for Argus Trading.

This module implements self-supervised learning to learn representations
from unlabeled market data using pretext tasks.
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


class PretextTask(Enum):
    """Self-supervised pretext tasks."""
    TEMPORAL_PREDICTION = auto()  # Predict next value
    CONTRASTIVE = auto()  # Similar vs dissimilar pairs
    AUTOENCODER = auto()  # Reconstruction
    MASKED_MODELING = auto()  # Predict masked values
    JIGSAW = auto()  # Reorder shuffled sequences


@dataclass
class SSLConfig:
    """Configuration for self-supervised learning."""
    tasks: List[PretextTask] = field(default_factory=lambda: [
        PretextTask.TEMPORAL_PREDICTION,
        PretextTask.CONTRASTIVE,
        PretextTask.AUTOENCODER
    ])
    embedding_dim: int = 64
    hidden_dim: int = 128
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50


class TemporalPredictor:
    """Predicts future values from past observations."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights = np.random.randn(input_dim, hidden_dim) * 0.1
        self.output_weights = np.random.randn(hidden_dim, input_dim) * 0.1

    def encode(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """Encode sequence into representation."""
        return np.tanh(sequence @ self.weights)

    def predict_next(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict next value in sequence."""
        encoded = self.encode(sequence)
        representation = np.mean(encoded, axis=0)
        prediction = representation @ self.output_weights
        return prediction

    def compute_loss(self, 
                    sequence: NDArray[np.float64],
                    actual_next: NDArray[np.float64]) -> float:
        """Compute prediction loss."""
        predicted = self.predict_next(sequence)
        return float(np.mean((predicted - actual_next) ** 2))


class ContrastiveLearner:
    """Learns representations using contrastive objectives."""

    def __init__(self, embedding_dim: int = 64, temperature: float = 0.1):
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.projection = np.random.randn(embedding_dim, embedding_dim) * 0.1

    def encode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Encode input to embedding."""
        embedding = np.tanh(x @ self.projection)
        # Normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def compute_similarity(self, 
                          embedding1: NDArray[np.float64],
                          embedding2: NDArray[np.float64]) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(embedding1, embedding2))

    def contrastive_loss(self,
                        positive_pairs: List[Tuple[NDArray, NDArray]],
                        negative_pairs: List[Tuple[NDArray, NDArray]]) -> float:
        """Compute contrastive loss."""
        loss = 0.0

        # Positive pairs - maximize similarity
        for anchor, positive in positive_pairs:
            anchor_emb = self.encode(anchor)
            positive_emb = self.encode(positive)
            similarity = self.compute_similarity(anchor_emb, positive_emb)
            loss -= similarity / self.temperature

        # Negative pairs - minimize similarity
        for anchor, negative in negative_pairs:
            anchor_emb = self.encode(anchor)
            negative_emb = self.encode(negative)
            similarity = self.compute_similarity(anchor_emb, negative_emb)
            loss += similarity / self.temperature

        return loss / (len(positive_pairs) + len(negative_pairs) + 1e-8)


class Autoencoder:
    """Autoencoder for learning representations."""

    def __init__(self, input_dim: int = 8, latent_dim: int = 16, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.encoder_w2 = np.random.randn(hidden_dim, latent_dim) * 0.1

        # Decoder
        self.decoder_w1 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.decoder_w2 = np.random.randn(hidden_dim, input_dim) * 0.1

    def encode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Encode input to latent representation."""
        hidden = np.tanh(x @ self.encoder_w1)
        latent = np.tanh(hidden @ self.encoder_w2)
        return latent

    def decode(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Decode latent representation to input space."""
        hidden = np.tanh(z @ self.decoder_w1)
        reconstructed = hidden @ self.decoder_w2
        return reconstructed

    def forward(self, x: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forward pass: encode and decode."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def reconstruction_loss(self, x: NDArray[np.float64]) -> float:
        """Compute reconstruction loss."""
        _, reconstructed = self.forward(x)
        return float(np.mean((x - reconstructed) ** 2))


class SelfSupervisedLearner:
    """Self-supervised learning system for market data."""

    def __init__(self, config: Optional[SSLConfig] = None):
        """Initialize the SSL system."""
        self.config = config or SSLConfig()

        # Initialize pretext task models
        self.temporal_predictor = TemporalPredictor(hidden_dim=self.config.hidden_dim)
        self.contrastive_learner = ContrastiveLearner(embedding_dim=self.config.embedding_dim)
        self.autoencoder = Autoencoder(latent_dim=self.config.embedding_dim, hidden_dim=self.config.hidden_dim)

        self.training_history: List[Dict[str, Any]] = []
        self.representation_cache: Dict[str, NDArray[np.float64]] = {}

    def pretrain(self, unlabeled_data: List[NDArray[np.float64]]) -> Dict[str, float]:
        """Pretrain on unlabeled market data using all pretext tasks."""
        logger.info(f"Starting self-supervised pretraining on {len(unlabeled_data)} samples")

        losses = {}

        # Task 1: Temporal Prediction
        if PretextTask.TEMPORAL_PREDICTION in self.config.tasks:
            losses["temporal_prediction"] = self._train_temporal_prediction(unlabeled_data)

        # Task 2: Contrastive Learning
        if PretextTask.CONTRASTIVE in self.config.tasks:
            losses["contrastive"] = self._train_contrastive(unlabeled_data)

        # Task 3: Autoencoder
        if PretextTask.AUTOENCODER in self.config.tasks:
            losses["autoencoder"] = self._train_autoencoder(unlabeled_data)

        # Task 4: Masked Modeling
        if PretextTask.MASKED_MODELING in self.config.tasks:
            losses["masked_modeling"] = self._train_masked_modeling(unlabeled_data)

        # Combined loss
        total_loss = sum(losses.values())
        losses["total"] = total_loss

        self.training_history.append({
            "samples": len(unlabeled_data),
            "losses": losses,
            "tasks": [t.name for t in self.config.tasks]
        })

        logger.info(f"Pretraining complete. Total loss: {total_loss:.4f}")
        return losses

    def _train_temporal_prediction(self, data: List[NDArray[np.float64]]) -> float:
        """Train temporal prediction pretext task."""
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(data) - 1, self.config.batch_size):
            batch = data[i:i + self.config.batch_size]

            for sequence in batch:
                if len(sequence) < 3:
                    continue

                # Split into input and target
                split_idx = len(sequence) // 2
                input_seq = sequence[:split_idx]
                target = sequence[split_idx:split_idx+1].flatten()

                loss = self.temporal_predictor.compute_loss(input_seq, target)
                total_loss += loss
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.debug(f"Temporal prediction loss: {avg_loss:.4f}")
        return avg_loss

    def _train_contrastive(self, data: List[NDArray[np.float64]]) -> float:
        """Train contrastive learning pretext task."""
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(data) - 2, self.config.batch_size):
            # Create positive pairs (different views of same data)
            positive_pairs = []
            negative_pairs = []

            for j in range(min(self.config.batch_size, len(data) - i - 2)):
                # Positive pair: original and augmented version
                original = data[i + j]
                augmented = self._augment_sequence(original)
                positive_pairs.append((original, augmented))

                # Negative pair: different samples
                negative_idx = random.randint(0, len(data) - 1)
                if negative_idx != i + j:
                    negative_pairs.append((original, data[negative_idx]))

            if positive_pairs and negative_pairs:
                loss = self.contrastive_learner.contrastive_loss(positive_pairs, negative_pairs)
                total_loss += loss
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.debug(f"Contrastive learning loss: {avg_loss:.4f}")
        return avg_loss

    def _train_autoencoder(self, data: List[NDArray[np.float64]]) -> float:
        """Train autoencoder pretext task."""
        total_loss = 0.0
        num_samples = 0

        for sample in data:
            loss = self.autoencoder.reconstruction_loss(sample)
            total_loss += loss
            num_samples += 1

        avg_loss = total_loss / max(num_samples, 1)
        logger.debug(f"Autoencoder reconstruction loss: {avg_loss:.4f}")
        return avg_loss

    def _train_masked_modeling(self, data: List[NDArray[np.float64]]) -> float:
        """Train masked modeling pretext task."""
        total_loss = 0.0
        num_samples = 0

        for sample in data:
            if len(sample) < 4:
                continue

            # Mask random positions
            mask_ratio = 0.3
            mask = np.random.rand(len(sample)) < mask_ratio
            masked_sample = sample.copy()
            masked_sample[mask] = 0.0

            # Try to reconstruct masked values
            latent = self.autoencoder.encode(masked_sample)
            reconstructed = self.autoencoder.decode(latent)

            # Loss only on masked positions
            if np.any(mask):
                loss = np.mean((reconstructed[mask] - sample[mask]) ** 2)
                total_loss += loss
                num_samples += 1

        avg_loss = total_loss / max(num_samples, 1)
        logger.debug(f"Masked modeling loss: {avg_loss:.4f}")
        return avg_loss

    def _augment_sequence(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """Create an augmented version of a sequence."""
        augmented = sequence.copy()

        # Random augmentation
        aug_type = random.choice(["noise", "scale", "crop", "shuffle"])

        if aug_type == "noise":
            augmented += np.random.randn(len(augmented)) * 0.1
        elif aug_type == "scale":
            augmented *= random.uniform(0.9, 1.1)
        elif aug_type == "crop" and len(augmented) > 4:
            crop_size = int(len(augmented) * 0.8)
            start = random.randint(0, len(augmented) - crop_size)
            augmented = augmented[start:start + crop_size]
        elif aug_type == "shuffle":
            indices = np.random.permutation(len(augmented))
            augmented = augmented[indices]

        return augmented

    def get_representation(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get learned representation for data."""
        # Use ensemble of all learned representations
        latent = self.autoencoder.encode(data)
        temporal = self.temporal_predictor.encode(data)
        
        # Combine representations
        combined = np.concatenate([latent, np.mean(temporal, axis=0)])
        
        # Project to embedding dimension
        if len(combined) >= self.config.embedding_dim:
            return combined[:self.config.embedding_dim]
        else:
            padded = np.zeros(self.config.embedding_dim)
            padded[:len(combined)] = combined
            return padded

    def get_ssl_summary(self) -> Dict[str, Any]:
        """Get summary of SSL training."""
        if not self.training_history:
            return {"status": "not_trained"}

        latest = self.training_history[-1]
        return {
            "total_samples": sum(h["samples"] for h in self.training_history),
            "training_runs": len(self.training_history),
            "latest_losses": latest["losses"],
            "tasks_used": latest["tasks"],
            "embedding_dim": self.config.embedding_dim
        }


class OnlineSSL(SelfSupervisedLearner):
    """Online self-supervised learning for continuous adaptation."""

    def __init__(self, config: Optional[SSLConfig] = None, update_frequency: int = 100):
        super().__init__(config)
        self.update_frequency = update_frequency
        self.buffer: List[NDArray[np.float64]] = []
        self.samples_since_update = 0

    def add_sample(self, sample: NDArray[np.float64]) -> None:
        """Add a new sample for online learning."""
        self.buffer.append(sample)
        self.samples_since_update += 1

        # Keep buffer manageable
        if len(self.buffer) > 10000:
            self.buffer = self.buffer[-10000:]

        # Update periodically
        if self.samples_since_update >= self.update_frequency:
            self._online_update()
            self.samples_since_update = 0

    def _online_update(self) -> None:
        """Perform online update with recent samples."""
        if len(self.buffer) < 50:
            return

        # Use recent samples for quick update
        recent_samples = self.buffer[-min(500, len(self.buffer)):]

        # Quick pretraining
        quick_config = SSLConfig(
            tasks=self.config.tasks,
            embedding_dim=self.config.embedding_dim,
            epochs=5  # Few epochs for online update
        )
        quick_ssl = SelfSupervisedLearner(quick_config)
        quick_ssl.pretrain(recent_samples)

        # Update weights (simplified)
        logger.debug("Online SSL update completed")


__all__ = [
    "SelfSupervisedLearner",
    "OnlineSSL",
    "SSLConfig",
    "PretextTask",
    "TemporalPredictor",
    "ContrastiveLearner",
    "Autoencoder"
]