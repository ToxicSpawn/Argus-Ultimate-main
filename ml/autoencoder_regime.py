"""
Autoencoder Market Regime Detector — uses a numpy-only autoencoder to learn
a latent representation of market features, then clusters latent codes to
identify market regimes.

Architecture:
  Encoder: input_dim → hidden_dim (ReLU) → latent_dim
  Decoder: latent_dim → hidden_dim (ReLU) → input_dim

High reconstruction error signals a novel/transition regime.

Optional: torch GPU acceleration if available, falls back to numpy.

Usage:
    detector = AutoencoderRegimeDetector(input_dim=10, latent_dim=3)
    detector.fit(historical_features, epochs=100, lr=0.01)
    result = detector.detect_regime(current_features)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Regime labels mapped from k-means cluster assignments
_DEFAULT_REGIME_LABELS = {
    0: "LOW_VOL",
    1: "TRENDING",
    2: "HIGH_VOL",
    3: "CRISIS",
    4: "RECOVERY",
}

# Reconstruction error threshold percentile for detecting transitions
_TRANSITION_PERCENTILE = 90


class AutoencoderRegimeDetector:
    """
    Numpy-based autoencoder for market regime detection.

    Learns a compressed latent representation of market features,
    then uses k-means clustering in latent space to assign regimes.
    """

    def __init__(
        self,
        input_dim: int = 10,
        latent_dim: int = 3,
        hidden_dim: int = 16,
        n_clusters: int = 4,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters

        # Xavier initialization
        self._W_enc1 = self._xavier_init(input_dim, hidden_dim)
        self._b_enc1 = np.zeros(hidden_dim)
        self._W_enc2 = self._xavier_init(hidden_dim, latent_dim)
        self._b_enc2 = np.zeros(latent_dim)

        self._W_dec1 = self._xavier_init(latent_dim, hidden_dim)
        self._b_dec1 = np.zeros(hidden_dim)
        self._W_dec2 = self._xavier_init(hidden_dim, input_dim)
        self._b_dec2 = np.zeros(input_dim)

        # K-means cluster centers (fit during training)
        self._cluster_centers: Optional[np.ndarray] = None  # (n_clusters, latent_dim)

        # Training stats
        self._trained = False
        self._train_errors: List[float] = []
        self._error_threshold = 0.0  # 90th percentile of training errors

        # Feature normalization
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Train autoencoder on historical market features.

        Args:
            features: (N, input_dim) array of market feature vectors.
            epochs: number of training epochs.
            lr: learning rate.
            batch_size: mini-batch size.

        Returns:
            Training summary dict.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        N = features.shape[0]
        if N < 5:
            logger.warning("AutoencoderRegime: too few samples (%d), need >= 5", N)
            return {"status": "insufficient_data", "n_samples": N}

        # Normalize
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std = np.where(self._std < 1e-8, 1.0, self._std)
        X = (features - self._mean) / self._std

        # Training loop
        loss_history: List[float] = []
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                batch_idx = idx[start : start + batch_size]
                batch = X[batch_idx]
                bs = batch.shape[0]

                # Forward
                h1 = self._relu(batch @ self._W_enc1 + self._b_enc1)
                z = h1 @ self._W_enc2 + self._b_enc2
                h2 = self._relu(z @ self._W_dec1 + self._b_dec1)
                x_hat = h2 @ self._W_dec2 + self._b_dec2

                # Loss: MSE
                diff = x_hat - batch
                loss = float(np.mean(diff ** 2))
                epoch_loss += loss

                # Backward (manual gradient computation)
                # dL/dx_hat = 2 * diff / (bs * input_dim)
                d_xhat = 2.0 * diff / (bs * self.input_dim)

                # Decoder layer 2
                d_W_dec2 = h2.T @ d_xhat
                d_b_dec2 = d_xhat.sum(axis=0)
                d_h2 = d_xhat @ self._W_dec2.T

                # ReLU grad for decoder hidden
                d_h2 = d_h2 * (h2 > 0).astype(float)

                # Decoder layer 1
                d_W_dec1 = z.T @ d_h2
                d_b_dec1 = d_h2.sum(axis=0)
                d_z = d_h2 @ self._W_dec1.T

                # Encoder layer 2
                d_W_enc2 = h1.T @ d_z
                d_b_enc2 = d_z.sum(axis=0)
                d_h1 = d_z @ self._W_enc2.T

                # ReLU grad for encoder hidden
                d_h1 = d_h1 * (h1 > 0).astype(float)

                # Encoder layer 1
                d_W_enc1 = batch.T @ d_h1
                d_b_enc1 = d_h1.sum(axis=0)

                # Gradient descent
                self._W_dec2 -= lr * d_W_dec2
                self._b_dec2 -= lr * d_b_dec2
                self._W_dec1 -= lr * d_W_dec1
                self._b_dec1 -= lr * d_b_dec1
                self._W_enc2 -= lr * d_W_enc2
                self._b_enc2 -= lr * d_b_enc2
                self._W_enc1 -= lr * d_W_enc1
                self._b_enc1 -= lr * d_b_enc1

                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

        # Compute latent codes for all training data
        latent_codes = self.encode(features)

        # K-means clustering on latent space
        self._cluster_centers = self._kmeans(latent_codes, self.n_clusters)

        # Compute reconstruction errors for threshold
        _, errors = self.reconstruct(features)
        self._train_errors = errors.tolist() if hasattr(errors, 'tolist') else list(errors)
        self._error_threshold = float(np.percentile(errors, _TRANSITION_PERCENTILE))

        self._trained = True
        logger.info(
            "AutoencoderRegime: trained on %d samples, final loss=%.6f, "
            "error_threshold=%.6f, %d clusters",
            N, loss_history[-1] if loss_history else 0.0,
            self._error_threshold, self.n_clusters,
        )

        return {
            "status": "trained",
            "n_samples": N,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "error_threshold": self._error_threshold,
            "n_clusters": self.n_clusters,
            "loss_history": loss_history,
        }

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Map features to latent space. Returns (N, latent_dim)."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        X = self._normalize(features)
        h1 = self._relu(X @ self._W_enc1 + self._b_enc1)
        z = h1 @ self._W_enc2 + self._b_enc2
        return z

    def reconstruct(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct features through the autoencoder.

        Returns (reconstructed, reconstruction_error) where error is per-sample MSE.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        X = self._normalize(features)
        h1 = self._relu(X @ self._W_enc1 + self._b_enc1)
        z = h1 @ self._W_enc2 + self._b_enc2
        h2 = self._relu(z @ self._W_dec1 + self._b_dec1)
        x_hat = h2 @ self._W_dec2 + self._b_dec2

        # Denormalize
        if self._mean is not None and self._std is not None:
            reconstructed = x_hat * self._std + self._mean
        else:
            reconstructed = x_hat

        # Per-sample MSE
        diff = X - x_hat
        errors = np.mean(diff ** 2, axis=1)

        return reconstructed, errors

    def detect_regime(self, features: np.ndarray) -> dict:
        """
        Detect current market regime.

        Encodes features, finds nearest cluster, checks reconstruction error.

        Returns:
            {regime, confidence, latent, reconstruction_error, is_transition}
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Encode
        z = self.encode(features)  # (1, latent_dim)
        z_vec = z[0]

        # Reconstruct
        _, errors = self.reconstruct(features)
        recon_error = float(errors[0])

        # Determine regime from cluster assignment
        if self._cluster_centers is not None and self._trained:
            dists = np.linalg.norm(self._cluster_centers - z_vec, axis=1)
            cluster_idx = int(np.argmin(dists))
            min_dist = float(dists[cluster_idx])

            # Confidence: inverse distance (closer = more confident)
            max_dist = float(np.max(dists))
            confidence = 1.0 - (min_dist / max(max_dist, 1e-8))
            confidence = max(0.0, min(1.0, confidence))

            regime_label = _DEFAULT_REGIME_LABELS.get(
                cluster_idx, f"REGIME_{cluster_idx}"
            )
        else:
            cluster_idx = 0
            confidence = 0.0
            regime_label = "UNKNOWN"

        # Transition detection: high reconstruction error = novel market state
        is_transition = recon_error > self._error_threshold if self._trained else False

        if is_transition:
            regime_label = "TRANSITION"
            confidence *= 0.5  # reduce confidence during transitions

        return {
            "regime": regime_label,
            "confidence": round(confidence, 4),
            "latent": z_vec.tolist(),
            "reconstruction_error": round(recon_error, 6),
            "is_transition": is_transition,
            "cluster_idx": cluster_idx,
        }

    def get_regime_map(self) -> dict:
        """Return cluster centers and their assigned labels."""
        if self._cluster_centers is None:
            return {"status": "not_trained", "clusters": {}}

        clusters = {}
        for i in range(self.n_clusters):
            label = _DEFAULT_REGIME_LABELS.get(i, f"REGIME_{i}")
            clusters[label] = {
                "center": self._cluster_centers[i].tolist(),
                "cluster_idx": i,
            }

        return {
            "status": "trained",
            "n_clusters": self.n_clusters,
            "latent_dim": self.latent_dim,
            "error_threshold": self._error_threshold,
            "clusters": clusters,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using stored mean/std."""
        if self._mean is not None and self._std is not None:
            return (X - self._mean) / self._std
        return X

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

    @staticmethod
    def _kmeans(
        data: np.ndarray,
        k: int,
        max_iter: int = 50,
    ) -> np.ndarray:
        """Simple k-means clustering. Returns (k, d) cluster centers."""
        N, d = data.shape
        if N < k:
            # Pad with duplicates
            centers = data.copy()
            while centers.shape[0] < k:
                centers = np.vstack([centers, data[0:1]])
            return centers[:k]

        # Initialize with k random samples
        idx = np.random.choice(N, size=k, replace=False)
        centers = data[idx].copy()

        for _ in range(max_iter):
            # Assign
            dists = np.zeros((N, k))
            for c in range(k):
                dists[:, c] = np.linalg.norm(data - centers[c], axis=1)
            assignments = np.argmin(dists, axis=1)

            # Update
            new_centers = np.zeros_like(centers)
            for c in range(k):
                mask = assignments == c
                if mask.any():
                    new_centers[c] = data[mask].mean(axis=0)
                else:
                    new_centers[c] = centers[c]

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        return centers
