"""Federated learning across ARGUS instances without sharing raw data.

This module implements a simple Federated Averaging (FedAvg) scheme
with optional differential privacy for sharing gradient updates
across multiple ARGUS instances. Each instance maintains a local
:class:`FederatedClient` with its own copy of the model weights; a
central :class:`FederatedAggregator` combines gradients it receives,
optionally injecting Gaussian noise for differential privacy, and the
top-level :class:`FederatedLearner` coordinates registration,
aggregation, and broadcast.

No networking is performed — the communication layer is abstracted
behind ``receive_gradient`` / ``broadcast_weights`` calls so this
module can be exercised in unit tests and integrated with any
transport (HTTP, Redis, gRPC) later.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Federated client
# ---------------------------------------------------------------------------


@dataclass
class FederatedClient:
    """One ARGUS instance participating in federated learning.

    Attributes:
        client_id: Unique identifier for this instance.
        weights: Local copy of the model weight vector.
        last_update_ts: UNIX timestamp of the last applied update.
        samples_seen: Total number of training samples observed.
        active: Whether the client is currently online.
    """

    client_id: str
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    last_update_ts: float = 0.0
    samples_seen: int = 0
    active: bool = True

    def apply_gradient(self, gradient: np.ndarray, learning_rate: float = 0.01) -> None:
        """Apply a gradient update to local weights."""
        if gradient.shape != self.weights.shape:
            if self.weights.size == 0:
                self.weights = np.zeros_like(gradient, dtype=float)
            else:
                raise ValueError(
                    f"gradient shape {gradient.shape} != weights shape {self.weights.shape}"
                )
        self.weights = self.weights - learning_rate * gradient
        self.last_update_ts = time.time()

    def set_weights(self, weights: np.ndarray) -> None:
        """Overwrite local weights from the server's global model."""
        self.weights = np.asarray(weights, dtype=float).copy()
        self.last_update_ts = time.time()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "weight_shape": list(self.weights.shape),
            "weight_norm": float(np.linalg.norm(self.weights)) if self.weights.size else 0.0,
            "last_update_ts": self.last_update_ts,
            "samples_seen": self.samples_seen,
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# Federated aggregator
# ---------------------------------------------------------------------------


class FederatedAggregator:
    """Combines client gradients into a single global update.

    Supports weighted FedAvg and optional per-client Gaussian noise
    injection for differential privacy. Clients that have dropped out
    (e.g. network partition) are skipped automatically.
    """

    def __init__(
        self,
        dp_noise_sigma: float = 0.0,
        clip_norm: Optional[float] = None,
    ) -> None:
        self.dp_noise_sigma = float(dp_noise_sigma)
        self.clip_norm = float(clip_norm) if clip_norm is not None else None
        self._pending: Dict[str, np.ndarray] = {}
        self._weights_per_client: Dict[str, float] = {}

    def add_gradient(self, client_id: str, gradient: np.ndarray, weight: float = 1.0) -> None:
        grad = np.asarray(gradient, dtype=float)
        if self.clip_norm is not None:
            norm = float(np.linalg.norm(grad))
            if norm > self.clip_norm > 0:
                grad = grad * (self.clip_norm / norm)
        if self.dp_noise_sigma > 0.0:
            grad = grad + np.random.normal(0.0, self.dp_noise_sigma, size=grad.shape)
        self._pending[client_id] = grad
        self._weights_per_client[client_id] = float(max(weight, 0.0))

    def aggregate(self) -> Optional[np.ndarray]:
        """Return the weighted average of pending gradients."""
        if not self._pending:
            return None
        shapes = {g.shape for g in self._pending.values()}
        if len(shapes) > 1:
            logger.warning("federated_aggregator: mismatched gradient shapes %s", shapes)
            # Pad all gradients to the maximum flattened length.
            max_len = max(g.size for g in self._pending.values())
            padded = {}
            for cid, g in self._pending.items():
                flat = g.flatten()
                if flat.size < max_len:
                    pad = np.zeros(max_len - flat.size, dtype=float)
                    flat = np.concatenate([flat, pad])
                padded[cid] = flat
            self._pending = padded
        total_weight = sum(self._weights_per_client.values()) or 1.0
        sample_grad = next(iter(self._pending.values()))
        avg = np.zeros_like(sample_grad, dtype=float)
        for cid, grad in self._pending.items():
            w = self._weights_per_client.get(cid, 1.0) / total_weight
            avg = avg + w * grad
        return avg

    def clear(self) -> None:
        self._pending.clear()
        self._weights_per_client.clear()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "pending_clients": list(self._pending.keys()),
            "dp_noise_sigma": self.dp_noise_sigma,
            "clip_norm": self.clip_norm,
        }


# ---------------------------------------------------------------------------
# Federated learner orchestrator
# ---------------------------------------------------------------------------


class FederatedLearner:
    """Coordinates federated clients, aggregation, and broadcast.

    Typical usage::

        learner = FederatedLearner(model_dim=10)
        learner.register_client('node-a')
        learner.register_client('node-b')
        learner.receive_gradient('node-a', np.random.randn(10))
        learner.receive_gradient('node-b', np.random.randn(10))
        learner.aggregate_and_broadcast()
        weights = learner.get_global_model()
    """

    def __init__(
        self,
        model_dim: int = 16,
        learning_rate: float = 0.05,
        dp_noise_sigma: float = 0.0,
        clip_norm: Optional[float] = 1.0,
        dropout_timeout_s: float = 60.0,
    ) -> None:
        self.model_dim = int(model_dim)
        self.learning_rate = float(learning_rate)
        self.dropout_timeout_s = float(dropout_timeout_s)
        self._clients: Dict[str, FederatedClient] = {}
        self._aggregator = FederatedAggregator(
            dp_noise_sigma=dp_noise_sigma,
            clip_norm=clip_norm,
        )
        self._global_weights = np.zeros(self.model_dim, dtype=float)
        self._round = 0

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def register_client(self, client_id: str) -> FederatedClient:
        """Add a new client and seed it with the current global model."""
        client = FederatedClient(
            client_id=client_id,
            weights=self._global_weights.copy(),
            last_update_ts=time.time(),
        )
        self._clients[client_id] = client
        logger.debug("federated_learner: registered client %s", client_id)
        return client

    def unregister_client(self, client_id: str) -> None:
        self._clients.pop(client_id, None)

    def _mark_dropouts(self) -> None:
        now = time.time()
        for client in self._clients.values():
            if client.last_update_ts == 0.0:
                continue
            if now - client.last_update_ts > self.dropout_timeout_s:
                client.active = False

    # ------------------------------------------------------------------
    # Core federated-learning loop
    # ------------------------------------------------------------------

    def receive_gradient(
        self,
        client_id: str,
        gradient: np.ndarray,
        sample_weight: float = 1.0,
    ) -> None:
        """Ingest a gradient vector from a client."""
        if client_id not in self._clients:
            self.register_client(client_id)
        client = self._clients[client_id]
        client.active = True
        client.last_update_ts = time.time()
        client.samples_seen += int(max(sample_weight, 0.0))
        grad = np.asarray(gradient, dtype=float).flatten()
        if grad.size != self.model_dim:
            if self._global_weights.size == 0:
                self.model_dim = grad.size
                self._global_weights = np.zeros(self.model_dim, dtype=float)
            else:
                logger.warning(
                    "federated_learner: gradient dim %d != model_dim %d; padding/truncating",
                    grad.size,
                    self.model_dim,
                )
                if grad.size < self.model_dim:
                    grad = np.concatenate([grad, np.zeros(self.model_dim - grad.size)])
                else:
                    grad = grad[: self.model_dim]
        self._aggregator.add_gradient(client_id, grad, sample_weight)

    def aggregate_and_broadcast(self) -> np.ndarray:
        """Perform one FedAvg round and distribute the updated weights."""
        self._mark_dropouts()
        avg = self._aggregator.aggregate()
        if avg is None:
            logger.debug("federated_learner: no pending gradients to aggregate")
            return self._global_weights.copy()
        self._global_weights = self._global_weights - self.learning_rate * avg
        self._aggregator.clear()
        self._round += 1
        for client in self._clients.values():
            if client.active:
                client.set_weights(self._global_weights)
        return self._global_weights.copy()

    def get_global_model(self) -> np.ndarray:
        return self._global_weights.copy()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        active_clients = [c.client_id for c in self._clients.values() if c.active]
        return {
            "model_dim": int(self.model_dim),
            "learning_rate": self.learning_rate,
            "round": self._round,
            "n_clients": len(self._clients),
            "active_clients": active_clients,
            "weight_norm": float(np.linalg.norm(self._global_weights)),
            "aggregator": self._aggregator.snapshot(),
            "clients": {cid: c.snapshot() for cid, c in self._clients.items()},
        }


__all__ = ["FederatedClient", "FederatedAggregator", "FederatedLearner"]
