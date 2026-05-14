"""Hybrid spatial-temporal dynamic GNN model for returns and volatility prediction."""

# pyright: reportMissingImports=false, reportConstantRedefinition=false, reportOptionalMemberAccess=false, reportCallIssue=false

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dynamic_graph import GraphSnapshot
from .spatial_gnn import SpatialGNN, SpatialGNNConfig
from .temporal_attention import TemporalAttentionConfig, TemporalAttentionEncoder

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


@dataclass(slots=True)
class HybridModelConfig:
    node_input_dim: int
    spatial_hidden_dim: int = 64
    temporal_hidden_dim: int = 64
    output_dim: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    regime_dim: int = 3
    use_residual: bool = True


class HybridSpatialTemporalModel:
    """Stack spatial message passing with temporal selective memory."""

    def __init__(self, config: HybridModelConfig) -> None:
        self.config = config
        self.spatial = SpatialGNN(
            SpatialGNNConfig(
                input_dim=config.node_input_dim,
                hidden_dim=config.spatial_hidden_dim,
                output_dim=config.temporal_hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        )
        self.temporal = TemporalAttentionEncoder(
            TemporalAttentionConfig(
                input_dim=config.temporal_hidden_dim,
                hidden_dim=config.temporal_hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        )
        self.last_attention: Dict[str, np.ndarray] = {}

        if _TORCH_AVAILABLE:
            self.regime_layer = nn.Linear(config.regime_dim, config.temporal_hidden_dim)
            self.prediction_head = nn.Sequential(
                nn.Linear(config.temporal_hidden_dim, config.temporal_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.temporal_hidden_dim, config.output_dim),
            )
        else:
            rng = np.random.default_rng(11)
            self.regime_layer = rng.normal(0.0, 0.05, size=(config.regime_dim, config.temporal_hidden_dim)).astype(np.float32)
            self.prediction_head = {
                "w1": rng.normal(0.0, 0.05, size=(config.temporal_hidden_dim, config.temporal_hidden_dim)).astype(np.float32),
                "b1": np.zeros((config.temporal_hidden_dim,), dtype=np.float32),
                "w2": rng.normal(0.0, 0.05, size=(config.temporal_hidden_dim, config.output_dim)).astype(np.float32),
                "b2": np.zeros((config.output_dim,), dtype=np.float32),
            }

    def forward(
        self,
        graph_sequence: Sequence[GraphSnapshot],
        regime_features: Optional[np.ndarray] = None,
        return_attention: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if not graph_sequence:
            raise ValueError("graph_sequence must not be empty")
        spatial_embeddings: List[np.ndarray] = []
        spatial_payload: Dict[str, List[np.ndarray]] = {}
        for snapshot in graph_sequence:
            node_features = snapshot.graph.node_features
            if node_features is None:
                raise ValueError("Each graph snapshot must include node_features")
            embedding, attention = self._spatial_forward(node_features, snapshot)
            spatial_embeddings.append(embedding)
            for relation, weights in attention.items():
                spatial_payload.setdefault(relation, []).append(weights)

        temporal_input = np.stack(spatial_embeddings, axis=1)
        temporal_output, temporal_payload = self._temporal_forward(temporal_input)
        node_state = temporal_output[:, -1, :]

        if regime_features is not None:
            regime_arr = np.asarray(regime_features, dtype=np.float32)
            regime_adjustment = self._regime_adjustment(regime_arr)
            node_state = node_state + regime_adjustment

        if self.config.use_residual:
            node_state = node_state + temporal_input[:, -1, :]

        predictions = self._prediction_forward(node_state)
        attention_payload: Dict[str, np.ndarray] = {
            "temporal_weights": temporal_payload["weights"],
            "temporal_memory_norm": temporal_payload["memory_norm"],
        }
        for relation, weights in spatial_payload.items():
            attention_payload[f"spatial_{relation}"] = np.stack(weights, axis=0)
        if return_attention:
            self.last_attention = attention_payload
        return predictions.astype(np.float32), attention_payload

    def _spatial_forward(self, node_features: np.ndarray, snapshot: GraphSnapshot) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if _TORCH_AVAILABLE and isinstance(self.spatial, nn.Module):
            output, payload = self.spatial.forward(
                torch.as_tensor(node_features, dtype=torch.float32),
                snapshot.graph,
                relation_adjacency=snapshot.relation_adjacency,
                return_attention=True,
            )
            return output.detach().cpu().numpy(), payload
        output, payload = self.spatial.forward(node_features, snapshot.graph, relation_adjacency=snapshot.relation_adjacency, return_attention=True)
        return np.asarray(output, dtype=np.float32), payload

    def _temporal_forward(self, temporal_input: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if _TORCH_AVAILABLE and isinstance(self.temporal, nn.Module):
            output, payload = self.temporal.forward(torch.as_tensor(temporal_input, dtype=torch.float32), return_attention=True)
            return output.detach().cpu().numpy(), payload
        return self.temporal.forward(temporal_input, return_attention=True)

    def _regime_adjustment(self, regime_features: np.ndarray) -> np.ndarray:
        regime_arr = np.asarray(regime_features, dtype=np.float32)
        if regime_arr.ndim == 1:
            regime_arr = np.repeat(regime_arr[None, :], repeats=1, axis=0)
        if _TORCH_AVAILABLE and hasattr(self.regime_layer, "forward"):
            regime_tensor = torch.as_tensor(regime_arr, dtype=torch.float32)
            adjustment = self.regime_layer(regime_tensor).detach().cpu().numpy()
        else:
            adjustment = regime_arr @ np.asarray(self.regime_layer, dtype=np.float32)
        if adjustment.shape[0] == 1:
            return np.repeat(adjustment, repeats=1, axis=0).astype(np.float32)
        return adjustment.astype(np.float32)

    def _prediction_forward(self, node_state: np.ndarray) -> np.ndarray:
        if _TORCH_AVAILABLE and hasattr(self.prediction_head, "forward"):
            return self.prediction_head(torch.as_tensor(node_state, dtype=torch.float32)).detach().cpu().numpy()
        hidden = np.maximum(0.0, node_state @ self.prediction_head["w1"] + self.prediction_head["b1"])
        return hidden @ self.prediction_head["w2"] + self.prediction_head["b2"]

    def predict(
        self,
        graph_sequence: Sequence[GraphSnapshot],
        regime_features: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        predictions, attention = self.forward(graph_sequence=graph_sequence, regime_features=regime_features, return_attention=True)
        return {
            "returns": predictions[:, 0],
            "volatility": predictions[:, 1] if predictions.shape[1] > 1 else np.zeros(predictions.shape[0], dtype=np.float32),
            "attention": attention,
        }
