"""Spatial graph attention layers for dynamic cross-asset message passing."""

# pyright: reportMissingImports=false, reportConstantRedefinition=false, reportOptionalMemberAccess=false, reportInvalidTypeForm=false

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .dynamic_graph import DynamicGraph

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


@dataclass(slots=True)
class SpatialGNNConfig:
    input_dim: int
    hidden_dim: int = 64
    output_dim: int = 64
    num_heads: int = 4
    num_relations: int = 4
    dropout: float = 0.1
    use_edge_features: bool = True

    def __post_init__(self) -> None:
        self.input_dim = int(self.input_dim)
        self.hidden_dim = int(self.hidden_dim)
        self.output_dim = int(self.output_dim)
        self.num_heads = max(1, int(self.num_heads))
        self.num_relations = max(1, int(self.num_relations))
        self.dropout = float(min(0.9, max(0.0, self.dropout)))


def _relation_adjacency(graph: DynamicGraph) -> Dict[str, np.ndarray]:
    relation_map: Dict[str, np.ndarray] = {}
    size = len(graph.nodes)
    for edge in graph.edges:
        matrix = relation_map.setdefault(edge.relation, np.zeros((size, size), dtype=np.float32))
        matrix[edge.source, edge.target] = float(edge.weight)
    if not relation_map:
        relation_map["correlation"] = np.zeros((size, size), dtype=np.float32)
    return relation_map


if _TORCH_AVAILABLE:

    class SpatialGNN(nn.Module):
        def __init__(self, config: SpatialGNNConfig) -> None:
            super().__init__()
            self.config = config
            self.query_proj = nn.Linear(config.input_dim, config.hidden_dim)
            self.key_proj = nn.Linear(config.input_dim, config.hidden_dim)
            self.value_proj = nn.Linear(config.input_dim, config.hidden_dim)
            self.relation_scale = nn.Parameter(torch.ones(config.num_relations))
            self.edge_gate = nn.Linear(4, 1) if config.use_edge_features else None
            self.out_proj = nn.Linear(config.hidden_dim, config.output_dim)
            self.dropout = nn.Dropout(config.dropout)
            self.last_attention: Dict[str, np.ndarray] = {}

        def forward(
            self,
            node_features: "torch.Tensor",
            graph: DynamicGraph,
            relation_adjacency: Optional[Dict[str, np.ndarray]] = None,
            return_attention: bool = True,
        ) -> Tuple["torch.Tensor", Dict[str, np.ndarray]]:
            if node_features.dim() != 2:
                raise ValueError("node_features must have shape (nodes, features)")
            relation_map = relation_adjacency or _relation_adjacency(graph)
            device = node_features.device
            query = self.query_proj(node_features)
            key = self.key_proj(node_features)
            value = self.value_proj(node_features)
            logits = (query @ key.T) / max(self.config.hidden_dim ** 0.5, 1.0)

            aggregated = torch.zeros_like(value)
            attention_payload: Dict[str, np.ndarray] = {}
            edge_feature_matrix = self._edge_feature_matrix(graph)
            relation_names = list(relation_map.keys())[: self.config.num_relations]
            for idx, relation in enumerate(relation_names):
                adjacency = torch.as_tensor(relation_map[relation], dtype=node_features.dtype, device=device)
                relation_logits = logits + torch.log(torch.clamp(adjacency.abs() + 1e-6, min=1e-6))
                relation_logits = relation_logits.masked_fill(adjacency == 0, -1e9)
                attention = torch.softmax(relation_logits, dim=-1)
                if self.edge_gate is not None and edge_feature_matrix.size:
                    gate = torch.sigmoid(self.edge_gate(torch.as_tensor(edge_feature_matrix, dtype=node_features.dtype, device=device)))
                    gate_matrix = torch.ones_like(adjacency)
                    for edge_idx, edge in enumerate(graph.edges):
                        gate_matrix[edge.source, edge.target] = gate[edge_idx, 0]
                    attention = attention * gate_matrix
                    attention = attention / torch.clamp(attention.sum(dim=-1, keepdim=True), min=1e-6)
                aggregated = aggregated + self.relation_scale[idx] * (attention @ value)
                attention_payload[relation] = attention.detach().cpu().numpy().astype(np.float32)

            output = self.out_proj(self.dropout(F.elu(aggregated + value)))
            if return_attention:
                self.last_attention = attention_payload
            return output, attention_payload

        @staticmethod
        def _edge_feature_matrix(graph: DynamicGraph) -> np.ndarray:
            rows = []
            for edge in graph.edges:
                rows.append(
                    [
                        float(edge.features.get("correlation", 0.0)),
                        float(edge.features.get("spillover", 0.0)),
                        float(edge.features.get("sector_similarity", 0.0)),
                        float(edge.features.get("strength", abs(edge.weight))),
                    ]
                )
            return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)

else:

    class SpatialGNN:  # type: ignore[no-redef]
        def __init__(self, config: SpatialGNNConfig) -> None:
            self.config = config
            rng = np.random.default_rng(7)
            self.w_query = rng.normal(0.0, 0.05, size=(config.input_dim, config.hidden_dim)).astype(np.float32)
            self.w_key = rng.normal(0.0, 0.05, size=(config.input_dim, config.hidden_dim)).astype(np.float32)
            self.w_value = rng.normal(0.0, 0.05, size=(config.input_dim, config.hidden_dim)).astype(np.float32)
            self.w_out = rng.normal(0.0, 0.05, size=(config.hidden_dim, config.output_dim)).astype(np.float32)
            self.last_attention: Dict[str, np.ndarray] = {}

        def forward(
            self,
            node_features: np.ndarray,
            graph: DynamicGraph,
            relation_adjacency: Optional[Dict[str, np.ndarray]] = None,
            return_attention: bool = True,
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            features = np.asarray(node_features, dtype=np.float32)
            relation_map = relation_adjacency or _relation_adjacency(graph)
            query = features @ self.w_query
            key = features @ self.w_key
            value = features @ self.w_value
            logits = (query @ key.T) / max(self.config.hidden_dim ** 0.5, 1.0)
            aggregated = np.zeros_like(value)
            payload: Dict[str, np.ndarray] = {}
            for relation in list(relation_map.keys())[: self.config.num_relations]:
                adjacency = np.asarray(relation_map[relation], dtype=np.float32)
                masked = np.where(adjacency != 0.0, logits + np.log(np.abs(adjacency) + 1e-6), -1e9)
                masked = masked - masked.max(axis=-1, keepdims=True)
                attention = np.exp(masked)
                attention = attention / np.clip(attention.sum(axis=-1, keepdims=True), 1e-6, None)
                aggregated = aggregated + attention @ value
                payload[relation] = attention.astype(np.float32)
            output = np.tanh(aggregated + value) @ self.w_out
            if return_attention:
                self.last_attention = payload
            return output.astype(np.float32), payload
