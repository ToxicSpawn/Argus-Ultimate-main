"""Graph Neural Network for cross-asset analysis.

Implements GNN-based models for modeling relationships between assets
using correlation, causality, and sector-based graph structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketGraph:
    """Represents a market graph with assets as nodes and relationships as edges."""

    nodes: List[str]
    edge_index: np.ndarray  # shape: (2, n_edges)
    edge_weights: np.ndarray  # shape: (n_edges,)
    node_features: np.ndarray  # shape: (n_nodes, n_features)
    timestamps: List[datetime] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.nodes) != self.node_features.shape[0]:
            raise ValueError(
                f"Number of nodes ({len(self.nodes)}) must match "
                f"node_features rows ({self.node_features.shape[0]})"
            )
        if self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, n_edges), got {self.edge_index.shape}")
        if self.edge_index.shape[1] != self.edge_weights.shape[0]:
            raise ValueError(
                f"edge_index columns ({self.edge_index.shape[1]}) must match "
                f"edge_weights length ({self.edge_weights.shape[0]})"
            )

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def node_to_idx(self) -> Dict[str, int]:
        return {node: idx for idx, node in enumerate(self.nodes)}


@dataclass
class TrainingResult:
    """Stores training metrics and history."""

    loss_history: List[float]
    final_loss: float
    epochs_completed: int
    model_params: Dict[str, Any] = field(default_factory=dict)


class GraphBuilder:
    """Builds MarketGraph instances from various data sources."""

    def __init__(self) -> None:
        logger.info("GraphBuilder initialized")

    def build_correlation_graph(
        self,
        returns: np.ndarray,
        assets: List[str],
        threshold: float = 0.3,
    ) -> MarketGraph:
        """Build a graph based on return correlations.

        Args:
            returns: Array of shape (n_assets, n_observations)
            assets: List of asset symbols
            threshold: Minimum absolute correlation to create an edge

        Returns:
            MarketGraph with correlation-based edges
        """
        if returns.shape[0] != len(assets):
            raise ValueError(
                f"returns rows ({returns.shape[0]}) must match assets count ({len(assets)})"
            )

        corr_matrix = np.corrcoef(returns)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        n_assets = len(assets)
        edge_sources = []
        edge_targets = []
        edge_weights = []

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                abs_corr = abs(corr_matrix[i, j])
                if abs_corr >= threshold:
                    edge_sources.append(i)
                    edge_targets.append(j)
                    edge_weights.append(corr_matrix[i, j])
                    edge_sources.append(j)
                    edge_targets.append(i)
                    edge_weights.append(corr_matrix[j, i])

        edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_weights_arr = np.array(edge_weights, dtype=np.float64)

        node_features = self._compute_node_features_from_returns(returns)

        graph = MarketGraph(
            nodes=assets,
            edge_index=edge_index,
            edge_weights=edge_weights_arr,
            node_features=node_features,
            timestamps=[datetime.now()],
        )

        logger.info(
            "Built correlation graph: %d nodes, %d edges (threshold=%.2f)",
            graph.n_nodes,
            graph.n_edges // 2,
            threshold,
        )
        return graph

    def build_causality_graph(
        self,
        causal_matrix: np.ndarray,
        assets: List[str],
        threshold: float = 0.1,
    ) -> MarketGraph:
        """Build a graph based on causal relationships.

        Args:
            causal_matrix: Matrix where causal_matrix[i, j] represents
                          causal strength from asset i to asset j
            assets: List of asset symbols
            threshold: Minimum causal strength to create an edge

        Returns:
            MarketGraph with causality-based edges
        """
        if causal_matrix.shape[0] != causal_matrix.shape[1]:
            raise ValueError("causal_matrix must be square")
        if causal_matrix.shape[0] != len(assets):
            raise ValueError(
                f"causal_matrix size ({causal_matrix.shape[0]}) must match assets count ({len(assets)})"
            )

        n_assets = len(assets)
        edge_sources = []
        edge_targets = []
        edge_weights = []

        for i in range(n_assets):
            for j in range(n_assets):
                if i != j and abs(causal_matrix[i, j]) >= threshold:
                    edge_sources.append(i)
                    edge_targets.append(j)
                    edge_weights.append(causal_matrix[i, j])

        edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_weights_arr = np.array(edge_weights, dtype=np.float64)

        node_features = np.random.randn(n_assets, 16) * 0.01

        graph = MarketGraph(
            nodes=assets,
            edge_index=edge_index,
            edge_weights=edge_weights_arr,
            node_features=node_features,
            timestamps=[datetime.now()],
        )

        logger.info(
            "Built causality graph: %d nodes, %d edges (threshold=%.2f)",
            graph.n_nodes,
            graph.n_edges,
            threshold,
        )
        return graph

    def build_sector_graph(
        self,
        assets: List[str],
        sectors: Dict[str, str],
        intra_sector_weight: float = 0.8,
        inter_sector_weight: float = 0.2,
    ) -> MarketGraph:
        """Build a graph based on sector classifications.

        Args:
            assets: List of asset symbols
            sectors: Dict mapping asset symbol to sector name
            intra_sector_weight: Edge weight for assets in same sector
            inter_sector_weight: Edge weight for assets in different sectors

        Returns:
            MarketGraph with sector-based edges
        """
        n_assets = len(assets)
        edge_sources = []
        edge_targets = []
        edge_weights = []

        sector_map = {asset: sectors.get(asset, "unknown") for asset in assets}

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if sector_map[assets[i]] == sector_map[assets[j]]:
                    weight = intra_sector_weight
                else:
                    weight = inter_sector_weight

                edge_sources.append(i)
                edge_targets.append(j)
                edge_weights.append(weight)
                edge_sources.append(j)
                edge_targets.append(i)
                edge_weights.append(weight)

        edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_weights_arr = np.array(edge_weights, dtype=np.float64)

        node_features = np.random.randn(n_assets, 16) * 0.01

        graph = MarketGraph(
            nodes=assets,
            edge_index=edge_index,
            edge_weights=edge_weights_arr,
            node_features=node_features,
            timestamps=[datetime.now()],
        )

        logger.info(
            "Built sector graph: %d nodes, %d edges",
            graph.n_nodes,
            graph.n_edges // 2,
        )
        return graph

    def update_edges(
        self,
        graph: MarketGraph,
        new_returns: np.ndarray,
        threshold: float = 0.3,
    ) -> MarketGraph:
        """Update graph edges based on new return data.

        Args:
            graph: Existing MarketGraph
            new_returns: New return data of shape (n_assets, n_observations)
            threshold: Correlation threshold for edge creation

        Returns:
            Updated MarketGraph with refreshed edges
        """
        if new_returns.shape[0] != graph.n_nodes:
            raise ValueError(
                f"new_returns rows ({new_returns.shape[0]}) must match graph nodes ({graph.n_nodes})"
            )

        corr_matrix = np.corrcoef(new_returns)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        edge_sources = []
        edge_targets = []
        edge_weights = []

        for i in range(graph.n_nodes):
            for j in range(i + 1, graph.n_nodes):
                abs_corr = abs(corr_matrix[i, j])
                if abs_corr >= threshold:
                    edge_sources.append(i)
                    edge_targets.append(j)
                    edge_weights.append(corr_matrix[i, j])
                    edge_sources.append(j)
                    edge_targets.append(i)
                    edge_weights.append(corr_matrix[j, i])

        edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_weights_arr = np.array(edge_weights, dtype=np.float64)

        updated_features = self._compute_node_features_from_returns(new_returns)

        updated_graph = MarketGraph(
            nodes=graph.nodes,
            edge_index=edge_index,
            edge_weights=edge_weights_arr,
            node_features=updated_features,
            timestamps=graph.timestamps + [datetime.now()],
        )

        logger.info(
            "Updated graph edges: %d nodes, %d edges (was %d)",
            updated_graph.n_nodes,
            updated_graph.n_edges // 2,
            graph.n_edges // 2,
        )
        return updated_graph

    @staticmethod
    def _compute_node_features_from_returns(returns: np.ndarray) -> np.ndarray:
        """Compute node features from return data.

        Args:
            returns: Array of shape (n_assets, n_observations)

        Returns:
            Node features of shape (n_assets, n_features)
        """
        n_assets = returns.shape[0]
        features_list = []

        mean_returns = np.mean(returns, axis=1, keepdims=True)
        std_returns = np.std(returns, axis=1, keepdims=True)
        skewness = np.mean(((returns - mean_returns) / (std_returns + 1e-8)) ** 3, axis=1, keepdims=True)
        kurtosis = np.mean(((returns - mean_returns) / (std_returns + 1e-8)) ** 4, axis=1, keepdims=True)
        max_returns = np.max(returns, axis=1, keepdims=True)
        min_returns = np.min(returns, axis=1, keepdims=True)
        var_95 = np.percentile(returns, 5, axis=1, keepdims=True)
        sharpe = mean_returns / (std_returns + 1e-8)

        features_list.extend([
            mean_returns, std_returns, skewness, kurtosis,
            max_returns, min_returns, var_95, sharpe,
        ])

        for window in [5, 10, 20]:
            if returns.shape[1] >= window:
                rolling_vol = np.array([
                    np.std(returns[i, -window:], ddof=1) if window > 1 else 0.0
                    for i in range(n_assets)
                ]).reshape(-1, 1)
                features_list.append(rolling_vol)

        n_features = sum(f.shape[1] for f in features_list)
        if n_features < 16:
            padding = np.zeros((n_assets, 16 - n_features))
            features_list.append(padding)

        node_features = np.hstack(features_list)[:, :64]

        node_features = (node_features - np.mean(node_features, axis=0, keepdims=True)) / (
            np.std(node_features, axis=0, keepdims=True) + 1e-8
        )

        return node_features


class GraphConvLayer:
    """Graph convolutional layer implementing message passing."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)

        self._weight_grad: Optional[np.ndarray] = None
        self._bias_grad: Optional[np.ndarray] = None

    def forward(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass through the graph convolution layer.

        Args:
            node_features: Node feature matrix of shape (n_nodes, in_features)
            edge_index: Edge indices of shape (2, n_edges)
            edge_weights: Edge weights of shape (n_edges,)

        Returns:
            Updated node features of shape (n_nodes, out_features)
        """
        messages = self.message_passing(node_features, edge_index)

        if edge_weights is not None:
            aggregated = self.aggregate_weighted(messages, edge_index, edge_weights)
        else:
            aggregated = self.aggregate(messages, edge_index)

        output = self.update(node_features, aggregated)

        return self._apply_activation(output)

    def message_passing(
        self,
        features: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """Compute messages for each edge.

        Args:
            features: Node features of shape (n_nodes, in_features)
            edge_index: Edge indices of shape (2, n_edges)

        Returns:
            Messages of shape (n_edges, out_features)
        """
        source_nodes = edge_index[0]
        transformed = features @ self.weight
        messages = transformed[source_nodes]
        return messages

    def aggregate(
        self,
        messages: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """Aggregate messages to nodes using mean aggregation.

        Args:
            messages: Messages of shape (n_edges, out_features)
            edge_index: Edge indices of shape (2, n_edges)

        Returns:
            Aggregated features of shape (n_nodes, out_features)
        """
        n_nodes = edge_index.max() + 1 if edge_index.size > 0 else 0
        aggregated = np.zeros((n_nodes, self.out_features))
        counts = np.zeros(n_nodes)

        target_nodes = edge_index[1]
        for idx in range(len(target_nodes)):
            aggregated[target_nodes[idx]] += messages[idx]
            counts[target_nodes[idx]] += 1.0

        counts = np.maximum(counts, 1.0)
        aggregated /= counts[:, np.newaxis]

        return aggregated

    def aggregate_weighted(
        self,
        messages: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: np.ndarray,
    ) -> np.ndarray:
        """Aggregate messages with edge weight weighting.

        Args:
            messages: Messages of shape (n_edges, out_features)
            edge_index: Edge indices of shape (2, n_edges)
            edge_weights: Edge weights of shape (n_edges,)

        Returns:
            Aggregated features of shape (n_nodes, out_features)
        """
        n_nodes = edge_index.max() + 1 if edge_index.size > 0 else 0
        aggregated = np.zeros((n_nodes, self.out_features))
        weight_sums = np.zeros(n_nodes)

        target_nodes = edge_index[1]
        abs_weights = np.abs(edge_weights)

        for idx in range(len(target_nodes)):
            w = abs_weights[idx]
            aggregated[target_nodes[idx]] += messages[idx] * w
            weight_sums[target_nodes[idx]] += w

        weight_sums = np.maximum(weight_sums, 1e-8)
        aggregated /= weight_sums[:, np.newaxis]

        return aggregated

    def update(
        self,
        node_features: np.ndarray,
        aggregated: np.ndarray,
    ) -> np.ndarray:
        """Combine node features with aggregated messages.

        Args:
            node_features: Original node features
            aggregated: Aggregated neighbor messages

        Returns:
            Updated node features
        """
        if node_features.shape[1] != self.in_features:
            node_features = node_features @ np.eye(node_features.shape[1], self.in_features)

        return aggregated + self.bias

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == "elu":
            return np.where(x > 0, x, np.exp(x) - 1.0)
        return x

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"weight": self.weight.copy(), "bias": self.bias.copy()}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self.weight = params["weight"].copy()
        self.bias = params["bias"].copy()


class GraphAttentionLayer:
    """Graph attention layer with multi-head attention."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout

        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"
        self.head_dim = out_features // n_heads

        scale = np.sqrt(2.0 / in_features)
        self.query_weights = np.random.randn(in_features, out_features) * scale
        self.key_weights = np.random.randn(in_features, out_features) * scale
        self.value_weights = np.random.randn(in_features, out_features) * scale

        self.output_weight = np.eye(out_features)

    def forward(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass through the attention layer.

        Args:
            node_features: Node features of shape (n_nodes, in_features)
            edge_index: Edge indices of shape (2, n_edges)
            edge_weights: Optional edge weights

        Returns:
            Attended node features of shape (n_nodes, out_features)
        """
        return self.multi_head_attention(node_features, edge_index)

    def compute_attention(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """Compute attention weights for edges.

        Args:
            node_features: Node features of shape (n_nodes, in_features)
            edge_index: Edge indices of shape (2, n_edges)

        Returns:
            Attention weights of shape (n_edges,)
        """
        queries = node_features @ self.query_weights
        keys = node_features @ self.key_weights

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        q_selected = queries[source_nodes]
        k_selected = keys[target_nodes]

        attention_logits = np.sum(q_selected * k_selected, axis=1) / np.sqrt(self.in_features)

        attention_weights = np.exp(attention_logits - np.max(attention_logits))
        attention_weights = attention_weights / (np.sum(attention_weights) + 1e-8)

        return attention_weights

    def multi_head_attention(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        n_heads: Optional[int] = None,
    ) -> np.ndarray:
        """Compute multi-head attention.

        Args:
            node_features: Node features of shape (n_nodes, in_features)
            edge_index: Edge indices of shape (2, n_edges)
            n_heads: Number of attention heads (uses self.n_heads if None)

        Returns:
            Attended node features of shape (n_nodes, out_features)
        """
        n_heads = n_heads or self.n_heads
        head_dim = self.out_features // n_heads
        n_nodes = node_features.shape[0]

        all_head_outputs = []

        for head_idx in range(n_heads):
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim

            q_head = node_features @ self.query_weights[:, start_idx:end_idx]
            k_head = node_features @ self.key_weights[:, start_idx:end_idx]
            v_head = node_features @ self.value_weights[:, start_idx:end_idx]

            source_nodes = edge_index[0]
            target_nodes = edge_index[1]

            q_selected = q_head[source_nodes]
            k_selected = k_head[target_nodes]

            attention_logits = np.sum(q_selected * k_selected, axis=1) / np.sqrt(head_dim)

            attention_weights = self._softmax_by_target(
                attention_logits, target_nodes, n_nodes
            )

            v_selected = v_head[source_nodes]
            head_output = np.zeros((n_nodes, head_dim))

            for idx in range(len(target_nodes)):
                head_output[target_nodes[idx]] += attention_weights[idx] * v_selected[idx]

            all_head_outputs.append(head_output)

        output = np.concatenate(all_head_outputs, axis=1)

        if self.dropout > 0:
            mask = (np.random.rand(*output.shape) > self.dropout).astype(np.float64)
            output = output * mask / (1.0 - self.dropout + 1e-8)

        return output

    @staticmethod
    def _softmax_by_target(
        logits: np.ndarray,
        targets: np.ndarray,
        n_nodes: int,
    ) -> np.ndarray:
        """Compute softmax grouped by target node."""
        result = np.zeros_like(logits)
        for node_idx in range(n_nodes):
            mask = targets == node_idx
            if np.any(mask):
                node_logits = logits[mask]
                node_logits = node_logits - np.max(node_logits)
                exp_logits = np.exp(node_logits)
                result[mask] = exp_logits / (np.sum(exp_logits) + 1e-8)
        return result

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "query_weights": self.query_weights.copy(),
            "key_weights": self.key_weights.copy(),
            "value_weights": self.value_weights.copy(),
            "output_weight": self.output_weight.copy(),
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self.query_weights = params["query_weights"].copy()
        self.key_weights = params["key_weights"].copy()
        self.value_weights = params["value_weights"].copy()
        self.output_weight = params["output_weight"].copy()


class GraphNeuralNetwork:
    """Multi-layer Graph Neural Network for cross-asset analysis."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        self.layers: List[GraphConvLayer] = []
        self.attention_layers: List[GraphAttentionLayer] = []

        self.layers.append(GraphConvLayer(input_dim, hidden_dim, activation=activation))
        for _ in range(n_layers - 2):
            self.layers.append(GraphConvLayer(hidden_dim, hidden_dim, activation=activation))
            self.attention_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout))
        self.layers.append(GraphConvLayer(hidden_dim, output_dim, activation="linear"))

        self._loss_history: List[float] = []

        logger.info(
            "GraphNeuralNetwork initialized: input_dim=%d, hidden_dim=%d, "
            "output_dim=%d, n_layers=%d, n_heads=%d",
            input_dim,
            hidden_dim,
            output_dim,
            n_layers,
            n_heads,
        )

    def forward(self, graph: MarketGraph) -> np.ndarray:
        """Forward pass through the GNN.

        Args:
            graph: MarketGraph with node features and edges

        Returns:
            Node embeddings of shape (n_nodes, output_dim)
        """
        features = graph.node_features.copy()
        edge_index = graph.edge_index
        edge_weights = graph.edge_weights

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                features = layer.forward(features, edge_index, edge_weights)

                if i < len(self.attention_layers):
                    attn_features = self.attention_layers[i].forward(
                        features, edge_index, edge_weights
                    )
                    features = 0.5 * features + 0.5 * attn_features
            else:
                features = layer.forward(features, edge_index, edge_weights)

        return features

    def predict(
        self,
        graph: MarketGraph,
        target_nodes: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Generate predictions for specified nodes.

        Args:
            graph: MarketGraph input
            target_nodes: List of node indices to predict (all if None)

        Returns:
            Predictions of shape (n_target_nodes, output_dim)
        """
        embeddings = self.forward(graph)

        if target_nodes is not None:
            return embeddings[target_nodes]
        return embeddings

    def train(
        self,
        graphs: List[MarketGraph],
        targets: List[np.ndarray],
        epochs: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = True,
    ) -> List[float]:
        """Train the GNN using gradient-free optimization.

        Args:
            graphs: List of MarketGraph instances for training
            targets: List of target arrays corresponding to each graph
            epochs: Number of training epochs
            learning_rate: Learning rate for parameter updates
            verbose: Whether to log training progress

        Returns:
            List of loss values per epoch
        """
        self._loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_samples = len(graphs)

            for graph, target in zip(graphs, targets):
                predictions = self.forward(graph)
                loss = self._compute_loss(predictions, target)
                epoch_loss += loss

                self._update_parameters(graph, target, learning_rate)

            avg_loss = epoch_loss / max(n_samples, 1)
            self._loss_history.append(float(avg_loss))

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                logger.info(
                    "GNN Training epoch %d/%d: loss=%.6f",
                    epoch + 1,
                    epochs,
                    avg_loss,
                )

        return self._loss_history

    def _compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Compute MSE loss between predictions and targets."""
        if predictions.shape != targets.shape:
            min_rows = min(predictions.shape[0], targets.shape[0])
            predictions = predictions[:min_rows]
            targets = targets[:min_rows]

        return float(np.mean((predictions - targets) ** 2))

    def _update_parameters(
        self,
        graph: MarketGraph,
        targets: np.ndarray,
        learning_rate: float,
    ) -> None:
        """Update parameters using numerical gradient approximation."""
        epsilon = 1e-4
        perturbation = np.random.randn(*graph.node_features.shape) * epsilon

        predictions = self.forward(graph)
        base_loss = self._compute_loss(predictions, targets)

        perturbed_features = graph.node_features + perturbation
        perturbed_graph = MarketGraph(
            nodes=graph.nodes,
            edge_index=graph.edge_index,
            edge_weights=graph.edge_weights,
            node_features=perturbed_features,
            timestamps=graph.timestamps,
        )
        perturbed_predictions = self.forward(perturbed_graph)
        perturbed_loss = self._compute_loss(perturbed_predictions, targets)

        gradient_direction = (perturbed_loss - base_loss) / (epsilon + 1e-8)

        for layer in self.layers:
            noise = np.random.randn(*layer.weight.shape) * learning_rate * 0.01
            if gradient_direction > 0:
                layer.weight -= noise
            else:
                layer.weight += noise

    def get_params(self) -> Dict[str, Any]:
        params = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "layers": [layer.get_params() for layer in self.layers],
            "attention_layers": [layer.get_params() for layer in self.attention_layers],
        }
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        for i, layer_params in enumerate(params.get("layers", [])):
            if i < len(self.layers):
                self.layers[i].set_params(layer_params)
        for i, attn_params in enumerate(params.get("attention_layers", [])):
            if i < len(self.attention_layers):
                self.attention_layers[i].set_params(attn_params)


class CrossAssetPredictor:
    """High-level predictor using GNN for cross-asset return prediction."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        self.model: Optional[GraphNeuralNetwork] = None
        self.graph_builder = GraphBuilder()
        self._is_trained = False

        logger.info("CrossAssetPredictor initialized")

    def build_model(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> GraphNeuralNetwork:
        """Build the GNN model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension

        Returns:
            Initialized GraphNeuralNetwork
        """
        self.model = GraphNeuralNetwork(
            input_dim=input_dim or self.input_dim,
            hidden_dim=hidden_dim or self.hidden_dim,
            output_dim=output_dim or self.output_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )
        return self.model

    def train(
        self,
        returns_data: Dict[str, np.ndarray],
        lookback: int = 60,
        epochs: int = 100,
        learning_rate: float = 0.01,
        threshold: float = 0.3,
    ) -> TrainingResult:
        """Train the cross-asset predictor.

        Args:
            returns_data: Dict mapping asset symbol to returns array
            lookback: Lookback window for building graphs
            epochs: Number of training epochs
            learning_rate: Learning rate
            threshold: Correlation threshold for graph building

        Returns:
            TrainingResult with loss history and metrics
        """
        if self.model is None:
            self.build_model()

        assets = list(returns_data.keys())
        n_assets = len(assets)

        min_length = min(len(r) for r in returns_data.values())
        if min_length < lookback:
            raise ValueError(
                f"Minimum return length ({min_length}) is less than lookback ({lookback})"
            )

        graphs = []
        targets = []

        n_windows = min_length - lookback
        for t in range(0, n_windows, max(lookback // 2, 1)):
            window_returns = np.array([
                returns_data[asset][t : t + lookback] for asset in assets
            ])

            graph = self.graph_builder.build_correlation_graph(
                returns=window_returns,
                assets=assets,
                threshold=threshold,
            )
            graphs.append(graph)

            future_returns = np.array([
                returns_data[asset][t + lookback] if t + lookback < len(returns_data[asset]) else 0.0
                for asset in assets
            ])
            target = np.tile(future_returns.reshape(-1, 1), (1, self.output_dim))
            targets.append(target)

        if not graphs:
            raise ValueError("No training windows generated from returns data")

        loss_history = self.model.train(
            graphs=graphs,
            targets=targets,
            epochs=epochs,
            learning_rate=learning_rate,
        )

        self._is_trained = True

        result = TrainingResult(
            loss_history=loss_history,
            final_loss=loss_history[-1] if loss_history else float("inf"),
            epochs_completed=epochs,
            model_params=self.model.get_params(),
        )

        logger.info(
            "CrossAssetPredictor training complete: %d epochs, final_loss=%.6f",
            epochs,
            result.final_loss,
        )
        return result

    def predict_next_returns(
        self,
        graph: MarketGraph,
    ) -> Dict[str, float]:
        """Predict next period returns for all assets in the graph.

        Args:
            graph: MarketGraph with current asset data

        Returns:
            Dict mapping asset symbol to predicted return
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() or train() first.")

        embeddings = self.model.forward(graph)

        predictions = {}
        for i, asset in enumerate(graph.nodes):
            pred = float(np.mean(embeddings[i]))
            predictions[asset] = pred

        logger.info("Predicted next returns for %d assets", len(predictions))
        return predictions

    def get_asset_embeddings(
        self,
        graph: MarketGraph,
    ) -> np.ndarray:
        """Get learned embeddings for all assets.

        Args:
            graph: MarketGraph with current asset data

        Returns:
            Node embeddings of shape (n_nodes, output_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() or train() first.")

        embeddings = self.model.forward(graph)

        logger.info(
            "Generated embeddings for %d assets, dim=%d",
            graph.n_nodes,
            embeddings.shape[1],
        )
        return embeddings

    def save_model(self, path: str) -> None:
        """Save model parameters to file."""
        if self.model is None:
            raise RuntimeError("No model to save")

        import json

        params = self.model.get_params()
        serializable_params = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                serializable_params[key] = value.tolist()
            elif isinstance(value, list):
                serializable_params[key] = [
                    {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in item.items()}
                    for item in value
                ]
            else:
                serializable_params[key] = value

        with open(path, "w") as f:
            json.dump(serializable_params, f)

        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load model parameters from file."""
        import json

        with open(path, "r") as f:
            params = json.load(f)

        if self.model is None:
            self.build_model(
                input_dim=params.get("input_dim", self.input_dim),
                hidden_dim=params.get("hidden_dim", self.hidden_dim),
                output_dim=params.get("output_dim", self.output_dim),
            )

        self.model.set_params(params)
        self._is_trained = True

        logger.info("Model loaded from %s", path)


class GraphAnalyzer:
    """Analyzes graph structure for insights about asset relationships."""

    def __init__(self) -> None:
        logger.info("GraphAnalyzer initialized")

    def compute_centrality(
        self,
        graph: MarketGraph,
    ) -> Dict[str, float]:
        """Compute degree centrality for each node.

        Args:
            graph: MarketGraph to analyze

        Returns:
            Dict mapping asset symbol to centrality score
        """
        centrality = {node: 0.0 for node in graph.nodes}

        if graph.n_edges == 0:
            return centrality

        edge_index = graph.edge_index
        edge_weights = graph.edge_weights

        for idx in range(graph.n_edges):
            source = edge_index[0, idx]
            target = edge_index[1, idx]
            weight = abs(edge_weights[idx])

            centrality[graph.nodes[source]] += weight
            centrality[graph.nodes[target]] += weight

        max_centrality = max(centrality.values()) if centrality else 1.0
        if max_centrality > 0:
            centrality = {k: v / max_centrality for k, v in centrality.items()}

        logger.info("Computed centrality for %d nodes", len(centrality))
        return centrality

    def detect_clusters(
        self,
        graph: MarketGraph,
        n_clusters: Optional[int] = None,
    ) -> List[List[str]]:
        """Detect clusters of assets using spectral clustering.

        Args:
            graph: MarketGraph to analyze
            n_clusters: Number of clusters (auto-determined if None)

        Returns:
            List of clusters, each cluster is a list of asset symbols
        """
        n_nodes = graph.n_nodes

        if n_nodes == 0:
            return []

        adjacency = np.zeros((n_nodes, n_nodes))
        for idx in range(graph.n_edges):
            source = graph.edge_index[0, idx]
            target = graph.edge_index[1, idx]
            weight = abs(graph.edge_weights[idx])
            adjacency[source, target] = weight

        degree = np.sum(adjacency, axis=1)
        degree_matrix = np.diag(degree)
        laplacian = degree_matrix - adjacency

        laplacian = laplacian + np.eye(n_nodes) * 1e-8

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        except np.linalg.LinAlgError:
            logger.warning("Eigenvalue decomposition failed, using fallback clustering")
            return self._fallback_clustering(graph)

        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        if n_clusters is None:
            n_clusters = self._estimate_n_clusters(eigenvalues, n_nodes)

        embedding = eigenvectors[:, :n_clusters]

        embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding_norm = np.maximum(embedding_norm, 1e-8)
        embedding = embedding / embedding_norm

        clusters = self._kmeans_clustering(embedding, n_clusters, max_iter=100)

        cluster_assets = []
        for cluster_idx in range(n_clusters):
            cluster_nodes = [graph.nodes[i] for i in range(n_nodes) if clusters[i] == cluster_idx]
            if cluster_nodes:
                cluster_assets.append(cluster_nodes)

        logger.info("Detected %d clusters", len(cluster_assets))
        return cluster_assets

    def identify_systemic_nodes(
        self,
        graph: MarketGraph,
        threshold: float = 0.7,
    ) -> List[str]:
        """Identify systemic/risk-critical nodes in the graph.

        Args:
            graph: MarketGraph to analyze
            threshold: Centrality threshold for systemic classification

        Returns:
            List of asset symbols identified as systemic
        """
        centrality = self.compute_centrality(graph)

        systemic_nodes = [
            node for node, score in centrality.items() if score >= threshold
        ]

        if not systemic_nodes:
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            n_top = max(1, len(sorted_nodes) // 10)
            systemic_nodes = [node for node, _ in sorted_nodes[:n_top]]

        logger.info(
            "Identified %d systemic nodes (threshold=%.2f)",
            len(systemic_nodes),
            threshold,
        )
        return systemic_nodes

    @staticmethod
    def _estimate_n_clusters(eigenvalues: np.ndarray, n_nodes: int) -> int:
        """Estimate optimal number of clusters using eigengap heuristic."""
        if len(eigenvalues) < 2:
            return 1

        gaps = np.diff(eigenvalues[:min(10, len(eigenvalues))])
        if len(gaps) == 0:
            return 2

        optimal_k = np.argmax(gaps) + 2
        return max(2, min(optimal_k, n_nodes // 2))

    @staticmethod
    def _kmeans_clustering(
        data: np.ndarray,
        n_clusters: int,
        max_iter: int = 100,
    ) -> np.ndarray:
        """Simple k-means clustering implementation."""
        n_samples = data.shape[0]

        if n_samples <= n_clusters:
            return np.arange(n_samples)

        np.random.seed(42)
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = data[indices].copy()

        labels = np.zeros(n_samples, dtype=np.int64)

        for _ in range(max_iter):
            distances = np.zeros((n_samples, n_clusters))
            for k in range(n_clusters):
                distances[:, k] = np.sum((data - centroids[k]) ** 2, axis=1)

            new_labels = np.argmin(distances, axis=1)

            if np.array_equal(labels, new_labels):
                break

            labels = new_labels

            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    centroids[k] = np.mean(data[mask], axis=0)

        return labels

    def _fallback_clustering(self, graph: MarketGraph) -> List[List[str]]:
        """Fallback clustering based on node degree."""
        centrality = self.compute_centrality(graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        n_clusters = max(2, len(sorted_nodes) // 3)
        clusters = [[] for _ in range(n_clusters)]

        for i, (node, _) in enumerate(sorted_nodes):
            clusters[i % n_clusters].append(node)

        return [c for c in clusters if c]
