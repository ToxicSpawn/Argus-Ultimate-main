"""End-to-end GNN training utilities for cross-asset market graphs.

Supports:
  - Correlation-driven graph construction from returns
  - Rolling window graph generation
  - GCN and GAT architectures
  - Node-level multi-task prediction (returns + volatility)
  - Graph-level pooled prediction
  - Model checkpointing and optional training visualisation

PyTorch Geometric is used when available. If it is missing, the module falls
back to dense PyTorch graph layers. If PyTorch itself is unavailable, graph
construction still works but training raises a clear RuntimeError.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    GATConv = None  # type: ignore[assignment]
    GCNConv = None  # type: ignore[assignment]
    global_mean_pool = None  # type: ignore[assignment]
    _PYG_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_AVAILABLE = False


@dataclass
class GraphConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100


@dataclass
class MarketGraph:
    nodes: List[str]
    edge_index: np.ndarray
    edge_weights: np.ndarray
    node_features: np.ndarray
    labels: np.ndarray


@dataclass
class _TrainingState:
    losses: List[float] = field(default_factory=list)
    node_losses: List[float] = field(default_factory=list)
    graph_losses: List[float] = field(default_factory=list)
    best_loss: float = float("inf")
    best_epoch: int = 0
    checkpoint_path: Optional[str] = None
    visualization_path: Optional[str] = None


if _TORCH_AVAILABLE:

    class _DenseGCNLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x: "torch.Tensor", adjacency: "torch.Tensor") -> "torch.Tensor":
            n_nodes = adjacency.shape[0]
            adjacency = adjacency + torch.eye(n_nodes, device=x.device, dtype=x.dtype)
            degree = adjacency.sum(dim=1).clamp_min(1e-6)
            d_inv_sqrt = torch.pow(degree, -0.5)
            normalized = d_inv_sqrt.unsqueeze(1) * adjacency * d_inv_sqrt.unsqueeze(0)
            return normalized @ self.linear(x)


    class _DenseGATLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim, bias=False)
            self.attn_src = nn.Parameter(torch.empty(out_dim))
            self.attn_dst = nn.Parameter(torch.empty(out_dim))
            self.dropout = nn.Dropout(dropout)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.uniform_(self.attn_src, -0.1, 0.1)
            nn.init.uniform_(self.attn_dst, -0.1, 0.1)

        def forward(self, x: "torch.Tensor", adjacency: "torch.Tensor") -> "torch.Tensor":
            h = self.linear(x)
            scores = (h @ self.attn_src).unsqueeze(1) + (h @ self.attn_dst).unsqueeze(0)
            scores = F.leaky_relu(scores, negative_slope=0.2)

            mask = adjacency > 0
            mask.fill_diagonal_(True)
            scores = scores.masked_fill(~mask, -1e9)
            attention = torch.softmax(scores, dim=-1)
            attention = self.dropout(attention)
            return attention @ h


    class _DenseGraphModel(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            dropout: float,
            architecture: str,
        ) -> None:
            super().__init__()
            self.architecture = architecture.lower()
            self.dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList()

            layer_in = input_dim
            for _ in range(max(1, num_layers)):
                if self.architecture == "gat":
                    layer = _DenseGATLayer(layer_in, hidden_dim, dropout=dropout)
                else:
                    layer = _DenseGCNLayer(layer_in, hidden_dim)
                self.layers.append(layer)
                layer_in = hidden_dim

            self.node_head = nn.Linear(hidden_dim, output_dim)
            self.graph_head = nn.Linear(hidden_dim, output_dim)

        def forward(
            self,
            x: "torch.Tensor",
            adjacency: "torch.Tensor",
        ) -> Dict[str, "torch.Tensor"]:
            hidden = x
            for layer in self.layers:
                hidden = layer(hidden, adjacency)
                hidden = F.relu(hidden)
                hidden = self.dropout(hidden)

            node_predictions = self.node_head(hidden)
            graph_embedding = hidden.mean(dim=0, keepdim=True)
            graph_prediction = self.graph_head(graph_embedding)
            return {
                "node_predictions": node_predictions,
                "graph_prediction": graph_prediction,
                "embeddings": hidden,
            }


    class _PyGGraphModel(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            dropout: float,
            architecture: str,
        ) -> None:
            super().__init__()
            self.architecture = architecture.lower()
            self.dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList()

            layer_in = input_dim
            for _ in range(max(1, num_layers)):
                if self.architecture == "gat":
                    layer = GATConv(layer_in, hidden_dim, heads=1, concat=False, dropout=dropout)
                else:
                    layer = GCNConv(layer_in, hidden_dim)
                self.layers.append(layer)
                layer_in = hidden_dim

            self.node_head = nn.Linear(hidden_dim, output_dim)
            self.graph_head = nn.Linear(hidden_dim, output_dim)

        def forward(
            self,
            x: "torch.Tensor",
            edge_index: "torch.Tensor",
            edge_weight: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            hidden = x
            for layer in self.layers:
                if self.architecture == "gat":
                    hidden = layer(hidden, edge_index)
                else:
                    hidden = layer(hidden, edge_index, edge_weight=edge_weight)
                hidden = F.relu(hidden)
                hidden = self.dropout(hidden)

            node_predictions = self.node_head(hidden)
            batch = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
            graph_embedding = global_mean_pool(hidden, batch)
            graph_prediction = self.graph_head(graph_embedding)
            return {
                "node_predictions": node_predictions,
                "graph_prediction": graph_prediction,
                "embeddings": hidden,
            }


class GNNTrainer:
    def __init__(
        self,
        architecture: str = "gcn",
        rolling_window: Optional[int] = None,
        checkpoint_dir: str = "checkpoints/gnn",
        visualization_dir: str = "artifacts/gnn",
        device: Optional[str] = None,
    ) -> None:
        self.architecture = architecture.lower()
        self.rolling_window = rolling_window
        self.checkpoint_dir = checkpoint_dir
        self.visualization_dir = visualization_dir
        self.device = self._resolve_device(device)
        self._last_returns: Optional[pd.DataFrame] = None
        self._last_features: Optional[pd.DataFrame] = None

    def build_adjacency_matrix(
        self,
        returns: pd.DataFrame,
        method: str = "correlation",
    ) -> np.ndarray:
        clean_returns = self._validate_returns(returns)
        if clean_returns.shape[1] == 0:
            raise ValueError("returns must contain at least one asset column")

        method_name = method.lower()
        if method_name == "correlation":
            adjacency = clean_returns.corr().fillna(0.0).to_numpy(dtype=np.float32)
        elif method_name == "covariance":
            covariance = clean_returns.cov().fillna(0.0).to_numpy(dtype=np.float32)
            scale = float(np.max(np.abs(covariance)))
            adjacency = covariance if scale <= 1e-12 else covariance / scale
        else:
            raise ValueError(f"unsupported adjacency method: {method}")

        adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency.astype(np.float32)

    def create_market_graph(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
    ) -> MarketGraph:
        clean_returns = self._validate_returns(returns)
        self._last_returns = clean_returns.copy()
        self._last_features = features.copy()

        nodes = clean_returns.columns.astype(str).tolist()
        edge_weights = self.build_adjacency_matrix(clean_returns)
        edge_index = (np.abs(edge_weights) > 0).astype(np.int64)
        node_features = self._build_node_feature_matrix(clean_returns, features)
        labels = self._build_labels(clean_returns)

        graph = MarketGraph(
            nodes=nodes,
            edge_index=edge_index,
            edge_weights=edge_weights,
            node_features=node_features,
            labels=labels,
        )
        logger.info(
            "Created market graph with %d nodes, %d features, %d tasks",
            len(nodes),
            node_features.shape[1],
            labels.shape[1],
        )
        return graph

    def train_gnn(self, graph: MarketGraph, config: GraphConfig) -> Dict[str, Any]:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GNN training")

        graphs = self._build_training_graphs(graph)
        model = self._build_model(graphs[0], config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        training_state = _TrainingState()

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.visualization_dir).mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_total_loss = 0.0
            epoch_node_loss = 0.0
            epoch_graph_loss = 0.0

            for batch_graphs in self._batch_graphs(graphs, config.batch_size):
                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=self.device)
                total_node_loss = 0.0
                total_graph_loss = 0.0

                for current_graph in batch_graphs:
                    outputs, node_labels, graph_labels = self._forward_graph(model, current_graph)
                    node_loss = F.mse_loss(outputs["node_predictions"], node_labels)
                    graph_loss = F.mse_loss(outputs["graph_prediction"], graph_labels)
                    loss = node_loss + 0.5 * graph_loss

                    total_loss = total_loss + loss
                    total_node_loss += float(node_loss.item())
                    total_graph_loss += float(graph_loss.item())

                batch_count = max(len(batch_graphs), 1)
                total_loss = total_loss / batch_count
                total_loss.backward()
                optimizer.step()

                epoch_total_loss += float(total_loss.item())
                epoch_node_loss += total_node_loss / batch_count
                epoch_graph_loss += total_graph_loss / batch_count

            num_batches = max(math.ceil(len(graphs) / max(config.batch_size, 1)), 1)
            avg_total_loss = epoch_total_loss / num_batches
            avg_node_loss = epoch_node_loss / num_batches
            avg_graph_loss = epoch_graph_loss / num_batches

            training_state.losses.append(avg_total_loss)
            training_state.node_losses.append(avg_node_loss)
            training_state.graph_losses.append(avg_graph_loss)

            if avg_total_loss < training_state.best_loss:
                training_state.best_loss = avg_total_loss
                training_state.best_epoch = epoch
                checkpoint_path = os.path.join(self.checkpoint_dir, "gnn_best_model.pt")
                self.save_model(
                    {
                        "model": model,
                        "config": config,
                        "architecture": self.architecture,
                        "feature_dim": graphs[0].node_features.shape[1],
                        "output_dim": graphs[0].labels.shape[1],
                    },
                    checkpoint_path,
                )
                training_state.checkpoint_path = checkpoint_path

            if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
                logger.info(
                    "GNN epoch %d/%d | loss=%.6f | node=%.6f | graph=%.6f",
                    epoch,
                    config.epochs,
                    avg_total_loss,
                    avg_node_loss,
                    avg_graph_loss,
                )

        visualization_path = self._save_training_visualization(training_state)
        if visualization_path is not None:
            training_state.visualization_path = visualization_path

        elapsed = time.time() - start_time
        return {
            "model": model,
            "config": config,
            "history": {
                "loss": training_state.losses,
                "node_loss": training_state.node_losses,
                "graph_loss": training_state.graph_losses,
            },
            "best_loss": training_state.best_loss,
            "best_epoch": training_state.best_epoch,
            "checkpoint_path": training_state.checkpoint_path,
            "visualization_path": training_state.visualization_path,
            "graphs_trained": len(graphs),
            "architecture": self.architecture,
            "feature_dim": graphs[0].node_features.shape[1],
            "output_dim": graphs[0].labels.shape[1],
            "pyg_enabled": _PYG_AVAILABLE,
            "training_time_s": elapsed,
        }

    def predict_with_gnn(self, graph: MarketGraph, model: Any) -> np.ndarray:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GNN inference")

        graph_model = self._unwrap_model(model)
        graph_model.eval()
        with torch.no_grad():
            outputs, _, _ = self._forward_graph(graph_model, graph)
        return outputs["node_predictions"].detach().cpu().numpy()

    def evaluate_gnn(self, model: Any, test_graph: MarketGraph) -> Dict[str, Any]:
        predictions = self.predict_with_gnn(test_graph, model)
        labels = np.asarray(test_graph.labels, dtype=np.float32)
        errors = predictions - labels
        rmse = np.sqrt(np.mean(np.square(errors), axis=0))
        mae = np.mean(np.abs(errors), axis=0)

        result = {
            "rmse_returns": float(rmse[0]) if rmse.size > 0 else 0.0,
            "rmse_volatility": float(rmse[1]) if rmse.size > 1 else 0.0,
            "mae_returns": float(mae[0]) if mae.size > 0 else 0.0,
            "mae_volatility": float(mae[1]) if mae.size > 1 else 0.0,
            "graph_mae": float(np.mean(mae)) if mae.size else 0.0,
            "num_nodes": len(test_graph.nodes),
        }
        logger.info("GNN evaluation complete: %s", result)
        return result

    def save_model(self, model: Any, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if _TORCH_AVAILABLE:
            payload = self._serialise_model_payload(model)
            if payload is not None:
                torch.save(payload, save_path)
                logger.info("Saved GNN model checkpoint to %s", save_path)
                return

        with save_path.open("wb") as handle:
            pickle.dump(model, handle)
        logger.info("Saved GNN model payload with pickle to %s", save_path)

    def load_model(self, path: str) -> Any:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(path)

        if _TORCH_AVAILABLE:
            try:
                payload = torch.load(load_path, map_location=self.device, weights_only=False)
                if isinstance(payload, dict) and {"state_dict", "metadata"}.issubset(payload):
                    metadata = payload["metadata"]
                    config_payload = metadata.get("config", {})
                    config = self._coerce_graph_config(config_payload)
                    model = self._instantiate_model_from_metadata(metadata, config)
                    model.load_state_dict(payload["state_dict"])
                    model.to(self.device)
                    model.eval()
                    logger.info("Loaded GNN checkpoint from %s", load_path)
                    return {
                        **metadata,
                        "model": model,
                        "config": config,
                    }
                logger.info("Loaded raw torch payload from %s", load_path)
                return payload
            except Exception as exc:
                logger.warning("Torch load failed for %s: %s; falling back to pickle", load_path, exc)

        with load_path.open("rb") as handle:
            payload = pickle.load(handle)
        logger.info("Loaded pickle payload from %s", load_path)
        return payload

    def create_rolling_graphs(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        window: int,
        step: int = 1,
    ) -> List[MarketGraph]:
        clean_returns = self._validate_returns(returns)
        if window < 2:
            raise ValueError("window must be at least 2")
        if clean_returns.shape[0] < window:
            return [self.create_market_graph(clean_returns, features)]

        graphs: List[MarketGraph] = []
        for end_idx in range(window, clean_returns.shape[0] + 1, max(step, 1)):
            window_returns = clean_returns.iloc[end_idx - window: end_idx]
            graphs.append(self.create_market_graph(window_returns, features))

        logger.info(
            "Created %d rolling market graphs using window=%d step=%d",
            len(graphs),
            window,
            step,
        )
        return graphs

    def _resolve_device(self, device: Optional[str]) -> str:
        if device is not None:
            return device
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _validate_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        clean_returns = returns.copy().sort_index()
        clean_returns = clean_returns.select_dtypes(include=[np.number]).fillna(0.0)
        if clean_returns.empty:
            raise ValueError("returns must contain numeric data")
        return clean_returns.astype(np.float32)

    def _build_node_feature_matrix(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
    ) -> np.ndarray:
        symbols = returns.columns.astype(str).tolist()
        derived = self._derive_return_features(returns)

        provided: Optional[np.ndarray] = None
        if isinstance(features, pd.DataFrame) and not features.empty:
            feature_frame = features.copy()
            feature_frame.index = feature_frame.index.map(str)
            feature_frame.columns = feature_frame.columns.map(str)

            if set(symbols).issubset(set(feature_frame.index)):
                aligned = feature_frame.loc[symbols]
                provided = aligned.select_dtypes(include=[np.number]).fillna(0.0).to_numpy(
                    dtype=np.float32
                )
            elif set(symbols).issubset(set(feature_frame.columns)):
                aligned = feature_frame[symbols].tail(1).T
                provided = aligned.fillna(0.0).to_numpy(dtype=np.float32)

        if provided is None or provided.size == 0:
            node_features = derived
        else:
            if provided.shape[0] != len(symbols):
                raise ValueError("features could not be aligned to asset symbols")
            node_features = np.concatenate([provided, derived], axis=1)

        return self._standardize_matrix(node_features)

    def _derive_return_features(self, returns: pd.DataFrame) -> np.ndarray:
        latest_return = returns.iloc[-1].to_numpy(dtype=np.float32)
        volatility = returns.std(axis=0).to_numpy(dtype=np.float32)
        momentum = returns.tail(min(5, len(returns))).mean(axis=0).to_numpy(dtype=np.float32)
        mean_return = returns.mean(axis=0).to_numpy(dtype=np.float32)
        return np.column_stack([latest_return, volatility, momentum, mean_return]).astype(np.float32)

    def _build_labels(self, returns: pd.DataFrame) -> np.ndarray:
        forward_window = returns.tail(min(5, len(returns)))
        future_return = forward_window.mean(axis=0).to_numpy(dtype=np.float32)
        future_volatility = forward_window.std(axis=0).fillna(0.0).to_numpy(dtype=np.float32)
        return np.column_stack([future_return, future_volatility]).astype(np.float32)

    def _standardize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(matrix, dtype=np.float32)
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (arr - mean) / std

    def _build_training_graphs(self, graph: MarketGraph) -> List[MarketGraph]:
        if self.rolling_window is None or self._last_returns is None or self._last_features is None:
            return [graph]

        try:
            rolling_graphs = self.create_rolling_graphs(
                self._last_returns,
                self._last_features,
                window=self.rolling_window,
            )
        except Exception as exc:
            logger.warning("Falling back to single graph training: %s", exc)
            return [graph]

        return rolling_graphs or [graph]

    def _batch_graphs(
        self,
        graphs: Sequence[MarketGraph],
        batch_size: int,
    ) -> List[List[MarketGraph]]:
        effective_batch = max(1, int(batch_size))
        return [list(graphs[idx: idx + effective_batch]) for idx in range(0, len(graphs), effective_batch)]

    def _build_model(self, graph: MarketGraph, config: GraphConfig) -> Any:
        input_dim = int(graph.node_features.shape[1])
        output_dim = int(graph.labels.shape[1])
        if _PYG_AVAILABLE:
            model = _PyGGraphModel(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=output_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                architecture=self.architecture,
            )
        else:
            model = _DenseGraphModel(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=output_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                architecture=self.architecture,
            )
        model.to(self.device)
        return model

    def _forward_graph(
        self,
        model: Any,
        graph: MarketGraph,
    ) -> Tuple[Dict[str, Any], "torch.Tensor", "torch.Tensor"]:
        node_features = torch.tensor(graph.node_features, dtype=torch.float32, device=self.device)
        node_labels = torch.tensor(graph.labels, dtype=torch.float32, device=self.device)
        graph_labels = node_labels.mean(dim=0, keepdim=True)

        if _PYG_AVAILABLE and isinstance(model, _PyGGraphModel):
            edge_index_np, edge_weight_np = self._dense_to_sparse(
                graph.edge_index,
                graph.edge_weights,
            )
            edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)
            edge_weight = torch.tensor(edge_weight_np, dtype=torch.float32, device=self.device)
            outputs = model(node_features, edge_index=edge_index, edge_weight=edge_weight)
        else:
            adjacency = torch.tensor(np.abs(graph.edge_weights), dtype=torch.float32, device=self.device)
            outputs = model(node_features, adjacency=adjacency)

        return outputs, node_labels, graph_labels

    def _dense_to_sparse(
        self,
        edge_index: np.ndarray,
        edge_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = np.where(edge_index > 0)
        if rows.size == 0:
            n_nodes = edge_index.shape[0]
            rows = np.arange(n_nodes)
            cols = np.arange(n_nodes)

        sparse_edge_index = np.vstack([rows, cols]).astype(np.int64)
        sparse_edge_weight = edge_weights[rows, cols].astype(np.float32)
        return sparse_edge_index, sparse_edge_weight

    def _save_training_visualization(self, training_state: _TrainingState) -> Optional[str]:
        if not _MATPLOTLIB_AVAILABLE or not training_state.losses:
            if not _MATPLOTLIB_AVAILABLE:
                logger.warning("matplotlib not available; skipping training visualisation")
            return None

        plot_path = os.path.join(self.visualization_dir, "gnn_training_curve.png")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(training_state.losses, label="total_loss", linewidth=2)
        ax.plot(training_state.node_losses, label="node_loss", linewidth=1.5)
        ax.plot(training_state.graph_losses, label="graph_loss", linewidth=1.5)
        ax.set_title("GNN Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        logger.info("Saved GNN training visualisation to %s", plot_path)
        return plot_path

    def _serialise_model_payload(self, model: Any) -> Optional[Dict[str, Any]]:
        model_obj = self._unwrap_model(model)
        if not hasattr(model_obj, "state_dict"):
            return None

        config = model.get("config") if isinstance(model, dict) else GraphConfig()
        config_payload = self._graph_config_to_dict(config)
        metadata = {
            "architecture": model.get("architecture", self.architecture)
            if isinstance(model, dict)
            else self.architecture,
            "feature_dim": model.get("feature_dim") if isinstance(model, dict) else None,
            "output_dim": model.get("output_dim") if isinstance(model, dict) else None,
            "pyg_enabled": _PYG_AVAILABLE and isinstance(model_obj, _PyGGraphModel),
            "config": config_payload,
        }
        if metadata["feature_dim"] is None or metadata["output_dim"] is None:
            return None

        return {
            "state_dict": model_obj.state_dict(),
            "metadata": metadata,
        }

    def _instantiate_model_from_metadata(self, metadata: Dict[str, Any], config: GraphConfig) -> Any:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to restore a GNN model")

        architecture = str(metadata.get("architecture", self.architecture)).lower()
        feature_dim = int(metadata["feature_dim"])
        output_dim = int(metadata["output_dim"])
        use_pyg = bool(metadata.get("pyg_enabled", False)) and _PYG_AVAILABLE

        if use_pyg:
            model = _PyGGraphModel(
                input_dim=feature_dim,
                hidden_dim=config.hidden_dim,
                output_dim=output_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                architecture=architecture,
            )
        else:
            model = _DenseGraphModel(
                input_dim=feature_dim,
                hidden_dim=config.hidden_dim,
                output_dim=output_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                architecture=architecture,
            )
        return model

    def _graph_config_to_dict(self, config: Any) -> Dict[str, Any]:
        if isinstance(config, GraphConfig):
            return {
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
            }
        if isinstance(config, dict):
            return {
                "hidden_dim": int(config.get("hidden_dim", 64)),
                "num_layers": int(config.get("num_layers", 3)),
                "dropout": float(config.get("dropout", 0.1)),
                "learning_rate": float(config.get("learning_rate", 0.001)),
                "batch_size": int(config.get("batch_size", 32)),
                "epochs": int(config.get("epochs", 100)),
            }
        return self._graph_config_to_dict(GraphConfig())

    def _coerce_graph_config(self, config: Any) -> GraphConfig:
        payload = self._graph_config_to_dict(config)
        return GraphConfig(**payload)

    def _unwrap_model(self, model: Any) -> Any:
        if isinstance(model, dict) and "model" in model:
            return model["model"]
        return model
