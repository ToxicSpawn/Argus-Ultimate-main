"""Dynamic graph structures and snapshot generation for cross-asset analysis."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        return datetime.now(timezone.utc)
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            pass
    if isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[s]").astype(np.int64)
        return datetime.fromtimestamp(int(seconds), tz=timezone.utc)
    return datetime.fromisoformat(str(value))


@dataclass(slots=True)
class DynamicEdge:
    source: int
    target: int
    weight: float
    relation: str = "correlation"
    features: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DynamicGraph:
    nodes: list[str]
    edges: list[DynamicEdge]
    timestamps: list[datetime]
    node_features: np.ndarray | None = None
    graph_features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = [str(node) for node in self.nodes]
        self.timestamps = [_coerce_timestamp(ts) for ts in self.timestamps]
        if self.node_features is not None:
            node_features = np.asarray(self.node_features, dtype=np.float32)
            if node_features.ndim != 2:
                raise ValueError("node_features must be 2D")
            if node_features.shape[0] != len(self.nodes):
                raise ValueError("node_features rows must match number of nodes")
            self.node_features = node_features

    @property
    def edge_index(self) -> np.ndarray:
        if not self.edges:
            return np.zeros((2, 0), dtype=np.int64)
        return np.asarray([[edge.source, edge.target] for edge in self.edges], dtype=np.int64).T

    @property
    def edge_weights(self) -> np.ndarray:
        if not self.edges:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray([edge.weight for edge in self.edges], dtype=np.float32)

    @property
    def edge_features(self) -> np.ndarray:
        if not self.edges:
            return np.zeros((0, 0), dtype=np.float32)
        feature_names = sorted({key for edge in self.edges for key in edge.features})
        if not feature_names:
            return np.zeros((len(self.edges), 0), dtype=np.float32)
        matrix = np.zeros((len(self.edges), len(feature_names)), dtype=np.float32)
        for row, edge in enumerate(self.edges):
            for col, feature_name in enumerate(feature_names):
                matrix[row, col] = float(edge.features.get(feature_name, 0.0))
        return matrix

    @property
    def relation_types(self) -> list[str]:
        return sorted({edge.relation for edge in self.edges})


@dataclass(slots=True)
class GraphSnapshot:
    timestamp: datetime
    graph: DynamicGraph
    adjacency: np.ndarray
    relation_adjacency: dict[str, np.ndarray] = field(default_factory=dict)


class DynamicGraphBuilder:
    """Build time-evolving market graphs from rolling windows of asset data."""

    def __init__(
        self,
        correlation_threshold: float = 0.25,
        spillover_threshold: float = 0.10,
        sector_weight: float = 0.35,
        max_edges_per_node: int | None = 12,
    ) -> None:
        self.correlation_threshold = float(correlation_threshold)
        self.spillover_threshold = float(spillover_threshold)
        self.sector_weight = float(sector_weight)
        self.max_edges_per_node: int | None = None if max_edges_per_node is None else int(max_edges_per_node)

    def build_graph(
        self,
        returns_window: np.ndarray,
        assets: Sequence[str],
        timestamp: Any,
        volatility_window: np.ndarray | None = None,
        sectors: Mapping[str, str] | None = None,
        industries: Mapping[str, str] | None = None,
        base_features: np.ndarray | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> GraphSnapshot:
        assets_list = [str(asset) for asset in assets]
        returns_arr = np.asarray(returns_window, dtype=np.float32)
        if returns_arr.ndim != 2:
            raise ValueError("returns_window must be 2D with shape (timesteps, assets)")
        if returns_arr.shape[1] != len(assets_list):
            raise ValueError("returns_window columns must match assets length")

        volatility_arr = (
            np.asarray(volatility_window, dtype=np.float32)
            if volatility_window is not None
            else self._rolling_volatility(returns_arr)
        )
        if volatility_arr.shape != returns_arr.shape:
            raise ValueError("volatility_window must match returns_window shape")

        corr = self._safe_corrcoef(returns_arr)
        spillover = self._detect_volatility_spillovers(returns_arr, volatility_arr)
        sector_scores = self._build_sector_similarity(assets_list, sectors=sectors, industries=industries)

        edges = self._build_edges(corr, spillover, sector_scores)
        node_features = base_features
        if node_features is None:
            node_features = self._build_node_features(returns_arr, volatility_arr, sector_scores)

        graph_timestamp = _coerce_timestamp(timestamp)
        graph = DynamicGraph(
            nodes=assets_list,
            edges=edges,
            timestamps=[graph_timestamp],
            node_features=node_features,
            graph_features={
                "avg_abs_correlation": float(np.mean(np.abs(corr))),
                "avg_spillover": float(np.mean(spillover)),
                "density": float(len(edges) / max(len(assets_list) * max(len(assets_list) - 1, 1), 1)),
            },
            metadata=dict(extra_metadata or {}),
        )

        relation_adjacency = {
            "correlation": np.where(np.abs(corr) >= self.correlation_threshold, corr, 0.0).astype(np.float32),
            "spillover": np.where(np.abs(spillover) >= self.spillover_threshold, spillover, 0.0).astype(np.float32),
            "sector": sector_scores.astype(np.float32),
        }
        adjacency = self._edges_to_adjacency(len(assets_list), edges)
        return GraphSnapshot(
            timestamp=graph_timestamp,
            graph=graph,
            adjacency=adjacency,
            relation_adjacency=relation_adjacency,
        )

    def generate_snapshots(
        self,
        returns: np.ndarray,
        assets: Sequence[str],
        timestamps: Sequence[Any],
        window_size: int = 32,
        volatility: np.ndarray | None = None,
        sectors: Mapping[str, str] | None = None,
        industries: Mapping[str, str] | None = None,
        node_features: np.ndarray | None = None,
    ) -> list[GraphSnapshot]:
        returns_arr = np.asarray(returns, dtype=np.float32)
        if returns_arr.ndim != 2:
            raise ValueError("returns must be 2D with shape (timesteps, assets)")
        if len(timestamps) != returns_arr.shape[0]:
            raise ValueError("timestamps length must match return rows")

        snapshots: list[GraphSnapshot] = []
        for end_idx in range(max(window_size, 2), returns_arr.shape[0] + 1):
            start_idx = max(0, end_idx - window_size)
            feature_slice = None
            if node_features is not None:
                feature_slice = np.asarray(node_features[end_idx - 1], dtype=np.float32)
            snapshots.append(
                self.build_graph(
                    returns_window=returns_arr[start_idx:end_idx],
                    assets=assets,
                    timestamp=timestamps[end_idx - 1],
                    volatility_window=None if volatility is None else np.asarray(volatility[start_idx:end_idx], dtype=np.float32),
                    sectors=sectors,
                    industries=industries,
                    base_features=feature_slice,
                    extra_metadata={"window_start": start_idx, "window_end": end_idx - 1},
                )
            )
        logger.info("Generated %d dynamic graph snapshots", len(snapshots))
        return snapshots

    def _build_edges(
        self,
        corr: np.ndarray,
        spillover: np.ndarray,
        sector_scores: np.ndarray,
    ) -> list[DynamicEdge]:
        n_assets = corr.shape[0]
        ranked_edges: list[tuple[float, DynamicEdge]] = []
        for src in range(n_assets):
            for dst in range(n_assets):
                if src == dst:
                    continue
                corr_value = float(corr[src, dst])
                spill_value = float(spillover[src, dst])
                sector_value = float(sector_scores[src, dst])
                relation_strength = 0.6 * abs(corr_value) + 0.25 * abs(spill_value) + 0.15 * sector_value
                if relation_strength < min(self.correlation_threshold, self.spillover_threshold):
                    continue

                relation = "correlation"
                if abs(spill_value) >= max(abs(corr_value), self.spillover_threshold):
                    relation = "spillover"
                elif sector_value > self.sector_weight:
                    relation = "sector"

                ranked_edges.append(
                    (
                        relation_strength,
                        DynamicEdge(
                            source=src,
                            target=dst,
                            weight=float(np.clip(0.7 * corr_value + 0.2 * spill_value + 0.1 * sector_value, -1.0, 1.0)),
                            relation=relation,
                            features={
                                "correlation": corr_value,
                                "spillover": spill_value,
                                "sector_similarity": sector_value,
                                "strength": relation_strength,
                            },
                        ),
                    )
                )

        ranked_edges.sort(key=lambda item: item[0], reverse=True)
        if self.max_edges_per_node is None:
            return [edge for _, edge in ranked_edges]

        counts = np.zeros(n_assets, dtype=np.int64)
        selected: list[DynamicEdge] = []
        for _, edge in ranked_edges:
            if counts[edge.source] >= self.max_edges_per_node:
                continue
            selected.append(edge)
            counts[edge.source] += 1
        return selected

    @staticmethod
    def _safe_corrcoef(values: np.ndarray) -> np.ndarray:
        if values.shape[0] < 2:
            return np.zeros((values.shape[1], values.shape[1]), dtype=np.float32)
        corr = np.corrcoef(values, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        return corr.astype(np.float32)

    @staticmethod
    def _rolling_volatility(returns: np.ndarray) -> np.ndarray:
        centered = returns - np.nanmean(returns, axis=0, keepdims=True)
        variance = np.nanmean(centered ** 2, axis=0, keepdims=True)
        return np.repeat(np.sqrt(np.maximum(variance, 1e-12)), returns.shape[0], axis=0).astype(np.float32)

    def _detect_volatility_spillovers(self, returns: np.ndarray, volatility: np.ndarray) -> np.ndarray:
        n_assets = returns.shape[1]
        spill = np.zeros((n_assets, n_assets), dtype=np.float32)
        diff_vol = np.diff(volatility, axis=0, prepend=volatility[:1])
        for src in range(n_assets):
            for dst in range(n_assets):
                if src == dst:
                    continue
                lagged_src = diff_vol[:-1, src]
                future_dst = diff_vol[1:, dst]
                same_time = returns[1:, dst] * returns[:-1, src]
                spill[src, dst] = float(
                    0.7 * self._safe_corr_1d(lagged_src, future_dst)
                    + 0.3 * self._safe_corr_1d(lagged_src, same_time)
                )
        return spill

    def _build_sector_similarity(
        self,
        assets: Sequence[str],
        sectors: Mapping[str, str] | None = None,
        industries: Mapping[str, str] | None = None,
    ) -> np.ndarray:
        size = len(assets)
        matrix = np.zeros((size, size), dtype=np.float32)
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                if i == j:
                    continue
                sector_match = float(
                    sectors is not None
                    and str(sectors.get(asset_i, "unknown")) == str(sectors.get(asset_j, "unknown"))
                )
                industry_match = float(
                    industries is not None
                    and str(industries.get(asset_i, "unknown")) == str(industries.get(asset_j, "unknown"))
                )
                matrix[i, j] = np.float32(0.6 * sector_match + 0.4 * industry_match)
        return matrix

    def _build_node_features(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        sector_scores: np.ndarray,
    ) -> np.ndarray:
        latest_returns = returns[-1]
        momentum = returns.mean(axis=0)
        vol = volatility[-1]
        connectedness = np.mean(np.abs(sector_scores), axis=1)
        features = np.stack([latest_returns, momentum, vol, connectedness], axis=1)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _edges_to_adjacency(n_nodes: int, edges: Iterable[DynamicEdge]) -> np.ndarray:
        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for edge in edges:
            adjacency[edge.source, edge.target] = float(edge.weight)
        return adjacency

    @staticmethod
    def _safe_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2:
            return 0.0
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
