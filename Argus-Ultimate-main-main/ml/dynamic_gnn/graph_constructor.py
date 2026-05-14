"""Graph construction utilities from rolling market data and metadata."""

# pyright: reportMissingImports=false, reportConstantRedefinition=false

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .dynamic_graph import DynamicGraphBuilder, GraphSnapshot

logger = logging.getLogger(__name__)

try:
    from scipy import stats as scipy_stats

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    scipy_stats = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False


class MarketGraphConstructor:
    """Construct dynamic graphs using correlation, causality, entropy, and metadata."""

    def __init__(
        self,
        correlation_threshold: float = 0.25,
        causality_threshold: float = 0.05,
        entropy_threshold: float = 0.01,
    ) -> None:
        self.correlation_threshold = float(correlation_threshold)
        self.causality_threshold = float(causality_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.graph_builder = DynamicGraphBuilder(correlation_threshold=correlation_threshold)

    def rolling_correlation_matrices(self, returns: np.ndarray, window_size: int = 32) -> np.ndarray:
        data = np.asarray(returns, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("returns must be 2D")
        matrices = []
        for end_idx in range(max(window_size, 2), data.shape[0] + 1):
            window = data[end_idx - window_size:end_idx]
            matrices.append(self.graph_builder._safe_corrcoef(window))
        return np.stack(matrices, axis=0) if matrices else np.zeros((0, data.shape[1], data.shape[1]), dtype=np.float32)

    def granger_causality_matrix(self, returns: np.ndarray, max_lag: int = 2) -> np.ndarray:
        data = np.asarray(returns, dtype=np.float32)
        n_assets = data.shape[1]
        matrix = np.zeros((n_assets, n_assets), dtype=np.float32)
        for src in range(n_assets):
            for dst in range(n_assets):
                if src == dst:
                    continue
                matrix[src, dst] = self._granger_score(data[:, src], data[:, dst], max_lag=max_lag)
        return matrix

    def transfer_entropy_matrix(self, returns: np.ndarray, bins: int = 8) -> np.ndarray:
        data = np.asarray(returns, dtype=np.float32)
        n_assets = data.shape[1]
        matrix = np.zeros((n_assets, n_assets), dtype=np.float32)
        for src in range(n_assets):
            for dst in range(n_assets):
                if src == dst:
                    continue
                matrix[src, dst] = self._transfer_entropy_score(data[:, src], data[:, dst], bins=bins)
        return matrix

    def integrate_metadata(
        self,
        assets: Sequence[str],
        sectors: Optional[Mapping[str, str]] = None,
        industries: Optional[Mapping[str, str]] = None,
        supply_chain_links: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        assets_list = [str(asset) for asset in assets]
        n_assets = len(assets_list)
        metadata = np.zeros((n_assets, n_assets), dtype=np.float32)
        for i, asset_i in enumerate(assets_list):
            for j, asset_j in enumerate(assets_list):
                if i == j:
                    continue
                sector_match = float(sectors is not None and sectors.get(asset_i) == sectors.get(asset_j))
                industry_match = float(industries is not None and industries.get(asset_i) == industries.get(asset_j))
                supply_chain_match = float(
                    supply_chain_links is not None and asset_j in set(supply_chain_links.get(asset_i, []))
                )
                metadata[i, j] = 0.45 * sector_match + 0.35 * industry_match + 0.20 * supply_chain_match
        return metadata

    def build_graph_sequence(
        self,
        returns: np.ndarray,
        assets: Sequence[str],
        timestamps: Sequence[Any],
        window_size: int = 32,
        sectors: Optional[Mapping[str, str]] = None,
        industries: Optional[Mapping[str, str]] = None,
        supply_chain_links: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> List[GraphSnapshot]:
        corr_seq = self.rolling_correlation_matrices(returns, window_size=window_size)
        gc_matrix = self.granger_causality_matrix(returns)
        te_matrix = self.transfer_entropy_matrix(returns)
        metadata = self.integrate_metadata(
            assets=assets,
            sectors=sectors,
            industries=industries,
            supply_chain_links=supply_chain_links,
        )

        snapshots = self.graph_builder.generate_snapshots(
            returns=returns,
            assets=assets,
            timestamps=timestamps,
            window_size=window_size,
            sectors=sectors,
            industries=industries,
        )

        if snapshots and corr_seq.shape[0] == len(snapshots):
            for idx, snapshot in enumerate(snapshots):
                snapshot.relation_adjacency["correlation"] = corr_seq[idx]
                snapshot.relation_adjacency["granger"] = np.where(np.abs(gc_matrix) >= self.causality_threshold, gc_matrix, 0.0)
                snapshot.relation_adjacency["transfer_entropy"] = np.where(np.abs(te_matrix) >= self.entropy_threshold, te_matrix, 0.0)
                snapshot.relation_adjacency["metadata"] = metadata
        logger.info("Built %d graph snapshots with causality and metadata relations", len(snapshots))
        return snapshots

    def _granger_score(self, source: np.ndarray, target: np.ndarray, max_lag: int = 2) -> float:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        max_lag = max(1, int(max_lag))
        if source.size <= max_lag + 2 or target.size <= max_lag + 2:
            return 0.0

        y = target[max_lag:]
        own_lags = np.column_stack([target[max_lag - lag:-lag] for lag in range(1, max_lag + 1)])
        joint_lags = np.column_stack(
            [target[max_lag - lag:-lag] for lag in range(1, max_lag + 1)]
            + [source[max_lag - lag:-lag] for lag in range(1, max_lag + 1)]
        )
        own_beta, *_ = np.linalg.lstsq(own_lags, y, rcond=None)
        joint_beta, *_ = np.linalg.lstsq(joint_lags, y, rcond=None)
        own_res = y - own_lags @ own_beta
        joint_res = y - joint_lags @ joint_beta
        own_var = float(np.mean(own_res ** 2))
        joint_var = float(np.mean(joint_res ** 2))
        if own_var <= 1e-12:
            return 0.0
        improvement = max(0.0, (own_var - joint_var) / own_var)
        if _SCIPY_AVAILABLE and scipy_stats is not None:
            try:
                dof_num = max_lag
                dof_den = max(len(y) - 2 * max_lag - 1, 1)
                f_score = ((own_var - joint_var) / max(dof_num, 1)) / max(joint_var / dof_den, 1e-12)
                p_value = float(1.0 - scipy_stats.f.cdf(f_score, dof_num, dof_den))
                improvement *= float(max(0.0, 1.0 - p_value))
            except Exception:
                logger.debug("F-test fallback triggered for Granger score", exc_info=True)
        return float(np.clip(improvement, 0.0, 1.0))

    def _transfer_entropy_score(self, source: np.ndarray, target: np.ndarray, bins: int = 8) -> float:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if source.size < 4 or target.size < 4:
            return 0.0
        src_disc = self._digitize(source[:-1], bins=bins)
        tgt_prev = self._digitize(target[:-1], bins=bins)
        tgt_next = self._digitize(target[1:], bins=bins)
        joint = np.stack([src_disc, tgt_prev, tgt_next], axis=1)
        p_xyz = self._empirical_prob(joint)
        p_xy = self._empirical_prob(joint[:, :2])
        p_yz = self._empirical_prob(joint[:, 1:])
        p_y = self._empirical_prob(tgt_prev[:, None])
        score = 0.0
        for key, prob_xyz in p_xyz.items():
            x, y, z = key
            denom = p_xy.get((x, y), 0.0) * p_yz.get((y, z), 0.0)
            numer = prob_xyz * p_y.get((y,), 0.0)
            if denom > 0.0 and numer > 0.0:
                score += prob_xyz * np.log((numer + 1e-12) / (denom + 1e-12))
        return float(max(0.0, score))

    @staticmethod
    def _digitize(values: np.ndarray, bins: int = 8) -> np.ndarray:
        quantiles = np.linspace(0.0, 1.0, max(2, bins + 1))
        edges = np.unique(np.quantile(values, quantiles))
        if edges.size < 2:
            return np.zeros_like(values, dtype=np.int64)
        return np.digitize(values, edges[1:-1], right=False).astype(np.int64)

    @staticmethod
    def _empirical_prob(values: np.ndarray) -> dict[tuple[int, ...], float]:
        unique, counts = np.unique(values, axis=0, return_counts=True)
        total = float(np.sum(counts))
        return {tuple(row.tolist()): float(count / total) for row, count in zip(unique, counts)}
