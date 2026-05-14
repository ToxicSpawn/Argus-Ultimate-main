"""
Graph Neural Network for Cross-Asset Flow — models how price moves
propagate between crypto assets using correlation-based adjacency and
message-passing aggregation.

Pure numpy/scipy implementation. No deep learning frameworks required.

Usage:
    gnn = AssetFlowGNN()
    for tick in ticks:
        gnn.update(tick)  # {asset: price}
    flows = gnn.propagate(n_hops=2)
    contagion = gnn.predict_contagion("BTC/USD", -5.0)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default crypto universe
DEFAULT_ASSETS = ["BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD", "ADA/USD"]

# Node feature indices
_F_RETURN = 0
_F_VOLATILITY = 1
_F_MOMENTUM = 2
_N_FEATURES = 3

# Minimum correlation to keep an edge (sparsification threshold)
_EDGE_THRESHOLD = 0.15


@dataclass
class FlowSignal:
    """Result of message-passing for a single asset."""
    flow_signal: float        # aggregated neighbor influence [-1, 1]
    influence_score: float    # how much this asset affects the graph
    leading_assets: List[str]
    lagging_assets: List[str]


class AssetFlowGNN:
    """
    Simplified GNN that models cross-asset price flow using:
      - Correlation-based adjacency matrix (edge weights)
      - Node features: return, volatility, momentum
      - Message passing: weighted mean aggregation of neighbor features
      - Granger-style lead-lag detection
      - Contagion simulation via shock propagation
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        lookback: int = 50,
    ) -> None:
        self.assets = list(assets or DEFAULT_ASSETS)
        self.lookback = max(10, lookback)
        self.n_assets = len(self.assets)
        self._asset_idx: Dict[str, int] = {a: i for i, a in enumerate(self.assets)}

        # Price history per asset
        self._prices: Dict[str, Deque[float]] = {
            a: deque(maxlen=self.lookback + 1) for a in self.assets
        }

        # Cached adjacency (recomputed when stale)
        self._adjacency: Optional[np.ndarray] = None
        self._adj_dirty = True
        self._updates_since_adj = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, prices: Dict[str, float]) -> None:
        """Update with latest prices for all tracked assets."""
        for asset in self.assets:
            p = prices.get(asset)
            if p is not None and p > 0:
                self._prices[asset].append(float(p))
        self._adj_dirty = True
        self._updates_since_adj += 1

    def compute_adjacency(self) -> np.ndarray:
        """
        Compute correlation-based adjacency matrix.
        Edge weight = max(0, |corr| - threshold) * sign(corr).
        Returns n_assets x n_assets matrix.
        """
        n = self.n_assets
        returns = self._compute_returns_matrix()
        if returns is None or returns.shape[1] < 5:
            # Not enough data — return identity-like matrix
            self._adjacency = np.eye(n) * 0.5
            self._adj_dirty = False
            return self._adjacency.copy()

        # Correlation matrix from returns
        corr = np.corrcoef(returns)
        # Handle NaN
        corr = np.nan_to_num(corr, nan=0.0)

        # Threshold + sparsify
        adj = np.where(
            np.abs(corr) > _EDGE_THRESHOLD,
            corr,
            0.0,
        )
        # Zero diagonal (no self-loops in message passing)
        np.fill_diagonal(adj, 0.0)

        self._adjacency = adj
        self._adj_dirty = False
        self._updates_since_adj = 0
        return adj.copy()

    def propagate(self, n_hops: int = 2) -> Dict[str, dict]:
        """
        Message passing: propagate signals through graph.

        At each hop, a node aggregates weighted mean of its neighbors'
        features. Edge weight = |adjacency|, feature = neighbor's latest
        [return, volatility, momentum] vector.

        Returns {asset: {flow_signal, influence_score, leading_assets, lagging_assets}}.
        """
        if self._adj_dirty or self._adjacency is None:
            self.compute_adjacency()

        adj = self._adjacency  # type: ignore[assignment]
        features = self._compute_node_features()  # (n_assets, 3)
        if features is None:
            return {a: {"flow_signal": 0.0, "influence_score": 0.0,
                        "leading_assets": [], "lagging_assets": []}
                    for a in self.assets}

        # Multi-hop message passing
        h = features.copy()  # (n, 3)
        for _ in range(max(1, n_hops)):
            abs_adj = np.abs(adj)
            # Row-normalise adjacency for weighted mean
            row_sum = abs_adj.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
            norm_adj = abs_adj / row_sum

            # Aggregate neighbors
            neighbor_msg = norm_adj @ h  # (n, 3)

            # Combine: h_new = 0.5 * h_self + 0.5 * neighbor_aggregation
            h = 0.5 * h + 0.5 * neighbor_msg

        # Compute per-asset results
        lead_lag = self.get_lead_lag()
        results: Dict[str, dict] = {}
        for asset in self.assets:
            idx = self._asset_idx[asset]
            # Flow signal: weighted return component from aggregation
            flow_signal = float(np.clip(h[idx, _F_RETURN], -1.0, 1.0))

            # Influence score: how connected this asset is (degree centrality)
            influence = float(np.abs(adj[idx]).sum() / max(self.n_assets - 1, 1))

            ll = lead_lag.get(asset, {})
            results[asset] = {
                "flow_signal": round(flow_signal, 6),
                "influence_score": round(influence, 4),
                "leading_assets": ll.get("leads", []),
                "lagging_assets": ll.get("lags", []),
            }

        return results

    def get_lead_lag(self) -> Dict[str, dict]:
        """
        Granger-style lead-lag detection using cross-correlation of returns.

        For each pair (A, B), compute cross-correlation at lag=1.
        If corr(A_t-1, B_t) > corr(B_t-1, A_t), A leads B.

        Returns {asset: {leads: [assets], lags: [assets], lead_strength: float}}.
        """
        returns_matrix = self._compute_returns_matrix()
        if returns_matrix is None or returns_matrix.shape[1] < 5:
            return {a: {"leads": [], "lags": [], "lead_strength": 0.0}
                    for a in self.assets}

        n = self.n_assets
        T = returns_matrix.shape[1]

        # Cross-correlation at lag 1
        result: Dict[str, dict] = {}
        lead_scores = np.zeros(n)

        for i in range(n):
            leads: List[str] = []
            lags: List[str] = []
            for j in range(n):
                if i == j:
                    continue
                # corr(r_i[t-1], r_j[t]) — does i lead j?
                r_i_lag = returns_matrix[i, :-1]
                r_j = returns_matrix[j, 1:]
                r_j_lag = returns_matrix[j, :-1]
                r_i = returns_matrix[i, 1:]

                if len(r_i_lag) < 3:
                    continue

                i_leads_j = self._safe_corr(r_i_lag, r_j)
                j_leads_i = self._safe_corr(r_j_lag, r_i)

                if i_leads_j > j_leads_i + 0.05:
                    leads.append(self.assets[j])
                    lead_scores[i] += i_leads_j - j_leads_i
                elif j_leads_i > i_leads_j + 0.05:
                    lags.append(self.assets[j])

            result[self.assets[i]] = {
                "leads": leads,
                "lags": lags,
                "lead_strength": round(float(lead_scores[i]), 4),
            }

        return result

    def predict_contagion(
        self,
        shocked_asset: str,
        shock_pct: float,
    ) -> Dict[str, float]:
        """
        Predict impact on other assets if shocked_asset moves by shock_pct%.

        Propagates the shock through the adjacency matrix with decay:
        impact_j = shock_pct * adj[shocked, j] * decay_factor

        Returns {asset: predicted_pct_change}.
        """
        if self._adj_dirty or self._adjacency is None:
            self.compute_adjacency()

        adj = self._adjacency  # type: ignore[assignment]
        if shocked_asset not in self._asset_idx:
            logger.warning("predict_contagion: %s not in asset universe", shocked_asset)
            return {}

        shock_idx = self._asset_idx[shocked_asset]
        impacts: Dict[str, float] = {}

        # Shock vector
        shock_vec = np.zeros(self.n_assets)
        shock_vec[shock_idx] = shock_pct

        # Propagate through adjacency (2 hops with decay)
        decay = 0.7
        current_shock = shock_vec.copy()
        total_impact = np.zeros(self.n_assets)

        for hop in range(3):
            propagated = adj.T @ current_shock  # (n,)
            propagated[shock_idx] = 0.0  # no self-feedback
            total_impact += propagated * (decay ** hop)
            current_shock = propagated * decay

        for i, asset in enumerate(self.assets):
            if asset == shocked_asset:
                impacts[asset] = shock_pct
            else:
                impacts[asset] = round(float(total_impact[i]), 4)

        return impacts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_returns_matrix(self) -> Optional[np.ndarray]:
        """
        Build (n_assets, T) matrix of log returns.
        Returns None if insufficient data.
        """
        min_len = min(len(self._prices[a]) for a in self.assets)
        if min_len < 3:
            return None

        T = min_len
        returns = np.zeros((self.n_assets, T - 1))
        for i, asset in enumerate(self.assets):
            prices = list(self._prices[asset])[-T:]
            p = np.array(prices, dtype=np.float64)
            p = np.where(p < 1e-12, 1e-12, p)
            returns[i] = np.diff(np.log(p))

        return returns

    def _compute_node_features(self) -> Optional[np.ndarray]:
        """
        Compute node feature matrix (n_assets, 3):
          [0] return: latest 1-bar return (normalized to ~[-1, 1])
          [1] volatility: rolling std of returns (normalized)
          [2] momentum: sign of cumulative 5-bar return
        """
        returns_matrix = self._compute_returns_matrix()
        if returns_matrix is None:
            return None

        n = self.n_assets
        features = np.zeros((n, _N_FEATURES))

        for i in range(n):
            rets = returns_matrix[i]
            if len(rets) == 0:
                continue
            # Latest return (scaled so typical crypto return maps to ~[-1,1])
            features[i, _F_RETURN] = np.clip(rets[-1] * 20.0, -1.0, 1.0)
            # Volatility (normalized: 1% daily vol → ~0.5)
            features[i, _F_VOLATILITY] = min(float(np.std(rets[-20:])) * 50.0, 1.0) if len(rets) >= 2 else 0.0
            # Momentum (5-bar cumulative, clipped)
            mom_window = min(5, len(rets))
            features[i, _F_MOMENTUM] = np.clip(float(np.sum(rets[-mom_window:])) * 10.0, -1.0, 1.0)

        return features

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Correlation with NaN protection."""
        if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if np.isfinite(c) else 0.0
