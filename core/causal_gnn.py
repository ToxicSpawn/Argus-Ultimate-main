"""Graph neural network with causal discovery for cross-asset relationships.

Upgrade (2026-04 peak-potential):
- Transfer Entropy estimation alongside Granger causality: TE captures
  nonlinear dependencies that linear Granger misses, improving edge quality
  on crypto assets with fat-tailed distributions.
- Attention-weighted message passing: each node aggregates neighbour messages
  weighted by a learned attention score rather than plain degree normalisation,
  allowing the GNN to focus on the most informative edges per-node.
- Rolling edge refresh: edges are re-discovered on a configurable cadence
  (every N new data points) rather than only on explicit calls, keeping the
  graph current during live trading.
- Shock confidence bands: predict_contagion now returns (mean, lower, upper)
  by running Monte Carlo perturbations of edge weights, giving a range of
  plausible contagion impacts.
- Multi-layer GNN: stacks two GNN layers with a residual connection so the
  network can capture two-hop causal propagation in its feature refinement.
- Incremental return ingestion: add_price_history appends efficiently and
  triggers an automatic edge refresh when the rolling counter fires.

All operations remain pure numpy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph representation
# ---------------------------------------------------------------------------


@dataclass
class AssetGraph:
    """Directed graph of assets and causal links."""

    nodes: List[str] = field(default_factory=list)
    edges: Dict[Tuple[str, str], float] = field(default_factory=dict)
    node_features: Dict[str, np.ndarray] = field(default_factory=dict)

    def add_node(self, symbol: str) -> None:
        if symbol not in self.nodes:
            self.nodes.append(symbol)
            self.node_features.setdefault(symbol, np.zeros(4, dtype=float))

    def add_edge(self, source: str, target: str, weight: float) -> None:
        if source == target:
            return
        self.add_node(source)
        self.add_node(target)
        self.edges[(source, target)] = float(weight)

    def neighbours(self, node: str) -> List[Tuple[str, float]]:
        return [(t, w) for (s, t), w in self.edges.items() if s == node]

    def adjacency_matrix(self) -> np.ndarray:
        n = len(self.nodes)
        idx = {s: i for i, s in enumerate(self.nodes)}
        a = np.zeros((n, n), dtype=float)
        for (src, tgt), w in self.edges.items():
            if src in idx and tgt in idx:
                a[idx[src], idx[tgt]] = w
        return a


# ---------------------------------------------------------------------------
# Attention-weighted GNN layer
# ---------------------------------------------------------------------------


class GNNLayer:
    """Single-hop graph convolution with attention weighting.

    Standard normalisation: A_norm = row-normalised (A + I).
    Attention: each edge weight is modulated by dot-product attention
    between source and target node features, then re-normalised per row.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 1337) -> None:
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / max(in_dim + out_dim, 1))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim, dtype=float)
        # Attention weight vector (concat of src+tgt projected features).
        self.a = rng.uniform(-0.1, 0.1, size=(in_dim * 2,))

    @staticmethod
    def _normalise(adj: np.ndarray) -> np.ndarray:
        a = adj + np.eye(adj.shape[0])
        deg = a.sum(axis=1, keepdims=True)
        deg = np.where(deg == 0, 1.0, deg)
        return a / deg

    def _attention_weights(self, h: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Compute attention-modulated adjacency."""
        n = h.shape[0]
        attn = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if adj[i, j] != 0 or i == j:
                    concat = np.concatenate([h[i], h[j]])
                    # Clamp dot product for numerical stability.
                    score = float(np.tanh(self.a[:len(concat)] @ concat))
                    attn[i, j] = score
        # Softmax per row over non-zero positions.
        for i in range(n):
            row = attn[i]
            mask = (adj[i] != 0)
            mask[i] = True  # self-loop
            if mask.sum() > 0:
                row_masked = row[mask]
                row_masked = row_masked - np.max(row_masked)
                exp_r = np.exp(row_masked)
                row[mask] = exp_r / (exp_r.sum() + 1e-12)
                row[~mask] = 0.0
            attn[i] = row
        return attn

    def forward(self, h: np.ndarray, adj: np.ndarray, use_attention: bool = True) -> np.ndarray:
        if h.shape[0] == 0:
            return h
        if use_attention:
            agg_matrix = self._attention_weights(h, adj)
        else:
            agg_matrix = self._normalise(adj)
        propagated = agg_matrix @ h
        out = propagated @ self.W + self.b
        return np.tanh(out)


# ---------------------------------------------------------------------------
# Causal GNN
# ---------------------------------------------------------------------------


class CausalGNN:
    """Causal discovery + attention GNN-based contagion model for assets.

    Typical usage::

        gnn = CausalGNN()
        gnn.add_asset('BTC/USD')
        gnn.add_price_history('BTC/USD', btc_returns)
        gnn.add_asset('ETH/USD')
        gnn.add_price_history('ETH/USD', eth_returns)
        edges = gnn.discover_causal_edges()
        mean, lo, hi = gnn.predict_contagion('BTC/USD', shock_size=-0.05)
    """

    def __init__(
        self,
        granger_lag: int = 2,
        min_history: int = 30,
        edge_threshold: float = 0.15,
        auto_refresh_every: int = 50,
        te_bins: int = 8,
        mc_samples: int = 64,
    ) -> None:
        self.graph = AssetGraph()
        self._returns: Dict[str, np.ndarray] = {}
        self._granger_lag = max(1, int(granger_lag))
        self._min_history = int(min_history)
        self._edge_threshold = float(edge_threshold)
        self._auto_refresh_every = int(auto_refresh_every)
        self._te_bins = int(te_bins)
        self._mc_samples = int(mc_samples)
        self._layer1 = GNNLayer(in_dim=4, out_dim=4, seed=1337)
        self._layer2 = GNNLayer(in_dim=4, out_dim=4, seed=7331)
        self._last_discovery_size = 0
        self._ingestion_counter = 0

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_asset(self, symbol: str) -> None:
        self.graph.add_node(symbol)
        self._returns.setdefault(symbol, np.array([], dtype=float))

    def add_price_history(self, symbol: str, returns: np.ndarray) -> None:
        self.add_asset(symbol)
        arr = np.asarray(returns, dtype=float).flatten()
        if arr.size == 0:
            return
        combined = np.concatenate([self._returns[symbol], arr])
        if combined.size > 4096:
            combined = combined[-4096:]
        self._returns[symbol] = combined
        self._update_node_features(symbol)
        self._ingestion_counter += int(arr.size)
        # Auto-refresh edges when enough new data has arrived.
        if self._ingestion_counter >= self._auto_refresh_every:
            self._ingestion_counter = 0
            self.discover_causal_edges()

    def _update_node_features(self, symbol: str) -> None:
        r = self._returns.get(symbol, np.array([]))
        if r.size == 0:
            self.graph.node_features[symbol] = np.zeros(4, dtype=float)
            return
        mean = float(np.mean(r))
        std = float(np.std(r) + 1e-12)
        skew = float(np.mean((r - mean) ** 3) / (std ** 3)) if std > 0 else 0.0
        last = float(r[-1])
        self.graph.node_features[symbol] = np.array([mean, std, skew, last], dtype=float)

    # ------------------------------------------------------------------
    # Causal discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _align(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = min(len(x), len(y))
        return x[-n:], y[-n:]

    def _granger_score(self, cause: np.ndarray, effect: np.ndarray) -> float:
        lag = self._granger_lag
        cause, effect = self._align(cause, effect)
        if len(effect) <= lag + 3:
            return 0.0

        def _build(series: np.ndarray, extras: Optional[List[np.ndarray]] = None) -> np.ndarray:
            rows = []
            for i in range(lag, len(effect)):
                feats = [1.0]
                feats.extend(series[i - lag: i].tolist())
                if extras is not None:
                    for e in extras:
                        feats.extend(e[i - lag: i].tolist())
                rows.append(feats)
            return np.asarray(rows, dtype=float)

        y = effect[lag:]
        x_restricted = _build(effect)
        x_full = _build(effect, extras=[cause])
        try:
            beta_r, *_ = np.linalg.lstsq(x_restricted, y, rcond=None)
            beta_f, *_ = np.linalg.lstsq(x_full, y, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        resid_r = y - x_restricted @ beta_r
        resid_f = y - x_full @ beta_f
        ss_r = float(np.sum(resid_r ** 2))
        ss_f = float(np.sum(resid_f ** 2))
        if ss_r <= 1e-12:
            return 0.0
        return float(max(0.0, min(1.0, (ss_r - ss_f) / ss_r)))

    def _transfer_entropy(self, source: np.ndarray, target: np.ndarray) -> float:
        """Approximate Transfer Entropy via histogram binning.

        TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
        Estimated using joint/marginal histogram counts.
        """
        source, target = self._align(source, target)
        lag = self._granger_lag
        if len(target) <= lag + 3:
            return 0.0
        bins = self._te_bins
        # Discretise via quantile binning for robustness.
        def _digitise(arr: np.ndarray) -> np.ndarray:
            pcts = np.linspace(0, 100, bins + 1)
            edges = np.percentile(arr, pcts)
            edges[-1] += 1e-10
            return np.digitize(arr, edges[1:-1])

        y_fut = _digitise(target[lag:])
        y_past = _digitise(target[:-lag][-len(y_fut):])
        x_past = _digitise(source[:-lag][-len(y_fut):])

        def _entropy(a: np.ndarray) -> float:
            _, counts = np.unique(a, return_counts=True)
            p = counts / counts.sum()
            return float(-np.sum(p * np.log(p + 1e-12)))

        def _joint_entropy(a: np.ndarray, b: np.ndarray) -> float:
            pairs = a * (bins + 1) + b
            return _entropy(pairs)

        def _cond_entropy(a: np.ndarray, given: np.ndarray) -> float:
            return _joint_entropy(a, given) - _entropy(given)

        try:
            h_yfut_given_ypast = _cond_entropy(y_fut, y_past)
            triplet = y_past * (bins + 1) ** 2 + x_past * (bins + 1) + y_fut
            _, counts_xyz = np.unique(triplet, return_counts=True)
            p_xyz = counts_xyz / counts_xyz.sum()
            pairs_xz = y_past * (bins + 1) + x_past
            _, counts_xz = np.unique(pairs_xz, return_counts=True)
            p_xz = counts_xz / counts_xz.sum()
            h_yfut_given_ypast_xpast = (
                float(-np.sum(p_xyz * np.log(p_xyz + 1e-12)))
                - float(-np.sum(p_xz * np.log(p_xz + 1e-12)))
            )
            te = h_yfut_given_ypast - h_yfut_given_ypast_xpast
            return float(max(0.0, min(1.0, te)))
        except Exception:
            return 0.0

    def _correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = self._align(a, b)
        if len(a) < 3:
            return 0.0
        a_std, b_std = float(np.std(a)), float(np.std(b))
        if a_std < 1e-12 or b_std < 1e-12:
            return 0.0
        return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))

    def discover_causal_edges(self) -> List[Tuple[str, str, float]]:
        """Run Granger + Transfer Entropy + correlation and populate edges.

        Edge weight = 0.4 * granger + 0.4 * transfer_entropy + 0.2 * correlation.
        Returns list of (source, target, weight) tuples.
        """
        discovered: List[Tuple[str, str, float]] = []
        self.graph.edges.clear()
        symbols = [s for s, r in self._returns.items() if len(r) >= self._min_history]
        for src in symbols:
            for tgt in symbols:
                if src == tgt:
                    continue
                g = self._granger_score(self._returns[src], self._returns[tgt])
                te = self._transfer_entropy(self._returns[src], self._returns[tgt])
                c = self._correlation(self._returns[src], self._returns[tgt])
                weight = 0.4 * g + 0.4 * te + 0.2 * c
                if abs(weight) >= self._edge_threshold:
                    self.graph.add_edge(src, tgt, float(np.clip(weight, -1.0, 1.0)))
                    discovered.append((src, tgt, float(weight)))
        self._last_discovery_size = len(discovered)
        logger.debug("causal_gnn: discovered %d edges (Granger+TE+corr)", len(discovered))
        return discovered

    # ------------------------------------------------------------------
    # Contagion forecasting with confidence bands
    # ------------------------------------------------------------------

    def predict_contagion(
        self,
        source: str,
        shock_size: float,
        n_hops: int = 3,
        decay: float = 0.7,
        confidence: bool = True,
    ) -> Dict[str, Any]:
        """Propagate a shock from ``source`` through the asset graph.

        Returns a dict mapping each asset symbol to:
        ``{'mean': float, 'lower': float, 'upper': float}`` if
        ``confidence=True``, else ``{'mean': float}``.

        Monte Carlo: edge weights are perturbed by N(0, 0.05) for each
        sample, giving a range of plausible outcomes.
        """
        if source not in self.graph.nodes:
            return {}
        n = len(self.graph.nodes)
        if n == 0:
            return {}
        idx = {s: i for i, s in enumerate(self.graph.nodes)}
        base_adj = self.graph.adjacency_matrix()
        src_idx = idx[source]

        def _propagate(adj: np.ndarray) -> np.ndarray:
            adj_norm = GNNLayer._normalise(adj)
            shock = np.zeros(n, dtype=float)
            shock[src_idx] = float(shock_size)
            impact = shock.copy()
            current = shock.copy()
            for _ in range(max(1, n_hops)):
                current = decay * (adj_norm.T @ current)
                impact += current
            return impact

        base_impact = _propagate(base_adj)
        if not confidence:
            return {sym: {"mean": float(base_impact[idx[sym]])} for sym in self.graph.nodes}

        rng = np.random.default_rng(42)
        samples = np.zeros((self._mc_samples, n), dtype=float)
        for i in range(self._mc_samples):
            noise = rng.normal(0.0, 0.05, size=base_adj.shape)
            perturbed = np.clip(base_adj + noise * (base_adj != 0), -1.0, 1.0)
            samples[i] = _propagate(perturbed)

        result: Dict[str, Any] = {}
        for sym in self.graph.nodes:
            si = idx[sym]
            mean_v = float(np.mean(samples[:, si]))
            lo = float(np.percentile(samples[:, si], 5))
            hi = float(np.percentile(samples[:, si], 95))
            result[sym] = {"mean": mean_v, "lower": lo, "upper": hi}
        return result

    # ------------------------------------------------------------------
    # Multi-layer GNN feature refinement
    # ------------------------------------------------------------------

    def refine_features(self) -> Dict[str, np.ndarray]:
        """Run two stacked GNN layers with a residual connection."""
        if not self.graph.nodes:
            return {}
        feats = np.stack([self.graph.node_features[s] for s in self.graph.nodes])
        adj = self.graph.adjacency_matrix()
        h1 = self._layer1.forward(feats, adj, use_attention=True)
        h2 = self._layer2.forward(h1, adj, use_attention=True)
        # Residual: add original features (same dim).
        h_out = np.tanh(h2 + feats)
        return {sym: h_out[i] for i, sym in enumerate(self.graph.nodes)}

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_nodes": len(self.graph.nodes),
            "n_edges": len(self.graph.edges),
            "last_discovery_size": self._last_discovery_size,
            "nodes": list(self.graph.nodes),
            "edges": [
                {"source": s, "target": t, "weight": float(w)}
                for (s, t), w in self.graph.edges.items()
            ],
            "history_lengths": {s: int(len(r)) for s, r in self._returns.items()},
            "ingestion_counter": self._ingestion_counter,
        }


__all__ = ["AssetGraph", "GNNLayer", "CausalGNN"]
