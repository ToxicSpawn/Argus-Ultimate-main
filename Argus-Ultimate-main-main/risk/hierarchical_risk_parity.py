#!/usr/bin/env python3
"""
Hierarchical Risk Parity (HRP) — Marcos Lopez de Prado's tree-based allocation.

Steps:
    1. Compute distance matrix from correlation of returns.
    2. Hierarchical (agglomerative, single-linkage) clustering.
    3. Quasi-diagonalisation of the covariance matrix.
    4. Recursive bisection to allocate weights.

Fully functional in pure Python; optionally accelerated with scipy.cluster.hierarchy.

Standalone usage:
    hrp = HierarchicalRiskParity()
    weights = hrp.compute_weights(returns_dict)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore[import-untyped]

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]

try:
    from scipy.cluster.hierarchy import linkage as scipy_linkage  # type: ignore[import-untyped]
    from scipy.spatial.distance import squareform as scipy_squareform  # type: ignore[import-untyped]

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Pure-Python linear algebra helpers
# ---------------------------------------------------------------------------

def _py_corr_cov(returns: Dict[str, List[float]], symbols: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Compute correlation and covariance matrices from return series (pure Python).

    Returns (correlation_matrix, covariance_matrix) as 2D lists.
    """
    n = len(symbols)
    T = min(len(returns[s]) for s in symbols)
    if T < 2:
        ident = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return ident, ident

    data = [returns[s][-T:] for s in symbols]
    means = [sum(row) / T for row in data]
    stds = []
    for i in range(n):
        var = sum((data[i][t] - means[i]) ** 2 for t in range(T)) / (T - 1)
        stds.append(math.sqrt(var) if var > 0 else 1e-10)

    cov = [[0.0] * n for _ in range(n)]
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            c = sum((data[i][t] - means[i]) * (data[j][t] - means[j]) for t in range(T)) / (T - 1)
            cov[i][j] = c
            cov[j][i] = c
            r = c / (stds[i] * stds[j])
            r = max(-1.0, min(1.0, r))
            corr[i][j] = r
            corr[j][i] = r
    return corr, cov


def _distance_matrix(corr: List[List[float]]) -> List[List[float]]:
    """Convert correlation matrix to distance: d(i,j) = sqrt(0.5*(1 - corr(i,j)))."""
    n = len(corr)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(max(0.0, 0.5 * (1.0 - corr[i][j])))
            dist[i][j] = d
            dist[j][i] = d
    return dist


# ---------------------------------------------------------------------------
# Pure-Python single-linkage clustering
# ---------------------------------------------------------------------------

def _single_linkage(dist: List[List[float]], n: int) -> List[Tuple[int, int, float, int]]:
    """
    Single-linkage agglomerative clustering (pure Python).

    Returns list of (cluster_a, cluster_b, distance, size) — same format as
    scipy.cluster.hierarchy.linkage output.
    """
    # Active clusters: id -> set of original indices
    active: Dict[int, set] = {i: {i} for i in range(n)}
    next_id = n
    result: List[Tuple[int, int, float, int]] = []

    # Current inter-cluster distances
    d: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d[(i, j)] = dist[i][j]

    for _ in range(n - 1):
        # Find closest pair
        best_pair: Optional[Tuple[int, int]] = None
        best_dist = float("inf")
        for (a, b), dval in d.items():
            if dval < best_dist:
                best_dist = dval
                best_pair = (a, b)

        if best_pair is None:
            break

        a, b = best_pair
        new_size = len(active[a]) + len(active[b])
        result.append((a, b, best_dist, new_size))

        # Create new cluster
        new_cluster = active[a] | active[b]
        active[next_id] = new_cluster
        del active[a]
        del active[b]

        # Update distances (single linkage = min)
        to_remove = [k for k in d if a in k or b in k]
        for k in to_remove:
            del d[k]

        for cid in active:
            if cid == next_id:
                continue
            # Min distance between any pair of original points
            min_d = float("inf")
            for pi in new_cluster:
                for pj in active[cid]:
                    dd = dist[pi][pj] if pi < pj else (dist[pj][pi] if pj < pi else 0.0)
                    if dd < min_d:
                        min_d = dd
            key = (min(cid, next_id), max(cid, next_id))
            d[key] = min_d

        next_id += 1

    return result


# ---------------------------------------------------------------------------
# Quasi-diagonalisation
# ---------------------------------------------------------------------------

def _get_quasi_diag(link: List[Tuple[int, int, float, int]], n: int) -> List[int]:
    """
    Quasi-diagonalise the covariance matrix using the dendrogram ordering.

    Returns a list of original indices in quasi-diagonal order.
    """
    # Build tree
    tree: Dict[int, List[int]] = {i: [i] for i in range(n)}
    for idx, (a, b, _, _) in enumerate(link):
        a_int, b_int = int(a), int(b)
        new_id = n + idx
        left = tree.get(a_int, [a_int])
        right = tree.get(b_int, [b_int])
        tree[new_id] = left + right

    # Root is last merge
    root = n + len(link) - 1
    return tree.get(root, list(range(n)))


# ---------------------------------------------------------------------------
# Recursive bisection
# ---------------------------------------------------------------------------

def _get_cluster_var(cov: Any, indices: List[int]) -> float:
    """Inverse-variance portfolio variance for a sub-cluster."""
    if _HAS_NUMPY:
        sub_cov = cov[np.ix_(indices, indices)]
        ivp = 1.0 / np.diag(sub_cov)
        ivp /= ivp.sum()
        return float(ivp @ sub_cov @ ivp)
    else:
        n = len(indices)
        diag = [cov[i][i] for i in indices]
        total_inv = sum(1.0 / d if d > 1e-15 else 1e15 for d in diag)
        if total_inv < 1e-15:
            return 1.0
        ivp = [(1.0 / d if d > 1e-15 else 1e15) / total_inv for d in diag]
        # w^T Sigma w
        var = 0.0
        for i_idx, i in enumerate(indices):
            for j_idx, j in enumerate(indices):
                var += ivp[i_idx] * ivp[j_idx] * cov[i][j]
        return var


def _recursive_bisect(cov: Any, sort_ix: List[int]) -> Dict[int, float]:
    """
    Recursively bisect sorted indices to assign HRP weights.

    Returns dict of original_index -> weight.
    """
    weights: Dict[int, float] = {i: 1.0 for i in sort_ix}
    clusters = [sort_ix]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            var_left = _get_cluster_var(cov, left)
            var_right = _get_cluster_var(cov, right)

            total_var = var_left + var_right
            if total_var < 1e-15:
                alpha = 0.5
            else:
                alpha = 1.0 - var_left / total_var

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1.0 - alpha)

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        clusters = new_clusters

    return weights


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) allocation following Lopez de Prado (2016).

    Parameters
    ----------
    min_history : int
        Minimum return observations per asset (default 20).
    """

    def __init__(self, min_history: int = 20):
        self.min_history = min_history
        self._last_weights: Dict[str, float] = {}
        self._last_compute_ms: float = 0.0
        logger.info(
            "HierarchicalRiskParity initialised (min_history=%d, numpy=%s, scipy=%s)",
            min_history, _HAS_NUMPY, _HAS_SCIPY,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_weights(self, returns: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute HRP portfolio weights.

        Parameters
        ----------
        returns : dict
            Symbol -> list of periodic returns.

        Returns
        -------
        dict
            Symbol -> weight (sums to ~1.0).
        """
        t0 = time.monotonic()
        symbols = sorted(returns.keys())
        n = len(symbols)

        if n == 0:
            return {}
        if n == 1:
            self._last_weights = {symbols[0]: 1.0}
            self._last_compute_ms = (time.monotonic() - t0) * 1000
            return {symbols[0]: 1.0}

        # Filter insufficient history
        valid = [s for s in symbols if len(returns[s]) >= self.min_history]
        if len(valid) < 2:
            logger.warning("HRP: fewer than 2 assets with sufficient history; equal weight fallback")
            w = {s: 1.0 / n for s in symbols}
            self._last_weights = w
            self._last_compute_ms = (time.monotonic() - t0) * 1000
            return w
        symbols = valid
        n = len(symbols)

        # Step 1: Correlation + covariance
        if _HAS_NUMPY:
            T = min(len(returns[s]) for s in symbols)
            mat = np.array([returns[s][-T:] for s in symbols])
            corr_mat = np.corrcoef(mat)
            cov_mat = np.cov(mat)
            if corr_mat.ndim == 0:
                corr_mat = np.array([[1.0]])
                cov_mat = np.array([[float(cov_mat)]])
            dist_mat = np.sqrt(np.clip(0.5 * (1.0 - corr_mat), 0.0, 1.0))
        else:
            corr_mat, cov_mat = _py_corr_cov(returns, symbols)
            dist_mat = _distance_matrix(corr_mat)

        # Step 2: Hierarchical clustering
        if _HAS_SCIPY and _HAS_NUMPY:
            import numpy as _np
            _dm = _np.array(dist_mat) if not isinstance(dist_mat, _np.ndarray) else dist_mat
            _dm = (_dm + _dm.T) / 2.0  # force symmetry for scipy
            _np.fill_diagonal(_dm, 0.0)
            condensed = scipy_squareform(_dm)
            link = scipy_linkage(condensed, method="single")
            link_tuples = [(int(r[0]), int(r[1]), float(r[2]), int(r[3])) for r in link]
        else:
            dist_list = dist_mat if not _HAS_NUMPY else dist_mat.tolist()
            link_tuples = _single_linkage(dist_list, n)

        # Step 3: Quasi-diagonalisation
        sort_ix = _get_quasi_diag(link_tuples, n)

        # Step 4: Recursive bisection
        raw_weights = _recursive_bisect(cov_mat, sort_ix)

        # Map back to symbols
        result: Dict[str, float] = {}
        total_w = sum(raw_weights.values())
        if total_w < 1e-15:
            result = {s: 1.0 / n for s in symbols}
        else:
            for idx, sym in enumerate(symbols):
                result[sym] = raw_weights.get(idx, 0.0) / total_w

        self._last_weights = result
        self._last_compute_ms = (time.monotonic() - t0) * 1000
        logger.debug("HRP weights computed in %.1f ms for %d assets", self._last_compute_ms, n)
        return result

    def get_clusters(
        self,
        returns: Dict[str, List[float]],
        n_clusters: int = 3,
    ) -> List[List[str]]:
        """
        Partition assets into *n_clusters* groups via dendrogram cutting.

        Parameters
        ----------
        returns : dict
            Symbol -> list of returns.
        n_clusters : int
            Number of clusters to produce (default 3).

        Returns
        -------
        list of list of str
            Each inner list is a cluster of symbol names.
        """
        symbols = sorted(returns.keys())
        n = len(symbols)
        if n <= n_clusters:
            return [[s] for s in symbols]

        # Compute correlation and cluster
        if _HAS_NUMPY:
            T = min(len(returns[s]) for s in symbols)
            mat = np.array([returns[s][-T:] for s in symbols])
            corr_mat = np.corrcoef(mat)
            if corr_mat.ndim == 0:
                corr_mat = np.array([[1.0]])
            dist_mat = np.sqrt(np.clip(0.5 * (1.0 - corr_mat), 0.0, 1.0))
        else:
            corr_mat, _ = _py_corr_cov(returns, symbols)
            dist_mat = _distance_matrix(corr_mat)

        if _HAS_SCIPY and _HAS_NUMPY:
            import numpy as _np
            _dm = _np.array(dist_mat) if not isinstance(dist_mat, _np.ndarray) else dist_mat
            _dm = (_dm + _dm.T) / 2.0
            _np.fill_diagonal(_dm, 0.0)
            condensed = scipy_squareform(_dm)
            link = scipy_linkage(condensed, method="single")
            link_tuples = [(int(r[0]), int(r[1]), float(r[2]), int(r[3])) for r in link]
        else:
            dist_list = dist_mat if not _HAS_NUMPY else dist_mat.tolist()
            link_tuples = _single_linkage(dist_list, n)

        # Cut dendrogram: use top (n - n_clusters) merges
        # Assign cluster labels by replaying merges
        labels: Dict[int, int] = {i: i for i in range(n)}
        cluster_map: Dict[int, set] = {i: {i} for i in range(n)}
        next_id = n

        # Replay all but last (n_clusters - 1) merges
        merges_to_replay = len(link_tuples) - (n_clusters - 1)
        for i in range(merges_to_replay):
            a, b, _, _ = link_tuples[i]
            a_int, b_int = int(a), int(b)
            merged = cluster_map.get(a_int, {a_int}) | cluster_map.get(b_int, {b_int})
            cluster_map[next_id] = merged
            # Propagate label
            label = min(merged)
            for idx in merged:
                if idx < n:
                    labels[idx] = label
            next_id += 1

        # Group by label
        groups: Dict[int, List[str]] = {}
        for idx in range(n):
            lbl = labels[idx]
            groups.setdefault(lbl, []).append(symbols[idx])

        return list(groups.values())

    def get_dendrogram_data(self, returns: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Return dendrogram data suitable for visualisation.

        Parameters
        ----------
        returns : dict
            Symbol -> list of returns.

        Returns
        -------
        dict
            Keys: ``symbols``, ``linkage`` (list of [a, b, dist, size]),
            ``sort_order`` (quasi-diagonal ordering).
        """
        symbols = sorted(returns.keys())
        n = len(symbols)
        if n < 2:
            return {"symbols": symbols, "linkage": [], "sort_order": list(range(n))}

        if _HAS_NUMPY:
            T = min(len(returns[s]) for s in symbols)
            mat = np.array([returns[s][-T:] for s in symbols])
            corr_mat = np.corrcoef(mat)
            if corr_mat.ndim == 0:
                corr_mat = np.array([[1.0]])
            dist_mat = np.sqrt(np.clip(0.5 * (1.0 - corr_mat), 0.0, 1.0))
        else:
            corr_mat, _ = _py_corr_cov(returns, symbols)
            dist_mat = _distance_matrix(corr_mat)

        if _HAS_SCIPY and _HAS_NUMPY:
            import numpy as _np
            _dm = _np.array(dist_mat) if not isinstance(dist_mat, _np.ndarray) else dist_mat
            _dm = (_dm + _dm.T) / 2.0
            _np.fill_diagonal(_dm, 0.0)
            condensed = scipy_squareform(_dm)
            link = scipy_linkage(condensed, method="single")
            link_tuples = [(int(r[0]), int(r[1]), float(r[2]), int(r[3])) for r in link]
        else:
            dist_list = dist_mat if not _HAS_NUMPY else dist_mat.tolist()
            link_tuples = _single_linkage(dist_list, n)

        sort_ix = _get_quasi_diag(link_tuples, n)

        return {
            "symbols": symbols,
            "linkage": [list(t) for t in link_tuples],
            "sort_order": sort_ix,
        }
