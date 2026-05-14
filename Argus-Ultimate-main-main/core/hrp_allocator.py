"""
core/hrp_allocator.py

Hierarchical Risk Parity (HRP) allocator using riskfolio-lib.
Falls back to equal-weight if riskfolio-lib is not installed.

Usage
-----
    from core.hrp_allocator import HRPAllocator
    import pandas as pd

    # returns: DataFrame of shape (T, N) — daily/hourly returns per asset
    alloc = HRPAllocator()
    weights = alloc.allocate(returns_df)  # {"BTC": 0.42, "ETH": 0.33, ...}
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _hrp_riskfolio(returns: pd.DataFrame) -> Dict[str, float]:
    import riskfolio as rp  # type: ignore
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit")
    w = port.optimization(model="HRP", codependence="pearson",
                          rm="MV", linkage="single", leaf_order=True)
    if w is None or w.empty:
        raise ValueError("riskfolio HRP returned empty weights")
    result = w["weights"].to_dict()
    return {k: float(v) for k, v in result.items()}


def _hrp_manual(returns: pd.DataFrame) -> Dict[str, float]:
    """
    Pure-numpy HRP fallback (de Prado 2016):
      1. Correlation → distance matrix
      2. Single-linkage hierarchical clustering
      3. Quasi-diagonalisation
      4. Recursive bisection
    """
    cov  = returns.cov().values
    corr = returns.corr().values
    n    = corr.shape[0]
    cols = list(returns.columns)

    # Distance matrix
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0, 1))

    # Single-linkage clustering via condensed distance
    from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore
    condensed = dist[np.triu_indices(n, k=1)]
    Z         = linkage(condensed, method="single")
    order     = leaves_list(Z)

    # Quasi-diagonal covariance
    cov_sorted = cov[np.ix_(order, order)]

    # Recursive bisection
    weights = np.ones(n)
    clusters: list[list[int]] = [list(range(n))]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            split      = len(cluster) // 2
            left_idx   = cluster[:split]
            right_idx  = cluster[split:]

            def _cluster_var(idxs: list[int]) -> float:
                sub_cov = cov_sorted[np.ix_(idxs, idxs)]
                inv_var = 1.0 / np.maximum(np.diag(sub_cov), 1e-12)
                w_      = inv_var / inv_var.sum()
                return float(w_ @ sub_cov @ w_)

            var_l = _cluster_var(left_idx)
            var_r = _cluster_var(right_idx)
            total = var_l + var_r
            alpha = 1.0 - var_l / total if total > 0 else 0.5

            weights[left_idx]  *= (1 - alpha)
            weights[right_idx] *= alpha

            if len(left_idx)  > 1: new_clusters.append(left_idx)
            if len(right_idx) > 1: new_clusters.append(right_idx)
        clusters = new_clusters

    # Re-order back to original column order
    final_w = np.zeros(n)
    for i, orig_idx in enumerate(order):
        final_w[orig_idx] = weights[i]

    total = final_w.sum()
    if total > 0:
        final_w /= total

    return {cols[i]: round(float(final_w[i]), 6) for i in range(n)}


class HRPAllocator:
    """
    Hierarchical Risk Parity allocator.

    Parameters
    ----------
    prefer_riskfolio : bool
        Try riskfolio-lib first; fall back to pure-numpy implementation.
    min_periods : int
        Minimum number of rows required in the returns DataFrame.
    """

    def __init__(self,
                 prefer_riskfolio: bool = True,
                 min_periods: int = 30) -> None:
        self.prefer_riskfolio = prefer_riskfolio
        self.min_periods      = min_periods

    def allocate(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Compute HRP weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Shape (T, N) — return series per asset column.
            Must have at least self.min_periods rows.

        Returns
        -------
        dict  {asset_name: weight}  — weights sum to ~1.0
        """
        returns = returns.dropna(axis=0, how="any")
        n_cols  = len(returns.columns)

        if n_cols == 0:
            logger.warning("HRPAllocator: empty returns DataFrame — returning {}")
            return {}

        if n_cols == 1:
            return {returns.columns[0]: 1.0}

        if len(returns) < self.min_periods:
            logger.warning(
                "HRPAllocator: only %d rows < min_periods %d — equal weight",
                len(returns), self.min_periods,
            )
            w = 1.0 / n_cols
            return {c: round(w, 6) for c in returns.columns}

        if self.prefer_riskfolio:
            try:
                return _hrp_riskfolio(returns)
            except Exception as exc:
                logger.warning("HRPAllocator: riskfolio failed (%s) — using manual HRP", exc)

        try:
            return _hrp_manual(returns)
        except Exception as exc:
            logger.error("HRPAllocator: manual HRP failed (%s) — equal weight", exc)
            w = 1.0 / n_cols
            return {c: round(w, 6) for c in returns.columns}
