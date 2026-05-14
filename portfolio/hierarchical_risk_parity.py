from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-10


class LinkageMethod(str, Enum):
    WARD = "ward"
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"


class OnlineAlgorithm(str, Enum):
    OLMAR = "olmar"
    PAMR = "pamr"
    CRP = "crp"


@dataclass(slots=True)
class PortfolioMetrics:
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    information_ratio: float
    annualized_return: float
    annualized_volatility: float
    total_return: float
    num_rebalances: int = 0


@dataclass(slots=True)
class HrpResult:
    weights: Dict[str, float]
    cluster_tree: List[List[int]]
    ordered_symbols: List[str]
    portfolio_variance: float
    portfolio_volatility: float


@dataclass(slots=True)
class RebalanceSignal:
    symbol: str
    current_weight: float
    target_weight: float
    delta_weight: float
    reason: str


def _compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns from price series (T x N)."""
    if prices.shape[0] < 2:
        return np.zeros((0, prices.shape[1]), dtype=float)
    return (prices[1:] / prices[:-1]) - 1.0


def _compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from price series (T x N)."""
    if prices.shape[0] < 2:
        return np.zeros((0, prices.shape[1]), dtype=float)
    return np.log(prices[1:] / prices[:-1])


def _sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns.mean(axis=0) - risk_free_rate / periods_per_year
    vol = returns.std(axis=0, ddof=1)
    vol = np.where(vol < _EPS, _EPS, vol)
    return float(np.sum(excess / vol) / np.sqrt(periods_per_year))


def _sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns.mean(axis=0) - risk_free_rate / periods_per_year
    downside = returns[returns < 0]
    if downside.shape[0] == 0:
        return float(np.sum(excess) * np.sqrt(periods_per_year))
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    downside_std = max(downside_std, _EPS)
    return float(np.sum(excess) / downside_std * np.sqrt(periods_per_year))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(np.min(drawdowns))


def _calmar_ratio(annualized_return: float, max_dd: float) -> float:
    if abs(max_dd) < _EPS:
        return 0.0
    return annualized_return / abs(max_dd)


def _information_ratio(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray,
                       periods_per_year: int = 252) -> float:
    active = portfolio_returns - benchmark_returns
    active_mean = active.mean()
    active_std = active.std(ddof=1)
    if active_std < _EPS:
        return 0.0
    return float(active_mean / active_std * np.sqrt(periods_per_year))


class HierarchicalClustering:
    """Build hierarchical clusters from asset return correlation structure."""

    def __init__(self, linkage_method: LinkageMethod = LinkageMethod.WARD) -> None:
        self.linkage_method = linkage_method
        self._linkage_matrix: Optional[np.ndarray] = None
        self._distance_matrix: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def compute_distance_matrix(self, returns: np.ndarray) -> np.ndarray:
        corr = np.corrcoef(returns.T)
        corr = np.clip(corr, -1.0, 1.0)
        dist = 0.5 * (1.0 - corr)
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        self._distance_matrix = dist
        return dist

    def cluster(self, returns: np.ndarray) -> np.ndarray:
        from scipy.cluster.hierarchy import linkage

        dist = self._distance_matrix
        if dist is None:
            dist = self.compute_distance_matrix(returns)

        n = dist.shape[0]
        condensed = self._to_condensed(dist)
        self._linkage_matrix = linkage(condensed, method=self.linkage_method.value)
        return self._linkage_matrix

    def get_clusters(self, returns: np.ndarray, max_clusters: Optional[int] = None) -> List[List[int]]:
        from scipy.cluster.hierarchy import fcluster

        if self._linkage_matrix is None:
            self.cluster(returns)

        n = self._linkage_matrix.shape[0] + 1
        if max_clusters is None:
            max_clusters = max(2, n // 3)

        labels = fcluster(self._linkage_matrix, t=max_clusters, criterion="maxclust")
        self._labels = labels

        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(idx)

        return [clusters[k] for k in sorted(clusters.keys())]

    def get_ordered_indices(self, returns: np.ndarray) -> List[int]:
        if self._linkage_matrix is None:
            self.cluster(returns)
        return self._seriation(self._linkage_matrix)

    @staticmethod
    def _to_condensed(dist: np.ndarray) -> np.ndarray:
        n = dist.shape[0]
        condensed = np.empty(n * (n - 1) // 2, dtype=float)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                condensed[idx] = dist[i, j]
                idx += 1
        return condensed

    @staticmethod
    def _seriation(linkage_matrix: np.ndarray) -> List[int]:
        n = linkage_matrix.shape[0] + 1
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]

        left = int(linkage_matrix[-1, 0])
        right = int(linkage_matrix[-1, 1])

        left_indices = HierarchicalClustering._get_subtree_indices(linkage_matrix, left)
        right_indices = HierarchicalClustering._get_subtree_indices(linkage_matrix, right)

        return left_indices + right_indices

    @staticmethod
    def _get_subtree_indices(linkage_matrix: np.ndarray, node_id: int) -> List[int]:
        n = linkage_matrix.shape[0] + 1
        if node_id < n:
            return [node_id]

        row_idx = node_id - n
        left = int(linkage_matrix[row_idx, 0])
        right = int(linkage_matrix[row_idx, 1])

        left_indices = HierarchicalClustering._get_subtree_indices(linkage_matrix, left)
        right_indices = HierarchicalClustering._get_subtree_indices(linkage_matrix, right)

        return left_indices + right_indices


class HRPWeighter:
    """Compute HRP weights via recursive bisection."""

    def __init__(self, returns: np.ndarray, linkage_matrix: np.ndarray,
                 ordered_indices: Optional[List[int]] = None) -> None:
        self.returns = returns
        self.linkage_matrix = linkage_matrix
        self.ordered_indices = ordered_indices
        self._cov_matrix: Optional[np.ndarray] = None
        self._variances: Optional[np.ndarray] = None

    def compute_weights(self) -> np.ndarray:
        n = self.returns.shape[1]
        self._cov_matrix = np.cov(self.returns.T)
        self._cov_matrix = (self._cov_matrix + self._cov_matrix.T) / 2.0
        self._variances = np.diag(self._cov_matrix).copy()

        if self.ordered_indices is None:
            ordered = HierarchicalClustering._seriation(self.linkage_matrix)
        else:
            ordered = list(self.ordered_indices)

        weights = self.recursive_bisection(ordered)
        weights = np.maximum(weights, 0.0)
        w_sum = weights.sum()
        if w_sum > _EPS:
            weights /= w_sum
        else:
            weights = np.ones(n, dtype=float) / n

        return weights

    def recursive_bisection(self, cluster_indices: List[int]) -> np.ndarray:
        n_total = self.returns.shape[1]
        weights = np.zeros(n_total, dtype=float)

        self._bisect_recursive(cluster_indices, weights)

        return weights

    def _bisect_recursive(self, indices: List[int], weights: np.ndarray) -> None:
        if len(indices) <= 1:
            if len(indices) == 1:
                weights[indices[0]] = 1.0
            return

        mid = len(indices) // 2
        left = indices[:mid]
        right = indices[mid:]

        var_left = self._cluster_variance(left)
        var_right = self._cluster_variance(right)

        alpha_left = 1.0 - var_left / (var_left + var_right) if (var_left + var_right) > _EPS else 0.5
        alpha_left = max(0.0, min(1.0, alpha_left))

        left_weights = np.zeros(len(left), dtype=float)
        right_weights = np.zeros(len(right), dtype=float)

        self._bisect_recursive(left, left_weights)
        self._bisect_recursive(right, right_weights)

        for i, idx in enumerate(left):
            weights[idx] = alpha_left * left_weights[i]
        for i, idx in enumerate(right):
            weights[idx] = (1.0 - alpha_left) * right_weights[i]

    def _cluster_variance(self, indices: List[int]) -> float:
        if self._cov_matrix is None or self._variances is None:
            return 1.0

        iv_weights = self.inverse_variance_weights(indices)
        cov_sub = self._cov_matrix[np.ix_(indices, indices)]
        var = float(iv_weights @ cov_sub @ iv_weights)
        return max(var, _EPS)

    def inverse_variance_weights(self, indices: List[int]) -> np.ndarray:
        if self._variances is None:
            return np.ones(len(indices), dtype=float) / max(len(indices), 1)

        sub_vars = self._variances[np.array(indices)]
        inv_vars = 1.0 / np.maximum(sub_vars, _EPS)
        w = inv_vars / inv_vars.sum()
        return w


class OnlinePortfolioSelector:
    """Online portfolio selection with OLMAR, PAMR, and CRP algorithms."""

    def __init__(
        self,
        algorithm: OnlineAlgorithm = OnlineAlgorithm.CRP,
        n_assets: int = 1,
        epsilon: float = 1.0,
        learning_rate: float = 0.1,
        aggressiveness: float = 500.0,
    ) -> None:
        self.algorithm = algorithm
        self.n_assets = n_assets
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.aggressiveness = aggressiveness
        self.weights = np.ones(n_assets, dtype=float) / n_assets
        self._t = 0

    def update(self, price_relative: np.ndarray) -> np.ndarray:
        """Update weights given price relative vector (1 + returns)."""
        self._t += 1

        if self.algorithm == OnlineAlgorithm.OLMAR:
            self._olmar_step(price_relative)
        elif self.algorithm == OnlineAlgorithm.PAMR:
            self._pamr_step(price_relative)
        elif self.algorithm == OnlineAlgorithm.CRP:
            pass  # CRP keeps uniform weights
        else:
            self._crp_step()

        self.weights = np.maximum(self.weights, 0.0)
        w_sum = self.weights.sum()
        if w_sum > _EPS:
            self.weights /= w_sum
        else:
            self.weights = np.ones(self.n_assets, dtype=float) / self.n_assets

        return self.weights.copy()

    def _olmar_step(self, price_relative: np.ndarray) -> None:
        mean_rel = self.epsilon
        m_t = price_relative @ self.weights
        if m_t < mean_rel:
            x_bar = np.mean(price_relative)
            diff = price_relative - x_bar
            norm_sq = np.dot(diff, diff)
            if norm_sq > _EPS:
                adjustment = max(0.0, (mean_rel - m_t)) / norm_sq * diff
                self.weights = self.weights + self.learning_rate * adjustment

    def _pamr_step(self, price_relative: np.ndarray) -> None:
        m_t = price_relative @ self.weights
        loss = max(0.0, m_t - self.epsilon)

        if loss > _EPS:
            x_bar = np.mean(price_relative)
            diff = price_relative - x_bar
            norm_sq = np.dot(diff, diff)

            if self.algorithm == OnlineAlgorithm.PAMR:
                tau = loss / norm_sq if norm_sq > _EPS else 0.0
            else:
                tau = min(self.aggressiveness, loss / norm_sq) if norm_sq > _EPS else 0.0

            self.weights = self.weights - tau * diff

    def _crp_step(self) -> None:
        self.weights = np.ones(self.n_assets, dtype=float) / self.n_assets

    def run_batch(self, price_relatives: np.ndarray) -> List[np.ndarray]:
        """Run online selection over a sequence of price relatives."""
        trajectory: List[np.ndarray] = []
        for rel in price_relatives:
            w = self.update(rel)
            trajectory.append(w.copy())
        return trajectory


class HrpPortfolioOptimizer:
    """Full HRP portfolio optimization with rebalancing and backtesting."""

    def __init__(
        self,
        symbols: List[str],
        linkage_method: LinkageMethod = LinkageMethod.WARD,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        rebalance_threshold: float = 0.05,
        transaction_cost: float = 0.001,
    ) -> None:
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.rebalance_threshold = rebalance_threshold
        self.transaction_cost = transaction_cost

        self._clustering = HierarchicalClustering(linkage_method)
        self._current_weights: Optional[np.ndarray] = None
        self._target_weights: Optional[np.ndarray] = None
        self._linkage_matrix: Optional[np.ndarray] = None
        self._ordered_indices: Optional[List[int]] = None
        self._cov_matrix: Optional[np.ndarray] = None
        self._rebalance_count = 0

    def optimize(self, prices: np.ndarray) -> HrpResult:
        returns = _compute_returns(prices)
        if returns.shape[0] < 2:
            logger.warning("Insufficient data for HRP optimization, using equal weights")
            eq = np.ones(self.n_assets, dtype=float) / self.n_assets
            return HrpResult(
                weights=dict(zip(self.symbols, eq.tolist())),
                cluster_tree=[],
                ordered_symbols=self.symbols.copy(),
                portfolio_variance=0.0,
                portfolio_volatility=0.0,
            )

        self._clustering.compute_distance_matrix(returns)
        self._linkage_matrix = self._clustering.cluster(returns)
        self._ordered_indices = self._clustering.get_ordered_indices(returns)

        weighter = HRPWeighter(returns, self._linkage_matrix, self._ordered_indices)
        weights = weighter.compute_weights()

        self._cov_matrix = np.cov(returns.T)
        self._cov_matrix = (self._cov_matrix + self._cov_matrix.T) / 2.0
        portfolio_var = float(weights @ self._cov_matrix @ weights)
        portfolio_vol = np.sqrt(max(portfolio_var, _EPS))

        self._target_weights = weights.copy()

        ordered_symbols = [self.symbols[i] for i in self._ordered_indices]
        clusters = self._clustering.get_clusters(returns)
        cluster_tree = clusters

        result = HrpResult(
            weights=dict(zip(self.symbols, weights.tolist())),
            cluster_tree=cluster_tree,
            ordered_symbols=ordered_symbols,
            portfolio_variance=portfolio_var,
            portfolio_volatility=float(portfolio_vol),
        )

        logger.info(
            "HRP optimization complete: %d assets, vol=%.4f, clusters=%d",
            self.n_assets, portfolio_vol, len(clusters),
        )
        return result

    def rebalance(
        self,
        prices: np.ndarray,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> List[RebalanceSignal]:
        result = self.optimize(prices)
        target = result.weights

        if current_weights is None:
            current = {s: 0.0 for s in self.symbols}
        else:
            current = current_weights.copy()

        signals: List[RebalanceSignal] = []
        for symbol in self.symbols:
            cw = current.get(symbol, 0.0)
            tw = target.get(symbol, 0.0)
            delta = tw - cw

            if abs(delta) >= self.rebalance_threshold:
                signals.append(RebalanceSignal(
                    symbol=symbol,
                    current_weight=cw,
                    target_weight=tw,
                    delta_weight=delta,
                    reason="hrp_rebalance",
                ))

        if signals:
            self._rebalance_count += 1
            self._current_weights = np.array([target.get(s, 0.0) for s in self.symbols])
            logger.info("HRP rebalance: %d signals generated", len(signals))

        return signals

    def backtest(
        self,
        prices: np.ndarray,
        rebalance_frequency: int = 20,
        initial_capital: float = 100000.0,
        online_algorithm: Optional[OnlineAlgorithm] = None,
    ) -> PortfolioMetrics:
        returns = _compute_returns(prices)
        n = returns.shape[0]

        if n < 2:
            return PortfolioMetrics(
                sharpe=0.0, sortino=0.0, max_drawdown=0.0,
                calmar=0.0, information_ratio=0.0,
                annualized_return=0.0, annualized_volatility=0.0,
                total_return=0.0, num_rebalances=0,
            )

        weights = np.ones(self.n_assets, dtype=float) / self.n_assets
        portfolio_returns: List[float] = []
        equity_curve: List[float] = [initial_capital]
        capital = initial_capital
        num_rebalances = 0

        online_selector: Optional[OnlinePortfolioSelector] = None
        if online_algorithm is not None:
            online_selector = OnlinePortfolioSelector(
                algorithm=online_algorithm,
                n_assets=self.n_assets,
            )

        for t in range(n):
            if t % rebalance_frequency == 0:
                window_end = min(t + rebalance_frequency, n)
                window_prices = prices[:window_end] if window_end > 0 else prices[:1]

                if window_prices.shape[0] >= 2:
                    try:
                        hrp_result = self.optimize(window_prices)
                        weights = np.array([hrp_result.weights.get(s, 0.0) for s in self.symbols])
                        num_rebalances += 1
                    except Exception:
                        logger.warning("HRP optimization failed at t=%d, using current weights", t)

            if online_selector is not None and t > 0:
                rel = 1.0 + returns[t - 1]
                weights = online_selector.update(rel)

            port_ret = float(weights @ returns[t])
            cost = self.transaction_cost * np.sum(np.abs(weights - (weights if t == 0 else weights)))
            net_ret = port_ret - cost
            portfolio_returns.append(net_ret)
            capital *= (1.0 + net_ret)
            equity_curve.append(capital)

        equity_arr = np.array(equity_curve)
        ret_arr = np.array(portfolio_returns)

        total_ret = (capital - initial_capital) / initial_capital
        n_periods = len(portfolio_returns)
        ann_ret = (1.0 + total_ret) ** (self.periods_per_year / max(n_periods, 1)) - 1.0
        ann_vol = float(ret_arr.std(ddof=1) * np.sqrt(self.periods_per_year)) if n_periods > 1 else 0.0

        sharpe = _sharpe_ratio(ret_arr.reshape(-1, 1), self.risk_free_rate, self.periods_per_year)
        sortino = _sortino_ratio(ret_arr.reshape(-1, 1), self.risk_free_rate, self.periods_per_year)
        max_dd = _max_drawdown(equity_arr)
        calmar = _calmar_ratio(ann_ret, max_dd)

        benchmark_returns = returns.mean(axis=1)
        ir = _information_ratio(ret_arr, benchmark_returns, self.periods_per_year)

        metrics = PortfolioMetrics(
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            calmar=calmar,
            information_ratio=ir,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            total_return=total_ret,
            num_rebalances=num_rebalances,
        )

        logger.info(
            "HRP backtest complete: sharpe=%.3f, sortino=%.3f, max_dd=%.3f, calmar=%.3f, IR=%.3f",
            sharpe, sortino, max_dd, calmar, ir,
        )
        return metrics

    def get_risk_contribution(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if self._cov_matrix is None:
            logger.warning("Covariance matrix not computed, run optimize() first")
            return {s: 1.0 / self.n_assets for s in self.symbols}

        if weights is None:
            w = self._target_weights
            if w is None:
                w = np.ones(self.n_assets, dtype=float) / self.n_assets
        else:
            w = np.array([weights.get(s, 0.0) for s in self.symbols])

        cov = self._cov_matrix
        marginal_risk = cov @ w
        risk_contrib = w * marginal_risk
        total_risk = risk_contrib.sum()

        if abs(total_risk) < _EPS:
            return {s: 1.0 / self.n_assets for s in self.symbols}

        risk_pct = risk_contrib / total_risk
        return dict(zip(self.symbols, risk_pct.tolist()))

    def get_cluster_structure(self) -> List[List[str]]:
        if self._clustering._labels is None:
            return [[s] for s in self.symbols]

        clusters: Dict[int, List[str]] = {}
        for idx, label in enumerate(self._clustering._labels):
            clusters.setdefault(int(label), []).append(self.symbols[idx])

        return [clusters[k] for k in sorted(clusters.keys())]

    def reset(self) -> None:
        self._current_weights = None
        self._target_weights = None
        self._linkage_matrix = None
        self._ordered_indices = None
        self._cov_matrix = None
        self._rebalance_count = 0
        self._clustering = HierarchicalClustering(self._clustering.linkage_method)
