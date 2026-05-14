"""Strategy comparison framework for Argus backtesting.

Compares multiple strategies on identical data with:
  - Standardized metrics (Sharpe, Sortino, Calmar, max DD, win rate)
  - Statistical significance tests (t-test, bootstrap)
  - Risk-adjusted rankings
  - Correlation analysis between strategy returns
  - Drawdown overlap analysis

Usage:
    comparator = StrategyComparison()
    comparator.add_strategy("momentum", momentum_result)
    comparator.add_strategy("mean_reversion", mrev_result)
    comparison = comparator.compare()
    print(comparison.rankings)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.backtest.metrics import BacktestMetrics, compute_metrics


@dataclass
class StrategyResult:
    """Container for a single strategy's backtest result."""
    name: str
    equity_curve: List[float]
    returns: List[float]
    trade_pnls: List[float]
    metrics: BacktestMetrics
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two strategies."""
    sharpe_diff: float         # Strategy A Sharpe - Strategy B Sharpe
    sortino_diff: float
    calmar_diff: float
    max_dd_diff: float         # Strategy A max DD - Strategy B max DD (negative = A worse)
    return_diff: float         # Total return difference
    win_rate_diff: float
    correlation: float         # Correlation of daily returns
    t_stat: float              # t-statistic for return difference
    p_value: float             # p-value from t-test
    bootstrap_alpha: float     # Bootstrap-estimated alpha (A - B)
    bootstrap_ci_low: float    # 95% CI lower bound
    bootstrap_ci_high: float   # 95% CI upper bound

    def to_dict(self) -> dict:
        return {
            "sharpe_diff": round(self.sharpe_diff, 4),
            "sortino_diff": round(self.sortino_diff, 4),
            "calmar_diff": round(self.calmar_diff, 4),
            "max_dd_diff": round(self.max_dd_diff, 4),
            "return_diff": round(self.return_diff, 4),
            "win_rate_diff": round(self.win_rate_diff, 4),
            "correlation": round(self.correlation, 4),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "bootstrap_alpha": round(self.bootstrap_alpha, 4),
            "bootstrap_ci": [round(self.bootstrap_ci_low, 4), round(self.bootstrap_ci_high, 4)],
        }


@dataclass
class RankingEntry:
    """Single strategy ranking entry."""
    rank: int
    strategy_name: str
    composite_score: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    total_return_pct: float
    win_rate: float
    n_trades: int

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "strategy": self.strategy_name,
            "composite_score": round(self.composite_score, 4),
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "calmar": round(self.calmar, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "win_rate": round(self.win_rate, 4),
            "n_trades": self.n_trades,
        }


@dataclass
class ComparisonResult:
    """Full comparison result."""
    rankings: List[RankingEntry]
    pairwise: Dict[str, ComparisonMetrics]
    correlation_matrix: Dict[str, Dict[str, float]]
    drawdown_overlap: Dict[str, Dict[str, float]]
    summary: Dict

    def to_dict(self) -> dict:
        return {
            "rankings": [r.to_dict() for r in self.rankings],
            "pairwise": {k: v.to_dict() for k, v in self.pairwise.items()},
            "correlation_matrix": self.correlation_matrix,
            "drawdown_overlap": self.drawdown_overlap,
            "summary": self.summary,
        }


def _ttest_independent(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Simple t-test for independent samples (two-tailed).

    Returns (t_stat, p_value). Uses Welch's t-test (unequal variance).
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1) if n_a > 1 else 0.0
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1) if n_b > 1 else 0.0

    se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a + var_b) > 0 else 1e-10
    t_stat = (mean_a - mean_b) / se

    # Approximate p-value using normal distribution for large samples
    # For small samples, use t-distribution approximation
    df = n_a + n_b - 2
    # Simple approximation: p ≈ 2 * (1 - Φ(|t|))
    z = abs(t_stat)
    # Abramowitz and Stegun approximation for normal CDF
    p_value = 2.0 * (1.0 - _normal_cdf(z))

    return t_stat, p_value


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using Abramowitz and Stegun formula."""
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0

    # Rational approximation
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    if x < 0:
        t = 1.0 / (1.0 + p * abs(x))
    else:
        t = 1.0 / (1.0 + p * x)

    pdf = math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
    cdf = 1.0 - pdf * (b1 * t + b2 * t * t + b3 * t ** 3 + b4 * t ** 4 + b5 * t ** 5)

    return cdf if x >= 0 else 1.0 - cdf


def _bootstrap_alpha(
    returns_a: Sequence[float],
    returns_b: Sequence[float],
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap estimate of alpha (A - B) with 95% CI.

    Returns (alpha, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    n = min(len(returns_a), len(returns_b))
    if n < 10:
        return 0.0, 0.0, 0.0

    a = np.array(returns_a[:n])
    b = np.array(returns_b[:n])
    alphas = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        alpha = np.mean(a[idx]) - np.mean(b[idx])
        alphas.append(float(alpha))

    alphas.sort()
    alpha = float(np.mean(alphas))
    ci_low = alphas[int(0.025 * n_bootstrap)]
    ci_high = alphas[int(0.975 * n_bootstrap)]

    return alpha, ci_low, ci_high


def _compute_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Pearson correlation between two series."""
    n = min(len(a), len(b))
    if n < 3:
        return 0.0

    a_arr = np.array(a[:n])
    b_arr = np.array(b[:n])

    mean_a = np.mean(a_arr)
    mean_b = np.mean(b_arr)
    std_a = np.std(a_arr)
    std_b = np.std(b_arr)

    if std_a < 1e-10 or std_b < 1e-10:
        return 0.0

    corr = np.mean((a_arr - mean_a) * (b_arr - mean_b)) / (std_a * std_b)
    return float(corr)


def _compute_drawdown_series(equity: Sequence[float]) -> List[float]:
    """Compute drawdown percentage series."""
    peak = equity[0] if equity else 0.0
    dd = []
    for eq in equity:
        if eq > peak:
            peak = eq
        dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0.0
        dd.append(dd_pct)
    return dd


def _drawdown_overlap(dd_a: Sequence[float], dd_b: Sequence[float]) -> float:
    """Compute fraction of time both strategies are in drawdown simultaneously."""
    n = min(len(dd_a), len(dd_b))
    if n == 0:
        return 0.0

    overlap = sum(1 for i in range(n) if dd_a[i] > 0 and dd_b[i] > 0)
    return overlap / n


class StrategyComparison:
    """Compare multiple strategies on standardized metrics.

    Usage:
        comparator = StrategyComparison()
        comparator.add_result("momentum", equity_curve, returns, trade_pnls)
        comparator.add_result("mean_reversion", equity_curve2, returns2, trade_pnls2)
        result = comparator.compare()
    """

    def __init__(self):
        self._strategies: Dict[str, StrategyResult] = {}

    def add_result(
        self,
        name: str,
        equity_curve: Sequence[float],
        returns: Optional[Sequence[float]] = None,
        trade_pnls: Optional[Sequence[float]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add a strategy result for comparison."""
        equity = list(equity_curve)
        if returns is None:
            returns = [0.0] + [
                (equity[i] - equity[i - 1]) / equity[i - 1]
                if equity[i - 1] != 0 else 0.0
                for i in range(1, len(equity))
            ]
        else:
            returns = list(returns)

        pnls = list(trade_pnls) if trade_pnls else []
        metrics = compute_metrics(equity, pnls if pnls else None)

        self._strategies[name] = StrategyResult(
            name=name,
            equity_curve=equity,
            returns=returns,
            trade_pnls=pnls,
            metrics=metrics,
            metadata=metadata or {},
        )

    def add_strategy_result(self, result: StrategyResult) -> None:
        """Add a pre-built StrategyResult."""
        self._strategies[result.name] = result

    def compare(
        self,
        weights: Optional[Dict[str, float]] = None,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
    ) -> ComparisonResult:
        """Run full comparison.

        Args:
            weights: Optional weights for composite score (default: equal weights)
            n_bootstrap: Number of bootstrap iterations for CI
            seed: Random seed for bootstrap

        Returns:
            ComparisonResult with rankings, pairwise comparisons, correlations
        """
        if not self._strategies:
            raise ValueError("No strategies added for comparison")

        # Compute composite scores and rankings
        rankings = self._compute_rankings(weights)

        # Pairwise comparisons
        pairwise: Dict[str, ComparisonMetrics] = {}
        names = list(self._strategies.keys())
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                key = f"{name_a}_vs_{name_b}"
                pairwise[key] = self._pairwise_compare(
                    name_a, name_b, n_bootstrap, seed
                )

        # Correlation matrix
        corr_matrix = self._correlation_matrix()

        # Drawdown overlap
        dd_overlap = self._drawdown_overlap_matrix()

        # Summary
        summary = self._compute_summary(rankings)

        return ComparisonResult(
            rankings=rankings,
            pairwise=pairwise,
            correlation_matrix=corr_matrix,
            drawdown_overlap=dd_overlap,
            summary=summary,
        )

    def _compute_rankings(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[RankingEntry]:
        """Compute composite scores and rank strategies."""
        if weights is None:
            weights = {
                "sharpe": 0.30,
                "sortino": 0.20,
                "calmar": 0.20,
                "max_dd": 0.15,
                "return": 0.10,
                "win_rate": 0.05,
            }

        scores: List[Tuple[str, float]] = []
        for name, result in self._strategies.items():
            m = result.metrics
            # Normalize each metric to [0, 1] range (higher is better)
            sharpe_norm = max(0, min(m.sharpe / 3.0, 1.0))  # Sharpe 0-3 → 0-1
            sortino_norm = max(0, min(m.sortino / 4.0, 1.0))
            calmar_norm = max(0, min(m.calmar / 5.0, 1.0))
            dd_norm = max(0, 1.0 - m.max_drawdown_pct / 50.0)  # 0% DD = 1.0, 50% DD = 0
            ret_norm = max(0, min(m.total_return_pct / 100.0, 1.0))
            wr_norm = m.win_rate  # Already 0-1

            composite = (
                weights.get("sharpe", 0) * sharpe_norm +
                weights.get("sortino", 0) * sortino_norm +
                weights.get("calmar", 0) * calmar_norm +
                weights.get("max_dd", 0) * dd_norm +
                weights.get("return", 0) * ret_norm +
                weights.get("win_rate", 0) * wr_norm
            )
            scores.append((name, composite))

        # Sort by composite score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        rankings = []
        for rank, (name, score) in enumerate(scores, 1):
            m = self._strategies[name].metrics
            rankings.append(RankingEntry(
                rank=rank,
                strategy_name=name,
                composite_score=score,
                sharpe=m.sharpe,
                sortino=m.sortino,
                calmar=m.calmar,
                max_drawdown_pct=m.max_drawdown_pct,
                total_return_pct=m.total_return_pct,
                win_rate=m.win_rate,
                n_trades=m.n_trades,
            ))

        return rankings

    def _pairwise_compare(
        self,
        name_a: str,
        name_b: str,
        n_bootstrap: int,
        seed: Optional[int],
    ) -> ComparisonMetrics:
        """Compare two strategies pairwise."""
        a = self._strategies[name_a]
        b = self._strategies[name_b]

        ma, mb = a.metrics, b.metrics

        # Correlation
        corr = _compute_correlation(a.returns, b.returns)

        # T-test
        t_stat, p_value = _ttest_independent(a.returns, b.returns)

        # Bootstrap alpha
        alpha, ci_low, ci_high = _bootstrap_alpha(
            a.returns, b.returns, n_bootstrap, seed
        )

        return ComparisonMetrics(
            sharpe_diff=ma.sharpe - mb.sharpe,
            sortino_diff=ma.sortino - mb.sortino,
            calmar_diff=ma.calmar - mb.calmar,
            max_dd_diff=ma.max_drawdown_pct - mb.max_drawdown_pct,
            return_diff=ma.total_return_pct - mb.total_return_pct,
            win_rate_diff=ma.win_rate - mb.win_rate,
            correlation=corr,
            t_stat=t_stat,
            p_value=p_value,
            bootstrap_alpha=alpha,
            bootstrap_ci_low=ci_low,
            bootstrap_ci_high=ci_high,
        )

    def _correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute return correlation matrix."""
        names = list(self._strategies.keys())
        matrix: Dict[str, Dict[str, float]] = {}

        for name_a in names:
            matrix[name_a] = {}
            for name_b in names:
                if name_a == name_b:
                    matrix[name_a][name_b] = 1.0
                else:
                    corr = _compute_correlation(
                        self._strategies[name_a].returns,
                        self._strategies[name_b].returns,
                    )
                    matrix[name_a][name_b] = round(corr, 4)

        return matrix

    def _drawdown_overlap_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute drawdown overlap matrix."""
        names = list(self._strategies.keys())
        matrix: Dict[str, Dict[str, float]] = {}

        for name_a in names:
            matrix[name_a] = {}
            dd_a = _compute_drawdown_series(self._strategies[name_a].equity_curve)
            for name_b in names:
                if name_a == name_b:
                    matrix[name_a][name_b] = 1.0
                else:
                    dd_b = _compute_drawdown_series(self._strategies[name_b].equity_curve)
                    overlap = _drawdown_overlap(dd_a, dd_b)
                    matrix[name_a][name_b] = round(overlap, 4)

        return matrix

    def _compute_summary(self, rankings: List[RankingEntry]) -> Dict:
        """Compute summary statistics."""
        if not rankings:
            return {}

        best = rankings[0]
        worst = rankings[-1]

        sharpes = [r.sharpe for r in rankings]
        drawdowns = [r.max_drawdown_pct for r in rankings]

        return {
            "n_strategies": len(rankings),
            "best_strategy": best.strategy_name,
            "best_sharpe": round(best.sharpe, 4),
            "worst_strategy": worst.strategy_name,
            "worst_sharpe": round(worst.sharpe, 4),
            "avg_sharpe": round(sum(sharpes) / len(sharpes), 4),
            "avg_max_dd": round(sum(drawdowns) / len(drawdowns), 4),
            "recommended": best.strategy_name if best.sharpe > 0.5 else "none",
        }
