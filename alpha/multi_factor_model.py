"""Multi-factor alpha model for signal generation.

Implements institutional-grade factor investing with momentum, value, quality,
volatility, and sentiment factors. Supports factor neutralization, signal
generation, and backtesting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MomentumType(Enum):
    """Momentum calculation type."""
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"


class FactorCategory(Enum):
    """Factor category for classification."""
    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"


@dataclass
class MomentumFactor:
    """Momentum factor configuration and scores."""
    name: str = "momentum"
    category: FactorCategory = FactorCategory.MOMENTUM
    momentum_type: MomentumType = MomentumType.CROSS_SECTIONAL
    lookbacks: list[int] = field(default_factory=lambda: [5, 20, 60, 120])
    weights: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    scores: np.ndarray | None = None
    description: str = "Price momentum across multiple time horizons"


@dataclass
class ValueFactor:
    """Value factor configuration and scores."""
    name: str = "value"
    category: FactorCategory = FactorCategory.VALUE
    metrics: list[str] = field(default_factory=lambda: ["pe_ratio", "pb_ratio", "ps_ratio"])
    weights: list[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])
    scores: np.ndarray | None = None
    description: str = "Relative valuation metrics"


@dataclass
class QualityFactor:
    """Quality factor configuration and scores."""
    name: str = "quality"
    category: FactorCategory = FactorCategory.QUALITY
    metrics: list[str] = field(default_factory=lambda: ["roe", "debt_to_equity", "earnings_stability"])
    weights: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    scores: np.ndarray | None = None
    description: str = "Earnings stability and balance sheet strength"


@dataclass
class VolatilityFactor:
    """Volatility factor configuration and scores."""
    name: str = "volatility"
    category: FactorCategory = FactorCategory.VOLATILITY
    lookback: int = 60
    annualization_factor: int = 252
    low_vol_preference: bool = True
    scores: np.ndarray | None = None
    description: str = "Low volatility anomaly factor"


@dataclass
class SentimentFactor:
    """Sentiment factor configuration and scores."""
    name: str = "sentiment"
    category: FactorCategory = FactorCategory.SENTIMENT
    sources: list[str] = field(default_factory=lambda: ["news", "social", "analyst"])
    weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    scores: np.ndarray | None = None
    description: str = "News and social media sentiment"


@dataclass
class FactorUniverse:
    """Container for asset universe data."""
    assets: list[str]
    prices: np.ndarray
    returns: np.ndarray | None = None
    dates: list[str] | None = None
    sectors: dict[str, str] | None = None
    market_caps: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.returns is None and self.prices is not None:
            self.returns = self._compute_returns()

    def _compute_returns(self) -> np.ndarray:
        """Compute simple returns from prices."""
        returns = np.diff(self.prices, axis=0) / self.prices[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        return returns

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_dates(self) -> int:
        return self.prices.shape[0] if self.prices is not None else 0


@dataclass
class Signal:
    """Trading signal generated from alpha scores."""
    asset: str
    direction: str
    strength: float
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorBacktestResult:
    """Result of factor backtesting."""
    factor_name: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_cumulative: float
    long_short_return: float
    long_return: float
    short_return: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    hit_rate: float
    decay_rates: list[float] = field(default_factory=list)


@dataclass
class TurnoverMetrics:
    """Turnover analysis metrics."""
    gross_turnover: float
    net_turnover: float
    avg_turnover: float
    max_turnover: float
    min_turnover: float


class FactorCalculator:
    """Computes factor scores from market data."""

    def __init__(self, universe: FactorUniverse) -> None:
        self.universe = universe
        logger.info(
            "FactorCalculator initialized with %d assets, %d dates",
            universe.n_assets,
            universe.n_dates,
        )

    def compute_momentum(
        self,
        returns: np.ndarray | None = None,
        lookbacks: list[int] | None = None,
        weights: list[float] | None = None,
    ) -> np.ndarray:
        """Compute momentum factor scores.

        Args:
            returns: (n_dates, n_assets) array of returns.
            lookbacks: List of lookback periods.
            weights: Weights for each lookback period.

        Returns:
            (n_assets,) array of momentum scores.
        """
        if returns is None:
            returns = self.universe.returns
        if lookbacks is None:
            lookbacks = [5, 20, 60, 120]
        if weights is None:
            weights = [0.1, 0.2, 0.3, 0.4]

        n_assets = returns.shape[1]
        momentum_scores = np.zeros(n_assets)

        for lookback, weight in zip(lookbacks, weights):
            if lookback > returns.shape[0]:
                logger.warning(
                    "Lookback %d exceeds available data (%d), skipping",
                    lookback,
                    returns.shape[0],
                )
                continue

            cumulative_returns = np.prod(1 + returns[-lookback:], axis=0) - 1
            momentum_scores += weight * cumulative_returns

        momentum_scores = self._cross_sectional_rank(momentum_scores)
        logger.info("Momentum factor computed with lookbacks: %s", lookbacks)
        return momentum_scores

    def compute_value(
        self,
        prices: np.ndarray | None = None,
        fundamentals: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute value factor scores from fundamentals.

        Args:
            prices: (n_dates, n_assets) array of prices.
            fundamentals: Dict of metric name to (n_assets,) array.

        Returns:
            (n_assets,) array of value scores.
        """
        if fundamentals is None:
            logger.warning("No fundamentals provided, returning zero scores")
            return np.zeros(self.universe.n_assets)

        n_assets = self.universe.n_assets
        value_scores = np.zeros(n_assets)

        for metric, values in fundamentals.items():
            if len(values) != n_assets:
                logger.warning(
                    "Metric %s has %d values, expected %d, skipping",
                    metric,
                    len(values),
                    n_assets,
                )
                continue

            normalized = self._z_score(values)
            value_scores += normalized

        value_scores = self._cross_sectional_rank(value_scores)
        logger.info("Value factor computed with metrics: %s", list(fundamentals.keys()))
        return value_scores

    def compute_quality(
        self,
        returns: np.ndarray | None = None,
        fundamentals: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute quality factor scores.

        Args:
            returns: (n_dates, n_assets) array of returns.
            fundamentals: Dict of quality metrics to (n_assets,) arrays.

        Returns:
            (n_assets,) array of quality scores.
        """
        if returns is None:
            returns = self.universe.returns
        if fundamentals is None:
            fundamentals = {}

        n_assets = returns.shape[1]
        quality_scores = np.zeros(n_assets)

        if "roe" in fundamentals:
            quality_scores += self._z_score(fundamentals["roe"])

        if "debt_to_equity" in fundamentals:
            quality_scores -= self._z_score(fundamentals["debt_to_equity"])

        if "earnings_stability" in fundamentals:
            quality_scores += self._z_score(fundamentals["earnings_stability"])

        if returns.shape[0] > 20:
            earnings_stability = self._compute_earnings_stability(returns)
            quality_scores += self._z_score(earnings_stability)

        quality_scores = self._cross_sectional_rank(quality_scores)
        logger.info("Quality factor computed")
        return quality_scores

    def compute_volatility(
        self,
        returns: np.ndarray | None = None,
        lookback: int = 60,
    ) -> np.ndarray:
        """Compute volatility factor scores (low vol anomaly).

        Args:
            returns: (n_dates, n_assets) array of returns.
            lookback: Lookback period for volatility calculation.

        Returns:
            (n_assets,) array of volatility scores (higher = lower vol).
        """
        if returns is None:
            returns = self.universe.returns

        if lookback > returns.shape[0]:
            lookback = returns.shape[0]
            logger.warning("Adjusted lookback to %d", lookback)

        recent_returns = returns[-lookback:]
        volatilities = np.std(recent_returns, axis=0)

        volatilities = np.where(volatilities == 0, np.mean(volatilities), volatilities)

        vol_scores = -self._z_score(volatilities)
        vol_scores = self._cross_sectional_rank(vol_scores)

        logger.info("Volatility factor computed with lookback %d", lookback)
        return vol_scores

    def compute_sentiment(
        self,
        news_data: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Compute sentiment factor scores.

        Args:
            news_data: Dict of asset symbol to sentiment score.

        Returns:
            (n_assets,) array of sentiment scores.
        """
        n_assets = self.universe.n_assets
        sentiment_scores = np.zeros(n_assets)

        if news_data is None:
            logger.warning("No news data provided, returning zero scores")
            return sentiment_scores

        for i, asset in enumerate(self.universe.assets):
            if asset in news_data:
                sentiment_scores[i] = news_data[asset]

        sentiment_scores = self._cross_sectional_rank(sentiment_scores)
        logger.info("Sentiment factor computed for %d assets", len(news_data))
        return sentiment_scores

    @staticmethod
    def _cross_sectional_rank(values: np.ndarray) -> np.ndarray:
        """Convert values to cross-sectional percentile ranks."""
        valid_mask = ~np.isnan(values)
        ranks = np.zeros_like(values)
        if valid_mask.any():
            valid_values = values[valid_mask]
            ranks[valid_mask] = stats.rankdata(valid_values) / len(valid_values)
        return ranks

    @staticmethod
    def _z_score(values: np.ndarray) -> np.ndarray:
        """Compute z-scores."""
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)
        return (values - np.mean(values)) / std

    @staticmethod
    def _compute_earnings_stability(returns: np.ndarray) -> np.ndarray:
        """Compute earnings stability as inverse of return variance."""
        rolling_var = np.var(returns[-60:], axis=0)
        stability = 1.0 / (rolling_var + 1e-8)
        return stability


class FactorNeutralizer:
    """Neutralizes factor exposures to unwanted risk factors."""

    def __init__(self) -> None:
        logger.info("FactorNeutralizer initialized")

    def neutralize_by_sector(
        self,
        scores: np.ndarray,
        sectors: dict[str, str],
    ) -> np.ndarray:
        """Neutralize scores by sector.

        Args:
            scores: (n_assets,) raw factor scores.
            sectors: Dict of asset to sector.

        Returns:
            (n_assets,) sector-neutralized scores.
        """
        neutralized = np.zeros_like(scores)
        sector_groups: dict[str, list[int]] = {}

        for i, asset in enumerate(sectors.keys()):
            sector = sectors[asset]
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(i)

        for sector, indices in sector_groups.items():
            sector_scores = scores[indices]
            sector_mean = np.mean(sector_scores)
            neutralized[indices] = sector_scores - sector_mean

        logger.info("Scores neutralized by sector (%d sectors)", len(sector_groups))
        return neutralized

    def neutralize_by_mcap(
        self,
        scores: np.ndarray,
        mcaps: dict[str, float],
    ) -> np.ndarray:
        """Neutralize scores by market cap.

        Args:
            scores: (n_assets,) raw factor scores.
            mcaps: Dict of asset to market cap.

        Returns:
            (n_assets,) market-cap-neutralized scores.
        """
        assets = list(mcaps.keys())
        market_caps = np.array([mcaps[a] for a in assets])
        log_mcaps = np.log(market_caps + 1e-8)

        valid_mask = ~np.isnan(scores) & ~np.isnan(log_mcaps)
        if valid_mask.sum() < 3:
            logger.warning("Insufficient data for market cap neutralization")
            return scores

        slope, intercept, _, _, _ = stats.linregress(
            log_mcaps[valid_mask], scores[valid_mask]
        )

        neutralized = scores.copy()
        neutralized[valid_mask] -= slope * log_mcaps[valid_mask] + intercept - np.mean(scores[valid_mask])

        logger.info("Scores neutralized by market cap")
        return neutralized

    def residualize(
        self,
        factor: np.ndarray,
        other_factors: np.ndarray,
    ) -> np.ndarray:
        """Residualize factor against other factors.

        Args:
            factor: (n_assets,) factor scores to residualize.
            other_factors: (n_assets, n_other) matrix of other factors.

        Returns:
            (n_assets,) residualized factor scores.
        """
        if other_factors.ndim == 1:
            other_factors = other_factors.reshape(-1, 1)

        valid_mask = ~np.isnan(factor) & ~np.any(np.isnan(other_factors), axis=1)
        if valid_mask.sum() < other_factors.shape[1] + 2:
            logger.warning("Insufficient data for residualization")
            return factor

        X = other_factors[valid_mask]
        y = factor[valid_mask]

        X_aug = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            residuals = y - X_aug @ beta
        except np.linalg.LinAlgError:
            logger.warning("SVD did not converge in residualization")
            return factor

        residualized = np.zeros_like(factor)
        residualized[valid_mask] = residuals
        residualized[~valid_mask] = factor[~valid_mask]

        logger.info("Factor residualized against %d other factors", other_factors.shape[1])
        return residualized


class AlphaSignalGenerator:
    """Generates trading signals from combined factor scores."""

    def __init__(self) -> None:
        logger.info("AlphaSignalGenerator initialized")

    def combine_factors(
        self,
        factors: dict[str, np.ndarray],
        weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Combine multiple factor scores into a single alpha score.

        Args:
            factors: Dict of factor name to (n_assets,) scores.
            weights: Dict of factor name to weight.

        Returns:
            (n_assets,) combined alpha scores.
        """
        if not factors:
            logger.warning("No factors provided")
            return np.array([])

        if weights is None:
            weights = {name: 1.0 / len(factors) for name in factors}

        n_assets = next(iter(factors.values())).shape[0]
        combined = np.zeros(n_assets)
        total_weight = 0.0

        for name, scores in factors.items():
            w = weights.get(name, 0.0)
            if w == 0:
                continue
            combined += w * scores
            total_weight += w

        if total_weight > 0:
            combined /= total_weight

        combined = self._normalize_scores(combined)
        logger.info("Combined %d factors into alpha scores", len(factors))
        return combined

    def generate_signals(
        self,
        alpha_scores: np.ndarray,
        threshold: float = 0.5,
        assets: list[str] | None = None,
        timestamp: str | None = None,
    ) -> list[Signal]:
        """Generate trading signals from alpha scores.

        Args:
            alpha_scores: (n_assets,) combined alpha scores.
            threshold: Threshold for signal generation (0-1).
            assets: List of asset symbols.
            timestamp: Signal timestamp.

        Returns:
            List of Signal objects.
        """
        signals = []

        for i, score in enumerate(alpha_scores):
            asset = assets[i] if assets else f"asset_{i}"

            if score > threshold:
                direction = "long"
                strength = score
            elif score < (1 - threshold):
                direction = "short"
                strength = 1 - score
            else:
                continue

            signals.append(
                Signal(
                    asset=asset,
                    direction=direction,
                    strength=float(strength),
                    timestamp=timestamp,
                    metadata={"alpha_score": float(score)},
                )
            )

        logger.info(
            "Generated %d signals (%d long, %d short)",
            len(signals),
            sum(1 for s in signals if s.direction == "long"),
            sum(1 for s in signals if s.direction == "short"),
        )
        return signals

    def rank_assets(
        self,
        alpha_scores: np.ndarray,
        assets: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Rank assets by alpha scores.

        Args:
            alpha_scores: (n_assets,) combined alpha scores.
            assets: List of asset symbols.

        Returns:
            List of (asset, score) tuples sorted by score descending.
        """
        if assets is None:
            assets = [f"asset_{i}" for i in range(len(alpha_scores))]

        ranked = sorted(
            zip(assets, alpha_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.info("Ranked %d assets by alpha scores", len(ranked))
        return ranked

    def compute_ic(
        self,
        factor: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """Compute information coefficient (rank IC) between factor and returns.

        Args:
            factor: (n_assets,) factor scores.
            returns: (n_assets,) forward returns.

        Returns:
            Spearman rank correlation (IC).
        """
        valid_mask = ~np.isnan(factor) & ~np.isnan(returns)
        if valid_mask.sum() < 3:
            logger.warning("Insufficient data for IC computation")
            return 0.0

        ic, _ = stats.spearmanr(factor[valid_mask], returns[valid_mask])
        return float(ic) if not np.isnan(ic) else 0.0

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val == min_val:
            return np.full_like(scores, 0.5)
        return (scores - min_val) / (max_val - min_val)


class FactorBacktester:
    """Backtests factor models and computes performance metrics."""

    def __init__(self) -> None:
        logger.info("FactorBacktester initialized")

    def backtest_factor(
        self,
        factor: np.ndarray,
        returns: np.ndarray,
        factor_name: str = "factor",
        n_quantiles: int = 5,
    ) -> FactorBacktestResult:
        """Backtest a factor against returns.

        Args:
            factor: (n_assets,) factor scores.
            returns: (n_dates, n_assets) forward returns.
            factor_name: Name of the factor.
            n_quantiles: Number of quantiles for long-short analysis.

        Returns:
            FactorBacktestResult with performance metrics.
        """
        ic_series = []
        for t in range(returns.shape[0]):
            ic = self._compute_ic(factor, returns[t])
            ic_series.append(ic)

        ic_array = np.array(ic_series)
        ic_mean = float(np.mean(ic_array))
        ic_std = float(np.std(ic_array))
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_cumulative = float(np.sum(ic_array))

        long_ret, short_ret, ls_ret = self._compute_long_short_returns(
            factor, returns, n_quantiles
        )

        sharpe = self._compute_sharpe(ls_ret)
        max_dd = self._compute_max_drawdown(ls_ret)
        hit_rate = float(np.mean(ic_array > 0))

        result = FactorBacktestResult(
            factor_name=factor_name,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_cumulative=ic_cumulative,
            long_short_return=float(np.sum(ls_ret)),
            long_return=float(np.sum(long_ret)),
            short_return=float(np.sum(short_ret)),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            turnover=0.0,
            hit_rate=hit_rate,
        )

        logger.info(
            "Backtest complete for %s: IC=%.4f, IR=%.4f, Sharpe=%.4f",
            factor_name,
            ic_mean,
            ic_ir,
            sharpe,
        )
        return result

    def compute_turnover(
        self,
        weights: np.ndarray,
    ) -> TurnoverMetrics:
        """Compute turnover metrics from weight changes.

        Args:
            weights: (n_periods, n_assets) array of portfolio weights.

        Returns:
            TurnoverMetrics object.
        """
        if weights.shape[0] < 2:
            return TurnoverMetrics(
                gross_turnover=0.0,
                net_turnover=0.0,
                avg_turnover=0.0,
                max_turnover=0.0,
                min_turnover=0.0,
            )

        changes = np.abs(np.diff(weights, axis=0))
        gross_to = np.sum(changes, axis=1)
        net_to = np.sum(np.abs(np.sum(changes, axis=1)))

        return TurnoverMetrics(
            gross_turnover=float(np.mean(gross_to)),
            net_turnover=float(net_to),
            avg_turnover=float(np.mean(gross_to)),
            max_turnover=float(np.max(gross_to)),
            min_turnover=float(np.min(gross_to)),
        )

    def factor_decay_analysis(
        self,
        factor: np.ndarray,
        returns: np.ndarray,
        max_lag: int = 20,
    ) -> list[float]:
        """Analyze factor decay over multiple periods.

        Args:
            factor: (n_assets,) factor scores.
            returns: (n_dates, n_assets) forward returns.
            max_lag: Maximum lag to analyze.

        Returns:
            List of IC values at each lag.
        """
        decay_rates = []
        effective_max_lag = min(max_lag, returns.shape[0])

        for lag in range(1, effective_max_lag + 1):
            if lag >= returns.shape[0]:
                decay_rates.append(0.0)
                continue

            ic = self._compute_ic(factor, returns[lag])
            decay_rates.append(ic)

        logger.info("Factor decay analysis complete: %d lags", len(decay_rates))
        return decay_rates

    @staticmethod
    def _compute_ic(factor: np.ndarray, returns: np.ndarray) -> float:
        """Compute rank IC."""
        valid_mask = ~np.isnan(factor) & ~np.isnan(returns)
        if valid_mask.sum() < 3:
            return 0.0
        ic, _ = stats.spearmanr(factor[valid_mask], returns[valid_mask])
        return float(ic) if not np.isnan(ic) else 0.0

    @staticmethod
    def _compute_long_short_returns(
        factor: np.ndarray,
        returns: np.ndarray,
        n_quantiles: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute long, short, and long-short returns."""
        n_assets = len(factor)
        q_size = max(1, n_assets // n_quantiles)

        sorted_indices = np.argsort(factor)
        long_indices = sorted_indices[-q_size:]
        short_indices = sorted_indices[:q_size]

        long_returns = np.mean(returns[:, long_indices], axis=1)
        short_returns = np.mean(returns[:, short_indices], axis=1)
        ls_returns = long_returns - short_returns

        return long_returns, short_returns, ls_returns

    @staticmethod
    def _compute_sharpe(returns: np.ndarray) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    @staticmethod
    def _compute_max_drawdown(returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.min(drawdowns))
