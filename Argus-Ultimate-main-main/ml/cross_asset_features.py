"""
Cross-Asset Feature Engineering.

Generates features that capture relationships between multiple assets:
- Cross-asset correlations (rolling, conditional)
- Lead-lag relationships
- Relative strength indicators
- Factor exposure features
- Sector/asset-class decomposition

Usage:
    builder = CrossAssetFeatureBuilder(assets=["BTC", "ETH", "SOL"])
    features = builder.build(features_dict)  # Dict of asset -> price series
    # Returns dict of cross-asset feature names -> values
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetFeatureConfig:
    """Configuration for cross-asset features."""

    # Feature types to generate
    include_correlations: bool = True
    include_lead_lag: bool = True
    include_relative_strength: bool = True
    include_factor_exposure: bool = False
    include_spread_features: bool = True

    # Rolling windows
    short_window: int = 20
    medium_window: int = 60
    long_window: int = 200

    # Correlation threshold for signal extraction
    correlation_threshold: float = 0.5
    lead_lag_max_lag: int = 10

    # Reference asset for relative strength
    reference_asset: str = "BTC"


class CrossAssetFeatureBuilder:
    """
    Build cross-asset features from multiple price series.

    Input: Dict[str, pd.DataFrame] or Dict[str, np.ndarray]
        Asset name -> OHLCV or returns data

    Output: Dict[str, float]
        Feature name -> value
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        config: Optional[CrossAssetFeatureConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.assets = assets or []
        self.config = config or CrossAssetFeatureConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._correlation_cache: Dict[Tuple[str, str], float] = {}
        self._lead_lag_cache: Dict[Tuple[str, str], int] = {}

    def set_assets(self, assets: List[str]) -> None:
        """Set the list of assets."""
        self.assets = list(assets)

    def build(
        self,
        data: Dict[str, Any],
        *,
        returns_only: bool = True,
    ) -> Dict[str, float]:
        """
        Build cross-asset features from data.

        Args:
            data: Dict of asset -> price/returns data
            returns_only: If True, assume data contains returns not prices

        Returns:
            Dict of feature names -> values
        """
        features: Dict[str, float] = {}

        # Extract returns from data
        returns_dict = self._extract_returns(data, returns_only)

        if len(returns_dict) < 2:
            # Need at least 2 assets for cross-asset features
            return features

        asset_list = list(returns_dict.keys())
        returns_array = np.column_stack([returns_dict[a] for a in asset_list])
        n_samples, n_assets = returns_array.shape

        # Build features based on config
        if self.config.include_correlations:
            features.update(self._build_correlation_features(returns_array, asset_list))

        if self.config.include_lead_lag:
            features.update(self._build_lead_lag_features(returns_array, asset_list))

        if self.config.include_relative_strength:
            features.update(
                self._build_relative_strength_features(returns_array, asset_list)
            )

        if self.config.include_spread_features:
            features.update(self._build_spread_features(returns_array, asset_list))

        if self.config.include_factor_exposure:
            features.update(self._build_factor_exposure(returns_array, asset_list))

        return features

    def _extract_returns(
        self,
        data: Dict[str, Any],
        returns_only: bool,
    ) -> Dict[str, np.ndarray]:
        """Extract returns arrays from data."""
        returns_dict: Dict[str, np.ndarray] = {}

        for asset, asset_data in data.items():
            if isinstance(asset_data, np.ndarray):
                if returns_only:
                    returns_dict[asset] = asset_data
                else:
                    # Compute returns from prices
                    returns_dict[asset] = self._compute_returns(asset_data)
            elif hasattr(asset_data, "close"):
                # Assume OHLCV DataFrame
                closes = asset_data.close.values
                returns_dict[asset] = self._compute_returns(closes)
            elif hasattr(asset_data, "values"):
                # Pandas Series
                returns_dict[asset] = self._compute_returns(asset_data.values)
            else:
                # Try to treat as dict with close key
                try:
                    closes = np.array(asset_data.get("close", asset_data.get("Close", [])))
                    if len(closes) > 0:
                        returns_dict[asset] = self._compute_returns(closes)
                except Exception:
                    pass

        return returns_dict

    def _compute_returns(self, prices: np.ndarray) -> np.ndarray:
        """Compute returns from prices."""
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 2:
            return np.zeros(len(prices))

        returns = np.zeros(len(prices))
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        # Replace NaN/inf
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        return returns

    def _build_correlation_features(
        self,
        returns: np.ndarray,
        assets: List[str],
    ) -> Dict[str, float]:
        """Build correlation-based features."""
        features: Dict[str, float] = {}
        n_samples, n_assets = returns.shape

        # Rolling correlations
        for window in [self.config.short_window, self.config.medium_window, self.config.long_window]:
            if n_samples >= window:
                window_returns = returns[-window:]
                corr_matrix = np.corrcoef(window_returns.T)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

                # Average correlation (excluding diagonal)
                mask = ~np.eye(n_assets, dtype=bool)
                avg_corr = float(np.mean(corr_matrix[mask]))
                features[f"cross_asset_avg_correlation_{window}"] = avg_corr

                # Max/min correlation
                features[f"cross_asset_max_correlation_{window}"] = float(np.max(corr_matrix[mask]))
                features[f"cross_asset_min_correlation_{window}"] = float(np.min(corr_matrix[mask]))

                # Correlation dispersion (std of off-diagonal)
                features[f"cross_asset_corr_dispersion_{window}"] = float(np.std(corr_matrix[mask]))

        # Correlation to reference asset
        if self.config.reference_asset in assets:
            ref_idx = assets.index(self.config.reference_asset)
            ref_returns = returns[:, ref_idx]

            for i, asset in enumerate(assets):
                if i != ref_idx:
                    asset_returns = returns[:, i]
                    corr = float(np.corrcoef(ref_returns, asset_returns)[0, 1])
                    corr = 0.0 if np.isnan(corr) else corr
                    features[f"correlation_to_{self.config.reference_asset}_{asset}"] = corr

        return features

    def _build_lead_lag_features(
        self,
        returns: np.ndarray,
        assets: List[str],
    ) -> Dict[str, float]:
        """Build lead-lag relationship features."""
        features: Dict[str, float] = {}
        n_samples, n_assets = returns.shape

        # Use short window for lead-lag
        window = min(self.config.short_window, n_samples)
        window_returns = returns[-window:]

        for i in range(n_assets):
            for j in range(n_assets):
                if i >= j:
                    continue

                asset_i = assets[i]
                asset_j = assets[j]
                key = (asset_i, asset_j)

                # Compute optimal lag
                if key in self._lead_lag_cache:
                    lag = self._lead_lag_cache[key]
                else:
                    lag = self._find_optimal_lag(
                        window_returns[:, i], window_returns[:, j]
                    )
                    self._lead_lag_cache[key] = lag

                features[f"lead_lag_{asset_i}_{asset_j}"] = float(lag)

        return features

    def _find_optimal_lag(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
    ) -> int:
        """Find optimal lag between two return series."""
        max_lag = min(self.config.lead_lag_max_lag, len(returns_a) // 2)
        if max_lag < 1:
            return 0

        best_corr = -1.0
        best_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = float(np.corrcoef(returns_a, returns_b)[0, 1])
            elif lag > 0:
                # A leads B
                corr = float(np.corrcoef(returns_a[:-lag], returns_b[lag:])[0, 1])
            else:
                # B leads A
                corr = float(np.corrcoef(returns_a[-lag:], returns_b[:lag])[0, 1])

            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_lag

    def _build_relative_strength_features(
        self,
        returns: np.ndarray,
        assets: List[str],
    ) -> Dict[str, float]:
        """Build relative strength indicators."""
        features: Dict[str, float] = {}
        n_samples, n_assets = returns.shape

        # Cumulative returns for each window
        for window in [self.config.short_window, self.config.medium_window]:
            if n_samples >= window:
                window_returns = returns[-window:]

                # Cumulative return for each asset
                cum_returns = np.sum(window_returns, axis=0)
                # Cumulative return for reference (first asset if not set)
                ref_asset = self.config.reference_asset
                if ref_asset in assets:
                    ref_idx = assets.index(ref_asset)
                    ref_cum = cum_returns[ref_idx]
                else:
                    ref_cum = np.mean(cum_returns)

                for i, asset in enumerate(assets):
                    if i != ref_idx if ref_asset in assets else True:
                        rs = cum_returns[i] - ref_cum
                        features[f"relative_strength_{asset}_vs_{ref_asset}_{window}"] = float(rs)

                # Relative strength spread
                features[f"relative_strength_spread_{window}"] = float(
                    np.max(cum_returns) - np.min(cum_returns)
                )

        return features

    def _build_spread_features(
        self,
        returns: np.ndarray,
        assets: List[str],
    ) -> Dict[str, float]:
        """Build pair spread features."""
        features: Dict[str, float] = {}
        n_samples, n_assets = returns.shape

        if n_assets < 2:
            return features

        # Pairwise spreads for top pairs (by correlation)
        window = min(self.config.short_window, n_samples)
        window_returns = returns[-window:]

        # Compute correlations for ranking
        correlations = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = float(np.corrcoef(window_returns[:, i], window_returns[:, j])[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                correlations.append((abs(corr), i, j))

        # Top 3 pairs by |correlation|
        correlations.sort(reverse=True)
        top_pairs = correlations[:3]

        for corr_abs, i, j in top_pairs:
            asset_i, asset_j = assets[i], assets[j]

            # Spread (difference in returns)
            spread = window_returns[:, i] - window_returns[:, j]
            features[f"spread_mean_{asset_i}_{asset_j}"] = float(np.mean(spread))
            features[f"spread_std_{asset_i}_{asset_j}"] = float(np.std(spread))
            features[f"spread_zscore_{asset_i}_{asset_j}"] = float(
                np.mean(spread) / (np.std(spread) + 1e-10)
            )

        return features

    def _build_factor_exposure(
        self,
        returns: np.ndarray,
        assets: List[str],
    ) -> Dict[str, float]:
        """Build factor exposure features (market, size, momentum)."""
        features: Dict[str, float] = {}
        n_samples, n_assets = returns.shape

        # Market factor = equal-weighted average
        market_returns = np.mean(returns, axis=1)

        # Compute exposures for each asset
        for i, asset in enumerate(assets):
            asset_returns = returns[:, i]

            # Beta to market
            market_var = np.var(market_returns)
            if market_var > 1e-10:
                covariance = np.cov(asset_returns, market_returns)[0, 1]
                beta = covariance / market_var
                features[f"market_beta_{asset}"] = float(beta)

            # Idiosyncratic volatility
            predicted = beta * market_returns if market_var > 1e-10 else 0
            residuals = asset_returns - predicted[:len(asset_returns)]
            features[f"idiosyncratic_vol_{asset}"] = float(np.std(residuals))

        # Average market beta
        betas = [v for k, v in features.items() if k.startswith("market_beta_")]
        if betas:
            features["cross_asset_avg_beta"] = float(np.mean(betas))

        return features

    def get_correlation_matrix(
        self,
        data: Dict[str, Any],
        window: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Get correlation matrix for assets."""
        returns_dict = self._extract_returns(data, returns_only=True)
        assets = list(returns_dict.keys())

        if len(assets) < 2:
            return np.array([]), assets

        returns_array = np.column_stack([returns_dict[a] for a in assets])
        n_samples, _ = returns_array.shape

        window = window or n_samples
        window = min(window, n_samples)
        window_returns = returns_array[-window:]

        corr_matrix = np.corrcoef(window_returns.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        return corr_matrix, assets


def create_cross_asset_builder(
    assets: List[str],
    config: Optional[CrossAssetFeatureConfig] = None,
) -> CrossAssetFeatureBuilder:
    """Factory function to create cross-asset builder."""
    return CrossAssetFeatureBuilder(assets=assets, config=config)


__all__ = [
    "CrossAssetFeatureBuilder",
    "CrossAssetFeatureConfig",
    "create_cross_asset_builder",
]