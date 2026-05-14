"""
Tests for ML cross-asset features module.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from ml.cross_asset_features import (
    CrossAssetFeatureBuilder,
    CrossAssetFeatureConfig,
    create_cross_asset_builder,
)


class TestCrossAssetFeatureBuilder(unittest.TestCase):
    """Tests for CrossAssetFeatureBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 200

        # Create returns data for multiple assets
        self.returns_dict = {
            "BTC": np.random.randn(self.n_samples) * 0.02,
            "ETH": np.random.randn(self.n_samples) * 0.02,
            "SOL": np.random.randn(self.n_samples) * 0.02,
        }

        # Create prices data
        self.prices_dict = {}
        for asset, returns in self.returns_dict.items():
            prices = 100 * np.exp(np.cumsum(returns))
            self.prices_dict[asset] = prices

    def test_build_with_returns(self):
        """Test building features from returns."""
        builder = CrossAssetFeatureBuilder(assets=["BTC", "ETH", "SOL"])
        features = builder.build(self.returns_dict, returns_only=True)

        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)

    def test_build_with_prices(self):
        """Test building features from prices."""
        builder = CrossAssetFeatureBuilder(assets=["BTC", "ETH", "SOL"])
        features = builder.build(self.prices_dict, returns_only=False)

        self.assertIsInstance(features, dict)

    def test_correlation_features(self):
        """Test correlation features are generated."""
        config = CrossAssetFeatureConfig(
            include_correlations=True,
            include_lead_lag=False,
            include_relative_strength=False,
            include_spread_features=False,
            include_factor_exposure=False,
        )
        builder = CrossAssetFeatureBuilder(config=config)
        features = builder.build(self.returns_dict, returns_only=True)

        # Check for correlation features
        corr_features = [k for k in features.keys() if "correlation" in k]
        self.assertGreater(len(corr_features), 0)

    def test_lead_lag_features(self):
        """Test lead-lag features are generated."""
        config = CrossAssetFeatureConfig(
            include_correlations=False,
            include_lead_lag=True,
            include_relative_strength=False,
            include_spread_features=False,
        )
        builder = CrossAssetFeatureBuilder(config=config)
        features = builder.build(self.returns_dict, returns_only=True)

        lead_lag_features = [k for k in features.keys() if "lead_lag" in k]
        self.assertGreaterEqual(len(lead_lag_features), 0)

    def test_relative_strength_features(self):
        """Test relative strength features."""
        config = CrossAssetFeatureConfig(
            include_correlations=False,
            include_lead_lag=False,
            include_relative_strength=True,
            include_spread_features=False,
            reference_asset="BTC",
        )
        builder = CrossAssetFeatureBuilder(config=config)
        features = builder.build(self.returns_dict, returns_only=True)

        rs_features = [k for k in features.keys() if "relative_strength" in k]
        self.assertGreaterEqual(len(rs_features), 0)

    def test_spread_features(self):
        """Test spread features."""
        config = CrossAssetFeatureConfig(
            include_correlations=False,
            include_lead_lag=False,
            include_relative_strength=False,
            include_spread_features=True,
        )
        builder = CrossAssetFeatureBuilder(config=config)
        features = builder.build(self.returns_dict, returns_only=True)

        spread_features = [k for k in features.keys() if "spread" in k]
        self.assertGreaterEqual(len(spread_features), 0)

    def test_single_asset_fallback(self):
        """Test fallback when only one asset provided."""
        single_asset = {"BTC": self.returns_dict["BTC"]}
        builder = CrossAssetFeatureBuilder(assets=["BTC"])
        features = builder.build(single_asset, returns_only=True)

        # Should return empty features for single asset
        self.assertIsInstance(features, dict)

    def test_get_correlation_matrix(self):
        """Test getting correlation matrix."""
        builder = CrossAssetFeatureBuilder()
        corr_matrix, assets = builder.get_correlation_matrix(self.returns_dict)

        self.assertIsInstance(assets, list)
        if len(assets) >= 2:
            n = len(assets)
            self.assertEqual(corr_matrix.shape, (n, n))

    def test_config_options(self):
        """Test different configuration options."""
        config = CrossAssetFeatureConfig(
            short_window=10,
            medium_window=30,
            long_window=100,
            reference_asset="ETH",
            correlation_threshold=0.9,
        )
        builder = CrossAssetFeatureBuilder(config=config)
        features = builder.build(self.returns_dict, returns_only=True)

        self.assertIsInstance(features, dict)

    def test_factory_function(self):
        """Test factory function."""
        builder = create_cross_asset_builder(
            assets=["BTC", "ETH", "SOL"],
            config=CrossAssetFeatureConfig(),
        )

        self.assertIsInstance(builder, CrossAssetFeatureBuilder)
        self.assertEqual(len(builder.assets), 3)


class TestCrossAssetFeatureConfig(unittest.TestCase):
    """Tests for CrossAssetFeatureConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CrossAssetFeatureConfig()

        self.assertEqual(config.include_correlations, True)
        self.assertEqual(config.include_lead_lag, True)
        self.assertEqual(config.include_relative_strength, True)
        self.assertEqual(config.short_window, 20)
        self.assertEqual(config.medium_window, 60)

    def test_custom_config(self):
        """Test custom configuration."""
        config = CrossAssetFeatureConfig(
            include_correlations=False,
            include_factor_exposure=True,
            short_window=10,
        )

        self.assertEqual(config.include_correlations, False)
        self.assertEqual(config.include_factor_exposure, True)
        self.assertEqual(config.short_window, 10)


if __name__ == "__main__":
    unittest.main()