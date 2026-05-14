"""Tests for meta-learning adjustment."""
from __future__ import annotations

import unittest

from argus_live.optimization.meta_learning import (
    MetaLearningAdjustment,
    compute_meta_learning_adjustment,
)


class TestMetaLearning(unittest.TestCase):

    def test_stressed_regime_boosts_lr_and_threshold(self):
        adj = compute_meta_learning_adjustment(
            regime="STRESSED",
            signal_noise=0.3,
            recent_model_error_bps=5.0,
        )
        self.assertIsInstance(adj, MetaLearningAdjustment)
        self.assertGreaterEqual(adj.learning_rate_multiplier, 1.0)
        self.assertGreaterEqual(adj.threshold_multiplier, 1.0)

    def test_noisy_signal_lowers_confidence(self):
        adj = compute_meta_learning_adjustment(
            regime="TRENDING",
            signal_noise=0.9,
            recent_model_error_bps=5.0,
        )
        self.assertLess(adj.confidence_multiplier, 1.0)

    def test_high_error_slows_learning(self):
        adj_low = compute_meta_learning_adjustment("STRESSED", 0.3, 5.0)
        adj_high = compute_meta_learning_adjustment("STRESSED", 0.3, 60.0)
        self.assertGreater(adj_low.learning_rate_multiplier, adj_high.learning_rate_multiplier)

    def test_ranging_regime_slower(self):
        adj = compute_meta_learning_adjustment("RANGING", 0.2, 5.0)
        self.assertLess(adj.learning_rate_multiplier, 1.0)

    def test_frozen_dataclass(self):
        adj = compute_meta_learning_adjustment("TRENDING", 0.3, 5.0)
        with self.assertRaises(AttributeError):
            adj.learning_rate_multiplier = 0.0  # type: ignore[misc]

    def test_crisis_regime(self):
        adj = compute_meta_learning_adjustment("CRISIS", 0.3, 5.0)
        self.assertGreaterEqual(adj.learning_rate_multiplier, 1.0)
        self.assertGreaterEqual(adj.threshold_multiplier, 1.0)

    def test_low_noise_full_confidence(self):
        adj = compute_meta_learning_adjustment("TRENDING", 0.1, 5.0)
        self.assertAlmostEqual(adj.confidence_multiplier, 1.0)


if __name__ == "__main__":
    unittest.main()
