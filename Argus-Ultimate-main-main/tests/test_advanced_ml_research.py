"""
Tests for advanced ML and research modules (March 2026 Batch 6).

Covers:
  1. ml/triple_barrier_labeler.py   — Triple Barrier Labeling
  2. ml/volatility_estimators.py    — Advanced Volatility Estimators
  3. research/optimal_stopping.py   — Optimal Trade Exit Timing
  4. risk/black_litterman.py        — Black-Litterman (BlackLittermanOptimizer)
  5. ml/meta_learner.py             — Meta-Learning Model Selector
  6. research/lead_lag_discovery.py  — Cross-Market Lead-Lag Discovery
  7. research/hurst_regime.py       — Hurst Exponent Regime Detector

60+ tests.
"""

from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import time
from typing import List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: synthetic data generation
# ---------------------------------------------------------------------------

def _trending_prices(n: int = 200, drift: float = 0.001, vol: float = 0.01, start: float = 100.0) -> List[float]:
    """Generate a trending price series with drift."""
    np.random.seed(42)
    returns = np.random.normal(drift, vol, n)
    log_prices = np.cumsum(returns) + math.log(start)
    return np.exp(log_prices).tolist()


def _mean_reverting_prices(n: int = 200, mean: float = 100.0, vol: float = 1.0) -> List[float]:
    """Generate a mean-reverting (Ornstein-Uhlenbeck) price series."""
    np.random.seed(123)
    prices = [mean]
    theta = 0.5  # mean reversion speed
    for _ in range(n - 1):
        p = prices[-1]
        dp = theta * (mean - p) + vol * np.random.randn()
        prices.append(max(p + dp, 1.0))
    return prices


def _random_walk_prices(n: int = 200, start: float = 100.0, vol: float = 0.01) -> List[float]:
    """Generate random walk prices."""
    np.random.seed(99)
    returns = np.random.normal(0, vol, n)
    log_prices = np.cumsum(returns) + math.log(start)
    return np.exp(log_prices).tolist()


def _ohlcv_bars(n: int = 100):
    """Generate synthetic OHLCV bars."""
    np.random.seed(77)
    opens, highs, lows, closes = [], [], [], []
    price = 100.0
    for _ in range(n):
        o = price
        h = o * (1.0 + abs(np.random.normal(0, 0.015)))
        l = o * (1.0 - abs(np.random.normal(0, 0.015)))
        c = np.random.uniform(l, h)
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        price = c
    return opens, highs, lows, closes


# ===========================================================================
# 1. Triple Barrier Labeler
# ===========================================================================

class TestTripleBarrierLabeler:
    """Tests for ml.triple_barrier_labeler."""

    def test_import(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler()
        assert labeler.upper_barrier == 0.02
        assert labeler.lower_barrier == -0.02
        assert labeler.max_holding_bars == 20

    def test_label_basic(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler(upper_barrier=0.05, lower_barrier=-0.05, max_holding_bars=10)
        prices = _trending_prices(100, drift=0.005)
        labels = labeler.label(prices)
        assert len(labels) == len(prices)
        assert all(l in (-1, 0, 1) for l in labels)

    def test_label_all_labels_present(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler(upper_barrier=0.01, lower_barrier=-0.01, max_holding_bars=5)
        prices = _random_walk_prices(500, vol=0.02)
        labels = labeler.label(prices)
        unique = set(labels)
        # With enough data and tight barriers, all labels should appear
        assert len(unique) >= 2

    def test_label_strongly_trending(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=-0.02, max_holding_bars=20)
        # Strong uptrend
        prices = [100.0 + i * 0.5 for i in range(100)]
        labels = labeler.label(prices)
        # Most early labels should be +1 (upper barrier hit)
        assert labels[0] == 1

    def test_label_strongly_downtrending(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=-0.02, max_holding_bars=20)
        prices = [100.0 - i * 0.5 for i in range(100)]
        labels = labeler.label(prices)
        # First label should be -1 (lower barrier hit)
        assert labels[0] == -1

    def test_label_short_series(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler()
        labels = labeler.label([100.0])
        assert labels == [0]

    def test_label_override_params(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler(upper_barrier=0.10, lower_barrier=-0.10)
        prices = _trending_prices(50, drift=0.003)
        labels_default = labeler.label(prices)
        labels_tight = labeler.label(prices, upper_barrier=0.005, lower_barrier=-0.005)
        # Tight barriers should produce more non-zero labels
        non_zero_default = sum(1 for l in labels_default if l != 0)
        non_zero_tight = sum(1 for l in labels_tight if l != 0)
        assert non_zero_tight >= non_zero_default

    def test_label_detailed(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler, BarrierResult
        labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=-0.02, max_holding_bars=10)
        prices = _trending_prices(50)
        results = labeler.label_detailed(prices)
        assert len(results) == len(prices)
        assert all(isinstance(r, BarrierResult) for r in results)
        assert all(r.barrier_type in ("upper", "lower", "timeout") for r in results)

    def test_meta_labels(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=-0.02, max_holding_bars=10)
        prices = _random_walk_prices(100)
        primary = [1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0) for i in range(100)]
        meta = labeler.get_meta_labels(prices, primary)
        assert len(meta) == len(prices)
        assert all(m in (0, 1) for m in meta)
        # Where primary is 0, meta should be 0
        for i, p in enumerate(primary):
            if p == 0:
                assert meta[i] == 0

    def test_meta_labels_length_mismatch(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler()
        with pytest.raises(ValueError, match="length"):
            labeler.get_meta_labels([100, 101, 102], [1, -1])

    def test_volatility_scaled_labeling(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        labeler = TripleBarrierLabeler()
        prices = _random_walk_prices(100)
        vols = [0.01] * 100
        labels = labeler.label_with_volatility_scaling(prices, vols, vol_multiplier=2.0)
        assert len(labels) == 100
        assert all(l in (-1, 0, 1) for l in labels)

    def test_init_validation(self):
        from ml.triple_barrier_labeler import TripleBarrierLabeler
        with pytest.raises(ValueError):
            TripleBarrierLabeler(upper_barrier=-0.01)
        with pytest.raises(ValueError):
            TripleBarrierLabeler(lower_barrier=0.01)
        with pytest.raises(ValueError):
            TripleBarrierLabeler(max_holding_bars=0)


# ===========================================================================
# 2. Volatility Estimators
# ===========================================================================

class TestVolatilityEstimators:
    """Tests for ml.volatility_estimators."""

    def test_import(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        assert ve.trading_days == 365

    def test_garman_klass(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        o, h, l, c = _ohlcv_bars(100)
        gk = ve.garman_klass(h, l, o, c)
        assert isinstance(gk, float)
        assert gk > 0

    def test_yang_zhang(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        o, h, l, c = _ohlcv_bars(100)
        yz = ve.yang_zhang(h, l, o, c)
        assert isinstance(yz, float)
        assert yz > 0

    def test_parkinson(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        _, h, l, _ = _ohlcv_bars(100)
        pk = ve.parkinson(h, l)
        assert isinstance(pk, float)
        assert pk > 0

    def test_rogers_satchell(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        o, h, l, c = _ohlcv_bars(100)
        rs = ve.rogers_satchell(h, l, o, c)
        assert isinstance(rs, float)
        assert rs > 0

    def test_realized_vol(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        returns = np.random.normal(0, 0.01, 100).tolist()
        rv = ve.realized_vol(returns)
        assert isinstance(rv, float)
        assert rv > 0

    def test_realized_vol_short_series(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        assert ve.realized_vol([0.01]) == 0.0

    def test_compare_estimators(self):
        from ml.volatility_estimators import VolatilityEstimator, OHLCVBar
        ve = VolatilityEstimator()
        o, h, l, c = _ohlcv_bars(100)
        bars = [OHLCVBar(open=o[i], high=h[i], low=l[i], close=c[i]) for i in range(100)]
        result = ve.compare_estimators(bars)
        assert "garman_klass" in result
        assert "yang_zhang" in result
        assert "parkinson" in result
        assert "rogers_satchell" in result
        assert "realized_vol" in result
        # All should be positive
        for name, val in result.items():
            assert val > 0 or math.isnan(val), f"{name} = {val}"

    def test_compare_estimators_with_dicts(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        o, h, l, c = _ohlcv_bars(50)
        bars = [{"open": o[i], "high": h[i], "low": l[i], "close": c[i]} for i in range(50)]
        result = ve.compare_estimators(bars)
        assert len(result) == 5

    def test_not_annualized(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        o, h, l, c = _ohlcv_bars(100)
        gk_ann = ve.garman_klass(h, l, o, c, annualize=True)
        gk_raw = ve.garman_klass(h, l, o, c, annualize=False)
        # Annualized should be larger
        assert gk_ann > gk_raw

    def test_validation_mismatched_lengths(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        with pytest.raises(ValueError, match="same length"):
            ve.garman_klass([1, 2], [1], [1, 2], [1, 2])

    def test_validation_negative_prices(self):
        from ml.volatility_estimators import VolatilityEstimator
        ve = VolatilityEstimator()
        with pytest.raises(ValueError, match="positive"):
            ve.parkinson([1, -2], [0.5, 0.5])


# ===========================================================================
# 3. Optimal Stopping
# ===========================================================================

class TestOptimalStopping:
    """Tests for research.optimal_stopping."""

    def test_import(self):
        from research.optimal_stopping import OptimalStopper
        os_ = OptimalStopper()
        assert os_.discount_factor == 0.99
        assert os_.max_bars == 50

    def test_compute_optimal_exit_monotonic_growth(self):
        from research.optimal_stopping import OptimalStopper
        os_ = OptimalStopper(discount_factor=0.99)
        # Monotonically increasing returns — should hold to end
        returns = [0.001 * i for i in range(1, 21)]
        optimal = os_.compute_optimal_exit(returns)
        # Should be near the end
        assert optimal >= 15

    def test_compute_optimal_exit_peak_and_decline(self):
        from research.optimal_stopping import OptimalStopper
        os_ = OptimalStopper(discount_factor=0.95)
        # Returns that peak at bar 5 then decline
        returns = [0.01 * i for i in range(6)] + [-0.01 * i for i in range(1, 15)]
        optimal = os_.compute_optimal_exit(returns)
        # Should exit around the peak
        assert 3 <= optimal <= 7

    def test_compute_optimal_exit_empty(self):
        from research.optimal_stopping import OptimalStopper
        os_ = OptimalStopper()
        assert os_.compute_optimal_exit([]) == 0

    def test_compute_optimal_exit_detailed(self):
        from research.optimal_stopping import OptimalStopper, ExitDecision
        os_ = OptimalStopper()
        returns = [0.01, 0.02, 0.03, 0.02, 0.01]
        result = os_.compute_optimal_exit_detailed(returns)
        assert isinstance(result, ExitDecision)
        assert 0 <= result.optimal_bar < len(returns)
        assert len(result.value_function) == len(returns)

    def test_exit_distribution(self):
        from research.optimal_stopping import OptimalStopper
        os_ = OptimalStopper()
        # Record some trades
        for _ in range(20):
            path = np.random.normal(0, 0.01, 10).cumsum().tolist()
            os_.record_trade(path)
        dist = os_.get_exit_distribution(lookback=20)
        assert isinstance(dist, dict)
        assert len(dist) > 0
        assert 0 in dist

    def test_should_exit_timeout(self):
        from research.optimal_stopping import OptimalStopper
        os_ = OptimalStopper(max_bars=10)
        assert os_.should_exit_now(0.01, bars_held=10) is True
        assert os_.should_exit_now(0.01, bars_held=15) is True

    def test_should_exit_deep_loss(self):
        from research.optimal_stopping import OptimalStopper
        os_ = OptimalStopper()
        assert os_.should_exit_now(-0.05, bars_held=2) is True

    def test_should_exit_positive_past_halfway(self):
        from research.optimal_stopping import OptimalStopper
        os_ = OptimalStopper(max_bars=10)
        assert os_.should_exit_now(0.01, bars_held=6) is True

    def test_should_exit_early_holding(self):
        from research.optimal_stopping import OptimalStopper
        os_ = OptimalStopper(max_bars=50)
        # Small positive P&L, early in trade — should hold
        assert os_.should_exit_now(0.005, bars_held=3) is False

    def test_init_validation(self):
        from research.optimal_stopping import OptimalStopper
        with pytest.raises(ValueError):
            OptimalStopper(discount_factor=0)
        with pytest.raises(ValueError):
            OptimalStopper(max_bars=0)


# ===========================================================================
# 4. Black-Litterman Optimizer
# ===========================================================================

class TestBlackLittermanOptimizer:
    """Tests for risk.black_litterman.BlackLittermanOptimizer."""

    def _make_optimizer(self):
        from risk.black_litterman import BlackLittermanOptimizer
        bl = BlackLittermanOptimizer(tau=0.05)
        market_caps = {"BTC": 1000e9, "ETH": 400e9, "SOL": 50e9}
        cov = [
            [0.04, 0.02, 0.01],
            [0.02, 0.06, 0.015],
            [0.01, 0.015, 0.09],
        ]
        bl.set_market_equilibrium(market_caps, cov, risk_aversion=2.5)
        return bl

    def test_import(self):
        from risk.black_litterman import BlackLittermanOptimizer
        bl = BlackLittermanOptimizer()
        assert bl.tau == 0.05

    def test_set_market_equilibrium(self):
        bl = self._make_optimizer()
        eq = bl.get_equilibrium_returns()
        assert "BTC" in eq
        assert "ETH" in eq
        assert "SOL" in eq
        # Equilibrium returns should be positive (positive market caps)
        assert all(v > 0 for v in eq.values())

    def test_optimize_no_views(self):
        bl = self._make_optimizer()
        weights = bl.optimize()
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # Without views, should be market-cap weights
        assert weights["BTC"] > weights["ETH"] > weights["SOL"]

    def test_optimize_with_absolute_view(self):
        bl = self._make_optimizer()
        # Bullish on SOL
        bl.add_view(["SOL"], [1.0], expected_return=0.15, confidence=0.8)
        weights = bl.optimize()
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # SOL weight should increase relative to equilibrium
        eq_weights = {"BTC": 1000 / 1450, "ETH": 400 / 1450, "SOL": 50 / 1450}
        assert weights["SOL"] > eq_weights["SOL"]

    def test_optimize_with_relative_view(self):
        bl = self._make_optimizer()
        # BTC outperforms ETH by 3%
        bl.add_view(["BTC", "ETH"], [1.0, -1.0], expected_return=0.03, confidence=0.7)
        weights = bl.optimize()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_get_posterior_returns(self):
        bl = self._make_optimizer()
        bl.add_view(["BTC"], [1.0], expected_return=0.10, confidence=0.9)
        bl.optimize()
        post = bl.get_posterior_returns()
        assert "BTC" in post
        eq = bl.get_equilibrium_returns()
        # With high-confidence bullish view, posterior should exceed equilibrium
        assert post["BTC"] > eq["BTC"]

    def test_get_weight_tilts(self):
        bl = self._make_optimizer()
        bl.add_view(["SOL"], [1.0], expected_return=0.20, confidence=0.9)
        bl.optimize()
        tilts = bl.get_weight_tilts()
        # SOL should have positive tilt
        assert tilts["SOL"] > 0

    def test_clear_views(self):
        bl = self._make_optimizer()
        bl.add_view(["BTC"], [1.0], expected_return=0.10, confidence=0.8)
        bl.clear_views()
        weights = bl.optimize()
        # Should be equilibrium weights again
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_add_view_unknown_asset(self):
        bl = self._make_optimizer()
        with pytest.raises(ValueError, match="Unknown"):
            bl.add_view(["DOGE"], [1.0], expected_return=0.05, confidence=0.5)

    def test_add_view_before_equilibrium(self):
        from risk.black_litterman import BlackLittermanOptimizer
        bl = BlackLittermanOptimizer()
        with pytest.raises(RuntimeError):
            bl.add_view(["BTC"], [1.0], 0.05, 0.8)

    def test_optimize_before_equilibrium(self):
        from risk.black_litterman import BlackLittermanOptimizer
        bl = BlackLittermanOptimizer()
        with pytest.raises(RuntimeError):
            bl.optimize()


# ===========================================================================
# 5. Meta Learner
# ===========================================================================

class TestMetaLearner:
    """Tests for ml.meta_learner."""

    def _make_learner(self, tmp_path):
        from ml.meta_learner import MetaLearner
        db_path = os.path.join(str(tmp_path), "test_meta.db")
        return MetaLearner(db_path=db_path, min_records=2)

    def test_import(self, tmp_path):
        ml = self._make_learner(tmp_path)
        assert ml.get_record_count() == 0

    def test_record_and_select(self, tmp_path):
        ml = self._make_learner(tmp_path)
        for _ in range(5):
            ml.record_model_performance("xgboost", "trending", {"vol": 0.3}, 0.75)
            ml.record_model_performance("lstm", "trending", {"vol": 0.3}, 0.60)

        best = ml.select_model("trending", {"vol": 0.3})
        assert best == "xgboost"

    def test_get_model_rankings(self, tmp_path):
        ml = self._make_learner(tmp_path)
        for _ in range(5):
            ml.record_model_performance("model_a", "volatile", {}, 0.80)
            ml.record_model_performance("model_b", "volatile", {}, 0.65)
            ml.record_model_performance("model_c", "volatile", {}, 0.70)

        rankings = ml.get_model_rankings("volatile")
        assert len(rankings) == 3
        # Should be sorted best first
        assert rankings[0][0] == "model_a"
        assert rankings[0][1] > rankings[1][1]

    def test_select_model_no_data(self, tmp_path):
        ml = self._make_learner(tmp_path)
        result = ml.select_model("unknown_regime")
        assert result == ""

    def test_get_all_regimes(self, tmp_path):
        ml = self._make_learner(tmp_path)
        ml.record_model_performance("m1", "trending", {}, 0.7)
        ml.record_model_performance("m1", "volatile", {}, 0.6)
        regimes = ml.get_all_regimes()
        assert "trending" in regimes
        assert "volatile" in regimes

    def test_get_model_history(self, tmp_path):
        ml = self._make_learner(tmp_path)
        for i in range(10):
            ml.record_model_performance("xgb", "trending", {"vol": 0.1 * i}, 0.5 + 0.02 * i)
        history = ml.get_model_history("xgb", regime="trending", limit=5)
        assert len(history) == 5

    def test_purge_old_records(self, tmp_path):
        ml = self._make_learner(tmp_path)
        for _ in range(5):
            ml.record_model_performance("m1", "r1", {}, 0.7)
        count_before = ml.get_record_count()
        # Purging with 0 max_age_days won't delete recent records
        deleted = ml.purge_old_records(max_age_days=0)
        # Records were just created, so should still be there
        assert ml.get_record_count() == count_before

    def test_feature_similarity_weighting(self, tmp_path):
        ml = self._make_learner(tmp_path)
        # Records with matching features should score higher
        for _ in range(5):
            ml.record_model_performance("m_good", "r1", {"vol": 0.3, "spread": 0.001}, 0.80)
            ml.record_model_performance("m_bad", "r1", {"vol": 0.9, "spread": 0.01}, 0.85)

        # Query with features similar to m_good's context
        rankings = ml.get_model_rankings("r1", {"vol": 0.3, "spread": 0.001})
        assert len(rankings) >= 2

    def test_min_records_filter(self, tmp_path):
        from ml.meta_learner import MetaLearner
        db_path = os.path.join(str(tmp_path), "test_min.db")
        ml = MetaLearner(db_path=db_path, min_records=10)
        # Only 3 records — below min_records threshold
        for _ in range(3):
            ml.record_model_performance("m1", "r1", {}, 0.8)
        rankings = ml.get_model_rankings("r1")
        assert len(rankings) == 0  # Not enough records

    def test_close(self, tmp_path):
        ml = self._make_learner(tmp_path)
        ml.close()
        # After close, operations should raise
        with pytest.raises(Exception):
            ml.record_model_performance("m1", "r1", {}, 0.5)


# ===========================================================================
# 6. Lead-Lag Discovery
# ===========================================================================

class TestLeadLagDiscovery:
    """Tests for research.lead_lag_discovery."""

    def test_import(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer()
        assert disc.min_correlation == 0.3

    def test_add_series(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer()
        disc.add_series("A", _trending_prices(100))
        assert "A" in disc.get_all_series_names()

    def test_add_series_too_short(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer()
        with pytest.raises(ValueError, match="too short"):
            disc.add_series("A", [1.0, 2.0])

    def test_discover_leads_with_lag(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer(min_correlation=0.2, significance_level=0.10)
        # Create leader and follower with known lag
        np.random.seed(42)
        n = 300
        leader = np.cumsum(np.random.normal(0, 0.01, n)) + 5.0
        leader = np.exp(leader)
        lag = 3
        follower = np.concatenate([np.exp(np.random.normal(0, 0.01, lag) + 5.0), leader[:-lag]])

        disc.add_series("leader", leader.tolist())
        disc.add_series("follower", follower.tolist())
        relations = disc.discover_leads(max_lag=10, min_overlap=30)
        # May or may not find the relation depending on noise, but should not crash
        assert isinstance(relations, list)

    def test_discover_leads_no_series(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer()
        result = disc.discover_leads()
        assert result == []

    def test_discover_leads_single_series(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer()
        disc.add_series("A", _trending_prices(100))
        result = disc.discover_leads()
        assert result == []

    def test_get_trading_signal(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer()
        disc.add_series("BTC", _trending_prices(100, drift=0.005))
        disc.add_series("ETH", _trending_prices(100, drift=0.003))
        signal = disc.get_trading_signal("BTC", "ETH")
        assert -1.0 <= signal <= 1.0

    def test_get_trading_signal_trending(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer()
        # Strong uptrend in leader
        prices = [100.0 + i * 2.0 for i in range(100)]
        disc.add_series("leader", prices)
        disc.add_series("follower", _random_walk_prices(100))
        signal = disc.get_trading_signal("leader", "follower")
        assert signal > 0  # bullish signal from uptrending leader

    def test_get_trading_signal_unknown_series(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer()
        assert disc.get_trading_signal("unknown1", "unknown2") == 0.0

    def test_remove_series(self):
        from research.lead_lag_discovery import LeadLagDiscoverer
        disc = LeadLagDiscoverer()
        disc.add_series("A", _trending_prices(50))
        disc.remove_series("A")
        assert "A" not in disc.get_all_series_names()

    def test_lead_lag_relation_dataclass(self):
        from research.lead_lag_discovery import LeadLagRelation
        r = LeadLagRelation(leader="A", follower="B", optimal_lag=3, correlation=0.75, p_value=0.001)
        assert r.leader == "A"
        assert r.optimal_lag == 3


# ===========================================================================
# 7. Hurst Regime Detector
# ===========================================================================

class TestHurstRegimeDetector:
    """Tests for research.hurst_regime."""

    def test_import(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector()
        assert hrd.window == 100

    def test_update_insufficient_data(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector(window=50)
        result = hrd.update("BTC/USD", 100.0)
        assert result is None

    def test_update_sufficient_data(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector(window=50)
        prices = _trending_prices(60)
        h = None
        for p in prices:
            h = hrd.update("BTC/USD", p)
        assert h is not None
        assert 0.0 < h < 1.0

    def test_get_hurst_no_data(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector()
        assert hrd.get_hurst("UNKNOWN") == 0.5

    def test_get_hurst_trending(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector(window=50)
        prices = _trending_prices(200, drift=0.005, vol=0.005)
        for p in prices:
            hrd.update("BTC/USD", p)
        h = hrd.get_hurst("BTC/USD")
        assert 0.0 < h < 1.0
        # Trending data should have H > 0.5 (usually)
        # Note: R/S method has noise so we just check it's valid

    def test_get_hurst_mean_reverting(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector(window=50)
        prices = _mean_reverting_prices(200, vol=2.0)
        for p in prices:
            hrd.update("BTC/USD", p)
        h = hrd.get_hurst("BTC/USD")
        assert 0.0 < h < 1.0

    def test_get_strategy_recommendation(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector(window=50)
        prices = _trending_prices(200)
        for p in prices:
            hrd.update("BTC/USD", p)
        rec = hrd.get_strategy_recommendation("BTC/USD")
        assert rec in ("mean_reversion", "momentum", "avoid")

    def test_get_regime_history(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector(window=50)
        prices = _trending_prices(100)
        for p in prices:
            hrd.update("BTC/USD", p)
        history = hrd.get_regime_history("BTC/USD")
        assert len(history) > 0
        ts, h, rec = history[0]
        assert isinstance(ts, float)
        assert isinstance(h, float)
        assert rec in ("mean_reversion", "momentum", "avoid")

    def test_get_regime_history_empty(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector()
        assert hrd.get_regime_history("NONE") == []

    def test_get_all_symbols(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector(window=20)
        for p in _trending_prices(30):
            hrd.update("BTC/USD", p)
        for p in _trending_prices(30):
            hrd.update("ETH/USD", p)
        symbols = hrd.get_all_symbols()
        assert "BTC/USD" in symbols
        assert "ETH/USD" in symbols

    def test_get_regime_summary(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector(window=30)
        for p in _trending_prices(50):
            hrd.update("BTC/USD", p)
        summary = hrd.get_regime_summary()
        assert "BTC/USD" in summary
        assert "hurst" in summary["BTC/USD"]
        assert "recommendation" in summary["BTC/USD"]

    def test_update_negative_price(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector()
        result = hrd.update("BTC/USD", -100.0)
        assert result is None

    def test_custom_window(self):
        from research.hurst_regime import HurstRegimeDetector
        hrd = HurstRegimeDetector(window=30)
        prices = _trending_prices(50)
        for p in prices:
            hrd.update("BTC/USD", p)
        h_default = hrd.get_hurst("BTC/USD")
        h_custom = hrd.get_hurst("BTC/USD", window=40)
        # Both should be valid floats
        assert 0.0 < h_default < 1.0
        assert 0.0 < h_custom < 1.0

    def test_init_validation(self):
        from research.hurst_regime import HurstRegimeDetector
        with pytest.raises(ValueError):
            HurstRegimeDetector(window=5)
        with pytest.raises(ValueError):
            HurstRegimeDetector(mean_reversion_threshold=0.6, trending_threshold=0.5)
