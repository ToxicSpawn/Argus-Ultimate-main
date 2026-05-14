"""Tests for data intelligence modules — Google Trends, stablecoin flows,
exchange reserves, confidence calibration, and anti-gaming layer.

50+ tests covering construction, signal generation, edge cases, persistence,
and randomisation properties.
"""

from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from data.alternative.google_trends_tracker import (
    GoogleTrendsTracker,
    TrendSignal,
)
from data.onchain.stablecoin_flow_tracker import (
    AggregateFlow,
    StablecoinFlow,
    StablecoinFlowTracker,
)
from data.onchain.exchange_reserve_monitor import (
    ExchangeReserveMonitor,
    ReserveChange,
)
from ml.confidence_calibrator import (
    CalibrationReport,
    ConfidenceCalibrator,
)
from risk.anti_gaming_layer import (
    AntiGamingLayer,
    ExecutionMask,
)


# ===================================================================
# Google Trends Tracker
# ===================================================================

class TestGoogleTrendsTracker:
    """Tests for GoogleTrendsTracker."""

    def test_construction(self):
        tracker = GoogleTrendsTracker()
        assert tracker._cache_ttl_s == 3600.0

    def test_construction_custom_ttl(self):
        tracker = GoogleTrendsTracker(cache_ttl_s=600.0)
        assert tracker._cache_ttl_s == 600.0

    def test_neutral_signal_without_pytrends(self):
        """Without pytrends installed, should return neutral signal."""
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None  # Force no-pytrends path
        sig = tracker.get_trend_score("bitcoin")
        assert isinstance(sig, TrendSignal)
        assert sig.keyword == "bitcoin"
        assert sig.score == 50
        assert sig.trend_direction == "stable"
        assert sig.change_pct == 0.0

    def test_cache_hit(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        sig1 = tracker.get_trend_score("ethereum")
        sig2 = tracker.get_trend_score("ethereum")
        assert sig1.timestamp == sig2.timestamp  # Same cached object

    def test_cache_miss_different_keyword(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        sig1 = tracker.get_trend_score("bitcoin")
        sig2 = tracker.get_trend_score("ethereum")
        assert sig1.keyword == "bitcoin"
        assert sig2.keyword == "ethereum"

    def test_cache_miss_different_timeframe(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        sig1 = tracker.get_trend_score("bitcoin", "now 7-d")
        sig2 = tracker.get_trend_score("bitcoin", "today 3-m")
        # Both neutral but separate cache entries
        assert sig1.keyword == sig2.keyword

    def test_clear_cache(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        tracker.get_trend_score("bitcoin")
        assert len(tracker._cache) == 1
        tracker.clear_cache()
        assert len(tracker._cache) == 0

    def test_multi_keyword_signal_neutral(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        result = tracker.get_multi_keyword_signal()
        assert isinstance(result, float)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_multi_keyword_signal_custom_list(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        result = tracker.get_multi_keyword_signal(["bitcoin", "eth"])
        assert -1.0 <= result <= 1.0

    def test_multi_keyword_signal_empty_list(self):
        tracker = GoogleTrendsTracker()
        result = tracker.get_multi_keyword_signal([])
        assert result == 0.0

    def test_is_euphoria_neutral(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        assert tracker.is_euphoria() is False  # score=50 < 80

    def test_is_capitulation_neutral(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        assert tracker.is_capitulation() is False  # score=50 > 20

    def test_is_euphoria_custom_threshold(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        # Neutral score=50, threshold=40 -> should be True
        assert tracker.is_euphoria(threshold=40) is True

    def test_is_capitulation_custom_threshold(self):
        tracker = GoogleTrendsTracker()
        tracker._pytrends = None
        assert tracker.is_capitulation(threshold=60) is True

    def test_trend_signal_dataclass(self):
        sig = TrendSignal(
            keyword="test",
            score=75,
            trend_direction="rising",
            change_pct=12.5,
        )
        assert sig.keyword == "test"
        assert sig.score == 75
        assert sig.trend_direction == "rising"
        assert sig.change_pct == 12.5
        assert sig.timestamp > 0


# ===================================================================
# Stablecoin Flow Tracker
# ===================================================================

class TestStablecoinFlowTracker:
    """Tests for StablecoinFlowTracker."""

    def test_construction(self):
        tracker = StablecoinFlowTracker()
        assert tracker._neutral_threshold_pct == 0.1

    def test_construction_custom(self):
        tracker = StablecoinFlowTracker(max_history_hours=48, neutral_threshold_pct=0.5)
        assert tracker._max_history_s == 48 * 3600.0

    def test_update_supply(self):
        tracker = StablecoinFlowTracker()
        tracker.update_supply("USDT", 100_000_000_000)
        assert "USDT" in tracker.tracked_tokens

    def test_get_flow_no_data(self):
        tracker = StablecoinFlowTracker()
        flow = tracker.get_flow("USDT")
        assert isinstance(flow, StablecoinFlow)
        assert flow.direction == "neutral"
        assert flow.signal_bias == 0.0

    def test_get_flow_single_snapshot(self):
        tracker = StablecoinFlowTracker()
        tracker.update_supply("USDT", 100e9)
        flow = tracker.get_flow("USDT")
        assert flow.direction == "neutral"  # Need at least 2 snapshots

    def test_get_flow_mint(self):
        tracker = StablecoinFlowTracker()
        now = time.time()
        tracker.update_supply("USDT", 100e9, timestamp=now - 3600)
        tracker.update_supply("USDT", 105e9, timestamp=now)  # 5% increase
        flow = tracker.get_flow("USDT", lookback_hours=2)
        assert flow.direction == "mint"
        assert flow.net_change > 0
        assert flow.signal_bias > 0  # Minting is bullish

    def test_get_flow_burn(self):
        tracker = StablecoinFlowTracker()
        now = time.time()
        tracker.update_supply("USDC", 50e9, timestamp=now - 3600)
        tracker.update_supply("USDC", 45e9, timestamp=now)  # 10% decrease
        flow = tracker.get_flow("USDC", lookback_hours=2)
        assert flow.direction == "burn"
        assert flow.net_change < 0
        assert flow.signal_bias < 0  # Burning is bearish

    def test_get_flow_neutral_small_change(self):
        tracker = StablecoinFlowTracker()
        now = time.time()
        tracker.update_supply("DAI", 5e9, timestamp=now - 3600)
        tracker.update_supply("DAI", 5.001e9, timestamp=now)  # Tiny change
        flow = tracker.get_flow("DAI", lookback_hours=2)
        assert flow.direction == "neutral"

    def test_case_insensitive_token(self):
        tracker = StablecoinFlowTracker()
        tracker.update_supply("usdt", 100e9)
        tracker.update_supply("Usdt", 101e9)
        assert "USDT" in tracker.tracked_tokens

    def test_aggregate_flow_no_data(self):
        tracker = StablecoinFlowTracker()
        agg = tracker.get_aggregate_flow()
        assert isinstance(agg, AggregateFlow)
        assert agg.overall_bias == 0.0
        assert len(agg.per_token) == 4  # default tokens

    def test_aggregate_flow_with_data(self):
        tracker = StablecoinFlowTracker()
        now = time.time()
        for tok, base in [("USDT", 100e9), ("USDC", 50e9)]:
            tracker.update_supply(tok, base, timestamp=now - 3600)
            tracker.update_supply(tok, base * 1.03, timestamp=now)
        agg = tracker.get_aggregate_flow(["USDT", "USDC"], lookback_hours=2)
        assert agg.total_net_change > 0

    def test_tracked_tokens(self):
        tracker = StablecoinFlowTracker()
        tracker.update_supply("DAI", 5e9)
        tracker.update_supply("USDT", 100e9)
        assert tracker.tracked_tokens == ["DAI", "USDT"]

    def test_stablecoin_flow_dataclass(self):
        flow = StablecoinFlow(
            token="USDT", net_change=1e8, pct_change=1.5,
            direction="mint", signal_bias=0.15,
        )
        assert flow.token == "USDT"


# ===================================================================
# Exchange Reserve Monitor
# ===================================================================

class TestExchangeReserveMonitor:
    """Tests for ExchangeReserveMonitor."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        return str(tmp_path / "test_reserves.db")

    def test_construction(self, tmp_db):
        mon = ExchangeReserveMonitor(db_path=tmp_db)
        assert Path(tmp_db).exists()

    def test_update_reserve(self, tmp_db):
        mon = ExchangeReserveMonitor(db_path=tmp_db)
        mon.update_reserve("binance", "BTC", 500_000.0)
        assert ("binance", "BTC") in mon.tracked_pairs

    def test_get_reserve_change_no_data(self, tmp_db):
        mon = ExchangeReserveMonitor(db_path=tmp_db)
        rc = mon.get_reserve_change("BTC")
        assert isinstance(rc, ReserveChange)
        assert rc.direction == "neutral"

    def test_get_reserve_change_inflow(self, tmp_db):
        mon = ExchangeReserveMonitor(db_path=tmp_db, neutral_threshold_pct=0.1)
        now = time.time()
        mon.update_reserve("binance", "BTC", 500_000, timestamp=now - 3600)
        mon.update_reserve("binance", "BTC", 550_000, timestamp=now)  # 10% increase
        rc = mon.get_reserve_change("BTC", lookback_hours=2)
        assert rc.direction == "inflow"
        assert rc.signal_bias < 0  # Inflow = bearish

    def test_get_reserve_change_outflow(self, tmp_db):
        mon = ExchangeReserveMonitor(db_path=tmp_db, neutral_threshold_pct=0.1)
        now = time.time()
        mon.update_reserve("kraken", "ETH", 10_000_000, timestamp=now - 3600)
        mon.update_reserve("kraken", "ETH", 9_000_000, timestamp=now)  # 10% decrease
        rc = mon.get_reserve_change("ETH", lookback_hours=2)
        assert rc.direction == "outflow"
        assert rc.signal_bias > 0  # Outflow = bullish

    def test_exchanges_with_inflow_outflow(self, tmp_db):
        mon = ExchangeReserveMonitor(db_path=tmp_db, neutral_threshold_pct=0.1)
        now = time.time()
        mon.update_reserve("binance", "BTC", 500_000, timestamp=now - 3600)
        mon.update_reserve("binance", "BTC", 550_000, timestamp=now)
        mon.update_reserve("kraken", "BTC", 100_000, timestamp=now - 3600)
        mon.update_reserve("kraken", "BTC", 80_000, timestamp=now)
        rc = mon.get_reserve_change("BTC", lookback_hours=2)
        assert "binance" in rc.exchanges_with_inflow
        assert "kraken" in rc.exchanges_with_outflow

    def test_get_all_assets_summary(self, tmp_db):
        mon = ExchangeReserveMonitor(db_path=tmp_db)
        now = time.time()
        mon.update_reserve("binance", "BTC", 500_000, timestamp=now - 3600)
        mon.update_reserve("binance", "BTC", 510_000, timestamp=now)
        mon.update_reserve("binance", "ETH", 1_000_000, timestamp=now - 3600)
        mon.update_reserve("binance", "ETH", 900_000, timestamp=now)
        summary = mon.get_all_assets_summary()
        assert "BTC" in summary
        assert "ETH" in summary

    def test_flush_and_load(self, tmp_db):
        mon = ExchangeReserveMonitor(db_path=tmp_db)
        now = time.time()
        mon.update_reserve("binance", "BTC", 500_000, timestamp=now)
        mon.flush()

        # Create new monitor, load from db
        mon2 = ExchangeReserveMonitor(db_path=tmp_db)
        count = mon2.load_history()
        assert count >= 1

    def test_reserve_change_dataclass(self):
        rc = ReserveChange(
            asset="BTC", net_change=-1000, pct_change=-2.0,
            direction="outflow",
            exchanges_with_inflow=[], exchanges_with_outflow=["binance"],
            signal_bias=0.2,
        )
        assert rc.asset == "BTC"
        assert rc.direction == "outflow"


# ===================================================================
# Confidence Calibrator
# ===================================================================

class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        return str(tmp_path / "test_calibration.db")

    def test_construction(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db)
        assert Path(tmp_db).exists()
        assert cal.tracked_models == []

    def test_record_prediction(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db)
        cal.record_prediction("model_a", 0.8, True)
        assert "model_a" in cal.tracked_models

    def test_record_clamps_confidence(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db)
        cal.record_prediction("model_a", 1.5, True)  # Should clamp to 1.0
        cal.record_prediction("model_a", -0.3, False)  # Should clamp to 0.0
        preds = cal._predictions["model_a"]
        assert preds[0][0] == 1.0
        assert preds[1][0] == 0.0

    def test_get_calibration_no_data(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db)
        report = cal.get_calibration("nonexistent")
        assert isinstance(report, CalibrationReport)
        assert report.total_predictions == 0
        assert report.ece == 0.0

    def test_get_calibration_with_data(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db)
        # Record many predictions — perfectly calibrated model
        for i in range(100):
            conf = (i % 10) / 10.0 + 0.05
            outcome = (i % 10) < int(conf * 10)
            cal.record_prediction("perfect", conf, outcome)
        report = cal.get_calibration("perfect")
        assert report.total_predictions == 100
        assert len(report.bins) > 0

    def test_get_calibration_overconfident(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db)
        # Model always says 0.9 but is only right 50% of the time
        for i in range(100):
            cal.record_prediction("overconf", 0.9, i % 2 == 0)
        report = cal.get_calibration("overconf")
        assert report.overconfident is True

    def test_calibrate_insufficient_data(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db, min_samples_for_calibration=50)
        cal.record_prediction("model_a", 0.7, True)
        # Only 1 sample, below threshold — should return raw value
        result = cal.calibrate("model_a", 0.8)
        assert result == 0.8

    def test_calibrate_with_enough_data(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db, min_samples_for_calibration=10)
        # Overconfident model: says 0.9 but only right 50%
        for i in range(50):
            cal.record_prediction("overconf", 0.9, i % 2 == 0)
        result = cal.calibrate("overconf", 0.9)
        assert 0.0 <= result <= 1.0
        # Should adjust down from 0.9 toward 0.5
        assert result < 0.85

    def test_calibrate_clamps_input(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db, min_samples_for_calibration=10)
        for i in range(20):
            cal.record_prediction("m", 0.5, True)
        result = cal.calibrate("m", 1.5)
        assert 0.0 <= result <= 1.0

    def test_reliability_diagram_data(self, tmp_db):
        cal = ConfidenceCalibrator(db_path=tmp_db)
        for i in range(50):
            cal.record_prediction("diag", i / 50.0, i % 3 == 0)
        data = cal.get_reliability_diagram_data("diag")
        assert "predicted_avgs" in data
        assert "actual_rates" in data
        assert "ece" in data
        assert data["perfect_line"] == [0.0, 1.0]

    def test_persistence_across_instances(self, tmp_db):
        cal1 = ConfidenceCalibrator(db_path=tmp_db)
        for i in range(10):
            cal1.record_prediction("persist_test", 0.7, True)

        cal2 = ConfidenceCalibrator(db_path=tmp_db)
        assert "persist_test" in cal2.tracked_models
        assert len(cal2._predictions["persist_test"]) == 10

    def test_calibration_report_dataclass(self):
        report = CalibrationReport(
            model_name="test",
            bins=[{"predicted_avg": 0.5, "actual_rate": 0.5, "count": 10}],
            ece=0.05,
            overconfident=False,
            underconfident=False,
        )
        assert report.model_name == "test"
        assert report.ece == 0.05


# ===================================================================
# Anti-Gaming Layer
# ===================================================================

class TestAntiGamingLayer:
    """Tests for AntiGamingLayer."""

    def test_construction(self):
        layer = AntiGamingLayer()
        assert layer._iceberg_prob == 0.3

    def test_construction_custom_iceberg(self):
        layer = AntiGamingLayer(iceberg_probability=0.7)
        assert layer._iceberg_prob == 0.7

    def test_randomize_order_size_in_range(self):
        layer = AntiGamingLayer()
        base = 100.0
        results = [layer.randomize_order_size(base, max_deviation_pct=10) for _ in range(100)]
        for r in results:
            assert 89.0 <= r <= 111.0  # small margin for float precision

    def test_randomize_order_size_zero(self):
        layer = AntiGamingLayer()
        assert layer.randomize_order_size(0.0) == 0.0

    def test_randomize_order_size_negative(self):
        layer = AntiGamingLayer()
        result = layer.randomize_order_size(-10.0)
        assert result == -10.0  # Passthrough for non-positive

    def test_randomize_order_size_varies(self):
        layer = AntiGamingLayer()
        results = set()
        for _ in range(20):
            results.add(round(layer.randomize_order_size(1000.0), 2))
        assert len(results) > 1  # Should produce different values

    def test_randomize_timing_non_negative(self):
        layer = AntiGamingLayer()
        for _ in range(50):
            delay = layer.randomize_timing(0.0, max_jitter_s=5.0)
            assert delay >= 0.0

    def test_randomize_timing_range(self):
        layer = AntiGamingLayer()
        base = 1.0
        for _ in range(50):
            delay = layer.randomize_timing(base, max_jitter_s=3.0)
            assert 1.0 <= delay <= 4.01

    def test_get_random_venue_single(self):
        layer = AntiGamingLayer()
        assert layer.get_random_venue(["kraken"]) == "kraken"

    def test_get_random_venue_multiple(self):
        layer = AntiGamingLayer()
        venues = ["kraken", "coinbase", "binance"]
        results = set()
        for _ in range(50):
            results.add(layer.get_random_venue(venues))
        assert len(results) >= 2  # Should use more than one venue

    def test_get_random_venue_weighted(self):
        layer = AntiGamingLayer()
        venues = ["kraken", "coinbase"]
        weights = [100.0, 0.001]
        results = [layer.get_random_venue(venues, weights) for _ in range(100)]
        # Kraken should dominate
        kraken_count = results.count("kraken")
        assert kraken_count > 80

    def test_get_random_venue_empty_raises(self):
        layer = AntiGamingLayer()
        with pytest.raises(ValueError):
            layer.get_random_venue([])

    def test_get_random_venue_mismatched_weights_raises(self):
        layer = AntiGamingLayer()
        with pytest.raises(ValueError):
            layer.get_random_venue(["a", "b"], [1.0])

    def test_should_split_order_below_threshold(self):
        layer = AntiGamingLayer()
        should, count = layer.should_split_order(100.0, threshold_usd=500.0)
        assert should is False
        assert count == 1

    def test_should_split_order_above_threshold(self):
        layer = AntiGamingLayer()
        should, count = layer.should_split_order(1000.0, threshold_usd=500.0)
        assert should is True
        assert 2 <= count <= 3

    def test_should_split_order_large(self):
        layer = AntiGamingLayer()
        should, count = layer.should_split_order(50000.0, threshold_usd=500.0)
        assert should is True
        assert 5 <= count <= 8

    def test_get_execution_mask(self):
        layer = AntiGamingLayer()
        mask = layer.get_execution_mask(
            base_size=1000.0,
            preferred_venues=["kraken", "coinbase"],
        )
        assert isinstance(mask, ExecutionMask)
        assert 0.85 <= mask.size_multiplier <= 1.15
        assert mask.delay_s >= 0
        assert mask.venue in ["kraken", "coinbase"]
        assert mask.split_count >= 1
        assert isinstance(mask.use_iceberg, bool)

    def test_get_execution_mask_defaults(self):
        layer = AntiGamingLayer()
        mask = layer.get_execution_mask()
        assert mask.venue == "primary"

    def test_execution_mask_dataclass(self):
        mask = ExecutionMask(
            size_multiplier=0.95,
            delay_s=1.5,
            venue="kraken",
            split_count=3,
            use_iceberg=True,
        )
        assert mask.size_multiplier == 0.95
        assert mask.use_iceberg is True

    def test_secure_uniform_range(self):
        layer = AntiGamingLayer()
        for _ in range(100):
            val = layer._secure_uniform(0.0, 1.0)
            assert 0.0 <= val <= 1.0

    def test_secure_uniform_degenerate(self):
        layer = AntiGamingLayer()
        assert layer._secure_uniform(5.0, 5.0) == 5.0
        assert layer._secure_uniform(5.0, 3.0) == 5.0  # lo >= hi


# ===================================================================
# Integration / cross-module sanity
# ===================================================================

class TestIntegration:
    """Light integration tests to confirm modules interoperate."""

    def test_all_modules_importable(self):
        """Confirm all modules import cleanly."""
        from data.alternative.google_trends_tracker import GoogleTrendsTracker
        from data.onchain.stablecoin_flow_tracker import StablecoinFlowTracker
        from data.onchain.exchange_reserve_monitor import ExchangeReserveMonitor
        from ml.confidence_calibrator import ConfidenceCalibrator
        from risk.anti_gaming_layer import AntiGamingLayer
        assert all([
            GoogleTrendsTracker, StablecoinFlowTracker,
            ExchangeReserveMonitor, ConfidenceCalibrator, AntiGamingLayer,
        ])

    def test_stablecoin_and_reserve_together(self, tmp_path):
        """Confirm stablecoin + reserve monitors can coexist."""
        sc = StablecoinFlowTracker()
        rm = ExchangeReserveMonitor(db_path=tmp_path / "res.db")
        now = time.time()
        sc.update_supply("USDT", 100e9, timestamp=now - 3600)
        sc.update_supply("USDT", 105e9, timestamp=now)
        rm.update_reserve("binance", "BTC", 500_000, timestamp=now - 3600)
        rm.update_reserve("binance", "BTC", 480_000, timestamp=now)

        sc_flow = sc.get_flow("USDT", lookback_hours=2)
        rm_change = rm.get_reserve_change("BTC", lookback_hours=2)

        # Stablecoin mint + exchange outflow = double bullish
        assert sc_flow.signal_bias > 0
        assert rm_change.signal_bias > 0
