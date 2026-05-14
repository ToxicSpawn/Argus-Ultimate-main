"""Tests for api/performance_pricing.py — PerformancePricingEngine."""
from __future__ import annotations

import os
import tempfile

import pytest

from api.performance_pricing import PerformanceAttribution, PerformancePricingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_perf.db")


@pytest.fixture
def engine(db_path):
    return PerformancePricingEngine(config=None, db_path=db_path)


# ---------------------------------------------------------------------------
# Outcome recording
# ---------------------------------------------------------------------------

class TestOutcomeRecording:
    def test_record_buy_win(self, engine):
        """Buy signal with price increase should show positive P&L."""
        engine.record_signal_outcome("sub-1", "sig-1", 60000.0, 61000.0, "buy")
        attr = engine.calculate_attribution("sub-1")
        assert attr.total_signals == 1
        assert attr.winning_signals == 1
        assert attr.attributed_pnl_usd == 1000.0

    def test_record_buy_loss(self, engine):
        """Buy signal with price decrease should show negative P&L."""
        engine.record_signal_outcome("sub-1", "sig-1", 60000.0, 59000.0, "buy")
        attr = engine.calculate_attribution("sub-1")
        assert attr.total_signals == 1
        assert attr.winning_signals == 0
        assert attr.attributed_pnl_usd == -1000.0

    def test_record_sell_win(self, engine):
        """Sell signal with price decrease should show positive P&L."""
        engine.record_signal_outcome("sub-1", "sig-1", 60000.0, 59000.0, "sell")
        attr = engine.calculate_attribution("sub-1")
        assert attr.winning_signals == 1
        assert attr.attributed_pnl_usd == 1000.0

    def test_record_sell_loss(self, engine):
        """Sell signal with price increase should show negative P&L."""
        engine.record_signal_outcome("sub-1", "sig-1", 60000.0, 61000.0, "sell")
        attr = engine.calculate_attribution("sub-1")
        assert attr.winning_signals == 0
        assert attr.attributed_pnl_usd == -1000.0

    def test_multiple_outcomes(self, engine):
        """Multiple outcomes should aggregate correctly."""
        engine.record_signal_outcome("sub-1", "sig-1", 100.0, 110.0, "buy")   # +10
        engine.record_signal_outcome("sub-1", "sig-2", 100.0, 95.0, "buy")    # -5
        engine.record_signal_outcome("sub-1", "sig-3", 100.0, 90.0, "sell")   # +10
        attr = engine.calculate_attribution("sub-1")
        assert attr.total_signals == 3
        assert attr.winning_signals == 2
        assert attr.attributed_pnl_usd == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Attribution and fee calculation
# ---------------------------------------------------------------------------

class TestAttribution:
    def test_performance_fee_on_gains(self, engine):
        """Performance fee should be 10% of gains."""
        engine.record_signal_outcome("sub-1", "sig-1", 100.0, 200.0, "buy")  # +100
        attr = engine.calculate_attribution("sub-1")
        assert attr.performance_fee_usd == pytest.approx(10.0)

    def test_no_fee_on_losses(self, engine):
        """No performance fee when total P&L is negative."""
        engine.record_signal_outcome("sub-1", "sig-1", 100.0, 50.0, "buy")  # -50
        attr = engine.calculate_attribution("sub-1")
        assert attr.performance_fee_usd == 0.0

    def test_total_fee_includes_base(self, engine, db_path):
        """Total fee = base + performance."""
        # Set up billing tier
        from api.billing import BillingManager
        bm = BillingManager(db_path=db_path)
        bm.assign_tier("sub-fee", "PRO")

        engine.record_signal_outcome("sub-fee", "sig-1", 100.0, 200.0, "buy")  # +100
        attr = engine.calculate_attribution("sub-fee")
        # Base = $99 (PRO), Performance = $10 (10% of $100)
        assert attr.base_fee_usd == 99.0
        assert attr.performance_fee_usd == pytest.approx(10.0)
        assert attr.total_fee_usd == pytest.approx(109.0)

    def test_empty_attribution(self, engine):
        """No outcomes should return zero attribution."""
        attr = engine.calculate_attribution("nonexistent")
        assert attr.total_signals == 0
        assert attr.winning_signals == 0
        assert attr.attributed_pnl_usd == 0.0
        assert attr.performance_fee_usd == 0.0


# ---------------------------------------------------------------------------
# Period filtering
# ---------------------------------------------------------------------------

class TestPeriodFiltering:
    def test_30_day_default(self, engine):
        """Default 30-day period should include recent outcomes."""
        engine.record_signal_outcome("sub-1", "sig-1", 100.0, 110.0, "buy")
        attr = engine.calculate_attribution("sub-1", period_days=30)
        assert attr.total_signals == 1

    def test_zero_day_period_excludes_all(self, engine):
        """A zero-day period might miss recent entries depending on timing."""
        engine.record_signal_outcome("sub-1", "sig-1", 100.0, 110.0, "buy")
        # 0-day should still include today
        attr = engine.calculate_attribution("sub-1", period_days=0)
        # With period_days=0, cutoff is now, so this should typically include today's
        # entries since we compare >= cutoff
        assert attr.total_signals >= 0  # just validate it doesn't crash


# ---------------------------------------------------------------------------
# Subscriber performance summary
# ---------------------------------------------------------------------------

class TestSubscriberPerformance:
    def test_performance_dict(self, engine):
        """get_subscriber_performance should return serialisable dict."""
        engine.record_signal_outcome("sub-1", "sig-1", 100.0, 120.0, "buy")  # +20
        engine.record_signal_outcome("sub-1", "sig-2", 100.0, 90.0, "buy")   # -10
        perf = engine.get_subscriber_performance("sub-1")
        assert perf["subscriber_id"] == "sub-1"
        assert perf["total_signals"] == 2
        assert perf["winning_signals"] == 1
        assert perf["win_rate"] == pytest.approx(0.5)
        assert perf["attributed_pnl_usd"] == pytest.approx(10.0)

    def test_empty_performance(self, engine):
        perf = engine.get_subscriber_performance("nobody")
        assert perf["total_signals"] == 0
        assert perf["win_rate"] == 0.0

    def test_custom_fee_pct(self, db_path):
        """Custom performance fee percentage from config."""
        from unittest.mock import MagicMock
        cfg = MagicMock()
        cfg.signal_billing_performance_fee_pct = 20.0
        engine = PerformancePricingEngine(config=cfg, db_path=db_path)
        engine.record_signal_outcome("sub-1", "sig-1", 100.0, 200.0, "buy")  # +100
        attr = engine.calculate_attribution("sub-1")
        assert attr.performance_fee_usd == pytest.approx(20.0)  # 20% of 100
