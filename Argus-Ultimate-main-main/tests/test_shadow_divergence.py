#!/usr/bin/env python3
"""Tests for monitoring.shadow_divergence — divergence tracking and alerts."""

from __future__ import annotations

import time
import pytest

from monitoring.shadow_divergence import ShadowDivergenceTracker, DivergenceReport
from ops.shadow_executor import ShadowTrade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    return ShadowDivergenceTracker(config=None)


@pytest.fixture
def sensitive_tracker():
    """Tracker with low alert threshold."""
    cfg = {"shadow_execution": {"divergence_alert_threshold_pct": 1.0}}
    return ShadowDivergenceTracker(config=cfg)


def _real_trade(symbol="BTC/AUD", side="buy", qty=1.0, price=100_000.0, filled=True, ts=None):
    return {
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "price": price,
        "filled": filled,
        "timestamp": ts or time.time(),
    }


def _shadow_trade(symbol="BTC/AUD", side="buy", qty=1.0, price=100_000.0, filled=True, ts=None):
    return ShadowTrade(
        symbol=symbol,
        side=side,
        quantity=qty,
        hypothetical_price=price,
        timestamp=ts or time.time(),
        would_have_filled=filled,
        reason="shadow_fill",
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class TestRecording:

    def test_record_real_trade(self, tracker):
        tracker.record_real_trade(_real_trade())
        assert tracker.real_count == 1

    def test_record_shadow_trade_from_dataclass(self, tracker):
        tracker.record_shadow_trade(_shadow_trade())
        assert tracker.shadow_count == 1

    def test_record_shadow_trade_from_dict(self, tracker):
        tracker.record_shadow_trade({
            "symbol": "BTC/AUD",
            "side": "buy",
            "quantity": 1.0,
            "price": 100_000.0,
            "filled": True,
            "timestamp": time.time(),
        })
        assert tracker.shadow_count == 1

    def test_clear(self, tracker):
        tracker.record_real_trade(_real_trade())
        tracker.record_shadow_trade(_shadow_trade())
        tracker.clear()
        assert tracker.real_count == 0
        assert tracker.shadow_count == 0


# ---------------------------------------------------------------------------
# Divergence calculation
# ---------------------------------------------------------------------------

class TestDivergenceCalculation:

    def test_no_trades_returns_zero_divergence(self, tracker):
        report = tracker.calculate_divergence()
        assert isinstance(report, DivergenceReport)
        assert report.fill_rate_divergence_pct == 0.0
        assert report.pnl_divergence_usd == 0.0

    def test_identical_trades_minimal_divergence(self, tracker):
        ts = time.time()
        tracker.record_real_trade(_real_trade(ts=ts))
        tracker.record_shadow_trade(_shadow_trade(ts=ts))
        report = tracker.calculate_divergence()
        assert report.fill_rate_divergence_pct == 0.0
        assert report.timing_divergence_ms == pytest.approx(0.0, abs=1.0)

    def test_fill_rate_divergence(self, tracker):
        # Real: 2 filled / 2 total = 100%
        tracker.record_real_trade(_real_trade(filled=True))
        tracker.record_real_trade(_real_trade(filled=True))
        # Shadow: 1 filled / 2 total = 50%
        tracker.record_shadow_trade(_shadow_trade(filled=True))
        tracker.record_shadow_trade(_shadow_trade(filled=False))
        report = tracker.calculate_divergence()
        assert report.fill_rate_divergence_pct == pytest.approx(50.0)

    def test_pnl_divergence(self, tracker):
        # Real: buy 1 BTC at 100k, sell 1 BTC at 110k -> pnl = 10k
        tracker.record_real_trade(_real_trade(side="buy", price=100_000.0))
        tracker.record_real_trade(_real_trade(side="sell", price=110_000.0))
        # Shadow: buy 1 BTC at 100k, sell 1 BTC at 105k -> pnl = 5k
        tracker.record_shadow_trade(_shadow_trade(side="buy", price=100_000.0))
        tracker.record_shadow_trade(_shadow_trade(side="sell", price=105_000.0))
        report = tracker.calculate_divergence()
        assert report.pnl_divergence_usd == pytest.approx(5000.0)

    def test_timing_divergence(self, tracker):
        ts_real = time.time()
        ts_shadow = ts_real + 0.1  # 100ms later
        tracker.record_real_trade(_real_trade(ts=ts_real))
        tracker.record_shadow_trade(_shadow_trade(ts=ts_shadow))
        report = tracker.calculate_divergence()
        assert report.timing_divergence_ms == pytest.approx(100.0, abs=5.0)


# ---------------------------------------------------------------------------
# Alert triggering
# ---------------------------------------------------------------------------

class TestAlerts:

    def test_no_alert_when_below_threshold(self, tracker):
        ts = time.time()
        tracker.record_real_trade(_real_trade(ts=ts))
        tracker.record_shadow_trade(_shadow_trade(ts=ts))
        report = tracker.calculate_divergence()
        assert report.alert_triggered is False

    def test_alert_on_high_fill_divergence(self, sensitive_tracker):
        # 100% vs 0% fill rate -> 100% divergence > 1% threshold
        sensitive_tracker.record_real_trade(_real_trade(filled=True))
        sensitive_tracker.record_shadow_trade(_shadow_trade(filled=False))
        report = sensitive_tracker.calculate_divergence()
        assert report.alert_triggered is True

    def test_alert_on_high_pnl_divergence(self, sensitive_tracker):
        sensitive_tracker.record_real_trade(_real_trade(side="sell", price=200_000.0))
        sensitive_tracker.record_shadow_trade(_shadow_trade(side="sell", price=100_000.0))
        report = sensitive_tracker.calculate_divergence()
        assert report.alert_triggered is True


# ---------------------------------------------------------------------------
# DivergenceReport dataclass
# ---------------------------------------------------------------------------

class TestDivergenceReport:

    def test_dataclass_fields(self):
        report = DivergenceReport(
            fill_rate_divergence_pct=5.0,
            pnl_divergence_usd=100.0,
            timing_divergence_ms=50.0,
            alert_triggered=False,
        )
        assert report.fill_rate_divergence_pct == 5.0
        assert report.alert_triggered is False
