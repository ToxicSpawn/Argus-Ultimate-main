"""
Tests for advanced execution and risk modules (March 2026 batch).

Covers:
    1. OrderFlowImbalancePredictor  (data/orderbook/order_flow_imbalance.py)
    2. FillProbabilityModel         (execution/fill_probability_model.py)
    3. BlackSwanDetector            (risk/black_swan_detector.py)
    4. PartialFillManager           (execution/partial_fill_manager.py)
    5. ContagionModel               (risk/contagion_model.py)

50+ tests total.
"""
from __future__ import annotations

import math
import os
import tempfile
import time
import pytest


# ---------------------------------------------------------------------------
# 1. OrderFlowImbalancePredictor
# ---------------------------------------------------------------------------

class TestOrderFlowImbalancePredictor:
    """Tests for OrderFlowImbalancePredictor."""

    def _make(self, **kwargs):
        from data.orderbook.order_flow_imbalance import OrderFlowImbalancePredictor
        return OrderFlowImbalancePredictor(**kwargs)

    def _bids(self, mid=65000, levels=10, base_size=1.0):
        return [(mid - i * 10, base_size) for i in range(levels)]

    def _asks(self, mid=65010, levels=10, base_size=1.0):
        return [(mid + i * 10, base_size) for i in range(levels)]

    def test_init(self):
        p = self._make()
        assert p.get_symbols() == []

    def test_update_and_get_imbalance_balanced(self):
        p = self._make()
        p.update_book("BTC/USD", self._bids(), self._asks())
        imb = p.get_imbalance("BTC/USD", depth_levels=5)
        assert -0.01 <= imb <= 0.01, f"Balanced book should be ~0, got {imb}"

    def test_imbalance_buy_pressure(self):
        p = self._make()
        bids = [(65000 - i * 10, 5.0) for i in range(10)]  # large bids
        asks = [(65010 + i * 10, 0.5) for i in range(10)]   # small asks
        p.update_book("BTC/USD", bids, asks)
        imb = p.get_imbalance("BTC/USD", depth_levels=5)
        assert imb > 0.5, f"Should show buy pressure, got {imb}"

    def test_imbalance_sell_pressure(self):
        p = self._make()
        bids = [(65000 - i * 10, 0.3) for i in range(10)]
        asks = [(65010 + i * 10, 4.0) for i in range(10)]
        p.update_book("BTC/USD", bids, asks)
        imb = p.get_imbalance("BTC/USD", depth_levels=5)
        assert imb < -0.5, f"Should show sell pressure, got {imb}"

    def test_imbalance_no_data(self):
        p = self._make()
        assert p.get_imbalance("UNKNOWN") == 0.0

    def test_imbalance_range(self):
        p = self._make()
        p.update_book("X", [(100, 999)], [(101, 0.001)])
        imb = p.get_imbalance("X", depth_levels=1)
        assert -1.0 <= imb <= 1.0

    def test_spoofing_not_detected_balanced(self):
        p = self._make()
        p.update_book("BTC/USD", self._bids(), self._asks())
        p.update_book("BTC/USD", self._bids(), self._asks())
        alert = p.detect_spoofing("BTC/USD")
        assert not alert.detected

    def test_spoofing_detected_large_top_bid(self):
        p = self._make()
        # Normal book
        p.update_book("BTC/USD", self._bids(), self._asks())
        # Book with huge top bid
        bids = [(65000, 50.0)] + [(65000 - (i + 1) * 10, 1.0) for i in range(9)]
        p.update_book("BTC/USD", bids, self._asks())
        # Now remove it
        p.update_book("BTC/USD", self._bids(base_size=1.0), self._asks())
        alert = p.detect_spoofing("BTC/USD")
        assert alert.detected
        assert alert.side == "bid"

    def test_spoofing_not_enough_history(self):
        p = self._make()
        p.update_book("BTC/USD", self._bids(), self._asks())
        alert = p.detect_spoofing("BTC/USD")
        assert not alert.detected

    def test_iceberg_detected(self):
        p = self._make()
        bids = [(65000, 0.1)] + [(65000 - (i + 1) * 10, 1.0) for i in range(9)]
        p.update_book("BTC/USD", bids, self._asks())
        # Many fills at 65000 with sizes much larger than visible 0.1
        fills = [{"price": 65000, "size": 0.5} for _ in range(5)]
        assert p.detect_iceberg("BTC/USD", fills) is True

    def test_iceberg_not_detected_few_fills(self):
        p = self._make()
        p.update_book("BTC/USD", self._bids(), self._asks())
        fills = [{"price": 65000, "size": 0.1}]
        assert p.detect_iceberg("BTC/USD", fills) is False

    def test_iceberg_uniform_fills(self):
        p = self._make()
        p.update_book("BTC/USD", self._bids(base_size=0.01), self._asks())
        fills = [{"price": 65000, "size": 1.0} for _ in range(6)]
        assert p.detect_iceberg("BTC/USD", fills) is True

    def test_prediction_neutral_insufficient_data(self):
        p = self._make()
        pred = p.get_short_term_prediction("BTC/USD")
        assert pred["direction"] == "neutral"
        assert pred["confidence"] == 0.0

    def test_prediction_after_updates(self):
        p = self._make()
        for i in range(5):
            bids = [(65000 - i * 10, 2.0 + i * 0.5) for _ in range(5)]
            asks = [(65010 + i * 10, 1.0) for _ in range(5)]
            p.update_book("BTC/USD", bids, asks)
        pred = p.get_short_term_prediction("BTC/USD")
        assert pred["direction"] in ("up", "down", "neutral")
        assert 0.0 <= pred["confidence"] <= 1.0
        assert -1.0 <= pred["imbalance_score"] <= 1.0

    def test_snapshot_count(self):
        p = self._make(history_depth=50)
        for _ in range(10):
            p.update_book("X", [(100, 1)], [(101, 1)])
        assert p.get_snapshot_count("X") == 10

    def test_history_depth_capped(self):
        p = self._make(history_depth=20)
        for _ in range(50):
            p.update_book("X", [(100, 1)], [(101, 1)])
        assert p.get_snapshot_count("X") == 20


# ---------------------------------------------------------------------------
# 2. FillProbabilityModel
# ---------------------------------------------------------------------------

class TestFillProbabilityModel:
    """Tests for FillProbabilityModel."""

    def _make(self, **kwargs):
        from execution.fill_probability_model import FillProbabilityModel
        tmp = tempfile.mktemp(suffix=".db")
        return FillProbabilityModel(db_path=tmp, min_samples=5, **kwargs)

    def test_init(self):
        m = self._make()
        assert m is not None

    def test_record_and_stats_empty(self):
        m = self._make()
        stats = m.get_fill_stats("BTC/USD")
        assert stats["total_orders"] == 0

    def test_record_single(self):
        m = self._make()
        m.record_limit_order("BTC/USD", 64950, "buy", 65000, 3.0, True, 800)
        stats = m.get_fill_stats("BTC/USD")
        assert stats["total_orders"] == 1
        assert stats["fill_rate"] == 1.0

    def test_fill_rate_mixed(self):
        m = self._make()
        for i in range(10):
            m.record_limit_order(
                "BTC/USD", 64950 + i, "buy", 65000, 3.0,
                filled=(i < 7), time_to_fill_ms=500 if i < 7 else None,
            )
        stats = m.get_fill_stats("BTC/USD")
        assert stats["total_orders"] == 10
        assert abs(stats["fill_rate"] - 0.7) < 0.01

    def test_predict_sigmoid_fallback(self):
        m = self._make()
        # No training data → sigmoid fallback
        prob = m.predict_fill_probability("BTC/USD", 64990, "buy", 65000, 3.0)
        assert 0.0 <= prob <= 1.0

    def test_predict_aggressive_higher_prob(self):
        m = self._make()
        # Aggressive (above mid for buy) should have higher prob than passive
        prob_aggressive = m.predict_fill_probability("BTC/USD", 65010, "buy", 65000, 5.0)
        prob_passive = m.predict_fill_probability("BTC/USD", 64900, "buy", 65000, 5.0)
        assert prob_aggressive > prob_passive

    def test_optimal_limit_price_buy(self):
        m = self._make()
        price = m.get_optimal_limit_price("BTC/USD", "buy", 65000, target_fill_prob=0.7)
        assert price > 0
        # For buy, optimal should be near or above mid for high fill prob
        assert price > 64500

    def test_optimal_limit_price_sell(self):
        m = self._make()
        price = m.get_optimal_limit_price("BTC/USD", "sell", 65000, target_fill_prob=0.7)
        assert price > 0

    def test_stats_by_side(self):
        m = self._make()
        m.record_limit_order("BTC/USD", 64950, "buy", 65000, 3.0, True, 500)
        m.record_limit_order("BTC/USD", 65050, "sell", 65000, 3.0, False)
        buy_stats = m.get_fill_stats("BTC/USD", side="buy")
        sell_stats = m.get_fill_stats("BTC/USD", side="sell")
        assert buy_stats["fill_rate"] == 1.0
        assert sell_stats["fill_rate"] == 0.0

    def test_distance_bps_buy(self):
        from execution.fill_probability_model import FillProbabilityModel
        d = FillProbabilityModel._compute_distance_bps(65050, "buy", 65000)
        assert d > 0  # above mid = aggressive for buy

    def test_distance_bps_sell(self):
        from execution.fill_probability_model import FillProbabilityModel
        d = FillProbabilityModel._compute_distance_bps(64950, "sell", 65000)
        assert d > 0  # below mid = aggressive for sell


# ---------------------------------------------------------------------------
# 3. BlackSwanDetector
# ---------------------------------------------------------------------------

class TestBlackSwanDetector:
    """Tests for BlackSwanDetector."""

    def _make(self, **kwargs):
        from risk.black_swan_detector import BlackSwanDetector
        return BlackSwanDetector(**kwargs)

    def test_init(self):
        d = self._make()
        assert d.get_system_risk_level() == "green"

    def test_no_anomaly_normal_data(self):
        d = self._make()
        for i in range(50):
            d.update_metrics("BTC/USD", price=65000 + i * 5, volume=100 + i)
        report = d.detect_anomalies("BTC/USD")
        assert not report.anomaly_detected
        assert report.severity == "low"
        assert report.recommendation == "monitor"

    def test_price_spike_anomaly(self):
        d = self._make()
        # Normal prices
        for i in range(100):
            d.update_metrics("BTC/USD", price=65000 + (i % 10) * 5, volume=100)
        # Massive spike
        d.update_metrics("BTC/USD", price=75000, volume=100)
        report = d.detect_anomalies("BTC/USD")
        assert report.anomaly_detected
        assert any("price_zscore" in t for t in report.triggers)

    def test_volume_spike_anomaly(self):
        d = self._make()
        for i in range(100):
            d.update_metrics("BTC/USD", price=65000, volume=100 + (i % 5))
        d.update_metrics("BTC/USD", price=65000, volume=100000)
        report = d.detect_anomalies("BTC/USD")
        assert report.anomaly_detected
        assert any("volume_zscore" in t for t in report.triggers)

    def test_funding_rate_extreme(self):
        d = self._make()
        for i in range(20):
            d.update_metrics("BTC/USD", price=65000, volume=100, funding_rate=0.0001)
        d.update_metrics("BTC/USD", price=65000, volume=100, funding_rate=0.05)
        report = d.detect_anomalies("BTC/USD")
        assert report.anomaly_detected
        assert any("funding_rate" in t for t in report.triggers)

    def test_oi_drop_anomaly(self):
        d = self._make()
        for i in range(20):
            d.update_metrics("BTC/USD", price=65000, volume=100, oi_change_pct=0.5)
        d.update_metrics("BTC/USD", price=65000, volume=100, oi_change_pct=-15.0)
        report = d.detect_anomalies("BTC/USD")
        assert report.anomaly_detected
        assert any("oi_drop" in t for t in report.triggers)

    def test_system_risk_green(self):
        d = self._make()
        assert d.get_system_risk_level() == "green"

    def test_system_risk_escalation(self):
        d = self._make()
        # Create anomalies in multiple symbols
        for sym in ["BTC/USD", "ETH/USD", "SOL/USD"]:
            for i in range(100):
                d.update_metrics(sym, price=1000 + (i % 5), volume=100)
            d.update_metrics(sym, price=5000, volume=100)
            d.detect_anomalies(sym)
        level = d.get_system_risk_level()
        assert level in ("yellow", "orange", "red")

    def test_insufficient_data(self):
        d = self._make()
        d.update_metrics("BTC/USD", price=65000, volume=100)
        report = d.detect_anomalies("BTC/USD")
        assert not report.anomaly_detected

    def test_anomaly_score_range(self):
        d = self._make()
        for i in range(100):
            d.update_metrics("BTC/USD", price=65000, volume=100)
        d.update_metrics("BTC/USD", price=99999, volume=999999)
        report = d.detect_anomalies("BTC/USD")
        assert 0.0 <= report.anomaly_score <= 1.0

    def test_get_active_anomalies(self):
        d = self._make()
        # Add some variance so std > 0, then spike
        import random
        rng = random.Random(42)
        for i in range(100):
            d.update_metrics("BTC/USD", price=65000 + rng.uniform(-100, 100), volume=100 + rng.uniform(-10, 10))
        d.update_metrics("BTC/USD", price=99999, volume=100)
        d.detect_anomalies("BTC/USD")
        actives = d.get_active_anomalies()
        assert len(actives) >= 1


# ---------------------------------------------------------------------------
# 4. PartialFillManager
# ---------------------------------------------------------------------------

class TestPartialFillManager:
    """Tests for PartialFillManager."""

    def _make(self, **kwargs):
        from execution.partial_fill_manager import PartialFillManager
        tmp = tempfile.mktemp(suffix=".db")
        return PartialFillManager(db_path=tmp, **kwargs)

    def test_init(self):
        m = self._make()
        assert m is not None

    def test_wait_when_mostly_filled_and_early(self):
        m = self._make()
        action = m.on_partial_fill(
            "o1", "BTC/USD", "buy",
            filled_qty=0.9, remaining_qty=0.1,
            fill_price=65000, elapsed_ms=2000,
        )
        assert action.action == "wait"

    def test_cancel_replace_low_fill_late(self):
        m = self._make()
        action = m.on_partial_fill(
            "o2", "BTC/USD", "buy",
            filled_qty=0.3, remaining_qty=0.7,
            fill_price=65000, elapsed_ms=35000,
        )
        assert action.action == "cancel_replace"
        assert action.new_price is not None
        assert action.new_price > 65000  # buy → more aggressive = higher

    def test_market_sweep_high_urgency(self):
        m = self._make()
        action = m.on_partial_fill(
            "o3", "BTC/USD", "buy",
            filled_qty=0.5, remaining_qty=0.5,
            fill_price=65000, elapsed_ms=1000,
            urgency=0.9,
        )
        assert action.action == "market_sweep"

    def test_cancel_replace_stale_order(self):
        m = self._make()
        action = m.on_partial_fill(
            "o4", "BTC/USD", "sell",
            filled_qty=0.7, remaining_qty=0.3,
            fill_price=65000, elapsed_ms=40000,
        )
        assert action.action == "cancel_replace"
        assert action.new_price < 65000  # sell → more aggressive = lower

    def test_wait_default(self):
        m = self._make()
        action = m.on_partial_fill(
            "o5", "BTC/USD", "buy",
            filled_qty=0.6, remaining_qty=0.4,
            fill_price=65000, elapsed_ms=10000,
        )
        assert action.action == "wait"

    def test_fill_rate_by_venue_empty(self):
        m = self._make()
        assert m.get_fill_rate_by_venue("kraken") == 0.0

    def test_fill_rate_by_venue(self):
        m = self._make()
        for i in range(5):
            m.on_partial_fill(
                f"o{i}", "BTC/USD", "buy",
                filled_qty=0.99 if i < 3 else 0.5,
                remaining_qty=0.01 if i < 3 else 0.5,
                fill_price=65000, elapsed_ms=1000,
                exchange="kraken",
            )
        rate = m.get_fill_rate_by_venue("kraken")
        assert 0.5 <= rate <= 0.7  # 3 out of 5 fully filled

    def test_stats(self):
        m = self._make()
        m.on_partial_fill("o1", "BTC/USD", "buy", 0.9, 0.1, 65000, 2000)
        m.on_partial_fill("o2", "BTC/USD", "buy", 0.3, 0.7, 65000, 35000)
        stats = m.get_stats("BTC/USD")
        assert stats["total_events"] == 2
        assert "action_counts" in stats

    def test_urgency_clamped(self):
        m = self._make()
        action = m.on_partial_fill(
            "o6", "BTC/USD", "buy",
            filled_qty=0.5, remaining_qty=0.5,
            fill_price=65000, elapsed_ms=1000,
            urgency=5.0,  # should be clamped to 1.0 → sweep
        )
        assert action.action == "market_sweep"


# ---------------------------------------------------------------------------
# 5. ContagionModel
# ---------------------------------------------------------------------------

class TestContagionModel:
    """Tests for ContagionModel."""

    def _make(self, **kwargs):
        from risk.contagion_model import ContagionModel
        return ContagionModel(**kwargs)

    def _feed_correlated(self, model, sym_a, sym_b, n=100, correlation="positive"):
        """Feed two symbols with correlated price movements."""
        import random
        random.seed(42)
        base = 65000
        t = time.time() - n * 60
        for i in range(n):
            move = random.gauss(0, 50)
            model.update_price(sym_a, base + move, timestamp=t + i * 60)
            if correlation == "positive":
                model.update_price(sym_b, (base / 20) + move / 20, timestamp=t + i * 60 + 5)
            else:
                model.update_price(sym_b, (base / 20) - move / 20, timestamp=t + i * 60 + 5)

    def test_init(self):
        m = self._make()
        assert m.get_symbols() == []

    def test_update_price(self):
        m = self._make()
        m.update_price("BTC/USD", 65000)
        assert "BTC/USD" in m.get_symbols()

    def test_detect_no_contagion_insufficient_data(self):
        m = self._make()
        m.update_price("BTC/USD", 65000)
        report = m.detect_contagion("BTC/USD")
        assert report.affected_symbols == []
        assert report.severity == "low"

    def test_detect_contagion_correlated_symbols(self):
        m = self._make(min_return_count=10)
        self._feed_correlated(m, "BTC/USD", "ETH/USD", n=100)
        report = m.detect_contagion("BTC/USD", threshold_pct=0.01)
        # Should find ETH correlated
        assert "ETH/USD" in report.correlation
        assert report.correlation["ETH/USD"] > 0.3

    def test_cascade_order(self):
        m = self._make(min_return_count=10)
        self._feed_correlated(m, "BTC/USD", "ETH/USD", n=100)
        self._feed_correlated(m, "BTC/USD", "SOL/USD", n=100)
        cascade = m.get_cascade_order("BTC/USD")
        assert isinstance(cascade, list)
        # Should include both other symbols
        assert len(cascade) >= 2

    def test_hedge_symbols_positive_corr(self):
        m = self._make(min_return_count=10)
        self._feed_correlated(m, "BTC/USD", "ETH/USD", n=100, correlation="positive")
        hedges = m.get_hedge_symbols(["BTC/USD"])
        # ETH is positively correlated, so NOT a hedge
        assert "ETH/USD" not in hedges

    def test_hedge_symbols_negative_corr(self):
        m = self._make(min_return_count=10)
        self._feed_correlated(m, "BTC/USD", "GOLD/USD", n=100, correlation="negative")
        hedges = m.get_hedge_symbols(["BTC/USD"])
        # GOLD is negatively correlated, should be a hedge
        assert "GOLD/USD" in hedges

    def test_empty_portfolio_hedges(self):
        m = self._make()
        assert m.get_hedge_symbols([]) == []

    def test_contagion_severity_low(self):
        from risk.contagion_model import ContagionModel
        sev = ContagionModel._classify_severity(0, {"X": 0.3}, 1.0)
        assert sev == "low"

    def test_contagion_severity_critical(self):
        from risk.contagion_model import ContagionModel
        sev = ContagionModel._classify_severity(5, {"X": 0.9, "Y": 0.8}, 12.0)
        assert sev == "critical"

    def test_pearson_correlation(self):
        from risk.contagion_model import ContagionModel
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        corr = ContagionModel._pearson_correlation(xs, ys)
        assert abs(corr - 1.0) < 0.001

    def test_pearson_negative(self):
        from risk.contagion_model import ContagionModel
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]
        corr = ContagionModel._pearson_correlation(xs, ys)
        assert abs(corr - (-1.0)) < 0.001
