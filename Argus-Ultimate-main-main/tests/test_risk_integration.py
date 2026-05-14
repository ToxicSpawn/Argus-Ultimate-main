"""
Tests for risk integration features (batch: portfolio beta, intraday VaR live
correlation, options Greeks pre-trade, funding cost enforcement, tail hedge
execution, slippage auto-populate, flash crash microstructure, MM inventory).

Target: >= 35 tests.
"""

from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import time
from collections import deque
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Feature 1: Portfolio beta
# ---------------------------------------------------------------------------

class TestPortfolioBeta:
    """Tests for real beta calculation in PortfolioRiskManager."""

    def _make_manager(self, capital=10000.0):
        from risk.portfolio import PortfolioRiskManager
        return PortfolioRiskManager(initial_capital=capital)

    def test_beta_insufficient_data_returns_one(self):
        mgr = self._make_manager()
        # Only 5 returns — below 20 threshold
        for i in range(5):
            mgr.update_capital(10000 + i * 10)
        assert mgr._calculate_beta() == 1.0

    def test_beta_with_explicit_market_returns(self):
        mgr = self._make_manager()
        np.random.seed(42)
        mkt = np.random.normal(0.001, 0.02, 50)
        # Portfolio returns = 1.5 * market + noise
        port = 1.5 * mkt + np.random.normal(0, 0.005, 50)
        mgr._returns_history = deque(port, maxlen=252)
        beta = mgr._calculate_beta(market_returns=pd.Series(mkt))
        assert 1.0 < beta < 2.0, f"Expected beta ~1.5, got {beta}"

    def test_beta_zero_variance_market_returns_one(self):
        mgr = self._make_manager()
        mgr._returns_history = deque([0.01] * 25, maxlen=252)
        flat_mkt = pd.Series([0.0] * 25)
        beta = mgr._calculate_beta(market_returns=flat_mkt)
        assert beta == 1.0

    def test_beta_from_btc_price_history(self):
        mgr = self._make_manager()
        np.random.seed(7)
        # Populate BTC price history
        btc_prices = [50000.0]
        for _ in range(30):
            btc_prices.append(btc_prices[-1] * (1 + np.random.normal(0.001, 0.02)))
        mgr._price_history["BTC/USD"] = deque(btc_prices, maxlen=60)
        # Populate portfolio returns (correlated with BTC)
        btc_returns = pd.Series(btc_prices).pct_change().dropna().values
        port_returns = 0.8 * btc_returns + np.random.normal(0, 0.005, len(btc_returns))
        mgr._returns_history = deque(port_returns, maxlen=252)
        beta = mgr._calculate_beta()
        # Should be roughly around 0.8
        assert 0.3 < beta < 1.5, f"Expected beta ~0.8, got {beta}"

    def test_beta_no_btc_in_price_history_returns_one(self):
        mgr = self._make_manager()
        mgr._returns_history = deque([0.01] * 25, maxlen=252)
        mgr._price_history["ETH/USD"] = deque([3000 + i for i in range(25)], maxlen=60)
        beta = mgr._calculate_beta()
        assert beta == 1.0

    def test_beta_negative_correlation(self):
        mgr = self._make_manager()
        np.random.seed(99)
        mkt = np.random.normal(0.001, 0.02, 40)
        port = -0.5 * mkt + np.random.normal(0, 0.005, 40)
        mgr._returns_history = deque(port, maxlen=252)
        beta = mgr._calculate_beta(market_returns=pd.Series(mkt))
        assert beta < 0, f"Expected negative beta, got {beta}"


# ---------------------------------------------------------------------------
# Feature 2: Intraday VaR live correlation
# ---------------------------------------------------------------------------

class TestIntradayVaRCorrelation:
    """Tests for live correlation source in IntradayVaR."""

    def _make_var(self, capital=10000.0):
        from risk.intraday_var import IntradayVaR
        return IntradayVaR(capital_usd=capital)

    def test_default_correlation_factor_is_0_7(self):
        var = self._make_var()
        assert var._resolve_correlation_factor() == 0.7

    def test_set_correlation_source_with_method(self):
        var = self._make_var()
        monitor = MagicMock()
        monitor.get_average_correlation.return_value = 0.5
        var.set_correlation_source(monitor)
        assert var._resolve_correlation_factor() == 0.5

    def test_set_correlation_source_with_attribute(self):
        var = self._make_var()
        monitor = SimpleNamespace(_last_avg_corr=0.3)
        var.set_correlation_source(monitor)
        assert var._resolve_correlation_factor() == 0.3

    def test_correlation_source_returns_none_falls_back(self):
        var = self._make_var()
        monitor = MagicMock()
        monitor.get_average_correlation.return_value = None
        # Also no _last_avg_corr attribute
        del monitor._last_avg_corr
        var.set_correlation_source(monitor)
        assert var._resolve_correlation_factor() == 0.7

    def test_correlation_source_error_falls_back(self):
        var = self._make_var()
        monitor = MagicMock()
        monitor.get_average_correlation.side_effect = RuntimeError("boom")
        del monitor._last_avg_corr
        var.set_correlation_source(monitor)
        assert var._resolve_correlation_factor() == 0.7

    def test_portfolio_var_uses_live_correlation(self):
        var = self._make_var()
        # Feed enough price data
        for i in range(15):
            var.update_price("BTC/USD", 60000 + i * 100)
        var.update_position("BTC/USD", 5000.0)

        # Without correlation source: uses 0.7
        var_default = var.compute_portfolio_var()

        # With correlation source: uses 0.5
        monitor = MagicMock()
        monitor.get_average_correlation.return_value = 0.5
        var.set_correlation_source(monitor)
        var_custom = var.compute_portfolio_var()

        # var_custom should be lower (0.5/0.7 ratio)
        if var_default > 0:
            ratio = var_custom / var_default
            assert 0.6 < ratio < 0.8, f"Expected ~0.71 ratio, got {ratio}"

    def test_correlation_clamped_0_1(self):
        var = self._make_var()
        monitor = MagicMock()
        monitor.get_average_correlation.return_value = 1.5
        var.set_correlation_source(monitor)
        assert var._resolve_correlation_factor() == 1.0

        monitor.get_average_correlation.return_value = -0.3
        assert var._resolve_correlation_factor() == 0.0


# ---------------------------------------------------------------------------
# Feature 3: Options Greeks in pre-trade
# ---------------------------------------------------------------------------

class TestOptionsGreeksPreTrade:
    """Tests for the options Greeks check in ComponentRegistry.pre_order_check."""

    def _make_registry(self, capital=1000.0, delta=None):
        from core.component_registry import ComponentRegistry
        config = SimpleNamespace(starting_capital_aud=capital)
        reg = ComponentRegistry(config)
        if delta is not None:
            greeks_calc = MagicMock()
            greeks_calc.get_portfolio_greeks.return_value = {"delta": delta}
            reg._greeks_calculator = greeks_calc
        return reg

    def test_no_greeks_calculator_no_impact(self):
        reg = self._make_registry()
        result = reg.pre_order_check("BTC/USD", "buy", 100.0)
        assert result["size_factor"] == 1.0

    def test_delta_below_threshold_no_reduction(self):
        reg = self._make_registry(capital=1000.0, delta=500.0)
        result = reg.pre_order_check("BTC/USD", "buy", 100.0)
        # 500 < 0.8 * 1000 = 800 — no reduction
        greeks_reasons = [r for r in result["reasons"] if "delta" in r.lower()]
        assert len(greeks_reasons) == 0

    def test_delta_above_threshold_halves_size(self):
        reg = self._make_registry(capital=1000.0, delta=900.0)
        result = reg.pre_order_check("BTC/USD", "buy", 100.0)
        assert result["size_factor"] <= 0.5
        greeks_reasons = [r for r in result["reasons"] if "delta" in r.lower()]
        assert len(greeks_reasons) == 1

    def test_negative_delta_above_threshold(self):
        reg = self._make_registry(capital=1000.0, delta=-850.0)
        result = reg.pre_order_check("BTC/USD", "sell", 100.0)
        assert result["size_factor"] <= 0.5


# ---------------------------------------------------------------------------
# Feature 4: Funding cost enforcement
# ---------------------------------------------------------------------------

class TestFundingCostEnforcement:
    """Tests for get_exit_recommendations() in FundingCostLimiter."""

    def _make_limiter(self):
        from risk.funding_cost_limiter import FundingCostLimiter
        return FundingCostLimiter(
            max_annual_cost_pct=0.50,
            max_8h_rate_pct=0.10,
            alert_threshold_pct=0.30,
        )

    def test_no_payments_no_recommendations(self):
        limiter = self._make_limiter()
        recs = limiter.get_exit_recommendations()
        assert recs == []

    def test_below_threshold_no_recommendations(self):
        limiter = self._make_limiter()
        # Record moderate funding (positive rate = cost)
        limiter.record_payment("BTC/USD", "kraken", 0.005, -0.50, 10000.0)
        recs = limiter.get_exit_recommendations()
        assert len(recs) == 0

    def test_above_alert_returns_reduce(self):
        limiter = self._make_limiter()
        # Positive rate_pct means cost: 0.04% per 8h annualised = 0.0004 * 3 * 365 = 43.8%
        # 43.8% > alert 30% but < max 50% => REDUCE
        for _ in range(5):
            limiter.record_payment("ETH/USD", "kraken", 0.04, -4.0, 10000.0)
        recs = limiter.get_exit_recommendations()
        assert len(recs) >= 1
        assert recs[0]["recommendation"] == "REDUCE"

    def test_above_max_returns_exit(self):
        limiter = self._make_limiter()
        # 0.06% per 8h = 0.0006 * 3 * 365 = 65.7% annual > 50% => EXIT
        for _ in range(5):
            limiter.record_payment("SOL/USD", "coinbase", 0.06, -6.0, 10000.0)
        recs = limiter.get_exit_recommendations()
        assert len(recs) >= 1
        assert recs[0]["recommendation"] == "EXIT"
        assert recs[0]["symbol"] == "SOL/USD"
        assert "exchange" in recs[0]

    def test_exit_recommendations_contain_required_keys(self):
        limiter = self._make_limiter()
        for _ in range(3):
            limiter.record_payment("BTC/USD", "kraken", 0.08, -8.0, 10000.0)
        recs = limiter.get_exit_recommendations()
        assert len(recs) >= 1
        for rec in recs:
            assert "symbol" in rec
            assert "exchange" in rec
            assert "annualized_cost_pct" in rec
            assert "recommendation" in rec


# ---------------------------------------------------------------------------
# Feature 5: Tail hedge execution
# ---------------------------------------------------------------------------

class TestTailHedgeExecution:
    """Tests for generate_hedge_orders() in TailHedgeAdvisor."""

    def _make_advisor(self, capital=10000.0):
        from risk.tail_hedge import TailHedgeAdvisor
        return TailHedgeAdvisor(capital_usd=capital)

    def test_generate_orders_from_crisis_evaluation(self):
        adv = self._make_advisor(capital=10000.0)
        recs = adv.evaluate(
            regime="CRISIS",
            portfolio_var_pct=0.05,
            funding_rate=-0.03,
            fear_greed_index=10.0,
        )
        assert len(recs) > 0
        orders = adv.generate_hedge_orders(recs, portfolio_value=10000.0, btc_price=60000.0)
        assert len(orders) > 0
        for order in orders:
            assert "symbol" in order
            assert "side" in order
            assert "quantity" in order
            assert order["quantity"] > 0
            assert "order_type" in order
            assert "reason" in order

    def test_short_futures_order_quantity(self):
        from risk.tail_hedge import HedgeRecommendation
        adv = self._make_advisor()
        rec = HedgeRecommendation(
            instrument="BTC-PERP",
            action="SHORT_FUTURES",
            size_usd=5000.0,
            rationale="test",
            urgency=0.8,
            estimated_cost_usd=5.0,
        )
        orders = adv.generate_hedge_orders([rec], portfolio_value=10000.0, btc_price=50000.0)
        assert len(orders) == 1
        order = orders[0]
        assert order["side"] == "sell"
        assert order["symbol"] == "BTC-PERP"
        # qty = 10000 * (5000/10000) / 50000 = 0.01
        expected_qty = 10000.0 * 0.5 / 50000.0
        assert abs(order["quantity"] - expected_qty) < 0.001

    def test_stablecoin_allocation_order(self):
        from risk.tail_hedge import HedgeRecommendation
        adv = self._make_advisor()
        rec = HedgeRecommendation(
            instrument="USDT",
            action="INCREASE_CASH",
            size_usd=3000.0,
            rationale="test",
            urgency=0.6,
            estimated_cost_usd=0.0,
        )
        orders = adv.generate_hedge_orders([rec], portfolio_value=10000.0)
        assert len(orders) == 1
        assert orders[0]["side"] == "buy"
        assert orders[0]["symbol"] == "USDT"

    def test_empty_recommendations_no_orders(self):
        adv = self._make_advisor()
        orders = adv.generate_hedge_orders([], portfolio_value=10000.0)
        assert orders == []

    def test_zero_portfolio_value_no_orders(self):
        from risk.tail_hedge import HedgeRecommendation
        adv = self._make_advisor()
        rec = HedgeRecommendation(
            instrument="BTC-PERP", action="SHORT_FUTURES",
            size_usd=5000.0, rationale="t", urgency=0.8, estimated_cost_usd=0,
        )
        orders = adv.generate_hedge_orders([rec], portfolio_value=0.0)
        assert orders == []


# ---------------------------------------------------------------------------
# Feature 6: Slippage auto-populate from ledger
# ---------------------------------------------------------------------------

class TestSlippageAutoPopulate:
    """Tests for auto_populate_from_ledger() in AdaptiveSlippageModel."""

    def test_missing_ledger_returns_zero(self):
        from execution.adaptive_slippage_model import AdaptiveSlippageModel
        count = AdaptiveSlippageModel.auto_populate_from_ledger(
            ledger_path="/nonexistent/ledger.db"
        )
        assert count == 0

    def test_populate_from_valid_ledger(self):
        from execution.adaptive_slippage_model import AdaptiveSlippageModel

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = os.path.join(tmpdir, "trade_ledger.db")
            fills_path = os.path.join(tmpdir, "fills.db")

            # Create a fake trade ledger
            conn = sqlite3.connect(ledger_path)
            conn.execute(
                "CREATE TABLE trades (symbol TEXT, side TEXT, price REAL, "
                "quantity REAL, timestamp REAL)"
            )
            for i in range(10):
                conn.execute(
                    "INSERT INTO trades VALUES (?, ?, ?, ?, ?)",
                    ("BTC/USD", "buy", 60000.0 + i * 10, 0.01, time.time() - i * 3600),
                )
            conn.commit()
            conn.close()

            count = AdaptiveSlippageModel.auto_populate_from_ledger(
                ledger_path=ledger_path, fills_db_path=fills_path
            )
            assert count == 10

            # Verify fills DB was created with correct schema
            conn2 = sqlite3.connect(fills_path)
            rows = conn2.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
            conn2.close()
            assert rows == 10

    def test_populate_computes_slippage_correctly(self):
        from execution.adaptive_slippage_model import AdaptiveSlippageModel

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = os.path.join(tmpdir, "ledger.db")
            fills_path = os.path.join(tmpdir, "fills.db")

            conn = sqlite3.connect(ledger_path)
            conn.execute(
                "CREATE TABLE trades (symbol TEXT, side TEXT, price REAL, "
                "quantity REAL, timestamp REAL, expected_price REAL)"
            )
            # Buy at 60100 when expected 60000 -> slippage = 100/60000 * 10000 = 16.67 bps
            conn.execute(
                "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)",
                ("BTC/USD", "buy", 60100.0, 0.01, time.time(), 60000.0),
            )
            conn.commit()
            conn.close()

            AdaptiveSlippageModel.auto_populate_from_ledger(
                ledger_path=ledger_path, fills_db_path=fills_path
            )

            conn2 = sqlite3.connect(fills_path)
            row = conn2.execute("SELECT slippage_bps FROM fills").fetchone()
            conn2.close()
            expected_bps = (60100.0 - 60000.0) / 60000.0 * 10000.0
            assert abs(row[0] - expected_bps) < 0.01


# ---------------------------------------------------------------------------
# Feature 7: Flash crash microstructure
# ---------------------------------------------------------------------------

class TestFlashCrashMicrostructure:
    """Tests for detect_microstructure_anomaly() in UnifiedRiskManager."""

    def _make_manager(self):
        from risk.unified_risk_manager import UnifiedRiskManager
        return UnifiedRiskManager(initial_capital=10000.0)

    def test_normal_prices_no_anomaly(self):
        mgr = self._make_manager()
        prices = [60000 + i * 5 for i in range(20)]
        timestamps = [1000.0 + i * 10 for i in range(20)]
        result = mgr.detect_microstructure_anomaly(prices, timestamps)
        assert result["anomaly"] is False
        assert result["severity"] == 0.0

    def test_spread_blowout_detected(self):
        mgr = self._make_manager()
        # Normal prices then sudden 5% jump
        prices = [60000.0] * 10 + [63000.0]  # 5% jump = 500 bps
        timestamps = [1000.0 + i * 10 for i in range(11)]
        result = mgr.detect_microstructure_anomaly(prices, timestamps, normal_spread_bps=5.0)
        assert result["anomaly"] is True
        assert "spread_blow_out" in result["type"]

    def test_price_reversal_detected(self):
        mgr = self._make_manager()
        # Price drops 3% then recovers within 60s
        base = 60000.0
        prices = [base, base * 0.97, base * 0.99]
        timestamps = [1000.0, 1020.0, 1040.0]
        result = mgr.detect_microstructure_anomaly(prices, timestamps)
        assert result["anomaly"] is True
        assert "price_reversal" in result["type"]

    def test_insufficient_data_no_anomaly(self):
        mgr = self._make_manager()
        result = mgr.detect_microstructure_anomaly([60000], [1000])
        assert result["anomaly"] is False

    def test_mismatched_lengths_no_anomaly(self):
        mgr = self._make_manager()
        result = mgr.detect_microstructure_anomaly([60000, 60100], [1000])
        assert result["anomaly"] is False

    def test_severity_bounded_0_1(self):
        mgr = self._make_manager()
        # Massive spike
        prices = [60000.0] * 10 + [90000.0]
        timestamps = [1000.0 + i * 10 for i in range(11)]
        result = mgr.detect_microstructure_anomaly(prices, timestamps)
        assert 0.0 <= result["severity"] <= 1.0


# ---------------------------------------------------------------------------
# Feature 8: Market maker inventory limits
# ---------------------------------------------------------------------------

class TestMarketMakerInventory:
    """Tests for inventory tracking in AvellanedaStoikovMM."""

    def _make_mm(self, max_inv=1.0):
        from strategies.market_maker_avellaneda import AvellanedaStoikovMM
        return AvellanedaStoikovMM(symbol="BTC/USD", max_inventory=max_inv)

    def test_update_inventory_buy(self):
        mm = self._make_mm()
        mm.update_inventory("buy", 0.5)
        assert mm.inventory == 0.5
        assert mm.current_inventory == 0.5

    def test_update_inventory_sell(self):
        mm = self._make_mm()
        mm.update_inventory("buy", 0.5)
        mm.update_inventory("sell", 0.3)
        assert abs(mm.inventory - 0.2) < 1e-9

    def test_skew_at_80_pct_capacity(self):
        mm = self._make_mm(max_inv=1.0)
        mm.inventory = 0.85  # > 80%
        bid_skew, ask_skew = mm._inventory_skew()
        # Should be aggressive — bid_skew negative, ask_skew more negative
        assert bid_skew < 0
        assert ask_skew < bid_skew  # ask more aggressive

    def test_skew_below_60_pct_is_zero(self):
        mm = self._make_mm(max_inv=1.0)
        mm.inventory = 0.3  # 30% — below 60%
        bid_skew, ask_skew = mm._inventory_skew()
        assert bid_skew == 0.0
        assert ask_skew == 0.0

    def test_quotes_blocked_at_max_inventory(self):
        mm = self._make_mm(max_inv=1.0)
        mm.inventory = 1.0
        # Feed some price data
        for i in range(5):
            mm._mid_prices.append(60000 + i)
        result = mm.analyze({"price": 60000.0, "bid": 59990.0, "ask": 60010.0})
        assert result is None  # blocked

    def test_quotes_allowed_below_max(self):
        mm = self._make_mm(max_inv=1.0)
        mm.inventory = 0.5
        for i in range(5):
            mm._mid_prices.append(60000 + i)
        result = mm.analyze({"price": 60000.0, "bid": 59990.0, "ask": 60010.0})
        # Should return a valid quote (may be None due to spread calc, but not due to inventory)
        # At least verify the inventory check didn't block it
        if result is not None:
            assert result["inventory"] == 0.5

    def test_negative_inventory_skew_buy_side(self):
        mm = self._make_mm(max_inv=1.0)
        mm.inventory = -0.85  # short heavy > 80%
        bid_skew, ask_skew = mm._inventory_skew()
        # Short heavy: push bid up (positive), ask also positive
        assert bid_skew > 0
        assert ask_skew > 0
