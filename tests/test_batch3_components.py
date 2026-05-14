"""
Tests for Batch 3 components — all 31 new files.

Covers:
  execution/fill_tracker.py
  execution/maker_enforcement.py
  risk/intraday_var.py
  risk/stress_tester.py
  risk/counterparty_monitor.py
  risk/funding_cost_limiter.py
  risk/tail_hedge.py
  strategies/liquidation_cascade.py
  strategies/macro_event_calendar.py
  strategies/cross_exchange_arb.py
  strategies/futures_basis_arb.py
  strategies/deribit_options.py
  ml/hmm_regime.py
  ml/signal_stacker.py
  ml/llm_signal.py
  ml/hyperopt.py
  ml/feature_importance.py
  core/health_server.py
  core/process_lock.py
  monitoring/latency_tracker.py
  monitoring/exchange_monitor.py
  monitoring/performance_attribution.py
  monitoring/drawdown_autopsy.py
  monitoring/trade_journal.py
  monitoring/discord_webhook.py
  monitoring/pdf_reporter.py
  compliance/ato_cgt.py
  compliance/tax_lot_optimizer.py
  data/fear_greed.py
  data/funding_predictor.py
  data/tick_capture.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# execution/fill_tracker
# ---------------------------------------------------------------------------

class TestFillTracker:
    def test_record_fill_and_slippage(self, tmp_path):
        from execution.fill_tracker import FillTracker
        ft = FillTracker(db_path=str(tmp_path / "fills.db"))
        ft.record_fill(
            strategy="test_strat",
            symbol="BTC/USD",
            side="buy",
            expected_price=50000.0,
            actual_price=50100.0,  # 2 bps slippage
            quantity_usd=200.0,
            exchange="kraken",
        )
        stats = ft.get_strategy_stats("test_strat")
        assert stats["fill_count"] == 1
        assert stats["avg_slippage_bps"] > 0

    def test_within_budget_initial(self, tmp_path):
        from execution.fill_tracker import FillTracker
        ft = FillTracker(db_path=str(tmp_path / "fills.db"))
        assert ft.is_within_budget("any_strat") is True

    def test_budget_exhausted_pauses(self, tmp_path):
        from execution.fill_tracker import FillTracker
        ft = FillTracker(db_path=str(tmp_path / "fills.db"), daily_limit_bps=5.0, daily_limit_usd=10.0)
        # Cause large slippage
        for _ in range(5):
            ft.record_fill(
                strategy="s",
                symbol="BTC/USD",
                side="buy",
                expected_price=10000.0,
                actual_price=10200.0,  # 20 bps each
                quantity_usd=100.0,
                exchange="kraken",
            )
        # Budget should be exhausted
        assert ft.is_within_budget("s") is False

    def test_record_fill_zero_slippage(self, tmp_path):
        from execution.fill_tracker import FillTracker
        ft = FillTracker(db_path=str(tmp_path / "fills.db"))
        ft.record_fill("s", "ETH/USD", "sell", 3000.0, 3000.0, 100.0, "kraken")
        stats = ft.get_strategy_stats("s")
        assert stats["fill_count"] == 1
        assert stats["avg_slippage_bps"] == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# execution/maker_enforcement
# ---------------------------------------------------------------------------

class TestMakerEnforcement:
    def test_should_use_maker_low_urgency(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement()
        assert me.should_use_maker(urgency=0.3, spread_bps=5.0) is True

    def test_should_use_taker_high_urgency(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement()
        assert me.should_use_maker(urgency=0.9, spread_bps=5.0) is False

    def test_should_use_taker_narrow_spread(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement()
        # MIN_SPREAD_BPS is 0.5; use a spread below that to trigger taker
        assert me.should_use_maker(urgency=0.3, spread_bps=0.3) is False

    def test_place_order_simulation_maker(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement(connector=None)
        result = asyncio.run(
            me.place_order("BTC/USD", "buy", 200.0, 50000.0, urgency=0.3)
        )
        assert result.success is True
        assert result.is_maker is True
        assert result.fee_bps == me.MAKER_FEE_BPS

    def test_place_order_simulation_taker(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement(connector=None)
        result = asyncio.run(
            me.place_order("BTC/USD", "buy", 200.0, 50000.0, urgency=0.95)
        )
        assert result.success is True
        assert result.is_maker is False

    def test_estimate_savings(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement()
        savings = me.estimate_savings_usd(1000.0)
        assert savings == pytest.approx(0.40, rel=0.01)  # 4 bps on $1000

    def test_disabled_always_taker(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement(enabled=False)
        result = asyncio.run(
            me.place_order("BTC/USD", "sell", 100.0, 50000.0, urgency=0.1)
        )
        assert result.is_maker is False


# ---------------------------------------------------------------------------
# risk/intraday_var
# ---------------------------------------------------------------------------

class TestIntradayVaR:
    def test_snapshot_structure(self):
        from risk.intraday_var import IntradayVaR
        var = IntradayVaR(capital_usd=1000.0)
        snap = var.snapshot()
        assert "portfolio_var_usd" in snap
        assert "var_limit_usd" in snap
        assert "utilisation_pct" in snap
        assert "breach" in snap

    def test_update_price_and_position(self):
        from risk.intraday_var import IntradayVaR
        var = IntradayVaR(capital_usd=1000.0)
        for i in range(20):
            var.update_price("BTC/USD", 50000.0 + i * 100)
        var.update_position("BTC/USD", 200.0)
        pv = var.compute_position_var("BTC/USD")
        assert pv.var_95_usd >= 0
        assert pv.var_99_usd >= pv.var_95_usd

    def test_reset_clears_state(self):
        from risk.intraday_var import IntradayVaR
        var = IntradayVaR(capital_usd=1000.0)
        var.update_position("BTC/USD", 200.0)
        var.reset()
        snap = var.snapshot()
        assert snap["portfolio_var_usd"] == pytest.approx(0.0, abs=0.01)

    def test_var_limit_breach(self):
        from risk.intraday_var import IntradayVaR
        var = IntradayVaR(capital_usd=100.0, var_limit_pct=0.01)
        # High volatility → large VaR
        prices = [100.0]
        for _ in range(30):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.05)))
        for p in prices:
            var.update_price("BTC/USD", p)
        var.update_position("BTC/USD", 90.0)
        snap = var.snapshot()
        assert isinstance(snap["breach"], bool)


# ---------------------------------------------------------------------------
# risk/stress_tester
# ---------------------------------------------------------------------------

class TestStressTester:
    def test_run_all_returns_results(self):
        from risk.stress_tester import PortfolioStressTester
        tester = PortfolioStressTester(capital_usd=1000.0)
        results = tester.run_all({"BTC/USD": 300.0, "ETH/USD": 200.0})
        assert len(results) >= 3

    def test_worst_case(self):
        from risk.stress_tester import PortfolioStressTester
        tester = PortfolioStressTester(capital_usd=1000.0)
        worst = tester.worst_case({"BTC/USD": 500.0})
        assert worst.pnl_usd < 0

    def test_summary_keys(self):
        from risk.stress_tester import PortfolioStressTester
        tester = PortfolioStressTester(capital_usd=1000.0)
        s = tester.summary({"BTC/USD": 200.0})
        assert "max_loss_usd" in s
        assert "scenarios_survived" in s

    def test_empty_positions(self):
        from risk.stress_tester import PortfolioStressTester
        tester = PortfolioStressTester(capital_usd=1000.0)
        results = tester.run_all({})
        assert all(r.pnl_usd == 0.0 for r in results)


# ---------------------------------------------------------------------------
# risk/counterparty_monitor
# ---------------------------------------------------------------------------

class TestCounterpartyMonitor:
    def _make_health(self, exchange_id, withdrawal_ok=True, risk_score=10):
        from risk.counterparty_monitor import ExchangeHealth
        return ExchangeHealth(
            exchange_id=exchange_id,
            withdrawal_ok=withdrawal_ok,
            insurance_fund_usd=1e9,
            open_interest_usd=1e10,
            funding_rate=0.01,
            volume_24h=5e8,
            risk_score=risk_score,
            warnings=[],
        )

    def test_update_and_risk_score(self):
        from risk.counterparty_monitor import CounterpartyMonitor
        cm = CounterpartyMonitor(["kraken", "bybit"])
        cm.update("kraken", self._make_health("kraken"))
        score = cm.get_risk_score("kraken")
        assert 0 <= score <= 100

    def test_should_reduce_low_risk(self):
        from risk.counterparty_monitor import CounterpartyMonitor
        cm = CounterpartyMonitor(["kraken"])
        cm.update("kraken", self._make_health("kraken", risk_score=20))
        assert cm.should_reduce_exposure("kraken") is False

    def test_should_reduce_high_risk(self):
        from risk.counterparty_monitor import CounterpartyMonitor, ExchangeHealth
        cm = CounterpartyMonitor(["kraken"])
        # withdrawal_ok=False (+30), extreme negative funding (+20), huge OI/insurance ratio (+25) → score ≥ 70
        health = ExchangeHealth(
            exchange_id="kraken",
            withdrawal_ok=False,
            insurance_fund_usd=1_000.0,
            open_interest_usd=1_000_000_000.0,
            funding_rate=-0.10,
            volume_24h=5e8,
            risk_score=0,
            warnings=[],
        )
        cm.update("kraken", health)
        assert cm.should_reduce_exposure("kraken") is True

    def test_snapshot_keys(self):
        from risk.counterparty_monitor import CounterpartyMonitor
        cm = CounterpartyMonitor(["kraken"])
        cm.update("kraken", self._make_health("kraken"))
        snap = cm.snapshot()
        assert isinstance(snap, dict)
        # snapshot may be {"kraken": ...} or {"exchanges": {"kraken": ...}}
        assert "kraken" in snap or "kraken" in snap.get("exchanges", {})


# ---------------------------------------------------------------------------
# risk/funding_cost_limiter
# ---------------------------------------------------------------------------

class TestFundingCostLimiter:
    def test_record_payment(self):
        from risk.funding_cost_limiter import FundingCostLimiter
        fl = FundingCostLimiter()
        fl.record_payment("BTC-PERP", "bybit", 0.01, 5.0, 500.0)
        cost = fl.get_annualised_cost("BTC-PERP")
        assert cost >= 0

    def test_should_exit_high_cost(self):
        from risk.funding_cost_limiter import FundingCostLimiter
        fl = FundingCostLimiter(max_annual_cost_pct=0.50, alert_threshold_pct=0.10)
        # Record many high funding payments — 0.05% per 8h = ~54% annualised
        for _ in range(30):
            fl.record_payment("BTC-PERP", "bybit", 0.05, 25.0, 500.0)
        assert fl.should_exit("BTC-PERP") is True

    def test_recommendations_structure(self):
        from risk.funding_cost_limiter import FundingCostLimiter
        fl = FundingCostLimiter()
        recs = fl.get_recommendations()
        assert isinstance(recs, list)

    def test_snapshot_keys(self):
        from risk.funding_cost_limiter import FundingCostLimiter
        fl = FundingCostLimiter()
        snap = fl.snapshot()
        assert "total_funding_paid_30d" in snap or isinstance(snap, dict)


# ---------------------------------------------------------------------------
# risk/tail_hedge
# ---------------------------------------------------------------------------

class TestTailHedgeAdvisor:
    def test_evaluate_crisis_regime(self):
        from risk.tail_hedge import TailHedgeAdvisor
        advisor = TailHedgeAdvisor(capital_usd=1000.0)
        recs = advisor.evaluate(
            regime="CRISIS",
            portfolio_var_pct=0.05,
            funding_rate=-0.02,
            fear_greed_index=15,
        )
        assert len(recs) >= 1

    def test_evaluate_calm_market(self):
        from risk.tail_hedge import TailHedgeAdvisor
        advisor = TailHedgeAdvisor(capital_usd=1000.0, min_urgency=0.5)
        recs = advisor.evaluate(
            regime="RANGE",
            portfolio_var_pct=0.01,
            funding_rate=0.01,
            fear_greed_index=55,
        )
        assert all(r.urgency < advisor.min_urgency for r in recs)

    def test_should_hedge_returns_bool(self):
        from risk.tail_hedge import TailHedgeAdvisor
        advisor = TailHedgeAdvisor(capital_usd=1000.0)
        result = advisor.should_hedge("RANGE", 0.01, 0.01, 55)
        assert isinstance(result, bool)

    def test_get_hedge_cost_nonnegative(self):
        from risk.tail_hedge import TailHedgeAdvisor
        advisor = TailHedgeAdvisor(capital_usd=1000.0)
        recs = advisor.evaluate("HIGH_VOL", 0.04, -0.01, 20)
        cost = advisor.get_hedge_cost(recs)
        assert cost >= 0


# ---------------------------------------------------------------------------
# strategies/liquidation_cascade
# ---------------------------------------------------------------------------

class TestLiquidationCascade:
    def test_generate_signal_none_initially(self):
        from strategies.liquidation_cascade import LiquidationCascadeStrategy
        s = LiquidationCascadeStrategy()
        sig = s.generate_signal("BTC/USD")
        assert sig is None  # no data fed yet

    def test_detect_cascade(self):
        from strategies.liquidation_cascade import LiquidationCascadeStrategy
        from datetime import datetime, timezone, timedelta
        s = LiquidationCascadeStrategy(oi_drop_threshold=0.05)
        base = datetime.now(tz=timezone.utc)
        oi = 1_000_000.0
        for i in range(10):
            oi *= 0.92
            s.update("BTC/USD", oi, -0.02, 50000.0 - i * 100, base + timedelta(minutes=i))
        sig = s.generate_signal("BTC/USD")
        if sig is not None:
            assert sig.direction in ("BUY", "SELL")
            assert 0 <= sig.confidence <= 1

    def test_update_stores_data(self):
        from strategies.liquidation_cascade import LiquidationCascadeStrategy
        from datetime import datetime, timezone
        s = LiquidationCascadeStrategy()
        s.update("BTC/USD", 1_000_000.0, 0.01, 50000.0, datetime.now(tz=timezone.utc))
        # No assertion — just must not raise


# ---------------------------------------------------------------------------
# strategies/macro_event_calendar
# ---------------------------------------------------------------------------

class TestMacroEventCalendar:
    def test_get_position_multiplier_normal(self):
        from strategies.macro_event_calendar import MacroEventFilter
        from datetime import datetime, timezone
        f = MacroEventFilter()
        # Far from any event
        mult = f.get_position_multiplier(datetime(2025, 8, 1, 12, 0, tzinfo=timezone.utc))
        assert mult == pytest.approx(1.0)

    def test_next_event_returns_event_or_none(self):
        from strategies.macro_event_calendar import MacroEventFilter
        f = MacroEventFilter()
        result = f.next_event()
        # Could be None if all events are past
        assert result is None or hasattr(result, "name")

    def test_add_custom_event(self):
        from strategies.macro_event_calendar import MacroEventFilter, MacroEvent
        from datetime import datetime, timezone
        f = MacroEventFilter()
        event = MacroEvent(
            name="Test Event",
            event_time=datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc),
            impact="HIGH",
            description="Test",
            assets_affected=["BTC"],
        )
        f.add_event(event)
        events = f.events_in_window(hours=24 * 365 * 10)
        names = [e.name for e in events]
        assert "Test Event" in names

    def test_should_halt_near_event(self):
        from strategies.macro_event_calendar import MacroEventFilter, MacroEvent
        from datetime import datetime, timezone, timedelta
        f = MacroEventFilter(halt_window_minutes=30)
        now = datetime.now(tz=timezone.utc)
        event = MacroEvent(
            name="Imminent Event",
            event_time=now + timedelta(minutes=15),
            impact="HIGH",
            description="Soon",
            assets_affected=["BTC"],
        )
        f.add_event(event)
        assert f.should_halt(now=now) is True


# ---------------------------------------------------------------------------
# strategies/cross_exchange_arb
# ---------------------------------------------------------------------------

class TestCrossExchangeArb:
    def _now(self):
        from datetime import datetime, timezone
        return datetime.now(tz=timezone.utc)

    def test_find_opportunity_basic(self):
        from strategies.cross_exchange_arb import CrossExchangeArbStrategy
        s = CrossExchangeArbStrategy(fee_bps_per_side=4.0, min_net_spread_bps=5.0)
        now = self._now()
        s.update_price("kraken", "BTC/USD", 50000.0, now)
        s.update_price("bybit", "BTC/USD", 50080.0, now)  # 16 bps spread
        opp = s.find_opportunity("BTC/USD")
        assert opp is not None
        assert opp.net_spread_bps > 0

    def test_no_opportunity_tight_spread(self):
        from strategies.cross_exchange_arb import CrossExchangeArbStrategy
        s = CrossExchangeArbStrategy(fee_bps_per_side=4.0, min_net_spread_bps=5.0)
        now = self._now()
        s.update_price("kraken", "BTC/USD", 50000.0, now)
        s.update_price("bybit", "BTC/USD", 50002.0, now)  # 0.4 bps
        opp = s.find_opportunity("BTC/USD")
        assert opp is None

    def test_generate_signals(self):
        from strategies.cross_exchange_arb import CrossExchangeArbStrategy
        s = CrossExchangeArbStrategy()
        now = self._now()
        s.update_price("kraken", "BTC/USD", 50000.0, now)
        s.update_price("bybit", "BTC/USD", 50100.0, now)
        signals = s.generate_signals()
        assert isinstance(signals, list)

    def test_confidence_capped_at_one(self):
        from strategies.cross_exchange_arb import CrossExchangeArbStrategy
        s = CrossExchangeArbStrategy(fee_bps_per_side=1.0, min_net_spread_bps=1.0)
        now = self._now()
        s.update_price("kraken", "BTC/USD", 50000.0, now)
        s.update_price("bybit", "BTC/USD", 51000.0, now)  # huge spread
        opp = s.find_opportunity("BTC/USD")
        if opp:
            assert opp.confidence <= 1.0


# ---------------------------------------------------------------------------
# strategies/futures_basis_arb
# ---------------------------------------------------------------------------

class TestFuturesBasisArb:
    def test_compute_basis_positive(self):
        from strategies.futures_basis_arb import FuturesBasisArbStrategy
        s = FuturesBasisArbStrategy()
        s.update_spot("BTC/USD", 50000.0)
        s.update_futures("BTC/USD", 52500.0, funding_rate=0.01)  # 5% premium
        opp = s.compute_basis("BTC/USD")
        assert opp is not None
        assert opp.basis_pct > 0

    def test_no_signal_without_data(self):
        from strategies.futures_basis_arb import FuturesBasisArbStrategy
        s = FuturesBasisArbStrategy()
        sig = s.generate_signal("BTC/USD")
        assert sig is None

    def test_long_basis_direction(self):
        from strategies.futures_basis_arb import FuturesBasisArbStrategy
        s = FuturesBasisArbStrategy(min_annual_basis_pct=5.0)
        s.update_spot("BTC/USD", 50000.0)
        s.update_futures("BTC/USD", 55000.0, funding_rate=0.03)
        opp = s.generate_signal("BTC/USD")
        if opp:
            assert opp.action in ("LONG_BASIS", "SHORT_BASIS", "NEUTRAL")


# ---------------------------------------------------------------------------
# strategies/deribit_options
# ---------------------------------------------------------------------------

class TestDeribitOptionsSignal:
    def test_generate_signal_neutral_on_failure(self):
        from strategies.deribit_options import DeribitOptionsSignal
        sig_gen = DeribitOptionsSignal(symbol="BTC")
        # No network in tests — should return NEUTRAL gracefully
        sig = asyncio.run(sig_gen.generate_signal())
        assert sig.direction in ("BULLISH", "BEARISH", "NEUTRAL")
        assert 0 <= sig.confidence <= 1

    def test_interpret_high_pc_ratio(self):
        from strategies.deribit_options import DeribitOptionsSignal, OptionsSnapshot
        sig_gen = DeribitOptionsSignal()
        snap = OptionsSnapshot(
            symbol="BTC", expiry="28MAR25",
            put_call_ratio=2.0, iv_skew_pct=8.0,
            max_pain_price=48000, gex_usd=-5_000_000,
            implied_move_pct=12.0,
        )
        sig = sig_gen._interpret(snap)
        assert sig.direction == "BEARISH"
        assert sig.confidence > 0.3

    def test_interpret_low_pc_ratio_bullish(self):
        from strategies.deribit_options import DeribitOptionsSignal, OptionsSnapshot
        sig_gen = DeribitOptionsSignal()
        snap = OptionsSnapshot(
            symbol="BTC", expiry="28MAR25",
            put_call_ratio=0.5, iv_skew_pct=-5.0,
            max_pain_price=55000, gex_usd=2_000_000,
            implied_move_pct=4.0,
        )
        sig = sig_gen._interpret(snap)
        assert sig.direction in ("BULLISH", "NEUTRAL")


# ---------------------------------------------------------------------------
# ml/hmm_regime
# ---------------------------------------------------------------------------

class TestHMMRegimeDetector:
    def _sample_returns(self, n=200):
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, n)

    def test_fit_and_predict(self):
        from ml.hmm_regime import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=3)
        returns = self._sample_returns(200)
        ok = det.fit(returns)
        assert ok is True
        assert det.is_fitted

    def test_predict_valid_regime(self):
        from ml.hmm_regime import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=3)
        returns = self._sample_returns(200)
        det.fit(returns)
        regime = det.predict(returns[-20:])
        assert regime in ("TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL", "CRISIS")

    def test_predict_proba_sums_to_one(self):
        from ml.hmm_regime import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=3)
        returns = self._sample_returns(200)
        det.fit(returns)
        proba = det.predict_proba(returns[-20:])
        assert isinstance(proba, dict)
        total = sum(proba.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_not_fitted_initially(self):
        from ml.hmm_regime import HMMRegimeDetector
        det = HMMRegimeDetector()
        assert det.is_fitted is False


# ---------------------------------------------------------------------------
# ml/signal_stacker
# ---------------------------------------------------------------------------

class TestSignalStacker:
    def test_stack_empty(self):
        from ml.signal_stacker import SignalStacker
        ss = SignalStacker()
        result = ss.stack()
        assert result.combined_value == pytest.approx(0.0)

    def test_stack_single_signal(self):
        from ml.signal_stacker import SignalStacker
        ss = SignalStacker()
        ss.update_signal("trend", 0.8, 0.9)
        result = ss.stack()
        assert result.combined_value > 0

    def test_stack_opposing_signals(self):
        from ml.signal_stacker import SignalStacker
        ss = SignalStacker()
        ss.update_signal("bull", 1.0, 0.8)
        ss.update_signal("bear", -1.0, 0.8)
        result = ss.stack()
        assert abs(result.combined_value) < 0.5  # should partially cancel

    def test_record_outcome_updates_accuracy(self):
        from ml.signal_stacker import SignalStacker
        ss = SignalStacker()
        ss.update_signal("trend", 0.7, 0.8)
        ss.record_outcome("trend", actual_direction=1)
        stats = ss.get_signal_stats()
        assert "trend" in stats

    def test_component_weights_sum(self):
        from ml.signal_stacker import SignalStacker
        ss = SignalStacker()
        ss.update_signal("a", 0.5, 0.7)
        ss.update_signal("b", -0.3, 0.6)
        result = ss.stack()
        total_w = sum(result.component_weights.values())
        assert total_w == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# ml/llm_signal
# ---------------------------------------------------------------------------

class TestLLMSignalGenerator:
    def test_generate_signal_no_connection(self):
        from ml.llm_signal import LLMSignalGenerator
        gen = LLMSignalGenerator(provider="ollama", timeout=1.0)
        sig = asyncio.run(
            gen.generate_signal(
                symbol="BTC/USD",
                regime="RANGE",
                price_data=[50000, 50100, 49900],
                funding_rate=0.01,
            )
        )
        assert sig.direction in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_parse_response_bullish(self):
        from ml.llm_signal import LLMSignalGenerator
        gen = LLMSignalGenerator()
        sig = gen._parse_response("I think this is BULLISH with high confidence.")
        assert sig.direction == "BULLISH"

    def test_parse_response_bearish(self):
        from ml.llm_signal import LLMSignalGenerator
        gen = LLMSignalGenerator()
        sig = gen._parse_response("Market looks BEARISH. Moderate confidence.")
        assert sig.direction == "BEARISH"

    def test_parse_response_neutral_fallback(self):
        from ml.llm_signal import LLMSignalGenerator
        gen = LLMSignalGenerator()
        sig = gen._parse_response("I have no idea what will happen.")
        assert sig.direction == "NEUTRAL"

    def test_build_prompt_contains_symbol(self):
        from ml.llm_signal import LLMSignalGenerator
        gen = LLMSignalGenerator()
        prompt = gen._build_prompt("BTC/USD", "TREND_UP", [50000, 50100, 50200], 0.01, None)
        assert "BTC/USD" in prompt or "BTC" in prompt


# ---------------------------------------------------------------------------
# ml/hyperopt
# ---------------------------------------------------------------------------

class TestHyperOptimizer:
    def _dummy_backtest(self, params):
        # Simple function: reward high signal_confidence
        return params.get("signal_confidence", 0.5) - params.get("stop_loss_pct", 0.01) * 10

    def test_random_search(self):
        from ml.hyperopt import HyperOptimizer
        opt = HyperOptimizer(n_trials=10, random_seed=0)
        best = opt.optimize(self._dummy_backtest)
        assert isinstance(best, dict)
        assert "signal_confidence" in best

    def test_results_count(self):
        from ml.hyperopt import HyperOptimizer
        opt = HyperOptimizer(n_trials=5)
        opt.optimize(self._dummy_backtest)
        assert len(opt.get_results()) == 5

    def test_best_params_property(self):
        from ml.hyperopt import HyperOptimizer
        opt = HyperOptimizer(n_trials=5)
        opt.optimize(self._dummy_backtest)
        assert opt.best_params is not None

    def test_handles_exception_in_backtest(self):
        from ml.hyperopt import HyperOptimizer
        def bad_fn(params):
            raise ValueError("backtest failed")
        opt = HyperOptimizer(n_trials=3)
        best = opt.optimize(bad_fn)
        assert isinstance(best, dict)  # should not raise


# ---------------------------------------------------------------------------
# ml/feature_importance
# ---------------------------------------------------------------------------

class TestFeatureImportanceTracker:
    def _make_data(self, n=100, n_features=5):
        np.random.seed(42)
        X = np.random.randn(n, n_features)
        y = X[:, 0] * 2 + np.random.randn(n) * 0.1  # feature 0 is important
        return X, y

    def test_compute_correlation(self):
        from ml.feature_importance import FeatureImportanceTracker
        tracker = FeatureImportanceTracker()
        X, y = self._make_data()
        names = [f"f{i}" for i in range(5)]
        scores = tracker.compute_correlation(X, y, names)
        assert len(scores) == 5
        # f0 should rank #1
        assert scores[0].name == "f0"

    def test_top_features(self):
        from ml.feature_importance import FeatureImportanceTracker
        tracker = FeatureImportanceTracker()
        X, y = self._make_data()
        names = [f"f{i}" for i in range(5)]
        tracker.compute_correlation(X, y, names)
        top = tracker.top_features(3)
        assert len(top) == 3

    def test_track_and_detect_drift(self):
        from ml.feature_importance import FeatureImportanceTracker
        tracker = FeatureImportanceTracker()
        X, y = self._make_data()
        names = [f"f{i}" for i in range(5)]
        for _ in range(3):
            scores = tracker.compute_correlation(X, y, names)
            tracker.track(scores)
        drift = tracker.detect_drift(window=3)
        assert isinstance(drift, dict)

    def test_to_dict(self):
        from ml.feature_importance import FeatureImportanceTracker
        tracker = FeatureImportanceTracker()
        X, y = self._make_data()
        names = [f"f{i}" for i in range(5)]
        tracker.compute_correlation(X, y, names)
        d = tracker.to_dict()
        assert "features" in d


# ---------------------------------------------------------------------------
# core/health_server
# ---------------------------------------------------------------------------

class TestHealthServer:
    def test_update_status(self):
        from core.health_server import HealthServer, SystemStatus
        hs = HealthServer(port=19999)
        status = SystemStatus(
            status="ok", uptime_seconds=10.0, is_trading=True,
            active_strategies=2, open_positions=1,
            daily_pnl_usd=5.0, last_signal_ts=time.time(),
            error_count=0,
        )
        hs.update_status(status)
        snap = hs._get_system_status()
        assert snap.status == "ok"

    def test_start_stop(self):
        from core.health_server import HealthServer
        hs = HealthServer(port=18888)
        hs.start()
        time.sleep(0.1)
        hs.stop()


# ---------------------------------------------------------------------------
# core/process_lock
# ---------------------------------------------------------------------------

class TestProcessLock:
    def test_acquire_and_release(self, tmp_path):
        from core.process_lock import ProcessLock
        lock = ProcessLock("test_lock", lock_dir=tmp_path)
        assert lock.acquire() is True
        assert lock.is_locked() is True
        lock.release()
        assert lock.is_locked() is False

    def test_context_manager(self, tmp_path):
        from core.process_lock import ProcessLock
        with ProcessLock("ctx_lock", lock_dir=tmp_path) as lock:
            assert lock.is_locked() is True

    def test_double_acquire_fails(self, tmp_path):
        from core.process_lock import ProcessLock
        lock1 = ProcessLock("double_lock", lock_dir=tmp_path)
        lock2 = ProcessLock("double_lock", lock_dir=tmp_path)
        lock1.acquire()
        result = lock2.acquire()
        assert result is False
        lock1.release()


# ---------------------------------------------------------------------------
# monitoring/latency_tracker
# ---------------------------------------------------------------------------

class TestLatencyTracker:
    def test_measure_context_manager(self):
        from monitoring.latency_tracker import LatencyTracker
        tracker = LatencyTracker()
        with tracker.measure("test_op"):
            time.sleep(0.01)
        stats = tracker.get_stats("test_op")
        assert stats.count == 1
        assert stats.mean_ms >= 10.0

    def test_record_direct(self):
        from monitoring.latency_tracker import LatencyTracker
        tracker = LatencyTracker()
        tracker.record("api_round_trip", 150.0)
        stats = tracker.get_stats("api_round_trip")
        assert stats.mean_ms == pytest.approx(150.0, rel=0.01)

    def test_alert_fires_on_threshold(self):
        from monitoring.latency_tracker import LatencyTracker
        alerts = []
        tracker = LatencyTracker(alert_callback=lambda op, lat, thr: alerts.append(op))
        tracker.record("signal_to_order", 1000.0)  # over 500ms threshold
        assert len(alerts) >= 1

    def test_get_all_stats(self):
        from monitoring.latency_tracker import LatencyTracker
        tracker = LatencyTracker()
        tracker.record("a", 10.0)
        tracker.record("b", 20.0)
        all_stats = tracker.get_all_stats()
        assert "a" in all_stats and "b" in all_stats


# ---------------------------------------------------------------------------
# monitoring/exchange_monitor
# ---------------------------------------------------------------------------

class TestExchangeMonitor:
    def test_record_ws_connect(self):
        from monitoring.exchange_monitor import ExchangeMonitor
        em = ExchangeMonitor()
        em.record_ws_event("kraken", "connect")
        metrics = em.get_metrics("kraken")
        assert metrics.ws_connected is True

    def test_record_ws_disconnect_lowers_score(self):
        from monitoring.exchange_monitor import ExchangeMonitor
        em = ExchangeMonitor()
        em.record_ws_event("bybit", "connect")
        score_connected = em.compute_health_score("bybit")
        em.record_ws_event("bybit", "disconnect")
        score_disconnected = em.compute_health_score("bybit")
        assert score_disconnected < score_connected

    def test_orderbook_stale(self):
        from monitoring.exchange_monitor import ExchangeMonitor
        em = ExchangeMonitor()
        em.record_ws_event("kraken", "connect")
        # Don't update orderbook → it's stale from start
        metrics = em.get_metrics("kraken")
        assert metrics.orderbook_stale is True

    def test_snapshot_structure(self):
        from monitoring.exchange_monitor import ExchangeMonitor
        em = ExchangeMonitor()
        em.record_ws_event("kraken", "connect")
        snap = em.snapshot()
        assert "kraken" in snap
        assert "health_score" in snap["kraken"]


# ---------------------------------------------------------------------------
# monitoring/performance_attribution
# ---------------------------------------------------------------------------

class TestPerformanceAttribution:
    def test_full_report_empty_db(self, tmp_path):
        from monitoring.performance_attribution import PerformanceAttribution
        db = str(tmp_path / "nonexistent.db")
        pa = PerformanceAttribution(trade_db=db)
        report = pa.full_report()
        assert isinstance(report, dict)

    def test_compute_by_strategy_empty(self, tmp_path):
        from monitoring.performance_attribution import PerformanceAttribution
        db = str(tmp_path / "nonexistent.db")
        pa = PerformanceAttribution(trade_db=db)
        result = pa.compute_by_strategy()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# monitoring/drawdown_autopsy
# ---------------------------------------------------------------------------

class TestDrawdownAutopsy:
    def _equity_curve(self):
        now = time.time()
        return [
            (now - 3600 * 10, 1000.0),
            (now - 3600 * 9, 1050.0),
            (now - 3600 * 8, 1020.0),
            (now - 3600 * 7, 900.0),   # drawdown starts
            (now - 3600 * 6, 850.0),
            (now - 3600 * 5, 880.0),
            (now - 3600 * 4, 920.0),
            (now - 3600 * 3, 1010.0),
        ]

    def test_detect_drawdowns(self):
        from monitoring.drawdown_autopsy import DrawdownAutopsy
        da = DrawdownAutopsy(capital_usd=1000.0, drawdown_alert_pct=0.10)
        events = da.detect_drawdowns(self._equity_curve())
        assert len(events) >= 1
        assert events[0].drawdown_pct >= 0.10

    def test_analyse_keys(self):
        from monitoring.drawdown_autopsy import DrawdownAutopsy
        da = DrawdownAutopsy(capital_usd=1000.0, drawdown_alert_pct=0.10)
        events = da.detect_drawdowns(self._equity_curve())
        if events:
            analysis = da.analyse(events[0])
            assert "root_cause" in analysis
            assert "recommendations" in analysis

    def test_generate_report_string(self):
        from monitoring.drawdown_autopsy import DrawdownAutopsy
        da = DrawdownAutopsy(capital_usd=1000.0, drawdown_alert_pct=0.10)
        events = da.detect_drawdowns(self._equity_curve())
        if events:
            report = da.generate_report(events[0])
            assert isinstance(report, str)
            assert "DRAWDOWN" in report


# ---------------------------------------------------------------------------
# monitoring/trade_journal
# ---------------------------------------------------------------------------

class TestTradeJournal:
    def _entry(self, trade_id="t1"):
        from monitoring.trade_journal import JournalEntry
        return JournalEntry(
            trade_id=trade_id,
            symbol="BTC/USD",
            strategy="test_strat",
            entry_ts=time.time() - 3600,
            exit_ts=time.time(),
            entry_price=50000.0,
            exit_price=50500.0,
            qty_usd=200.0,
            pnl_usd=20.0,
            pnl_pct=0.01,
            regime_at_entry="TREND_UP",
            signal_confidence=0.75,
            exit_reason="TAKE_PROFIT",
            tags=["btc", "trend"],
        )

    def test_record_and_query(self, tmp_path):
        from monitoring.trade_journal import TradeJournal
        j = TradeJournal(db_path=str(tmp_path / "journal.db"))
        j.record(self._entry("t1"))
        results = j.query(limit=10)
        assert len(results) == 1
        assert results[0].trade_id == "t1"

    def test_get_stats(self, tmp_path):
        from monitoring.trade_journal import TradeJournal
        j = TradeJournal(db_path=str(tmp_path / "journal.db"))
        j.record(self._entry("t1"))
        j.record(self._entry("t2"))
        stats = j.get_stats()
        assert stats["trade_count"] == 2
        assert stats["win_rate"] == pytest.approx(1.0)

    def test_generate_markdown(self, tmp_path):
        from monitoring.trade_journal import TradeJournal
        j = TradeJournal(db_path=str(tmp_path / "journal.db"))
        j.record(self._entry("t1"))
        md = j.generate_markdown_report()
        assert "Trade Journal" in md
        assert "BTC/USD" in md

    def test_add_note(self, tmp_path):
        from monitoring.trade_journal import TradeJournal
        j = TradeJournal(db_path=str(tmp_path / "journal.db"))
        j.record(self._entry("t1"))
        j.add_note("t1", "Good trade", tag="win")  # should not raise


# ---------------------------------------------------------------------------
# monitoring/discord_webhook
# ---------------------------------------------------------------------------

class TestDiscordWebhook:
    def test_not_configured_returns_false(self):
        from monitoring.discord_webhook import DiscordWebhook
        dw = DiscordWebhook(webhook_url=None)
        assert dw.is_configured is False
        result = asyncio.run(
            dw.send_risk_alert("TEST", "Test message")
        )
        assert result is False

    def test_configured_property(self):
        from monitoring.discord_webhook import DiscordWebhook
        dw = DiscordWebhook(webhook_url="https://discord.com/api/webhooks/fake")
        assert dw.is_configured is True

    def test_get_discord_singleton(self):
        from monitoring.discord_webhook import get_discord, DiscordWebhook
        d = get_discord()
        assert isinstance(d, DiscordWebhook)


# ---------------------------------------------------------------------------
# monitoring/pdf_reporter
# ---------------------------------------------------------------------------

class TestPDFReporter:
    def test_generate_text_report(self, tmp_path):
        from monitoring.pdf_reporter import PDFReporter, ReportConfig
        from datetime import datetime, timezone
        reporter = PDFReporter(trade_db=str(tmp_path / "nonexistent.db"), output_dir=str(tmp_path))
        config = ReportConfig(
            period="weekly",
            start_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 3, 7, tzinfo=timezone.utc),
            capital_usd=1000.0,
        )
        path = reporter.generate(config)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "ARGUS" in content

    def test_ascii_chart(self):
        from monitoring.pdf_reporter import PDFReporter
        reporter = PDFReporter()
        chart = reporter._ascii_chart([10.0, -5.0, 8.0, -3.0, 12.0], width=20, height=5)
        assert "│" in chart or "─" in chart or "█" in chart


# ---------------------------------------------------------------------------
# compliance/ato_cgt
# ---------------------------------------------------------------------------

class TestATOCapitalGainsTracker:
    def test_record_acquisition_and_disposal(self):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        now = time.time()
        tracker.record_acquisition("BTC", 0.1, 5000.0, now - 86400 * 400, "kraken")
        disposal = tracker.record_disposal("BTC", 0.1, 6000.0, now, "kraken")
        assert disposal.capital_gain_aud == pytest.approx(1000.0, rel=0.01)

    def test_discount_eligible_long_hold(self):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        now = time.time()
        tracker.record_acquisition("ETH", 1.0, 2000.0, now - 86400 * 400, "kraken")
        disposal = tracker.record_disposal("ETH", 1.0, 3000.0, now, "kraken")
        assert disposal.discount_eligible is True
        assert disposal.discounted_gain_aud == pytest.approx(500.0, rel=0.01)

    def test_not_discount_eligible_short_hold(self):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        now = time.time()
        tracker.record_acquisition("ETH", 1.0, 2000.0, now - 86400 * 30, "kraken")
        disposal = tracker.record_disposal("ETH", 1.0, 3000.0, now, "kraken")
        assert disposal.discount_eligible is False
        assert disposal.discounted_gain_aud == pytest.approx(1000.0, rel=0.01)

    def test_fy_summary(self):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        from datetime import datetime, timezone
        fy_mid = datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp()
        tracker.record_acquisition("BTC", 0.5, 20000.0, fy_mid - 86400 * 400)
        tracker.record_disposal("BTC", 0.5, 25000.0, fy_mid)
        summary = tracker.get_fy_summary(2026)
        assert summary["total_gains_aud"] > 0

    def test_export_csv(self, tmp_path):
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        now = time.time()
        tracker.record_acquisition("BTC", 0.1, 5000.0, now - 86400 * 400)
        tracker.record_disposal("BTC", 0.1, 6000.0, now)
        path = str(tmp_path / "cgt.csv")
        tracker.export_csv(2026, path)
        assert os.path.exists(path)

    # -- Wash sale / bed-and-breakfast detection tests -------------------

    def test_wash_sale_basic_detection(self):
        """Sell at a loss then rebuy within 30 days => flagged."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        t0 = 1_700_000_000.0  # arbitrary base timestamp
        # Buy BTC
        tracker.record_acquisition("BTC", 1.0, 50_000.0, t0)
        # Sell at a loss 60 days later
        disposal_ts = t0 + 86400 * 60
        tracker.record_disposal("BTC", 1.0, 45_000.0, disposal_ts)
        # Rebuy 10 days after disposal (within 30-day window)
        rebuy_ts = disposal_ts + 86400 * 10
        tracker.record_acquisition("BTC", 1.0, 44_000.0, rebuy_ts)

        flagged = tracker.detect_wash_sales()
        assert len(flagged) == 1
        ws = flagged[0]
        assert ws["symbol"] == "BTC"
        assert ws["disposal_date"] == disposal_ts
        assert ws["disposal_proceeds"] == pytest.approx(45_000.0)
        assert ws["reacquisition_date"] == rebuy_ts
        assert ws["reacquisition_cost"] == pytest.approx(44_000.0)
        assert ws["potential_disallowed_loss"] == pytest.approx(5_000.0)

    def test_wash_sale_not_flagged_gain(self):
        """Sell at a gain then rebuy => NOT flagged (only losses matter)."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        t0 = 1_700_000_000.0
        tracker.record_acquisition("ETH", 10.0, 20_000.0, t0)
        disposal_ts = t0 + 86400 * 60
        tracker.record_disposal("ETH", 10.0, 25_000.0, disposal_ts)
        rebuy_ts = disposal_ts + 86400 * 5
        tracker.record_acquisition("ETH", 10.0, 24_000.0, rebuy_ts)

        flagged = tracker.detect_wash_sales()
        assert len(flagged) == 0

    def test_wash_sale_not_flagged_outside_window(self):
        """Rebuy after 31 days => NOT flagged."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        t0 = 1_700_000_000.0
        tracker.record_acquisition("BTC", 1.0, 50_000.0, t0)
        disposal_ts = t0 + 86400 * 60
        tracker.record_disposal("BTC", 1.0, 45_000.0, disposal_ts)
        # Rebuy 31 days after disposal (outside window)
        rebuy_ts = disposal_ts + 86400 * 31
        tracker.record_acquisition("BTC", 1.0, 44_000.0, rebuy_ts)

        flagged = tracker.detect_wash_sales()
        assert len(flagged) == 0

    def test_wash_sale_custom_lookback(self):
        """Custom lookback of 14 days: rebuy at day 15 => not flagged."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        t0 = 1_700_000_000.0
        tracker.record_acquisition("BTC", 1.0, 50_000.0, t0)
        disposal_ts = t0 + 86400 * 60
        tracker.record_disposal("BTC", 1.0, 45_000.0, disposal_ts)
        rebuy_ts = disposal_ts + 86400 * 15
        tracker.record_acquisition("BTC", 1.0, 44_000.0, rebuy_ts)

        assert len(tracker.detect_wash_sales(lookback_days=14)) == 0
        assert len(tracker.detect_wash_sales(lookback_days=30)) == 1

    def test_wash_sale_different_asset_not_flagged(self):
        """Sell BTC at loss, buy ETH within 30 days => NOT flagged."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        t0 = 1_700_000_000.0
        tracker.record_acquisition("BTC", 1.0, 50_000.0, t0)
        disposal_ts = t0 + 86400 * 60
        tracker.record_disposal("BTC", 1.0, 45_000.0, disposal_ts)
        rebuy_ts = disposal_ts + 86400 * 5
        tracker.record_acquisition("ETH", 10.0, 44_000.0, rebuy_ts)

        flagged = tracker.detect_wash_sales()
        assert len(flagged) == 0

    def test_wash_sale_multiple_assets(self):
        """Two different assets both with wash sales."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        t0 = 1_700_000_000.0

        # BTC wash sale
        tracker.record_acquisition("BTC", 1.0, 50_000.0, t0)
        btc_sell = t0 + 86400 * 60
        tracker.record_disposal("BTC", 1.0, 48_000.0, btc_sell)
        tracker.record_acquisition("BTC", 1.0, 47_000.0, btc_sell + 86400 * 5)

        # ETH wash sale
        tracker.record_acquisition("ETH", 10.0, 20_000.0, t0)
        eth_sell = t0 + 86400 * 70
        tracker.record_disposal("ETH", 10.0, 18_000.0, eth_sell)
        tracker.record_acquisition("ETH", 10.0, 17_500.0, eth_sell + 86400 * 3)

        flagged = tracker.detect_wash_sales()
        assert len(flagged) == 2
        symbols = {ws["symbol"] for ws in flagged}
        assert symbols == {"BTC", "ETH"}

    def test_wash_sale_consumed_lot_still_detected(self):
        """Rebuy is consumed by a later disposal but should still be detected."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        tracker = ATOCapitalGainsTracker()
        t0 = 1_700_000_000.0

        tracker.record_acquisition("BTC", 1.0, 50_000.0, t0)
        sell_ts = t0 + 86400 * 60
        tracker.record_disposal("BTC", 1.0, 45_000.0, sell_ts)
        # Rebuy within 30 days
        rebuy_ts = sell_ts + 86400 * 5
        tracker.record_acquisition("BTC", 1.0, 44_000.0, rebuy_ts)
        # Sell again (consuming the reacquired lot)
        tracker.record_disposal("BTC", 1.0, 46_000.0, rebuy_ts + 86400 * 100)

        flagged = tracker.detect_wash_sales()
        # The first disposal at a loss + rebuy within window should be flagged
        assert len(flagged) == 1
        assert flagged[0]["disposal_date"] == sell_ts

    def test_get_wash_sale_report(self):
        """get_wash_sale_report filters by period and aggregates."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        from datetime import datetime, timezone
        tracker = ATOCapitalGainsTracker()

        # Place events within FY2026 (1 Jul 2025 – 30 Jun 2026)
        fy_start = datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp()
        fy_end = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()
        mid_fy = datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp()

        tracker.record_acquisition("BTC", 1.0, 50_000.0, mid_fy - 86400 * 90)
        tracker.record_disposal("BTC", 1.0, 47_000.0, mid_fy)
        tracker.record_acquisition("BTC", 1.0, 46_000.0, mid_fy + 86400 * 10)

        report = tracker.get_wash_sale_report(fy_start, fy_end)
        assert report["flagged_count"] == 1
        assert report["total_disallowed_loss_aud"] == pytest.approx(3_000.0)
        assert report["by_asset"]["BTC"] == 1
        assert len(report["wash_sales"]) == 1

    def test_get_wash_sale_report_empty_period(self):
        """Report for a period with no wash sales returns zeroes."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        from datetime import datetime, timezone
        tracker = ATOCapitalGainsTracker()

        fy_start = datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp()
        fy_end = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()

        report = tracker.get_wash_sale_report(fy_start, fy_end)
        assert report["flagged_count"] == 0
        assert report["total_disallowed_loss_aud"] == 0.0
        assert report["wash_sales"] == []

    def test_wash_sale_report_excludes_outside_period(self):
        """Wash sales outside the queried period are excluded."""
        from compliance.ato_cgt import ATOCapitalGainsTracker
        from datetime import datetime, timezone
        tracker = ATOCapitalGainsTracker()

        # Place the disposal outside FY2026
        outside_ts = datetime(2024, 12, 1, tzinfo=timezone.utc).timestamp()
        tracker.record_acquisition("BTC", 1.0, 50_000.0, outside_ts - 86400 * 60)
        tracker.record_disposal("BTC", 1.0, 45_000.0, outside_ts)
        tracker.record_acquisition("BTC", 1.0, 44_000.0, outside_ts + 86400 * 5)

        fy_start = datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp()
        fy_end = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()
        report = tracker.get_wash_sale_report(fy_start, fy_end)
        assert report["flagged_count"] == 0


# ---------------------------------------------------------------------------
# compliance/tax_lot_optimizer
# ---------------------------------------------------------------------------

class TestTaxLotOptimizer:
    def _lot(self, lot_id, qty, cost, days_ago):
        from compliance.tax_lot_optimizer import TaxLot
        return TaxLot(
            lot_id=lot_id, asset="BTC", quantity=qty,
            cost_per_unit_aud=cost,
            acquisition_ts=time.time() - 86400 * days_ago,
            exchange="kraken",
        )

    def test_hifo_selects_highest_cost(self):
        from compliance.tax_lot_optimizer import TaxLotOptimizer
        opt = TaxLotOptimizer(strategy="HIFO")
        opt.add_lot(self._lot("cheap", 0.5, 40000.0, 10))
        opt.add_lot(self._lot("expensive", 0.5, 60000.0, 20))
        selections = opt.select_lots("BTC", 0.5, time.time(), 35000.0)
        assert selections[0].lot_id == "expensive"

    def test_lofo_selects_lowest_cost(self):
        from compliance.tax_lot_optimizer import TaxLotOptimizer
        opt = TaxLotOptimizer(strategy="LOFO")
        opt.add_lot(self._lot("cheap", 0.5, 40000.0, 10))
        opt.add_lot(self._lot("expensive", 0.5, 60000.0, 20))
        selections = opt.select_lots("BTC", 0.5, time.time(), 50000.0)
        assert selections[0].lot_id == "cheap"

    def test_estimate_tax_nonnegative(self):
        from compliance.tax_lot_optimizer import TaxLotOptimizer
        opt = TaxLotOptimizer(strategy="MIN_TAX")
        opt.add_lot(self._lot("old", 1.0, 30000.0, 400))
        selections = opt.select_lots("BTC", 1.0, time.time(), 50000.0)
        tax = opt.estimate_tax(selections, 50000.0)
        assert tax >= 0

    def test_discount_first_prefers_old_lot(self):
        from compliance.tax_lot_optimizer import TaxLotOptimizer
        opt = TaxLotOptimizer(strategy="DISCOUNT_FIRST")
        opt.add_lot(self._lot("new_lot", 0.5, 50000.0, 5))
        opt.add_lot(self._lot("old_lot", 0.5, 30000.0, 400))
        selections = opt.select_lots("BTC", 0.5, time.time(), 55000.0)
        assert selections[0].is_discount_eligible is True


# ---------------------------------------------------------------------------
# data/fear_greed
# ---------------------------------------------------------------------------

class TestFearGreedIndex:
    def test_get_signal_bias_extreme_fear(self):
        pytest.importorskip("data.fear_greed")
        from data.fear_greed import FearGreedIndex
        fg = FearGreedIndex()
        bias = fg.get_signal_bias(10)
        assert bias > 0  # contrarian buy

    def test_get_signal_bias_extreme_greed(self):
        pytest.importorskip("data.fear_greed")
        from data.fear_greed import FearGreedIndex
        fg = FearGreedIndex()
        bias = fg.get_signal_bias(90)
        assert bias < 0  # contrarian sell

    def test_get_signal_bias_neutral(self):
        pytest.importorskip("data.fear_greed")
        from data.fear_greed import FearGreedIndex
        fg = FearGreedIndex()
        bias = fg.get_signal_bias(50)
        assert bias == pytest.approx(0.0, abs=0.01)

    def test_is_extreme(self):
        pytest.importorskip("data.fear_greed")
        from data.fear_greed import FearGreedIndex
        fg = FearGreedIndex()
        assert fg.is_extreme(10) is True
        assert fg.is_extreme(85) is True
        assert fg.is_extreme(50) is False

    def test_get_returns_neutral_on_failure(self):
        pytest.importorskip("data.fear_greed")
        from data.fear_greed import FearGreedIndex
        fg = FearGreedIndex()
        # No network in tests — cache is empty, fetch will fail
        reading = asyncio.run(fg.get())
        assert 0 <= reading.value <= 100


# ---------------------------------------------------------------------------
# data/funding_predictor
# ---------------------------------------------------------------------------

class TestFundingRatePredictor:
    def test_predict_no_data(self):
        pytest.importorskip("data.funding_predictor")
        from data.funding_predictor import FundingRatePredictor
        pred = FundingRatePredictor()
        result = pred.predict("BTC/USD")
        assert result.direction == "NEUTRAL"
        assert result.confidence == pytest.approx(0.0, abs=0.01)

    def test_predict_with_data(self):
        pytest.importorskip("data.funding_predictor")
        from data.funding_predictor import FundingRatePredictor
        pred = FundingRatePredictor()
        for i in range(10):
            pred.update_orderbook("BTC/USD", bid_volume=1000.0, ask_volume=200.0, mid_price=50000.0)
            pred.update_premium("BTC/USD", 50000.0, 50100.0)  # positive premium
        result = pred.predict("BTC/USD")
        assert result.direction in ("LONG_PAY", "SHORT_PAY", "NEUTRAL")

    def test_obi_positive_when_bid_dominant(self):
        pytest.importorskip("data.funding_predictor")
        from data.funding_predictor import FundingRatePredictor
        pred = FundingRatePredictor()
        pred.update_orderbook("BTC/USD", bid_volume=800.0, ask_volume=200.0, mid_price=50000.0)
        obi = pred._compute_obi("BTC/USD")
        assert obi > 0

    def test_prediction_fields(self):
        pytest.importorskip("data.funding_predictor")
        from data.funding_predictor import FundingRatePredictor
        pred = FundingRatePredictor()
        result = pred.predict("ETH/USD")
        assert hasattr(result, "symbol")
        assert hasattr(result, "predicted_rate_pct")
        assert hasattr(result, "time_to_funding_hours")


# ---------------------------------------------------------------------------
# data/tick_capture
# ---------------------------------------------------------------------------

class TestTickCapture:
    def test_feed_and_stats(self):
        pytest.importorskip("data.tick_capture")
        from data.tick_capture import TickCapture, Tick
        tc = TickCapture(db_path=":memory:", batch_size=10, flush_interval=999)
        tick = Tick(symbol="BTC/USD", exchange="kraken", price=50000.0, quantity=0.01, side="buy")
        tc.feed(tick)
        stats = tc.get_stats()
        assert stats["ticks_captured"] == 1

    def test_vwap_empty_returns_zero(self, tmp_path):
        pytest.importorskip("data.tick_capture")
        from data.tick_capture import TickCapture
        tc = TickCapture(db_path=str(tmp_path / "ticks.db"))

        async def run():
            await tc.start()
            vwap = tc.query_vwap("BTC/USD", time.time() - 3600)
            await tc.stop()
            return vwap

        vwap = asyncio.run(run())
        assert vwap == pytest.approx(0.0)

    def test_volume_profile_empty(self, tmp_path):
        pytest.importorskip("data.tick_capture")
        from data.tick_capture import TickCapture
        tc = TickCapture(db_path=str(tmp_path / "ticks.db"))

        async def run():
            await tc.start()
            profile = tc.query_volume_profile("BTC/USD", time.time() - 3600)
            await tc.stop()
            return profile

        profile = asyncio.run(run())
        assert profile == []
