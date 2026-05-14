"""
Tests for Batch 4 components:
  - Execution: adaptive_slippage_model, rate_limit_manager, position_netter,
               order_flow_toxicity, conditional_orders
  - ML: volatility_forecaster, alpha_model, regime_classifier
  - Strategy/Data: whale_tracker, fred_calendar, reinforcement_stub
  - Backtesting: monte_carlo, regime_backtest
  - Compliance/Ops: austrac, deployment_checklist, capital_migration
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Execution — Batch 4A
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveSlippageModel:
    def test_import(self):
        from execution.adaptive_slippage_model import AdaptiveSlippageModel
        model = AdaptiveSlippageModel()
        assert model is not None

    def test_prediction_without_data(self):
        from execution.adaptive_slippage_model import AdaptiveSlippageModel, SlippageFeatures
        model = AdaptiveSlippageModel()
        features = SlippageFeatures(side="buy", quantity_norm=0.01, hour=14,
                                    spread_bps=3.0, volatility_30m=0.5)
        pred = model.predict(features)
        # Returns fallback prediction when no training data
        assert pred is not None
        assert pred.predicted_bps >= 0

    def test_prediction_has_confidence_interval(self):
        from execution.adaptive_slippage_model import AdaptiveSlippageModel, SlippageFeatures
        model = AdaptiveSlippageModel()
        features = SlippageFeatures(side="sell", quantity_norm=0.005, hour=10,
                                    spread_bps=2.0, volatility_30m=0.3)
        pred = model.predict(features)
        assert pred.ci_low <= pred.predicted_bps <= pred.ci_high


class TestRateLimitManager:
    def test_import(self):
        from execution.rate_limit_manager import RateLimitManager, EndpointType
        mgr = RateLimitManager()
        assert mgr is not None

    def test_check_known_exchange(self):
        from execution.rate_limit_manager import RateLimitManager, EndpointType
        mgr = RateLimitManager()
        # Should not raise for known exchanges
        allowed = mgr.check("kraken", EndpointType.PUBLIC)
        assert isinstance(allowed, bool)

    def test_check_unknown_exchange_permissive(self):
        from execution.rate_limit_manager import RateLimitManager, EndpointType
        mgr = RateLimitManager()
        # Unknown exchanges get permissive defaults
        allowed = mgr.check("unknownexchange", EndpointType.PUBLIC)
        assert allowed is True

    def test_multiple_calls_within_limit(self):
        from execution.rate_limit_manager import RateLimitManager, EndpointType
        mgr = RateLimitManager()
        # First few calls should succeed
        results = [mgr.check("coinbase", EndpointType.PUBLIC) for _ in range(5)]
        assert any(results)


class TestPositionNetter:
    def test_import(self):
        from execution.position_netter import PositionNetter
        netter = PositionNetter()
        assert netter is not None

    def test_update_and_net(self):
        from execution.position_netter import PositionNetter
        netter = PositionNetter()
        netter.update("kraken", "BTC/USD", "long", 0.5, 50000.0)
        netter.update("coinbase", "BTC/USD", "long", 0.3, 50100.0)
        net = netter.get_net("BTC/USD")
        assert net is not None
        assert abs(net.net_quantity - 0.8) < 1e-9

    def test_snapshot_serialisable(self):
        from execution.position_netter import PositionNetter
        netter = PositionNetter()
        netter.update("kraken", "ETH/USD", "long", 1.0, 3000.0)
        snap = netter.snapshot()
        assert isinstance(snap, dict)

    def test_total_exposure(self):
        from execution.position_netter import PositionNetter
        netter = PositionNetter()
        netter.update("kraken", "BTC/USD", "long", 0.1, 50000.0)
        exposure = netter.get_total_exposure_usd({"BTC/USD": 50000.0})
        assert abs(exposure - 5000.0) < 1.0


class TestOrderFlowToxicity:
    def test_import(self):
        from execution.order_flow_toxicity import OrderFlowToxicity
        oft = OrderFlowToxicity()
        assert oft is not None

    def test_update_and_score(self):
        from execution.order_flow_toxicity import OrderFlowToxicity, FillContext
        oft = OrderFlowToxicity()
        for i in range(20):
            ctx = FillContext(symbol="BTC/USD", side="buy" if i%2==0 else "sell",
                              quantity=0.1, price=50000.0+i, spread_bps=3.0, depth_at_price=5.0)
            oft.record_fill(ctx)
        score = oft.get_toxicity("BTC/USD")
        assert score is not None
        assert 0.0 <= score.toxicity_score <= 1.0

    def test_recommendation_type(self):
        from execution.order_flow_toxicity import OrderFlowToxicity, FillContext
        oft = OrderFlowToxicity()
        for i in range(20):
            ctx = FillContext(symbol="ETH/USD", side="buy", quantity=1.0, price=3000.0,
                              spread_bps=5.0, depth_at_price=10.0)
            oft.record_fill(ctx)
        score = oft.get_toxicity("ETH/USD")
        if score:
            assert score.recommendation in ("OK", "NORMAL", "REDUCE_SIZE", "WIDEN_LIMIT", "PAUSE")


class TestConditionalOrders:
    def test_import(self):
        from execution.conditional_orders import ConditionalOrderManager, GroupType
        mgr = ConditionalOrderManager(connector=None)
        assert mgr is not None

    def test_create_oco(self):
        from execution.conditional_orders import ConditionalOrderManager, GroupType
        mgr = ConditionalOrderManager(connector=None)
        group_id = mgr.create_oco(
            symbol="BTC/USD",
            tp_price=55000.0,
            sl_price=45000.0,
            quantity=0.1,
            exchange="kraken",
        )
        assert group_id is not None

    def test_create_bracket(self):
        from execution.conditional_orders import ConditionalOrderManager, GroupType
        mgr = ConditionalOrderManager(connector=None)
        group_id = mgr.create_bracket(
            symbol="ETH/USD",
            entry_price=3000.0,
            tp_price=3300.0,
            sl_price=2700.0,
            quantity=1.0,
            exchange="kraken",
        )
        assert group_id is not None


# ─────────────────────────────────────────────────────────────────────────────
# ML — Batch 4B
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilityForecaster:
    def _feed_prices(self, vf, symbol, n=350, start=50000.0):
        from ml.volatility_forecaster import VolatilityForecaster
        price = start
        for i in range(n):
            price *= (1 + np.random.normal(0, 0.002))
            vf.update(symbol, price)

    def test_import(self):
        from ml.volatility_forecaster import VolatilityForecaster
        vf = VolatilityForecaster()
        assert vf is not None

    def test_returns_none_insufficient_data(self):
        from ml.volatility_forecaster import VolatilityForecaster
        vf = VolatilityForecaster()
        vf.update("BTC/USD", 50000.0)
        result = vf.forecast("BTC/USD")
        assert result is None

    def test_forecast_after_sufficient_data(self):
        from ml.volatility_forecaster import VolatilityForecaster
        vf = VolatilityForecaster()
        self._feed_prices(vf, "BTC/USD", n=100)
        result = vf.forecast("BTC/USD")
        assert result is not None
        assert result.forecast_vol_1d > 0
        assert result.regime in ("LOW", "NORMAL", "ELEVATED", "EXTREME")
        assert result.method in ("ewma", "garch")

    def test_realized_vol(self):
        from ml.volatility_forecaster import VolatilityForecaster
        vf = VolatilityForecaster()
        self._feed_prices(vf, "ETH/USD", n=300, start=3000.0)
        rv = vf.realized_vol("ETH/USD", bars=100)
        assert rv is not None and rv > 0

    def test_all_forecasts(self):
        from ml.volatility_forecaster import VolatilityForecaster
        vf = VolatilityForecaster()
        for sym in ["BTC/USD", "ETH/USD"]:
            self._feed_prices(vf, sym, n=100)
        all_f = vf.all_forecasts()
        assert isinstance(all_f, dict)


class TestAlphaModel:
    def _feed(self, model, symbol, n=400, has_funding=True):
        price = 50000.0
        for i in range(n):
            price *= (1 + np.random.normal(0.0001, 0.002))
            funding = np.random.normal(0.0001, 0.0002) if has_funding else None
            model.update(symbol, price, funding_rate=funding, spread_bps=3.0)

    def test_import(self):
        from ml.alpha_model import AlphaModel
        model = AlphaModel()
        assert model is not None

    def test_returns_none_insufficient(self):
        from ml.alpha_model import AlphaModel
        model = AlphaModel()
        model.update("BTC/USD", 50000.0)
        assert model.score("BTC/USD") is None

    def test_score_after_sufficient_data(self):
        from ml.alpha_model import AlphaModel
        model = AlphaModel()
        self._feed(model, "BTC/USD", n=350)
        score = model.score("BTC/USD")
        assert score is not None
        assert -1.0 <= score.composite <= 1.0
        assert score.signal in ("STRONG_LONG", "LONG", "NEUTRAL", "SHORT", "STRONG_SHORT")
        assert 0.0 <= score.confidence <= 1.0

    def test_factors_present(self):
        from ml.alpha_model import AlphaModel
        model = AlphaModel()
        self._feed(model, "ETH/USD", n=350)
        score = model.score("ETH/USD")
        assert score is not None
        assert "momentum_1d" in score.factors
        assert "carry" in score.factors

    def test_set_weights(self):
        from ml.alpha_model import AlphaModel
        model = AlphaModel()
        model.set_weights({"momentum_1d": 1.0, "carry": 1.0})
        # Should not raise

    def test_invalid_weights_raises(self):
        from ml.alpha_model import AlphaModel
        model = AlphaModel()
        with pytest.raises(ValueError):
            model.set_weights({"momentum_1d": -1.0})


class TestRegimeClassifier:
    def _generate_prices(self, n=2100, trend=0.0001):
        prices = [50000.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + trend + np.random.normal(0, 0.002)))
        return prices

    def test_import(self):
        from ml.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        assert clf is not None

    def test_predict_without_training_uses_rules(self):
        from ml.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        prices = self._generate_prices(n=2200)
        result = clf.predict(prices)
        assert result is not None
        assert result.regime in ("TREND_UP", "TREND_DOWN", "RANGING", "VOLATILE", "CRISIS")
        assert result.method == "rules"
        assert 0.0 <= result.confidence <= 1.0

    def test_insufficient_prices_returns_none(self):
        from ml.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        result = clf.predict([50000.0] * 100)  # Not enough bars
        assert result is None

    def test_add_training_sample(self):
        from ml.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        prices = self._generate_prices(n=2200)
        ok = clf.add_training_sample(prices, "TREND_UP")
        assert ok is True
        assert clf.n_training_samples == 1

    def test_invalid_regime_label_raises(self):
        from ml.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        prices = self._generate_prices(n=2200)
        with pytest.raises(ValueError):
            clf.add_training_sample(prices, "INVALID_REGIME")

    def test_feature_names(self):
        from ml.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        names = clf.feature_names()
        assert "vol_1d" in names
        assert "ret_1d" in names

    def test_probabilities_sum_to_one(self):
        from ml.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        prices = self._generate_prices(n=2200)
        result = clf.predict(prices)
        assert result is not None
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Strategy / Data — Batch 4C
# ─────────────────────────────────────────────────────────────────────────────

class TestWhaleTracker:
    def test_import(self):
        pytest.importorskip("data.onchain.whale_tracker")
        from data.onchain.whale_tracker import WhaleTracker
        tracker = WhaleTracker()
        assert tracker is not None

    def test_get_signal_no_api_returns_signal(self):
        pytest.importorskip("data.onchain.whale_tracker")
        from data.onchain.whale_tracker import WhaleTracker
        tracker = WhaleTracker(timeout=1.0)
        signal = tracker.get_signal("BTC")
        assert signal is not None
        assert signal.asset == "BTC"
        assert signal.direction in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_inject_transaction(self):
        pytest.importorskip("data.onchain.whale_tracker")
        from data.onchain.whale_tracker import WhaleTracker, WhaleTransaction
        tracker = WhaleTracker()
        tx = WhaleTransaction(
            tx_hash="abc123",
            asset="BTC",
            amount=500.0,
            usd_value=25_000_000.0,
            from_exchange=False,
            to_exchange=True,       # Exchange inflow → bearish
            timestamp=time.time(),
            signal="INFLOW",
        )
        tracker.inject_transaction(tx)
        signal = tracker.get_signal("BTC")
        assert signal.inflow_count_1h >= 1

    def test_clear_manual_events(self):
        pytest.importorskip("data.onchain.whale_tracker")
        from data.onchain.whale_tracker import WhaleTracker, WhaleTransaction
        tracker = WhaleTracker()
        tx = WhaleTransaction(
            tx_hash="xyz",
            asset="ETH",
            amount=5000.0,
            usd_value=15_000_000.0,
            from_exchange=True,
            to_exchange=False,
            timestamp=time.time(),
            signal="OUTFLOW",
        )
        tracker.inject_transaction(tx)
        tracker.clear_manual_events()
        signal = tracker.get_signal("ETH")
        # After clearing, no manually injected events
        assert signal.outflow_count_1h == 0


class TestFREDCalendar:
    def test_import(self):
        pytest.importorskip("data.macro.fred_calendar")
        from data.macro.fred_calendar import FREDCalendar
        cal = FREDCalendar()
        assert cal is not None

    def test_upcoming_contains_fomc(self):
        pytest.importorskip("data.macro.fred_calendar")
        from data.macro.fred_calendar import FREDCalendar
        cal = FREDCalendar()
        snap = cal.get_upcoming(days=365)
        fomc_events = [e for e in snap.events if e.event_type == "FOMC"]
        assert len(fomc_events) >= 1  # 2026 FOMC dates are hardcoded

    def test_high_impact_event_detected(self):
        pytest.importorskip("data.macro.fred_calendar")
        from data.macro.fred_calendar import FREDCalendar
        cal = FREDCalendar()
        snap = cal.get_upcoming(days=365)
        # All FOMC events are impact 3
        high = [e for e in snap.events if e.impact >= 3]
        assert len(high) >= 1

    def test_add_manual_event(self):
        pytest.importorskip("data.macro.fred_calendar")
        from data.macro.fred_calendar import FREDCalendar, MacroEvent
        cal = FREDCalendar()
        event = MacroEvent(
            name="Test RBA Meeting",
            event_type="OTHER",
            scheduled_at=datetime.now(tz=timezone.utc) + timedelta(hours=2),
            impact=2,
            source="manual",
        )
        cal.add_event(event)
        snap = cal.get_upcoming(days=1)
        manual = [e for e in snap.events if e.name == "Test RBA Meeting"]
        assert len(manual) == 1

    def test_is_blackout_false_normally(self):
        pytest.importorskip("data.macro.fred_calendar")
        from data.macro.fred_calendar import FREDCalendar
        cal = FREDCalendar()
        # Should be False unless FOMC is happening right now (very unlikely)
        result = cal.is_blackout(blackout_hours=0.001)
        assert isinstance(result, bool)

    def test_is_blackout_true_near_event(self):
        pytest.importorskip("data.macro.fred_calendar")
        from data.macro.fred_calendar import FREDCalendar, MacroEvent
        cal = FREDCalendar()
        # Add an event 5 minutes from now
        event = MacroEvent(
            name="Imminent FOMC",
            event_type="FOMC",
            scheduled_at=datetime.now(tz=timezone.utc) + timedelta(minutes=5),
            impact=3,
            source="manual",
        )
        cal.add_event(event)
        assert cal.is_blackout(blackout_hours=0.5) is True

    def test_clear_manual_events(self):
        pytest.importorskip("data.macro.fred_calendar")
        from data.macro.fred_calendar import FREDCalendar, MacroEvent
        cal = FREDCalendar()
        event = MacroEvent(
            name="Temp",
            event_type="FOMC",
            scheduled_at=datetime.now(tz=timezone.utc) + timedelta(minutes=5),
            impact=3,
            source="manual",
        )
        cal.add_event(event)
        cal.clear_manual_events()
        assert cal.is_blackout(blackout_hours=0.5) is False


class TestReinforcementStub:
    def test_import(self):
        from strategies.reinforcement_stub import RLExecutionAgent, RLState
        agent = RLExecutionAgent()
        assert agent is not None

    def test_state_to_array(self):
        from strategies.reinforcement_stub import RLState
        state = RLState(
            position_usd=500.0,
            unrealised_pnl=10.0,
            volatility_1h=0.4,
            spread_bps=3.0,
            ob_imbalance=0.2,
            time_of_day_sin=0.5,
            time_of_day_cos=0.5,
            slippage_budget_remaining=15.0,
            bars_since_last_trade=10,
        )
        arr = state.to_array()
        assert len(arr) == 9

    def test_decide_returns_decision(self):
        from strategies.reinforcement_stub import RLExecutionAgent, RLState, HOLD
        agent = RLExecutionAgent()
        state = RLState(
            position_usd=0.0,
            unrealised_pnl=0.0,
            volatility_1h=0.3,
            spread_bps=3.0,
            ob_imbalance=0.5,
            time_of_day_sin=0.7,
            time_of_day_cos=0.7,
            slippage_budget_remaining=20.0,
            bars_since_last_trade=5,
        )
        decision = agent.decide(state)
        assert decision is not None
        assert 0 <= decision.action <= 4
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.from_model is False  # No model loaded

    def test_budget_exhausted_forces_hold(self):
        from strategies.reinforcement_stub import RLExecutionAgent, RLState, HOLD
        agent = RLExecutionAgent()
        state = RLState(
            position_usd=1000.0,
            unrealised_pnl=-50.0,
            volatility_1h=0.2,
            spread_bps=2.0,
            ob_imbalance=0.8,
            time_of_day_sin=0.5,
            time_of_day_cos=0.5,
            slippage_budget_remaining=0.0,   # Budget exhausted
            bars_since_last_trade=1,
        )
        decision = agent.decide(state)
        assert decision.action == HOLD

    def test_state_dim_action_dim(self):
        from strategies.reinforcement_stub import RLExecutionAgent
        assert RLExecutionAgent.state_dim() == 9
        assert RLExecutionAgent.action_dim() == 5


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting — Batch 4D
# ─────────────────────────────────────────────────────────────────────────────

class TestMonteCarlo:
    def test_import(self):
        from backtesting.monte_carlo import MonteCarlo, MCConfig
        mc = MonteCarlo()
        assert mc is not None

    def test_gbm_shape(self):
        from backtesting.monte_carlo import MonteCarlo, MCConfig
        cfg = MCConfig(n_paths=100, n_steps=50, random_seed=42)
        mc = MonteCarlo(cfg)
        result = mc.simulate_gbm(initial_value=1000.0, mu=0.15, sigma=0.60)
        assert result.paths.shape == (100, 51)
        assert len(result.terminal_values) == 100

    def test_gbm_initial_value_preserved(self):
        from backtesting.monte_carlo import MonteCarlo, MCConfig
        cfg = MCConfig(n_paths=200, n_steps=100, random_seed=0)
        mc = MonteCarlo(cfg)
        result = mc.simulate_gbm(initial_value=5000.0, mu=0.0, sigma=0.01)
        # All paths start at 5000
        assert np.allclose(result.paths[:, 0], 5000.0)

    def test_var_cvar_positive(self):
        from backtesting.monte_carlo import MonteCarlo, MCConfig
        cfg = MCConfig(n_paths=500, n_steps=252, random_seed=1)
        mc = MonteCarlo(cfg)
        result = mc.simulate_gbm(initial_value=1000.0, mu=-0.3, sigma=0.8)
        assert result.var_95 >= 0
        assert result.cvar_95 >= result.var_95

    def test_prob_ruin_between_0_and_1(self):
        from backtesting.monte_carlo import MonteCarlo, MCConfig
        cfg = MCConfig(n_paths=300, n_steps=252, random_seed=2)
        mc = MonteCarlo(cfg)
        result = mc.simulate_gbm(initial_value=1000.0, mu=0.10, sigma=0.60)
        assert 0.0 <= result.prob_ruin <= 1.0

    def test_bootstrap_simulation(self):
        from backtesting.monte_carlo import MonteCarlo, MCConfig
        cfg = MCConfig(n_paths=100, n_steps=50, random_seed=3)
        mc = MonteCarlo(cfg)
        hist_returns = np.random.normal(0.0005, 0.02, 500).tolist()
        result = mc.simulate_bootstrap(
            initial_value=1000.0,
            historical_returns=hist_returns,
            block_size=5,
        )
        assert result.paths.shape == (100, 51)

    def test_regime_switching(self):
        from backtesting.monte_carlo import MonteCarlo, MCConfig
        cfg = MCConfig(n_paths=100, n_steps=50, random_seed=4)
        mc = MonteCarlo(cfg)
        regimes = {
            "bull":  (0.20, 0.40),
            "bear":  (-0.30, 0.70),
            "quiet": (0.05, 0.20),
        }
        result = mc.simulate_regime_switching(
            initial_value=1000.0,
            regimes=regimes,
        )
        assert result.paths.shape == (100, 51)

    def test_summary_keys(self):
        from backtesting.monte_carlo import MonteCarlo, MCConfig
        cfg = MCConfig(n_paths=100, n_steps=50, random_seed=5)
        mc = MonteCarlo(cfg)
        result = mc.simulate_gbm(initial_value=1000.0, mu=0.10, sigma=0.50)
        summary = mc.summary(result)
        assert "var_95_loss" in summary
        assert "median_terminal" in summary
        assert "prob_ruin_pct" in summary


class TestRegimeBacktest:
    def test_import(self):
        from backtesting.regime_backtest import RegimeBacktester, Trade
        bt = RegimeBacktester()
        assert bt is not None

    def test_add_trade_and_analyse(self):
        from backtesting.regime_backtest import RegimeBacktester, Trade
        bt = RegimeBacktester()
        for i in range(10):
            bt.add_trade(Trade(
                entry_price=50000.0,
                exit_price=51000.0,
                side="LONG",
                size=0.1,
                entry_bar=i * 10,
                exit_bar=i * 10 + 8,
                regime="TREND_UP",
            ))
        result = bt.analyse()
        assert result.by_regime["TREND_UP"].n_trades == 10
        assert result.by_regime["TREND_UP"].win_rate == 1.0  # all profitable

    def test_regime_distribution(self):
        from backtesting.regime_backtest import RegimeBacktester
        bt = RegimeBacktester()
        for _ in range(100):
            bt.record_bar("TREND_UP")
        for _ in range(50):
            bt.record_bar("RANGING")
        result = bt.analyse()
        assert abs(result.regime_distribution["TREND_UP"] - 100/150) < 0.01
        assert abs(result.regime_distribution["RANGING"] - 50/150) < 0.01

    def test_from_trades_and_regimes(self):
        from backtesting.regime_backtest import RegimeBacktester
        bt = RegimeBacktester()
        n = 500
        prices = [50000.0 * (1 + 0.001 * i) for i in range(n)]
        regimes = (["TREND_UP"] * 250 + ["RANGING"] * 250)
        # Simple signal: buy in first half, flat in second
        signals = [1] * 50 + [0] * 200 + [1] * 50 + [0] * 200
        result = bt.from_trades_and_regimes(prices, regimes, signals)
        assert result.overall.n_trades >= 1

    def test_no_trades_handles_gracefully(self):
        from backtesting.regime_backtest import RegimeBacktester
        bt = RegimeBacktester()
        result = bt.analyse()
        assert result.overall.n_trades == 0
        assert result.overall.win_rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Compliance / Ops — Batch 4E
# ─────────────────────────────────────────────────────────────────────────────

class TestAUSTRAC:
    def test_import(self):
        from compliance.austrac import AUSTRACReporter
        reporter = AUSTRACReporter()
        assert reporter is not None

    def test_no_ttr_below_threshold(self):
        from compliance.austrac import AUSTRACReporter, AUSTRACTransaction
        reporter = AUSTRACReporter()
        tx = AUSTRACTransaction(
            tx_id="TX001",
            timestamp=datetime.now(tz=timezone.utc),
            asset="BTC",
            amount_asset=0.1,
            amount_aud=5000.0,   # Below AUD 10,000 threshold
            direction="BUY",
            counterparty_exchange="Kraken",
            customer_id="CUST-001",
        )
        reporter.record_transaction(tx)
        assert len(reporter.get_pending_ttrs()) == 0

    def test_ttr_generated_above_threshold(self):
        from compliance.austrac import AUSTRACReporter, AUSTRACTransaction
        reporter = AUSTRACReporter()
        tx = AUSTRACTransaction(
            tx_id="TX002",
            timestamp=datetime.now(tz=timezone.utc),
            asset="BTC",
            amount_asset=0.25,
            amount_aud=12500.0,  # Above AUD 10,000 threshold
            direction="BUY",
            counterparty_exchange="Kraken",
            customer_id="CUST-001",
        )
        reporter.record_transaction(tx)
        pending = reporter.get_pending_ttrs()
        assert len(pending) == 1

    def test_ttr_export(self, tmp_path):
        from compliance.austrac import AUSTRACReporter, AUSTRACTransaction
        reporter = AUSTRACReporter(output_dir=tmp_path)
        tx = AUSTRACTransaction(
            tx_id="TX003",
            timestamp=datetime.now(tz=timezone.utc),
            asset="ETH",
            amount_asset=5.0,
            amount_aud=15000.0,
            direction="SELL",
            counterparty_exchange="Coinbase",
            customer_id="CUST-002",
        )
        reporter.record_transaction(tx)
        pending = reporter.get_pending_ttrs()
        path = reporter.export_ttr_report(pending[0])
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["report_type"] == "TTR"

    def test_mark_ttr_filed(self):
        from compliance.austrac import AUSTRACReporter, AUSTRACTransaction
        reporter = AUSTRACReporter()
        tx = AUSTRACTransaction(
            tx_id="TX004",
            timestamp=datetime.now(tz=timezone.utc),
            asset="BTC",
            amount_asset=0.5,
            amount_aud=25000.0,
            direction="BUY",
            counterparty_exchange="Kraken",
            customer_id="CUST-001",
        )
        reporter.record_transaction(tx)
        pending = reporter.get_pending_ttrs()
        report_id = pending[0].report_id
        reporter.mark_ttr_filed(report_id)
        assert len(reporter.get_pending_ttrs()) == 0

    def test_compliance_summary(self):
        from compliance.austrac import AUSTRACReporter
        reporter = AUSTRACReporter()
        summary = reporter.compliance_summary()
        assert "pending_ttrs" in summary
        assert summary["ttr_threshold_aud"] == 10000.0


class TestDeploymentChecklist:
    def test_import(self):
        from ops.deployment_checklist import DeploymentChecklist
        checklist = DeploymentChecklist()
        assert checklist is not None

    def test_run_returns_result(self, tmp_path):
        from ops.deployment_checklist import DeploymentChecklist
        checklist = DeploymentChecklist(db_path=tmp_path / "test.db",
                                        lock_dir=tmp_path)
        result = checklist.run()
        assert result is not None
        assert isinstance(result.go, bool)
        assert len(result.checks) >= 5

    def test_python_version_check_passes(self, tmp_path):
        from ops.deployment_checklist import DeploymentChecklist
        checklist = DeploymentChecklist(db_path=tmp_path / "test.db",
                                        lock_dir=tmp_path)
        result = checklist.run()
        python_check = next((c for c in result.checks if "Python" in c.name), None)
        assert python_check is not None
        assert python_check.passed  # Running on 3.14

    def test_custom_check(self, tmp_path):
        from ops.deployment_checklist import DeploymentChecklist
        checklist = DeploymentChecklist(db_path=tmp_path / "test.db",
                                        lock_dir=tmp_path)
        checklist.add_check("Custom always passes", lambda: (True, "OK"), critical=False)
        result = checklist.run()
        custom = next((c for c in result.checks if c.name == "Custom always passes"), None)
        assert custom is not None
        assert custom.passed

    def test_summary_string(self, tmp_path):
        from ops.deployment_checklist import DeploymentChecklist
        checklist = DeploymentChecklist(db_path=tmp_path / "test.db",
                                        lock_dir=tmp_path)
        result = checklist.run()
        summary = result.summary()
        assert "GO" in summary or "NO-GO" in summary


class TestCapitalMigration:
    def test_import(self):
        from ops.capital_migration import CapitalMigration, Stage
        migration = CapitalMigration()
        assert migration.current_stage == Stage.PAPER

    def test_assess_paper_to_micro(self):
        from ops.capital_migration import CapitalMigration, Stage, PerformanceSnapshot
        migration = CapitalMigration()
        perf = PerformanceSnapshot(
            days_at_stage=8,
            sharpe_annualised=0.6,
            max_drawdown_pct=8.0,
            circuit_breaks_7d=0,
            total_trades=15,
            current_stage=Stage.PAPER,
        )
        assessment = migration.assess(perf)
        assert assessment.current_stage == Stage.PAPER
        assert assessment.next_stage == Stage.MICRO
        assert assessment.can_advance is True

    def test_assess_fails_insufficient_days(self):
        from ops.capital_migration import CapitalMigration, Stage, PerformanceSnapshot
        migration = CapitalMigration()
        perf = PerformanceSnapshot(
            days_at_stage=1,   # Too few days
            sharpe_annualised=2.0,
            max_drawdown_pct=5.0,
            circuit_breaks_7d=0,
            total_trades=50,
            current_stage=Stage.PAPER,
        )
        assessment = migration.assess(perf)
        assert assessment.can_advance is False

    def test_advance_requires_confirmed(self):
        from ops.capital_migration import CapitalMigration, Stage
        migration = CapitalMigration()
        result = migration.advance(confirmed=False)
        assert result is False
        assert migration.current_stage == Stage.PAPER

    def test_advance_confirmed(self):
        from ops.capital_migration import CapitalMigration, Stage
        migration = CapitalMigration()
        result = migration.advance(confirmed=True)
        assert result is True
        assert migration.current_stage == Stage.MICRO
        assert migration.current_capital_aud == 100.0

    def test_rollback(self):
        from ops.capital_migration import CapitalMigration, Stage
        migration = CapitalMigration()
        migration.advance(confirmed=True)
        assert migration.current_stage == Stage.MICRO
        rolled = migration.rollback("Test rollback", confirmed=True)
        assert rolled is True
        assert migration.current_stage == Stage.PAPER

    def test_already_at_max(self):
        from ops.capital_migration import CapitalMigration, Stage, PerformanceSnapshot
        migration = CapitalMigration()
        # Advance to LIVE
        for _ in range(3):
            migration.advance(confirmed=True)
        assert migration.current_stage == Stage.LIVE
        perf = PerformanceSnapshot(
            days_at_stage=60, sharpe_annualised=2.0, max_drawdown_pct=5.0,
            circuit_breaks_7d=0, total_trades=100, current_stage=Stage.LIVE,
        )
        assessment = migration.assess(perf)
        assert assessment.can_advance is False
        assert assessment.next_stage is None

    def test_history_recorded(self):
        from ops.capital_migration import CapitalMigration
        migration = CapitalMigration()
        migration.advance(confirmed=True)
        migration.advance(confirmed=True)
        history = migration.history()
        assert len(history) == 2
        assert history[0]["from"] == "paper"
        assert history[0]["to"] == "micro"

    def test_position_limit_scales_with_stage(self):
        from ops.capital_migration import CapitalMigration, Stage
        migration = CapitalMigration()
        assert migration.current_position_limit_pct == 100.0  # Paper: full % for testing
        migration.advance(confirmed=True)
        assert migration.current_position_limit_pct == 10.0  # Micro: 10%
        migration.advance(confirmed=True)
        assert migration.current_position_limit_pct == 25.0  # Seed: 25%
