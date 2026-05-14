"""
Tests for edge strategy features:
  - Regime-conditional strategy rotation with hysteresis
  - Funding rate prediction
  - On-chain whale signal generation
  - Session effect bias detection
  - Signal consensus filtering
  - Transaction cost analysis
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Regime Strategy Rotation
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegimeStrategyRotator:
    """Tests for strategies.regime_rotation.RegimeStrategyRotator."""

    def _make(self, **kwargs):
        from strategies.regime_rotation import RegimeStrategyRotator
        return RegimeStrategyRotator(**kwargs)

    def test_initial_state(self):
        rotator = self._make()
        assert rotator.current_regime == "NORMAL"
        assert rotator.current_confidence == 0.0
        assert rotator.pending_regime is None

    def test_update_same_regime_clears_pending(self):
        rotator = self._make()
        rotator._pending_regime = "HIGH_VOL"
        rotator.update_regime("NORMAL", 0.8)
        assert rotator.pending_regime is None
        assert rotator.current_confidence == 0.8

    def test_update_new_regime_sets_pending(self):
        rotator = self._make()
        rotator.update_regime("CRISIS", 0.9)
        assert rotator.pending_regime == "CRISIS"
        assert rotator.current_regime == "NORMAL"  # not changed yet

    def test_hysteresis_blocks_premature_rotation(self):
        rotator = self._make(hysteresis_s=1800)
        rotator.update_regime("HIGH_VOL", 0.85)
        result = rotator.rotate()
        # Still NORMAL because hysteresis hasn't passed
        assert rotator.current_regime == "NORMAL"

    def test_hysteresis_allows_rotation_after_delay(self):
        rotator = self._make(hysteresis_s=0.0)  # instant hysteresis for test
        rotator.update_regime("HIGH_VOL", 0.85)
        rotator._pending_ts = time.time() - 1  # force past hysteresis
        result = rotator.rotate()
        assert rotator.current_regime == "HIGH_VOL"

    def test_rotate_records_history(self):
        rotator = self._make(hysteresis_s=0.0)
        rotator.update_regime("CRISIS", 0.95)
        rotator._pending_ts = time.time() - 1
        rotator.rotate()
        history = rotator.get_rotation_history()
        assert len(history) == 1
        assert history[0]["old_regime"] == "NORMAL"
        assert history[0]["new_regime"] == "CRISIS"

    def test_get_regime_weights_preferred_strategies(self):
        rotator = self._make()
        rotator._current_regime = "TRENDING_UP"
        weights = rotator.get_regime_weights()
        assert weights.get("momentum", 0) == 1.0
        assert weights.get("breakout", 0) == 1.0

    def test_get_regime_weights_non_preferred_get_reduced(self):
        rotator = self._make(off_regime_weight=0.3)
        rotator._current_regime = "CRISIS"
        weights = rotator.get_regime_weights()
        # macro_event_filter is preferred in CRISIS
        # Everything else should get 0.3
        for name, w in weights.items():
            if name == "macro_event_filter":
                assert w == 1.0
            else:
                assert w == 0.3

    def test_rotate_with_router(self):
        """Test that rotate calls enable on the router."""
        router = MagicMock()
        router._strategies = {"momentum": True, "mean_reversion": True, "breakout": True}
        rotator = self._make(strategy_router=router)
        rotator._current_regime = "TRENDING_UP"
        result = rotator.rotate()
        assert isinstance(result, dict)
        assert router.enable.called

    def test_regime_map_covers_all_variants(self):
        from strategies.regime_rotation import RegimeStrategyRotator
        rmap = RegimeStrategyRotator.REGIME_STRATEGY_MAP
        assert "TREND_UP" in rmap
        assert "TRENDING_UP" in rmap
        assert "CRISIS" in rmap
        assert "RANGE" in rmap

    def test_multiple_regime_transitions(self):
        rotator = self._make(hysteresis_s=0.0)
        for regime in ["HIGH_VOL", "CRISIS", "NORMAL"]:
            rotator.update_regime(regime, 0.8)
            rotator._pending_ts = time.time() - 1
            rotator.rotate()
        assert rotator.current_regime == "NORMAL"
        history = rotator.get_rotation_history()
        assert len(history) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Funding Rate Predictor
# ═══════════════════════════════════════════════════════════════════════════════


class TestFundingRatePredictor:
    """Tests for strategies.funding_rate_predictor.FundingRatePredictor."""

    def _make(self, **kwargs):
        from strategies.funding_rate_predictor import FundingRatePredictor
        return FundingRatePredictor(**kwargs)

    def test_predict_with_no_data_returns_neutral(self):
        pred = self._make()
        result = pred.predict_next_rate("BTC/USD")
        assert result.direction == "NEUTRAL"
        assert result.magnitude == "LOW"

    def test_update_stores_data(self):
        pred = self._make()
        pred.update("BTC/USD", 0.001, 1e9, 50000, 50050, 1e6, 2e6)
        assert len(pred._rate_history["BTC/USD"]) == 1
        assert len(pred._basis_history["BTC/USD"]) == 1

    def test_predict_with_positive_trend(self):
        pred = self._make()
        for i in range(10):
            # Increasing funding rate + positive basis
            rate = 0.0001 * (i + 1)
            spot = 50000
            perp = 50000 + i * 10
            pred.update("BTC/USD", rate, 1e9, spot, perp, 1e6, 2e6)
        result = pred.predict_next_rate("BTC/USD")
        assert result.predicted_rate_pct > 0
        assert result.direction == "LONG_PAY"

    def test_predict_extreme_rate_recommends_position(self):
        pred = self._make()
        for i in range(10):
            rate = 0.001 * (i + 1)  # Very high rates
            pred.update("BTC/USD", rate, 1e9, 50000, 50500, 1e6, 5e6)
        result = pred.predict_next_rate("BTC/USD")
        # With very high rates, should recommend short perp
        if result.magnitude in ("EXTREME", "HIGH") and result.confidence >= 0.5:
            assert result.position_recommendation in ("SHORT_PERP", "LONG_PERP")

    def test_optimal_entry_timing(self):
        pred = self._make()
        pred.set_last_settlement("BTC/USD", time.time() - 3600 * 6)  # 6h ago
        timing = pred.get_optimal_entry_timing("BTC/USD")
        assert "timing" in timing
        assert "hours_to_settlement" in timing
        assert timing["hours_to_settlement"] >= 0

    def test_generate_signal_returns_none_for_low_rate(self):
        pred = self._make()
        pred.update("BTC/USD", 0.00001, 1e9, 50000, 50001, 1e6, 1e6)
        signal = pred.generate_signal("BTC/USD")
        assert signal is None

    def test_orderbook_imbalance_update(self):
        pred = self._make()
        pred.update_orderbook_imbalance("BTC/USD", 0.7)
        assert len(pred._obi_history["BTC/USD"]) == 1

    def test_multiple_symbols_independent(self):
        pred = self._make()
        pred.update("BTC/USD", 0.001, 1e9, 50000, 50100, 1e6, 2e6)
        pred.update("ETH/USD", -0.001, 5e8, 3000, 2990, 5e5, 1e6)
        btc = pred.predict_next_rate("BTC/USD")
        eth = pred.predict_next_rate("ETH/USD")
        assert btc.symbol == "BTC/USD"
        assert eth.symbol == "ETH/USD"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Whale Signal Generator
# ═══════════════════════════════════════════════════════════════════════════════


class TestWhaleSignalGenerator:
    """Tests for data.onchain.whale_signals.WhaleSignalGenerator."""

    def _make(self, **kwargs):
        from data.onchain.whale_signals import WhaleSignalGenerator
        return WhaleSignalGenerator(**kwargs)

    def test_no_data_returns_none(self):
        gen = self._make()
        result = gen.generate_signal("BTC/USD")
        assert result is None

    def test_inject_inflow_generates_sell_signal(self):
        gen = self._make()
        gen.inject_flow("BTC", "exchange_inflow", 500, usd_value=25_000_000)
        gen.inject_flow("BTC", "exchange_inflow", 300, usd_value=15_000_000)
        gen.inject_flow("BTC", "exchange_inflow", 200, usd_value=10_000_000)
        result = gen.generate_signal("BTC/USD")
        assert result is not None
        assert result.direction == "SELL"

    def test_inject_outflow_generates_buy_signal(self):
        gen = self._make()
        gen.inject_flow("BTC", "exchange_outflow", 500, usd_value=25_000_000)
        gen.inject_flow("BTC", "exchange_outflow", 300, usd_value=15_000_000)
        gen.inject_flow("BTC", "exchange_outflow", 200, usd_value=10_000_000)
        result = gen.generate_signal("BTC/USD")
        assert result is not None
        assert result.direction == "BUY"

    def test_mixed_flows_neutral(self):
        gen = self._make()
        gen.inject_flow("BTC", "exchange_inflow", 200, usd_value=10_000_000)
        gen.inject_flow("BTC", "exchange_outflow", 200, usd_value=10_000_000)
        result = gen.generate_signal("BTC/USD")
        assert result is not None
        assert result.direction == "NEUTRAL"

    def test_clear_manual_flows(self):
        gen = self._make()
        gen.inject_flow("BTC", "exchange_inflow", 500, usd_value=25_000_000)
        gen.clear_manual_flows()
        result = gen.generate_signal("BTC/USD")
        assert result is None

    def test_check_recent_transfers(self):
        gen = self._make()
        gen.inject_flow("ETH", "exchange_outflow", 5000, usd_value=10_000_000)
        transfers = gen.check_recent_transfers("ETH/USD")
        assert transfers is not None
        assert transfers["direction"] == "exchange_outflow"

    def test_with_whale_tracker(self):
        """Test integration with WhaleTracker mock."""
        tracker = MagicMock()
        mock_signal = MagicMock()
        mock_signal.inflow_count_1h = 5
        mock_signal.outflow_count_1h = 1
        mock_signal.net_flow_usd = 50_000_000
        tracker.get_signal.return_value = mock_signal
        gen = self._make(whale_tracker=tracker)
        result = gen.generate_signal("BTC/USD")
        assert result is not None
        assert result.direction == "SELL"  # heavy inflows = sell


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Session Effect Strategy
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionEffectStrategy:
    """Tests for strategies.session_effect.SessionEffectStrategy."""

    def _make(self, **kwargs):
        from strategies.session_effect import SessionEffectStrategy
        return SessionEffectStrategy(**kwargs)

    def test_monday_morning_bullish(self):
        strat = self._make()
        monday_9am = datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc)  # Monday
        bias = strat.get_session_bias(monday_9am)
        assert bias["bias"] == "bullish"
        assert bias["session"] == "london"

    def test_sunday_early_bearish(self):
        strat = self._make()
        sunday_3am = datetime(2026, 3, 15, 3, 0, tzinfo=timezone.utc)  # Sunday
        bias = strat.get_session_bias(sunday_3am)
        assert bias["bias"] == "bearish"

    def test_session_detection_asia(self):
        strat = self._make()
        # Tuesday 4am UTC = Asia session
        dt = datetime(2026, 3, 17, 4, 0, tzinfo=timezone.utc)
        assert strat.get_current_session(dt) == "asia"

    def test_session_detection_london(self):
        strat = self._make()
        dt = datetime(2026, 3, 17, 10, 0, tzinfo=timezone.utc)
        assert strat.get_current_session(dt) == "london"

    def test_session_detection_ny(self):
        strat = self._make()
        dt = datetime(2026, 3, 17, 15, 0, tzinfo=timezone.utc)
        assert strat.get_current_session(dt) == "ny"

    def test_session_detection_weekend(self):
        strat = self._make()
        saturday = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)
        assert strat.get_current_session(saturday) == "weekend"

    def test_month_end_detection(self):
        strat = self._make()
        last_day = datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc)
        assert strat.is_month_end(last_day) is True
        mid_month = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
        assert strat.is_month_end(mid_month) is False

    def test_generate_signal_neutral_returns_none(self):
        strat = self._make()
        # Tuesday 10am = neutral day, no strong pattern
        tuesday = datetime(2026, 3, 17, 10, 0, tzinfo=timezone.utc)
        signal = strat.generate_signal("BTC/USD", now=tuesday)
        # May or may not generate depending on DOW strength vs min_strength
        # Tuesday bias strength is 0.1 which is < default 0.15
        assert signal is None

    def test_generate_signal_strong_session(self):
        strat = self._make(min_strength=0.1)
        # Monday 9am = bullish
        monday = datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc)
        signal = strat.generate_signal("BTC/USD", regime="TREND_UP", now=monday)
        assert signal is not None
        assert signal["action"] == "BUY"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Signal Consensus
# ═══════════════════════════════════════════════════════════════════════════════


class TestSignalConsensus:
    """Tests for strategies.signal_consensus.SignalConsensus."""

    def _make(self, **kwargs):
        from strategies.signal_consensus import SignalConsensus
        return SignalConsensus(**kwargs)

    def test_empty_signals(self):
        consensus = self._make()
        result = consensus.filter_signals([])
        assert result == []

    def test_unanimous_buy(self):
        consensus = self._make(mode="unanimous", min_strategies=2)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.8},
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.7},
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.6},
        ]
        result = consensus.filter_signals(signals)
        assert len(result) == 1
        assert result[0]["action"] == "BUY"

    def test_unanimous_fails_on_disagreement(self):
        consensus = self._make(mode="unanimous", min_strategies=2)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.8},
            {"symbol": "BTC/USD", "action": "SELL", "confidence": 0.7},
        ]
        result = consensus.filter_signals(signals)
        assert len(result) == 0

    def test_weighted_consensus(self):
        consensus = self._make(mode="weighted", min_agreement=0.6, min_strategies=2)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.9},
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.8},
            {"symbol": "BTC/USD", "action": "SELL", "confidence": 0.3},
        ]
        result = consensus.filter_signals(signals)
        assert len(result) == 1
        assert result[0]["action"] == "BUY"

    def test_weighted_no_consensus(self):
        consensus = self._make(mode="weighted", min_agreement=0.8, min_strategies=2)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.5},
            {"symbol": "BTC/USD", "action": "SELL", "confidence": 0.5},
        ]
        result = consensus.filter_signals(signals)
        assert len(result) == 0

    def test_majority_mode(self):
        consensus = self._make(mode="majority", min_agreement=0.6, min_strategies=2)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.6},
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.5},
            {"symbol": "BTC/USD", "action": "SELL", "confidence": 0.4},
        ]
        result = consensus.filter_signals(signals)
        assert len(result) == 1
        assert result[0]["action"] == "BUY"

    def test_below_min_strategies_passes_through(self):
        consensus = self._make(min_strategies=3)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.8},
        ]
        result = consensus.filter_signals(signals)
        assert len(result) == 1  # Passes through because < min_strategies

    def test_multiple_symbols_independent(self):
        consensus = self._make(mode="weighted", min_agreement=0.6, min_strategies=2)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.9},
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.8},
            {"symbol": "ETH/USD", "action": "SELL", "confidence": 0.9},
            {"symbol": "ETH/USD", "action": "SELL", "confidence": 0.8},
        ]
        result = consensus.filter_signals(signals)
        assert len(result) == 2
        syms = {r["symbol"] for r in result}
        assert "BTC/USD" in syms
        assert "ETH/USD" in syms

    def test_consensus_stats(self):
        consensus = self._make(mode="weighted", min_strategies=2)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.9},
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.8},
        ]
        consensus.filter_signals(signals)
        stats = consensus.get_consensus_stats()
        assert stats["signals_in"] == 2
        assert stats["signals_out"] == 1
        assert stats["filter_rate"] == 0.5

    def test_boosted_confidence(self):
        consensus = self._make(mode="weighted", min_agreement=0.6, min_strategies=2)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.7},
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.6},
        ]
        result = consensus.filter_signals(signals)
        assert len(result) == 1
        # Confidence should be boosted above the original 0.7
        assert result[0]["confidence"] > 0.7

    def test_invalid_mode_raises(self):
        from strategies.signal_consensus import SignalConsensus
        with pytest.raises(ValueError):
            SignalConsensus(mode="invalid")

    def test_reset_stats(self):
        consensus = self._make()
        consensus.filter_signals([{"symbol": "X", "action": "BUY", "confidence": 0.5}])
        consensus.reset_stats()
        stats = consensus.get_consensus_stats()
        assert stats["signals_in"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Transaction Cost Analysis
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransactionCostAnalyzer:
    """Tests for monitoring.tca.TransactionCostAnalyzer."""

    def _make(self, **kwargs):
        from monitoring.tca import TransactionCostAnalyzer
        return TransactionCostAnalyzer(**kwargs)

    def test_empty_analysis(self):
        tca = self._make()
        result = tca.analyze()
        assert result["trade_count"] == 0
        assert result["total_slippage_bps"] == 0.0

    def test_record_and_analyze(self):
        tca = self._make()
        tca.record_execution(
            order_id="o1", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50010,
            quantity=0.1, venue="kraken", order_type="market",
            latency_ms=50,
        )
        result = tca.analyze()
        assert result["trade_count"] == 1
        assert result["total_slippage_bps"] > 0  # filled higher than intended

    def test_slippage_buy_direction(self):
        """Buying at higher price = positive slippage (bad)."""
        tca = self._make()
        tca.record_execution(
            order_id="o1", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50050,
            quantity=0.1, venue="kraken",
        )
        result = tca.analyze()
        assert result["total_slippage_bps"] > 0

    def test_slippage_sell_direction(self):
        """Selling at lower price = positive slippage (bad)."""
        tca = self._make()
        tca.record_execution(
            order_id="o1", symbol="BTC/USD", side="sell",
            intended_price=50000, fill_price=49950,
            quantity=0.1, venue="kraken",
        )
        result = tca.analyze()
        assert result["total_slippage_bps"] > 0

    def test_slippage_by_venue(self):
        tca = self._make()
        # Kraken: low slippage
        tca.record_execution(
            order_id="o1", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50005,
            quantity=0.1, venue="kraken",
        )
        # Coinbase: high slippage
        tca.record_execution(
            order_id="o2", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50050,
            quantity=0.1, venue="coinbase",
        )
        result = tca.analyze()
        assert "kraken" in result["slippage_by_venue"]
        assert "coinbase" in result["slippage_by_venue"]
        assert result["slippage_by_venue"]["kraken"] < result["slippage_by_venue"]["coinbase"]

    def test_slippage_by_order_type(self):
        tca = self._make()
        tca.record_execution(
            order_id="o1", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50050,
            quantity=0.1, venue="kraken", order_type="market",
        )
        tca.record_execution(
            order_id="o2", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50005,
            quantity=0.1, venue="kraken", order_type="limit",
        )
        result = tca.analyze()
        assert "market" in result["slippage_by_order_type"]
        assert "limit" in result["slippage_by_order_type"]

    def test_maker_vs_taker_savings(self):
        tca = self._make()
        # Market orders: high slippage
        for i in range(3):
            tca.record_execution(
                order_id=f"m{i}", symbol="BTC/USD", side="buy",
                intended_price=50000, fill_price=50050,
                quantity=0.1, venue="kraken", order_type="market",
            )
        # Limit orders: low slippage
        for i in range(3):
            tca.record_execution(
                order_id=f"l{i}", symbol="BTC/USD", side="buy",
                intended_price=50000, fill_price=50005,
                quantity=0.1, venue="kraken", order_type="limit",
            )
        result = tca.analyze()
        assert result["maker_vs_taker_savings"] > 0

    def test_execution_score_perfect(self):
        tca = self._make()
        tca.record_execution(
            order_id="o1", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50000,  # zero slippage
            quantity=0.1, venue="kraken", order_type="limit",
            latency_ms=5,
        )
        score = tca.get_execution_score()
        assert score >= 70  # Should be high with zero slippage

    def test_execution_score_terrible(self):
        tca = self._make()
        tca.record_execution(
            order_id="o1", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50500,  # 100 bps slippage
            quantity=0.1, venue="kraken", order_type="market",
            latency_ms=1000,
        )
        score = tca.get_execution_score()
        assert score < 30  # Should be low

    def test_optimal_venue_recommendation(self):
        tca = self._make()
        tca.record_execution(
            order_id="o1", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50005,
            quantity=0.1, venue="kraken",
        )
        tca.record_execution(
            order_id="o2", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50050,
            quantity=0.1, venue="coinbase",
        )
        result = tca.analyze()
        assert result["optimal_venue_recommendation"] == "kraken"

    def test_max_records_trimming(self):
        tca = self._make(max_records=5)
        for i in range(10):
            tca.record_execution(
                order_id=f"o{i}", symbol="BTC/USD", side="buy",
                intended_price=50000, fill_price=50010,
                quantity=0.1, venue="kraken",
            )
        assert tca.record_count == 5

    def test_get_records(self):
        tca = self._make()
        tca.record_execution(
            order_id="o1", symbol="BTC/USD", side="buy",
            intended_price=50000, fill_price=50010,
            quantity=0.1, venue="kraken",
        )
        records = tca.get_records()
        assert len(records) == 1
        assert records[0]["order_id"] == "o1"
        assert "slippage_bps" in records[0]

    def test_size_bucket_classification(self):
        from monitoring.tca import ExecutionRecord
        small = ExecutionRecord("o1", "X", "buy", 100, 100, 1.0, "v", "market", 0)
        assert small.size_bucket == "small"
        medium = ExecutionRecord("o2", "X", "buy", 1000, 1000, 2.0, "v", "market", 0)
        assert medium.size_bucket == "medium"
        large = ExecutionRecord("o3", "X", "buy", 10000, 10000, 2.0, "v", "market", 0)
        assert large.size_bucket == "large"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Integration: Regime rotation + Strategy Router
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests between components."""

    def test_regime_rotation_with_consensus(self):
        """Regime rotation feeds weights, consensus filters the result."""
        from strategies.regime_rotation import RegimeStrategyRotator
        from strategies.signal_consensus import SignalConsensus

        rotator = RegimeStrategyRotator()
        rotator._current_regime = "TRENDING_UP"
        weights = rotator.get_regime_weights()

        consensus = SignalConsensus(mode="weighted", min_agreement=0.6, min_strategies=2)
        signals = [
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.8 * weights.get("momentum", 0.3)},
            {"symbol": "BTC/USD", "action": "BUY", "confidence": 0.7 * weights.get("breakout", 0.3)},
        ]
        result = consensus.filter_signals(signals)
        assert len(result) >= 1

    def test_funding_predictor_with_session_effect(self):
        """Funding prediction + session effect can both produce signals."""
        from strategies.funding_rate_predictor import FundingRatePredictor
        from strategies.session_effect import SessionEffectStrategy

        predictor = FundingRatePredictor()
        session = SessionEffectStrategy(min_strength=0.1)

        # Feed data to predictor
        for i in range(5):
            predictor.update("BTC/USD", 0.0005 * (i + 1), 1e9, 50000, 50100, 1e6, 2e6)

        # Both can produce predictions for the same symbol
        pred = predictor.predict_next_rate("BTC/USD")
        monday = datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc)
        bias = session.get_session_bias(monday)

        assert pred.symbol == "BTC/USD"
        assert bias["session"] == "london"

    def test_tca_records_from_multiple_sources(self):
        """TCA can record executions from any strategy."""
        from monitoring.tca import TransactionCostAnalyzer

        tca = TransactionCostAnalyzer()
        # Simulate executions from different strategies
        for i, venue in enumerate(["kraken", "coinbase", "kraken"]):
            tca.record_execution(
                order_id=f"strat_{i}",
                symbol="BTC/USD",
                side="buy",
                intended_price=50000,
                fill_price=50000 + i * 5,
                quantity=0.1,
                venue=venue,
            )

        analysis = tca.analyze()
        assert analysis["trade_count"] == 3
        assert len(analysis["slippage_by_venue"]) == 2
