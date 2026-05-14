"""
Tests for strategies.strategy_router.StrategyRouter

Covers:
  - Registration, enable/disable
  - generate_all_signals with mock strategies
  - Fault isolation (one failure doesn't crash others)
  - Signal conversion from different return types
  - Deduplication (same symbol, same direction -> merge)
  - Conflict resolution (BUY vs SELL same symbol)
  - MacroEventFilter applied as multiplier
  - MTFConfluence confirmation boost/penalty
  - Config-driven enable/disable
  - Timeout handling for slow strategies
  - Empty OHLCV handling
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from strategies.strategy_router import StrategyRouter

# ---------------------------------------------------------------------------
# Helpers: minimal TradingSignal and strategy stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeSignal:
    """Mimics core.types.Signal fields."""
    symbol: str = "BTC/USD"
    action: str = "BUY"
    confidence: float = 0.7
    strength: float = 0.6
    entry_price: float = 50000.0
    stop_loss: Optional[float] = 49000.0
    take_profit: Optional[float] = 52000.0
    reasoning: str = "test signal"
    strategy_name: str = "test"


@dataclass
class _FakeArbOpportunity:
    """Mimics ArbitrageOpportunity from cross_exchange_arb."""
    symbol: str = "BTC"
    cheap_exchange: str = "kraken"
    expensive_exchange: str = "coinbase"
    cheap_price: float = 49900.0
    expensive_price: float = 50100.0
    net_spread_bps: float = 20.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class _FakeBasisOpportunity:
    """Mimics BasisOpportunity from futures_basis_arb."""
    symbol: str = "BTC"
    spot_price: float = 50000.0
    futures_price: float = 50500.0
    annual_basis_pct: float = 12.0
    action: str = "BUY_SPOT"


@dataclass
class _FakeArbSignal:
    """Mimics ArbSignal from delta_neutral_perp_arb."""
    symbol: str = "BTC/USD"
    action: str = "ENTER"
    predicted_funding_bps: float = 5.0
    annual_rate_pct: float = 18.0
    basis_bps: float = 10.0
    reason: str = "positive funding"
    timestamp: float = field(default_factory=time.time)


@dataclass
class _FakeVolArbSignal:
    """Mimics VolArbSignal from volatility_arb."""
    symbol: str = "BTC/USD"
    action: str = "SELL_VOL"
    vol_premium_pct: float = 8.0
    iv_pct: float = 60.0
    rv_pct: float = 52.0
    hedge_delta: float = 0.3
    position_size_usd: float = 200.0
    reason: str = "IV premium"


@dataclass
class _FakeLiquidationSignal:
    """Mimics LiquidationSignal."""
    symbol: str = "BTC/USD"
    direction: str = "BUY"
    confidence: float = 0.75
    oi_drop_pct: float = 0.08
    funding_rate: float = -0.01
    estimated_cascade_size_usd: float = 5_000_000.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _FakeOptionsSignal:
    """Mimics OptionsSignal from deribit_options."""
    symbol: str = "BTC"
    direction: str = "BULLISH"
    confidence: float = 0.65
    rationale: str = "high put/call skew"
    iv_percentile: float = 80.0
    timestamp: float = field(default_factory=time.time)


class _MockStrategy:
    """Simple mock that returns a configurable signal."""

    def __init__(self, result=None, is_async=False, delay=0.0, should_raise=False):
        self._result = result
        self._is_async = is_async
        self._delay = delay
        self._should_raise = should_raise

    async def generate_signal(self, symbol, ohlcv, regime, **kwargs):
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_raise:
            raise RuntimeError("strategy crashed")
        return self._result


class _MockAnalyzeStrategy:
    """Mock with analyze(market_data) interface."""

    def __init__(self, result=None):
        self._result = result

    async def analyze(self, market_data):
        return self._result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegistration:
    """Test strategy registration, enable, disable."""

    def test_register_strategy(self):
        router = StrategyRouter()
        mock = _MockStrategy()
        router.register("test_strat", mock, enabled=True)
        assert "test_strat" in router.get_active_strategies()

    def test_register_disabled(self):
        router = StrategyRouter()
        mock = _MockStrategy()
        router.register("test_strat", mock, enabled=False)
        assert "test_strat" not in router.get_active_strategies()

    def test_enable_disable(self):
        router = StrategyRouter()
        mock = _MockStrategy()
        router.register("test_strat", mock, enabled=True)
        assert "test_strat" in router.get_active_strategies()
        router.disable("test_strat")
        assert "test_strat" not in router.get_active_strategies()
        router.enable("test_strat")
        assert "test_strat" in router.get_active_strategies()

    def test_get_strategy_stats(self):
        router = StrategyRouter()
        mock = _MockStrategy()
        router.register("test_strat", mock)
        stats = router.get_strategy_stats()
        assert "test_strat" in stats
        assert stats["test_strat"]["calls"] == 0
        assert stats["test_strat"]["enabled"] is True

    def test_filter_strategies_excluded_from_active(self):
        router = StrategyRouter()
        router.register("macro_event_filter", MagicMock(), enabled=True)
        router.register("mtf_confluence", MagicMock(), enabled=True)
        # Filters should not appear in active strategies list
        assert "macro_event_filter" not in router.get_active_strategies()
        assert "mtf_confluence" not in router.get_active_strategies()


class TestGenerateAllSignals:
    """Test signal generation from various strategy types."""

    @pytest.mark.asyncio
    async def test_generate_signal_strategy(self):
        """Standard generate_signal(symbol, ohlcv, regime) strategies."""
        router = StrategyRouter()
        signal = _FakeSignal(action="BUY", confidence=0.7, entry_price=50000.0)
        mock = _MockStrategy(result=signal)
        router.register("peak_alpha", mock, enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "TRENDING_UP")
        assert len(signals) == 1
        assert signals[0].action == "BUY"
        assert signals[0].confidence >= 0.7  # may be slightly boosted

    @pytest.mark.asyncio
    async def test_analyze_strategy(self):
        """analyze(market_data) strategies like stat_arb_cointegration."""
        router = StrategyRouter()
        result = [{"action": "BUY", "symbol": "BTC/USD", "price": 50000, "confidence": 0.6}]
        mock = _MockAnalyzeStrategy(result=result)
        router.register("stat_arb_cointegration", mock, enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "MEAN_REVERT", {"price": 50000})
        assert len(signals) == 1
        assert signals[0].action == "BUY"

    @pytest.mark.asyncio
    async def test_disabled_strategy_produces_no_signals(self):
        router = StrategyRouter()
        signal = _FakeSignal()
        mock = _MockStrategy(result=signal)
        router.register("peak_alpha", mock, enabled=False)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_none_result_skipped(self):
        router = StrategyRouter()
        mock = _MockStrategy(result=None)
        router.register("peak_alpha", mock, enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_hold_signal_skipped(self):
        router = StrategyRouter()
        signal = _FakeSignal(action="HOLD")
        mock = _MockStrategy(result=signal)
        router.register("peak_alpha", mock, enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        assert len(signals) == 0


class TestFaultIsolation:
    """One strategy failure doesn't crash others."""

    @pytest.mark.asyncio
    async def test_one_failure_doesnt_block_others(self):
        router = StrategyRouter()
        # First strategy will fail
        failing = _MockStrategy(should_raise=True)
        router.register("peak_alpha", failing, enabled=True)
        # Second strategy will succeed
        good_signal = _FakeSignal(action="BUY", confidence=0.8)
        succeeding = _MockStrategy(result=good_signal)
        router.register("mean_reversion", succeeding, enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        # Should get signal from succeeding strategy despite failure
        assert len(signals) >= 1
        # Stats should record the error
        stats = router.get_strategy_stats()
        assert stats["peak_alpha"]["errors"] == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Strategy that exceeds timeout is skipped."""
        router = StrategyRouter(timeout_s=0.1)
        slow = _MockStrategy(result=_FakeSignal(), delay=1.0)
        router.register("peak_alpha", slow, enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        assert len(signals) == 0
        stats = router.get_strategy_stats()
        assert stats["peak_alpha"]["timeouts"] == 1


class TestSignalConversion:
    """Test conversion from various return types to TradingSignal."""

    @pytest.mark.asyncio
    async def test_convert_arb_opportunity(self):
        """ArbitrageOpportunity -> TradingSignal."""
        router = StrategyRouter()

        class _ArbStrat:
            def generate_signals(self):
                return [_FakeArbOpportunity()]

        router.register("cross_exchange_arb", _ArbStrat(), enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        assert len(signals) == 1
        assert signals[0].action == "BUY"
        assert "arb spread" in signals[0].reasoning

    @pytest.mark.asyncio
    async def test_convert_basis_opportunity(self):
        """BasisOpportunity -> TradingSignal."""
        router = StrategyRouter()

        class _BasisStrat:
            def generate_signal(self, symbol):
                return _FakeBasisOpportunity()

        router.register("futures_basis_arb", _BasisStrat(), enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        assert len(signals) == 1
        assert "basis" in signals[0].reasoning

    @pytest.mark.asyncio
    async def test_convert_arb_signal(self):
        """ArbSignal (delta_neutral) -> TradingSignal."""
        router = StrategyRouter()

        class _DeltaStrat:
            def evaluate(self, symbol, spot, perp, funding, time_h):
                return _FakeArbSignal()

        router.register("delta_neutral_perp_arb", _DeltaStrat(), enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN", {"price": 50000})
        assert len(signals) == 1
        assert signals[0].action == "BUY"  # ENTER -> BUY

    @pytest.mark.asyncio
    async def test_convert_vol_arb_signal(self):
        """VolArbSignal -> TradingSignal."""
        router = StrategyRouter()

        class _VolStrat:
            def evaluate(self, symbol):
                return _FakeVolArbSignal()

        router.register("volatility_arb", _VolStrat(), enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN", {"price": 50000})
        assert len(signals) == 1
        assert signals[0].action == "SELL"  # SELL_VOL -> SELL
        assert "vol premium" in signals[0].reasoning

    @pytest.mark.asyncio
    async def test_convert_liquidation_signal(self):
        """LiquidationSignal -> TradingSignal."""
        router = StrategyRouter()

        class _LiqStrat:
            def generate_signal(self, symbol):
                return _FakeLiquidationSignal()

        router.register("liquidation_cascade", _LiqStrat(), enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN", {"price": 50000})
        assert len(signals) == 1
        assert signals[0].action == "BUY"
        assert signals[0].confidence == 0.75

    @pytest.mark.asyncio
    async def test_convert_options_signal(self):
        """OptionsSignal -> TradingSignal."""
        router = StrategyRouter()

        class _OptStrat:
            async def generate_signal(self):
                return _FakeOptionsSignal()

        router.register("deribit_options", _OptStrat(), enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN", {"price": 50000})
        assert len(signals) == 1
        assert signals[0].action == "BUY"  # BULLISH -> BUY
        assert "BTC/USD" in signals[0].symbol

    @pytest.mark.asyncio
    async def test_convert_dict_signal(self):
        """Dict-based signals."""
        router = StrategyRouter()

        class _DictStrat:
            async def analyze(self, market_data):
                return {"action": "SELL", "symbol": "ETH/USD", "price": 3000, "confidence": 0.65, "reason": "overbought"}

        router.register("stat_arb_cointegration", _DictStrat(), enabled=True)
        signals = await router.generate_all_signals("ETH/USD", None, "UNKNOWN", {"price": 3000})
        assert len(signals) == 1
        assert signals[0].action == "SELL"
        assert signals[0].symbol == "ETH/USD"

    @pytest.mark.asyncio
    async def test_convert_market_maker_dict(self):
        """Market maker dict with bid_price/ask_price.

        Market maker generates both BUY (bid) and SELL (ask) for the same
        symbol.  Deduplication sees a conflict with gap=0 (<0.3) and cancels
        both — which is the correct safety behaviour for opposing signals
        from a single strategy.  When market_maker is the *only* strategy the
        net output is therefore zero signals.
        """
        router = StrategyRouter()

        class _MmStrat:
            def analyze(self, market_data):
                return {"bid_price": 49900, "ask_price": 50100, "symbol": "BTC/USD"}

        router.register("market_maker", _MmStrat(), enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN", {"price": 50000})
        # Both BUY and SELL at equal confidence -> conflict resolution cancels both
        assert len(signals) == 0
        # But the conversion itself produced 2 signals (check stats)
        stats = router.get_strategy_stats()
        assert stats["market_maker"]["signals_produced"] == 2


class TestDeduplication:
    """Signal deduplication and conflict resolution."""

    @pytest.mark.asyncio
    async def test_same_direction_merged_with_boost(self):
        """Multiple strategies agreeing -> take best + boost 10%."""
        router = StrategyRouter()
        sig1 = _FakeSignal(action="BUY", confidence=0.6, entry_price=50000)
        sig2 = _FakeSignal(action="BUY", confidence=0.8, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=sig1), enabled=True)
        router.register("momentum", _MockStrategy(result=sig2), enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "TRENDING_UP")
        assert len(signals) == 1
        # Best confidence was 0.8, boosted by 10%
        assert signals[0].confidence >= 0.8 * 1.10 - 0.001

    @pytest.mark.asyncio
    async def test_conflict_cancels_when_gap_small(self):
        """BUY vs SELL with small confidence gap -> cancel both."""
        router = StrategyRouter()
        buy_sig = _FakeSignal(action="BUY", confidence=0.6, entry_price=50000)
        sell_sig = _FakeSignal(action="SELL", confidence=0.7, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=buy_sig), enabled=True)
        router.register("mean_reversion", _MockStrategy(result=sell_sig), enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        # Gap is 0.1 < 0.3 -> both cancelled
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_conflict_keeps_winner_when_gap_large(self):
        """BUY vs SELL with large confidence gap -> keep winner."""
        router = StrategyRouter()
        buy_sig = _FakeSignal(action="BUY", confidence=0.9, entry_price=50000)
        sell_sig = _FakeSignal(action="SELL", confidence=0.4, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=buy_sig), enabled=True)
        router.register("mean_reversion", _MockStrategy(result=sell_sig), enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        # Gap is 0.5 > 0.3 -> keep BUY
        assert len(signals) == 1
        assert signals[0].action == "BUY"

    @pytest.mark.asyncio
    async def test_single_signal_no_dedup(self):
        """Single strategy signal passes through without change."""
        router = StrategyRouter()
        sig = _FakeSignal(action="BUY", confidence=0.75, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=sig), enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "TRENDING_UP")
        assert len(signals) == 1
        assert signals[0].confidence == 0.75


class TestMacroEventFilter:
    """MacroEventFilter applied as position multiplier."""

    @pytest.mark.asyncio
    async def test_macro_filter_reduces_confidence(self):
        router = StrategyRouter()
        sig = _FakeSignal(action="BUY", confidence=0.8, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=sig), enabled=True)

        # Mock macro filter with multiplier 0.3 (reduce window)
        macro = MagicMock()
        macro.get_position_multiplier.return_value = 0.3
        router.register("macro_event_filter", macro, enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "TRENDING_UP")
        assert len(signals) == 1
        # Confidence should be reduced: 0.8 * 0.3 = 0.24
        assert abs(signals[0].confidence - 0.24) < 0.01

    @pytest.mark.asyncio
    async def test_macro_filter_halt_drops_all(self):
        router = StrategyRouter()
        sig = _FakeSignal(action="BUY", confidence=0.9, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=sig), enabled=True)

        macro = MagicMock()
        macro.get_position_multiplier.return_value = 0.0  # halt
        router.register("macro_event_filter", macro, enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "TRENDING_UP")
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_macro_filter_no_effect_when_1(self):
        router = StrategyRouter()
        sig = _FakeSignal(action="BUY", confidence=0.8, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=sig), enabled=True)

        macro = MagicMock()
        macro.get_position_multiplier.return_value = 1.0
        router.register("macro_event_filter", macro, enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "TRENDING_UP")
        assert len(signals) == 1
        assert signals[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_macro_filter_disabled(self):
        router = StrategyRouter()
        sig = _FakeSignal(action="BUY", confidence=0.8, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=sig), enabled=True)

        macro = MagicMock()
        macro.get_position_multiplier.return_value = 0.0
        router.register("macro_event_filter", macro, enabled=False)  # disabled

        signals = await router.generate_all_signals("BTC/USD", None, "TRENDING_UP")
        assert len(signals) == 1  # filter not applied


class TestMTFConfluence:
    """MTFConfluence confirmation boost/penalty."""

    @pytest.mark.asyncio
    async def test_mtf_boosts_when_agrees(self):
        router = StrategyRouter()
        sig = _FakeSignal(action="BUY", confidence=0.7, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=sig), enabled=True)

        mtf = MagicMock()
        mtf.check.return_value = (True, 0.8, "3/3 timeframes agree")
        router.register("mtf_confluence", mtf, enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "TRENDING_UP")
        assert len(signals) == 1
        # Boosted by 15%: 0.7 * 1.15 = 0.805
        assert signals[0].confidence >= 0.7 * 1.15 - 0.001

    @pytest.mark.asyncio
    async def test_mtf_penalizes_when_disagrees(self):
        router = StrategyRouter()
        sig = _FakeSignal(action="BUY", confidence=0.7, entry_price=50000)
        router.register("peak_alpha", _MockStrategy(result=sig), enabled=True)

        mtf = MagicMock()
        mtf.check.return_value = (False, 0.3, "1/3 timeframes agree")
        router.register("mtf_confluence", mtf, enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        assert len(signals) == 1
        # Reduced by 20%: 0.7 * 0.80 = 0.56
        assert signals[0].confidence <= 0.7 * 0.80 + 0.001


class TestConfigDriven:
    """Config-driven enable/disable."""

    def test_config_flags_respected(self):
        router = StrategyRouter()
        router.register("peak_alpha", _MockStrategy(), enabled=True)
        router.register("scalping", _MockStrategy(), enabled=False)
        assert "peak_alpha" in router.get_active_strategies()
        assert "scalping" not in router.get_active_strategies()


class TestEmptyOHLCV:
    """Strategies handle empty/None OHLCV gracefully."""

    @pytest.mark.asyncio
    async def test_none_ohlcv(self):
        """Strategy returning None when OHLCV is None."""
        router = StrategyRouter()
        mock = _MockStrategy(result=None)
        router.register("peak_alpha", mock, enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_empty_market_data(self):
        router = StrategyRouter()
        mock = _MockStrategy(result=_FakeSignal(action="BUY", confidence=0.6))
        router.register("peak_alpha", mock, enabled=True)
        signals = await router.generate_all_signals("BTC/USD", None, "UNKNOWN", {})
        assert len(signals) == 1


class TestMultipleSymbols:
    """Deduplication handles different symbols independently."""

    @pytest.mark.asyncio
    async def test_different_symbols_not_merged(self):
        router = StrategyRouter()

        class _BtcStrat:
            async def generate_signal(self, symbol, ohlcv, regime, **kw):
                return _FakeSignal(symbol="BTC/USD", action="BUY", confidence=0.7, entry_price=50000)

        class _EthStrat:
            async def generate_signal(self, symbol, ohlcv, regime, **kw):
                return _FakeSignal(symbol="ETH/USD", action="BUY", confidence=0.6, entry_price=3000)

        router.register("peak_alpha", _BtcStrat(), enabled=True)
        router.register("momentum", _EthStrat(), enabled=True)

        signals = await router.generate_all_signals("BTC/USD", None, "TRENDING_UP")
        # Two different symbols, should not be merged
        assert len(signals) == 2
        symbols = {s.symbol for s in signals}
        assert "BTC/USD" in symbols
        assert "ETH/USD" in symbols


class TestStatsTracking:
    """Verify stats are updated correctly."""

    @pytest.mark.asyncio
    async def test_stats_increment(self):
        router = StrategyRouter()
        sig = _FakeSignal(action="BUY", confidence=0.7)
        router.register("peak_alpha", _MockStrategy(result=sig), enabled=True)

        await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        stats = router.get_strategy_stats()
        assert stats["peak_alpha"]["calls"] == 1
        assert stats["peak_alpha"]["signals_produced"] == 1
        assert stats["peak_alpha"]["errors"] == 0
        assert stats["peak_alpha"]["avg_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_error_stats(self):
        router = StrategyRouter()
        router.register("peak_alpha", _MockStrategy(should_raise=True), enabled=True)

        await router.generate_all_signals("BTC/USD", None, "UNKNOWN")
        stats = router.get_strategy_stats()
        assert stats["peak_alpha"]["errors"] == 1
        assert stats["peak_alpha"]["signals_produced"] == 0
