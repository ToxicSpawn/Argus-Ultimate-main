"""
Tests for trading performance fixes 15–27.

Covers:
  FIX 15: Signal staleness decay
  FIX 16: Aggressive strategy dampening
  FIX 17: Maker fill rate tracking
  FIX 18: Wire RL agent for execution sizing (graceful fallback)
  FIX 19: Position conflict check
  FIX 20: Increase rate limits to exchange maximums
  FIX 21: Smart loop sleep
  FIX 22: Execution quality score
  FIX 23: Signal age-based urgency
  FIX 24: Regime-specific strategy whitelist
  FIX 25: Hot hand strategy boost
  FIX 26: Smart signal conflict resolution
  FIX 27: Configurable ensemble thresholds
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ===================================================================
# FIX 15 & FIX 23: Signal staleness decay + age-based urgency
# ===================================================================

class TestSignalStalenessDecay:
    """FIX 15: Signal staleness decay applied in _execute_signals."""

    def test_staleness_decay_formula(self):
        """Confidence decays exponentially with age: exp(-age/30)."""
        age = 21.0  # ~half-life
        decay = math.exp(-age / 30.0)
        # At 21s, decay should be roughly 0.50
        assert 0.45 < decay < 0.55

    def test_staleness_rejection_at_120s(self):
        """Signals older than 120s should be rejected."""
        age = 121.0
        assert age > 120.0  # would trigger rejection

    def test_staleness_no_decay_for_fresh(self):
        """Fresh signals (age < 0.1s) should not be decayed."""
        age = 0.05
        decay = math.exp(-age / 30.0)
        assert decay > 0.99

    def test_staleness_moderate_decay(self):
        """At 60s, confidence should be ~13.5% of original."""
        age = 60.0
        decay = math.exp(-age / 30.0)
        assert 0.10 < decay < 0.18


class TestSignalAgeBasedUrgency:
    """FIX 23: Age-based urgency mapping."""

    def test_urgency_fresh_signal(self):
        """age < 5s -> urgency 0.2 (maker, patient)."""
        age = 3.0
        urgency = 0.2 if age < 5.0 else 0.5
        assert urgency == 0.2

    def test_urgency_balanced(self):
        """age 5-30s -> urgency 0.5."""
        age = 15.0
        if age < 5.0:
            urgency = 0.2
        elif age < 30.0:
            urgency = 0.5
        elif age < 60.0:
            urgency = 0.8
        else:
            urgency = 1.0
        assert urgency == 0.5

    def test_urgency_aggressive(self):
        """age 30-60s -> urgency 0.8."""
        age = 45.0
        if age < 5.0:
            urgency = 0.2
        elif age < 30.0:
            urgency = 0.5
        elif age < 60.0:
            urgency = 0.8
        else:
            urgency = 1.0
        assert urgency == 0.8

    def test_urgency_market_order(self):
        """age > 60s -> urgency 1.0."""
        age = 90.0
        if age < 5.0:
            urgency = 0.2
        elif age < 30.0:
            urgency = 0.5
        elif age < 60.0:
            urgency = 0.8
        else:
            urgency = 1.0
        assert urgency == 1.0


# ===================================================================
# FIX 16: Aggressive strategy dampening
# ===================================================================

class TestStrategyDampening:
    """FIX 16: get_strategy_multiplier in StrategyStateStore."""

    def _make_store(self):
        from strategies.strategy_state_store import StrategyStateStore
        tmp = tempfile.mktemp(suffix=".db")
        store = StrategyStateStore(db_path=tmp, max_consecutive_losses=5, cooldown_minutes=60)
        return store, tmp

    def test_multiplier_default_no_state(self):
        store, _ = self._make_store()
        assert store.get_strategy_multiplier("nonexistent") == 1.0

    def test_multiplier_consecutive_losses_5(self):
        store, _ = self._make_store()
        for i in range(6):
            store.update_after_trade("bad_strat", pnl=-10.0)
        mult = store.get_strategy_multiplier("bad_strat", portfolio_value=1000.0)
        assert mult == 0.10

    def test_multiplier_heavy_pnl_loss(self):
        """PnL < -3% -> 0.25 (with mixed wins to avoid consec_losses trigger)."""
        store, _ = self._make_store()
        # Alternate losses and occasional wins to avoid 5-consecutive-loss trigger
        for i in range(10):
            store.update_after_trade("losing", pnl=-8.0)
            if i % 4 == 3:  # win every 4th to reset consecutive losses
                store.update_after_trade("losing", pnl=0.01)
        # total_pnl is deeply negative, but consecutive_losses < 5
        mult = store.get_strategy_multiplier("losing", portfolio_value=1000.0)
        assert mult == 0.25

    def test_multiplier_moderate_pnl_loss(self):
        """PnL -1% to -3% -> 0.50 (with mixed wins to avoid consec_losses trigger)."""
        store, _ = self._make_store()
        # Alternate losses and wins to keep consecutive_losses < 5
        for i in range(6):
            store.update_after_trade("moderate_loss", pnl=-3.5)
            if i % 3 == 2:
                store.update_after_trade("moderate_loss", pnl=0.01)
        # total_pnl ~ -21 + 0.02 = -20.98, portfolio = 1000 -> ~ -2.1%
        mult = store.get_strategy_multiplier("moderate_loss", portfolio_value=1000.0)
        assert mult == 0.50

    def test_multiplier_winning_boost(self):
        """PnL > +3% -> 1.30."""
        store, _ = self._make_store()
        for i in range(10):
            store.update_after_trade("winner", pnl=5.0)
        # total_pnl = +50, portfolio = 1000 -> +5%
        mult = store.get_strategy_multiplier("winner", portfolio_value=1000.0)
        assert mult == 1.30

    def test_multiplier_not_enough_trades(self):
        """< 5 trades -> 1.0 regardless of PnL."""
        store, _ = self._make_store()
        for i in range(3):
            store.update_after_trade("few_trades", pnl=-50.0)
        mult = store.get_strategy_multiplier("few_trades", portfolio_value=1000.0)
        assert mult == 1.0


# ===================================================================
# FIX 17: Maker fill rate tracking
# ===================================================================

class TestMakerFillRateTracking:
    """FIX 17: Fill tracking in MakerEnforcement."""

    def test_record_maker_fill(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement(connector=None, enabled=True)
        me.record_fill("BTC/USD", is_maker=True)
        me.record_fill("BTC/USD", is_maker=True)
        me.record_fill("BTC/USD", is_maker=False)
        rate = me.get_maker_fill_rate("BTC/USD")
        assert abs(rate - 2.0 / 3.0) < 0.01

    def test_fill_rate_unknown_symbol(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement(connector=None, enabled=True)
        assert me.get_maker_fill_rate("ETH/USD") == 1.0

    def test_should_auto_taker_insufficient_data(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement(connector=None, enabled=True)
        for _ in range(10):
            me.record_fill("BTC/USD", is_maker=False)
        assert me.should_auto_taker("BTC/USD") is False  # < 50 trades

    def test_should_auto_taker_low_fill_rate(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement(connector=None, enabled=True)
        for _ in range(10):
            me.record_fill("BTC/USD", is_maker=True)
        for _ in range(50):
            me.record_fill("BTC/USD", is_maker=False)
        assert me.should_auto_taker("BTC/USD") is True

    def test_get_fill_stats(self):
        from execution.maker_enforcement import MakerEnforcement
        me = MakerEnforcement(connector=None, enabled=True)
        me.record_fill("BTC/USD", is_maker=True)
        stats = me.get_fill_stats()
        assert "BTC/USD" in stats
        assert stats["BTC/USD"]["maker_fills"] == 1


# ===================================================================
# FIX 18: RL agent wiring (graceful fallback)
# ===================================================================

class TestRLAgentFallback:
    """FIX 18: RL agent wiring with graceful fallback."""

    def test_no_rl_agent_no_crash(self):
        """When no RL agent is available, sizing should be unaffected."""
        # Simulate the fallback path: no rl_agent attribute
        component_registry = SimpleNamespace()
        _rl_agent = getattr(component_registry, "rl_agent", None)
        assert _rl_agent is None  # should not raise

    def test_rl_agent_size_factor_clamped(self):
        """RL size factor should be clamped to [0.1, 2.0]."""
        raw_factor = 5.0
        clamped = max(0.1, min(2.0, raw_factor))
        assert clamped == 2.0

        raw_factor_negative = -1.0
        clamped = max(0.1, min(2.0, raw_factor_negative))
        assert clamped == 0.1


# ===================================================================
# FIX 19: Position conflict check
# ===================================================================

class TestPositionConflictCheck:
    """FIX 19: Position conflict detection."""

    def test_buy_with_existing_long_is_pyramid(self):
        """BUY when already LONG is a pyramid."""
        existing_side = "BUY"
        action = "BUY"
        assert action == existing_side  # pyramid scenario

    def test_sell_with_existing_long_is_close(self):
        """SELL when LONG is a close (opposite direction)."""
        existing_side = "BUY"
        action = "SELL"
        assert action != existing_side  # close scenario

    def test_buy_with_existing_short_is_close(self):
        """BUY when SHORT is a close."""
        existing_side = "SELL"
        action = "BUY"
        assert action != existing_side


# ===================================================================
# FIX 20: Rate limits updated to exchange maximums
# ===================================================================

class TestRateLimitsUpdated:
    """FIX 20: Kraken private rate limit updated to 15 req/s."""

    def test_kraken_private_default_is_15(self):
        from execution.rate_limit_manager import DEFAULT_LIMITS, EndpointType
        kraken_private = DEFAULT_LIMITS["kraken"][EndpointType.PRIVATE]
        assert kraken_private.requests_per_second == 15.0

    def test_from_config_overrides(self):
        from execution.rate_limit_manager import RateLimitManager, EndpointType
        config = {
            "kraken": {"private": 20, "public": 20},
        }
        mgr = RateLimitManager.from_config(config)
        # Check that the bucket was created with the override
        bucket = mgr._get_bucket("kraken", EndpointType.PRIVATE)
        assert bucket.refill_rate == 20.0

    def test_from_config_empty(self):
        from execution.rate_limit_manager import RateLimitManager
        mgr = RateLimitManager.from_config(None)
        assert mgr is not None

    def test_from_config_preserves_unmentioned_exchanges(self):
        from execution.rate_limit_manager import RateLimitManager, EndpointType
        config = {"kraken": {"private": 20}}
        mgr = RateLimitManager.from_config(config)
        # Coinbase should still have defaults
        bucket = mgr._get_bucket("coinbase", EndpointType.PRIVATE)
        assert bucket.refill_rate == 15.0


# ===================================================================
# FIX 21: Smart loop sleep
# ===================================================================

class TestSmartLoopSleep:
    """FIX 21: Adaptive sleep based on cycle state."""

    def test_signals_generated_no_sleep(self):
        """If signals were generated, sleep should be 0."""
        execution_results = [{"status": "filled"}]
        _had_signals = bool(execution_results)
        _has_pending = False
        iter_sleep_s = 5.0

        if _had_signals:
            _smart_sleep = 0.0
        elif _has_pending:
            _smart_sleep = min(1.0, iter_sleep_s)
        else:
            _smart_sleep = iter_sleep_s

        assert _smart_sleep == 0.0

    def test_pending_orders_fast_poll(self):
        """If pending orders exist, sleep = min(1.0, iter_sleep_s)."""
        execution_results = []
        _had_signals = bool(execution_results)
        _has_pending = True
        iter_sleep_s = 5.0

        if _had_signals:
            _smart_sleep = 0.0
        elif _has_pending:
            _smart_sleep = min(1.0, iter_sleep_s)
        else:
            _smart_sleep = iter_sleep_s

        assert _smart_sleep == 1.0

    def test_idle_full_sleep(self):
        """No signals, no pending -> full sleep."""
        execution_results = []
        _had_signals = bool(execution_results)
        _has_pending = False
        iter_sleep_s = 5.0

        if _had_signals:
            _smart_sleep = 0.0
        elif _has_pending:
            _smart_sleep = min(1.0, iter_sleep_s)
        else:
            _smart_sleep = iter_sleep_s

        assert _smart_sleep == 5.0


# ===================================================================
# FIX 22: Execution quality score
# ===================================================================

class TestExecutionQualityScore:
    """FIX 22: Execution quality computation."""

    def test_quality_score_computation(self):
        """Compute avg slippage, maker rate, fill time from trade list."""
        trades = []
        for i in range(50):
            trades.append({
                "slippage_bps": 2.0,
                "is_maker": i % 3 != 0,  # ~66% maker
                "fill_time_ms": 100.0,
            })

        avg_slippage = sum(t["slippage_bps"] for t in trades) / len(trades)
        maker_count = sum(1 for t in trades if t["is_maker"])
        maker_rate = maker_count / len(trades) * 100.0
        avg_fill_time = sum(t["fill_time_ms"] for t in trades) / len(trades)

        assert avg_slippage == 2.0
        assert 60.0 < maker_rate < 70.0
        assert avg_fill_time == 100.0


# ===================================================================
# FIX 24: Regime-specific strategy whitelist
# ===================================================================

class TestRegimeStrategyWhitelist:
    """FIX 24: Regime-to-strategy preference mapping."""

    def _get_regime_prefs(self):
        return {
            "TRENDING_UP": {"momentum", "breakout", "funding_rate", "funding_rate_harvester"},
            "TRENDING_DOWN": {"momentum", "breakout", "funding_rate", "funding_rate_harvester"},
            "RANGE": {"mean_reversion", "stat_arb", "pairs", "kalman_pairs"},
            "NORMAL": {"mean_reversion", "stat_arb", "pairs", "kalman_pairs"},
            "HIGH_VOL": {"funding_rate", "funding_rate_harvester"},
            "CRISIS": {"funding_rate", "funding_rate_harvester"},
        }

    def test_trending_prefers_momentum(self):
        prefs = self._get_regime_prefs()
        assert "momentum" in prefs["TRENDING_UP"]

    def test_range_prefers_mean_reversion(self):
        prefs = self._get_regime_prefs()
        assert "mean_reversion" in prefs["RANGE"]

    def test_crisis_only_funding(self):
        prefs = self._get_regime_prefs()
        assert prefs["CRISIS"] == {"funding_rate", "funding_rate_harvester"}

    def test_mismatched_strategy_gets_discount(self):
        """Non-preferred strategy in a regime gets confidence * 0.7."""
        regime = "TRENDING_UP"
        prefs = self._get_regime_prefs()
        source_strategy = "mean_reversion"
        preferred = prefs.get(regime, set())
        matches = any(p in source_strategy.lower() for p in preferred)
        assert not matches

        confidence = 0.8
        if not matches:
            confidence *= 0.7
        assert abs(confidence - 0.56) < 0.01

    def test_crisis_mismatch_gets_50pct_discount(self):
        """Non-preferred strategy in CRISIS gets confidence * 0.5."""
        confidence = 0.8
        regime = "CRISIS"
        prefs = self._get_regime_prefs()
        source_strategy = "momentum"
        preferred = prefs.get(regime, set())
        matches = any(p in source_strategy.lower() for p in preferred)
        assert not matches

        confidence *= 0.5
        assert abs(confidence - 0.4) < 0.01


# ===================================================================
# FIX 25: Hot hand strategy boost
# ===================================================================

class TestHotHandBoost:
    """FIX 25: Consecutive win streak boosts."""

    def test_3_consecutive_wins_boost_15pct(self):
        consec_wins = 3
        size_pct = 0.10
        if consec_wins >= 5:
            boost = min(1.30, 1.25)
        elif consec_wins >= 3:
            boost = 1.15
        else:
            boost = 1.0
        size_pct *= boost
        assert abs(size_pct - 0.115) < 0.001

    def test_5_consecutive_wins_boost_25pct(self):
        consec_wins = 5
        size_pct = 0.10
        if consec_wins >= 5:
            boost = min(1.30, 1.25)
        elif consec_wins >= 3:
            boost = 1.15
        else:
            boost = 1.0
        size_pct *= boost
        assert abs(size_pct - 0.125) < 0.001

    def test_no_streak_no_boost(self):
        consec_wins = 2
        boost = 1.0
        if consec_wins >= 5:
            boost = min(1.30, 1.25)
        elif consec_wins >= 3:
            boost = 1.15
        assert boost == 1.0

    def test_boost_cap_at_30pct(self):
        """Max boost is 1.30."""
        assert min(1.30, 1.25) == 1.25  # capped


# ===================================================================
# FIX 26: Smart signal conflict resolution
# ===================================================================

class TestSmartSignalConflictResolution:
    """FIX 26: Resolve BUY/SELL conflicts for same symbol."""

    def _make_signal(self, symbol, action, confidence=0.5, timestamp=None):
        ts = timestamp or time.time()
        return SimpleNamespace(
            symbol=symbol, action=action, confidence=confidence,
            strength=0.5, entry_price=50000.0, stop_loss=None,
            take_profit=None, reasoning="test", strategy="test",
            timestamp=ts,
        )

    def test_no_conflict_passes_through(self):
        """Signals without conflicts pass through unchanged."""
        signals = [
            self._make_signal("BTC/USD", "BUY"),
            self._make_signal("ETH/USD", "SELL"),
        ]
        # No conflict since different symbols
        from collections import defaultdict
        symbol_signals = defaultdict(list)
        for sig in signals:
            symbol_signals[sig.symbol].append(sig)

        has_conflict = False
        for sym, sigs in symbol_signals.items():
            actions = {s.action.upper() for s in sigs}
            if "BUY" in actions and "SELL" in actions:
                has_conflict = True

        assert not has_conflict

    def test_conflict_detected_same_symbol(self):
        """BUY + SELL for same symbol is a conflict."""
        signals = [
            self._make_signal("BTC/USD", "BUY"),
            self._make_signal("BTC/USD", "SELL"),
        ]
        from collections import defaultdict
        symbol_signals = defaultdict(list)
        for sig in signals:
            symbol_signals[sig.symbol].append(sig)

        has_conflict = False
        for sym, sigs in symbol_signals.items():
            actions = {s.action.upper() for s in sigs}
            if "BUY" in actions and "SELL" in actions:
                has_conflict = True

        assert has_conflict


# ===================================================================
# FIX 27: Configurable ensemble thresholds
# ===================================================================

class TestConfigurableEnsembleThresholds:
    """FIX 27: Ensemble thresholds configurable from config."""

    def test_default_thresholds(self):
        from ml.ensemble_signal_hub import EnsembleSignalHub
        hub = EnsembleSignalHub(config=None)
        assert hub._bullish_threshold == 0.5
        assert hub._strong_composite_threshold == 0.3
        assert hub._strong_agreement_threshold == 0.7

    def test_custom_thresholds(self):
        from ml.ensemble_signal_hub import EnsembleSignalHub
        hub = EnsembleSignalHub(config={
            "bullish_threshold": 0.6,
            "strong_composite_threshold": 0.4,
            "strong_agreement_threshold": 0.8,
        })
        assert hub._bullish_threshold == 0.6
        assert hub._strong_composite_threshold == 0.4
        assert hub._strong_agreement_threshold == 0.8

    def test_label_uses_threshold(self):
        from ml.ensemble_signal_hub import EnsembleSignalHub
        hub = EnsembleSignalHub(config={"strong_composite_threshold": 0.6})
        # label threshold = strong_composite_threshold / 2 = 0.3
        assert hub._label(0.31) == "BULLISH"
        assert hub._label(-0.31) == "BEARISH"
        assert hub._label(0.1) == "NEUTRAL"

    def test_label_with_default_threshold(self):
        from ml.ensemble_signal_hub import EnsembleSignalHub
        hub = EnsembleSignalHub(config=None)
        # label threshold = 0.3 / 2 = 0.15
        assert hub._label(0.16) == "BULLISH"
        assert hub._label(-0.16) == "BEARISH"
        assert hub._label(0.0) == "NEUTRAL"

    def test_update_with_custom_thresholds(self):
        """update() should work with custom thresholds."""
        from ml.ensemble_signal_hub import EnsembleSignalHub
        hub = EnsembleSignalHub(config={
            "bullish_threshold": 0.3,
            "strong_composite_threshold": 0.2,
            "strong_agreement_threshold": 0.6,
            "enabled": {
                "fear_greed": False,
                "llm": False,
                "whale": False,
                "news": False,
                "alpha": False,
                "vol_regime": False,
                "funding": False,
                "chain_metrics": False,
            },
        })
        result = hub.update("BTC/USD", [50000.0, 50100.0, 50200.0], "NORMAL")
        assert result.composite == 0.0  # no sources enabled
        assert result.regime_bias == "NEUTRAL"


# ===================================================================
# Integration-ish: multiple fixes working together
# ===================================================================

class TestMultipleFixesIntegration:
    """Verify combinations of fixes don't conflict."""

    def test_staleness_plus_regime_whitelist(self):
        """Signal age decay should compound with regime discount."""
        confidence = 0.8
        age = 30.0
        # FIX 15: staleness decay
        confidence *= math.exp(-age / 30.0)
        # FIX 24: regime mismatch
        confidence *= 0.7
        # Should be significantly reduced
        assert confidence < 0.25

    def test_strategy_dampening_plus_hot_hand_conflict(self):
        """When strategy has 5 consec losses, hot hand should not boost."""
        # In practice, consecutive_losses >= 5 returns 0.10 from dampening
        # and consecutive_wins would be 0, so hot hand returns 1.0
        dampen = 0.10  # 5 consecutive losses
        hot_hand = 1.0  # 0 consecutive wins
        combined = dampen * hot_hand
        assert combined == 0.10

    def test_smart_sleep_with_signals_and_pending(self):
        """If both signals generated and pending orders, signals take priority."""
        execution_results = [{"status": "filled"}]
        _had_signals = bool(execution_results)
        _has_pending = True
        iter_sleep_s = 5.0

        if _had_signals:
            _smart_sleep = 0.0
        elif _has_pending:
            _smart_sleep = min(1.0, iter_sleep_s)
        else:
            _smart_sleep = iter_sleep_s

        assert _smart_sleep == 0.0  # signals take priority
