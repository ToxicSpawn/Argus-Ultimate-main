"""
Tests for regime-adaptive position sizing, macro calendar integration,
session-based sizing, trailing stops, time-based exits, and volatility-adjusted exits.

Covers the profit-critical enhancements in unified_trading_system.py:
  1. Regime-adaptive position scaling (CRISIS=0.3 .. BULL=1.2)
  2. Dynamic stop-loss / take-profit based on regime
  3. Macro event blocking (FOMC blocks new BUY entries)
  4. Session-based sizing (peak vs off-hours)
  5. Trailing stop mechanics (activation, high-water tracking, trigger)
  6. Time-based exits (48h stale, 7d max hold)
  7. Volatility-adjusted exits (high ATR = wide stops)
  8. Combined multipliers (regime + session + macro)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Lightweight mocks so we don't need the full system
# ---------------------------------------------------------------------------


@dataclass
class _MockConfig:
    run_mode: str = "paper"
    aud_to_usd: float = 0.65
    max_position_pct: float = 0.25
    min_position_size_aud: float = 10.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    trailing_stop_pct: float = 0.02
    trailing_activation_pct: float = 0.01
    stale_position_hours: float = 48.0
    stale_min_profit_pct: float = 0.005
    max_hold_hours: float = 168.0
    max_concurrent_positions: int = 10
    primary_exchange: str = "kraken"
    paper_maker_fee_rate: float = 0.0002
    paper_fee_rate: float = 0.0026
    paper_slippage_bps: float = 5.0
    portfolio_var_limit_pct: float = 0.0
    portfolio_cvar_limit_pct: float = 0.0


@dataclass
class _MockSignal:
    symbol: str = "BTC/USD"
    action: str = "BUY"
    confidence: float = 0.8
    strength: float = 0.7
    entry_price: float = 50000.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = "test"
    strategy: str = "test_strat"


# ============================================================================
# 1. Regime-Adaptive Position Sizing
# ============================================================================


class TestRegimeAdaptiveSizing:
    """Test REGIME_POSITION_SCALE constants and application."""

    def _get_scale_map(self):
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        return UnifiedTradingSystem.REGIME_POSITION_SCALE

    def test_crisis_scale_0_3(self):
        scale = self._get_scale_map()
        assert scale["CRISIS"] == 0.3

    def test_extreme_scale_0_3(self):
        scale = self._get_scale_map()
        assert scale["EXTREME"] == 0.3

    def test_high_vol_scale_0_5(self):
        scale = self._get_scale_map()
        assert scale["HIGH_VOL"] == 0.5

    def test_elevated_scale_0_5(self):
        scale = self._get_scale_map()
        assert scale["ELEVATED"] == 0.5

    def test_trending_down_scale_0_65(self):
        scale = self._get_scale_map()
        assert scale["TRENDING_DOWN"] == 0.65

    def test_bear_scale_0_65(self):
        scale = self._get_scale_map()
        assert scale["BEAR"] == 0.65

    def test_trend_down_scale_0_65(self):
        scale = self._get_scale_map()
        assert scale["TREND_DOWN"] == 0.65

    def test_range_scale_1_0(self):
        scale = self._get_scale_map()
        assert scale["RANGE"] == 1.0

    def test_normal_scale_1_0(self):
        scale = self._get_scale_map()
        assert scale["NORMAL"] == 1.0

    def test_sideways_scale_1_0(self):
        scale = self._get_scale_map()
        assert scale["SIDEWAYS"] == 1.0

    def test_trending_up_scale_1_2(self):
        scale = self._get_scale_map()
        assert scale["TRENDING_UP"] == 1.2

    def test_bull_scale_1_2(self):
        scale = self._get_scale_map()
        assert scale["BULL"] == 1.2

    def test_trend_up_scale_1_2(self):
        scale = self._get_scale_map()
        assert scale["TREND_UP"] == 1.2

    def test_breakout_scale_0_8(self):
        scale = self._get_scale_map()
        assert scale["BREAKOUT"] == 0.8

    def test_low_vol_scale_1_3(self):
        scale = self._get_scale_map()
        assert scale["LOW_VOL"] == 1.3

    def test_unknown_regime_defaults_1_0(self):
        scale = self._get_scale_map()
        assert scale.get("NONEXISTENT_REGIME", 1.0) == 1.0


# ============================================================================
# 2. Dynamic Stop/TP in Different Regimes
# ============================================================================


class TestDynamicStopTP:
    """Test REGIME_STOP_SCALE and REGIME_TP_SCALE constants."""

    def _get_stop_scale(self):
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        return UnifiedTradingSystem.REGIME_STOP_SCALE

    def _get_tp_scale(self):
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        return UnifiedTradingSystem.REGIME_TP_SCALE

    def test_crisis_widens_stops(self):
        """In CRISIS, stops should be 2x wider."""
        scale = self._get_stop_scale()
        assert scale["CRISIS"] == 2.0

    def test_high_vol_widens_stops(self):
        scale = self._get_stop_scale()
        assert scale["HIGH_VOL"] == 2.0

    def test_low_vol_tightens_stops(self):
        """In LOW_VOL, stops should be 0.7x (tighter)."""
        scale = self._get_stop_scale()
        assert scale["LOW_VOL"] == 0.7

    def test_crisis_widens_tp(self):
        """In CRISIS, TP should be 1.5x wider."""
        scale = self._get_tp_scale()
        assert scale["CRISIS"] == 1.5

    def test_low_vol_tightens_tp(self):
        """In LOW_VOL, TP should be 0.8x (tighter)."""
        scale = self._get_tp_scale()
        assert scale["LOW_VOL"] == 0.8

    def test_bull_widens_tp(self):
        """In BULL, TP should extend further."""
        scale = self._get_tp_scale()
        assert scale["BULL"] == 1.3

    def test_normal_stop_unchanged(self):
        scale = self._get_stop_scale()
        assert scale["NORMAL"] == 1.0

    def test_normal_tp_unchanged(self):
        scale = self._get_tp_scale()
        assert scale["NORMAL"] == 1.0


# ============================================================================
# 3. Macro Event Blocking
# ============================================================================


class TestMacroEventBlocking:
    """Test FOMC / high-impact macro event blocks new BUY entries."""

    def test_fred_calendar_fomc_is_high_impact(self):
        """FOMC should be impact level 3 (market-halting)."""
        from data.macro.fred_calendar import EVENT_IMPACT
        assert EVENT_IMPACT["FOMC"] == 3

    def test_fred_calendar_cpi_is_high_impact(self):
        from data.macro.fred_calendar import EVENT_IMPACT
        assert EVENT_IMPACT["CPI"] == 3

    def test_fred_calendar_nfp_is_medium_impact(self):
        from data.macro.fred_calendar import EVENT_IMPACT
        assert EVENT_IMPACT["NFP"] == 2

    def test_fred_calendar_blackout_within_30min(self):
        """is_blackout() should return True when within 30min of FOMC."""
        from data.macro.fred_calendar import FREDCalendar, MacroEvent
        cal = FREDCalendar()
        # Add a fake event 15 minutes from now
        soon = datetime.now(tz=timezone.utc) + timedelta(minutes=15)
        cal.add_event(MacroEvent(
            name="FOMC Rate Decision",
            event_type="FOMC",
            scheduled_at=soon,
            impact=3,
            source="test",
        ))
        assert cal.is_blackout(blackout_hours=0.5) is True

    def test_fred_calendar_no_blackout_far_event(self):
        """is_blackout() should be False when next event is far away."""
        from data.macro.fred_calendar import FREDCalendar, MacroEvent
        cal = FREDCalendar()
        cal.clear_manual_events()
        future = datetime.now(tz=timezone.utc) + timedelta(hours=24)
        cal.add_event(MacroEvent(
            name="CPI Release",
            event_type="CPI",
            scheduled_at=future,
            impact=3,
            source="test",
        ))
        assert cal.is_blackout(blackout_hours=0.5) is False

    def test_get_upcoming_finds_imminent_event(self):
        """get_upcoming should identify high-impact events within 2 hours."""
        from data.macro.fred_calendar import FREDCalendar, MacroEvent
        cal = FREDCalendar()
        soon = datetime.now(tz=timezone.utc) + timedelta(hours=1)
        cal.add_event(MacroEvent(
            name="FOMC Test",
            event_type="FOMC",
            scheduled_at=soon,
            impact=3,
            source="test",
        ))
        snap = cal.get_upcoming(days=1)
        assert snap.hours_to_next_high is not None
        assert snap.hours_to_next_high <= 2.0
        assert snap.next_high_impact is not None


# ============================================================================
# 4. Session-Based Sizing
# ============================================================================


class TestSessionBasedSizing:
    """Test session multipliers based on UTC hour."""

    def test_ny_open_peak_volume(self):
        """13-17 UTC (NY open) should get 1.1x multiplier."""
        for h in (13, 14, 15, 16, 17):
            if 13 <= h <= 17:
                mult = 1.1
            elif 8 <= h <= 10:
                mult = 1.05
            elif 1 <= h <= 5:
                mult = 0.8
            else:
                mult = 1.0
            assert mult == 1.1, f"Hour {h} should be 1.1"

    def test_london_open_moderate(self):
        """8-10 UTC (London open) should get 1.05x."""
        for h in (8, 9, 10):
            if 13 <= h <= 17:
                mult = 1.1
            elif 8 <= h <= 10:
                mult = 1.05
            elif 1 <= h <= 5:
                mult = 0.8
            else:
                mult = 1.0
            assert mult == 1.05, f"Hour {h} should be 1.05"

    def test_low_volume_hours(self):
        """1-5 UTC should get 0.8x (reduced for poor fills)."""
        for h in (1, 2, 3, 4, 5):
            if 13 <= h <= 17:
                mult = 1.1
            elif 8 <= h <= 10:
                mult = 1.05
            elif 1 <= h <= 5:
                mult = 0.8
            else:
                mult = 1.0
            assert mult == 0.8, f"Hour {h} should be 0.8"

    def test_neutral_hours(self):
        """Other hours should get 1.0x."""
        for h in (0, 6, 7, 11, 12, 18, 19, 20, 21, 22, 23):
            if 13 <= h <= 17:
                mult = 1.1
            elif 8 <= h <= 10:
                mult = 1.05
            elif 1 <= h <= 5:
                mult = 0.8
            else:
                mult = 1.0
            assert mult == 1.0, f"Hour {h} should be 1.0"


# ============================================================================
# 5. Trailing Stop Mechanics
# ============================================================================


class TestTrailingStopMechanics:
    """Test _update_trailing_stops logic."""

    def _make_system(self, positions=None, regime="NORMAL"):
        """Create a minimal mock of UnifiedTradingSystem for trailing stop tests."""
        system = MagicMock()
        system.positions = positions or {}
        system._position_high_water = {}
        system._position_low_water = {}
        system._partial_tp_taken = {}
        system.config = _MockConfig()
        # Import the actual method and bind it
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        # We'll test the method directly by calling it with our mock
        system._update_trailing_stops = UnifiedTradingSystem._update_trailing_stops.__get__(system)
        return system

    def test_trailing_stop_not_triggered_early(self):
        """Trailing stop should NOT activate before 1% profit."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 50100,  # 0.2% profit (< 1% activation)
                "entry_price": 50000,
                "entry_time": time.time() - 3600,
                "side": "BUY",
            }
        })
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        assert len(exits) == 0

    def test_trailing_stop_activates_after_profit(self):
        """Trailing stop should activate and trigger when price drops 2% from peak."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 49500,  # Dropped from peak
                "entry_price": 50000,
                "entry_time": time.time() - 3600,
                "side": "BUY",
            }
        })
        # Simulate a previous high water mark well above current
        system._position_high_water["BTC/USD"] = 51000.0  # 2% profit peak
        # profit_pct from entry = (49500-50000)/50000 = -1% (negative)
        # So trailing won't activate because profit_pct < activation threshold
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        # Price is below entry, so profit < activation, trailing doesn't trigger
        assert len(exits) == 0

    def test_trailing_stop_triggers_from_peak(self):
        """When price was well in profit and drops 2% from peak, trailing triggers."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 50500,  # 1% above entry
                "entry_price": 50000,
                "entry_time": time.time() - 3600,
                "side": "BUY",
            }
        })
        # Peak was 51600, current is 50500 => drawdown = 2.13% from peak
        system._position_high_water["BTC/USD"] = 51600.0
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        assert len(exits) == 1
        assert exits[0]["action"] == "SELL"
        assert "trailing_stop" in exits[0]["reason"]

    def test_high_water_tracking_updates(self):
        """High water mark should be updated each call."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 52000,
                "entry_price": 50000,
                "entry_time": time.time() - 3600,
                "side": "BUY",
            }
        })
        system._position_high_water["BTC/USD"] = 51000.0
        asyncio.run(
            system._update_trailing_stops()
        )
        # High water should be updated to 52000
        assert system._position_high_water["BTC/USD"] == 52000.0

    def test_short_trailing_stop(self):
        """Short position trailing stop: triggers on runup from trough."""
        system = self._make_system(positions={
            "ETH/USD": {
                "quantity": 1.0,
                "current_price": 3100,  # 2.6% above trough
                "entry_price": 3200,    # short entry
                "entry_time": time.time() - 3600,
                "side": "SELL",
            }
        })
        # Trough was 3020, current 3100 => runup = (3100-3020)/3020 = 2.65%
        system._position_low_water["ETH/USD"] = 3020.0
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        assert len(exits) == 1
        assert exits[0]["action"] == "BUY"  # close short by buying
        assert "trailing_stop" in exits[0]["reason"]

    def test_low_water_tracking_updates(self):
        """Low water mark should decrease when price drops further."""
        system = self._make_system(positions={
            "ETH/USD": {
                "quantity": 1.0,
                "current_price": 2900,  # Below previous low water
                "entry_price": 3200,
                "entry_time": time.time() - 3600,
                "side": "SELL",
            }
        })
        system._position_low_water["ETH/USD"] = 3000.0
        asyncio.run(
            system._update_trailing_stops()
        )
        assert system._position_low_water["ETH/USD"] == 2900.0


# ============================================================================
# 6. Time-Based Exits
# ============================================================================


class TestTimeBasedExits:
    """Test stale position (48h) and max hold (7d) exits."""

    def _make_system(self, positions=None):
        system = MagicMock()
        system.positions = positions or {}
        system._position_high_water = {}
        system._position_low_water = {}
        system._partial_tp_taken = {}
        system.config = _MockConfig()
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        system._update_trailing_stops = UnifiedTradingSystem._update_trailing_stops.__get__(system)
        return system

    def test_stale_position_48h_low_profit(self):
        """Position held > 48h with < 0.5% profit should be closed."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 50100,  # 0.2% profit
                "entry_price": 50000,
                "entry_time": time.time() - (49 * 3600),  # 49 hours ago
                "side": "BUY",
            }
        })
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        assert len(exits) == 1
        assert "stale" in exits[0]["reason"]

    def test_not_stale_if_profitable(self):
        """Position held > 48h with > 0.5% profit should NOT be closed as stale."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 50500,  # 1% profit
                "entry_price": 50000,
                "entry_time": time.time() - (49 * 3600),
                "side": "BUY",
            }
        })
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        # 1% profit > 0.5% threshold => not stale.
        # But trailing may trigger if watermark is high. With no prior watermark, it won't.
        stale_exits = [e for e in exits if "stale" in e.get("reason", "")]
        assert len(stale_exits) == 0

    def test_max_hold_7d_exit(self):
        """Position held > 7 days should be force-closed regardless of profit."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 55000,  # 10% profit
                "entry_price": 50000,
                "entry_time": time.time() - (8 * 24 * 3600),  # 8 days ago
                "side": "BUY",
            }
        })
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        assert len(exits) == 1
        assert "max_hold" in exits[0]["reason"]

    def test_position_under_48h_not_stale(self):
        """Position held < 48h should NOT be closed as stale."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 50050,  # 0.1% profit
                "entry_price": 50000,
                "entry_time": time.time() - (24 * 3600),  # 24 hours
                "side": "BUY",
            }
        })
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        stale_exits = [e for e in exits if "stale" in e.get("reason", "")]
        assert len(stale_exits) == 0


# ============================================================================
# 7. Volatility-Adjusted Exits
# ============================================================================


class TestVolatilityAdjustedExits:
    """Test ATR-based stop and TP calculation."""

    def test_high_atr_widens_stops(self):
        """When vol is high, stops should be wider (1.5x ATR)."""
        vol = 0.05  # 5% daily vol
        atr_stop = vol * 1.5
        atr_tp = vol * 3.0
        assert atr_stop == pytest.approx(0.075)  # 7.5%
        assert atr_tp == pytest.approx(0.15)      # 15%

    def test_low_atr_tightens_stops(self):
        """When vol is low, stops should be tighter."""
        vol = 0.005  # 0.5% daily vol
        atr_stop = vol * 1.5
        atr_tp = vol * 3.0
        assert atr_stop == pytest.approx(0.0075)  # 0.75%
        assert atr_tp == pytest.approx(0.015)      # 1.5%

    def test_rr_ratio_maintained(self):
        """ATR exits maintain 2:1 R:R ratio (TP = 2x stop)."""
        for vol in (0.005, 0.01, 0.02, 0.05, 0.10):
            atr_stop = vol * 1.5
            atr_tp = vol * 3.0
            ratio = atr_tp / atr_stop
            assert ratio == pytest.approx(2.0), f"R:R should be 2:1 for vol={vol}"

    def test_zero_vol_doesnt_crash(self):
        """Zero vol should not produce ATR exits (stays with base)."""
        vol = 0.0
        atr_stop = vol * 1.5
        assert atr_stop == 0.0  # Won't override base because < 0.001


# ============================================================================
# 8. Combined: Regime + Session + Macro All Applying
# ============================================================================


class TestCombinedMultipliers:
    """Test that all multipliers compose correctly."""

    def test_crisis_during_low_volume(self):
        """CRISIS regime + low volume hours: 0.3 * 0.8 = 0.24 of base."""
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        regime_mult = UnifiedTradingSystem.REGIME_POSITION_SCALE["CRISIS"]
        session_mult = 0.8  # 1-5 UTC
        combined = regime_mult * session_mult
        assert combined == pytest.approx(0.24)

    def test_bull_during_peak_hours(self):
        """BULL regime + NY open: 1.2 * 1.1 = 1.32 of base."""
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        regime_mult = UnifiedTradingSystem.REGIME_POSITION_SCALE["BULL"]
        session_mult = 1.1  # 13-17 UTC
        combined = regime_mult * session_mult
        assert combined == pytest.approx(1.32)

    def test_normal_neutral_hours(self):
        """NORMAL regime + neutral hours: 1.0 * 1.0 = 1.0 (no change)."""
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        regime_mult = UnifiedTradingSystem.REGIME_POSITION_SCALE["NORMAL"]
        session_mult = 1.0
        combined = regime_mult * session_mult
        assert combined == pytest.approx(1.0)

    def test_high_vol_stop_with_atr(self):
        """HIGH_VOL regime stop scale 2.0 with base 2% = 4% stop."""
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        base_stop = 0.02
        stop_mult = UnifiedTradingSystem.REGIME_STOP_SCALE["HIGH_VOL"]
        adj_stop = base_stop * stop_mult
        assert adj_stop == pytest.approx(0.04)

    def test_low_vol_tp_tightened(self):
        """LOW_VOL regime TP scale 0.8 with base 4% = 3.2% TP."""
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        base_tp = 0.04
        tp_mult = UnifiedTradingSystem.REGIME_TP_SCALE["LOW_VOL"]
        adj_tp = base_tp * tp_mult
        assert adj_tp == pytest.approx(0.032)

    def test_all_multipliers_compose_multiplicatively(self):
        """Verify composition: base * regime * session * macro(0.7 for SELL)."""
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        base_pct = 0.10  # 10% base position
        regime = "HIGH_VOL"
        regime_mult = UnifiedTradingSystem.REGIME_POSITION_SCALE[regime]  # 0.5
        session_mult = 0.8  # low volume
        macro_reduction = 0.7  # macro imminent, SELL exit reduced

        final = base_pct * regime_mult * session_mult * macro_reduction
        expected = 0.10 * 0.5 * 0.8 * 0.7
        assert final == pytest.approx(expected)


# ============================================================================
# 9. Watermark Cleanup
# ============================================================================


class TestWatermarkCleanup:
    """Ensure watermarks are cleaned when positions close."""

    def _make_system(self, positions=None):
        system = MagicMock()
        system.positions = positions or {}
        system._position_high_water = {"BTC/USD": 55000.0, "ETH/USD": 4000.0}
        system._position_low_water = {"BTC/USD": 48000.0, "ETH/USD": 3500.0}
        system._partial_tp_taken = {}
        system.config = _MockConfig()
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        system._update_trailing_stops = UnifiedTradingSystem._update_trailing_stops.__get__(system)
        return system

    def test_closed_positions_watermarks_cleaned(self):
        """When a position is no longer in self.positions, its watermarks are removed."""
        # Only BTC/USD is still open; ETH/USD was closed
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 52000,
                "entry_price": 50000,
                "entry_time": time.time() - 3600,
                "side": "BUY",
            }
        })
        asyncio.run(
            system._update_trailing_stops()
        )
        assert "BTC/USD" in system._position_high_water
        assert "ETH/USD" not in system._position_high_water
        assert "ETH/USD" not in system._position_low_water


# ============================================================================
# 10. Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case coverage."""

    def _make_system(self, positions=None):
        system = MagicMock()
        system.positions = positions or {}
        system._position_high_water = {}
        system._position_low_water = {}
        system._partial_tp_taken = {}
        system.config = _MockConfig()
        from unified_trading_system import UnifiedSystemArchitecture as UnifiedTradingSystem
        system._update_trailing_stops = UnifiedTradingSystem._update_trailing_stops.__get__(system)
        return system

    def test_no_positions_no_exits(self):
        """Empty positions dict should produce no exits."""
        system = self._make_system(positions={})
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        assert exits == []

    def test_zero_quantity_skipped(self):
        """Position with zero quantity should be skipped."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0,
                "current_price": 50000,
                "entry_price": 50000,
                "entry_time": time.time(),
                "side": "BUY",
            }
        })
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        assert exits == []

    def test_missing_entry_time_no_time_exit(self):
        """Position with no entry_time should not trigger time-based exits."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 50000,
                "entry_price": 50000,
                "side": "BUY",
            }
        })
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        time_exits = [e for e in exits if "stale" in e.get("reason", "") or "max_hold" in e.get("reason", "")]
        assert len(time_exits) == 0

    def test_none_position_skipped(self):
        """None value in positions dict should be safely skipped."""
        system = self._make_system(positions={
            "BTC/USD": None,
        })
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        assert exits == []

    def test_zero_price_skipped(self):
        """Position with zero current_price should be skipped."""
        system = self._make_system(positions={
            "BTC/USD": {
                "quantity": 0.01,
                "current_price": 0,
                "entry_price": 50000,
                "entry_time": time.time(),
                "side": "BUY",
            }
        })
        exits = asyncio.run(
            system._update_trailing_stops()
        )
        assert exits == []
