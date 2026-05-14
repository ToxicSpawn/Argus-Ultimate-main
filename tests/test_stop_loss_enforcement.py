#!/usr/bin/env python3
"""
Tests for stop-loss auto-execution, margin enforcement, and per-strategy risk limits.

Covers:
- Fixed stop triggers
- Trailing stop high-water mark updates and triggers
- Time-based stops
- Multiple positions (only breached ones close)
- No false triggers
- Margin enforcement / deleverage
- Strategy risk limits and cooldowns
- Ordering: stops first, then margin
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from risk.unified_risk_manager import (
    UnifiedRiskManager,
    StrategyRiskLimits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rm(**kwargs) -> UnifiedRiskManager:
    """Create a UnifiedRiskManager with sensible test defaults."""
    defaults = dict(
        initial_capital=10000.0,
        max_daily_loss=0.02,
        max_leverage=3.0,
    )
    defaults.update(kwargs)
    return UnifiedRiskManager(**defaults)


def _long_position(entry_price: float, quantity: float = 1.0, current_price: float = 0.0, side: str = "long"):
    return {
        "entry_price": entry_price,
        "quantity": quantity,
        "current_price": current_price or entry_price,
        "side": side,
    }


def _short_position(entry_price: float, quantity: float = 1.0, current_price: float = 0.0):
    return _long_position(entry_price, quantity, current_price, side="short")


# ===========================================================================
# Fixed Stop Tests
# ===========================================================================

class TestFixedStop:

    def test_fixed_stop_triggers_on_price_drop_long(self):
        """Fixed stop triggers when price drops below entry * (1 - stop_pct)."""
        rm = _make_rm()
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=48900)}
        prices = {"BTC/USD": 48900.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.02)
        assert len(stops) == 1
        assert stops[0]["symbol"] == "BTC/USD"
        assert stops[0]["side"] == "SELL"
        assert "fixed_stop_loss" in stops[0]["reason"]

    def test_fixed_stop_triggers_on_price_rise_short(self):
        """Fixed stop triggers for short when price rises above entry * (1 + stop_pct)."""
        rm = _make_rm()
        positions = {"ETH/USD": _short_position(3000, 1.0, current_price=3070)}
        prices = {"ETH/USD": 3070.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.02)
        assert len(stops) == 1
        assert stops[0]["side"] == "BUY"
        assert "fixed_stop_loss" in stops[0]["reason"]

    def test_no_trigger_when_price_above_stop_long(self):
        """No stop when price is still above the fixed stop level."""
        rm = _make_rm()
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=49500)}
        prices = {"BTC/USD": 49500.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.02)
        assert len(stops) == 0

    def test_no_trigger_when_price_below_stop_short(self):
        """No stop for short when price is below the stop level."""
        rm = _make_rm()
        positions = {"ETH/USD": _short_position(3000, 1.0, current_price=2950)}
        prices = {"ETH/USD": 2950.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.02)
        assert len(stops) == 0

    def test_fixed_stop_exact_boundary_long(self):
        """Stop triggers at exact boundary (entry * (1 - pct))."""
        rm = _make_rm()
        entry = 10000.0
        stop_pct = 0.05
        stop_price = entry * (1.0 - stop_pct)  # 9500
        positions = {"BTC/USD": _long_position(entry, 1.0, current_price=stop_price)}
        prices = {"BTC/USD": stop_price}
        stops = rm.check_stops(positions, prices, stop_loss_pct=stop_pct)
        assert len(stops) == 1

    def test_correct_quantity_in_stop(self):
        """Stop signal carries the full position quantity."""
        rm = _make_rm()
        positions = {"BTC/USD": _long_position(50000, 0.5, current_price=45000)}
        prices = {"BTC/USD": 45000.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.05)
        assert len(stops) == 1
        assert stops[0]["quantity"] == 0.5


# ===========================================================================
# Trailing Stop Tests
# ===========================================================================

class TestTrailingStop:

    def test_trailing_stop_updates_high_water_mark(self):
        """update_trailing_stops updates the high-water mark."""
        rm = _make_rm()
        rm.update_trailing_stops("BTC/USD", 50000)
        rm.update_trailing_stops("BTC/USD", 55000)
        rm.update_trailing_stops("BTC/USD", 53000)  # Should NOT lower it
        assert rm._trailing_highs["BTC/USD"] == 55000

    def test_trailing_stop_triggers_on_reversal(self):
        """Trailing stop triggers when price drops trail_pct below high water mark."""
        rm = _make_rm()
        # Entry at 50000, price went to 55000 then dropped
        rm.update_trailing_stops("BTC/USD", 55000)
        trail_pct = 0.02
        # Trail stop = 55000 * (1 - 0.02) = 53900
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=53800)}
        prices = {"BTC/USD": 53800.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.10, trail_pct=trail_pct)
        assert len(stops) == 1
        assert "trailing_stop" in stops[0]["reason"]

    def test_trailing_stop_no_trigger_when_above_trail(self):
        """Trailing stop does not trigger when price is above trail level."""
        rm = _make_rm()
        rm.update_trailing_stops("BTC/USD", 55000)
        trail_pct = 0.02
        # Trail stop = 55000 * 0.98 = 53900
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=54000)}
        prices = {"BTC/USD": 54000.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.10, trail_pct=trail_pct)
        assert len(stops) == 0

    def test_trailing_stop_ignored_if_no_profit_yet(self):
        """Trailing stop should not trigger if high water is at entry (no profit run-up)."""
        rm = _make_rm()
        # High water mark equals entry price - no run-up
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=49500)}
        prices = {"BTC/USD": 49500.0}
        # Fixed stop at 50000 * 0.90 = 45000, so no fixed trigger
        # Trail: high=50000 (set by check_stops), but high <= entry so trailing skipped
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.10, trail_pct=0.02)
        assert len(stops) == 0


# ===========================================================================
# Time Stop Tests
# ===========================================================================

class TestTimeStop:

    def test_time_stop_closes_after_max_hours(self):
        """Time stop triggers when position held longer than max_hold_hours."""
        rm = _make_rm()
        # Register entry time 100 hours ago
        rm.register_entry_time("BTC/USD", datetime.now() - timedelta(hours=100))
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=51000)}
        prices = {"BTC/USD": 51000.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.10, trail_pct=0.10, max_hold_hours=72.0)
        assert len(stops) == 1
        assert "time_stop" in stops[0]["reason"]

    def test_time_stop_no_trigger_before_max_hours(self):
        """Time stop does not trigger before max_hold_hours."""
        rm = _make_rm()
        rm.register_entry_time("BTC/USD", datetime.now() - timedelta(hours=10))
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=51000)}
        prices = {"BTC/USD": 51000.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.10, trail_pct=0.10, max_hold_hours=72.0)
        assert len(stops) == 0

    def test_time_stop_returns_correct_side_for_short(self):
        """Time stop for a short position returns BUY side."""
        rm = _make_rm()
        rm.register_entry_time("ETH/USD", datetime.now() - timedelta(hours=100))
        positions = {"ETH/USD": _short_position(3000, 1.0, current_price=2900)}
        prices = {"ETH/USD": 2900.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.10, trail_pct=0.10, max_hold_hours=72.0)
        assert len(stops) == 1
        assert stops[0]["side"] == "BUY"


# ===========================================================================
# Multiple Position Tests
# ===========================================================================

class TestMultiplePositions:

    def test_only_breached_positions_close(self):
        """Only positions that breach their stop get closed."""
        rm = _make_rm()
        positions = {
            "BTC/USD": _long_position(50000, 0.1, current_price=48000),  # Breached (4%)
            "ETH/USD": _long_position(3000, 1.0, current_price=2990),   # Safe (0.33%)
            "SOL/USD": _long_position(100, 10.0, current_price=97),     # Breached (3%)
        }
        prices = {"BTC/USD": 48000.0, "ETH/USD": 2990.0, "SOL/USD": 97.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.02)
        symbols_closed = {s["symbol"] for s in stops}
        assert "BTC/USD" in symbols_closed
        assert "SOL/USD" in symbols_closed
        assert "ETH/USD" not in symbols_closed

    def test_zero_quantity_skipped(self):
        """Positions with zero quantity are skipped."""
        rm = _make_rm()
        positions = {"BTC/USD": _long_position(50000, 0.0, current_price=40000)}
        prices = {"BTC/USD": 40000.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.02)
        assert len(stops) == 0

    def test_none_position_skipped(self):
        """None positions are gracefully skipped."""
        rm = _make_rm()
        positions = {"BTC/USD": None}
        prices = {"BTC/USD": 40000.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.02)
        assert len(stops) == 0


# ===========================================================================
# Margin Enforcement Tests
# ===========================================================================

class TestMarginEnforcement:

    def test_overleveraged_triggers_force_close(self):
        """When leverage exceeds max, enforce_margin returns closures."""
        rm = _make_rm(max_leverage=2.0)
        positions = {
            "BTC/USD": _long_position(50000, 0.5, current_price=50000),  # 25000 notional
            "ETH/USD": _long_position(3000, 5.0, current_price=3000),    # 15000 notional
        }
        # Total notional: 40000, capital: 10000 => 4x leverage > 2x max
        prices = {"BTC/USD": 50000.0, "ETH/USD": 3000.0}
        closures = rm.enforce_margin(positions, prices, total_capital=10000.0)
        assert len(closures) > 0
        total_closed_notional = sum(c["quantity_to_close"] * prices[c["symbol"]] for c in closures)
        # After closing, remaining notional should be <= 2x * 10000 = 20000
        remaining = 40000.0 - total_closed_notional
        assert remaining <= 20000.0 + 0.01  # small float tolerance

    def test_under_leverage_no_action(self):
        """When leverage is under max, no closures."""
        rm = _make_rm(max_leverage=3.0)
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=50000)}  # 5000 notional
        prices = {"BTC/USD": 50000.0}
        closures = rm.enforce_margin(positions, prices, total_capital=10000.0)
        assert len(closures) == 0

    def test_deleverage_cuts_largest_loser_first(self):
        """Deleverage should close worst-performing positions first."""
        rm = _make_rm(max_leverage=2.0)
        positions = {
            "WINNER": _long_position(100, 100, current_price=110),    # +1000 PnL, 11000 notional
            "LOSER":  _long_position(100, 100, current_price=90),     # -1000 PnL, 9000 notional
        }
        prices = {"WINNER": 110.0, "LOSER": 90.0}
        closures = rm.deleverage(positions, 2.0, prices, total_capital=5000.0)
        # Total notional = 20000, target = 10000, excess = 10000
        # LOSER (pnl=-1000) should be closed first
        assert len(closures) >= 1
        assert closures[0]["symbol"] == "LOSER"

    def test_deleverage_partial_close(self):
        """When only a partial close is needed to reach target leverage."""
        rm = _make_rm(max_leverage=2.0)
        # Single position: 15000 notional, capital 10000 => 1.5x (under 2x)
        positions = {"BTC/USD": _long_position(100, 150, current_price=100)}
        prices = {"BTC/USD": 100.0}
        closures = rm.deleverage(positions, 1.0, prices, total_capital=10000.0)
        # Target: 10000, notional: 15000, excess: 5000
        assert len(closures) == 1
        assert closures[0]["quantity_to_close"] == pytest.approx(50.0, rel=0.01)

    def test_enforce_margin_zero_capital(self):
        """Zero capital should return empty list, not crash."""
        rm = _make_rm()
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=50000)}
        closures = rm.enforce_margin(positions, {"BTC/USD": 50000.0}, total_capital=0.0)
        assert closures == []


# ===========================================================================
# Strategy Risk Limits Tests
# ===========================================================================

class TestStrategyLimits:

    def test_daily_loss_blocks_new_trades(self):
        """Strategy exceeding daily loss limit is blocked."""
        rm = _make_rm(initial_capital=10000.0)
        limits = StrategyRiskLimits(max_daily_loss_pct=2.0)
        rm.set_strategy_limits(limits)
        # Daily loss of 250 on 10000 = 2.5% > 2.0%
        allowed, reason = rm.check_strategy_limits("momentum", daily_pnl=-250.0)
        assert not allowed
        assert "strategy_daily_loss" in reason

    def test_daily_loss_within_limit_allowed(self):
        """Strategy within daily loss limit is allowed."""
        rm = _make_rm(initial_capital=10000.0)
        limits = StrategyRiskLimits(max_daily_loss_pct=5.0)
        rm.set_strategy_limits(limits)
        allowed, reason = rm.check_strategy_limits("momentum", daily_pnl=-100.0)
        assert allowed

    def test_consecutive_losses_trigger_cooldown(self):
        """5 consecutive losses should trigger a cooldown."""
        rm = _make_rm()
        limits = StrategyRiskLimits(max_consecutive_losses=5, cooldown_after_loss_streak_minutes=60)
        rm.set_strategy_limits(limits)
        allowed, reason = rm.check_strategy_limits("scalper", daily_pnl=0.0, consecutive_losses=5)
        assert not allowed
        assert "cooldown" in reason

    def test_consecutive_losses_under_limit_allowed(self):
        """Under the consecutive loss limit, trading is allowed."""
        rm = _make_rm()
        limits = StrategyRiskLimits(max_consecutive_losses=5)
        rm.set_strategy_limits(limits)
        allowed, reason = rm.check_strategy_limits("scalper", daily_pnl=0.0, consecutive_losses=3)
        assert allowed

    def test_no_limits_configured_always_allows(self):
        """Without configured limits, all strategies are allowed."""
        rm = _make_rm()
        allowed, reason = rm.check_strategy_limits("any_strategy")
        assert allowed
        assert "no_limits" in reason

    def test_record_strategy_trade_tracks_losses(self):
        """record_strategy_trade properly updates consecutive loss count."""
        rm = _make_rm()
        limits = StrategyRiskLimits(max_consecutive_losses=3, cooldown_after_loss_streak_minutes=30)
        rm.set_strategy_limits(limits)
        rm.record_strategy_trade("test_strat", -10.0)
        rm.record_strategy_trade("test_strat", -20.0)
        rm.record_strategy_trade("test_strat", -5.0)
        allowed, reason = rm.check_strategy_limits("test_strat")
        assert not allowed
        assert "cooldown" in reason

    def test_winning_trade_resets_consecutive_losses(self):
        """A winning trade resets the consecutive loss counter."""
        rm = _make_rm()
        limits = StrategyRiskLimits(max_consecutive_losses=3)
        rm.set_strategy_limits(limits)
        rm.record_strategy_trade("test_strat", -10.0)
        rm.record_strategy_trade("test_strat", -20.0)
        rm.record_strategy_trade("test_strat", 50.0)  # Win resets
        allowed, reason = rm.check_strategy_limits("test_strat")
        assert allowed


# ===========================================================================
# Ordering / Integration Tests
# ===========================================================================

class TestStopMarginOrdering:

    def test_stops_check_before_margin(self):
        """Verify that check_stops and enforce_margin can run sequentially."""
        rm = _make_rm(max_leverage=2.0)
        positions = {
            "BTC/USD": _long_position(50000, 0.5, current_price=48000),  # Stop hit
            "ETH/USD": _long_position(3000, 5.0, current_price=3000),    # No stop
        }
        prices = {"BTC/USD": 48000.0, "ETH/USD": 3000.0}

        # Step 1: stops fire first
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.02)
        assert len(stops) >= 1  # BTC/USD should be stopped

        # Step 2: simulate removing stopped position, then check margin
        remaining = {k: v for k, v in positions.items() if k not in {s["symbol"] for s in stops}}
        margin = rm.enforce_margin(remaining, prices, total_capital=10000.0)
        # ETH has 15000 notional vs 10000 capital = 1.5x, under 2.0x
        assert len(margin) == 0

    def test_clear_position_tracking(self):
        """clear_position_tracking removes trailing/time data for a symbol."""
        rm = _make_rm()
        rm.update_trailing_stops("BTC/USD", 55000)
        rm.register_entry_time("BTC/USD", datetime.now() - timedelta(hours=50))
        rm.clear_position_tracking("BTC/USD")
        assert "BTC/USD" not in rm._trailing_highs
        assert "BTC/USD" not in rm._position_entry_times


class TestStrategyRiskLimitsDataclass:

    def test_defaults(self):
        """StrategyRiskLimits has expected defaults."""
        limits = StrategyRiskLimits()
        assert limits.max_daily_loss_pct == 2.0
        assert limits.max_consecutive_losses == 5
        assert limits.max_position_pct == 10.0
        assert limits.cooldown_after_loss_streak_minutes == 60

    def test_custom_values(self):
        """StrategyRiskLimits accepts custom values."""
        limits = StrategyRiskLimits(
            max_daily_loss_pct=1.0,
            max_consecutive_losses=3,
            max_position_pct=5.0,
            cooldown_after_loss_streak_minutes=30,
        )
        assert limits.max_daily_loss_pct == 1.0
        assert limits.max_consecutive_losses == 3


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_positions_no_crash(self):
        """Empty positions dict should return empty list."""
        rm = _make_rm()
        assert rm.check_stops({}, {}) == []
        assert rm.enforce_margin({}, {}, 10000.0) == []

    def test_missing_prices_skipped(self):
        """Positions without prices are gracefully skipped."""
        rm = _make_rm()
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=0.0)}
        stops = rm.check_stops(positions, {}, stop_loss_pct=0.02)
        assert len(stops) == 0

    def test_negative_prices_skipped(self):
        """Negative prices are skipped."""
        rm = _make_rm()
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=-1)}
        stops = rm.check_stops(positions, {"BTC/USD": -1.0}, stop_loss_pct=0.02)
        assert len(stops) == 0

    def test_zero_stop_loss_pct(self):
        """Zero stop_loss_pct means stop is at entry price (triggers easily)."""
        rm = _make_rm()
        positions = {"BTC/USD": _long_position(50000, 0.1, current_price=50000)}
        prices = {"BTC/USD": 50000.0}
        stops = rm.check_stops(positions, prices, stop_loss_pct=0.0)
        # Price == entry == stop level, should trigger (<=)
        assert len(stops) == 1
