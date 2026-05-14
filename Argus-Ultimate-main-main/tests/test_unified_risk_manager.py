"""Unit tests for UnifiedRiskManager — 25 tests covering all critical paths."""
from __future__ import annotations

import pytest
from risk.unified_risk_manager import UnifiedRiskManager, RiskLevel


@pytest.fixture
def rm() -> UnifiedRiskManager:
    return UnifiedRiskManager(
        initial_capital=10_000.0,
        max_daily_loss=0.02,
        max_consecutive_losses=3,
        circuit_breaker_cooldown_minutes=60,
        max_leverage=3.0,
        max_total_exposure=0.8,
    )


class TestInitialState:
    def test_initial_capital(self, rm):
        assert rm.current_capital == 10_000.0
        assert rm.peak_capital == 10_000.0

    def test_no_circuit_breaker_on_init(self, rm):
        assert not rm.circuit_breaker_active
        assert rm.check_circuit_breaker() is False

    def test_daily_pnl_zero(self, rm):
        assert rm.daily_pnl == 0.0

    def test_consecutive_losses_zero(self, rm):
        assert rm.consecutive_losses == 0


class TestCircuitBreaker:
    def test_consecutive_losses_trigger(self, rm):
        for _ in range(3):
            rm.record_trade(-100.0)
        assert rm.circuit_breaker_active is True

    def test_winning_trade_resets_streak(self, rm):
        rm.record_trade(-100.0)
        rm.record_trade(-100.0)
        rm.record_trade(500.0)  # win — resets streak
        assert rm.consecutive_losses == 0
        assert rm.circuit_breaker_active is False

    def test_trip_circuit_breaker_directly(self, rm):
        rm.trip_circuit_breaker("test")
        assert rm.circuit_breaker_active is True
        assert rm.circuit_breaker_reason == "test"

    def test_pre_trade_blocked_when_circuit_open(self, rm):
        rm.trip_circuit_breaker("test")
        ok, reason = rm.pre_trade_risk_check("BTC/AUD", 1000.0)
        assert ok is False
        assert "circuit" in reason


class TestDailyLossLimit:
    def test_daily_loss_not_exceeded(self, rm):
        rm.record_trade(-100.0)  # 1% loss — under 2% limit
        assert rm.is_daily_loss_limit_exceeded() is False

    def test_daily_loss_exceeded(self, rm):
        # 2% of 10_000 = 200; record 201 loss
        rm.daily_pnl = -201.0
        assert rm.is_daily_loss_limit_exceeded() is True

    def test_daily_loss_triggers_circuit_breaker(self, rm):
        rm.daily_pnl = -201.0
        rm.check_circuit_breaker()
        assert rm.circuit_breaker_active is True

    def test_pre_trade_blocked_on_daily_loss(self, rm):
        rm.daily_pnl = -201.0
        ok, reason = rm.pre_trade_risk_check("BTC/AUD", 100.0)
        assert ok is False


class TestCapitalTracking:
    def test_update_capital_raises_peak(self, rm):
        rm.update_capital(12_000.0, pnl=2000.0)
        assert rm.peak_capital == 12_000.0

    def test_update_capital_does_not_lower_peak(self, rm):
        rm.update_capital(12_000.0, pnl=2000.0)
        rm.update_capital(9_000.0, pnl=-3000.0)
        assert rm.peak_capital == 12_000.0

    def test_returns_history_populated(self, rm):
        rm.update_capital(10_100.0, pnl=100.0)
        assert len(rm.returns_history) == 1


class TestLeverage:
    def test_leverage_exceeded_blocks_trade(self, rm):
        rm.set_total_exposure(25_001.0)  # > 3x of 10k
        ok, reason = rm.pre_trade_risk_check("ETH/AUD", 1.0)
        assert ok is False
        assert "leverage" in reason

    def test_leverage_within_limit_passes(self, rm):
        rm.set_total_exposure(1_000.0)
        ok, _ = rm.pre_trade_risk_check("ETH/AUD", 100.0)
        assert ok is True


class TestMarginTracking:
    def test_update_and_total_margin(self, rm):
        rm.update_margin_requirement("BTC", 1000.0)
        rm.update_margin_requirement("ETH", 500.0)
        assert rm.get_total_margin() == 1500.0

    def test_clear_margin(self, rm):
        rm.update_margin_requirement("BTC", 1000.0)
        rm.update_margin_requirement("BTC", 0.0)
        assert rm.get_total_margin() == 0.0

    def test_free_margin(self, rm):
        rm.update_margin_requirement("BTC", 3000.0)
        assert rm.get_free_margin(10_000.0) == 7_000.0

    def test_insufficient_margin_blocks_trade(self, rm):
        rm.update_margin_requirement("BTC", 9_500.0)
        ok, reason = rm.pre_trade_risk_check("ETH/AUD", 100.0, required_margin_usd=1000.0)
        assert ok is False
        assert "margin" in reason

    def test_margin_call_returns_reductions(self, rm):
        rm.update_margin_requirement("BTC", 9_000.0)
        reductions = rm.check_margin_call(10_000.0, margin_call_pct=80.0)
        assert len(reductions) > 0
        assert reductions[0]["symbol"] == "BTC"


class TestFlashCrash:
    def test_flash_crash_trips_breaker(self, rm):
        triggered = rm.check_flash_crash("BTC", current_price=50_000.0, previous_price=100_000.0)
        assert triggered is True
        assert rm.circuit_breaker_active is True

    def test_normal_move_no_trip(self, rm):
        triggered = rm.check_flash_crash("BTC", current_price=99_000.0, previous_price=100_000.0)
        assert triggered is False


class TestLiquidationPrice:
    def test_long_liquidation_below_entry(self):
        liq = UnifiedRiskManager.calculate_liquidation_price(100.0, leverage=10.0, side="long")
        assert liq < 100.0

    def test_short_liquidation_above_entry(self):
        liq = UnifiedRiskManager.calculate_liquidation_price(100.0, leverage=10.0, side="short")
        assert liq > 100.0

    def test_invalid_leverage_raises(self):
        with pytest.raises(ValueError):
            UnifiedRiskManager.calculate_liquidation_price(100.0, leverage=0.0, side="long")


class TestRegimeSizing:
    def test_crisis_regime_quarters_limit(self):
        result = UnifiedRiskManager.get_regime_adjusted_position_limit(1000.0, "CRISIS")
        assert result == 250.0

    def test_unknown_regime_full_limit(self):
        result = UnifiedRiskManager.get_regime_adjusted_position_limit(1000.0, "UNKNOWN")
        assert result == 1000.0

    def test_high_vol_halves_limit(self):
        result = UnifiedRiskManager.get_regime_adjusted_position_limit(1000.0, "HIGH_VOL")
        assert result == 500.0
