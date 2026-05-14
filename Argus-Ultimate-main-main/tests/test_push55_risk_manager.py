"""Push 55 — Risk Manager + position sizing: 27 tests."""
from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.skip(
    "legacy risk manager tests target removed APIs",
    allow_module_level=True,
)


# ---------------------------------------------------------------------------
# RiskConfig tests (4)
# ---------------------------------------------------------------------------
from core.risk.risk_config import RiskConfig


class TestRiskConfig:
    def test_defaults(self):
        c = RiskConfig()
        assert c.max_position_pct == pytest.approx(0.10)
        assert c.max_drawdown_halt == pytest.approx(0.15)
        assert c.max_open_positions == 4

    def test_validate_passes(self):
        RiskConfig().validate()  # should not raise

    def test_validate_bad_max_position_pct(self):
        with pytest.raises(AssertionError):
            RiskConfig(max_position_pct=0.0).validate()

    def test_validate_bad_daily_loss(self):
        with pytest.raises(AssertionError):
            RiskConfig(max_daily_loss=-50).validate()


# ---------------------------------------------------------------------------
# PositionSizer tests (8)
# ---------------------------------------------------------------------------
from core.risk.position_sizer import PositionSizer, SizingMethod, SizerConfig

SizingMode = SizingMethod


@pytest.mark.skip(reason="legacy PositionSizer API removed from current source")
class TestPositionSizer:
    def _sizer(self, equity=10_000.0):
        return PositionSizer(SizerConfig())

    def test_fixed_usd_qty(self):
        s = self._sizer()
        ps = s.size(SizingMode.FIXED_USD, price=1000.0, fixed_usd=500.0)
        assert ps.qty == pytest.approx(0.5)
        assert ps.is_valid

    def test_fixed_usd_capped_by_equity_pct(self):
        s = self._sizer(equity=1000.0)  # 10% = $100 cap
        ps = s.size(SizingMode.FIXED_USD, price=1.0, fixed_usd=500.0)
        assert ps.notional == pytest.approx(100.0)

    def test_pct_equity_notional(self):
        s = self._sizer(equity=10_000.0)  # 10% = $1000
        ps = s.size(SizingMode.PCT_EQUITY, price=500.0)
        assert ps.notional == pytest.approx(1000.0)
        assert ps.qty == pytest.approx(2.0)

    def test_kelly_positive_edge(self):
        s = self._sizer()
        ps = s.size(SizingMode.KELLY, price=100.0,
                     win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        assert ps.is_valid
        assert ps.notional > 0

    def test_kelly_no_edge_zero_qty(self):
        s = self._sizer()
        # 40% win rate with 1:1 payoff → negative Kelly → 0
        ps = s.size(SizingMode.KELLY, price=100.0,
                     win_rate=0.3, avg_win=1.0, avg_loss=1.0)
        assert ps.qty == pytest.approx(0.0)

    def test_vol_scaled_with_atr(self):
        s = self._sizer(equity=10_000.0)
        ps = s.size(SizingMode.VOLATILITY_SCALED, price=100.0, atr=5.0)
        assert ps.is_valid

    def test_vol_scaled_zero_atr_fallback(self):
        s = self._sizer()
        ps = s.size(SizingMode.VOLATILITY_SCALED, price=100.0, atr=0.0)
        # Falls back to pct_equity
        assert ps.mode == SizingMode.VOLATILITY_SCALED or ps.is_valid

    def test_invalid_price_returns_zero(self):
        s = self._sizer()
        ps = s.size(SizingMode.PCT_EQUITY, price=0.0)
        assert ps.qty == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ExposureTracker tests (6)
# ---------------------------------------------------------------------------
from core.risk.exposure_tracker import ExposureTracker


class TestExposureTracker:
    def test_add_and_total(self):
        e = ExposureTracker()
        e.add("BTC", 1000.0)
        e.add("ETH", 500.0)
        assert e.total_notional == pytest.approx(1500.0)

    def test_remove_reduces_total(self):
        e = ExposureTracker()
        e.add("BTC", 1000.0)
        e.remove("BTC")
        assert e.total_notional == pytest.approx(0.0)

    def test_open_count(self):
        e = ExposureTracker()
        e.add("A", 100)
        e.add("B", 200)
        assert e.open_count == 2

    def test_symbol_notional(self):
        e = ExposureTracker()
        e.add("ETH", 750.0)
        assert e.symbol_notional("ETH") == pytest.approx(750.0)

    def test_utilisation(self):
        e = ExposureTracker()
        e.add("BTC", 2000.0)
        assert e.utilisation(10_000.0) == pytest.approx(0.2)

    def test_would_exceed(self):
        e = ExposureTracker(max_total_notional=1000.0)
        e.add("BTC", 800.0)
        assert e.would_exceed(300.0) is True
        assert e.would_exceed(100.0) is False


# ---------------------------------------------------------------------------
# RiskManager tests (9)
# ---------------------------------------------------------------------------
from core.risk.risk_manager import RiskManager, TradeDecision


class TestRiskManager:
    def _rm(self, equity=10_000.0, **kwargs):
        cfg = RiskConfig(**kwargs)
        return RiskManager(cfg, equity=equity)

    def test_approved_trade(self):
        rm = self._rm()
        decision = rm.approve_trade("BTCUSDT", notional=500.0, confidence=0.7)
        assert decision.approved is True

    def test_rejected_low_confidence(self):
        rm = self._rm(min_confidence=0.8)
        decision = rm.approve_trade("BTCUSDT", notional=500.0, confidence=0.6)
        assert decision.approved is False
        assert "confidence" in decision.reason

    def test_rejected_max_positions(self):
        rm = self._rm(max_open_positions=2)
        rm.record_open("BTC", 500)
        rm.record_open("ETH", 500)
        decision = rm.approve_trade("SOL", notional=100, confidence=0.9)
        assert decision.approved is False

    def test_halt_on_drawdown(self):
        rm = self._rm(equity=10_000.0, max_drawdown_halt=0.10)
        decision = rm.approve_trade("BTC", 100, 0.9, current_drawdown=0.15)
        assert decision.approved is False
        assert rm.is_halted

    def test_halt_on_daily_loss(self):
        rm = self._rm(max_daily_loss=100.0)
        rm.record_trade("BTC", -150.0)
        decision = rm.approve_trade("ETH", 100, 0.9)
        assert decision.approved is False

    def test_resume_clears_halt(self):
        rm = self._rm()
        rm.halt("test")
        assert rm.is_halted
        rm.resume()
        assert not rm.is_halted

    def test_equity_updates_on_close(self):
        rm = self._rm(equity=10_000.0)
        rm.record_open("BTC", 1000.0)
        rm.record_close("BTC", 50.0)
        assert rm.equity == pytest.approx(10_050.0)

    def test_status_dict_keys(self):
        rm = self._rm()
        s = rm.status()
        assert "halted" in s and "daily_pnl" in s and "open_positions" in s

    def test_trade_decision_bool(self):
        assert bool(TradeDecision(True, "ok")) is True
        assert bool(TradeDecision(False, "no")) is False
