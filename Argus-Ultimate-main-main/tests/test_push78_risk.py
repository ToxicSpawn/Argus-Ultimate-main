"""Push 78 — Tests: RiskEvent, RiskEventBus, RiskManager,
PositionSizer, MarginWatcher. 32 tests.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# RiskEvent + RiskEventBus (5)
# ---------------------------------------------------------------------------

class TestRiskEventBus:
    def test_emit_and_receive(self):
        from core.risk.risk_event import RiskEventBus, RiskEvent, RiskEventType
        bus = RiskEventBus()
        received = []
        bus.subscribe(received.append)
        bus.emit(RiskEvent(RiskEventType.KILL_SWITCH, "test"))
        assert len(received) == 1

    def test_history_stored(self):
        from core.risk.risk_event import RiskEventBus, RiskEvent, RiskEventType
        bus = RiskEventBus()
        bus.emit(RiskEvent(RiskEventType.MARGIN_SOFT, "soft"))
        assert len(bus.history) == 1

    def test_filter_by_type(self):
        from core.risk.risk_event import RiskEventBus, RiskEvent, RiskEventType
        bus = RiskEventBus()
        bus.emit(RiskEvent(RiskEventType.KILL_SWITCH, "ks"))
        bus.emit(RiskEvent(RiskEventType.MARGIN_SOFT, "soft"))
        ks = bus.events_of_type(RiskEventType.KILL_SWITCH)
        assert len(ks) == 1

    def test_handler_exception_ignored(self):
        from core.risk.risk_event import RiskEventBus, RiskEvent, RiskEventType
        bus = RiskEventBus()
        bus.subscribe(lambda e: 1 / 0)
        bus.emit(RiskEvent(RiskEventType.KILL_SWITCH, "x"))  # no raise

    def test_clear_history(self):
        from core.risk.risk_event import RiskEventBus, RiskEvent, RiskEventType
        bus = RiskEventBus()
        bus.emit(RiskEvent(RiskEventType.KILL_SWITCH, "x"))
        bus.clear_history()
        assert len(bus.history) == 0


# ---------------------------------------------------------------------------
# RiskManager (12)
# ---------------------------------------------------------------------------

class TestRiskManager:
    def _rm(self, **kwargs):
        from core.risk.risk_manager import RiskManager, RiskConfig
        cfg = RiskConfig(initial_equity=10_000, **kwargs)
        return RiskManager(config=cfg)

    def test_order_allowed_initially(self):
        rm = self._rm()
        ok, msg = rm.check_order_allowed("BTCUSDT", 1000)
        assert ok

    def test_kill_switch_blocks(self):
        rm = self._rm()
        rm.activate_kill_switch("test")
        ok, _ = rm.check_order_allowed("BTCUSDT", 1000)
        assert not ok

    def test_kill_switch_reset(self):
        rm = self._rm()
        rm.activate_kill_switch("test")
        rm.reset_kill_switch()
        ok, _ = rm.check_order_allowed("BTCUSDT", 1000)
        assert ok

    def test_daily_loss_blocks(self):
        rm = self._rm(daily_loss_limit_pct=1.0, kill_on_daily_loss=False)
        # 10000 * 1% = 100 → lose 200
        rm._daily_pnl = -200
        ok, _ = rm.check_order_allowed("BTCUSDT", 100)
        assert not ok

    def test_portfolio_heat_blocks(self):
        rm = self._rm(max_portfolio_heat=0.5)
        rm.update_open_notional("BTCUSDT", 6000)  # 60% heat on 10k equity
        ok, _ = rm.check_order_allowed("BTCUSDT", 1000)
        assert not ok

    def test_symbol_drawdown_blocks(self):
        rm = self._rm(max_symbol_drawdown_pct=5.0)
        rm.update_symbol_pnl("BTCUSDT", 0)  # init state
        state = rm._symbol_states["BTCUSDT"]
        state.peak_equity = 10000
        state.current_pnl = -600  # 6% drawdown
        ok, _ = rm.check_order_allowed("BTCUSDT", 100)
        assert not ok

    def test_var_computed(self):
        rm = self._rm()
        for r in [-2, -1, 0, 1, 2, -3, -0.5, 0.5, -1.5, 1.5, -2.5, 0.3]:
            rm.record_return(r)
        var95, cvar95 = rm.compute_var()
        assert var95 >= 0
        assert cvar95 >= 0

    def test_var_zero_insufficient_data(self):
        rm = self._rm()
        var95, _ = rm.compute_var()
        assert var95 == 0.0

    def test_var_99_gt_var_95(self):
        rm = self._rm()
        for r in list(range(-10, 10)) + [-5, -8, -12, -1]:
            rm.record_return(float(r))
        v95, _ = rm.compute_var(0.95)
        v99, _ = rm.compute_var(0.99)
        assert v99 >= v95

    def test_update_equity(self):
        rm = self._rm()
        rm.update_equity(20000)
        assert rm._equity == 20000

    def test_stats_dict(self):
        rm = self._rm()
        s = rm.stats
        assert "kill_switch" in s and "var_95" in s

    def test_force_daily_reset(self):
        rm = self._rm()
        rm._daily_pnl = -500
        rm.force_daily_reset()
        assert rm._daily_pnl == 0.0


# ---------------------------------------------------------------------------
# PositionSizer (10)
# ---------------------------------------------------------------------------

class TestPositionSizer:
    def _sizer(self, method="kelly"):
        from core.risk.position_sizer import PositionSizer, SizerConfig, SizingMethod
        cfg = SizerConfig(method=SizingMethod(method))
        return PositionSizer(cfg)

    def test_kelly_positive(self):
        s = self._sizer("kelly")
        qty = s.size(equity=10000, price=50000, strength=0.8)
        assert qty > 0

    def test_kelly_zero_price(self):
        s = self._sizer("kelly")
        qty = s.size(equity=10000, price=0, strength=0.8)
        assert qty == 0

    def test_kelly_zero_strength(self):
        s = self._sizer("kelly")
        qty = s.size(equity=10000, price=50000, strength=0.0)
        assert qty == 0.0

    def test_fixed_frac_with_atr(self):
        s = self._sizer("fixed_frac")
        qty = s.size(equity=10000, price=50000, atr=500)
        assert qty > 0

    def test_fixed_frac_no_atr_uses_default(self):
        s = self._sizer("fixed_frac")
        qty = s.size(equity=10000, price=50000, atr=0)
        assert qty > 0  # uses price*1% as stop

    def test_vol_adjusted_high_vol_reduces_size(self):
        s = self._sizer("vol_adjusted")
        low_vol  = s.size(equity=10000, price=50000, realised_vol_pct=10)
        high_vol = s.size(equity=10000, price=50000, realised_vol_pct=50)
        assert high_vol < low_vol

    def test_max_position_cap(self):
        from core.risk.position_sizer import PositionSizer, SizerConfig, SizingMethod
        cfg = SizerConfig(kelly_fraction=10.0, max_position_pct=5.0)  # huge kelly, small cap
        s   = PositionSizer(cfg)
        qty = s.size(equity=10000, price=50000, strength=1.0)
        max_qty = (10000 * 0.05) / 50000
        assert qty <= max_qty + 1e-9

    def test_audit_trail(self):
        s = self._sizer("kelly")
        s.size(equity=10000, price=50000, strength=0.5)
        assert len(s.audit) == 1

    def test_realised_vol_flat_prices(self):
        s = self._sizer()
        prices = [50000.0] * 50
        vol = s.realised_vol(prices)
        assert vol == pytest.approx(0.0, abs=1e-6)

    def test_realised_vol_rising(self):
        s = self._sizer()
        prices = [50000 * (1.01 ** i) for i in range(50)]
        vol = s.realised_vol(prices)
        assert vol > 0


# ---------------------------------------------------------------------------
# MarginWatcher (5)
# ---------------------------------------------------------------------------

class TestMarginWatcher:
    def test_instantiates(self):
        from core.risk.margin_watcher import MarginWatcher
        mw = MarginWatcher()
        assert mw is not None

    def test_check_sync_ok(self):
        from core.risk.margin_watcher import MarginWatcher, MarginConfig
        mw = MarginWatcher(config=MarginConfig(soft_threshold=0.7))
        ratio, level = mw.check_margin_sync(equity=10000, used_margin=5000)
        assert level == "OK"
        assert ratio == pytest.approx(0.5)

    def test_check_sync_soft(self):
        from core.risk.margin_watcher import MarginWatcher, MarginConfig
        mw = MarginWatcher(config=MarginConfig(soft_threshold=0.7))
        ratio, level = mw.check_margin_sync(equity=10000, used_margin=7500)
        assert level == "SOFT"

    def test_check_sync_hard(self):
        from core.risk.margin_watcher import MarginWatcher, MarginConfig
        mw = MarginWatcher(config=MarginConfig(hard_threshold=0.85))
        ratio, level = mw.check_margin_sync(equity=10000, used_margin=9000)
        assert level == "HARD"

    def test_stats_dict(self):
        from core.risk.margin_watcher import MarginWatcher
        mw = MarginWatcher()
        s = mw.stats
        assert "last_ratio" in s and "poll_count" in s
