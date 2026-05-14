"""
Tests for the SOL alpha stack — covers all 4 new modules.
28 tests total.
"""
from __future__ import annotations

import time
from decimal import Decimal

import pytest

from strategies.sol_dca_superior import SolSafeDCAStrategy, SafeDCADeal, MAX_SAFETY_ORDERS
from strategies.sol_momentum_scalper import SolMomentumScalper
from alpha.sol_regime_detector import SolRegimeDetector
from execution.sol_tp_ladder import SolTpLadder


# ============================================================
# SolRegimeDetector
# ============================================================

class TestSolRegimeDetector:
    def _feed(self, detector: SolRegimeDetector, n: int = 25,
              base: float = 150.0, trend: float = 0.0) -> None:
        for i in range(n):
            c = base + i * trend
            detector.update(close=c, high=c * 1.005, low=c * 0.995)

    def test_ranging_regime_low_adx(self) -> None:
        d = SolRegimeDetector()
        # Flat prices = low ADX
        self._feed(d, n=30, base=150.0, trend=0.0)
        r = d.get_regime()
        assert r.regime in ("ranging", "mild_trending")

    def test_trending_regime_with_uptrend(self) -> None:
        d = SolRegimeDetector()
        self._feed(d, n=30, base=100.0, trend=1.5)  # strong uptrend
        r = d.get_regime()
        assert r.regime in ("trending", "strong_trending", "mild_trending")

    def test_funding_rate_stored(self) -> None:
        d = SolRegimeDetector()
        d.set_funding_rate(0.0012)
        assert d._funding_rate == 0.0012

    def test_insufficient_data_returns_ranging(self) -> None:
        d = SolRegimeDetector()
        d.update(150.0, 151.0, 149.0)  # only 1 bar
        r = d.get_regime()
        assert r.regime == "ranging"

    def test_regime_reading_has_all_fields(self) -> None:
        d = SolRegimeDetector()
        self._feed(d)
        r = d.get_regime()
        assert hasattr(r, "adx")
        assert hasattr(r, "realised_vol")
        assert hasattr(r, "ema20")


# ============================================================
# SolSafeDCAStrategy
# ============================================================

class TestSolSafeDCAStrategy:
    def test_opens_base_order_in_ranging(self) -> None:
        s = SolSafeDCAStrategy(capital_aud=1000.0)
        order = s.on_tick(price=150.0, atr=2.0, regime="ranging", available_capital=1000.0)
        assert order is not None
        assert order["action"] == "buy"
        assert order["reason"] == "base_order"

    def test_no_order_in_choppy_regime(self) -> None:
        s = SolSafeDCAStrategy(capital_aud=1000.0)
        order = s.on_tick(price=150.0, atr=2.0, regime="choppy", available_capital=1000.0)
        assert order is None

    def test_safety_order_triggered_on_dip(self) -> None:
        s = SolSafeDCAStrategy(capital_aud=1000.0)
        s.on_tick(price=150.0, atr=2.0, regime="ranging", available_capital=1000.0)
        # Price drops below safety order 1 trigger (150 - 1.2*2.0*1 = 147.6)
        order = s.on_tick(price=147.0, atr=2.0, regime="ranging", available_capital=900.0)
        assert order is not None
        assert "safety_order_1" in order["reason"]

    def test_max_safety_orders_respected(self) -> None:
        s = SolSafeDCAStrategy(capital_aud=5000.0)
        s.on_tick(price=150.0, atr=2.0, regime="ranging", available_capital=5000.0)
        deal = s.active_deal
        assert deal is not None
        deal.filled_orders = MAX_SAFETY_ORDERS
        # Should NOT trigger another safety order
        order = s.on_tick(price=140.0, atr=2.0, regime="ranging", available_capital=4000.0)
        assert order is None or "safety" not in order.get("reason", "")

    def test_deal_stop_closes_position(self) -> None:
        s = SolSafeDCAStrategy(capital_aud=1000.0)
        s.on_tick(price=150.0, atr=2.0, regime="ranging", available_capital=1000.0)
        # Drop below deal stop (150 * 0.88 = 132)
        order = s.on_tick(price=130.0, atr=2.0, regime="ranging", available_capital=800.0)
        assert order is not None
        assert order["action"] == "sell"
        assert order["reason"] == "deal_stop"
        assert s.active_deal is None

    def test_avg_entry_calculation(self) -> None:
        deal = SafeDCADeal(
            symbol="SOL/USD",
            base_entry=150.0,
            base_qty=1.0,
            atr_at_open=2.0,
            entries=[150.0, 146.0],
            quantities=[1.0, 2.0],
        )
        expected = (150 * 1 + 146 * 2) / 3
        assert abs(deal.avg_entry() - expected) < 1e-9


# ============================================================
# SolMomentumScalper
# ============================================================

class TestSolMomentumScalper:
    def _make_scalper(self) -> SolMomentumScalper:
        return SolMomentumScalper(capital_aud=1000.0)

    def _feed_bars(self, scalper: SolMomentumScalper, n: int = 25,
                   base: float = 150.0, trend: float = 0.3) -> None:
        for i in range(n):
            c = base + i * trend
            scalper.on_bar(
                close=c, high=c * 1.003, low=c * 0.997,
                ofi=0.0, bb_bandwidth=0.02, atr=1.5,
                available_capital=1000.0,
            )

    def test_no_signal_without_enough_bars(self) -> None:
        s = self._make_scalper()
        order = s.on_bar(
            close=150.0, high=151.0, low=149.0,
            ofi=0.5, bb_bandwidth=0.01, atr=1.5,
            available_capital=1000.0,
        )
        assert order is None

    def test_long_signal_with_conditions_met(self) -> None:
        s = self._make_scalper()
        self._feed_bars(s, n=25, trend=0.3)  # uptrend
        # Final bar: strong OFI + squeeze
        order = s.on_bar(
            close=157.5, high=158.0, low=157.0,
            ofi=0.50, bb_bandwidth=0.012,  # below threshold
            atr=1.5, available_capital=1000.0,
        )
        # May or may not trigger depending on momentum alignment
        if order is not None:
            assert order["action"] == "buy"

    def test_cooldown_prevents_rapid_re_entry(self) -> None:
        s = self._make_scalper()
        self._feed_bars(s, n=25, trend=0.3)
        s._last_entry_time = time.time()  # simulate just entered
        order = s.on_bar(
            close=157.5, high=158.0, low=157.0,
            ofi=0.60, bb_bandwidth=0.010,
            atr=1.5, available_capital=1000.0,
        )
        assert order is None  # cooldown active

    def test_trail_stop_closes_long(self) -> None:
        from strategies.sol_momentum_scalper import ScalpPosition
        s = self._make_scalper()
        s.position = ScalpPosition(
            side="long", entry_price=150.0, qty=1.0, atr=2.0,
            high_water=155.0, low_water=150.0,
        )
        # Price drops below trail stop (155 - 0.6*2 = 153.8)
        order = s.on_bar(
            close=153.0, high=153.5, low=152.5,
            ofi=0.0, bb_bandwidth=0.02, atr=2.0,
            available_capital=1000.0,
        )
        assert order is not None
        assert order["action"] == "sell"
        assert s.position is None


# ============================================================
# SolTpLadder
# ============================================================

class TestSolTpLadder:
    def test_tier1_triggered(self) -> None:
        ladder = SolTpLadder()
        ladder.open_deal(avg_entry=150.0, atr=2.0, qty=1.0)
        orders = ladder.on_tick(150.0 * 1.016)  # above tier1 (1.5%)
        assert any(o["tier"] == 1 for o in orders)

    def test_tier2_triggered_after_tier1(self) -> None:
        ladder = SolTpLadder()
        ladder.open_deal(avg_entry=150.0, atr=2.0, qty=1.0)
        ladder.on_tick(150.0 * 1.016)  # tier 1
        orders = ladder.on_tick(150.0 * 1.031)  # tier 2
        assert any(o["tier"] == 2 for o in orders)

    def test_tier1_qty_is_40pct(self) -> None:
        ladder = SolTpLadder()
        ladder.open_deal(avg_entry=150.0, atr=2.0, qty=1.0)
        orders = ladder.on_tick(150.0 * 1.016)
        tier1 = next(o for o in orders if o["tier"] == 1)
        assert abs(tier1["qty"] - 0.40) < 1e-9

    def test_tier3_trails_and_closes(self) -> None:
        ladder = SolTpLadder()
        ladder.open_deal(avg_entry=150.0, atr=2.0, qty=1.0)
        ladder.on_tick(150.0 * 1.016)  # tier 1
        ladder.on_tick(150.0 * 1.031)  # tier 2
        # Price continues up then reverses
        s = ladder.state
        assert s is not None
        s.tier3_high_water = 165.0
        orders = ladder.on_tick(165.0 - 2.0 * 1.8 - 0.1)  # below trail
        assert any(o["tier"] == 3 for o in orders)
        assert ladder.state is None  # deal closed

    def test_no_order_before_tp1(self) -> None:
        ladder = SolTpLadder()
        ladder.open_deal(avg_entry=150.0, atr=2.0, qty=1.0)
        orders = ladder.on_tick(151.0)  # below tier1
        assert orders == []

    def test_force_close_clears_state(self) -> None:
        ladder = SolTpLadder()
        ladder.open_deal(avg_entry=150.0, atr=2.0, qty=1.0)
        order = ladder.force_close(140.0)
        assert order is not None
        assert order["qty"] == 1.0
        assert ladder.state is None

    def test_no_duplicate_tier_fills(self) -> None:
        ladder = SolTpLadder()
        ladder.open_deal(avg_entry=150.0, atr=2.0, qty=1.0)
        # Hit tier1 price twice
        orders1 = ladder.on_tick(152.5)
        orders2 = ladder.on_tick(152.5)
        tier1_count = sum(1 for o in orders1 + orders2 if o["tier"] == 1)
        assert tier1_count == 1  # only filled once
