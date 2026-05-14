"""
tests_unified/test_contingency_orders.py
=========================================
Unit tests for:
  - ConditionalOrderManager  (conditional_orders.py)
  - ContingencyExecutor      (contingency_orders.py)
  - AdvancedOrderSpec flags  (order_types_advanced.py)

All tests run fully offline (no exchange connectivity required).
"""
from __future__ import annotations

import asyncio
import pytest

from execution.conditional_orders import (
    ConditionalOrderManager,
    GroupStatus,
    GroupType,
    LegStatus,
    LegType,
)
from execution.contingency_orders import ContingencyExecutor, OTOStatus
from execution.order_types_advanced import (
    AdvancedOrderSpec,
    TimeInForce,
    normalize_to_venue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# ConditionalOrderManager — OCO
# ---------------------------------------------------------------------------

class TestOCO:
    def setup_method(self):
        self.mgr = ConditionalOrderManager(connector=None)

    def test_create_oco_returns_group_id(self):
        gid = self.mgr.create_oco(
            symbol="BTC/USDT", tp_price=66000, sl_price=63000,
            quantity=0.1, exchange="kraken",
        )
        assert gid in self.mgr._groups

    def test_oco_has_two_legs(self):
        gid = self.mgr.create_oco(
            symbol="BTC/USDT", tp_price=66000, sl_price=63000,
            quantity=0.1, exchange="kraken",
        )
        group = self.mgr._groups[gid]
        assert len(group.legs) == 2
        types = {leg.leg_type for leg in group.legs}
        assert LegType.TP in types and LegType.SL in types

    def test_oco_tp_fill_cancels_sl(self):
        gid = self.mgr.create_oco(
            symbol="BTC/USDT", tp_price=66000, sl_price=63000,
            quantity=0.1, exchange="kraken",
        )
        group = self.mgr._groups[gid]
        tp_leg = next(l for l in group.legs if l.leg_type == LegType.TP)
        sl_leg = next(l for l in group.legs if l.leg_type == LegType.SL)
        self.mgr.on_fill(tp_leg.order_id, filled_qty=0.1, fill_price=66010)
        assert tp_leg.status == LegStatus.FILLED
        assert sl_leg.status == LegStatus.CANCELLED
        assert group.status == GroupStatus.FILLED

    def test_oco_sl_fill_cancels_tp(self):
        gid = self.mgr.create_oco(
            symbol="BTC/USDT", tp_price=66000, sl_price=63000,
            quantity=0.1, exchange="kraken",
        )
        group = self.mgr._groups[gid]
        tp_leg = next(l for l in group.legs if l.leg_type == LegType.TP)
        sl_leg = next(l for l in group.legs if l.leg_type == LegType.SL)
        self.mgr.on_fill(sl_leg.order_id, filled_qty=0.1, fill_price=62990)
        assert sl_leg.status == LegStatus.FILLED
        assert tp_leg.status == LegStatus.CANCELLED
        assert group.status == GroupStatus.FILLED

    def test_cancel_group(self):
        gid = self.mgr.create_oco(
            symbol="BTC/USDT", tp_price=66000, sl_price=63000,
            quantity=0.1, exchange="kraken",
        )
        self.mgr.cancel_group(gid)
        group = self.mgr._groups[gid]
        assert group.status == GroupStatus.CANCELLED
        for leg in group.legs:
            assert leg.status == LegStatus.CANCELLED


# ---------------------------------------------------------------------------
# ConditionalOrderManager — Bracket
# ---------------------------------------------------------------------------

class TestBracket:
    def setup_method(self):
        self.mgr = ConditionalOrderManager(connector=None)

    def test_create_bracket_three_legs(self):
        gid = self.mgr.create_bracket(
            symbol="ETH/USDT", entry_price=3000, tp_price=3300,
            sl_price=2850, quantity=1.0, exchange="bybit",
        )
        group = self.mgr._groups[gid]
        assert len(group.legs) == 3
        assert group.group_type == GroupType.BRACKET

    def test_bracket_entry_fill_moves_to_partial(self):
        gid = self.mgr.create_bracket(
            symbol="ETH/USDT", entry_price=3000, tp_price=3300,
            sl_price=2850, quantity=1.0, exchange="bybit",
        )
        group = self.mgr._groups[gid]
        entry_leg = next(l for l in group.legs if l.leg_type == LegType.ENTRY)
        self.mgr.on_fill(entry_leg.order_id, filled_qty=1.0, fill_price=3005)
        assert group.status == GroupStatus.PARTIAL

    def test_bracket_tp_fill_after_entry(self):
        gid = self.mgr.create_bracket(
            symbol="ETH/USDT", entry_price=3000, tp_price=3300,
            sl_price=2850, quantity=1.0, exchange="bybit",
        )
        group = self.mgr._groups[gid]
        entry_leg = next(l for l in group.legs if l.leg_type == LegType.ENTRY)
        tp_leg    = next(l for l in group.legs if l.leg_type == LegType.TP)
        sl_leg    = next(l for l in group.legs if l.leg_type == LegType.SL)
        self.mgr.on_fill(entry_leg.order_id, filled_qty=1.0, fill_price=3005)
        self.mgr.on_fill(tp_leg.order_id,    filled_qty=1.0, fill_price=3300)
        assert tp_leg.status == LegStatus.FILLED
        assert sl_leg.status == LegStatus.CANCELLED
        assert group.status == GroupStatus.FILLED


# ---------------------------------------------------------------------------
# ConditionalOrderManager — Trailing Stop
# ---------------------------------------------------------------------------

class TestTrailingStop:
    def setup_method(self):
        self.mgr = ConditionalOrderManager(connector=None)

    def test_trailing_stop_advances_with_price(self):
        gid = self.mgr.create_trailing_stop(
            symbol="BTC/USDT", trail_pct=0.02, quantity=0.1,
            exchange="kraken", initial_price=65000,
        )
        group = self.mgr._groups[gid]
        old_stop = group.__dict__["_stop_price"]
        self.mgr.on_price_update("BTC/USDT", 67000)
        new_stop = group.__dict__["_stop_price"]
        assert new_stop > old_stop

    def test_trailing_stop_does_not_retreat(self):
        gid = self.mgr.create_trailing_stop(
            symbol="BTC/USDT", trail_pct=0.02, quantity=0.1,
            exchange="kraken", initial_price=65000,
        )
        group = self.mgr._groups[gid]
        self.mgr.on_price_update("BTC/USDT", 67000)
        stop_after_advance = group.__dict__["_stop_price"]
        self.mgr.on_price_update("BTC/USDT", 64000)
        assert group.__dict__["_stop_price"] == stop_after_advance

    def test_trailing_stop_triggers(self):
        gid = self.mgr.create_trailing_stop(
            symbol="BTC/USDT", trail_pct=0.02, quantity=0.1,
            exchange="kraken", initial_price=65000,
        )
        group = self.mgr._groups[gid]
        # Price falls through stop (65000 * 0.98 = 63700)
        self.mgr.on_price_update("BTC/USDT", 63000)
        assert group.status == GroupStatus.FILLED


# ---------------------------------------------------------------------------
# ContingencyExecutor — simulation mode
# ---------------------------------------------------------------------------

class TestContingencyExecutorSim:
    def setup_method(self):
        self.ex = ContingencyExecutor(connector=None)

    def test_submit_oco_returns_group_id(self):
        gid = run(self.ex.submit_oco(
            symbol="BTC/USDT", tp_price=66000, sl_price=63000,
            quantity=0.1, exchange="kraken",
        ))
        assert gid

    def test_submit_bracket_returns_group_id(self):
        gid = run(self.ex.submit_bracket(
            symbol="ETH/USDT", entry_price=3000, tp_price=3300,
            sl_price=2850, quantity=1.0, exchange="bybit",
        ))
        assert gid

    def test_submit_trailing_stop_returns_group_id(self):
        gid = run(self.ex.submit_trailing_stop(
            symbol="BTC/USDT", trail_pct=0.02, quantity=0.1,
            exchange="kraken", initial_price=65000,
        ))
        assert gid

    def test_submit_oto_returns_group_id(self):
        gid = run(self.ex.submit_oto(
            symbol="SOL/USDT",
            trigger_price=150.0,
            trigger_side="buy",
            trigger_qty=10.0,
            dependent_specs=[
                {"type": "limit", "side": "sell", "price": 160.0, "amount": 10.0},
                {"type": "stop",  "side": "sell", "price": 145.0, "amount": 10.0},
            ],
            exchange="bybit",
        ))
        assert gid
        assert gid in self.ex._oto_groups

    def test_on_fill_bracket_activates_oco(self):
        gid = run(self.ex.submit_bracket(
            symbol="ETH/USDT", entry_price=3000, tp_price=3300,
            sl_price=2850, quantity=1.0, exchange="bybit",
        ))
        group = self.ex._cond_mgr._groups[gid]
        entry_leg = next(l for l in group.legs if l.leg_type == LegType.ENTRY)
        # Simulate exchange fill using synthetic ID
        run(self.ex.on_fill(
            order_id=entry_leg.order_id,
            filled_qty=1.0,
            fill_price=3005.0,
        ))
        assert group.status == GroupStatus.PARTIAL

    def test_on_fill_oto_activates_dependents(self):
        gid = run(self.ex.submit_oto(
            symbol="SOL/USDT",
            trigger_price=150.0,
            trigger_side="buy",
            trigger_qty=10.0,
            dependent_specs=[
                {"type": "limit", "side": "sell", "price": 160.0, "amount": 10.0},
            ],
            exchange="bybit",
        ))
        oto = self.ex._oto_groups[gid]
        run(self.ex.on_fill(
            order_id=oto.trigger_order_id,
            filled_qty=10.0,
            fill_price=151.0,
        ))
        assert oto.status == OTOStatus.ACTIVE
        assert len(oto.submitted_dependent_ids) == 1

    def test_cancel_group(self):
        gid = run(self.ex.submit_oco(
            symbol="BTC/USDT", tp_price=66000, sl_price=63000,
            quantity=0.1, exchange="kraken",
        ))
        run(self.ex.cancel_group(gid))
        group = self.ex._cond_mgr._groups[gid]
        assert group.status == GroupStatus.CANCELLED

    def test_price_update_advances_trailing_stop(self):
        gid = run(self.ex.submit_trailing_stop(
            symbol="BTC/USDT", trail_pct=0.02, quantity=0.1,
            exchange="kraken", initial_price=65000,
        ))
        group = self.ex._cond_mgr._groups[gid]
        old_stop = group.__dict__["_stop_price"]
        run(self.ex.on_price_update("BTC/USDT", 68000))
        new_stop = group.__dict__["_stop_price"]
        assert new_stop > old_stop

    def test_snapshot_contains_oto_section(self):
        run(self.ex.submit_oto(
            symbol="SOL/USDT", trigger_price=150.0, trigger_side="buy",
            trigger_qty=10.0, dependent_specs=[], exchange="bybit",
        ))
        snap = self.ex.snapshot()
        assert "oto" in snap
        assert len(snap["oto"]) == 1


# ---------------------------------------------------------------------------
# AdvancedOrderSpec — post_only / reduce_only / GTD flags
# ---------------------------------------------------------------------------

class TestAdvancedOrderSpec:
    def test_post_only_bybit(self):
        spec = AdvancedOrderSpec(
            symbol="BTC/USDT", side="buy", size=0.01,
            order_type="limit", price=65000, post_only=True,
        )
        payload = normalize_to_venue(spec, venue="bybit", mid_price=65000)
        assert payload.get("postOnly") is True

    def test_post_only_generic_venue(self):
        spec = AdvancedOrderSpec(
            symbol="BTC/USDT", side="buy", size=0.01,
            order_type="limit", price=65000, post_only=True,
        )
        payload = normalize_to_venue(spec, venue="other", mid_price=65000)
        assert payload.get("post_only") is True

    def test_reduce_only_bybit(self):
        spec = AdvancedOrderSpec(
            symbol="BTC/USDT", side="sell", size=0.01,
            order_type="limit", price=66000, reduce_only=True,
        )
        payload = normalize_to_venue(spec, venue="bybit", mid_price=66000)
        assert payload.get("reduceOnly") is True

    def test_gtd_time_in_force(self):
        spec = AdvancedOrderSpec(
            symbol="ETH/USDT", side="buy", size=1.0,
            order_type="limit", price=3000,
            time_in_force=TimeInForce.GTD,
            expire_time="2026-05-01T00:00:00Z",
        )
        payload = normalize_to_venue(spec, venue="kraken", mid_price=3000)
        assert payload["time_in_force"] == "GTD"
        assert payload["expire_time"] == "2026-05-01T00:00:00Z"

    def test_fok_market_order_skips_tif(self):
        spec = AdvancedOrderSpec(
            symbol="BTC/USDT", side="buy", size=0.01,
            order_type="market",
            time_in_force=TimeInForce.FOK,
        )
        payload = normalize_to_venue(spec, venue="bybit", mid_price=65000)
        assert payload["type"] == "market"
        assert "time_in_force" not in payload

    def test_ioc_limit(self):
        spec = AdvancedOrderSpec(
            symbol="SOL/USDT", side="sell", size=10.0,
            order_type="limit", price=160.0,
            time_in_force=TimeInForce.IOC,
        )
        payload = normalize_to_venue(spec, venue="bybit", mid_price=160.0)
        assert payload["time_in_force"] == "IOC"

    def test_pegged_buy_price_below_mid(self):
        spec = AdvancedOrderSpec(
            symbol="BTC/USDT", side="buy", size=0.01,
            order_type="limit", pegged=True, peg_offset_bps=-10,
        )
        payload = normalize_to_venue(spec, venue="bybit", mid_price=65000)
        # -10 bps below mid = 65000 * (1 - 10/10000) = 64935
        assert payload["price"] == pytest.approx(64935.0)

    def test_hidden_flag_passed_through(self):
        spec = AdvancedOrderSpec(
            symbol="BTC/USDT", side="buy", size=0.01,
            order_type="limit", price=65000, hidden=True,
        )
        payload = normalize_to_venue(spec, venue="kraken", mid_price=65000)
        assert payload.get("hidden") is True
