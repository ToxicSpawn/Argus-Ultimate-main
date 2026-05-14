"""Push 97 — Tests for AlertAction bridge (v8.33.0)."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.alerts.alert_action_bridge import (
    ActionRule,
    ActionType,
    AlertActionBridge,
    DEFAULT_ACTION_RULES,
)
from core.alerts.correlation_hedge import (
    CorrelationHedge,
    CorrelationWindow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(
    has_rm: bool = True,
    has_om: bool = True,
) -> MagicMock:
    ctx = MagicMock()
    if has_rm:
        ctx.risk_manager = MagicMock()
        ctx.risk_manager.activate_kill_switch = MagicMock()
    else:
        ctx.risk_manager = None
    if has_om:
        ctx.order_manager = MagicMock()
        ctx.order_manager.stats = {
            "positions": {
                "BTCUSDT": {"side": "LONG", "qty": 0.5, "notional": 25000.0},
            }
        }
        ctx.order_manager.submit_reduce = AsyncMock()
        ctx.order_manager.submit_market = AsyncMock()
    else:
        ctx.order_manager = None
    return ctx


def make_event(title: str) -> MagicMock:
    e = MagicMock()
    e.title = title
    return e


# ---------------------------------------------------------------------------
# ActionRule
# ---------------------------------------------------------------------------

class TestActionRule:
    def test_defaults(self):
        r = ActionRule(alert_name="foo", action_type=ActionType.NOTIFY_ONLY)
        assert r.enabled
        assert r.cooldown_s == 60.0

    def test_wildcard_pattern(self):
        r = ActionRule(alert_name="drawdown*", action_type=ActionType.REDUCE_POSITION)
        assert AlertActionBridge._matches(r.alert_name, "drawdown_pct")
        assert not AlertActionBridge._matches(r.alert_name, "vol_spike")


# ---------------------------------------------------------------------------
# AlertActionBridge
# ---------------------------------------------------------------------------

class TestAlertActionBridge:
    def test_register_rules(self):
        bridge = AlertActionBridge(make_ctx())
        bridge.register_rules(DEFAULT_ACTION_RULES)
        assert len(bridge.rules) == len(DEFAULT_ACTION_RULES)

    def test_add_remove_rule(self):
        bridge = AlertActionBridge(make_ctx())
        rule = ActionRule(alert_name="test_rule", action_type=ActionType.NOTIFY_ONLY)
        bridge.add_rule(rule)
        assert any(r.alert_name == "test_rule" for r in bridge.rules)
        removed = bridge.remove_rule("test_rule")
        assert removed
        assert not any(r.alert_name == "test_rule" for r in bridge.rules)

    def test_remove_nonexistent_returns_false(self):
        bridge = AlertActionBridge(make_ctx())
        assert not bridge.remove_rule("nonexistent")

    @pytest.mark.asyncio
    async def test_on_alert_notify_only(self):
        bridge = AlertActionBridge(make_ctx())
        rule = ActionRule(
            alert_name="test_notify",
            action_type=ActionType.NOTIFY_ONLY,
            cooldown_s=0.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("test_notify"))
        await asyncio.sleep(0.05)  # let task complete
        assert bridge.stats["total_triggered"] == 1

    @pytest.mark.asyncio
    async def test_on_alert_kill_switch(self):
        ctx = make_ctx()
        bridge = AlertActionBridge(ctx)
        rule = ActionRule(
            alert_name="kill_switch_auto_pct",
            action_type=ActionType.KILL_SWITCH_AUTO,
            cooldown_s=0.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("kill_switch_auto_pct"))
        await asyncio.sleep(0.05)
        ctx.risk_manager.activate_kill_switch.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_alert_reduce_position(self):
        ctx = make_ctx()
        bridge = AlertActionBridge(ctx)
        rule = ActionRule(
            alert_name="drawdown_pct",
            action_type=ActionType.REDUCE_POSITION,
            reduce_pct=0.5,
            cooldown_s=0.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("drawdown_pct"))
        await asyncio.sleep(0.05)
        assert bridge.stats["total_triggered"] == 1

    @pytest.mark.asyncio
    async def test_cooldown_blocks_repeated_trigger(self):
        bridge = AlertActionBridge(make_ctx())
        rule = ActionRule(
            alert_name="test_cool",
            action_type=ActionType.NOTIFY_ONLY,
            cooldown_s=9999.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("test_cool"))
        await bridge.on_alert(make_event("test_cool"))
        await asyncio.sleep(0.05)
        assert bridge.stats["total_triggered"] == 1
        assert bridge.stats["total_blocked"] == 1

    @pytest.mark.asyncio
    async def test_no_rm_kill_switch_fails_gracefully(self):
        ctx = make_ctx(has_rm=False)
        bridge = AlertActionBridge(ctx)
        rule = ActionRule(
            alert_name="kill_switch_auto_pct",
            action_type=ActionType.KILL_SWITCH_AUTO,
            cooldown_s=0.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("kill_switch_auto_pct"))
        await asyncio.sleep(0.05)
        result = bridge.stats["last_actions"][-1]
        assert not result["success"]

    @pytest.mark.asyncio
    async def test_no_om_reduce_fails_gracefully(self):
        ctx = make_ctx(has_om=False)
        bridge = AlertActionBridge(ctx)
        rule = ActionRule(
            alert_name="drawdown_pct",
            action_type=ActionType.REDUCE_POSITION,
            cooldown_s=0.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("drawdown_pct"))
        await asyncio.sleep(0.05)
        result = bridge.stats["last_actions"][-1]
        assert not result["success"]

    @pytest.mark.asyncio
    async def test_no_positions_reduce_succeeds_gracefully(self):
        ctx = make_ctx()
        ctx.order_manager.stats = {"positions": {}}
        bridge = AlertActionBridge(ctx)
        rule = ActionRule(
            alert_name="drawdown_pct",
            action_type=ActionType.REDUCE_POSITION,
            cooldown_s=0.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("drawdown_pct"))
        await asyncio.sleep(0.05)
        result = bridge.stats["last_actions"][-1]
        assert result["success"]

    def test_stats_structure(self):
        bridge = AlertActionBridge(make_ctx())
        s = bridge.stats
        assert "rules" in s
        assert "total_triggered" in s
        assert "history_len" in s

    @pytest.mark.asyncio
    async def test_hedge_delta_action(self):
        ctx = make_ctx()
        bridge = AlertActionBridge(ctx)
        rule = ActionRule(
            alert_name="vol_spike_ratio",
            action_type=ActionType.HEDGE_DELTA,
            hedge_ratio=1.0,
            cooldown_s=0.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("vol_spike_ratio"))
        await asyncio.sleep(0.05)
        assert bridge.stats["total_triggered"] == 1

    @pytest.mark.asyncio
    async def test_unmatched_alert_not_triggered(self):
        bridge = AlertActionBridge(make_ctx())
        rule = ActionRule(
            alert_name="specific_rule",
            action_type=ActionType.NOTIFY_ONLY,
            cooldown_s=0.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("different_alert"))
        await asyncio.sleep(0.05)
        assert bridge.stats["total_triggered"] == 0

    @pytest.mark.asyncio
    async def test_disabled_rule_not_triggered(self):
        bridge = AlertActionBridge(make_ctx())
        rule = ActionRule(
            alert_name="test_disabled",
            action_type=ActionType.NOTIFY_ONLY,
            enabled=False,
            cooldown_s=0.0,
        )
        bridge.add_rule(rule)
        await bridge.on_alert(make_event("test_disabled"))
        await asyncio.sleep(0.05)
        assert bridge.stats["total_triggered"] == 0


# ---------------------------------------------------------------------------
# CorrelationWindow
# ---------------------------------------------------------------------------

class TestCorrelationWindow:
    def test_returns_none_before_min_samples(self):
        w = CorrelationWindow(window=60)
        for _ in range(5):
            w.update(0.01, 0.01)
        assert w.correlation() is None

    def test_perfect_correlation(self):
        w = CorrelationWindow(window=60)
        for i in range(20):
            w.update(float(i), float(i))
        corr = w.correlation()
        assert corr is not None
        assert abs(corr - 1.0) < 1e-6

    def test_negative_correlation(self):
        w = CorrelationWindow(window=60)
        for i in range(20):
            w.update(float(i), float(-i))
        corr = w.correlation()
        assert corr is not None
        assert corr < -0.99

    def test_zero_variance_returns_none(self):
        w = CorrelationWindow(window=60)
        for _ in range(20):
            w.update(1.0, 1.0)   # constant → zero variance
        # correlation() returns None for zero-variance series
        assert w.correlation() is None


# ---------------------------------------------------------------------------
# CorrelationHedge
# ---------------------------------------------------------------------------

class TestCorrelationHedge:
    def test_no_positions_no_events(self):
        ctx = MagicMock()
        ctx.order_manager = MagicMock()
        ctx.order_manager.stats = {"positions": {}}
        hedge = CorrelationHedge(ctx, threshold=0.8)
        result = asyncio.get_event_loop().run_until_complete(
            hedge.check_and_hedge()
        )
        assert result == []

    def test_stats_structure(self):
        hedge = CorrelationHedge(MagicMock(), threshold=0.8)
        s = hedge.stats
        assert "tracked_pairs" in s
        assert "hedge_events" in s
