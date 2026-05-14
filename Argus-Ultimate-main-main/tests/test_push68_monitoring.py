"""Push 68 — Tests: ArgusMetrics, AlertRuleEngine, AlertRule,
TelegramNotifier (dry-run), DiscordNotifier (dry-run),
AlertManager. 26 tests.
"""
from __future__ import annotations
import asyncio
import sys
import time
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# ArgusMetrics (4)
# ---------------------------------------------------------------------------

class TestArgusMetrics:
    def test_instantiates_without_prometheus(self):
        from core.monitoring.metrics import ArgusMetrics
        m = ArgusMetrics()
        assert m is not None

    def test_update_risk_snapshot_no_error(self):
        from core.monitoring.metrics import ArgusMetrics
        m = ArgusMetrics()
        m.update_risk_snapshot(
            halted=False, drawdown_pct=1.5, cvar_95=0.02,
            cvar_99=0.04, daily_pnl=250.0, equity=10250.0,
            position_count=2
        )

    def test_record_fill_no_error(self):
        from core.monitoring.metrics import ArgusMetrics
        m = ArgusMetrics()
        m.record_fill(latency_ms=12.5, algorithm="PPO", side="buy")

    def test_update_kelly_no_error(self):
        from core.monitoring.metrics import ArgusMetrics
        m = ArgusMetrics()
        m.update_kelly(0.15)


# ---------------------------------------------------------------------------
# AlertRule (5)
# ---------------------------------------------------------------------------

class TestAlertRule:
    def test_fires_when_gt_threshold(self):
        from core.monitoring.alert_rules import AlertRule, Severity
        val = [0.0]
        rule = AlertRule(
            name="test", metric_fn=lambda: val[0],
            threshold=0.5, comparator="gt",
            severity=Severity.WARN, cooldown_secs=0.0
        )
        val[0] = 0.8
        event = rule.evaluate()
        assert event is not None
        assert event.rule_name == "test"

    def test_does_not_fire_below_threshold(self):
        from core.monitoring.alert_rules import AlertRule
        rule = AlertRule(
            name="t", metric_fn=lambda: 0.2,
            threshold=0.5, comparator="gt", cooldown_secs=0.0
        )
        assert rule.evaluate() is None

    def test_cooldown_prevents_repeat_fire(self):
        from core.monitoring.alert_rules import AlertRule
        rule = AlertRule(
            name="t", metric_fn=lambda: 1.0,
            threshold=0.5, comparator="gt",
            cooldown_secs=9999.0
        )
        rule.evaluate()  # first fire
        rule.last_fired = time.time()  # force cooldown active
        assert rule.evaluate() is None

    def test_lt_comparator(self):
        from core.monitoring.alert_rules import AlertRule
        rule = AlertRule(
            name="t", metric_fn=lambda: -600.0,
            threshold=-500.0, comparator="lt", cooldown_secs=0.0
        )
        assert rule.evaluate() is not None

    def test_disabled_rule_never_fires(self):
        from core.monitoring.alert_rules import AlertRule
        rule = AlertRule(
            name="t", metric_fn=lambda: 9999.0,
            threshold=0.0, comparator="gt",
            cooldown_secs=0.0, enabled=False
        )
        assert rule.evaluate() is None


# ---------------------------------------------------------------------------
# AlertRuleEngine (3)
# ---------------------------------------------------------------------------

class TestAlertRuleEngine:
    def test_evaluate_all_returns_list(self):
        from core.monitoring.alert_rules import AlertRuleEngine, AlertRule
        eng = AlertRuleEngine()
        eng.add_rule(AlertRule(
            name="r1", metric_fn=lambda: 1.0,
            threshold=0.5, comparator="gt", cooldown_secs=0.0
        ))
        result = eng.evaluate_all()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_disable_rule(self):
        from core.monitoring.alert_rules import AlertRuleEngine, AlertRule
        eng = AlertRuleEngine()
        eng.add_rule(AlertRule(
            name="r1", metric_fn=lambda: 1.0,
            threshold=0.0, comparator="gt", cooldown_secs=0.0
        ))
        eng.disable("r1")
        assert eng.evaluate_all() == []

    def test_default_rules_build(self):
        from core.monitoring.alert_rules import AlertRuleEngine
        rules = AlertRuleEngine.default_rules(
            get_halted=lambda: 0,
            get_drawdown=lambda: 0.0,
            get_cvar_95=lambda: 0.0,
            get_cvar_99=lambda: 0.0,
            get_daily_pnl=lambda: 0.0,
            get_fill_latency_p99=lambda: 0.0,
        )
        assert len(rules) == 7


# ---------------------------------------------------------------------------
# TelegramNotifier dry-run (4)
# ---------------------------------------------------------------------------

class TestTelegramNotifier:
    def _make_event(self):
        from core.monitoring.alert_rules import AlertEvent, Severity
        return AlertEvent(
            rule_name="test", severity=Severity.WARN,
            message="Test alert", value=0.8, threshold=0.5
        )

    def test_dry_run_send_returns_true(self):
        from core.monitoring.telegram_notifier import TelegramNotifier, TelegramConfig
        n = TelegramNotifier(TelegramConfig(dry_run=True))
        async def run():
            return await n.send_alert(self._make_event())
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is True

    def test_sent_count_increments(self):
        from core.monitoring.telegram_notifier import TelegramNotifier, TelegramConfig
        n = TelegramNotifier(TelegramConfig(dry_run=True))
        async def run():
            await n.send_alert(self._make_event())
            await n.send_alert(self._make_event())
        asyncio.get_event_loop().run_until_complete(run())
        assert n.total_sent == 2

    def test_rate_limit_blocks_excess(self):
        from core.monitoring.telegram_notifier import TelegramNotifier, TelegramConfig
        n = TelegramNotifier(TelegramConfig(dry_run=True, max_per_minute=2))
        async def run():
            for _ in range(5):
                await n.send_text("x")
        asyncio.get_event_loop().run_until_complete(run())
        assert n.total_sent == 2

    def test_no_token_dry_run_passes(self):
        from core.monitoring.telegram_notifier import TelegramNotifier, TelegramConfig
        n = TelegramNotifier(TelegramConfig(bot_token="", dry_run=True))
        async def run():
            return await n.send_text("hello")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is True


# ---------------------------------------------------------------------------
# DiscordNotifier dry-run (4)
# ---------------------------------------------------------------------------

class TestDiscordNotifier:
    def _make_event(self):
        from core.monitoring.alert_rules import AlertEvent, Severity
        return AlertEvent(
            rule_name="test", severity=Severity.CRITICAL,
            message="Critical alert", value=0.12, threshold=0.10
        )

    def test_dry_run_send_returns_true(self):
        from core.monitoring.discord_notifier import DiscordNotifier, DiscordConfig
        n = DiscordNotifier(DiscordConfig(dry_run=True))
        async def run():
            return await n.send_alert(self._make_event())
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is True

    def test_sent_count_increments(self):
        from core.monitoring.discord_notifier import DiscordNotifier, DiscordConfig
        n = DiscordNotifier(DiscordConfig(dry_run=True))
        async def run():
            await n.send_alert(self._make_event())
        asyncio.get_event_loop().run_until_complete(run())
        assert n.total_sent == 1

    def test_embed_has_correct_color(self):
        from core.monitoring.discord_notifier import DiscordNotifier, DiscordConfig, _SEVERITY_COLOR
        from core.monitoring.alert_rules import Severity
        n = DiscordNotifier(DiscordConfig(dry_run=True))
        event = self._make_event()
        payload = n._build_embed(event)
        assert payload["embeds"][0]["color"] == _SEVERITY_COLOR[Severity.CRITICAL]

    def test_rate_limit_blocks_excess(self):
        from core.monitoring.discord_notifier import DiscordNotifier, DiscordConfig
        n = DiscordNotifier(DiscordConfig(dry_run=True, max_per_minute=1))
        async def run():
            await n.send_alert(self._make_event())
            result = await n.send_alert(self._make_event())
            return result
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is False


# ---------------------------------------------------------------------------
# AlertManager (6)
# ---------------------------------------------------------------------------

class TestAlertManager:
    def test_instantiates(self):
        from core.monitoring.alert_manager import AlertManager
        mgr = AlertManager()
        assert mgr is not None

    def test_add_rule_and_evaluate(self):
        from core.monitoring.alert_manager import AlertManager, AlertManagerConfig
        from core.monitoring.alert_rules import AlertRule, Severity
        from core.monitoring.telegram_notifier import TelegramConfig
        from core.monitoring.discord_notifier import DiscordConfig
        cfg = AlertManagerConfig(
            telegram=TelegramConfig(dry_run=True),
            discord=DiscordConfig(dry_run=True),
            min_dispatch_severity=Severity.INFO,
        )
        mgr = AlertManager(cfg)
        mgr.add_rule(AlertRule(
            name="r", metric_fn=lambda: 1.0,
            threshold=0.5, comparator="gt",
            severity=Severity.WARN, cooldown_secs=0.0
        ))
        async def run():
            return await mgr.evaluate()
        fired = asyncio.get_event_loop().run_until_complete(run())
        assert len(fired) == 1

    def test_dispatched_count_increments(self):
        from core.monitoring.alert_manager import AlertManager, AlertManagerConfig
        from core.monitoring.alert_rules import AlertRule, Severity
        from core.monitoring.telegram_notifier import TelegramConfig
        from core.monitoring.discord_notifier import DiscordConfig
        cfg = AlertManagerConfig(
            telegram=TelegramConfig(dry_run=True),
            discord=DiscordConfig(dry_run=True),
            min_dispatch_severity=Severity.INFO,
        )
        mgr = AlertManager(cfg)
        mgr.add_rule(AlertRule(
            name="r", metric_fn=lambda: 1.0,
            threshold=0.5, comparator="gt",
            severity=Severity.WARN, cooldown_secs=0.0
        ))
        async def run():
            await mgr.evaluate()
        asyncio.get_event_loop().run_until_complete(run())
        assert mgr.dispatched_count == 1

    def test_severity_filter_blocks_info(self):
        from core.monitoring.alert_manager import AlertManager, AlertManagerConfig
        from core.monitoring.alert_rules import AlertRule, Severity
        from core.monitoring.telegram_notifier import TelegramConfig
        from core.monitoring.discord_notifier import DiscordConfig
        cfg = AlertManagerConfig(
            telegram=TelegramConfig(dry_run=True),
            discord=DiscordConfig(dry_run=True),
            min_dispatch_severity=Severity.WARN,  # INFO blocked
        )
        mgr = AlertManager(cfg)
        mgr.add_rule(AlertRule(
            name="r", metric_fn=lambda: 1.0,
            threshold=0.5, comparator="gt",
            severity=Severity.INFO, cooldown_secs=0.0
        ))
        async def run():
            return await mgr.evaluate()
        asyncio.get_event_loop().run_until_complete(run())
        assert mgr.dispatched_count == 0

    def test_starts_and_stops(self):
        from core.monitoring.alert_manager import AlertManager
        mgr = AlertManager()
        async def run():
            await mgr.start()
            await asyncio.sleep(0.05)
            await mgr.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert not mgr.is_running

    def test_eval_count_increments_in_loop(self):
        from core.monitoring.alert_manager import AlertManager, AlertManagerConfig
        cfg = AlertManagerConfig(evaluate_interval_secs=0.05)
        mgr = AlertManager(cfg)
        async def run():
            await mgr.start()
            await asyncio.sleep(0.22)
            await mgr.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert mgr.eval_count >= 2
