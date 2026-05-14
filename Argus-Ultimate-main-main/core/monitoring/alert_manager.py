"""Push 68 — Unified AlertManager: metrics -> rules -> notifications.

Wires together:
  ArgusMetrics  (Prometheus)
  AlertRuleEngine (threshold evaluation)
  TelegramNotifier + DiscordNotifier (dispatch)

Run evaluate() on every bar or every N seconds.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

from core.monitoring.alert_rules import AlertEvent, AlertRule, AlertRuleEngine, Severity
from core.monitoring.telegram_notifier import TelegramNotifier, TelegramConfig
from core.monitoring.discord_notifier import DiscordNotifier, DiscordConfig
from core.monitoring.metrics import ArgusMetrics


@dataclass
class AlertManagerConfig:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    # Minimum severity to dispatch (filter INFO in prod)
    min_dispatch_severity: Severity = Severity.WARN
    evaluate_interval_secs: float = 10.0


class AlertManager:
    """Unified alert orchestrator for Argus.

    Usage:
        mgr = AlertManager(config, metrics)
        mgr.add_rules(AlertRuleEngine.default_rules(...))
        await mgr.start()   # starts background evaluation loop
        await mgr.stop()
    """

    def __init__(
        self,
        config: AlertManagerConfig | None = None,
        metrics: ArgusMetrics | None = None,
    ):
        self.cfg = config or AlertManagerConfig()
        self.metrics = metrics or ArgusMetrics()
        self._rule_engine = AlertRuleEngine()
        self._telegram = TelegramNotifier(self.cfg.telegram)
        self._discord = DiscordNotifier(self.cfg.discord)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._dispatched: List[AlertEvent] = []
        self._eval_count: int = 0

    def add_rules(self, rules: List[AlertRule]) -> None:
        for r in rules:
            self._rule_engine.add_rule(r)

    def add_rule(self, rule: AlertRule) -> None:
        self._rule_engine.add_rule(rule)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._eval_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def evaluate(self) -> List[AlertEvent]:
        """Evaluate all rules and dispatch fired alerts."""
        self._eval_count += 1
        fired = self._rule_engine.evaluate_all()
        for event in fired:
            if self._should_dispatch(event):
                await self._dispatch(event)
        return fired

    async def _eval_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.cfg.evaluate_interval_secs)
            if not self._running:
                break
            try:
                await self.evaluate()
            except Exception:
                pass

    async def _dispatch(self, event: AlertEvent) -> None:
        self._dispatched.append(event)
        tasks = []
        if self.cfg.telegram.bot_token or self.cfg.telegram.dry_run:
            tasks.append(self._telegram.send_alert(event))
        if self.cfg.discord.webhook_url or self.cfg.discord.dry_run:
            tasks.append(self._discord.send_alert(event))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _should_dispatch(self, event: AlertEvent) -> bool:
        severity_order = [
            Severity.INFO, Severity.WARN,
            Severity.CRITICAL, Severity.EMERGENCY
        ]
        min_idx = severity_order.index(self.cfg.min_dispatch_severity)
        evt_idx = severity_order.index(event.severity)
        return evt_idx >= min_idx

    @property
    def dispatched_count(self) -> int:
        return len(self._dispatched)

    @property
    def eval_count(self) -> int:
        return self._eval_count

    @property
    def is_running(self) -> bool:
        return self._running
