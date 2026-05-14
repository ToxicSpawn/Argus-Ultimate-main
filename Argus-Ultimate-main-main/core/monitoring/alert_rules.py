"""Push 68 — Alert rule engine for Argus.

Defines named threshold rules that fire when metric values
cross configured thresholds, with cooldown to prevent spam.

Severity levels:
  INFO      — informational, no action needed
  WARN      — attention recommended
  CRITICAL  — immediate action required
  EMERGENCY — system halt recommended
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class Severity(str, Enum):
    INFO      = "INFO"
    WARN      = "WARN"
    CRITICAL  = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class AlertEvent:
    rule_name: str
    severity: Severity
    message: str
    value: float
    threshold: float
    fired_at: float = field(default_factory=time.time)


@dataclass
class AlertRule:
    """A single named threshold alert rule.

    Args:
        name:           Unique rule identifier
        metric_fn:      Callable returning current metric value (float)
        threshold:      Trigger threshold
        comparator:     "gt" | "lt" | "gte" | "lte" | "eq"
        severity:       Alert severity
        message_tmpl:   f-string template: {value}, {threshold} available
        cooldown_secs:  Minimum seconds between repeated alerts
        enabled:        Can be toggled at runtime
    """
    name: str
    metric_fn: Callable[[], float]
    threshold: float
    comparator: str = "gt"           # gt | lt | gte | lte | eq
    severity: Severity = Severity.WARN
    message_tmpl: str = "{name} breached {threshold} (current: {value:.4f})"
    cooldown_secs: float = 300.0     # 5 minutes default
    enabled: bool = True

    # Runtime state
    last_fired: float = field(default=0.0, repr=False)
    fire_count: int = field(default=0, repr=False)

    def evaluate(self) -> Optional[AlertEvent]:
        """Evaluate rule. Returns AlertEvent if triggered, else None."""
        if not self.enabled:
            return None
        now = time.time()
        if now - self.last_fired < self.cooldown_secs:
            return None
        value = self.metric_fn()
        if self._triggered(value):
            self.last_fired = now
            self.fire_count += 1
            msg = self.message_tmpl.format(
                name=self.name, value=value, threshold=self.threshold
            )
            return AlertEvent(
                rule_name=self.name,
                severity=self.severity,
                message=msg,
                value=value,
                threshold=self.threshold,
            )
        return None

    def _triggered(self, value: float) -> bool:
        c = self.comparator
        if c == "gt":  return value > self.threshold
        if c == "lt":  return value < self.threshold
        if c == "gte": return value >= self.threshold
        if c == "lte": return value <= self.threshold
        if c == "eq":  return abs(value - self.threshold) < 1e-9
        return False


class AlertRuleEngine:
    """Evaluates a set of AlertRules and collects fired events."""

    # Pre-built default rules for Argus
    @staticmethod
    def default_rules(
        get_halted: Callable,
        get_drawdown: Callable,
        get_cvar_95: Callable,
        get_cvar_99: Callable,
        get_daily_pnl: Callable,
        get_fill_latency_p99: Callable,
    ) -> List[AlertRule]:
        return [
            AlertRule(
                name="argus_risk_halted",
                metric_fn=get_halted,
                threshold=0.5, comparator="gt",
                severity=Severity.EMERGENCY,
                message_tmpl="🚨 EMERGENCY: Argus risk system HALTED",
                cooldown_secs=60,
            ),
            AlertRule(
                name="drawdown_critical",
                metric_fn=get_drawdown,
                threshold=5.0, comparator="gt",
                severity=Severity.CRITICAL,
                message_tmpl="🔴 CRITICAL: Drawdown {value:.2f}% > {threshold}%",
                cooldown_secs=300,
            ),
            AlertRule(
                name="drawdown_warn",
                metric_fn=get_drawdown,
                threshold=3.0, comparator="gt",
                severity=Severity.WARN,
                message_tmpl="⚠️ WARN: Drawdown {value:.2f}% > {threshold}%",
                cooldown_secs=600,
            ),
            AlertRule(
                name="cvar_95_breach",
                metric_fn=get_cvar_95,
                threshold=0.05, comparator="gt",
                severity=Severity.CRITICAL,
                message_tmpl="🔴 CRITICAL: CVaR-95 {value:.3f} > {threshold}",
                cooldown_secs=300,
            ),
            AlertRule(
                name="cvar_99_breach",
                metric_fn=get_cvar_99,
                threshold=0.10, comparator="gt",
                severity=Severity.EMERGENCY,
                message_tmpl="🚨 EMERGENCY: CVaR-99 {value:.3f} > {threshold}",
                cooldown_secs=60,
            ),
            AlertRule(
                name="daily_pnl_loss",
                metric_fn=get_daily_pnl,
                threshold=-500.0, comparator="lt",
                severity=Severity.CRITICAL,
                message_tmpl="🔴 CRITICAL: Daily PnL ${value:.2f} < ${threshold:.2f}",
                cooldown_secs=300,
            ),
            AlertRule(
                name="fill_latency_high",
                metric_fn=get_fill_latency_p99,
                threshold=500.0, comparator="gt",
                severity=Severity.WARN,
                message_tmpl="⚠️ WARN: p99 fill latency {value:.0f}ms > {threshold:.0f}ms",
                cooldown_secs=120,
            ),
        ]

    def __init__(self, rules: List[AlertRule] | None = None):
        self._rules: List[AlertRule] = rules or []
        self._fired_events: List[AlertEvent] = []

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def evaluate_all(self) -> List[AlertEvent]:
        fired = []
        for rule in self._rules:
            event = rule.evaluate()
            if event:
                fired.append(event)
                self._fired_events.append(event)
        return fired

    def get_history(self, last_n: int = 50) -> List[AlertEvent]:
        return self._fired_events[-last_n:]

    def disable(self, name: str) -> bool:
        for r in self._rules:
            if r.name == name:
                r.enabled = False
                return True
        return False

    def enable(self, name: str) -> bool:
        for r in self._rules:
            if r.name == name:
                r.enabled = True
                return True
        return False
