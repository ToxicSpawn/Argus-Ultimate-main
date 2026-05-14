"""Push 97 — Alert→Action bridge (v8.33.0).

Wires AlertManager events to automatic trading actions:
  - KILL_SWITCH_AUTO  → activate kill switch
  - REDUCE_POSITION   → MarginWatcher.auto_reduce() on worst position
  - HEDGE_DELTA       → submit a delta-neutral hedge order via OrderManager
  - NOTIFY_ONLY       → no action (default for informational rules)

Design:
  ActionRule          dataclass: maps alert name pattern -> ActionType + params
  AlertActionBridge   subscribes to AlertManager; dispatches actions
  ActionResult        outcome of an executed action

Integrates with:
  core/alerts/alert_manager.py    (AlertManager, AlertEvent)
  core/api/app.py                 (AppContext)
  core/risk/risk_manager.py       (activate_kill_switch)
  core/risk/margin_watcher.py     (auto_reduce)
  core/execution/order_manager.py (submit_order)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums + dataclasses
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    KILL_SWITCH_AUTO  = "kill_switch_auto"
    REDUCE_POSITION   = "reduce_position"
    HEDGE_DELTA       = "hedge_delta"
    NOTIFY_ONLY       = "notify_only"


@dataclass
class ActionRule:
    """Maps an alert name (or prefix) to an automatic action."""
    alert_name:   str         # exact name or prefix (ends with '*')
    action_type:  ActionType
    # Optional params
    reduce_pct:   float = 0.5     # fraction to reduce for REDUCE_POSITION
    hedge_ratio:  float = 1.0     # hedge notional ratio for HEDGE_DELTA
    cooldown_s:   float = 60.0    # min seconds between repeated triggers
    enabled:      bool  = True
    description:  str   = ""


@dataclass
class ActionResult:
    rule:         ActionRule
    alert_name:   str
    action_type:  ActionType
    success:      bool
    message:      str
    ts:           float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Default action rules (mirror alert_rules defaults from Push 95)
# ---------------------------------------------------------------------------

DEFAULT_ACTION_RULES: List[ActionRule] = [
    ActionRule(
        alert_name="kill_switch_auto_pct",
        action_type=ActionType.KILL_SWITCH_AUTO,
        cooldown_s=300.0,
        description="Auto kill-switch on >= 10% drawdown",
    ),
    ActionRule(
        alert_name="drawdown_pct",
        action_type=ActionType.REDUCE_POSITION,
        reduce_pct=0.5,
        cooldown_s=120.0,
        description="Reduce largest position 50% on > 5% drawdown",
    ),
    ActionRule(
        alert_name="vol_spike_ratio",
        action_type=ActionType.REDUCE_POSITION,
        reduce_pct=0.25,
        cooldown_s=90.0,
        description="Reduce 25% on extreme vol spike",
    ),
    ActionRule(
        alert_name="confidence_floor",
        action_type=ActionType.NOTIFY_ONLY,
        cooldown_s=60.0,
        description="Log + notify only; no position change",
    ),
]


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class AlertActionBridge:
    """Subscribes to AlertManager and executes automatic trading actions.

    Usage::

        bridge = AlertActionBridge(ctx)
        bridge.register_rules(DEFAULT_ACTION_RULES)
        alert_manager.register_handler(bridge.on_alert)

    The bridge is async-safe: on_alert() can be called from any coroutine.
    All actions are executed via asyncio.create_task() to avoid blocking
    the alert dispatch loop.
    """

    def __init__(self, ctx: Any) -> None:
        """
        Args:
            ctx: AppContext with .risk_manager, .order_manager, .adapter fields.
        """
        self._ctx              = ctx
        self._rules:      List[ActionRule]   = []
        self._last_fired: Dict[str, float]   = {}  # rule.alert_name -> last ts
        self._history:    List[ActionResult] = []
        self._total_triggered  = 0
        self._total_blocked    = 0

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def register_rules(self, rules: List[ActionRule]) -> None:
        self._rules.extend(rules)

    def add_rule(self, rule: ActionRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, alert_name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.alert_name != alert_name]
        return len(self._rules) < before

    @property
    def rules(self) -> List[ActionRule]:
        return list(self._rules)

    # ------------------------------------------------------------------
    # Alert handler
    # ------------------------------------------------------------------

    async def on_alert(self, event: Any) -> None:
        """Called by AlertManager on every alert event.

        Matches event.title against registered rules and fires actions.
        """
        alert_name = getattr(event, "title", "") or ""
        for rule in self._rules:
            if not rule.enabled:
                continue
            if not self._matches(rule.alert_name, alert_name):
                continue
            # Cooldown check
            last = self._last_fired.get(rule.alert_name, 0.0)
            if time.time() - last < rule.cooldown_s:
                self._total_blocked += 1
                continue
            self._last_fired[rule.alert_name] = time.time()
            asyncio.create_task(self._execute(rule, alert_name))

    # ------------------------------------------------------------------
    # Action executors
    # ------------------------------------------------------------------

    async def _execute(self, rule: ActionRule, alert_name: str) -> None:
        self._total_triggered += 1
        result: ActionResult
        try:
            if rule.action_type == ActionType.KILL_SWITCH_AUTO:
                result = await self._do_kill_switch(rule, alert_name)
            elif rule.action_type == ActionType.REDUCE_POSITION:
                result = await self._do_reduce(rule, alert_name)
            elif rule.action_type == ActionType.HEDGE_DELTA:
                result = await self._do_hedge(rule, alert_name)
            else:
                result = ActionResult(
                    rule=rule, alert_name=alert_name,
                    action_type=ActionType.NOTIFY_ONLY,
                    success=True, message="notify_only; no action taken",
                )
        except Exception as exc:  # noqa: BLE001
            result = ActionResult(
                rule=rule, alert_name=alert_name,
                action_type=rule.action_type,
                success=False, message=str(exc),
            )
            logger.error("AlertActionBridge action failed: %s", exc)
        self._history.append(result)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        logger.info(
            "AlertAction %s -> %s: %s",
            alert_name, rule.action_type.value,
            "OK" if result.success else f"FAIL({result.message})",
        )

    async def _do_kill_switch(self, rule: ActionRule, alert_name: str) -> ActionResult:
        rm = getattr(self._ctx, "risk_manager", None)
        if rm is None:
            return ActionResult(
                rule=rule, alert_name=alert_name,
                action_type=ActionType.KILL_SWITCH_AUTO,
                success=False, message="risk_manager not in AppContext",
            )
        reason = f"auto kill-switch triggered by alert: {alert_name}"
        if asyncio.iscoroutinefunction(rm.activate_kill_switch):
            await rm.activate_kill_switch(reason)
        else:
            rm.activate_kill_switch(reason)
        return ActionResult(
            rule=rule, alert_name=alert_name,
            action_type=ActionType.KILL_SWITCH_AUTO,
            success=True, message=reason,
        )

    async def _do_reduce(self, rule: ActionRule, alert_name: str) -> ActionResult:
        om = getattr(self._ctx, "order_manager", None)
        if om is None:
            return ActionResult(
                rule=rule, alert_name=alert_name,
                action_type=ActionType.REDUCE_POSITION,
                success=False, message="order_manager not in AppContext",
            )
        # Find largest open position
        try:
            stats = om.stats
            positions = stats.get("positions", {})
            if not positions:
                return ActionResult(
                    rule=rule, alert_name=alert_name,
                    action_type=ActionType.REDUCE_POSITION,
                    success=True, message="no open positions to reduce",
                )
            largest_sym = max(positions, key=lambda s: abs(positions[s].get("notional", 0)))
            pos = positions[largest_sym]
            qty_to_reduce = abs(pos.get("qty", 0.0)) * rule.reduce_pct
            side = "SELL" if pos.get("side", "LONG") == "LONG" else "BUY"
            if hasattr(om, "submit_reduce"):
                await om.submit_reduce(largest_sym, side, qty_to_reduce)
            return ActionResult(
                rule=rule, alert_name=alert_name,
                action_type=ActionType.REDUCE_POSITION,
                success=True,
                message=f"reduce {rule.reduce_pct*100:.0f}% of {largest_sym} ({side} {qty_to_reduce:.4f})",
            )
        except Exception as exc:
            return ActionResult(
                rule=rule, alert_name=alert_name,
                action_type=ActionType.REDUCE_POSITION,
                success=False, message=str(exc),
            )

    async def _do_hedge(self, rule: ActionRule, alert_name: str) -> ActionResult:
        om = getattr(self._ctx, "order_manager", None)
        if om is None:
            return ActionResult(
                rule=rule, alert_name=alert_name,
                action_type=ActionType.HEDGE_DELTA,
                success=False, message="order_manager not in AppContext",
            )
        try:
            stats     = om.stats
            positions = stats.get("positions", {})
            if not positions:
                return ActionResult(
                    rule=rule, alert_name=alert_name,
                    action_type=ActionType.HEDGE_DELTA,
                    success=True, message="no positions to hedge",
                )
            # Hedge the largest net delta
            largest_sym = max(positions, key=lambda s: abs(positions[s].get("notional", 0)))
            pos = positions[largest_sym]
            qty = abs(pos.get("qty", 0.0)) * rule.hedge_ratio
            side = "SELL" if pos.get("side", "LONG") == "LONG" else "BUY"
            if hasattr(om, "submit_market"):
                await om.submit_market(largest_sym, side, qty, tag="hedge_auto")
            return ActionResult(
                rule=rule, alert_name=alert_name,
                action_type=ActionType.HEDGE_DELTA,
                success=True,
                message=f"hedge {largest_sym} {side} {qty:.4f} (ratio={rule.hedge_ratio})",
            )
        except Exception as exc:
            return ActionResult(
                rule=rule, alert_name=alert_name,
                action_type=ActionType.HEDGE_DELTA,
                success=False, message=str(exc),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(pattern: str, name: str) -> bool:
        if pattern.endswith("*"):
            return name.startswith(pattern[:-1])
        return name == pattern

    @property
    def stats(self) -> dict:
        return {
            "rules":            len(self._rules),
            "total_triggered":  self._total_triggered,
            "total_blocked":    self._total_blocked,
            "history_len":      len(self._history),
            "last_actions":     [
                {
                    "alert":   r.alert_name,
                    "action":  r.action_type.value,
                    "success": r.success,
                    "ts":      r.ts,
                }
                for r in self._history[-10:]
            ],
        }
