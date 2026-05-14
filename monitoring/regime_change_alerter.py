"""
Batch 3 – Regime Change Alerter
=================================
Detects and broadcasts regime transitions to registered alert channels
(log, webhook, internal callback).  Consumes the
`_latest_regime_label` field maintained by the trading loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from unified_trading_system import UnifiedSystemArchitecture

logger = logging.getLogger(__name__)


class RegimeChangeAlerter:
    """
    Watches `system._latest_regime_label` each cycle and fires alerts
    when a transition is detected.

    Alert channels:
    - Python logger (always active)
    - Registered async callbacks (e.g. Telegram, Slack, webhook)
    """

    def __init__(self, system: "UnifiedSystemArchitecture") -> None:
        self._sys = system
        self._prev_regime: str = ""
        self._transition_count: int = 0
        self._last_transition_ts: float = 0.0
        self._callbacks: List[Callable[[str, str, float], Any]] = []
        # Cooldown: avoid duplicate alerts within N seconds
        self._cooldown_seconds: float = float(
            getattr(getattr(system, "config", None), "regime_alerter_cooldown_seconds", 60.0) or 60.0
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(self, cb: Callable[[str, str, float], Any]) -> None:
        """
        Register an alert callback.  Signature: cb(from_regime, to_regime, ts).
        May be sync or async; async callbacks are wrapped in create_task.
        """
        self._callbacks.append(cb)

    def check(self) -> None:
        """
        Call once per trading loop cycle to check for regime transitions.
        Lightweight – only fires on actual change + cooldown.
        """
        current = str(
            getattr(self._sys, "_latest_regime_label", "")
            or getattr(self._sys, "_last_regime_consensus", {}).get("regime", "")
            or ""
        ).strip().lower()

        if not current:
            return
        if current == self._prev_regime:
            return

        now = time.time()
        if (now - self._last_transition_ts) < self._cooldown_seconds:
            return

        prev = self._prev_regime
        self._prev_regime = current
        self._transition_count += 1
        self._last_transition_ts = now

        logger.info(
            "REGIME CHANGE: %s → %s (transition #%d)",
            prev or "<init>",
            current,
            self._transition_count,
        )
        self._fire_callbacks(prev, current, now)

    def stats(self) -> Dict[str, Any]:
        return {
            "current_regime": self._prev_regime,
            "transition_count": self._transition_count,
            "last_transition_ts": self._last_transition_ts,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire_callbacks(self, from_regime: str, to_regime: str, ts: float) -> None:
        for cb in list(self._callbacks):
            try:
                result = cb(from_regime, to_regime, ts)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(result)
                        else:
                            loop.run_until_complete(result)
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("RegimeChangeAlerter callback error: %s", exc)
