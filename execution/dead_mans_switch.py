#!/usr/bin/env python3
"""
Dead Man's Switch — safety mechanism that detects system unresponsiveness.

If the trading system fails to call ``heartbeat()`` within ``timeout_minutes``
(default 30), the switch triggers:
1. Sends an alert via Discord/Telegram webhook.
2. Optionally closes all open positions (configurable).

The switch is designed to be called every trading cycle.  It is thread-safe
and uses no background threads (the check is performed on each ``check()``
or ``heartbeat()`` call).

Usage::

    switch = DeadMansSwitch(
        timeout_minutes=30,
        alert_webhook_url="https://discord.com/api/webhooks/...",
        close_positions_on_timeout=True,
    )
    # Every cycle:
    switch.heartbeat()
    # Periodic health check:
    if not switch.check():
        logger.critical("Dead man's switch triggered!")
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_MINUTES = 30
_MAX_ALERT_FREQUENCY_SECONDS = 300  # Don't spam alerts more than once per 5 min


# ---------------------------------------------------------------------------
# Dead Man's Switch
# ---------------------------------------------------------------------------


class DeadMansSwitch:
    """Safety mechanism that detects system unresponsiveness.

    Parameters
    ----------
    timeout_minutes : int
        Minutes of silence before triggering.  Default 30.
    alert_webhook_url : str, optional
        Discord or Telegram webhook URL for alerts.
    close_positions_on_timeout : bool
        If True, call the position-close callback when triggered.
    close_callback : callable, optional
        Function to call to close all positions.  Signature: ``() -> None``.
    alert_callback : callable, optional
        Custom alert function.  Signature: ``(message: str) -> None``.
        If not provided, the webhook URL is used.
    """

    def __init__(
        self,
        timeout_minutes: int = _DEFAULT_TIMEOUT_MINUTES,
        alert_webhook_url: Optional[str] = None,
        close_positions_on_timeout: bool = False,
        close_callback: Optional[Callable[[], None]] = None,
        alert_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._timeout_seconds = max(60, timeout_minutes * 60)
        self._webhook_url = alert_webhook_url
        self._close_on_timeout = close_positions_on_timeout
        self._close_callback = close_callback
        self._alert_callback = alert_callback

        self._last_heartbeat: float = time.time()
        self._last_alert_ts: float = 0.0
        self._triggered = False
        self._trigger_count = 0
        self._lock = threading.Lock()

        logger.info(
            "DeadMansSwitch initialised — timeout=%d min, close_on_timeout=%s, "
            "webhook=%s",
            timeout_minutes, close_positions_on_timeout,
            "configured" if alert_webhook_url else "none",
        )

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(self) -> None:
        """Reset the dead man's switch timer.

        Call this every trading cycle to indicate the system is alive.
        """
        with self._lock:
            self._last_heartbeat = time.time()
            if self._triggered:
                logger.info(
                    "DeadMansSwitch: system recovered — heartbeat received "
                    "after trigger",
                )
                self._triggered = False

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(self) -> bool:
        """Check whether the system is still alive.

        Returns
        -------
        bool
            True if the last heartbeat is within the timeout window (alive).
            False if the switch has triggered (dead).
        """
        with self._lock:
            elapsed = time.time() - self._last_heartbeat
            alive = elapsed < self._timeout_seconds

            if not alive and not self._triggered:
                self._triggered = True
                self._trigger_count += 1
                logger.critical(
                    "DeadMansSwitch TRIGGERED — no heartbeat for %.1f min "
                    "(timeout=%d min, trigger #%d)",
                    elapsed / 60.0,
                    self._timeout_seconds / 60,
                    self._trigger_count,
                )
                # Release lock before potentially slow I/O
                self._lock.release()
                try:
                    self._on_trigger(elapsed)
                finally:
                    self._lock.acquire()

            return alive

    def _on_trigger(self, elapsed_seconds: float) -> None:
        """Handle a trigger event: alert and optionally close positions."""
        msg = (
            f"DEAD MAN'S SWITCH TRIGGERED\n"
            f"No heartbeat for {elapsed_seconds / 60.0:.1f} minutes "
            f"(threshold: {self._timeout_seconds / 60:.0f} min).\n"
            f"Trigger count: {self._trigger_count}\n"
            f"Close positions: {self._close_on_timeout}"
        )

        # Send alert (rate-limited)
        now = time.time()
        if now - self._last_alert_ts >= _MAX_ALERT_FREQUENCY_SECONDS:
            self._send_alert(msg)
            self._last_alert_ts = now

        # Close positions if configured
        if self._close_on_timeout and self._close_callback:
            try:
                logger.warning("DeadMansSwitch: closing all positions")
                self._close_callback()
            except Exception:
                logger.exception("DeadMansSwitch: error closing positions")

    def _send_alert(self, message: str) -> None:
        """Send alert via callback or webhook."""
        if self._alert_callback:
            try:
                self._alert_callback(message)
                return
            except Exception:
                logger.exception("DeadMansSwitch: alert callback failed")

        if not self._webhook_url:
            logger.warning(
                "DeadMansSwitch: no alert channel configured — logging only"
            )
            return

        try:
            # Detect Discord vs Telegram webhook format
            if "discord.com" in self._webhook_url:
                payload = json.dumps({"content": message}).encode("utf-8")
                content_type = "application/json"
            elif "api.telegram.org" in self._webhook_url:
                # Telegram: webhook URL should be full sendMessage URL
                payload = json.dumps({"text": message}).encode("utf-8")
                content_type = "application/json"
            else:
                payload = json.dumps({"text": message}).encode("utf-8")
                content_type = "application/json"

            req = urllib.request.Request(
                self._webhook_url,
                data=payload,
                headers={"Content-Type": content_type},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info(
                    "DeadMansSwitch: alert sent (status %d)", resp.status,
                )
        except Exception:
            logger.exception("DeadMansSwitch: failed to send webhook alert")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_last_heartbeat(self) -> datetime:
        """Return the datetime of the last heartbeat (UTC)."""
        with self._lock:
            return datetime.fromtimestamp(self._last_heartbeat, tz=timezone.utc)

    def get_silence_duration(self) -> timedelta:
        """Return how long since the last heartbeat."""
        with self._lock:
            return timedelta(seconds=time.time() - self._last_heartbeat)

    @property
    def is_triggered(self) -> bool:
        """Whether the switch is currently in triggered state."""
        with self._lock:
            return self._triggered

    @property
    def trigger_count(self) -> int:
        """Total number of times the switch has triggered."""
        with self._lock:
            return self._trigger_count

    @property
    def timeout_minutes(self) -> int:
        """Configured timeout in minutes."""
        return self._timeout_seconds // 60
