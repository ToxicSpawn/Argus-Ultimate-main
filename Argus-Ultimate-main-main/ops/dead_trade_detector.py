#!/usr/bin/env python3
"""
Dead Trade Detector — detects when the system stops producing trades.

Monitors trade frequency and alerts when no trade has been executed for a
configurable period (default 4 hours during market hours).  Provides
possible reasons for the silence based on system state.

Crypto markets are 24/7, so "market hours" means any time.  The detector
considers periods of low volume (UTC 2–5) as reduced-expectation windows.

Usage::

    detector = DeadTradeDetector()
    detector.record_trade(time.time())
    # Periodic check:
    alert = detector.check(expected_trades_per_hour=2)
    if alert:
        print(f"No trades for {alert.hours_since_last_trade:.1f}h")
        print(alert.possible_reasons)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ALERT_HOURS = 4.0        # Alert if no trade for this many hours
_LOW_VOLUME_HOURS_UTC = {2, 3, 4, 5}  # Reduced expectations during Asian early morning
_LOW_VOLUME_MULTIPLIER = 2.0      # Allow 2x the normal silence during low-volume hours


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeadTradeAlert:
    """Alert indicating the system has stopped trading."""

    hours_since_last_trade: float
    expected_trades: float           # expected trades in the silent period
    possible_reasons: List[str]
    last_trade_time: Optional[datetime] = None
    severity: str = "warning"        # "warning" or "critical"


# ---------------------------------------------------------------------------
# Dead Trade Detector
# ---------------------------------------------------------------------------


class DeadTradeDetector:
    """Detects when the trading system stops producing trades.

    Parameters
    ----------
    alert_after_hours : float
        Hours of silence before raising an alert.  Default 4.0.
    system_state_callback : callable, optional
        Function returning a dict of system state for reason diagnosis.
        Expected keys: ``exchange_status`` (str), ``circuit_breaker_active``
        (bool), ``signals_suppressed`` (bool), ``process_healthy`` (bool).
    """

    def __init__(
        self,
        alert_after_hours: float = _DEFAULT_ALERT_HOURS,
        system_state_callback: Optional[Callable[[], Dict[str, object]]] = None,
    ) -> None:
        self._alert_hours = max(0.5, alert_after_hours)
        self._state_callback = system_state_callback
        self._trade_timestamps: List[float] = []
        self._max_history = 1000  # keep last N trade timestamps
        self._last_alert_ts: float = 0.0
        self._alert_cooldown_seconds = 1800.0  # 30 min between alerts

        logger.info(
            "DeadTradeDetector initialised — alert_after=%.1fh", self._alert_hours,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_trade(self, timestamp: Optional[float] = None) -> None:
        """Record that a trade was executed.

        Parameters
        ----------
        timestamp : float, optional
            Epoch timestamp of the trade.  Defaults to now.
        """
        ts = timestamp if timestamp is not None else time.time()
        self._trade_timestamps.append(ts)

        # Trim history
        if len(self._trade_timestamps) > self._max_history:
            self._trade_timestamps = self._trade_timestamps[-self._max_history:]

        logger.debug("DeadTradeDetector: trade recorded at %s", ts)

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------

    def check(
        self, expected_trades_per_hour: float = 2.0
    ) -> Optional[DeadTradeAlert]:
        """Check whether the system has stopped trading.

        Parameters
        ----------
        expected_trades_per_hour : float
            Baseline expected trade frequency.

        Returns
        -------
        DeadTradeAlert or None
            Alert if silence exceeds threshold, None if healthy.
        """
        now = time.time()

        if not self._trade_timestamps:
            # No trades ever recorded — cannot alert meaningfully
            return None

        last_trade = max(self._trade_timestamps)
        hours_silent = (now - last_trade) / 3600.0

        # Adjust threshold for low-volume hours
        current_hour_utc = datetime.fromtimestamp(now, tz=timezone.utc).hour
        threshold_hours = self._alert_hours
        if current_hour_utc in _LOW_VOLUME_HOURS_UTC:
            threshold_hours *= _LOW_VOLUME_MULTIPLIER

        if hours_silent < threshold_hours:
            return None

        # Rate-limit alerts
        if now - self._last_alert_ts < self._alert_cooldown_seconds:
            return None

        self._last_alert_ts = now

        expected_trades = hours_silent * expected_trades_per_hour
        reasons = self._diagnose_reasons(hours_silent)

        severity = "critical" if hours_silent > threshold_hours * 2 else "warning"

        alert = DeadTradeAlert(
            hours_since_last_trade=round(hours_silent, 2),
            expected_trades=round(expected_trades, 1),
            possible_reasons=reasons,
            last_trade_time=datetime.fromtimestamp(last_trade, tz=timezone.utc),
            severity=severity,
        )

        logger.warning(
            "DeadTradeDetector: ALERT — no trades for %.1fh (expected ~%.0f), "
            "severity=%s, reasons=%s",
            hours_silent, expected_trades, severity, reasons,
        )
        return alert

    def _diagnose_reasons(self, hours_silent: float) -> List[str]:
        """Attempt to diagnose why trading stopped.

        Uses the system state callback if available, otherwise returns
        generic possible reasons.
        """
        reasons: List[str] = []

        # Query system state if callback is available
        state: Dict[str, object] = {}
        if self._state_callback:
            try:
                state = self._state_callback()
            except Exception:
                logger.exception("DeadTradeDetector: state callback failed")
                reasons.append("system state unavailable (callback error)")

        # Check exchange status
        exchange_status = state.get("exchange_status", "unknown")
        if exchange_status in ("down", "maintenance", "degraded"):
            reasons.append("exchange down")
        elif exchange_status == "unknown":
            reasons.append("exchange down")  # plausible default

        # Check circuit breaker
        if state.get("circuit_breaker_active"):
            reasons.append("circuit breaker active")

        # Check signal suppression
        if state.get("signals_suppressed"):
            reasons.append("all signals suppressed")

        # Check process health
        if state.get("process_healthy") is False:
            reasons.append("process hung")

        # Generic reasons if we have no state info
        if not reasons:
            reasons = [
                "all signals suppressed",
                "exchange down",
                "process hung",
                "market closed",
                "circuit breaker active",
            ]

        return reasons

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_trade_frequency(self, hours: float = 24.0) -> float:
        """Return trades per hour over the specified lookback.

        Parameters
        ----------
        hours : float
            Lookback period in hours.

        Returns
        -------
        float
            Average trades per hour.
        """
        cutoff = time.time() - hours * 3600.0
        recent = [ts for ts in self._trade_timestamps if ts >= cutoff]
        if not recent or hours <= 0:
            return 0.0
        return len(recent) / hours

    def get_last_trade_time(self) -> Optional[datetime]:
        """Return the datetime of the most recent trade (UTC)."""
        if not self._trade_timestamps:
            return None
        return datetime.fromtimestamp(
            max(self._trade_timestamps), tz=timezone.utc
        )

    def get_hours_since_last_trade(self) -> Optional[float]:
        """Return hours since the last trade, or None if no trades recorded."""
        if not self._trade_timestamps:
            return None
        return (time.time() - max(self._trade_timestamps)) / 3600.0
