"""
drawdown_monitor.py — Live drawdown monitor wired to IncidentReporter circuit-breaker.

Tracks equity curve in real-time and fires graduated alerts + hard stops
when drawdown thresholds are breached.

Threshold levels (configurable)
---------------------------------
  WARNING  : 5%  drawdown — alert only, continue trading
  CRITICAL : 10% drawdown — alert + reduce position sizes by 50%
  HALT     : 15% drawdown — alert + halt all new orders immediately
  EMERGENCY: 20% drawdown — alert + close all positions + stop bot

Features
--------
- Tracks peak equity and rolling drawdown continuously
- Cooldown between repeated alerts (avoids spam)
- Daily P&L reset at configurable UTC hour
- Integrates with IncidentReporter for Discord + email alerts
- Thread-safe for use with asyncio and background tasks
- Emits structured DrawdownEvent for downstream logging
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DrawdownLevel(str, Enum):
    NORMAL    = "NORMAL"
    WARNING   = "WARNING"
    CRITICAL  = "CRITICAL"
    HALT      = "HALT"
    EMERGENCY = "EMERGENCY"


ACTION_MAP = {
    DrawdownLevel.WARNING  : "alert_only",
    DrawdownLevel.CRITICAL : "reduce_size",
    DrawdownLevel.HALT     : "halt_orders",
    DrawdownLevel.EMERGENCY: "emergency_stop",
}


@dataclass
class DrawdownEvent:
    level: str
    drawdown_pct: float
    peak_equity: float
    current_equity: float
    action: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DrawdownMonitor:
    """
    Real-time drawdown monitor with graduated circuit-breaker.

    Parameters
    ----------
    initial_equity      : float  starting account equity
    warning_pct         : float  drawdown % to trigger WARNING (default 5)
    critical_pct        : float  drawdown % to trigger CRITICAL (default 10)
    halt_pct            : float  drawdown % to trigger HALT (default 15)
    emergency_pct       : float  drawdown % to trigger EMERGENCY (default 20)
    alert_cooldown_sec  : float  minimum seconds between repeated alerts (default 300)
    daily_reset_hour_utc: int    UTC hour to reset daily peak equity (default 0)
    on_halt             : optional async callable() — called on HALT level
    on_emergency        : optional async callable() — called on EMERGENCY level
    on_reduce_size      : optional async callable(factor: float) — called on CRITICAL
    reporter            : optional IncidentReporter instance
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        warning_pct: float = 5.0,
        critical_pct: float = 10.0,
        halt_pct: float = 15.0,
        emergency_pct: float = 20.0,
        alert_cooldown_sec: float = 300.0,
        daily_reset_hour_utc: int = 0,
        on_halt: Optional[Callable] = None,
        on_emergency: Optional[Callable] = None,
        on_reduce_size: Optional[Callable] = None,
        reporter: Optional[Any] = None,
    ) -> None:
        self._initial_equity      = float(initial_equity)
        self._peak_equity         = float(initial_equity)
        self._current_equity      = float(initial_equity)
        self._daily_peak          = float(initial_equity)
        self._daily_reset_hour    = daily_reset_hour_utc
        self._last_daily_reset    = self._today_utc()

        self._thresholds = {
            DrawdownLevel.WARNING  : float(warning_pct),
            DrawdownLevel.CRITICAL : float(critical_pct),
            DrawdownLevel.HALT     : float(halt_pct),
            DrawdownLevel.EMERGENCY: float(emergency_pct),
        }

        self._cooldown_sec  = alert_cooldown_sec
        self._last_alert_ts: Dict[str, float] = {}
        self._current_level = DrawdownLevel.NORMAL
        self._halted        = False
        self._emergency     = False

        self._on_halt        = on_halt
        self._on_emergency   = on_emergency
        self._on_reduce_size = on_reduce_size
        self._reporter       = reporter

        self._event_log: List[DrawdownEvent] = []
        self._equity_history: List[tuple] = []  # (timestamp, equity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, equity: float) -> Optional[DrawdownEvent]:
        """
        Update equity and check drawdown thresholds.
        Call this after every trade or on a periodic tick.

        Returns DrawdownEvent if a threshold was breached, else None.
        """
        equity = float(equity)
        self._current_equity = equity
        self._equity_history.append((time.time(), equity))

        # Keep history bounded
        if len(self._equity_history) > 10_000:
            self._equity_history = self._equity_history[-5_000:]

        # Daily reset check
        self._check_daily_reset()

        # Update peaks
        if equity > self._peak_equity:
            self._peak_equity = equity
        if equity > self._daily_peak:
            self._daily_peak = equity

        # Calculate drawdown from all-time peak
        dd_pct = self._drawdown_pct(equity, self._peak_equity)
        level  = self._classify(dd_pct)

        if level == DrawdownLevel.NORMAL:
            self._current_level = DrawdownLevel.NORMAL
            return None

        # Check cooldown
        if not self._cooldown_elapsed(level):
            return None

        event = DrawdownEvent(
            level=level.value,
            drawdown_pct=round(dd_pct, 4),
            peak_equity=round(self._peak_equity, 4),
            current_equity=round(equity, 4),
            action=ACTION_MAP.get(level, "alert_only"),
            metadata={
                "daily_dd_pct": round(
                    self._drawdown_pct(equity, self._daily_peak), 4
                ),
                "from_initial_pct": round(
                    self._drawdown_pct(equity, self._initial_equity), 4
                ),
            },
        )
        self._event_log.append(event)
        self._last_alert_ts[level.value] = time.time()
        self._current_level = level

        await self._dispatch(event)
        return event

    def record_trade_pnl(self, pnl: float) -> None:
        """Convenience: update equity by adding a trade P&L."""
        asyncio.create_task(self.update(self._current_equity + pnl))

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def is_emergency(self) -> bool:
        return self._emergency

    @property
    def current_drawdown_pct(self) -> float:
        return self._drawdown_pct(self._current_equity, self._peak_equity)

    @property
    def current_level(self) -> str:
        return self._current_level.value

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def current_equity(self) -> float:
        return self._current_equity

    def event_log(self) -> List[Dict[str, Any]]:
        return [
            {
                "level"          : e.level,
                "drawdown_pct"   : e.drawdown_pct,
                "peak_equity"    : e.peak_equity,
                "current_equity" : e.current_equity,
                "action"         : e.action,
                "timestamp"      : e.timestamp,
                "metadata"       : e.metadata,
            }
            for e in self._event_log
        ]

    def max_drawdown_pct(self) -> float:
        """Maximum drawdown recorded since monitor started."""
        if len(self._equity_history) < 2:
            return 0.0
        equities = [e for _, e in self._equity_history]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = self._drawdown_pct(eq, peak)
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 4)

    def reset_halt(self) -> None:
        """Manually clear halt state (operator intervention after review)."""
        self._halted = False
        self._current_level = DrawdownLevel.NORMAL
        logger.warning("DrawdownMonitor: halt manually cleared")

    def reset_emergency(self) -> None:
        """Manually clear emergency state."""
        self._emergency = False
        self._halted = False
        self._current_level = DrawdownLevel.NORMAL
        logger.warning("DrawdownMonitor: emergency manually cleared")

    # ------------------------------------------------------------------
    # Dispatch actions
    # ------------------------------------------------------------------

    async def _dispatch(self, event: DrawdownEvent) -> None:
        level = DrawdownLevel(event.level)
        msg = (
            f"\U0001f6a8 DRAWDOWN {event.level} | "
            f"DD={event.drawdown_pct:.2f}% | "
            f"Equity={event.current_equity:.2f} (peak={event.peak_equity:.2f}) | "
            f"Action: {event.action}"
        )
        logger.warning(msg)

        # Fire IncidentReporter if available
        if self._reporter:
            try:
                await self._reporter.alert(
                    title=f"Drawdown {event.level}",
                    message=msg,
                    level=event.level,
                    metadata=event.metadata,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("DrawdownMonitor reporter error: %s", exc)

        if level == DrawdownLevel.CRITICAL:
            self._current_level = DrawdownLevel.CRITICAL
            if self._on_reduce_size:
                try:
                    await self._call(self._on_reduce_size, 0.5)
                except Exception as exc:  # noqa: BLE001
                    logger.error("on_reduce_size callback error: %s", exc)

        elif level == DrawdownLevel.HALT:
            self._halted = True
            if self._on_halt:
                try:
                    await self._call(self._on_halt)
                except Exception as exc:  # noqa: BLE001
                    logger.error("on_halt callback error: %s", exc)

        elif level == DrawdownLevel.EMERGENCY:
            self._halted    = True
            self._emergency = True
            if self._on_emergency:
                try:
                    await self._call(self._on_emergency)
                except Exception as exc:  # noqa: BLE001
                    logger.error("on_emergency callback error: %s", exc)

    @staticmethod
    async def _call(fn: Callable, *args: Any) -> None:
        if asyncio.iscoroutinefunction(fn):
            await fn(*args)
        else:
            fn(*args)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify(self, dd_pct: float) -> DrawdownLevel:
        if dd_pct >= self._thresholds[DrawdownLevel.EMERGENCY]:
            return DrawdownLevel.EMERGENCY
        if dd_pct >= self._thresholds[DrawdownLevel.HALT]:
            return DrawdownLevel.HALT
        if dd_pct >= self._thresholds[DrawdownLevel.CRITICAL]:
            return DrawdownLevel.CRITICAL
        if dd_pct >= self._thresholds[DrawdownLevel.WARNING]:
            return DrawdownLevel.WARNING
        return DrawdownLevel.NORMAL

    def _cooldown_elapsed(self, level: DrawdownLevel) -> bool:
        last = self._last_alert_ts.get(level.value, 0.0)
        return (time.time() - last) >= self._cooldown_sec

    @staticmethod
    def _drawdown_pct(equity: float, peak: float) -> float:
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - equity) / peak * 100.0)

    def _check_daily_reset(self) -> None:
        import datetime
        now_hour = datetime.datetime.utcnow().hour
        today    = datetime.datetime.utcnow().date()
        if today > self._last_daily_reset and now_hour >= self._daily_reset_hour:
            self._daily_peak     = self._current_equity
            self._last_daily_reset = today
            logger.info("DrawdownMonitor: daily peak reset to %.2f", self._daily_peak)

    @staticmethod
    def _today_utc():
        import datetime
        return datetime.datetime.utcnow().date()
