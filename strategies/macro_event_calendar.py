"""
Macro Event Calendar — avoids trading around high-impact macro events.

High-impact events that move crypto markets:
  - US CPI (monthly)
  - US FOMC meetings (8x per year)
  - Non-Farm Payrolls (monthly)
  - SEC crypto rulings
  - Bitcoin ETF approval/rejection dates

Strategy:
  1. Maintain calendar of upcoming events
  2. Within T-2h to T+1h of any HIGH impact event: reduce position size by 70%
  3. Within T-30min: halt new entries entirely
  4. Track event outcomes to update model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# Sentinel representing "all crypto assets"
ALL_ASSETS: List[str] = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA"]


@dataclass
class MacroEvent:
    """A scheduled macroeconomic event with known market impact."""
    name: str
    event_time: datetime
    impact: str                      # "HIGH", "MEDIUM", or "LOW"
    description: str
    assets_affected: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.impact not in ("HIGH", "MEDIUM", "LOW"):
            raise ValueError(
                f"impact must be 'HIGH', 'MEDIUM', or 'LOW'; got {self.impact!r}"
            )
        # Normalise to UTC-aware if naive
        if self.event_time.tzinfo is None:
            self.event_time = self.event_time.replace(tzinfo=timezone.utc)

    def minutes_until(self, now: datetime) -> float:
        """Return signed minutes from *now* to the event (negative if past)."""
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta = self.event_time - now
        return delta.total_seconds() / 60.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MacroEvent({self.name!r}, {self.event_time.isoformat()}, "
            f"impact={self.impact!r})"
        )


def _utc(year: int, month: int, day: int, hour: int = 19, minute: int = 0) -> datetime:
    """Convenience helper — returns a UTC-aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Hardcoded 2026 macro event calendar
# ---------------------------------------------------------------------------
# FOMC meeting *end* dates (decision released ~19:00 UTC / 2 PM ET):
#   Jan 28-29, Mar 18-19, May 6-7, Jun 17-18, Jul 29-30,
#   Sep 16-17, Oct 28-29, Dec 9-10
# CPI releases happen roughly the 2nd week of each month at 12:30 UTC (8:30 AM ET).
# NFP releases first Friday of each month at 12:30 UTC.

_DEFAULT_EVENTS_2026: List[MacroEvent] = [
    # ---- FOMC decisions ----
    MacroEvent(
        name="FOMC Rate Decision",
        event_time=_utc(2026, 1, 29),
        impact="HIGH",
        description="Federal Open Market Committee interest rate decision — Jan 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="FOMC Rate Decision",
        event_time=_utc(2026, 3, 19),
        impact="HIGH",
        description="Federal Open Market Committee interest rate decision — Mar 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="FOMC Rate Decision",
        event_time=_utc(2026, 5, 7),
        impact="HIGH",
        description="Federal Open Market Committee interest rate decision — May 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="FOMC Rate Decision",
        event_time=_utc(2026, 6, 18),
        impact="HIGH",
        description="Federal Open Market Committee interest rate decision — Jun 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="FOMC Rate Decision",
        event_time=_utc(2026, 7, 30),
        impact="HIGH",
        description="Federal Open Market Committee interest rate decision — Jul 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="FOMC Rate Decision",
        event_time=_utc(2026, 9, 17),
        impact="HIGH",
        description="Federal Open Market Committee interest rate decision — Sep 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="FOMC Rate Decision",
        event_time=_utc(2026, 10, 29),
        impact="HIGH",
        description="Federal Open Market Committee interest rate decision — Oct 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="FOMC Rate Decision",
        event_time=_utc(2026, 12, 10),
        impact="HIGH",
        description="Federal Open Market Committee interest rate decision — Dec 2026",
        assets_affected=ALL_ASSETS,
    ),
    # ---- US CPI releases (2nd week, ~12:30 UTC) ----
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 1, 14, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — January 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 2, 11, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — February 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 3, 11, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — March 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 4, 8, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — April 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 5, 13, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — May 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 6, 10, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — June 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 7, 8, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — July 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 8, 12, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — August 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 9, 9, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — September 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 10, 14, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — October 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 11, 12, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — November 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US CPI",
        event_time=_utc(2026, 12, 9, 12, 30),
        impact="HIGH",
        description="US Consumer Price Index — December 2026",
        assets_affected=ALL_ASSETS,
    ),
    # ---- Non-Farm Payrolls (first Friday of each month) ----
    MacroEvent(
        name="US Non-Farm Payrolls",
        event_time=_utc(2026, 1, 9, 12, 30),
        impact="HIGH",
        description="US Non-Farm Payrolls — January 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US Non-Farm Payrolls",
        event_time=_utc(2026, 2, 6, 12, 30),
        impact="HIGH",
        description="US Non-Farm Payrolls — February 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US Non-Farm Payrolls",
        event_time=_utc(2026, 3, 6, 12, 30),
        impact="HIGH",
        description="US Non-Farm Payrolls — March 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US Non-Farm Payrolls",
        event_time=_utc(2026, 4, 3, 12, 30),
        impact="HIGH",
        description="US Non-Farm Payrolls — April 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US Non-Farm Payrolls",
        event_time=_utc(2026, 5, 1, 12, 30),
        impact="HIGH",
        description="US Non-Farm Payrolls — May 2026",
        assets_affected=ALL_ASSETS,
    ),
    MacroEvent(
        name="US Non-Farm Payrolls",
        event_time=_utc(2026, 6, 5, 12, 30),
        impact="HIGH",
        description="US Non-Farm Payrolls — June 2026",
        assets_affected=ALL_ASSETS,
    ),
    # ---- Spot Bitcoin ETF regulatory review dates ----
    MacroEvent(
        name="SEC Crypto Asset Review",
        event_time=_utc(2026, 4, 30, 20, 0),
        impact="HIGH",
        description="SEC deadline for crypto ETF product applications — Q2 2026",
        assets_affected=["BTC", "ETH"],
    ),
    MacroEvent(
        name="SEC Crypto Asset Review",
        event_time=_utc(2026, 9, 30, 20, 0),
        impact="HIGH",
        description="SEC deadline for crypto ETF product applications — Q3 2026",
        assets_affected=["BTC", "ETH"],
    ),
]


class MacroEventFilter:
    """
    Filters trading activity around high-impact macroeconomic events.

    Within ``reduce_window_minutes`` of any HIGH-impact event the position
    size multiplier is reduced to ``reduce_factor``.  Within
    ``halt_window_minutes`` the multiplier drops to 0.0 (no new entries).

    Parameters
    ----------
    custom_events : list of MacroEvent, optional
        Additional events to include alongside the hardcoded defaults.
    halt_window_minutes : int
        Minutes before/after an event during which new entries are halted.
        Default 30.
    reduce_window_minutes : int
        Minutes before/after an event during which position size is reduced.
        Default 120.
    reduce_factor : float
        Position size multiplier applied during the reduce window.
        Default 0.30 (i.e. positions shrink to 30 % of normal).
    """

    DEFAULT_EVENTS: List[MacroEvent] = _DEFAULT_EVENTS_2026

    def __init__(
        self,
        custom_events: Optional[List[MacroEvent]] = None,
        halt_window_minutes: int = 30,
        reduce_window_minutes: int = 120,
        reduce_factor: float = 0.30,
    ) -> None:
        if halt_window_minutes >= reduce_window_minutes:
            raise ValueError(
                "halt_window_minutes must be less than reduce_window_minutes"
            )
        if not (0.0 < reduce_factor < 1.0):
            raise ValueError("reduce_factor must be in (0, 1)")

        self.halt_window_minutes = halt_window_minutes
        self.reduce_window_minutes = reduce_window_minutes
        self.reduce_factor = reduce_factor

        self._events: List[MacroEvent] = list(self.DEFAULT_EVENTS)
        if custom_events:
            self._events.extend(custom_events)

        logger.info(
            "MacroEventFilter initialised with %d events (halt=%dmin reduce=%dmin factor=%.2f)",
            len(self._events),
            halt_window_minutes,
            reduce_window_minutes,
            reduce_factor,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_event(self, event: MacroEvent) -> None:
        """Append a new :class:`MacroEvent` to the calendar."""
        self._events.append(event)
        logger.debug("add_event: %r", event)

    def get_position_multiplier(self, now: Optional[datetime] = None) -> float:
        """
        Return the position size multiplier for the current moment.

        Returns
        -------
        float
            ``0.0``  — within halt window (no new entries).
            ``reduce_factor`` — within reduce window (reduced size).
            ``1.0``  — no nearby events.
        """
        now = self._normalise_now(now)

        for event in self._events:
            if event.impact != "HIGH":
                continue
            minutes = abs(event.minutes_until(now))
            if minutes <= self.halt_window_minutes:
                logger.info(
                    "MacroEventFilter HALT: %s in %.1f min",
                    event.name,
                    event.minutes_until(now),
                )
                return 0.0
            if minutes <= self.reduce_window_minutes:
                logger.info(
                    "MacroEventFilter REDUCE: %s in %.1f min → multiplier=%.2f",
                    event.name,
                    event.minutes_until(now),
                    self.reduce_factor,
                )
                return self.reduce_factor

        return 1.0

    def should_halt(self, now: Optional[datetime] = None) -> bool:
        """Return ``True`` if new entries should be completely halted."""
        return self.get_position_multiplier(now) == 0.0

    def next_event(self, now: Optional[datetime] = None) -> Optional[MacroEvent]:
        """
        Return the soonest upcoming event (regardless of impact).

        Returns ``None`` if no future events are scheduled.
        """
        now = self._normalise_now(now)
        future = [e for e in self._events if e.event_time > now]
        if not future:
            return None
        return min(future, key=lambda e: e.event_time)

    def events_in_window(
        self,
        hours: float = 24,
        now: Optional[datetime] = None,
    ) -> List[MacroEvent]:
        """
        Return all events scheduled within the next *hours* hours.

        Results are sorted by event_time ascending.
        """
        now = self._normalise_now(now)
        cutoff = now + timedelta(hours=hours)
        upcoming = [
            e for e in self._events if now <= e.event_time <= cutoff
        ]
        return sorted(upcoming, key=lambda e: e.event_time)

    def all_events(self) -> List[MacroEvent]:
        """Return a copy of the full event calendar sorted by event_time."""
        return sorted(self._events, key=lambda e: e.event_time)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_now(now: Optional[datetime]) -> datetime:
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now
