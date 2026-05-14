"""
Event-driven universe: temporarily include/exclude symbols around macro/news events.

Stub: returns empty exclude/include sets; integrate real calendar (earnings, FOMC, etc.) when available.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EventCalendar:
    """
    Optional calendar for event-driven universe changes.
    get_exclude_symbols() / get_include_symbols() return symbols to exclude or force-include
    for the current time window (e.g. around FOMC, earnings).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._events: List[Dict[str, Any]] = []  # {symbol, start_utc, end_utc, action: exclude|include}

    def add_event(self, symbol: str, start_utc: datetime, end_utc: datetime, action: str = "exclude") -> None:
        """Register an event window. action: 'exclude' or 'include'."""
        self._events.append({
            "symbol": str(symbol),
            "start_utc": start_utc,
            "end_utc": end_utc,
            "action": str(action).lower() or "exclude",
        })

    def get_exclude_symbols(self, at_utc: Optional[datetime] = None) -> Set[str]:
        """Symbols to exclude from universe in the current window."""
        now = at_utc or datetime.now(timezone.utc)
        out: Set[str] = set()
        for e in self._events:
            if e.get("action") != "exclude":
                continue
            if e["start_utc"] <= now <= e["end_utc"]:
                out.add(str(e["symbol"]))
        return out

    def get_include_symbols(self, at_utc: Optional[datetime] = None) -> Set[str]:
        """Symbols to force-include (e.g. high-importance events)."""
        now = at_utc or datetime.now(timezone.utc)
        out: Set[str] = set()
        for e in self._events:
            if e.get("action") != "include":
                continue
            if e["start_utc"] <= now <= e["end_utc"]:
                out.add(str(e["symbol"]))
        return out
