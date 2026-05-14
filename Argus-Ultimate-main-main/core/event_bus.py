#!/usr/bin/env python3
"""
Event-Driven Architecture Bus — publish/subscribe with priority ordering.

Supports both synchronous and async callbacks, priority-based dispatch,
event history for debugging, and thread-safe operation.

Standalone usage:
    bus = EventBus()
    bus.subscribe("price_update", my_handler, priority=10)
    bus.publish("price_update", {"symbol": "BTC/AUD", "price": 95000.0})
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known event types (informational — the bus accepts any string)
# ---------------------------------------------------------------------------

KNOWN_EVENT_TYPES: Set[str] = {
    "price_update",
    "signal_generated",
    "order_placed",
    "order_filled",
    "risk_breach",
    "regime_change",
    "anomaly_detected",
    "cycle_complete",
}


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Subscriber:
    """A registered callback with its priority."""

    callback: Callable
    priority: int  # higher = called first
    is_async: bool


@dataclass(slots=True)
class _EventRecord:
    """Historical record of a published event."""

    event_type: str
    data: Dict[str, Any]
    timestamp: float
    handler_count: int
    dispatch_time_ms: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EventBus:
    """
    Thread-safe publish/subscribe event bus with priority dispatch.

    Parameters
    ----------
    history_size : int
        Number of recent events to retain for debugging (default 1000).
    warn_slow_ms : float
        Log a warning if a single handler takes longer than this (default 50 ms).
    """

    def __init__(self, history_size: int = 1000, warn_slow_ms: float = 50.0):
        self.history_size = history_size
        self.warn_slow_ms = warn_slow_ms

        # event_type -> sorted list of _Subscriber (highest priority first)
        self._subscribers: Dict[str, List[_Subscriber]] = {}
        self._lock = threading.Lock()

        # Event history ring buffer
        self._history: Deque[_EventRecord] = deque(maxlen=history_size)

        # Aggregate stats
        self._total_published: int = 0
        self._total_dispatch_ms: float = 0.0

        logger.info("EventBus initialised (history_size=%d)", history_size)

    # ------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # ------------------------------------------------------------------

    def subscribe(self, event_type: str, callback: Callable, priority: int = 0) -> None:
        """
        Register a handler for *event_type*.

        Parameters
        ----------
        event_type : str
            The event name to listen for.
        callback : callable
            Sync or async function to invoke. Receives ``(event_type, data)`` args.
        priority : int
            Higher priority handlers run first (default 0).
        """
        is_async = asyncio.iscoroutinefunction(callback)
        sub = _Subscriber(callback=callback, priority=priority, is_async=is_async)

        with self._lock:
            subs = self._subscribers.setdefault(event_type, [])
            subs.append(sub)
            subs.sort(key=lambda s: s.priority, reverse=True)

        if event_type not in KNOWN_EVENT_TYPES:
            logger.debug("EventBus: subscribed to custom event type '%s'", event_type)

        logger.debug(
            "EventBus: subscribed %s to '%s' (priority=%d, async=%s)",
            callback.__name__ if hasattr(callback, "__name__") else str(callback),
            event_type, priority, is_async,
        )

    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Remove a handler for *event_type*.

        Parameters
        ----------
        event_type : str
            The event name.
        callback : callable
            The exact callback reference to remove.

        Returns
        -------
        bool
            True if the callback was found and removed.
        """
        with self._lock:
            subs = self._subscribers.get(event_type)
            if not subs:
                return False
            before = len(subs)
            self._subscribers[event_type] = [s for s in subs if s.callback is not callback]
            removed = before - len(self._subscribers[event_type])
            if removed:
                logger.debug("EventBus: unsubscribed from '%s'", event_type)
            return removed > 0

    # ------------------------------------------------------------------
    # Publish (sync)
    # ------------------------------------------------------------------

    def publish(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> int:
        """
        Fire an event synchronously to all subscribers (priority order).

        Async callbacks are skipped in sync publish — use ``publish_async`` for those.

        Parameters
        ----------
        event_type : str
            Event name.
        data : dict, optional
            Event payload.

        Returns
        -------
        int
            Number of handlers invoked.
        """
        data = data or {}
        t0 = time.monotonic()

        with self._lock:
            subs = list(self._subscribers.get(event_type, []))

        invoked = 0
        for sub in subs:
            if sub.is_async:
                # Try to schedule in running loop, otherwise skip
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(sub.callback(event_type, data))
                    invoked += 1
                except RuntimeError:
                    logger.debug(
                        "EventBus: skipping async handler %s (no running event loop)",
                        getattr(sub.callback, "__name__", "?"),
                    )
                continue

            handler_t0 = time.monotonic()
            try:
                sub.callback(event_type, data)
                invoked += 1
            except Exception:
                logger.error(
                    "EventBus: handler %s raised on '%s'",
                    getattr(sub.callback, "__name__", "?"), event_type,
                    exc_info=True,
                )
            elapsed_ms = (time.monotonic() - handler_t0) * 1000
            if elapsed_ms > self.warn_slow_ms:
                logger.warning(
                    "EventBus: slow handler %s took %.1f ms on '%s'",
                    getattr(sub.callback, "__name__", "?"), elapsed_ms, event_type,
                )

        dispatch_ms = (time.monotonic() - t0) * 1000
        self._record_event(event_type, data, invoked, dispatch_ms)
        return invoked

    # ------------------------------------------------------------------
    # Publish (async)
    # ------------------------------------------------------------------

    async def publish_async(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> int:
        """
        Fire an event asynchronously; awaits async handlers, calls sync handlers normally.

        Parameters
        ----------
        event_type : str
            Event name.
        data : dict, optional
            Event payload.

        Returns
        -------
        int
            Number of handlers invoked.
        """
        data = data or {}
        t0 = time.monotonic()

        with self._lock:
            subs = list(self._subscribers.get(event_type, []))

        invoked = 0
        for sub in subs:
            handler_t0 = time.monotonic()
            try:
                if sub.is_async:
                    await sub.callback(event_type, data)
                else:
                    sub.callback(event_type, data)
                invoked += 1
            except Exception:
                logger.error(
                    "EventBus: handler %s raised on '%s'",
                    getattr(sub.callback, "__name__", "?"), event_type,
                    exc_info=True,
                )
            elapsed_ms = (time.monotonic() - handler_t0) * 1000
            if elapsed_ms > self.warn_slow_ms:
                logger.warning(
                    "EventBus: slow handler %s took %.1f ms on '%s'",
                    getattr(sub.callback, "__name__", "?"), elapsed_ms, event_type,
                )

        dispatch_ms = (time.monotonic() - t0) * 1000
        self._record_event(event_type, data, invoked, dispatch_ms)
        return invoked

    # ------------------------------------------------------------------
    # Stats & history
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return bus statistics.

        Returns
        -------
        dict
            Keys: events_published, handlers_registered, avg_dispatch_time_ms,
            event_types, history_length.
        """
        with self._lock:
            total_handlers = sum(len(subs) for subs in self._subscribers.values())
            event_types = sorted(self._subscribers.keys())

        avg_ms = (
            self._total_dispatch_ms / self._total_published
            if self._total_published > 0
            else 0.0
        )

        return {
            "events_published": self._total_published,
            "handlers_registered": total_handlers,
            "avg_dispatch_time_ms": round(avg_ms, 3),
            "event_types": event_types,
            "history_length": len(self._history),
        }

    def get_history(self, event_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return recent event records, optionally filtered by type.

        Parameters
        ----------
        event_type : str, optional
            Filter to this event type.
        limit : int
            Maximum records to return (default 50).

        Returns
        -------
        list of dict
        """
        records = list(self._history)
        if event_type:
            records = [r for r in records if r.event_type == event_type]
        records = records[-limit:]
        return [
            {
                "event_type": r.event_type,
                "data": r.data,
                "timestamp": r.timestamp,
                "handler_count": r.handler_count,
                "dispatch_time_ms": round(r.dispatch_time_ms, 3),
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_event(
        self, event_type: str, data: Dict[str, Any], handler_count: int, dispatch_ms: float
    ) -> None:
        """Append to history ring buffer and update aggregate stats."""
        self._history.append(
            _EventRecord(
                event_type=event_type,
                data=data,
                timestamp=time.time(),
                handler_count=handler_count,
                dispatch_time_ms=dispatch_ms,
            )
        )
        self._total_published += 1
        self._total_dispatch_ms += dispatch_ms
