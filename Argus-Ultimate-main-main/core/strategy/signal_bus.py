"""Push 75 — AsyncSignalBus: async pub/sub signal dispatcher.

Connects strategy tick() outputs to order manager handlers.

Features:
  - subscribe(handler, strategy_ids=None, symbols=None)
  - publish(signal) — fan-out to matching subscribers
  - async handlers supported (sync handlers are wrapped)
  - Dead-letter queue for failed handlers
  - Ring-buffer signal history (capacity 1000)
  - Stats: published, delivered, dropped, dlq_size
"""
from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from core.strategy.signal import Signal


@dataclass
class Subscription:
    handler:      Callable
    strategy_ids: Optional[Set[str]] = None   # None = all
    symbols:      Optional[Set[str]] = None   # None = all
    sub_id:       str = ""

    def matches(self, signal: Signal) -> bool:
        if self.strategy_ids and signal.strategy_id not in self.strategy_ids:
            return False
        if self.symbols and signal.symbol not in self.symbols:
            return False
        return True


@dataclass
class DeadLetterEntry:
    signal:    Signal
    handler:   str
    error:     str
    failed_at: float = field(default_factory=time.time)


class AsyncSignalBus:
    """Async pub/sub bus for trading signals.

    Usage:
        bus = AsyncSignalBus()
        bus.subscribe(my_handler, symbols={"BTCUSDT"})
        await bus.publish(signal)
    """

    def __init__(self, history_capacity: int = 1000):
        self._subscriptions: List[Subscription] = []
        self._history: Deque[Signal] = deque(maxlen=history_capacity)
        self._dlq:    List[DeadLetterEntry] = []
        self._published:  int = 0
        self._delivered:  int = 0
        self._dropped:    int = 0
        self._sub_counter: int = 0

    def subscribe(
        self,
        handler: Callable,
        strategy_ids: Optional[Set[str]] = None,
        symbols: Optional[Set[str]] = None,
    ) -> str:
        """Register a handler. Returns sub_id for later unsubscribe."""
        self._sub_counter += 1
        sub_id = f"sub_{self._sub_counter}"
        self._subscriptions.append(Subscription(
            handler=handler,
            strategy_ids=strategy_ids,
            symbols=symbols,
            sub_id=sub_id,
        ))
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        before = len(self._subscriptions)
        self._subscriptions = [
            s for s in self._subscriptions if s.sub_id != sub_id
        ]
        return len(self._subscriptions) < before

    async def publish(self, signal: Signal) -> int:
        """Publish signal to all matching subscribers. Returns delivery count."""
        self._published += 1
        self._history.append(signal)
        delivered = 0
        for sub in self._subscriptions:
            if not sub.matches(signal):
                continue
            try:
                result = sub.handler(signal)
                if inspect.isawaitable(result):
                    await result
                delivered += 1
                self._delivered += 1
            except Exception as e:
                self._dropped += 1
                self._dlq.append(DeadLetterEntry(
                    signal=signal,
                    handler=repr(sub.handler),
                    error=str(e)[:200],
                ))
        return delivered

    def publish_sync(self, signal: Signal) -> None:
        """Fire-and-forget sync wrapper (creates asyncio task)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.publish(signal))
            else:
                loop.run_until_complete(self.publish(signal))
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[Signal]:
        return list(self._history)

    @property
    def dlq(self) -> List[DeadLetterEntry]:
        return self._dlq

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "published":    self._published,
            "delivered":    self._delivered,
            "dropped":      self._dropped,
            "dlq_size":     len(self._dlq),
            "subscribers":  len(self._subscriptions),
        }

    def clear_dlq(self) -> int:
        n = len(self._dlq)
        self._dlq.clear()
        return n

    def __len__(self) -> int:
        return len(self._subscriptions)
