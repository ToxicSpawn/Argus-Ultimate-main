"""
SignalMemory — short-lived queue for valid signals that arrived at WAIT readiness.

When a strong signal arrives but ARGUS is in WAIT state (conditions poor but
improving), the signal is queued here rather than discarded. Each cycle, the
queue is checked: if conditions have improved to READY or PRIME, the queued
signals are promoted back into the execution pipeline.

Prevents valid signals from being lost just because system health momentarily dipped.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.moment_readiness import ReadinessLabel, ReadinessState

logger = logging.getLogger(__name__)


@dataclass
class QueuedSignal:
    """A signal deferred for later re-evaluation."""

    signal: Any                         # Original trading signal object
    queued_at_cycle: int
    expires_at_cycle: int               # queued_at + max_queue_cycles
    strategy_name: str
    original_readiness_score: float     # Score at time of queueing
    min_readiness: str = "READY"        # Minimum label required to promote
    queued_at_ts: float = field(default_factory=time.time)


class SignalMemory:
    """
    FIFO queue that holds deferred signals for up to max_queue_cycles.

    Signals are promoted (returned for execution) when readiness reaches
    READY or PRIME. PRIME conditions also promote signals that were queued
    with a high original score even if their min_readiness is CAUTIOUS.
    """

    def __init__(
        self,
        max_queue_cycles: int = 5,
        max_queued: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = (config or {}).get("self_directed_trading") or {}
        self.max_queue_cycles = int(cfg.get("sm_max_queue_cycles", max_queue_cycles))
        self.max_queued = int(cfg.get("sm_max_queued", max_queued))

        self._queue: deque[QueuedSignal] = deque()

        # Stats
        self._total_queued = 0
        self._total_promoted = 0
        self._total_expired = 0
        self._promoted_this_cycle = 0
        self._expired_this_cycle = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def queue(
        self,
        signal: Any,
        cycle: int,
        strategy_name: str,
        readiness_score: float,
    ) -> bool:
        """
        Add a signal to the queue.

        Returns True if queued, False if rejected (queue full).
        The oldest signal is NOT dropped to make room — we reject the new one.
        FIFO: first-queued is first-promoted.
        """
        if len(self._queue) >= self.max_queued:
            logger.debug(
                "SignalMemory: queue full (%d/%d) — rejecting signal from %s",
                len(self._queue),
                self.max_queued,
                strategy_name,
            )
            return False

        qs = QueuedSignal(
            signal=signal,
            queued_at_cycle=cycle,
            expires_at_cycle=cycle + self.max_queue_cycles,
            strategy_name=strategy_name,
            original_readiness_score=readiness_score,
        )
        self._queue.append(qs)
        self._total_queued += 1
        logger.debug(
            "SignalMemory: queued signal from %s (cycle=%d, score=%.1f, expires=%d)",
            strategy_name,
            cycle,
            readiness_score,
            qs.expires_at_cycle,
        )
        return True

    def pop_promotable(
        self,
        current_readiness: "ReadinessState",
        current_cycle: int,
    ) -> List[Any]:
        """
        Return all queued signals that should be promoted given current conditions.

        Promotion criteria:
        - current_readiness.label is PRIME or READY, AND signal not expired, OR
        - current_readiness.label is CAUTIOUS AND original_readiness_score > 70
          (strong signal queued during bad timing — promote cautiously)

        Promoted signals are removed from the queue.
        """
        self._promoted_this_cycle = 0

        _raw_label = getattr(current_readiness, "label", "WAIT")
        # Support both ReadinessLabel enum and plain strings
        label = str(_raw_label.value if hasattr(_raw_label, "value") else _raw_label)
        if label not in ("PRIME", "READY", "CAUTIOUS"):
            return []

        promotable: List[QueuedSignal] = []
        remaining: deque[QueuedSignal] = deque()

        for qs in self._queue:
            if current_cycle >= qs.expires_at_cycle:
                # Expired — will be handled by expire_stale; skip here
                remaining.append(qs)
                continue

            should_promote = False
            if label in ("PRIME", "READY"):
                should_promote = True
            elif label == "CAUTIOUS" and qs.original_readiness_score > 70.0:
                # Strong signal that arrived at bad timing — promote cautiously
                should_promote = True

            if should_promote:
                promotable.append(qs)
                self._total_promoted += 1
                self._promoted_this_cycle += 1
                logger.debug(
                    "SignalMemory: promoting signal from %s (queued=%d, score=%.1f)",
                    qs.strategy_name,
                    qs.queued_at_cycle,
                    qs.original_readiness_score,
                )
            else:
                remaining.append(qs)

        self._queue = remaining
        return [qs.signal for qs in promotable]

    def expire_stale(self, current_cycle: int) -> int:
        """
        Remove signals that have exceeded their TTL.

        Returns count of expired signals removed.
        """
        self._expired_this_cycle = 0
        remaining: deque[QueuedSignal] = deque()

        for qs in self._queue:
            if current_cycle >= qs.expires_at_cycle:
                self._expired_this_cycle += 1
                self._total_expired += 1
                logger.debug(
                    "SignalMemory: expiring signal from %s (queued=%d, expired=%d)",
                    qs.strategy_name,
                    qs.queued_at_cycle,
                    qs.expires_at_cycle,
                )
            else:
                remaining.append(qs)

        removed = self._expired_this_cycle
        if removed:
            logger.info("SignalMemory: expired %d stale signal(s) this cycle", removed)
        self._queue = remaining
        return removed

    def snapshot(self) -> Dict[str, Any]:
        """Return a summary dict suitable for the advisory."""
        return {
            "queued": len(self._queue),
            "promoted_this_cycle": self._promoted_this_cycle,
            "expired_this_cycle": self._expired_this_cycle,
            "total_queued_lifetime": self._total_queued,
            "total_promoted_lifetime": self._total_promoted,
            "total_expired_lifetime": self._total_expired,
            "max_queued": self.max_queued,
            "max_queue_cycles": self.max_queue_cycles,
        }

    def __len__(self) -> int:
        return len(self._queue)

    def __repr__(self) -> str:
        return (
            f"SignalMemory(queued={len(self._queue)}, "
            f"total_promoted={self._total_promoted}, "
            f"total_expired={self._total_expired})"
        )
