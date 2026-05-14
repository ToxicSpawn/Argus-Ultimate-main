"""Push 94 — RegimeHistoryBuffer: fixed-size ring-buffer for regime transitions.

Stores the last N regime transitions with full context snapshots.
Thread-safe, zero external dependencies.

Public API
----------
    buf = RegimeHistoryBuffer(maxlen=200)
    buf.record(regime="HIGH_VOL", context={...})
    buf.transitions           -> list[RegimeTransition]
    buf.latest                -> RegimeTransition | None
    buf.since(ts)             -> list[RegimeTransition]
    buf.stats()               -> RegimeHistoryStats
    buf.clear()               -> None
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class RegimeTransition:
    """Single regime transition event."""
    seq:          int            # monotonic sequence number (1-based)
    timestamp:    float          # unix epoch seconds (time.time())
    from_regime:  Optional[str]  # None for the very first record
    to_regime:    str
    duration_secs: Optional[float]  # seconds spent in from_regime; None for first
    context:      Dict[str, Any] = field(default_factory=dict)

    @property
    def iso(self) -> str:
        """ISO-8601 UTC string for the transition timestamp."""
        import datetime
        return datetime.datetime.utcfromtimestamp(self.timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq":           self.seq,
            "timestamp":     self.timestamp,
            "iso":           self.iso,
            "from_regime":   self.from_regime,
            "to_regime":     self.to_regime,
            "duration_secs": self.duration_secs,
            "context":       self.context,
        }


@dataclass
class RegimeHistoryStats:
    """Aggregate stats computed from the ring-buffer."""
    total_transitions: int
    unique_regimes:    List[str]
    regime_counts:     Dict[str, int]
    avg_duration_secs: Optional[float]   # None when < 2 transitions
    min_duration_secs: Optional[float]
    max_duration_secs: Optional[float]
    current_regime:    Optional[str]
    current_since:     Optional[float]
    current_duration_secs: Optional[float]  # time in current regime so far

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_transitions":     self.total_transitions,
            "unique_regimes":        self.unique_regimes,
            "regime_counts":         self.regime_counts,
            "avg_duration_secs":     self.avg_duration_secs,
            "min_duration_secs":     self.min_duration_secs,
            "max_duration_secs":     self.max_duration_secs,
            "current_regime":        self.current_regime,
            "current_since":         self.current_since,
            "current_duration_secs": self.current_duration_secs,
        }


class RegimeHistoryBuffer:
    """Thread-safe fixed-size ring-buffer for regime transition history.

    Parameters
    ----------
    maxlen : int
        Maximum number of transitions to keep in memory (default 200).
    """

    def __init__(self, maxlen: int = 200) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._maxlen   = maxlen
        self._buf: Deque[RegimeTransition] = deque(maxlen=maxlen)
        self._lock     = threading.Lock()
        self._seq      = 0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        regime: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[RegimeTransition]:
        """Record a regime transition.

        Only appends if `regime` differs from the last recorded regime.
        Returns the new RegimeTransition, or None if no transition occurred
        (i.e. regime unchanged).

        Parameters
        ----------
        regime    : new regime label (e.g. "HIGH_VOL", "TRENDING", "QUIET")
        context   : optional dict of scalar metrics captured at transition
        timestamp : override unix epoch; defaults to time.time()
        """
        ts = timestamp if timestamp is not None else time.time()
        ctx = context or {}

        with self._lock:
            latest = self._buf[-1] if self._buf else None

            if latest is not None and latest.to_regime == regime:
                return None  # no transition — same regime

            from_regime  = latest.to_regime if latest else None
            duration     = (ts - latest.timestamp) if latest else None
            self._seq   += 1

            transition = RegimeTransition(
                seq=self._seq,
                timestamp=ts,
                from_regime=from_regime,
                to_regime=regime,
                duration_secs=round(duration, 3) if duration is not None else None,
                context=ctx,
            )
            self._buf.append(transition)
            return transition

    def clear(self) -> None:
        """Wipe all recorded transitions and reset sequence counter."""
        with self._lock:
            self._buf.clear()
            self._seq = 0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def transitions(self) -> List[RegimeTransition]:
        """Return all buffered transitions oldest-first (snapshot copy)."""
        with self._lock:
            return list(self._buf)

    @property
    def latest(self) -> Optional[RegimeTransition]:
        """Most-recent transition, or None if buffer is empty."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def since(self, timestamp: float) -> List[RegimeTransition]:
        """Return all transitions with timestamp >= `timestamp`."""
        with self._lock:
            return [t for t in self._buf if t.timestamp >= timestamp]

    def last_n(self, n: int) -> List[RegimeTransition]:
        """Return the most recent `n` transitions."""
        with self._lock:
            items = list(self._buf)
            return items[-n:] if n < len(items) else items

    def stats(self) -> RegimeHistoryStats:
        """Compute aggregate stats over the current buffer contents."""
        with self._lock:
            items = list(self._buf)

        if not items:
            return RegimeHistoryStats(
                total_transitions=0,
                unique_regimes=[],
                regime_counts={},
                avg_duration_secs=None,
                min_duration_secs=None,
                max_duration_secs=None,
                current_regime=None,
                current_since=None,
                current_duration_secs=None,
            )

        counts: Dict[str, int] = {}
        for t in items:
            counts[t.to_regime] = counts.get(t.to_regime, 0) + 1

        durations = [t.duration_secs for t in items if t.duration_secs is not None]
        avg_dur = round(sum(durations) / len(durations), 3) if durations else None
        min_dur = round(min(durations), 3) if durations else None
        max_dur = round(max(durations), 3) if durations else None

        latest      = items[-1]
        current_dur = round(time.time() - latest.timestamp, 3)

        return RegimeHistoryStats(
            total_transitions=len(items),
            unique_regimes=sorted(counts.keys()),
            regime_counts=counts,
            avg_duration_secs=avg_dur,
            min_duration_secs=min_dur,
            max_duration_secs=max_dur,
            current_regime=latest.to_regime,
            current_since=latest.timestamp,
            current_duration_secs=current_dur,
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def __repr__(self) -> str:
        return f"RegimeHistoryBuffer(len={len(self)}, maxlen={self._maxlen})"
