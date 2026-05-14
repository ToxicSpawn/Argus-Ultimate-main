"""Push 95 — RegimeHistoryWiring: zero-intrusion adapter to feed RegimeHistoryBuffer
from any RegimeDetector without modifying detector code.

Usage
-----
    from core.regime_history_wiring import wire_regime_history

    buf = RegimeHistoryBuffer(maxlen=500)
    wire_regime_history(app_context, regime_detector=det, buf=buf)
    # AppContext.regime_history is now set and fed on every transition.

How it works
------------
1.  `wire_regime_history` sets `app_context.regime_history = buf`.
2.  It wraps `regime_detector.on_transition` (if present) OR patches
    `regime_detector.snapshot` to detect changes on each poll.
3.  Alternatively, `RegimeTransitionCallback` can be registered directly
    with any object that accepts a callback list.

All patterns are zero-copy and thread-safe (the buffer is already locked).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

from core.regime_history_buffer import RegimeHistoryBuffer


# ---------------------------------------------------------------------------
# Core wiring helper
# ---------------------------------------------------------------------------

def wire_regime_history(
    app_context: Any,
    regime_detector: Any,
    buf: Optional[RegimeHistoryBuffer] = None,
    maxlen: int = 500,
    context_extractor: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> RegimeHistoryBuffer:
    """Wire a RegimeHistoryBuffer into an AppContext + RegimeDetector.

    Tries three attachment strategies in order:
      1. `regime_detector.register_transition_callback(cb)` — explicit hook
      2. `regime_detector.on_transition` attribute — assign callable
      3. Snapshot-poll fallback via `SnapshotPoller` (starts a daemon thread)

    Parameters
    ----------
    app_context      : AppContext instance to attach the buffer to
    regime_detector  : Any object with a `.snapshot()` method or transition hook
    buf              : existing RegimeHistoryBuffer; created if None
    maxlen           : buffer size if buf is None (default 500)
    context_extractor: optional fn(snapshot) -> dict for enriching transition context

    Returns
    -------
    RegimeHistoryBuffer
        The (possibly newly created) buffer, already wired.
    """
    if buf is None:
        buf = RegimeHistoryBuffer(maxlen=maxlen)

    app_context.regime_history = buf

    def _callback(regime: str, context: Optional[Dict[str, Any]] = None) -> None:
        buf.record(regime=regime, context=context or {})

    # Strategy 1 — explicit callback registry
    if hasattr(regime_detector, "register_transition_callback"):
        regime_detector.register_transition_callback(_callback)
        return buf

    # Strategy 2 — direct on_transition attribute
    if hasattr(regime_detector, "on_transition"):
        _prev = getattr(regime_detector, "on_transition", None)

        def _chained(regime: str, context: Optional[Dict[str, Any]] = None) -> None:
            _callback(regime, context)
            if callable(_prev):
                _prev(regime, context)

        regime_detector.on_transition = _chained
        return buf

    # Strategy 3 — snapshot-poll fallback
    _extractor = context_extractor or _default_context_extractor
    poller = SnapshotPoller(
        detector=regime_detector,
        buf=buf,
        context_extractor=_extractor,
    )
    poller.start()
    # Store reference on detector so it isn't GC'd
    regime_detector._history_poller = poller
    return buf


def _default_context_extractor(snapshot: Any) -> Dict[str, Any]:
    """Extract scalar metrics from a snapshot object or dict."""
    if snapshot is None:
        return {}
    if isinstance(snapshot, dict):
        keys = ("vol_ratio", "trend_score", "bb_pos", "autocorr", "confidence", "tick_count")
        return {k: snapshot[k] for k in keys if k in snapshot}
    ctx: Dict[str, Any] = {}
    for attr in ("vol_ratio", "trend_score", "bb_pos", "autocorr", "confidence", "tick_count"):
        val = getattr(snapshot, attr, None)
        if val is not None:
            ctx[attr] = val
    return ctx


# ---------------------------------------------------------------------------
# Snapshot-poll fallback
# ---------------------------------------------------------------------------

class SnapshotPoller:
    """Daemon thread that polls detector.snapshot() and records regime changes.

    Used when the detector has no explicit transition hook.
    Poll interval is 250 ms — sufficient for regime-level events.
    """

    def __init__(
        self,
        detector: Any,
        buf: RegimeHistoryBuffer,
        context_extractor: Callable[[Any], Dict[str, Any]],
        poll_interval: float = 0.25,
    ) -> None:
        self._detector        = detector
        self._buf             = buf
        self._extractor       = context_extractor
        self._poll_interval   = poll_interval
        self._last_regime: Optional[str] = None
        self._thread          = threading.Thread(
            target=self._run, daemon=True, name="SnapshotPoller"
        )
        self._stop_event      = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self._detector.snapshot()
                if snap is None:
                    pass
                else:
                    regime = self._extract_regime(snap)
                    if regime and regime != self._last_regime:
                        ctx = self._extractor(snap)
                        self._buf.record(regime=regime, context=ctx)
                        self._last_regime = regime
            except Exception:
                pass  # never crash the poller
            self._stop_event.wait(timeout=self._poll_interval)

    @staticmethod
    def _extract_regime(snap: Any) -> Optional[str]:
        if isinstance(snap, dict):
            r = snap.get("regime")
            return r.value if hasattr(r, "value") else str(r) if r else None
        r = getattr(snap, "regime", None)
        if r is None:
            return None
        return r.value if hasattr(r, "value") else str(r)


# ---------------------------------------------------------------------------
# Standalone callback adapter
# ---------------------------------------------------------------------------

class RegimeTransitionCallback:
    """Lightweight adapter for detectors that accept a callback list.

    Example
    -------
        cb = RegimeTransitionCallback(buf)
        regime_detector.transition_callbacks.append(cb)
        # Now every regime change is recorded in buf.
    """

    def __init__(
        self,
        buf: RegimeHistoryBuffer,
        context_extractor: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ) -> None:
        self._buf       = buf
        self._extractor = context_extractor or _default_context_extractor

    def __call__(self, regime: str, context: Optional[Dict[str, Any]] = None) -> None:
        self._buf.record(regime=regime, context=context or {})

    def from_snapshot(self, snap: Any) -> None:
        """Alternative entry point — extract regime + context from a snapshot."""
        regime = SnapshotPoller._extract_regime(snap)
        if regime:
            ctx = self._extractor(snap)
            self._buf.record(regime=regime, context=ctx)
