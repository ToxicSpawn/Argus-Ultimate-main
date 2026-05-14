"""
Distributed Tracing — lightweight span-based tracing for ARGUS internals.

Tracks request flow through the signal → risk → execution pipeline with
parent/child span relationships. Stores up to 10,000 traces in memory;
older traces are evicted FIFO.

Usage::

    from core.tracing import Tracer, traced

    tracer = Tracer()

    # Manual span management
    ctx = tracer.start_span("process_signal")
    # ... do work ...
    tracer.end_span(ctx)

    # Decorator
    @traced("risk_check")
    def check_risk(signal):
        ...

    # Retrieve full trace tree
    tree = tracer.export_traces(trace_id)
"""
from __future__ import annotations

import functools
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_TRACES = 10_000


@dataclass
class TraceContext:
    """A single span within a distributed trace."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds, or None if span is still open."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0


class Tracer:
    """
    In-process distributed tracer with parent/child span support.

    Thread-safe. Stores up to ``MAX_TRACES`` distinct trace_ids;
    evicts oldest when the cap is reached.
    """

    def __init__(self, max_traces: int = MAX_TRACES) -> None:
        self._max_traces = max_traces
        self._lock = threading.Lock()
        # OrderedDict preserves insertion order for FIFO eviction
        self._traces: OrderedDict[str, List[TraceContext]] = OrderedDict()

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    def start_span(
        self,
        operation: str,
        parent: Optional[TraceContext] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> TraceContext:
        """
        Begin a new span.

        Parameters
        ----------
        operation : str
            Human-readable name of the operation (e.g. ``"risk_check"``).
        parent : TraceContext, optional
            If provided, the new span inherits the parent's ``trace_id``.
        tags : dict, optional
            Arbitrary key/value metadata attached to the span.

        Returns
        -------
        TraceContext
            The open span context. Pass to :meth:`end_span` when done.
        """
        trace_id = parent.trace_id if parent else uuid.uuid4().hex[:16]
        span_id = uuid.uuid4().hex[:16]
        parent_span_id = parent.span_id if parent else None

        ctx = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.monotonic(),
            tags=dict(tags) if tags else {},
        )

        with self._lock:
            # Evict oldest traces if at capacity
            while len(self._traces) >= self._max_traces and trace_id not in self._traces:
                self._traces.popitem(last=False)
            self._traces.setdefault(trace_id, []).append(ctx)

        return ctx

    def end_span(self, ctx: TraceContext) -> None:
        """Close an open span by recording its end time."""
        ctx.end_time = time.monotonic()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> List[TraceContext]:
        """Return all spans for a given trace_id, or empty list if not found."""
        with self._lock:
            return list(self._traces.get(trace_id, []))

    def export_traces(self, trace_id: str) -> dict:
        """
        Export a full trace as a nested dict tree.

        Returns
        -------
        dict
            ``{"trace_id": ..., "spans": [...]}`` where each span includes
            ``children`` containing nested child spans.
        """
        spans = self.get_trace(trace_id)
        if not spans:
            return {"trace_id": trace_id, "spans": []}

        # Build span dict keyed by span_id
        span_map: Dict[str, dict] = {}
        for s in spans:
            span_map[s.span_id] = {
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "operation": s.operation,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration_ms": s.duration_ms,
                "tags": s.tags,
                "children": [],
            }

        # Build tree
        roots: List[dict] = []
        for sid, sdict in span_map.items():
            pid = sdict["parent_span_id"]
            if pid and pid in span_map:
                span_map[pid]["children"].append(sdict)
            else:
                roots.append(sdict)

        return {"trace_id": trace_id, "spans": roots}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def trace_count(self) -> int:
        with self._lock:
            return len(self._traces)

    def clear(self) -> None:
        """Remove all stored traces."""
        with self._lock:
            self._traces.clear()


# ---------------------------------------------------------------------------
# Global tracer instance (singleton)
# ---------------------------------------------------------------------------

_global_tracer: Optional[Tracer] = None
_tracer_lock = threading.Lock()


def get_tracer() -> Tracer:
    """Return the global Tracer singleton, creating it on first call."""
    global _global_tracer
    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = Tracer()
    return _global_tracer


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def traced(operation_name: str) -> Callable:
    """
    Decorator that wraps a function in a tracing span.

    Usage::

        @traced("risk_check")
        def evaluate_risk(signal):
            ...

    The span is automatically closed on return or exception.
    If the function receives a keyword argument ``_parent_span``, it is
    used as the parent context.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            parent = kwargs.pop("_parent_span", None)
            ctx = tracer.start_span(operation_name, parent=parent)
            try:
                result = fn(*args, **kwargs)
                ctx.tags["status"] = "ok"
                return result
            except Exception as exc:
                ctx.tags["status"] = "error"
                ctx.tags["error"] = str(exc)
                raise
            finally:
                tracer.end_span(ctx)

        return wrapper

    return decorator
