"""
Distributed tracing stub (OpenTelemetry-style).

Propagate trace context across components; no-op if OpenTelemetry not installed.
Use for: request/cycle correlation, latency spans across data → brain → execution.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)


def get_trace_id() -> Optional[str]:
    return _trace_id.get()


def get_span_id() -> Optional[str]:
    return _span_id.get()


def set_trace_context(trace_id: Optional[str] = None, span_id: Optional[str] = None) -> None:
    """Set current trace/span for this context."""
    _trace_id.set(trace_id or str(uuid.uuid4()))
    _span_id.set(span_id or str(uuid.uuid4())[:16])


def clear_trace_context() -> None:
    _trace_id.set(None)
    _span_id.set(None)


def current_context() -> Dict[str, str]:
    """Return dict suitable for logging or propagation (e.g. correlation_id)."""
    tid = get_trace_id()
    sid = get_span_id()
    out: Dict[str, str] = {}
    if tid:
        out["trace_id"] = tid
    if sid:
        out["span_id"] = sid
    return out


def start_span(name: str, **attrs: Any) -> "Span":
    """Start a span (stub). Use with 'with start_span(...):' if Span supports it."""
    try:
        from opentelemetry import trace
        return trace.get_current_span()
    except ImportError:
        return _NoopSpan()


class _NoopSpan:
    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass
