"""
Enhanced distributed tracing for the Argus trading system.

This module provides in-process span tracking with optional OpenTelemetry
integration, async-safe context propagation, span decorators, trace sampling,
and exporters compatible with Jaeger and Zipkin style payloads.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import random
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, MutableMapping, Optional, TypeVar, cast

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
except ImportError:  # pragma: no cover - optional dependency
    otel_trace = None
    SpanKind = None
    TraceContextTextMapPropagator = None

MAX_TRACES = 10_000
DEFAULT_SAMPLE_RATE = 1.0
_UTC = timezone.utc
F = TypeVar("F", bound=Callable[..., Any])

_current_trace_id: ContextVar[Optional[str]] = ContextVar("argus_trace_id", default=None)
_current_span_id: ContextVar[Optional[str]] = ContextVar("argus_span_id", default=None)
_current_span: ContextVar[Optional["TraceSpan"]] = ContextVar("argus_current_span", default=None)


@dataclass
class TraceSpan:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"


class _SpanScope:
    def __init__(
        self,
        tracer: "ArgusTracer",
        operation_name: str,
        parent_context: Optional[Any] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        status_on_error: str = "error",
    ) -> None:
        self._tracer = tracer
        self._operation_name = operation_name
        self._parent_context = parent_context
        self._attributes = attributes or {}
        self._status_on_error = status_on_error
        self._span: Optional[TraceSpan] = None

    def __enter__(self) -> TraceSpan:
        self._span = self._tracer.start_span(self._operation_name, parent_context=self._parent_context)
        for key, value in self._attributes.items():
            self._tracer.set_span_attribute(self._span, key, value)
        return self._span

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> None:
        if self._span is None:
            return
        if exc is not None:
            self._tracer.add_span_event(
                self._span,
                "exception",
                {"type": getattr(exc_type, "__name__", "Exception"), "message": str(exc)},
            )
            self._tracer.end_span(self._span, status=self._status_on_error)
            return
        self._tracer.end_span(self._span)


class ArgusTracer:
    """Enhanced tracer with optional OpenTelemetry integration."""

    def __init__(
        self,
        service_name: str = "argus-trading-system",
        max_traces: int = MAX_TRACES,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        high_volume_sample_rates: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.service_name = service_name
        self._max_traces = max_traces
        self._default_sample_rate = self._normalize_sample_rate(sample_rate)
        self._high_volume_sample_rates = {
            key: self._normalize_sample_rate(value)
            for key, value in (high_volume_sample_rates or {
                "trade.execution": 1.0,
                "signal.generation": 0.25,
                "risk.check": 0.5,
                "ml.inference": 0.35,
            }).items()
        }
        self._lock = threading.RLock()
        self._traces: "OrderedDict[str, List[TraceSpan]]" = OrderedDict()
        self._active_spans: Dict[str, TraceSpan] = {}
        self._otel_spans: Dict[str, Any] = {}
        self._otel_context_managers: Dict[str, Any] = {}
        self._context_tokens: Dict[str, tuple[Token[Any], Token[Any], Token[Any]]] = {}
        self._sampled_spans: Dict[str, bool] = {}
        self._span_start_perf: Dict[str, float] = {}
        self._latency_by_component: Dict[str, List[float]] = defaultdict(list)
        self._latency_limit = 10_000
        self._otel_tracer = otel_trace.get_tracer(service_name) if otel_trace is not None else None
        self._propagator = TraceContextTextMapPropagator() if TraceContextTextMapPropagator is not None else None

    def current_span(self) -> Optional[TraceSpan]:
        return _current_span.get()

    def current_context(self) -> Dict[str, str]:
        span = self.current_span()
        if span is None:
            return {}
        return {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id or "",
            "service_name": span.service_name,
            "operation_name": span.operation_name,
        }

    def attach_context(self, context: Optional[Mapping[str, Any]]) -> Optional[TraceSpan]:
        if not context:
            return None
        trace_id = self._stringify(context.get("trace_id"))
        span_id = self._stringify(context.get("span_id"))
        if not trace_id or not span_id:
            return None
        parent_span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=self._stringify(context.get("parent_span_id")) or None,
            operation_name=self._stringify(context.get("operation_name")) or "propagated-context",
            service_name=self._stringify(context.get("service_name")) or self.service_name,
            start_time=datetime.now(tz=_UTC),
            end_time=datetime.now(tz=_UTC),
            duration_ms=0.0,
            tags={},
            logs=[],
            status="ok",
        )
        _current_trace_id.set(parent_span.trace_id)
        _current_span_id.set(parent_span.span_id)
        _current_span.set(parent_span)
        return parent_span

    def inject_context(self, carrier: Optional[MutableMapping[str, str]] = None) -> Dict[str, str]:
        output: Dict[str, str] = dict(carrier or {})
        output.update({k: v for k, v in self.current_context().items() if v})
        if self._propagator is not None and otel_trace is not None:
            try:
                self._propagator.inject(output)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("OpenTelemetry context injection failed: %s", exc)
        return output

    def span(
        self,
        operation_name: str,
        parent_context: Optional[Any] = None,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> _SpanScope:
        return _SpanScope(self, operation_name, parent_context=parent_context, attributes=attributes)

    def start_span(self, operation_name: str, parent_context: Optional[Any] = None) -> TraceSpan:
        parent_span = self._resolve_parent_context(parent_context)
        trace_id = parent_span.trace_id if parent_span else self._generate_id(32)
        span_id = self._generate_id(16)
        sampled = self._should_sample(operation_name, parent_span)
        service_name = self._service_name_for_operation(operation_name)
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span.span_id if parent_span else None,
            operation_name=operation_name,
            service_name=service_name,
            start_time=datetime.now(tz=_UTC),
            tags={
                "service.name": service_name,
                "component": service_name,
                "sampled": str(sampled).lower(),
            },
            logs=[],
            status="ok",
        )
        tokens = (
            _current_trace_id.set(trace_id),
            _current_span_id.set(span_id),
            _current_span.set(span),
        )
        with self._lock:
            self._active_spans[span_id] = span
            self._context_tokens[span_id] = tokens
            self._sampled_spans[span_id] = sampled
            self._span_start_perf[span_id] = time.perf_counter()
            if sampled:
                self._append_span(span)
        self._start_otel_span(span, parent_span)
        logger.debug(
            "Started trace span operation=%s trace_id=%s span_id=%s parent_span_id=%s sampled=%s",
            operation_name,
            trace_id,
            span_id,
            span.parent_span_id,
            sampled,
        )
        return span

    def end_span(self, span: TraceSpan, status: str = "ok") -> None:
        now = datetime.now(tz=_UTC)
        span.end_time = now
        with self._lock:
            perf_start = self._span_start_perf.pop(span.span_id, None)
        if perf_start is None:
            span.duration_ms = max((now - span.start_time).total_seconds() * 1000.0, 0.0)
        else:
            span.duration_ms = max((time.perf_counter() - perf_start) * 1000.0, 0.0)
        span.status = status
        span.tags["status"] = status
        self._record_latency(span)
        self._end_otel_span(span, status=status)
        with self._lock:
            self._active_spans.pop(span.span_id, None)
            self._sampled_spans.pop(span.span_id, None)
            tokens = self._context_tokens.pop(span.span_id, None)
        if tokens is not None:
            self._reset_context(tokens)
        logger.debug(
            "Ended trace span operation=%s trace_id=%s span_id=%s duration_ms=%.3f status=%s",
            span.operation_name,
            span.trace_id,
            span.span_id,
            span.duration_ms,
            status,
        )

    def add_span_event(self, span: TraceSpan, name: str, attributes: dict) -> None:
        event = {
            "timestamp": datetime.now(tz=_UTC).isoformat(),
            "name": name,
            "attributes": self._stringify_dict(attributes or {}),
        }
        span.logs.append(event)
        otel_span = self._otel_spans.get(span.span_id)
        if otel_span is not None:
            try:
                otel_span.add_event(name, event["attributes"])
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("OpenTelemetry add_event failed: %s", exc)

    def set_span_attribute(self, span: TraceSpan, key: str, value: str) -> None:
        text_value = self._stringify(value)
        span.tags[str(key)] = text_value
        otel_span = self._otel_spans.get(span.span_id)
        if otel_span is not None:
            try:
                otel_span.set_attribute(str(key), text_value)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("OpenTelemetry set_attribute failed: %s", exc)

    def trace_trade_execution(self, trade_data: dict) -> TraceSpan:
        span = self.start_span("trade.execution")
        for key, value in trade_data.items():
            self.set_span_attribute(span, f"trade.{key}", value)
        self.add_span_event(span, "trade.execution.received", trade_data)
        return span

    def trace_signal_generation(self, signal_data: dict) -> TraceSpan:
        span = self.start_span("signal.generation")
        for key, value in signal_data.items():
            self.set_span_attribute(span, f"signal.{key}", value)
        self.add_span_event(span, "signal.generation.started", signal_data)
        return span

    def trace_risk_check(self, risk_data: dict) -> TraceSpan:
        span = self.start_span("risk.check")
        for key, value in risk_data.items():
            self.set_span_attribute(span, f"risk.{key}", value)
        self.add_span_event(span, "risk.check.started", risk_data)
        return span

    def trace_ml_inference(self, inference_data: dict) -> TraceSpan:
        span = self.start_span("ml.inference")
        for key, value in inference_data.items():
            self.set_span_attribute(span, f"ml.{key}", value)
        self.add_span_event(span, "ml.inference.started", inference_data)
        return span

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        with self._lock:
            return sorted(
                [self._clone_span(span) for span in self._traces.get(trace_id, [])],
                key=lambda item: item.start_time,
            )

    def export_traces(self, format: str = "json") -> str:
        trace_payload = self._exportable_traces()
        latency_payload = self._latency_summary()
        normalized = format.lower()
        if normalized == "json":
            return json.dumps(
                {
                    "service_name": self.service_name,
                    "trace_count": len(trace_payload),
                    "latency_attribution": latency_payload,
                    "traces": trace_payload,
                },
                indent=2,
                sort_keys=True,
            )
        if normalized == "jaeger":
            return json.dumps(
                {
                    "data": [self._to_jaeger_trace(trace) for trace in trace_payload],
                    "total": len(trace_payload),
                },
                indent=2,
                sort_keys=True,
            )
        if normalized == "zipkin":
            spans: List[Dict[str, Any]] = []
            for trace in trace_payload:
                spans.extend(self._to_zipkin_spans(trace))
            return json.dumps(spans, indent=2, sort_keys=True)
        raise ValueError(f"Unsupported export format: {format}")

    def traced(
        self,
        operation_name: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            op_name = operation_name or fn.__name__

            if hasattr(fn, "__call__") and _is_async_callable(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.span(op_name, attributes=attributes) as span:
                        if attributes:
                            self.add_span_event(span, "function.decorated", dict(attributes))
                        return await cast(Callable[..., Awaitable[Any]], fn)(*args, **kwargs)

                return cast(F, async_wrapper)

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(op_name, attributes=attributes) as span:
                    if attributes:
                        self.add_span_event(span, "function.decorated", dict(attributes))
                    return fn(*args, **kwargs)

            return cast(F, wrapper)

        return decorator

    def _append_span(self, span: TraceSpan) -> None:
        while len(self._traces) >= self._max_traces and span.trace_id not in self._traces:
            self._traces.popitem(last=False)
        self._traces.setdefault(span.trace_id, []).append(span)
        self._traces.move_to_end(span.trace_id)

    def _clone_span(self, span: TraceSpan) -> TraceSpan:
        return TraceSpan(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            operation_name=span.operation_name,
            service_name=span.service_name,
            start_time=span.start_time,
            end_time=span.end_time,
            duration_ms=span.duration_ms,
            tags=dict(span.tags),
            logs=[dict(item) for item in span.logs],
            status=span.status,
        )

    def _resolve_parent_context(self, parent_context: Optional[Any]) -> Optional[TraceSpan]:
        if isinstance(parent_context, TraceSpan):
            return parent_context
        if isinstance(parent_context, Mapping):
            trace_id = self._stringify(parent_context.get("trace_id"))
            span_id = self._stringify(parent_context.get("span_id"))
            if trace_id and span_id:
                return TraceSpan(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=self._stringify(parent_context.get("parent_span_id")) or None,
                    operation_name=self._stringify(parent_context.get("operation_name")) or "propagated-context",
                    service_name=self._stringify(parent_context.get("service_name")) or self.service_name,
                    start_time=datetime.now(tz=_UTC),
                    end_time=datetime.now(tz=_UTC),
                    duration_ms=0.0,
                    tags={},
                    logs=[],
                    status="ok",
                )
        return _current_span.get()

    def _record_latency(self, span: TraceSpan) -> None:
        component = span.tags.get("component") or span.service_name
        values = self._latency_by_component[component]
        values.append(span.duration_ms)
        if len(values) > self._latency_limit:
            del values[0 : len(values) - self._latency_limit]

    def _latency_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for component, values in self._latency_by_component.items():
            if not values:
                continue
            ordered = sorted(values)
            summary[component] = {
                "count": float(len(ordered)),
                "avg_ms": round(sum(ordered) / len(ordered), 6),
                "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 6),
                "p99_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))], 6),
            }
        return summary

    def _exportable_traces(self) -> List[Dict[str, Any]]:
        with self._lock:
            trace_ids = list(self._traces.keys())
        output: List[Dict[str, Any]] = []
        for trace_id in trace_ids:
            spans = [self._span_to_dict(span) for span in self.get_trace(trace_id)]
            if spans:
                output.append({"trace_id": trace_id, "spans": spans})
        return output

    def _span_to_dict(self, span: TraceSpan) -> Dict[str, Any]:
        span_dict = asdict(span)
        span_dict["start_time"] = span.start_time.isoformat()
        span_dict["end_time"] = span.end_time.isoformat() if span.end_time else None
        return span_dict

    def _to_jaeger_trace(self, trace_payload: Dict[str, Any]) -> Dict[str, Any]:
        processes: Dict[str, Dict[str, Any]] = {}
        jaeger_spans: List[Dict[str, Any]] = []
        for span in trace_payload["spans"]:
            service_name = span["service_name"]
            process_id = f"process-{service_name}"
            if process_id not in processes:
                processes[process_id] = {
                    "serviceName": service_name,
                    "tags": [{"key": "service.name", "type": "string", "value": service_name}],
                }
            jaeger_spans.append(
                {
                    "traceID": span["trace_id"],
                    "spanID": span["span_id"],
                    "operationName": span["operation_name"],
                    "references": ([{
                        "refType": "CHILD_OF",
                        "traceID": span["trace_id"],
                        "spanID": span["parent_span_id"],
                    }] if span["parent_span_id"] else []),
                    "startTime": self._iso_to_epoch_microseconds(span["start_time"]),
                    "duration": int(float(span.get("duration_ms", 0.0)) * 1000.0),
                    "tags": [
                        {"key": key, "type": "string", "value": value}
                        for key, value in span.get("tags", {}).items()
                    ],
                    "logs": [
                        {
                            "timestamp": self._iso_to_epoch_microseconds(log["timestamp"]),
                            "fields": [
                                {"key": attr_key, "type": "string", "value": attr_value}
                                for attr_key, attr_value in log.get("attributes", {}).items()
                            ] + [{"key": "event", "type": "string", "value": log.get("name", "")}],
                        }
                        for log in span.get("logs", [])
                    ],
                    "processID": process_id,
                    "warnings": None,
                }
            )
        return {
            "traceID": trace_payload["trace_id"],
            "spans": jaeger_spans,
            "processes": processes,
        }

    def _to_zipkin_spans(self, trace_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for span in trace_payload["spans"]:
            output.append(
                {
                    "traceId": span["trace_id"],
                    "id": span["span_id"],
                    "parentId": span["parent_span_id"],
                    "name": span["operation_name"],
                    "timestamp": self._iso_to_epoch_microseconds(span["start_time"]),
                    "duration": int(float(span.get("duration_ms", 0.0)) * 1000.0),
                    "kind": "INTERNAL",
                    "localEndpoint": {"serviceName": span["service_name"]},
                    "tags": span.get("tags", {}),
                    "annotations": [
                        {
                            "timestamp": self._iso_to_epoch_microseconds(log["timestamp"]),
                            "value": log.get("name", "event"),
                        }
                        for log in span.get("logs", [])
                    ],
                }
            )
        return output

    def _start_otel_span(self, span: TraceSpan, parent_span: Optional[TraceSpan]) -> None:
        if self._otel_tracer is None or otel_trace is None:
            return
        try:
            attributes = dict(span.tags)
            attributes.update(
                {
                    "argus.trace_id": span.trace_id,
                    "argus.span_id": span.span_id,
                    "argus.parent_span_id": span.parent_span_id or "",
                }
            )
            if parent_span is not None and parent_span.span_id in self._otel_spans:
                parent_otel_span = self._otel_spans[parent_span.span_id]
                context = otel_trace.set_span_in_context(parent_otel_span)
                manager = self._otel_tracer.start_as_current_span(
                    span.operation_name,
                    context=context,
                    kind=SpanKind.INTERNAL if SpanKind is not None else None,
                    attributes=attributes,
                )
            else:
                manager = self._otel_tracer.start_as_current_span(
                    span.operation_name,
                    kind=SpanKind.INTERNAL if SpanKind is not None else None,
                    attributes=attributes,
                )
            otel_span = manager.__enter__()
            self._otel_context_managers[span.span_id] = manager
            self._otel_spans[span.span_id] = otel_span
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("OpenTelemetry span start failed: %s", exc)

    def _end_otel_span(self, span: TraceSpan, status: str) -> None:
        otel_span = self._otel_spans.pop(span.span_id, None)
        manager = self._otel_context_managers.pop(span.span_id, None)
        if otel_span is not None:
            try:
                otel_span.set_attribute("argus.status", status)
                otel_span.set_attribute("argus.duration_ms", span.duration_ms)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("OpenTelemetry attribute finalization failed: %s", exc)
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("OpenTelemetry span end failed: %s", exc)

    def _should_sample(self, operation_name: str, parent_span: Optional[TraceSpan]) -> bool:
        if parent_span is not None:
            return parent_span.tags.get("sampled", "true").lower() == "true"
        sample_rate = self._high_volume_sample_rates.get(operation_name, self._default_sample_rate)
        if sample_rate >= 1.0:
            return True
        if sample_rate <= 0.0:
            return False
        return random.random() <= sample_rate

    def _service_name_for_operation(self, operation_name: str) -> str:
        if operation_name.startswith("trade."):
            return "trade-execution"
        if operation_name.startswith("signal."):
            return "signal-generation"
        if operation_name.startswith("risk."):
            return "risk-management"
        if operation_name.startswith("ml."):
            return "ml-inference"
        return self.service_name

    def _normalize_sample_rate(self, sample_rate: float) -> float:
        return max(0.0, min(1.0, float(sample_rate)))

    def _generate_id(self, length: int) -> str:
        return uuid.uuid4().hex[:length]

    def _stringify_dict(self, values: Mapping[str, Any]) -> Dict[str, str]:
        return {str(key): self._stringify(value) for key, value in values.items()}

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, sort_keys=True, default=str)
            except TypeError:
                return str(value)
        return str(value)

    def _reset_context(self, tokens: tuple[Token[Any], Token[Any], Token[Any]]) -> None:
        trace_token, span_token, current_span_token = tokens
        _current_trace_id.reset(trace_token)
        _current_span_id.reset(span_token)
        _current_span.reset(current_span_token)

    def _iso_to_epoch_microseconds(self, timestamp: str) -> int:
        dt = datetime.fromisoformat(timestamp)
        return int(dt.timestamp() * 1_000_000)


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn)


_global_tracer: Optional[ArgusTracer] = None
_global_lock = threading.Lock()


def get_argus_tracer() -> ArgusTracer:
    global _global_tracer
    if _global_tracer is None:
        with _global_lock:
            if _global_tracer is None:
                _global_tracer = ArgusTracer()
    return _global_tracer


def traced(operation_name: Optional[str] = None, attributes: Optional[Mapping[str, Any]] = None) -> Callable[[F], F]:
    return get_argus_tracer().traced(operation_name=operation_name, attributes=attributes)
