from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


LaneExecuteFn = Callable[[str, List[Any], str], Awaitable[List[Dict[str, Any]]]]


@dataclass
class ExecutionLaneStats:
    symbol: str
    enqueued_signals: int = 0
    dropped_signals: int = 0
    executed_batches: int = 0
    executed_signals: int = 0
    errors: int = 0
    last_latency_ms: float = 0.0
    latency_samples_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=256))

    def record_execution(self, *, signals: int, latency_ms: float, had_error: bool) -> None:
        self.executed_batches += 1
        self.executed_signals += int(max(0, signals))
        self.last_latency_ms = float(max(0.0, latency_ms))
        self.latency_samples_ms.append(self.last_latency_ms)
        if had_error:
            self.errors += 1

    def snapshot(self) -> Dict[str, Any]:
        samples = sorted(float(x) for x in list(self.latency_samples_ms))
        p50 = samples[int((len(samples) - 1) * 0.50)] if samples else 0.0
        p90 = samples[int((len(samples) - 1) * 0.90)] if samples else 0.0
        return {
            "symbol": str(self.symbol),
            "enqueued_signals": int(self.enqueued_signals),
            "dropped_signals": int(self.dropped_signals),
            "executed_batches": int(self.executed_batches),
            "executed_signals": int(self.executed_signals),
            "errors": int(self.errors),
            "last_latency_ms": float(self.last_latency_ms),
            "latency_p50_ms": float(p50),
            "latency_p90_ms": float(p90),
        }


class ExecutionLane:
    """Isolated per-symbol lane with bounded queue and serialized execution."""

    def __init__(self, *, symbol: str, batch_size: int = 8, max_queue: int = 128) -> None:
        self.symbol = str(symbol or "UNKNOWN")
        self.batch_size = max(1, int(batch_size or 1))
        self.max_queue = max(1, int(max_queue or 1))
        self.queue: Deque[Any] = deque()
        self.lock = asyncio.Lock()
        self.stats = ExecutionLaneStats(symbol=self.symbol)

    @property
    def pending(self) -> int:
        return int(len(self.queue))

    def enqueue(self, signals: List[Any]) -> Tuple[int, int]:
        accepted = 0
        dropped = 0
        for sig in list(signals or []):
            if len(self.queue) >= self.max_queue:
                dropped += 1
                continue
            self.queue.append(sig)
            accepted += 1
        self.stats.enqueued_signals += int(accepted)
        self.stats.dropped_signals += int(dropped)
        return int(accepted), int(dropped)

    async def drain(
        self,
        *,
        execute_fn: LaneExecuteFn,
        correlation_id: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        out: List[Dict[str, Any]] = []
        errors: List[str] = []
        async with self.lock:
            while self.queue:
                batch: List[Any] = []
                while self.queue and len(batch) < self.batch_size:
                    batch.append(self.queue.popleft())
                t0 = time.perf_counter()
                had_error = False
                try:
                    result = await execute_fn(self.symbol, batch, correlation_id)
                    if isinstance(result, list):
                        out.extend(result)
                except Exception as exc:
                    had_error = True
                    errors.append(str(exc))
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.stats.record_execution(signals=len(batch), latency_ms=latency_ms, had_error=had_error)
        return out, errors


class ExecutionMeshError(RuntimeError):
    def __init__(self, message: str, *, summary: Dict[str, Any]) -> None:
        super().__init__(message)
        self.summary = dict(summary or {})


class ExecutionMeshCoordinator:
    """
    In-process execution mesh coordinator.

    Isolation properties:
    - each symbol gets its own bounded queue
    - each lane executes under its own lock
    - one lane failure can optionally halt the full cycle (fail-closed)
    """

    OVERFLOW_LANE = "__overflow__"

    def __init__(
        self,
        *,
        max_lanes: int = 8,
        max_queue_per_lane: int = 128,
        batch_size: int = 8,
        parallel_lanes: bool = True,
        halt_on_lane_error: bool = True,
        allowed_symbols: Optional[List[str]] = None,
        max_queue_size: int = 1000,
    ) -> None:
        self.max_lanes = max(1, int(max_lanes or 1))
        self.max_queue_per_lane = max(1, int(max_queue_per_lane or 1))
        self.batch_size = max(1, int(batch_size or 1))
        self.parallel_lanes = bool(parallel_lanes)
        self.halt_on_lane_error = bool(halt_on_lane_error)
        self.allowed_symbols = {str(s).upper() for s in list(allowed_symbols or []) if str(s).strip()}
        self.max_queue_size = max(1, int(max_queue_size))
        self._lanes: Dict[str, ExecutionLane] = {}

    def _signal_symbol(self, signal: Any) -> str:
        if isinstance(signal, dict):
            symbol = signal.get("symbol")
        else:
            symbol = getattr(signal, "symbol", None)
        sym = str(symbol or "").strip().upper()
        return sym if sym else "UNKNOWN"

    def _get_or_create_lane(self, symbol: str) -> ExecutionLane:
        sym = str(symbol or "UNKNOWN").upper()
        if self.allowed_symbols and sym not in self.allowed_symbols:
            sym = self.OVERFLOW_LANE
        if sym in self._lanes:
            return self._lanes[sym]
        if len(self._lanes) >= self.max_lanes:
            sym = self.OVERFLOW_LANE
            if sym in self._lanes:
                return self._lanes[sym]
            # If we are at lane cap and overflow does not exist, reuse deterministic oldest lane.
            oldest_key = sorted(self._lanes.keys())[0]
            return self._lanes[oldest_key]
        lane = ExecutionLane(symbol=sym, batch_size=self.batch_size, max_queue=self.max_queue_per_lane)
        self._lanes[sym] = lane
        return lane

    def _total_queued(self) -> int:
        """Total items queued across all lanes."""
        return sum(lane.pending for lane in self._lanes.values())

    def queue_pressure(self) -> float:
        """Return queue pressure as a ratio 0.0 to 1.0."""
        total = self._total_queued()
        return min(1.0, total / self.max_queue_size)

    def lanes_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {k: lane.stats.snapshot() for k, lane in sorted(self._lanes.items())}

    async def execute_cycle(
        self,
        signals: List[Any],
        *,
        execute_fn: LaneExecuteFn,
        correlation_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        total_in = int(len(signals or []))
        if total_in <= 0:
            return [], {
                "enabled": True,
                "input_signals": 0,
                "accepted_signals": 0,
                "dropped_signals": 0,
                "rejected_backpressure": 0,
                "lanes_active": 0,
                "lane_errors": 0,
                "lane_stats": self.lanes_snapshot(),
            }

        # Backpressure: reject entire batch if queue is full
        current_queued = self._total_queued()
        if current_queued >= self.max_queue_size:
            logger.warning(
                "Execution mesh backpressure: queue full (%d/%d), rejecting %d signals",
                current_queued, self.max_queue_size, total_in,
            )
            return [], {
                "enabled": True,
                "input_signals": int(total_in),
                "accepted_signals": 0,
                "dropped_signals": 0,
                "rejected_backpressure": int(total_in),
                "lanes_active": 0,
                "lane_errors": 0,
                "lane_stats": self.lanes_snapshot(),
            }

        accepted = 0
        dropped = 0
        rejected_bp = 0
        lane_counts: Dict[str, int] = {}
        for sig in list(signals or []):
            # Per-signal backpressure check
            if self._total_queued() >= self.max_queue_size:
                rejected_bp += 1
                logger.warning("Execution mesh backpressure: rejecting signal (queue %d/%d)",
                               self._total_queued(), self.max_queue_size)
                continue
            sym = self._signal_symbol(sig)
            lane = self._get_or_create_lane(sym)
            a, d = lane.enqueue([sig])
            accepted += int(a)
            dropped += int(d)
            lane_counts[lane.symbol] = int(lane_counts.get(lane.symbol, 0) + a)

        lane_symbols = sorted([k for k, lane in self._lanes.items() if lane.pending > 0])
        lane_outputs: Dict[str, List[Dict[str, Any]]] = {}
        lane_errors: List[Dict[str, Any]] = []

        async def _run_lane(sym: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
            lane = self._lanes[sym]
            lane_corr = f"{correlation_id}:{sym}" if correlation_id else sym
            out, errs = await lane.drain(execute_fn=execute_fn, correlation_id=lane_corr)
            return sym, out, errs

        if self.parallel_lanes and len(lane_symbols) > 1:
            tasks = [asyncio.create_task(_run_lane(sym)) for sym in lane_symbols]
            done = await asyncio.gather(*tasks)
            for sym, out, errs in done:
                lane_outputs[sym] = out
                for e in errs:
                    lane_errors.append({"symbol": sym, "error": str(e)})
        else:
            for sym in lane_symbols:
                out, errs = await self._lanes[sym].drain(
                    execute_fn=execute_fn,
                    correlation_id=(f"{correlation_id}:{sym}" if correlation_id else sym),
                )
                lane_outputs[sym] = out
                for e in errs:
                    lane_errors.append({"symbol": sym, "error": str(e)})

        merged: List[Dict[str, Any]] = []
        for sym in lane_symbols:
            merged.extend(list(lane_outputs.get(sym) or []))

        summary = {
            "enabled": True,
            "input_signals": int(total_in),
            "accepted_signals": int(accepted),
            "dropped_signals": int(dropped),
            "rejected_backpressure": int(rejected_bp),
            "lanes_active": int(len(lane_symbols)),
            "lane_signal_counts": dict(lane_counts),
            "lane_errors": int(len(lane_errors)),
            "lane_error_details": list(lane_errors),
            "lane_stats": self.lanes_snapshot(),
            "results_count": int(len(merged)),
        }

        if lane_errors and self.halt_on_lane_error:
            first = lane_errors[0]
            raise ExecutionMeshError(
                f"execution mesh lane failure ({first.get('symbol')}): {first.get('error')}",
                summary=summary,
            )
        return merged, summary

