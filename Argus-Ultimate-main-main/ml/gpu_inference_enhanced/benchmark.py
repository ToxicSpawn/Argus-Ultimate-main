"""Benchmark helpers for low-latency inference runtimes."""

# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false, reportDeprecated=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedImport=false, reportConstantRedefinition=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from .gpu_inference_engine import GPUInferenceEngine

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import pynvml  # type: ignore[import-untyped]

    _HAS_NVML = True
except ImportError:
    pynvml = None  # type: ignore[assignment]
    _HAS_NVML = False


@dataclass
class BenchmarkResult:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float
    throughput_items_per_sec: float
    iterations: int
    batch_size: int
    gpu_utilization_pct: Optional[float] = None
    gpu_memory_utilization_pct: Optional[float] = None


class InferenceBenchmark:
    """Run latency and throughput benchmarks against an inference engine."""

    def __init__(self, engine: GPUInferenceEngine) -> None:
        self.engine = engine

    def run(
        self,
        sample_input: Sequence[Sequence[float]] | Sequence[float],
        iterations: int = 200,
        warmup_iterations: int = 10,
    ) -> BenchmarkResult:
        if not _HAS_NUMPY:
            raise RuntimeError("NumPy is required for benchmarking")
        batch = np.asarray(sample_input, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        for _ in range(max(0, warmup_iterations)):
            self.engine.infer(batch)

        latencies = []
        t0 = time.perf_counter()
        for _ in range(max(1, iterations)):
            response = self.engine.infer(batch)
            latencies.append(response.latency_ms)
        elapsed = max(time.perf_counter() - t0, 1e-9)

        arr = np.asarray(latencies, dtype=float)
        gpu_stats = self._gpu_stats()
        result = BenchmarkResult(
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            avg_ms=float(arr.mean()),
            throughput_items_per_sec=float((iterations * batch.shape[0]) / elapsed),
            iterations=iterations,
            batch_size=int(batch.shape[0]),
            gpu_utilization_pct=gpu_stats.get("utilization_pct"),
            gpu_memory_utilization_pct=gpu_stats.get("memory_utilization_pct"),
        )
        logger.info(
            "InferenceBenchmark: p50=%.4fms p95=%.4fms p99=%.4fms throughput=%.2f/s",
            result.p50_ms,
            result.p95_ms,
            result.p99_ms,
            result.throughput_items_per_sec,
        )
        return result

    def throughput_test(
        self,
        sample_input: Sequence[Sequence[float]] | Sequence[float],
        duration_seconds: float = 5.0,
    ) -> Dict[str, Any]:
        if not _HAS_NUMPY:
            raise RuntimeError("NumPy is required for benchmarking")
        batch = np.asarray(sample_input, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        processed = 0
        started = time.perf_counter()
        while time.perf_counter() - started < duration_seconds:
            self.engine.infer(batch)
            processed += int(batch.shape[0])

        elapsed = max(time.perf_counter() - started, 1e-9)
        gpu_stats = self._gpu_stats()
        return {
            "duration_seconds": elapsed,
            "processed_items": processed,
            "throughput_items_per_sec": processed / elapsed,
            "gpu_utilization_pct": gpu_stats.get("utilization_pct"),
            "gpu_memory_utilization_pct": gpu_stats.get("memory_utilization_pct"),
        }

    def _gpu_stats(self) -> Dict[str, Optional[float]]:
        if not _HAS_NVML:
            return {"utilization_pct": None, "memory_utilization_pct": None}
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_utilization = (mem.used / mem.total) * 100.0 if mem.total else 0.0
            return {
                "utilization_pct": float(util.gpu),
                "memory_utilization_pct": float(memory_utilization),
            }
        except Exception:  # noqa: BLE001
            logger.warning("InferenceBenchmark: failed to query NVML", exc_info=True)
            return {"utilization_pct": None, "memory_utilization_pct": None}
