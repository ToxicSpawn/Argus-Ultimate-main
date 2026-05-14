"""GPU-friendly feature preprocessing with streaming updates and CPU fallback."""

# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false, reportDeprecated=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedImport=false, reportConstantRedefinition=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import cupy as cp  # type: ignore[import-untyped]

    _HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False


@dataclass
class FeaturePipelineStats:
    total_batches: int = 0
    total_rows: int = 0
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    streaming_updates: int = 0
    device: str = "cpu"


@dataclass
class _StreamingState:
    window_size: int
    samples: Deque[Any] = field(default_factory=deque)
    mean: Optional[Any] = None
    std: Optional[Any] = None


class FeaturePipelineGPU:
    """Preprocess features on GPU when CuPy is available."""

    def __init__(
        self,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        window_size: int = 512,
        eps: float = 1e-6,
        use_gpu: bool = True,
    ) -> None:
        if not _HAS_NUMPY:
            raise RuntimeError("NumPy is required for FeaturePipelineGPU")
        self._xp = cp if (use_gpu and _HAS_CUPY) else np
        self._device = "gpu" if self._xp is cp else "cpu"
        self._eps = eps
        self._lock = threading.Lock()
        self._state = _StreamingState(window_size=window_size, samples=deque(maxlen=window_size))
        self._stats = FeaturePipelineStats(device=self._device)
        self._latencies: Deque[float] = deque(maxlen=1024)

        self._mean = self._as_backend_array(mean) if mean is not None else None
        self._std = self._as_backend_array(std) if std is not None else None
        logger.info("FeaturePipelineGPU initialised (device=%s)", self._device)

    @property
    def device(self) -> str:
        return self._device

    def preprocess_batch(self, features: Sequence[Sequence[float]], include_deltas: bool = True) -> Any:
        """Normalize and enrich a feature batch."""
        t0 = time.perf_counter()
        batch = self._as_backend_array(features)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        normalized = self._normalize(batch)
        outputs = [normalized]
        if include_deltas:
            deltas = self._xp.zeros_like(normalized)
            if normalized.shape[0] > 1:
                deltas[1:] = normalized[1:] - normalized[:-1]
            outputs.append(deltas)
        enriched = self._xp.concatenate(outputs, axis=1)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._record_stats(latency_ms, rows=int(batch.shape[0]))
        return self.to_cpu(enriched)

    def normalize(self, features: Sequence[Sequence[float]] | Sequence[float]) -> Any:
        batch = self._as_backend_array(features)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        return self.to_cpu(self._normalize(batch))

    def update_streaming(self, features: Sequence[float]) -> Dict[str, Any]:
        """Update rolling statistics for streaming feature normalization."""
        sample = np.asarray(features, dtype=np.float32)
        with self._lock:
            self._state.samples.append(sample)
            stacked = np.stack(list(self._state.samples), axis=0)
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            std = np.where(std < self._eps, 1.0, std)
            self._state.mean = mean
            self._state.std = std
            self._mean = self._as_backend_array(mean)
            self._std = self._as_backend_array(std)
            self._stats.streaming_updates += 1
        return {
            "window_size": len(self._state.samples),
            "device": self._device,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "device": self._stats.device,
            "total_batches": self._stats.total_batches,
            "total_rows": self._stats.total_rows,
            "last_latency_ms": round(self._stats.last_latency_ms, 6),
            "avg_latency_ms": round(self._stats.avg_latency_ms, 6),
            "streaming_updates": self._stats.streaming_updates,
        }

    def _normalize(self, batch: Any) -> Any:
        if self._mean is None or self._std is None:
            mean = batch.mean(axis=0)
            std = batch.std(axis=0)
            std = self._xp.where(std < self._eps, 1.0, std)
        else:
            mean = self._mean
            std = self._std
        return (batch - mean) / (std + self._eps)

    def _as_backend_array(self, value: Any) -> Any:
        if value is None:
            return None
        return self._xp.asarray(value, dtype=self._xp.float32)

    def to_cpu(self, value: Any) -> Any:
        if self._xp is cp:
            return cp.asnumpy(value)
        return np.asarray(value)

    def _record_stats(self, latency_ms: float, rows: int) -> None:
        self._latencies.append(latency_ms)
        self._stats.total_batches += 1
        self._stats.total_rows += rows
        self._stats.last_latency_ms = latency_ms
        self._stats.avg_latency_ms = sum(self._latencies) / len(self._latencies)
