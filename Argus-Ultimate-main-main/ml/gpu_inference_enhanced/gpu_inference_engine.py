"""High-performance inference engine with GPU-first execution and CPU fallback."""

# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false, reportDeprecated=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedImport=false, reportConstantRedefinition=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _HAS_ORT = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _HAS_ORT = False

try:
    import tensorrt as trt  # type: ignore[import-untyped]

    _HAS_TENSORRT = True
except ImportError:
    trt = None  # type: ignore[assignment]
    _HAS_TENSORRT = False

try:
    import pycuda.driver as cuda  # type: ignore[import-untyped]

    _HAS_PYCUDA = True
except ImportError:
    cuda = None  # type: ignore[assignment]
    _HAS_PYCUDA = False


@dataclass
class InferenceResponse:
    """Normalized output from one inference call."""

    outputs: Dict[str, Any]
    latency_ms: float
    batch_size: int
    device: str
    success: bool
    model_version: Optional[str] = None
    error: Optional[str] = None


@dataclass
class InferenceMetrics:
    """Rolling latency and throughput counters."""

    total_requests: int = 0
    total_items: int = 0
    total_errors: int = 0
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_items_per_sec: float = 0.0


class _LatencyTracker:
    def __init__(self, maxlen: int = 2048) -> None:
        self._latencies: Deque[float] = deque(maxlen=maxlen)
        self._timestamps: Deque[tuple[float, int]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, latency_ms: float, batch_size: int) -> InferenceMetrics:
        now = time.time()
        with self._lock:
            self._latencies.append(latency_ms)
            self._timestamps.append((now, batch_size))
            while self._timestamps and now - self._timestamps[0][0] > 60.0:
                self._timestamps.popleft()

            arr = np.asarray(self._latencies, dtype=float) if _HAS_NUMPY else list(self._latencies)
            if _HAS_NUMPY:
                p50 = float(np.percentile(arr, 50)) if len(arr) else 0.0
                p95 = float(np.percentile(arr, 95)) if len(arr) else 0.0
                p99 = float(np.percentile(arr, 99)) if len(arr) else 0.0
                avg = float(arr.mean()) if len(arr) else 0.0
            else:
                ordered = sorted(arr)
                avg = sum(ordered) / len(ordered) if ordered else 0.0
                p50 = ordered[int((len(ordered) - 1) * 0.50)] if ordered else 0.0
                p95 = ordered[int((len(ordered) - 1) * 0.95)] if ordered else 0.0
                p99 = ordered[int((len(ordered) - 1) * 0.99)] if ordered else 0.0

            recent_items = sum(size for _, size in self._timestamps)
            window = max(now - self._timestamps[0][0], 1e-6) if self._timestamps else 1.0
            throughput = recent_items / window

        return InferenceMetrics(
            last_latency_ms=latency_ms,
            avg_latency_ms=avg,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            throughput_items_per_sec=throughput,
        )


class _TensorBufferPool:
    """Reusable host/device buffers keyed by shape and dtype."""

    def __init__(self) -> None:
        self._host_cache: Dict[tuple[tuple[int, ...], str], list[Any]] = defaultdict(list)
        self._device_cache: Dict[tuple[int, str], list[Any]] = defaultdict(list)
        self._lock = threading.Lock()

    def acquire_host(self, shape: Sequence[int], dtype: Any) -> Any:
        key = (tuple(int(x) for x in shape), str(dtype))
        with self._lock:
            if self._host_cache[key]:
                return self._host_cache[key].pop()
        return np.empty(shape, dtype=dtype)

    def release_host(self, array: Any) -> None:
        key = (tuple(int(x) for x in array.shape), str(array.dtype))
        with self._lock:
            self._host_cache[key].append(array)

    def acquire_device(self, size_bytes: int) -> Any:
        key = (int(size_bytes), "device")
        with self._lock:
            if self._device_cache[key]:
                return self._device_cache[key].pop()
        if _HAS_PYCUDA:
            return cuda.mem_alloc(size_bytes)
        return None

    def release_device(self, allocation: Any, size_bytes: int) -> None:
        if allocation is None:
            return
        key = (int(size_bytes), "device")
        with self._lock:
            self._device_cache[key].append(allocation)


class _CudaStreamManager:
    """Simple round-robin CUDA stream manager."""

    def __init__(self, stream_count: int = 2) -> None:
        self._streams: list[Any] = []
        self._index = 0
        self._lock = threading.Lock()
        if _HAS_PYCUDA:
            self._streams = [cuda.Stream() for _ in range(max(1, stream_count))]

    def next_stream(self) -> Any:
        if not self._streams:
            return None
        with self._lock:
            stream = self._streams[self._index]
            self._index = (self._index + 1) % len(self._streams)
            return stream


class GPUInferenceEngine:
    """Unified inference engine using TensorRT, ONNX Runtime, or CPU fallback."""

    def __init__(
        self,
        engine_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        cpu_fallback: Optional[Any] = None,
        model_version: Optional[str] = None,
        preferred_device: str = "auto",
        max_batch_size: int = 64,
        stream_count: int = 2,
    ) -> None:
        self.engine_path = engine_path
        self.onnx_path = onnx_path
        self.cpu_fallback = cpu_fallback
        self.model_version = model_version
        self.preferred_device = preferred_device
        self.max_batch_size = max_batch_size

        self._device = "cpu"
        self._runtime = None
        self._engine = None
        self._context = None
        self._onnx_session = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._binding_indices: Dict[str, int] = {}
        self._stream_manager = _CudaStreamManager(stream_count=stream_count)
        self._pool = _TensorBufferPool()
        self._latency_tracker = _LatencyTracker()
        self._metrics = InferenceMetrics()
        self._lock = threading.Lock()

        self._initialise_runtime()

    @property
    def device(self) -> str:
        return self._device

    @property
    def metrics(self) -> InferenceMetrics:
        return self._metrics

    @property
    def is_gpu_enabled(self) -> bool:
        return self._device.startswith("gpu") or self._device.startswith("cuda")

    def infer(self, features: Any) -> InferenceResponse:
        """Run batched inference synchronously."""
        batch = self._prepare_batch(features)
        batch_size = int(batch.shape[0])
        t0 = time.perf_counter()
        try:
            if self._context is not None and self._engine is not None:
                outputs = self._infer_tensorrt(batch)
            elif self._onnx_session is not None:
                outputs = self._infer_onnx(batch)
            else:
                outputs = self._infer_cpu(batch)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._update_metrics(latency_ms, batch_size, success=True)
            return InferenceResponse(
                outputs=outputs,
                latency_ms=latency_ms,
                batch_size=batch_size,
                device=self._device,
                success=True,
                model_version=self.model_version,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPUInferenceEngine: primary inference failed, falling back to CPU: %s", exc)
            try:
                outputs = self._infer_cpu(batch)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._update_metrics(latency_ms, batch_size, success=False)
                return InferenceResponse(
                    outputs=outputs,
                    latency_ms=latency_ms,
                    batch_size=batch_size,
                    device="cpu",
                    success=True,
                    model_version=self.model_version,
                    error=str(exc),
                )
            except Exception as fallback_exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._update_metrics(latency_ms, batch_size, success=False)
                logger.error("GPUInferenceEngine: CPU fallback failed", exc_info=True)
                return InferenceResponse(
                    outputs={},
                    latency_ms=latency_ms,
                    batch_size=batch_size,
                    device="cpu",
                    success=False,
                    model_version=self.model_version,
                    error=str(fallback_exc),
                )

    async def infer_async(self, features: Any) -> InferenceResponse:
        """Run inference without blocking the event loop."""
        return await asyncio.to_thread(self.infer, features)

    def warmup(self, sample_input: Optional[Any] = None, runs: int = 3) -> None:
        """Prime the runtime to reduce first-request latency spikes."""
        if sample_input is None:
            sample_input = np.zeros((1, 1), dtype=np.float32)
        for _ in range(max(1, runs)):
            self.infer(sample_input)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "device": self._device,
            "model_version": self.model_version,
            "total_requests": self._metrics.total_requests,
            "total_items": self._metrics.total_items,
            "total_errors": self._metrics.total_errors,
            "last_latency_ms": round(self._metrics.last_latency_ms, 6),
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 6),
            "p50_latency_ms": round(self._metrics.p50_latency_ms, 6),
            "p95_latency_ms": round(self._metrics.p95_latency_ms, 6),
            "p99_latency_ms": round(self._metrics.p99_latency_ms, 6),
            "throughput_items_per_sec": round(self._metrics.throughput_items_per_sec, 6),
        }

    def _initialise_runtime(self) -> None:
        if self.engine_path and _HAS_TENSORRT and _HAS_PYCUDA:
            try:
                logger.info("GPUInferenceEngine: loading TensorRT engine %s", self.engine_path)
                trt_logger = trt.Logger(trt.Logger.WARNING)
                self._runtime = trt.Runtime(trt_logger)
                engine_bytes = Path(self.engine_path).read_bytes()
                self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
                self._context = self._engine.create_execution_context() if self._engine else None
                if self._context is not None:
                    self._binding_indices = {
                        self._engine.get_tensor_name(i): i for i in range(self._engine.num_io_tensors)
                    }
                    self._input_names = [
                        self._engine.get_tensor_name(i)
                        for i in range(self._engine.num_io_tensors)
                        if self._engine.get_tensor_mode(self._engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
                    ]
                    self._output_names = [
                        self._engine.get_tensor_name(i)
                        for i in range(self._engine.num_io_tensors)
                        if self._engine.get_tensor_mode(self._engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
                    ]
                    self._device = "gpu:tensorrt"
                    return
            except Exception:  # noqa: BLE001
                logger.warning("GPUInferenceEngine: TensorRT load failed", exc_info=True)

        if self.onnx_path and _HAS_ORT:
            providers = ["CPUExecutionProvider"]
            try:
                available = set(ort.get_available_providers())
                if self.preferred_device != "cpu" and "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                sess_options = ort.SessionOptions()
                sess_options.inter_op_num_threads = 1
                sess_options.intra_op_num_threads = 1
                self._onnx_session = ort.InferenceSession(self.onnx_path, sess_options, providers=providers)
                self._input_names = [item.name for item in self._onnx_session.get_inputs()]
                self._output_names = [item.name for item in self._onnx_session.get_outputs()]
                active = self._onnx_session.get_providers()
                self._device = "gpu:onnxruntime" if "CUDAExecutionProvider" in active else "cpu"
                logger.info("GPUInferenceEngine: ONNX Runtime providers=%s", active)
                return
            except Exception:  # noqa: BLE001
                logger.warning("GPUInferenceEngine: ONNX Runtime init failed", exc_info=True)

        self._device = "cpu"
        logger.info("GPUInferenceEngine: using CPU fallback runtime")

    def _prepare_batch(self, features: Any) -> Any:
        if not _HAS_NUMPY:
            raise RuntimeError("NumPy is required for GPUInferenceEngine")
        batch = np.asarray(features, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if batch.shape[0] > self.max_batch_size:
            raise ValueError(f"Batch size {batch.shape[0]} exceeds max_batch_size={self.max_batch_size}")
        if np.any(np.isnan(batch)) or np.any(np.isinf(batch)):
            raise ValueError("Input batch contains NaN or Inf")
        return np.ascontiguousarray(batch)

    def _infer_tensorrt(self, batch: Any) -> Dict[str, Any]:
        with self._lock:
            if not self._input_names:
                raise RuntimeError("TensorRT engine has no input tensors")
            stream = self._stream_manager.next_stream()
            input_name = self._input_names[0]
            input_shape = tuple(int(dim) for dim in batch.shape)
            self._context.set_input_shape(input_name, input_shape)

            host_input = self._pool.acquire_host(batch.shape, np.float32)
            np.copyto(host_input, batch)
            input_bytes = int(host_input.nbytes)
            device_input = self._pool.acquire_device(input_bytes)
            if device_input is None:
                self._pool.release_host(host_input)
                raise RuntimeError("PyCUDA device allocation unavailable")

            bindings: list[int] = [0] * self._engine.num_io_tensors
            cuda.memcpy_htod_async(device_input, host_input, stream)
            bindings[self._binding_indices[input_name]] = int(device_input)

            host_outputs: Dict[str, Any] = {}
            output_allocations: list[tuple[Any, int]] = []
            for output_name in self._output_names:
                output_shape = tuple(int(dim) for dim in self._context.get_tensor_shape(output_name))
                host_output = self._pool.acquire_host(output_shape, np.float32)
                device_output = self._pool.acquire_device(int(host_output.nbytes))
                if device_output is None:
                    raise RuntimeError("PyCUDA output allocation unavailable")
                bindings[self._binding_indices[output_name]] = int(device_output)
                output_allocations.append((device_output, int(host_output.nbytes)))
                host_outputs[output_name] = host_output

            if not self._context.execute_async_v2(bindings=bindings, stream_handle=stream.handle):
                raise RuntimeError("TensorRT execution returned False")

            for output_name, (device_output, size_bytes) in zip(self._output_names, output_allocations):
                cuda.memcpy_dtoh_async(host_outputs[output_name], device_output, stream)
                self._pool.release_device(device_output, size_bytes)

            stream.synchronize()
            self._pool.release_device(device_input, input_bytes)
            self._pool.release_host(host_input)

            result = {name: array.copy() for name, array in host_outputs.items()}
            for array in host_outputs.values():
                self._pool.release_host(array)
            return result

    def _infer_onnx(self, batch: Any) -> Dict[str, Any]:
        inputs = {self._input_names[0]: batch}
        outputs = self._onnx_session.run(self._output_names, inputs)
        return {name: value for name, value in zip(self._output_names, outputs)}

    def _infer_cpu(self, batch: Any) -> Dict[str, Any]:
        if self.cpu_fallback is None:
            return {"output": batch}
        if hasattr(self.cpu_fallback, "predict"):
            prediction = self.cpu_fallback.predict(batch)
        else:
            prediction = self.cpu_fallback(batch)
        return {"output": prediction}

    def _update_metrics(self, latency_ms: float, batch_size: int, success: bool) -> None:
        snapshot = self._latency_tracker.record(latency_ms, batch_size=batch_size)
        self._metrics.total_requests += 1
        self._metrics.total_items += batch_size
        if not success:
            self._metrics.total_errors += 1
        self._metrics.last_latency_ms = snapshot.last_latency_ms
        self._metrics.avg_latency_ms = snapshot.avg_latency_ms
        self._metrics.p50_latency_ms = snapshot.p50_latency_ms
        self._metrics.p95_latency_ms = snapshot.p95_latency_ms
        self._metrics.p99_latency_ms = snapshot.p99_latency_ms
        self._metrics.throughput_items_per_sec = snapshot.throughput_items_per_sec
