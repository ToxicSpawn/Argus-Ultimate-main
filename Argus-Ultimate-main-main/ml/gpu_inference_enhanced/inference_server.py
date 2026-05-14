"""FastAPI inference server with batching, health checks, and GPU-aware routing."""

# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false, reportDeprecated=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedImport=false, reportConstantRedefinition=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportGeneralTypeIssues=false, reportMissingTypeArgument=false, reportOptionalCall=false, reportCallIssue=false

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .model_registry import GPUModelRegistry

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    _FASTAPI_AVAILABLE = False

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


class InferenceRequest(BaseModel):
    model_name: str
    features: list[list[float]] | list[float]
    request_id: Optional[str] = None


class BatchInferenceRequest(BaseModel):
    model_name: str
    batch: list[list[float]]
    request_id: Optional[str] = None


@dataclass
class _PendingRequest:
    model_name: str
    features: Any
    request_id: str
    future: asyncio.Future
    enqueued_at: float


class _RequestBatcher:
    """Aggregates short-lived requests into micro-batches."""

    def __init__(self, registry: GPUModelRegistry, max_batch_size: int = 16, max_wait_ms: float = 0.5) -> None:
        self._registry = registry
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._queues: Dict[str, deque[_PendingRequest]] = defaultdict(deque)
        self._tasks: Dict[str, asyncio.Task] = {}

    async def submit(self, model_name: str, features: Any, request_id: str) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        pending = _PendingRequest(
            model_name=model_name,
            features=features,
            request_id=request_id,
            future=future,
            enqueued_at=time.perf_counter(),
        )
        queue = self._queues[model_name]
        queue.append(pending)
        if model_name not in self._tasks or self._tasks[model_name].done():
            self._tasks[model_name] = asyncio.create_task(self._flush_loop(model_name))
        return await future

    async def _flush_loop(self, model_name: str) -> None:
        await asyncio.sleep(self._max_wait_ms / 1000.0)
        queue = self._queues[model_name]
        if not queue:
            return
        batch = []
        entries = []
        while queue and len(entries) < self._max_batch_size:
            item = queue.popleft()
            entries.append(item)
            batch.append(item.features)

        try:
            response = self._registry.predict(model_name, batch, request_id=entries[0].request_id)
            outputs = response.outputs
            for entry in entries:
                if not entry.future.done():
                    entry.future.set_result(
                        {
                            "request_id": entry.request_id,
                            "device": response.device,
                            "latency_ms": response.latency_ms,
                            "outputs": outputs,
                            "model_version": response.model_version,
                            "batched": len(entries) > 1,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference batch failed for %s", model_name, exc_info=True)
            for entry in entries:
                if not entry.future.done():
                    entry.future.set_exception(exc)


class InferenceServer:
    """Production-oriented FastAPI inference server for trading models."""

    def __init__(
        self,
        registry: GPUModelRegistry,
        gpu_ids: Optional[list[int]] = None,
        max_batch_size: int = 16,
        max_batch_wait_ms: float = 0.5,
    ) -> None:
        if not _FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI is required to create InferenceServer")
        self.registry = registry
        self.gpu_ids = gpu_ids or self._discover_gpu_ids()
        self._batcher = _RequestBatcher(
            registry=registry,
            max_batch_size=max_batch_size,
            max_wait_ms=max_batch_wait_ms,
        )
        self._request_count = 0
        self._error_count = 0
        self._latencies = deque(maxlen=2048)
        self.app = FastAPI(title="Argus GPU Inference Server", version="1.0.0")
        self._mount_routes()

    def _mount_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> Dict[str, Any]:
            return {
                "status": "ok",
                "gpu_ids": self.gpu_ids,
                "models": self.registry.get_model_stats(),
                "requests": self._request_count,
                "errors": self._error_count,
            }

        @self.app.get("/metrics")
        async def metrics() -> Dict[str, Any]:
            latency_avg = (sum(self._latencies) / len(self._latencies)) if self._latencies else 0.0
            return {
                "requests_total": self._request_count,
                "errors_total": self._error_count,
                "avg_latency_ms": round(latency_avg, 6),
                "gpu_count": len(self.gpu_ids),
                "models": self.registry.get_model_stats(),
            }

        @self.app.post("/infer")
        async def infer(payload: InferenceRequest) -> Dict[str, Any]:
            request_id = payload.request_id or str(uuid.uuid4())
            t0 = time.perf_counter()
            try:
                result = await self._batcher.submit(payload.model_name, payload.features, request_id=request_id)
                self._request_count += 1
                self._latencies.append((time.perf_counter() - t0) * 1000.0)
                result["gpu_assignment"] = self._choose_gpu(payload.model_name, request_id)
                return result
            except KeyError as exc:
                self._error_count += 1
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                self._error_count += 1
                logger.error("InferenceServer: request failed", exc_info=True)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @self.app.post("/infer_batch")
        async def infer_batch(payload: BatchInferenceRequest) -> Dict[str, Any]:
            request_id = payload.request_id or str(uuid.uuid4())
            t0 = time.perf_counter()
            try:
                response = self.registry.predict(payload.model_name, payload.batch, request_id=request_id)
                self._request_count += 1
                self._latencies.append((time.perf_counter() - t0) * 1000.0)
                return {
                    "request_id": request_id,
                    "device": response.device,
                    "latency_ms": response.latency_ms,
                    "batch_size": response.batch_size,
                    "outputs": response.outputs,
                    "model_version": response.model_version,
                    "gpu_assignment": self._choose_gpu(payload.model_name, request_id),
                }
            except KeyError as exc:
                self._error_count += 1
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                self._error_count += 1
                logger.error("InferenceServer: batch request failed", exc_info=True)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _choose_gpu(self, model_name: str, request_id: str) -> Dict[str, Any]:
        if not self.gpu_ids:
            return {"gpu_id": None, "mode": "cpu_fallback"}
        bucket = int(uuid.UUID(hashlib_uuid(model_name, request_id)).int % len(self.gpu_ids))
        return {"gpu_id": self.gpu_ids[bucket], "mode": "round_robin_hash"}

    @staticmethod
    def _discover_gpu_ids() -> list[int]:
        value = os.environ.get("ARGUS_GPU_IDS")
        if not value:
            return [0]
        return [int(part.strip()) for part in value.split(",") if part.strip()]


def hashlib_uuid(model_name: str, request_id: str) -> str:
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, "argus-gpu-inference")
    return str(uuid.uuid5(namespace, f"{model_name}:{request_id}"))


def create_app(
    registry: GPUModelRegistry,
    gpu_ids: Optional[list[int]] = None,
    max_batch_size: int = 16,
    max_batch_wait_ms: float = 0.5,
) -> Optional[Any]:
    """Create a FastAPI app or return None when FastAPI is unavailable."""
    if not _FASTAPI_AVAILABLE:
        logger.warning("create_app: FastAPI not installed")
        return None
    server = InferenceServer(
        registry=registry,
        gpu_ids=gpu_ids,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
    return server.app
