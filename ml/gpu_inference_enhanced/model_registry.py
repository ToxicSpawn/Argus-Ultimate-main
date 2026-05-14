"""Model registry with versioning, A/B routing, and automatic CPU fallback."""

# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false, reportDeprecated=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedImport=false, reportConstantRedefinition=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional

from .gpu_inference_engine import GPUInferenceEngine, InferenceResponse

logger = logging.getLogger(__name__)


@dataclass
class ModelRecord:
    """Model metadata and runtime bindings."""

    name: str
    version: str
    engine: GPUInferenceEngine
    cpu_fallback: Optional[Callable[[Any], Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    traffic_weight: float = 1.0
    registered_at: float = field(default_factory=time.time)
    active: bool = True
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    fallback_invocations: int = 0
    recent_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=256))


@dataclass
class ModelRoutingResult:
    """Chosen model version for a request."""

    model_name: str
    version: str
    device: str
    traffic_bucket: float


class GPUModelRegistry:
    """Register models, route requests, and track production performance."""

    def __init__(self, unhealthy_latency_ms: float = 10.0, error_rate_threshold: float = 0.10) -> None:
        self._registry: Dict[str, Dict[str, ModelRecord]] = defaultdict(dict)
        self._primary_version: Dict[str, str] = {}
        self._ab_overrides: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
        self._unhealthy_latency_ms = unhealthy_latency_ms
        self._error_rate_threshold = error_rate_threshold

    def register_model(
        self,
        model_name: str,
        version: str,
        engine: GPUInferenceEngine,
        cpu_fallback: Optional[Callable[[Any], Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        traffic_weight: float = 1.0,
        make_primary: bool = True,
    ) -> None:
        record = ModelRecord(
            name=model_name,
            version=version,
            engine=engine,
            cpu_fallback=cpu_fallback,
            metadata=metadata or {},
            traffic_weight=max(0.0, traffic_weight),
        )
        with self._lock:
            self._registry[model_name][version] = record
            if make_primary or model_name not in self._primary_version:
                self._primary_version[model_name] = version
        logger.info("GPUModelRegistry: registered %s:%s on %s", model_name, version, engine.device)

    def set_primary_version(self, model_name: str, version: str) -> None:
        with self._lock:
            if version not in self._registry.get(model_name, {}):
                raise KeyError(f"Unknown model version {model_name}:{version}")
            self._primary_version[model_name] = version

    def configure_ab_test(self, model_name: str, weights: Dict[str, float]) -> None:
        with self._lock:
            known_versions = self._registry.get(model_name, {})
            if not known_versions:
                raise KeyError(f"No models registered for {model_name}")
            for version in weights:
                if version not in known_versions:
                    raise KeyError(f"Unknown model version {model_name}:{version}")
            total = sum(max(0.0, value) for value in weights.values())
            if total <= 0.0:
                raise ValueError("A/B weights must sum to more than zero")
            self._ab_overrides[model_name] = {key: value / total for key, value in weights.items()}

    def predict(self, model_name: str, features: Any, request_id: Optional[str] = None) -> InferenceResponse:
        routing = self.route_request(model_name, request_id=request_id)
        record = self._registry[model_name][routing.version]
        response = record.engine.infer(features)
        self._record_outcome(record, response)

        if self._should_fallback(record, response) and record.cpu_fallback is not None:
            logger.warning("GPUModelRegistry: falling back to CPU for %s:%s", model_name, record.version)
            record.fallback_invocations += 1
            engine = GPUInferenceEngine(cpu_fallback=record.cpu_fallback, model_version=record.version)
            response = engine.infer(features)
            self._record_outcome(record, response)

        return response

    async def predict_async(self, model_name: str, features: Any, request_id: Optional[str] = None) -> InferenceResponse:
        routing = self.route_request(model_name, request_id=request_id)
        record = self._registry[model_name][routing.version]
        response = await record.engine.infer_async(features)
        self._record_outcome(record, response)
        return response

    def route_request(self, model_name: str, request_id: Optional[str] = None) -> ModelRoutingResult:
        with self._lock:
            versions = self._registry.get(model_name, {})
            if not versions:
                raise KeyError(f"Model not registered: {model_name}")
            weights = self._ab_overrides.get(model_name)
            if not weights:
                primary = self._primary_version[model_name]
                record = versions[primary]
                return ModelRoutingResult(model_name=model_name, version=primary, device=record.engine.device, traffic_bucket=1.0)

            bucket = self._stable_bucket(model_name, request_id or f"{time.time_ns()}")
            cursor = 0.0
            selected_version = self._primary_version[model_name]
            for version, weight in weights.items():
                cursor += weight
                if bucket <= cursor:
                    selected_version = version
                    break
            record = versions[selected_version]
            return ModelRoutingResult(
                model_name=model_name,
                version=selected_version,
                device=record.engine.device,
                traffic_bucket=bucket,
            )

    def get_model_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        names = [model_name] if model_name else sorted(self._registry.keys())
        payload: Dict[str, Any] = {}
        for name in names:
            payload[name] = {}
            for version, record in self._registry.get(name, {}).items():
                error_rate = (record.total_errors / record.total_requests) if record.total_requests else 0.0
                payload[name][version] = {
                    "active": record.active,
                    "device": record.engine.device,
                    "avg_latency_ms": round(record.avg_latency_ms, 6),
                    "last_latency_ms": round(record.last_latency_ms, 6),
                    "total_requests": record.total_requests,
                    "total_errors": record.total_errors,
                    "error_rate": round(error_rate, 6),
                    "fallback_invocations": record.fallback_invocations,
                    "engine_metrics": record.engine.get_metrics(),
                }
        return payload

    def _record_outcome(self, record: ModelRecord, response: InferenceResponse) -> None:
        record.total_requests += 1
        record.last_latency_ms = response.latency_ms
        record.recent_latencies.append(response.latency_ms)
        record.avg_latency_ms = sum(record.recent_latencies) / len(record.recent_latencies)
        if not response.success or response.error:
            record.total_errors += 1

    def _should_fallback(self, record: ModelRecord, response: InferenceResponse) -> bool:
        if not response.success:
            return True
        if response.device.startswith("cpu"):
            return False
        error_rate = (record.total_errors / record.total_requests) if record.total_requests else 0.0
        return response.latency_ms > self._unhealthy_latency_ms or error_rate > self._error_rate_threshold

    @staticmethod
    def _stable_bucket(model_name: str, request_id: str) -> float:
        digest = hashlib.sha256(f"{model_name}:{request_id}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF
