"""TensorRT compilation utilities with engine caching and versioning."""

# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAny=false, reportExplicitAny=false, reportDeprecated=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedImport=false, reportConstantRedefinition=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportGeneralTypeIssues=false

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

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
class TensorRTCompilationConfig:
    """Compilation parameters for a TensorRT engine build."""

    precision: str = "fp16"
    min_batch_size: int = 1
    optimal_batch_size: int = 8
    max_batch_size: int = 64
    workspace_size_bytes: int = 1 << 30
    enable_dynamic_shapes: bool = True
    calibration_cache_file: Optional[str] = None
    input_name: Optional[str] = None
    input_shape: Optional[Sequence[int]] = None
    strict_types: bool = False


@dataclass
class EngineArtifact:
    """Represents a compiled engine and associated metadata."""

    model_name: str
    version: str
    onnx_path: str
    engine_path: Optional[str]
    metadata_path: Optional[str]
    created_at: float
    precision: str
    batch_range: Dict[str, int]
    success: bool
    device: str = "cpu"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class _NullCalibrator:
    """Placeholder calibrator when INT8 calibration cannot be enabled."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


class _EntropyCalibrator(getattr(trt, "IInt8EntropyCalibrator2", object)):
    """Minimal TensorRT INT8 calibrator backed by NumPy batches."""

    def __init__(self, batches: Sequence[Any], cache_path: Optional[str]) -> None:
        if not (_HAS_TENSORRT and _HAS_PYCUDA and _HAS_NUMPY):
            raise RuntimeError("INT8 calibration requires TensorRT, PyCUDA, and NumPy")
        super().__init__()
        self._batches = [np.ascontiguousarray(batch.astype(np.float32)) for batch in batches]
        self._cache_path = cache_path
        self._index = 0
        self._allocation = None
        if self._batches:
            self._allocation = cuda.mem_alloc(self._batches[0].nbytes)

    def get_batch_size(self) -> int:
        if not self._batches:
            return 0
        return int(self._batches[0].shape[0])

    def get_batch(self, names: Sequence[str]) -> Optional[list[int]]:
        del names
        if self._index >= len(self._batches) or self._allocation is None:
            return None
        batch = self._batches[self._index]
        cuda.memcpy_htod(self._allocation, batch)
        self._index += 1
        return [int(self._allocation)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if not self._cache_path:
            return None
        path = Path(self._cache_path)
        if path.exists():
            return path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        if not self._cache_path:
            return
        path = Path(self._cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cache)


class TensorRTCompiler:
    """Compile ONNX models into cached TensorRT engines when available."""

    def __init__(self, cache_dir: str | os.PathLike[str]) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._logger = trt.Logger(trt.Logger.WARNING) if _HAS_TENSORRT else None
        logger.info(
            "TensorRTCompiler initialised (tensorrt=%s, pycuda=%s)",
            _HAS_TENSORRT,
            _HAS_PYCUDA,
        )

    def compile_model(
        self,
        model_name: str,
        onnx_path: str | os.PathLike[str],
        config: Optional[TensorRTCompilationConfig] = None,
        version: Optional[str] = None,
        calibration_batches: Optional[Sequence[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_rebuild: bool = False,
    ) -> EngineArtifact:
        """Compile or reuse a cached TensorRT engine."""
        config = config or TensorRTCompilationConfig()
        resolved_onnx = Path(onnx_path)
        if not resolved_onnx.exists():
            return self._failure_artifact(
                model_name=model_name,
                version=version or "missing",
                onnx_path=resolved_onnx,
                config=config,
                error=f"ONNX model not found: {resolved_onnx}",
            )

        resolved_version = version or self._compute_version(resolved_onnx, config)
        engine_dir = self.cache_dir / model_name / resolved_version
        engine_dir.mkdir(parents=True, exist_ok=True)
        engine_path = engine_dir / f"{model_name}_{config.precision}.engine"
        metadata_path = engine_dir / f"{model_name}_{config.precision}.json"

        if engine_path.exists() and metadata_path.exists() and not force_rebuild:
            logger.info("TensorRTCompiler: using cached engine for %s:%s", model_name, resolved_version)
            cached = self._load_metadata(metadata_path)
            return EngineArtifact(
                model_name=model_name,
                version=resolved_version,
                onnx_path=str(resolved_onnx),
                engine_path=str(engine_path),
                metadata_path=str(metadata_path),
                created_at=float(cached.get("created_at", time.time())),
                precision=config.precision,
                batch_range={
                    "min": config.min_batch_size,
                    "optimal": config.optimal_batch_size,
                    "max": config.max_batch_size,
                },
                success=True,
                device="gpu",
                metadata=cached,
            )

        if not _HAS_TENSORRT:
            error = "TensorRT not installed; engine compilation skipped"
            logger.warning("TensorRTCompiler: %s", error)
            artifact = self._failure_artifact(
                model_name=model_name,
                version=resolved_version,
                onnx_path=resolved_onnx,
                config=config,
                error=error,
            )
            self._write_metadata(metadata_path, artifact, extra=metadata or {})
            artifact.metadata_path = str(metadata_path)
            return artifact

        try:
            engine_bytes, build_metadata = self._build_engine(
                resolved_onnx,
                config=config,
                calibration_batches=calibration_batches,
            )
            engine_path.write_bytes(engine_bytes)
            artifact = EngineArtifact(
                model_name=model_name,
                version=resolved_version,
                onnx_path=str(resolved_onnx),
                engine_path=str(engine_path),
                metadata_path=str(metadata_path),
                created_at=time.time(),
                precision=config.precision,
                batch_range={
                    "min": config.min_batch_size,
                    "optimal": config.optimal_batch_size,
                    "max": config.max_batch_size,
                },
                success=True,
                device="gpu",
                metadata={**build_metadata, **(metadata or {})},
            )
            self._write_metadata(metadata_path, artifact)
            logger.info(
                "TensorRTCompiler: compiled %s:%s (%s)",
                model_name,
                resolved_version,
                config.precision,
            )
            return artifact
        except Exception as exc:  # noqa: BLE001
            logger.error("TensorRTCompiler: compilation failed for %s", model_name, exc_info=True)
            artifact = self._failure_artifact(
                model_name=model_name,
                version=resolved_version,
                onnx_path=resolved_onnx,
                config=config,
                error=str(exc),
            )
            self._write_metadata(metadata_path, artifact, extra=metadata or {})
            artifact.metadata_path = str(metadata_path)
            return artifact

    def _build_engine(
        self,
        onnx_path: Path,
        config: TensorRTCompilationConfig,
        calibration_batches: Optional[Sequence[Any]],
    ) -> tuple[bytes, Dict[str, Any]]:
        builder = trt.Builder(self._logger)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, self._logger)

        if not parser.parse(onnx_path.read_bytes()):
            errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
            raise RuntimeError("; ".join(errors) or "ONNX parse failure")

        build_config = builder.create_builder_config()
        build_config.max_workspace_size = int(config.workspace_size_bytes)
        if config.strict_types:
            build_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        precision = config.precision.lower()
        if precision == "fp16" and builder.platform_has_fast_fp16:
            build_config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if builder.platform_has_fast_int8:
                build_config.set_flag(trt.BuilderFlag.INT8)
                calibrator = self._create_calibrator(config, calibration_batches)
                if isinstance(calibrator, _EntropyCalibrator):
                    build_config.int8_calibrator = calibrator
                else:
                    logger.warning("TensorRTCompiler: INT8 requested but %s", calibrator.reason)
                    if builder.platform_has_fast_fp16:
                        build_config.set_flag(trt.BuilderFlag.FP16)
            else:
                logger.warning("TensorRTCompiler: INT8 unsupported on this platform; using FP16/FP32")
                if builder.platform_has_fast_fp16:
                    build_config.set_flag(trt.BuilderFlag.FP16)

        if config.enable_dynamic_shapes and network.num_inputs > 0:
            profile = builder.create_optimization_profile()
            for idx in range(network.num_inputs):
                tensor = network.get_input(idx)
                raw_shape = tuple(int(dim if dim > 0 else 1) for dim in tensor.shape)
                batchless_shape = raw_shape[1:] if len(raw_shape) > 1 else (1,)
                min_shape = (config.min_batch_size, *batchless_shape)
                opt_shape = (config.optimal_batch_size, *batchless_shape)
                max_shape = (config.max_batch_size, *batchless_shape)
                profile.set_shape(tensor.name, min_shape, opt_shape, max_shape)
            build_config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, build_config)
        if serialized is None:
            raise RuntimeError("TensorRT returned an empty serialized engine")

        return bytes(serialized), {
            "platform_has_fast_fp16": bool(builder.platform_has_fast_fp16),
            "platform_has_fast_int8": bool(builder.platform_has_fast_int8),
            "num_inputs": int(network.num_inputs),
            "num_outputs": int(network.num_outputs),
            "dynamic_shapes": bool(config.enable_dynamic_shapes),
        }

    def _create_calibrator(
        self,
        config: TensorRTCompilationConfig,
        calibration_batches: Optional[Sequence[Any]],
    ) -> _EntropyCalibrator | _NullCalibrator:
        if not calibration_batches:
            return _NullCalibrator("no calibration batches were provided")
        if not _HAS_PYCUDA:
            return _NullCalibrator("PyCUDA is not installed")
        cache_path = config.calibration_cache_file
        return _EntropyCalibrator(calibration_batches, cache_path)

    def _compute_version(self, onnx_path: Path, config: TensorRTCompilationConfig) -> str:
        digest = hashlib.sha256()
        digest.update(onnx_path.read_bytes())
        digest.update(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
        return digest.hexdigest()[:16]

    def _failure_artifact(
        self,
        model_name: str,
        version: str,
        onnx_path: Path,
        config: TensorRTCompilationConfig,
        error: str,
    ) -> EngineArtifact:
        return EngineArtifact(
            model_name=model_name,
            version=version,
            onnx_path=str(onnx_path),
            engine_path=None,
            metadata_path=None,
            created_at=time.time(),
            precision=config.precision,
            batch_range={
                "min": config.min_batch_size,
                "optimal": config.optimal_batch_size,
                "max": config.max_batch_size,
            },
            success=False,
            device="cpu",
            error=error,
            metadata={"fallback": "cpu"},
        )

    @staticmethod
    def _load_metadata(path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.warning("TensorRTCompiler: failed to read metadata %s", path, exc_info=True)
            return {}

    @staticmethod
    def _write_metadata(path: Path, artifact: EngineArtifact, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            **artifact.metadata,
            **(extra or {}),
            "model_name": artifact.model_name,
            "version": artifact.version,
            "onnx_path": artifact.onnx_path,
            "engine_path": artifact.engine_path,
            "created_at": artifact.created_at,
            "precision": artifact.precision,
            "batch_range": artifact.batch_range,
            "success": artifact.success,
            "device": artifact.device,
            "error": artifact.error,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
