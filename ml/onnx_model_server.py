#!/usr/bin/env python3
"""
ONNX Model Serving — thread-safe inference with latency tracking.

Loads ONNX models via onnxruntime (if available) and provides a unified
``predict()`` interface.  Includes optional sklearn-to-ONNX export via
skl2onnx.

Falls back gracefully when onnxruntime is not installed — all methods
return sentinel values and log warnings.

Standalone usage:
    server = ONNXModelServer()
    server.load_model("regime", "models/regime.onnx")
    preds = server.predict("regime", [[0.1, 0.2, 0.3]])
"""

from __future__ import annotations

import logging
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _HAS_ORT = True
    logger.debug("onnxruntime %s available", ort.__version__)
except ImportError:
    _HAS_ORT = False
    ort = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import-untyped]

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]

try:
    from skl2onnx import convert_sklearn  # type: ignore[import-untyped]
    from skl2onnx.common.data_types import FloatTensorType  # type: ignore[import-untyped]

    _HAS_SKL2ONNX = True
except ImportError:
    _HAS_SKL2ONNX = False


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass
class _ModelEntry:
    """Metadata + session for one loaded ONNX model."""

    name: str
    path: str
    session: Any  # ort.InferenceSession or None
    input_name: str
    input_shape: Optional[List[int]]
    output_names: List[str]
    loaded_at: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Latency tracking
    latencies_ms: List[float] = field(default_factory=list)
    call_count: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ONNXModelServer:
    """
    Thread-safe ONNX model loader and inference server.

    Parameters
    ----------
    models_dir : str
        Default directory for model files (default ``models/``).
    max_latency_samples : int
        How many inference latencies to retain per model for p99 calculation
        (default 1000).
    """

    def __init__(self, models_dir: str = "models", max_latency_samples: int = 1000):
        self.models_dir = models_dir
        self.max_latency_samples = max_latency_samples
        self._models: Dict[str, _ModelEntry] = {}
        self._global_lock = threading.Lock()

        status = "available" if _HAS_ORT else "NOT INSTALLED"
        logger.info("ONNXModelServer initialised (onnxruntime=%s)", status)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, name: str, path: str) -> bool:
        """
        Load an ONNX model from disk.

        Parameters
        ----------
        name : str
            Unique model identifier.
        path : str
            Filesystem path to ``.onnx`` file.

        Returns
        -------
        bool
            True if loaded successfully, False otherwise.
        """
        if not _HAS_ORT:
            logger.warning(
                "ONNXModelServer.load_model('%s'): onnxruntime not installed — skipping", name
            )
            return False

        resolved = Path(path)
        if not resolved.exists():
            # Try relative to models_dir
            resolved = Path(self.models_dir) / path
        if not resolved.exists():
            logger.error("ONNXModelServer: model file not found: %s", resolved)
            return False

        try:
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(resolved), sess_options)

            inputs = session.get_inputs()
            input_name = inputs[0].name if inputs else "input"
            input_shape = inputs[0].shape if inputs else None
            output_names = [o.name for o in session.get_outputs()]

            entry = _ModelEntry(
                name=name,
                path=str(resolved),
                session=session,
                input_name=input_name,
                input_shape=input_shape,
                output_names=output_names,
                loaded_at=time.time(),
            )

            with self._global_lock:
                self._models[name] = entry

            logger.info(
                "ONNXModelServer: loaded '%s' from %s (inputs=%s, outputs=%s)",
                name, resolved, input_name, output_names,
            )
            return True

        except Exception:
            logger.error("ONNXModelServer: failed to load '%s'", name, exc_info=True)
            return False

    def predict(self, name: str, features: List[List[float]]) -> Optional[List[Any]]:
        """
        Run inference on a loaded model.

        Parameters
        ----------
        name : str
            Model identifier (from ``load_model``).
        features : list of list of float
            2D input: rows = samples, columns = features.

        Returns
        -------
        list or None
            Model output (flattened if single output), or None on failure.
        """
        if not _HAS_ORT:
            logger.warning("ONNXModelServer.predict('%s'): onnxruntime not installed", name)
            return None

        with self._global_lock:
            entry = self._models.get(name)

        if entry is None:
            logger.warning("ONNXModelServer.predict: model '%s' not loaded", name)
            return None

        with entry.lock:
            t0 = time.monotonic()
            try:
                if _HAS_NUMPY:
                    input_array = np.array(features, dtype=np.float32)
                else:
                    input_array = features  # type: ignore[assignment]

                result = entry.session.run(
                    entry.output_names,
                    {entry.input_name: input_array},
                )

                elapsed_ms = (time.monotonic() - t0) * 1000
                entry.latencies_ms.append(elapsed_ms)
                if len(entry.latencies_ms) > self.max_latency_samples:
                    entry.latencies_ms = entry.latencies_ms[-self.max_latency_samples:]
                entry.call_count += 1

                # Flatten single-output models
                if len(result) == 1:
                    out = result[0]
                    if _HAS_NUMPY and hasattr(out, "tolist"):
                        out = out.tolist()
                    # Flatten nested single-element lists
                    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                        if len(out[0]) == 1:
                            out = [row[0] for row in out]
                    return out
                return result

            except Exception:
                logger.error("ONNXModelServer: inference failed for '%s'", name, exc_info=True)
                return None

    def get_latency_stats(self, name: str) -> Dict[str, Any]:
        """
        Return inference latency statistics for a model.

        Parameters
        ----------
        name : str
            Model identifier.

        Returns
        -------
        dict
            Keys: avg_ms, p99_ms, min_ms, max_ms, count.
        """
        with self._global_lock:
            entry = self._models.get(name)

        if entry is None:
            return {"avg_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "count": 0}

        lats = list(entry.latencies_ms)
        if not lats:
            return {"avg_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "count": 0}

        lats_sorted = sorted(lats)
        p99_idx = max(0, int(len(lats_sorted) * 0.99) - 1)

        return {
            "avg_ms": round(statistics.mean(lats), 3),
            "p99_ms": round(lats_sorted[p99_idx], 3),
            "min_ms": round(lats_sorted[0], 3),
            "max_ms": round(lats_sorted[-1], 3),
            "count": entry.call_count,
        }

    def export_sklearn_to_onnx(
        self,
        model: Any,
        name: str,
        input_shape: Tuple[int, ...],
    ) -> Optional[str]:
        """
        Export a scikit-learn model to ONNX format.

        Parameters
        ----------
        model : sklearn estimator
            A fitted scikit-learn model.
        name : str
            Output filename (without extension).
        input_shape : tuple of int
            Expected input shape, e.g. ``(1, 10)`` for 10 features.

        Returns
        -------
        str or None
            Path to the saved ``.onnx`` file, or None on failure.
        """
        if not _HAS_SKL2ONNX:
            logger.warning(
                "ONNXModelServer.export_sklearn_to_onnx: skl2onnx not installed — cannot export"
            )
            return None

        try:
            os.makedirs(self.models_dir, exist_ok=True)
            out_path = os.path.join(self.models_dir, f"{name}.onnx")

            initial_type = [("float_input", FloatTensorType([None, input_shape[-1]]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)

            with open(out_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info("ONNXModelServer: exported sklearn model to %s", out_path)
            return out_path

        except Exception:
            logger.error("ONNXModelServer: export failed for '%s'", name, exc_info=True)
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all loaded models with metadata.

        Returns
        -------
        list of dict
            Each dict has keys: name, path, input_name, input_shape,
            output_names, loaded_at, call_count.
        """
        with self._global_lock:
            entries = list(self._models.values())

        return [
            {
                "name": e.name,
                "path": e.path,
                "input_name": e.input_name,
                "input_shape": e.input_shape,
                "output_names": e.output_names,
                "loaded_at": e.loaded_at,
                "call_count": e.call_count,
            }
            for e in entries
        ]

    def unload_model(self, name: str) -> bool:
        """
        Unload a model and free its resources.

        Parameters
        ----------
        name : str
            Model identifier.

        Returns
        -------
        bool
            True if the model was found and removed.
        """
        with self._global_lock:
            entry = self._models.pop(name, None)

        if entry is None:
            return False

        logger.info("ONNXModelServer: unloaded model '%s'", name)
        return True
