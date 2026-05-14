"""
GPU-accelerated inference pipeline for batched ML predictions.

Registers models, batches feature vectors onto GPU tensors, runs forward
passes, and returns predictions with latency tracking.
Falls back gracefully to CPU/numpy when torch or CUDA unavailable.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

# Detect torch availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY_MB = torch.cuda.get_device_properties(0).total_memory / 1e6
    else:
        GPU_NAME = "N/A"
        GPU_MEMORY_MB = 0
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    GPU_NAME = "N/A"
    GPU_MEMORY_MB = 0
    torch = None

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class InferenceResult:
    """Result from a single model prediction."""
    prediction: Any
    confidence: float = 0.0
    latency_ms: float = 0.0
    device_used: str = "cpu"
    cache_hit: bool = False


@dataclass
class RegisteredModel:
    """Metadata for a registered model."""
    name: str
    model: Any
    input_dim: int
    model_type: str = "torch"  # "torch", "sklearn", "callable"
    predict_fn: Optional[Callable] = None
    total_predictions: int = 0
    total_latency_ms: float = 0.0


class GPUInferencePipeline:
    """Batched GPU inference for all ML models."""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            if CUDA_AVAILABLE:
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = device

        self._models: Dict[str, RegisteredModel] = {}
        self._total_predictions = 0
        self._total_latency_ms = 0.0
        self._oom_fallbacks = 0

        logger.info(
            "GPUInferencePipeline: device=%s, torch=%s, cuda=%s, gpu=%s (%.0f MB)",
            self._device, TORCH_AVAILABLE, CUDA_AVAILABLE, GPU_NAME, GPU_MEMORY_MB,
        )

    @property
    def device(self) -> str:
        return self._device

    def register_model(self, name: str, model: Any, input_dim: int = 0,
                       model_type: str = "auto", predict_fn: Optional[Callable] = None):
        """Register a model for batched inference."""
        if model_type == "auto":
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                model_type = "torch"
            elif predict_fn is not None:
                model_type = "callable"
            elif hasattr(model, "predict"):
                model_type = "sklearn"
            else:
                model_type = "callable"

        self._models[name] = RegisteredModel(
            name=name, model=model, input_dim=input_dim,
            model_type=model_type, predict_fn=predict_fn,
        )
        logger.info("GPUInferencePipeline: registered model '%s' (type=%s, dim=%d)", name, model_type, input_dim)

    def unregister_model(self, name: str):
        """Remove a registered model."""
        self._models.pop(name, None)

    def predict_batch(self, model_name: str, features_list: List[Any]) -> List[InferenceResult]:
        """Batch predict using registered model."""
        if model_name not in self._models:
            logger.warning("GPUInferencePipeline: model '%s' not registered", model_name)
            return [InferenceResult(prediction=None, confidence=0.0) for _ in features_list]

        if not features_list:
            return []

        reg = self._models[model_name]
        t0 = time.monotonic()
        results = []

        try:
            if reg.model_type == "torch" and TORCH_AVAILABLE:
                results = self._predict_torch(reg, features_list)
            elif reg.model_type == "sklearn":
                results = self._predict_sklearn(reg, features_list)
            elif reg.model_type == "callable" and reg.predict_fn is not None:
                results = self._predict_callable(reg, features_list)
            else:
                # Fallback: try predict method
                results = self._predict_generic(reg, features_list)
        except Exception as exc:
            logger.error("GPUInferencePipeline: batch predict failed for '%s': %s", model_name, exc)
            results = [InferenceResult(prediction=None, confidence=0.0) for _ in features_list]

        elapsed_ms = (time.monotonic() - t0) * 1000
        reg.total_predictions += len(features_list)
        reg.total_latency_ms += elapsed_ms
        self._total_predictions += len(features_list)
        self._total_latency_ms += elapsed_ms

        # Stamp latency on results
        per_item_ms = elapsed_ms / max(len(results), 1)
        for r in results:
            r.latency_ms = per_item_ms

        return results

    def _predict_torch(self, reg: RegisteredModel, features_list: List[Any]) -> List[InferenceResult]:
        """Run batched torch inference on GPU."""
        device = self._device

        # Convert to tensor
        if np is not None and isinstance(features_list[0], np.ndarray):
            batch = torch.tensor(np.stack(features_list), dtype=torch.float32)
        elif isinstance(features_list[0], (list, tuple)):
            batch = torch.tensor(features_list, dtype=torch.float32)
        elif isinstance(features_list[0], torch.Tensor):
            batch = torch.stack(features_list)
        else:
            batch = torch.tensor(features_list, dtype=torch.float32)

        try:
            batch = batch.to(device)
            reg.model.to(device)
            reg.model.eval()
            with torch.no_grad():
                output = reg.model(batch)

            # Convert output to predictions
            if output.dim() == 1:
                preds = output.cpu().tolist()
            elif output.dim() == 2 and output.shape[1] == 1:
                preds = output.squeeze(1).cpu().tolist()
            else:
                # Classification: argmax + softmax confidence
                confs = torch.softmax(output, dim=1)
                pred_classes = torch.argmax(confs, dim=1).cpu().tolist()
                pred_confs = confs.max(dim=1).values.cpu().tolist()
                return [
                    InferenceResult(prediction=p, confidence=c, device_used=device)
                    for p, c in zip(pred_classes, pred_confs)
                ]

            return [
                InferenceResult(prediction=p, confidence=abs(p), device_used=device)
                for p in preds
            ]

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._oom_fallbacks += 1
                logger.warning("GPUInferencePipeline: CUDA OOM, falling back to CPU")
                torch.cuda.empty_cache()
                return self._predict_torch_cpu(reg, features_list)
            raise

    def _predict_torch_cpu(self, reg: RegisteredModel, features_list: List[Any]) -> List[InferenceResult]:
        """CPU fallback for torch models."""
        if np is not None and isinstance(features_list[0], np.ndarray):
            batch = torch.tensor(np.stack(features_list), dtype=torch.float32)
        else:
            batch = torch.tensor(features_list, dtype=torch.float32)

        reg.model.to("cpu")
        reg.model.eval()
        with torch.no_grad():
            output = reg.model(batch)

        preds = output.squeeze().cpu().tolist()
        if isinstance(preds, (int, float)):
            preds = [preds]
        return [
            InferenceResult(prediction=p, confidence=abs(p) if isinstance(p, (int, float)) else 0.0, device_used="cpu")
            for p in preds
        ]

    def _predict_sklearn(self, reg: RegisteredModel, features_list: List[Any]) -> List[InferenceResult]:
        """Sklearn model batch predict."""
        if np is not None:
            X = np.array(features_list)
        else:
            X = features_list

        preds = reg.model.predict(X)
        confs = [0.5] * len(preds)
        if hasattr(reg.model, "predict_proba"):
            try:
                probs = reg.model.predict_proba(X)
                prob_confs = [float(max(row)) for row in probs]
                if len(prob_confs) == len(preds):
                    confs = prob_confs
            except Exception as _e:
                logger.debug("gpu_inference error: %s", _e)

        return [
            InferenceResult(prediction=p, confidence=c, device_used="cpu")
            for p, c in zip(list(preds), confs)
        ]

    def _predict_callable(self, reg: RegisteredModel, features_list: List[Any]) -> List[InferenceResult]:
        """Callable function batch predict."""
        results = []
        for feat in features_list:
            try:
                pred = reg.predict_fn(feat)
                if isinstance(pred, dict):
                    results.append(InferenceResult(
                        prediction=pred.get("prediction", pred),
                        confidence=float(pred.get("confidence", 0.5)),
                        device_used="cpu",
                    ))
                else:
                    results.append(InferenceResult(prediction=pred, confidence=0.5, device_used="cpu"))
            except Exception as exc:
                logger.debug("GPUInferencePipeline: callable predict failed: %s", exc)
                results.append(InferenceResult(prediction=None, confidence=0.0))
        return results

    def _predict_generic(self, reg: RegisteredModel, features_list: List[Any]) -> List[InferenceResult]:
        """Generic fallback — try model.predict()."""
        if hasattr(reg.model, "predict"):
            return self._predict_sklearn(reg, features_list)
        return [InferenceResult(prediction=None, confidence=0.0) for _ in features_list]

    def predict_all_models(self, features_dict: Dict[str, List[Any]]) -> Dict[str, List[InferenceResult]]:
        """Run all registered models on their respective features."""
        results = {}
        for model_name, features in features_dict.items():
            if model_name in self._models:
                results[model_name] = self.predict_batch(model_name, features)
        return results

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        model_stats = {}
        for name, reg in self._models.items():
            avg_ms = reg.total_latency_ms / max(reg.total_predictions, 1)
            model_stats[name] = {
                "type": reg.model_type,
                "predictions": reg.total_predictions,
                "avg_latency_ms": round(avg_ms, 3),
            }

        gpu_mem_used = 0.0
        if CUDA_AVAILABLE:
            try:
                gpu_mem_used = torch.cuda.memory_allocated() / 1e6
            except Exception as _e:
                logger.debug("gpu_inference error: %s", _e)

        return {
            "device": self._device,
            "torch_available": TORCH_AVAILABLE,
            "cuda_available": CUDA_AVAILABLE,
            "gpu_name": GPU_NAME,
            "gpu_memory_total_mb": round(GPU_MEMORY_MB, 0),
            "gpu_memory_used_mb": round(gpu_mem_used, 1),
            "models_loaded": len(self._models),
            "total_predictions": self._total_predictions,
            "avg_latency_ms": round(self._total_latency_ms / max(self._total_predictions, 1), 3),
            "oom_fallbacks": self._oom_fallbacks,
            "models": model_stats,
        }
