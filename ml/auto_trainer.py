"""
Automated model training scheduler.

Registers ML models with training functions and intervals, checks which
are due for retraining each cycle, and runs training in background threads
so it never blocks the trading loop.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Detect GPU
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = "cpu"
    torch = None


@dataclass
class TrainingSchedule:
    """Schedule for a registered model."""
    model_name: str
    trainer_func: Callable
    interval_hours: float = 24.0
    min_samples: int = 1000
    last_trained: float = 0.0  # monotonic timestamp
    last_trained_wall: float = 0.0  # wall clock timestamp
    training_count: int = 0
    last_duration_seconds: float = 0.0
    is_training: bool = False
    last_error: Optional[str] = None
    samples_available: int = 0


@dataclass
class TrainingResult:
    """Result from a training run."""
    model_name: str
    success: bool
    duration_seconds: float = 0.0
    error: Optional[str] = None
    device_used: str = "cpu"
    samples_used: int = 0


class AutoTrainer:
    """Automated model training scheduler with GPU support."""

    def __init__(self, models_dir: str = "models/", device: str = "auto"):
        self._models_dir = models_dir
        self._device = DEVICE if device == "auto" else device
        self._schedules: Dict[str, TrainingSchedule] = {}
        self._training_lock = asyncio.Lock()
        self._results_history: List[TrainingResult] = []
        self._max_history = 100

        logger.info("AutoTrainer: device=%s, models_dir=%s", self._device, models_dir)

    def register(self, model_name: str, trainer_func: Callable,
                 interval_hours: float = 24.0, min_samples: int = 1000):
        """Register a model for periodic retraining."""
        self._schedules[model_name] = TrainingSchedule(
            model_name=model_name,
            trainer_func=trainer_func,
            interval_hours=interval_hours,
            min_samples=min_samples,
        )
        logger.info(
            "AutoTrainer: registered '%s' (every %.1f hours, min %d samples)",
            model_name, interval_hours, min_samples,
        )

    def unregister(self, model_name: str):
        """Remove a model from training schedule."""
        self._schedules.pop(model_name, None)

    def update_samples(self, model_name: str, samples: int):
        """Update available sample count for a model."""
        if model_name in self._schedules:
            self._schedules[model_name].samples_available = samples

    def is_due(self, model_name: str) -> bool:
        """Check if a model is due for retraining."""
        if model_name not in self._schedules:
            return False
        sched = self._schedules[model_name]
        if sched.is_training:
            return False
        if sched.samples_available < sched.min_samples:
            return False
        if sched.last_trained == 0:
            return True  # Never trained
        elapsed_hours = (time.monotonic() - sched.last_trained) / 3600.0
        return elapsed_hours >= sched.interval_hours

    def get_due_models(self) -> List[str]:
        """Return list of model names due for retraining."""
        return [name for name in self._schedules if self.is_due(name)]

    async def check_and_train(self) -> List[TrainingResult]:
        """Check all models, retrain any that are due. Non-blocking."""
        due = self.get_due_models()
        if not due:
            return []

        results = []
        for model_name in due:
            result = await self._train_model(model_name)
            results.append(result)

        return results

    async def _train_model(self, model_name: str) -> TrainingResult:
        """Train a single model in background thread."""
        sched = self._schedules[model_name]

        if sched.is_training:
            return TrainingResult(
                model_name=model_name,
                success=False,
                error="already_training",
            )

        async with self._training_lock:
            sched.is_training = True
            t0 = time.monotonic()

            try:
                loop = asyncio.get_running_loop()
                # Run training function in thread pool (non-blocking)
                await loop.run_in_executor(None, sched.trainer_func)

                duration = time.monotonic() - t0
                sched.last_trained = time.monotonic()
                sched.last_trained_wall = time.time()
                sched.training_count += 1
                sched.last_duration_seconds = duration
                sched.last_error = None

                result = TrainingResult(
                    model_name=model_name,
                    success=True,
                    duration_seconds=duration,
                    device_used=self._device,
                    samples_used=sched.samples_available,
                )

                logger.info(
                    "AutoTrainer: '%s' trained successfully in %.1fs on %s (%d samples)",
                    model_name, duration, self._device, sched.samples_available,
                )

            except Exception as exc:
                duration = time.monotonic() - t0
                sched.last_error = str(exc)

                result = TrainingResult(
                    model_name=model_name,
                    success=False,
                    duration_seconds=duration,
                    error=str(exc),
                )

                logger.error(
                    "AutoTrainer: '%s' training failed after %.1fs: %s",
                    model_name, duration, exc,
                )

            finally:
                sched.is_training = False

            self._results_history.append(result)
            if len(self._results_history) > self._max_history:
                self._results_history = self._results_history[-self._max_history:]

            return result

    def force_retrain(self, model_name: str) -> bool:
        """Mark a model for immediate retraining (next check_and_train call)."""
        if model_name not in self._schedules:
            return False
        self._schedules[model_name].last_trained = 0
        self._schedules[model_name].samples_available = max(
            self._schedules[model_name].samples_available,
            self._schedules[model_name].min_samples,
        )
        logger.info("AutoTrainer: '%s' marked for immediate retraining", model_name)
        return True

    def get_schedule(self) -> dict:
        """Return training schedule for all models."""
        result = {}
        now = time.monotonic()
        for name, sched in self._schedules.items():
            hours_since = (now - sched.last_trained) / 3600.0 if sched.last_trained > 0 else float("inf")
            hours_until = max(0, sched.interval_hours - hours_since)
            result[name] = {
                "interval_hours": sched.interval_hours,
                "min_samples": sched.min_samples,
                "samples_available": sched.samples_available,
                "last_trained_ago_hours": round(hours_since, 2),
                "next_due_hours": round(hours_until, 2),
                "training_count": sched.training_count,
                "last_duration_seconds": round(sched.last_duration_seconds, 1),
                "is_training": sched.is_training,
                "last_error": sched.last_error,
                "is_due": self.is_due(name),
            }
        return result

    def get_history(self, limit: int = 20) -> List[dict]:
        """Return recent training results."""
        return [
            {
                "model": r.model_name,
                "success": r.success,
                "duration_s": round(r.duration_seconds, 1),
                "device": r.device_used,
                "error": r.error,
            }
            for r in self._results_history[-limit:]
        ]

    def snapshot(self) -> dict:
        """Full state snapshot for dashboard/monitoring."""
        return {
            "device": self._device,
            "cuda_available": CUDA_AVAILABLE,
            "registered_models": len(self._schedules),
            "total_trainings": sum(s.training_count for s in self._schedules.values()),
            "currently_training": [n for n, s in self._schedules.items() if s.is_training],
            "due_for_training": self.get_due_models(),
            "schedule": self.get_schedule(),
        }
