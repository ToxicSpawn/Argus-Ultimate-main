"""
adaptive/auto_model_manager.py --- Automatic Model Lifecycle Management.

Monitors ML model health (staleness, accuracy, drift, data availability)
and produces concrete ModelAction recommendations: ok, retrain, replace,
or disable.  Optionally triggers retraining pipelines and schedules
nightly checks.

Usage::

    mgr = AutoModelManager(config=cfg_section)
    actions = mgr.check_all_models(model_registry)
    for a in actions:
        if a.action == "retrain":
            mgr.auto_retrain(a.model_name, trainer=my_trainer)

Standalone --- no hard imports on the rest of the ARGUS tree at module load.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelAction:
    """A lifecycle recommendation for a single model."""

    model_name: str
    action: str              # "ok" | "retrain" | "replace" | "disable"
    reason: str
    priority: int            # 1 = highest
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# AutoModelManager
# ---------------------------------------------------------------------------

class AutoModelManager:
    """Automatic model lifecycle management.

    Parameters
    ----------
    config : dict, optional
        ``auto_model_manager`` section from unified config.
    """

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._stale_days: int = int(cfg.get("stale_days", 7))
        self._drift_threshold: float = float(cfg.get("drift_threshold", 0.3))
        self._accuracy_drop_pct: float = float(cfg.get("accuracy_drop_pct", 10.0))
        self._min_accuracy: float = float(cfg.get("min_accuracy", 0.45))
        self._min_samples_for_disable: int = int(cfg.get("min_samples_for_disable", 100))
        self._new_data_threshold: int = int(cfg.get("new_data_threshold", 1000))
        self._nightly_check_hour: int = int(cfg.get("nightly_check_hour", 3))  # 3 AM
        self._retrain_history: List[Dict[str, Any]] = []
        self._last_check_ts: float = 0.0

        logger.info(
            "AutoModelManager initialised (stale=%dd, drift_thresh=%.2f, accuracy_drop=%.0f%%)",
            self._stale_days, self._drift_threshold, self._accuracy_drop_pct,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all_models(
        self,
        model_registry: Dict[str, Dict[str, Any]],
    ) -> List[ModelAction]:
        """Evaluate all registered models and return lifecycle actions.

        Parameters
        ----------
        model_registry : dict
            model_name -> {
                last_train_ts (float, epoch), accuracy (float 0-1),
                peak_accuracy (float), drift_score (float 0-1),
                samples_since_train (int), is_active (bool)
            }

        Returns
        -------
        list[ModelAction]
            Actions sorted by priority (1 = most urgent).
        """
        if not self._enabled:
            return []

        self._last_check_ts = time.time()
        actions: List[ModelAction] = []

        for name, m in model_registry.items():
            action = self._evaluate_model(name, m)
            if action:
                actions.append(action)

        actions.sort(key=lambda a: a.priority)

        if actions:
            logger.info(
                "AutoModelManager: %d of %d models need attention: %s",
                len(actions), len(model_registry),
                ", ".join(f"{a.model_name}({a.action})" for a in actions),
            )
        return actions

    def auto_retrain(
        self,
        model_name: str,
        *,
        trainer: Any = None,
    ) -> bool:
        """Trigger retraining for a model.

        Parameters
        ----------
        model_name : str
            Name of the model to retrain.
        trainer : object, optional
            An object with a ``retrain(model_name: str)`` method.

        Returns
        -------
        bool
            True if retrain was triggered successfully.
        """
        entry = {
            "model_name": model_name,
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "success": False,
        }

        if trainer is None:
            logger.warning(
                "auto_retrain('%s'): no trainer provided, logging request only.", model_name,
            )
            entry["reason"] = "no_trainer"
            self._retrain_history.append(entry)
            return False

        try:
            if hasattr(trainer, "retrain"):
                trainer.retrain(model_name)
            elif callable(trainer):
                trainer(model_name)
            else:
                logger.warning("Trainer object has no retrain() method or is not callable.")
                entry["reason"] = "invalid_trainer"
                self._retrain_history.append(entry)
                return False

            entry["success"] = True
            self._retrain_history.append(entry)
            logger.info("Successfully triggered retrain for model '%s'", model_name)
            return True
        except Exception:
            logger.exception("Failed to retrain model '%s'", model_name)
            entry["reason"] = "exception"
            self._retrain_history.append(entry)
            return False

    def schedule_nightly_check(self) -> Dict[str, Any]:
        """Return scheduling metadata for a nightly model health check.

        This does not actually register a cron job; it returns the parameters
        so the caller (e.g. SelfImprovementOrchestrator) can wire it up.
        """
        return {
            "hour": self._nightly_check_hour,
            "minute": 0,
            "callback": "auto_model_manager.check_all_models",
            "description": f"Nightly model health check at {self._nightly_check_hour:02d}:00 UTC",
        }

    @property
    def retrain_history(self) -> List[Dict[str, Any]]:
        return list(self._retrain_history)

    @property
    def last_check_ts(self) -> float:
        return self._last_check_ts

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        name: str,
        m: Dict[str, Any],
    ) -> Optional[ModelAction]:
        """Evaluate a single model and return an action if needed."""
        last_train_ts = float(m.get("last_train_ts", 0))
        accuracy = float(m.get("accuracy", 0.5))
        peak_accuracy = float(m.get("peak_accuracy", accuracy))
        drift_score = float(m.get("drift_score", 0.0))
        samples_since = int(m.get("samples_since_train", 0))
        is_active = bool(m.get("is_active", True))

        if not is_active:
            return None

        now = time.time()
        age_days = (now - last_train_ts) / 86400.0 if last_train_ts > 0 else 999.0

        reasons: List[str] = []
        best_priority = 5  # lower = more urgent

        # Check 1: Staleness
        if age_days > self._stale_days:
            reasons.append(f"stale ({age_days:.0f}d since last train, threshold={self._stale_days}d)")
            best_priority = min(best_priority, 3)

        # Check 2: Feature drift
        if drift_score > self._drift_threshold:
            reasons.append(f"feature drift {drift_score:.2f} > {self._drift_threshold:.2f}")
            best_priority = min(best_priority, 2)

        # Check 3: Accuracy degradation
        if peak_accuracy > 0:
            drop_pct = ((peak_accuracy - accuracy) / peak_accuracy) * 100
            if drop_pct > self._accuracy_drop_pct:
                reasons.append(
                    f"accuracy dropped {drop_pct:.0f}% from peak "
                    f"({peak_accuracy:.3f} -> {accuracy:.3f})"
                )
                best_priority = min(best_priority, 3)

        # Check 4: Consistently wrong -> disable
        if accuracy < self._min_accuracy and samples_since >= self._min_samples_for_disable:
            return ModelAction(
                model_name=name,
                action="disable",
                reason=(
                    f"Model '{name}' accuracy is {accuracy:.3f} "
                    f"(below {self._min_accuracy:.2f}) over {samples_since} samples. Disabling."
                ),
                priority=1,
            )

        # Check 5: New data available
        if samples_since >= self._new_data_threshold and not reasons:
            reasons.append(f"{samples_since} new samples available for retraining")
            best_priority = min(best_priority, 4)

        if reasons:
            return ModelAction(
                model_name=name,
                action="retrain",
                reason=f"Model '{name}': {'; '.join(reasons)}.",
                priority=best_priority,
            )

        return None
