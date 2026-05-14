"""
ModelPerformanceExporter — Bridges model_manager._registry → advisory.

Reads ModelManager._registry every export_interval_cycles cycles and
produces a structured advisory["model_performance"] payload containing:
  - Per-model: is_loaded, is_stale, staleness_score, last_accuracy,
               trained_at, version
  - Aggregates: models_loaded, models_stale, avg_staleness,
                retrain_queue_size, from_cache

Output: advisory["model_performance"]
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelPerformanceSnapshot:
    models_loaded: int
    models_stale: int
    avg_staleness: float        # 0.0 (fresh) → 1.0 (fully stale)
    retrain_queue_size: int
    from_cache: bool
    per_model: Dict[str, Dict[str, Any]]
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ModelPerformanceExporter
# ---------------------------------------------------------------------------

class ModelPerformanceExporter:
    """
    Exports model_manager._registry contents to advisory.

    Parameters
    ----------
    model_manager           : ModelManager instance (optional)
    export_interval_cycles  : how often to re-scan (default 50)
    max_staleness_hours     : hours until staleness_score reaches 1.0 (default 168)
    config                  : optional config object
    """

    def __init__(
        self,
        model_manager: Optional[Any] = None,
        export_interval_cycles: int = 50,
        max_staleness_hours: float = 168.0,
        config: Optional[Any] = None,
    ) -> None:
        self.model_manager           = model_manager
        self.export_interval_cycles  = max(1, int(export_interval_cycles))
        self.max_staleness_hours     = max(1.0, float(max_staleness_hours))
        self.config                  = config

        self._last_export_cycle: int = -999999
        self._last_snapshot: Optional[ModelPerformanceSnapshot] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def export(self, cycle: int) -> Dict[str, Any]:
        """
        Scan model_manager and return advisory payload.
        Rate-limited to export_interval_cycles.
        Returns cached snapshot when called too frequently.
        """
        is_fresh = (cycle - self._last_export_cycle) < self.export_interval_cycles

        if is_fresh and self._last_snapshot is not None:
            snap = self._last_snapshot
            return self._to_dict(snap, from_cache=True)

        # Perform fresh scan
        snap = self._build_snapshot()
        self._last_snapshot = snap
        self._last_export_cycle = cycle
        return self._to_dict(snap, from_cache=False)

    def snapshot(self) -> Dict[str, Any]:
        """Return last cached advisory payload (or empty defaults)."""
        if self._last_snapshot is None:
            return {
                "models_loaded": 0,
                "models_stale": 0,
                "avg_staleness": 0.0,
                "retrain_queue_size": 0,
                "from_cache": False,
                "per_model": {},
            }
        return self._to_dict(self._last_snapshot, from_cache=True)

    # ── Internal ────────────────────────────────────────────────────────────

    def _build_snapshot(self) -> ModelPerformanceSnapshot:
        """Scan model_manager._registry and build a snapshot."""
        per_model: Dict[str, Dict[str, Any]] = {}

        if self.model_manager is None:
            return ModelPerformanceSnapshot(
                models_loaded=0,
                models_stale=0,
                avg_staleness=0.0,
                retrain_queue_size=0,
                from_cache=False,
                per_model={},
            )

        registry: Dict[str, Any] = {}
        try:
            registry = getattr(self.model_manager, "_registry", {}) or {}
        except Exception as exc:
            logger.debug("ModelPerformanceExporter: cannot read _registry: %s", exc)

        models_loaded = 0
        models_stale  = 0
        staleness_scores: List[float] = []

        for name, meta in registry.items():
            try:
                is_loaded   = bool(meta.get("loaded", False))
                is_stale    = bool(meta.get("is_stale", False))
                last_acc    = float(meta.get("last_accuracy", 0.0) or 0.0)
                trained_at  = meta.get("trained_at")
                version     = str(meta.get("version", "unknown") or "unknown")

                stale_score = self._staleness_score(meta)
                staleness_scores.append(stale_score)

                if is_loaded:
                    models_loaded += 1
                if is_stale:
                    models_stale += 1

                per_model[str(name)] = {
                    "is_loaded":       is_loaded,
                    "is_stale":        is_stale,
                    "staleness_score": round(stale_score, 4),
                    "last_accuracy":   round(last_acc, 4),
                    "trained_at":      trained_at,
                    "version":         version,
                }
            except Exception as exc:
                logger.debug(
                    "ModelPerformanceExporter: error reading model %s: %s", name, exc
                )

        avg_staleness = (
            sum(staleness_scores) / len(staleness_scores) if staleness_scores else 0.0
        )

        # Retrain queue size
        retrain_queue_size = 0
        try:
            rq = getattr(self.model_manager, "_retrain_queue", None)
            if rq is not None:
                retrain_queue_size = len(rq)
        except Exception:
            pass

        return ModelPerformanceSnapshot(
            models_loaded      = models_loaded,
            models_stale       = models_stale,
            avg_staleness      = round(avg_staleness, 4),
            retrain_queue_size = retrain_queue_size,
            from_cache         = False,
            per_model          = per_model,
        )

    def _staleness_score(self, meta: Dict[str, Any]) -> float:
        """
        Compute staleness score 0.0 (fresh) → 1.0 (max_staleness_hours old).
        Uses meta["trained_at"] (Unix timestamp) if available.
        Falls back to meta["is_stale"] boolean (0.5 if stale).
        """
        trained_at = meta.get("trained_at")
        if trained_at is not None:
            try:
                age_hours = (time.time() - float(trained_at)) / 3600.0
                return min(1.0, max(0.0, age_hours / self.max_staleness_hours))
            except (TypeError, ValueError):
                pass
        # Fallback: boolean staleness
        if meta.get("is_stale"):
            return 0.5
        return 0.0

    @staticmethod
    def _to_dict(snap: ModelPerformanceSnapshot, from_cache: bool) -> Dict[str, Any]:
        return {
            "models_loaded":       snap.models_loaded,
            "models_stale":        snap.models_stale,
            "avg_staleness":       snap.avg_staleness,
            "retrain_queue_size":  snap.retrain_queue_size,
            "from_cache":          from_cache,
            "per_model":           snap.per_model,
            "ts":                  snap.ts,
        }
