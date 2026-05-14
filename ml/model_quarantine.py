"""Drift-triggered model quarantine and fallback routing.

This module is intentionally small and ML-layer only. It records degraded model
versions, checks whether a model should be avoided, and returns a safe fallback
model id without coupling to the large trading coordinator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from argus_live.config.quarantine import ConfigQuarantineStore, QuarantineRecord


@dataclass
class ModelRouteDecision:
    """Routing result for one model selection request."""

    requested_model_id: str
    active_model_id: str
    quarantined: bool
    fallback_used: bool
    reason: str
    config_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_model_id": self.requested_model_id,
            "active_model_id": self.active_model_id,
            "quarantined": self.quarantined,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "config_hash": self.config_hash,
        }


class ModelQuarantineManager:
    """Persist quarantine records and route degraded models to fallbacks."""

    def __init__(
        self,
        path: str | Path = "data/model_quarantine.jsonl",
        fallback_models: Optional[Dict[str, str]] = None,
        drift_threshold: float = 0.5,
        error_rate_threshold: float = 0.3,
        confidence_threshold: float = 0.2,
    ) -> None:
        self.store = ConfigQuarantineStore(path)
        self.fallback_models = dict(fallback_models or {})
        self.drift_threshold = float(drift_threshold)
        self.error_rate_threshold = float(error_rate_threshold)
        self.confidence_threshold = float(confidence_threshold)

    def config_hash_for(
        self,
        model_id: str,
        *,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        feature_hash: Optional[str] = None,
    ) -> str:
        parts = [model_name or "", model_id, version or "", feature_hash or ""]
        payload = ":".join(str(part) for part in parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def register_fallback(self, model_id: str, fallback_model_id: str) -> None:
        self.fallback_models[str(model_id)] = str(fallback_model_id)

    def should_quarantine(self, health: Any) -> tuple[bool, List[str]]:
        """Evaluate a ModelHealth-like object or dict for quarantine triggers."""
        data = self._health_to_dict(health)
        reasons: List[str] = []

        status = str(data.get("status", "")).lower()
        if status == "critical":
            reasons.append("critical_health_status")

        drift_score = float(data.get("drift_score", 0.0) or 0.0)
        if drift_score >= self.drift_threshold:
            reasons.append(f"drift_score_{drift_score:.3f}")

        error_rate = float(data.get("error_rate", 0.0) or 0.0)
        if error_rate >= self.error_rate_threshold:
            reasons.append(f"error_rate_{error_rate:.3f}")

        avg_confidence = float(data.get("avg_confidence", 1.0) or 0.0)
        if avg_confidence <= self.confidence_threshold:
            reasons.append(f"low_confidence_{avg_confidence:.3f}")

        return bool(reasons), reasons

    def quarantine_model(
        self,
        *,
        run_id: str,
        model_id: str,
        reasons: List[str],
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        feature_hash: Optional[str] = None,
        decision: str = "quarantine",
    ) -> QuarantineRecord:
        config_hash = self.config_hash_for(
            model_id,
            model_name=model_name,
            version=version,
            feature_hash=feature_hash,
        )
        return self.store.quarantine(
            run_id=run_id,
            config_hash=config_hash,
            decision=decision,
            reasons=reasons,
        )

    def route_model(
        self,
        model_id: str,
        *,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        feature_hash: Optional[str] = None,
    ) -> ModelRouteDecision:
        config_hash = self.config_hash_for(
            model_id,
            model_name=model_name,
            version=version,
            feature_hash=feature_hash,
        )
        if not self.store.is_quarantined(config_hash):
            return ModelRouteDecision(model_id, model_id, False, False, "model_allowed", config_hash)

        fallback = self.fallback_models.get(model_id)
        if fallback:
            return ModelRouteDecision(model_id, fallback, True, True, "quarantined_using_fallback", config_hash)
        return ModelRouteDecision(model_id, model_id, True, False, "quarantined_no_fallback", config_hash)

    def evaluate_and_route(
        self,
        *,
        run_id: str,
        model_id: str,
        health: Any,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        feature_hash: Optional[str] = None,
    ) -> ModelRouteDecision:
        should_quarantine, reasons = self.should_quarantine(health)
        if should_quarantine:
            self.quarantine_model(
                run_id=run_id,
                model_id=model_id,
                reasons=reasons,
                model_name=model_name,
                version=version,
                feature_hash=feature_hash,
            )
        return self.route_model(
            model_id,
            model_name=model_name,
            version=version,
            feature_hash=feature_hash,
        )

    @staticmethod
    def _health_to_dict(health: Any) -> Dict[str, Any]:
        if isinstance(health, dict):
            return health
        if hasattr(health, "to_dict"):
            return health.to_dict()
        return {
            "status": getattr(health, "status", "unknown"),
            "drift_score": getattr(health, "drift_score", 0.0),
            "error_rate": getattr(health, "error_rate", 0.0),
            "avg_confidence": getattr(health, "avg_confidence", 1.0),
        }
