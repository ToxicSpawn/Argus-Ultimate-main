"""Train-serve consistency checks and serving-time validation for streaming features."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from typing import Any
from collections.abc import Mapping, Sequence

from .feature_registry import FeatureRegistry, FeatureQualityRecord

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ValidationResult:
    entity_id: str
    valid: bool
    missing_features: List[str] = field(default_factory=list)
    stale_features: List[str] = field(default_factory=list)
    schema_violations: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class ConsistencyAlert:
    feature_name: str
    message: str
    severity: str
    created_at: datetime = field(default_factory=_utc_now)


class ConsistencyGuarantor:
    """Prevents train-serve skew and validates features before inference."""

    def __init__(self, *, registry: FeatureRegistry, drift_threshold: float = 0.2) -> None:
        self.registry = registry
        self.drift_threshold = float(drift_threshold)
        self._alerts: List[ConsistencyAlert] = []

    def validate_serving_payload(
        self,
        entity_id: str,
        payload: Mapping[str, Any],
        *,
        feature_timestamps: Optional[Mapping[str, datetime]] = None,
        required_features: Optional[Sequence[str]] = None,
    ) -> ValidationResult:
        feature_timestamps = dict(feature_timestamps or {})
        missing: List[str] = []
        stale: List[str] = []
        schema_violations: List[str] = []
        now = _utc_now()

        if required_features is None:
            definitions = self.registry.list_features()
        else:
            definitions = [definition for definition in self.registry.list_features() if definition.name in set(required_features)]

        for definition in definitions:
            if definition.name not in payload:
                missing.append(definition.name)
                continue
            value = payload[definition.name]
            if not self._matches_dtype(value, definition.dtype):
                schema_violations.append(f"{definition.name}: expected {definition.dtype}")
            ts = feature_timestamps.get(definition.name)
            if ts is not None and (now - ts).total_seconds() > definition.freshness_sla_seconds:
                stale.append(definition.name)

        valid = not missing and not stale and not schema_violations
        result = ValidationResult(
            entity_id=entity_id,
            valid=valid,
            missing_features=missing,
            stale_features=stale,
            schema_violations=schema_violations,
        )
        if not valid:
            logger.warning("Serving validation failed for %s missing=%s stale=%s schema=%s", entity_id, missing, stale, schema_violations)
        return result

    def detect_train_serve_skew(
        self,
        *,
        feature_name: str,
        training_values: Sequence[float],
        serving_values: Sequence[float],
    ) -> float:
        train = [float(value) for value in training_values]
        serve = [float(value) for value in serving_values]
        if not train or not serve:
            return 0.0
        train_mean = sum(train) / len(train)
        serve_mean = sum(serve) / len(serve)
        train_std = math.sqrt(sum((value - train_mean) ** 2 for value in train) / len(train))
        serve_std = math.sqrt(sum((value - serve_mean) ** 2 for value in serve) / len(serve))
        mean_gap = abs(train_mean - serve_mean) / max(abs(train_mean), 1e-12)
        std_gap = abs(train_std - serve_std) / max(abs(train_std), 1e-12)
        score = float(max(mean_gap, std_gap))
        if score >= self.drift_threshold:
            message = f"Train-serve skew detected for {feature_name}: score={score:.4f}"
            self._alerts.append(ConsistencyAlert(feature_name=feature_name, message=message, severity="warning"))
            self.registry.record_quality(
                FeatureQualityRecord(
                    feature_name=feature_name,
                    timestamp=_utc_now(),
                    null_rate=0.0,
                    freshness_lag_seconds=0.0,
                    drift_score=score,
                    notes="train_serve_skew",
                )
            )
            logger.warning(message)
        return score

    def drift_alerts(self) -> List[ConsistencyAlert]:
        return list(self._alerts)

    def _matches_dtype(self, value: Any, dtype: str) -> bool:
        dtype_upper = dtype.upper()
        if dtype_upper in {"FLOAT", "DOUBLE", "INT", "INT64"}:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if dtype_upper in {"STRING", "STR"}:
            return isinstance(value, str)
        if dtype_upper in {"BOOL", "BOOLEAN"}:
            return isinstance(value, bool)
        return True
