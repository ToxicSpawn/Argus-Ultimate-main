"""Feature registry for metadata, schema, dependencies, and quality monitoring."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class FeatureDefinition:
    name: str
    dtype: str
    description: str = ""
    owner: str = "ml"
    tags: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    freshness_sla_seconds: int = 60
    entity_keys: List[str] = field(default_factory=lambda: ["symbol"])


@dataclass(slots=True)
class FeatureQualityRecord:
    feature_name: str
    timestamp: datetime
    null_rate: float
    freshness_lag_seconds: float
    drift_score: float
    notes: str = ""


class FeatureRegistry:
    """In-memory registry for streaming feature metadata and monitoring."""

    def __init__(self) -> None:
        self._definitions: Dict[str, FeatureDefinition] = {}
        self._quality_history: Dict[str, List[FeatureQualityRecord]] = {}
        self._lock = threading.RLock()

    def register(self, definition: FeatureDefinition) -> None:
        with self._lock:
            self._definitions[definition.name] = definition
            self._quality_history.setdefault(definition.name, [])
        logger.info("Registered feature %s dtype=%s", definition.name, definition.dtype)

    def get(self, feature_name: str) -> Optional[FeatureDefinition]:
        with self._lock:
            return self._definitions.get(feature_name)

    def list_features(self) -> List[FeatureDefinition]:
        with self._lock:
            return list(self._definitions.values())

    def search(self, query: str = "", *, tags: Optional[Mapping[str, str]] = None) -> List[FeatureDefinition]:
        query_lower = query.lower().strip()
        tags = dict(tags or {})
        results: List[FeatureDefinition] = []
        with self._lock:
            for definition in self._definitions.values():
                if query_lower and query_lower not in definition.name.lower() and query_lower not in definition.description.lower():
                    continue
                if tags and any(definition.tags.get(key) != value for key, value in tags.items()):
                    continue
                results.append(definition)
        return results

    def dependency_graph(self) -> Dict[str, List[str]]:
        with self._lock:
            return {name: list(definition.dependencies) for name, definition in self._definitions.items()}

    def dependencies_for(self, feature_name: str) -> List[str]:
        definition = self.get(feature_name)
        return list(definition.dependencies) if definition else []

    def record_quality(self, record: FeatureQualityRecord) -> None:
        with self._lock:
            self._quality_history.setdefault(record.feature_name, []).append(record)
            self._quality_history[record.feature_name] = self._quality_history[record.feature_name][-1000:]

    def latest_quality(self, feature_name: str) -> Optional[FeatureQualityRecord]:
        with self._lock:
            records = self._quality_history.get(feature_name, [])
            return records[-1] if records else None

    def quality_summary(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for feature_name, records in self._quality_history.items():
                if not records:
                    continue
                latest = records[-1]
                summary[feature_name] = {
                    "latest_timestamp": latest.timestamp.isoformat(),
                    "null_rate": latest.null_rate,
                    "freshness_lag_seconds": latest.freshness_lag_seconds,
                    "drift_score": latest.drift_score,
                }
        return summary
