"""End-to-end streaming feature pipeline orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .consistency_guarantor import ConsistencyGuarantor, ValidationResult
from .feature_computation import FeatureComputationEngine, FeatureFrame
from .feature_registry import FeatureDefinition, FeatureQualityRecord, FeatureRegistry
from .offline_store import FeatureLineageRecord, OfflineFeatureStore
from .online_store import RedisOnlineFeatureStore
from .streaming_ingestion import StreamingIngestionService

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class PipelineHealth:
    running: bool = False
    last_success_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    processed_events: int = 0
    failed_events: int = 0
    retries: int = 0
    validation_failures: int = 0
    avg_cycle_latency_ms: float = 0.0


class StreamingFeaturePipeline:
    """Coordinates ingestion, feature computation, online/offline storage, and validation."""

    def __init__(
        self,
        *,
        ingestion: Optional[StreamingIngestionService] = None,
        computation: Optional[FeatureComputationEngine] = None,
        online_store: Optional[RedisOnlineFeatureStore] = None,
        offline_store: Optional[OfflineFeatureStore] = None,
        registry: Optional[FeatureRegistry] = None,
        consistency: Optional[ConsistencyGuarantor] = None,
        max_retries: int = 3,
        online_ttl_seconds: int = 300,
    ) -> None:
        self.ingestion = ingestion or StreamingIngestionService()
        self.computation = computation or FeatureComputationEngine()
        self.online_store = online_store or RedisOnlineFeatureStore(default_ttl_seconds=online_ttl_seconds)
        self.offline_store = offline_store or OfflineFeatureStore()
        self.registry = registry or FeatureRegistry()
        self.consistency = consistency or ConsistencyGuarantor(registry=self.registry)
        self.max_retries = max(1, int(max_retries))
        self.online_ttl_seconds = int(online_ttl_seconds)
        self.health = PipelineHealth()
        self._latency_samples_ms: List[float] = []

    def bootstrap_default_registry(self) -> None:
        defaults = [
            FeatureDefinition(name="rsi_14", dtype="FLOAT", description="14-period RSI", tags={"category": "technical"}),
            FeatureDefinition(name="macd", dtype="FLOAT", description="MACD line", tags={"category": "technical"}),
            FeatureDefinition(name="bollinger_width", dtype="FLOAT", description="Bollinger band width", tags={"category": "technical"}),
            FeatureDefinition(name="order_book_spread_bps", dtype="FLOAT", description="Top-of-book spread in bps", tags={"category": "order_book"}),
            FeatureDefinition(name="order_book_imbalance", dtype="FLOAT", description="Depth imbalance", tags={"category": "order_book"}),
        ]
        for definition in defaults:
            if self.registry.get(definition.name) is None:
                self.registry.register(definition)

    def process_once(self) -> int:
        start = time.perf_counter()
        self.health.running = True
        processed = 0
        try:
            processed += self._process_events(self.ingestion.consume_market_data(), event_type="market_data")
            processed += self._process_events(self.ingestion.consume_trade_events(), event_type="trade_events")
            processed += self._process_events(self.ingestion.consume_order_book_updates(), event_type="order_book")
            self.health.last_success_at = _utc_now()
            return processed
        except Exception:
            self.health.last_error_at = _utc_now()
            logger.exception("Streaming feature pipeline cycle failed")
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._latency_samples_ms.append(elapsed_ms)
            self._latency_samples_ms[:] = self._latency_samples_ms[-1000:]
            if self._latency_samples_ms:
                self.health.avg_cycle_latency_ms = sum(self._latency_samples_ms) / len(self._latency_samples_ms)

    def _process_events(self, events: Sequence[Any], *, event_type: str) -> int:
        processed = 0
        for event in events:
            retries = 0
            while True:
                try:
                    frame = self._compute_frame(event_type=event_type, payload=event.payload)
                    self._persist_frame(frame)
                    processed += 1
                    self.health.processed_events += 1
                    break
                except Exception as exc:
                    retries += 1
                    self.health.failed_events += 1
                    self.health.last_error_at = _utc_now()
                    if retries >= self.max_retries:
                        self.ingestion.handle_processing_error(event, exc)
                        logger.exception("Dropped event after %d retries topic=%s", retries, event.topic)
                        break
                    self.health.retries += 1
                    logger.warning("Retrying event topic=%s attempt=%d error=%s", event.topic, retries, exc)
        return processed

    def _compute_frame(self, *, event_type: str, payload: Mapping[str, Any]) -> FeatureFrame:
        if event_type == "market_data":
            return self.computation.process_market_data(payload)
        if event_type == "trade_events":
            return self.computation.process_trade_event(payload)
        if event_type == "order_book":
            return self.computation.process_order_book(payload)
        raise ValueError(f"Unsupported event type: {event_type}")

    def _persist_frame(self, frame: FeatureFrame) -> ValidationResult:
        records = self.online_store.write_features(
            frame.entity_id,
            frame.features,
            event_ts=frame.timestamp,
            ttl_seconds=self.online_ttl_seconds,
            source=frame.source,
        )
        offline_rows = [
            {
                "entity_id": record.entity_id,
                "feature_name": record.feature_name,
                "value": record.value,
                "event_ts": record.event_ts,
                "source": record.source,
                "version": record.version,
            }
            for record in records
        ]
        self.offline_store.write_feature_batch(offline_rows)
        self.offline_store.record_lineage(
            [
                FeatureLineageRecord(
                    feature_name=record.feature_name,
                    entity_id=record.entity_id,
                    source=record.source,
                    computation_version="v1",
                    event_ts=record.event_ts,
                    metadata={"version": record.version},
                )
                for record in records
            ]
        )

        timestamps = {record.feature_name: record.event_ts for record in records}
        validation = self.consistency.validate_serving_payload(
            frame.entity_id,
            frame.features,
            feature_timestamps=timestamps,
            required_features=list(frame.features.keys()),
        )
        if not validation.valid:
            self.health.validation_failures += 1

        for record in records:
            freshness_lag = max((_utc_now() - record.event_ts).total_seconds(), 0.0)
            self.registry.record_quality(
                FeatureQualityRecord(
                    feature_name=record.feature_name,
                    timestamp=_utc_now(),
                    null_rate=1.0 if record.value is None else 0.0,
                    freshness_lag_seconds=freshness_lag,
                    drift_score=0.0,
                )
            )
        return validation

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "health": {
                "running": self.health.running,
                "last_success_at": self.health.last_success_at.isoformat() if self.health.last_success_at else None,
                "last_error_at": self.health.last_error_at.isoformat() if self.health.last_error_at else None,
                "processed_events": self.health.processed_events,
                "failed_events": self.health.failed_events,
                "retries": self.health.retries,
                "validation_failures": self.health.validation_failures,
                "avg_cycle_latency_ms": round(self.health.avg_cycle_latency_ms, 6),
            },
            "ingestion": self.ingestion.stats,
            "online_store": self.online_store.stats,
            "registry_quality": self.registry.quality_summary(),
        }

    def close(self) -> None:
        self.health.running = False
        self.ingestion.close()
        self.online_store.close()
