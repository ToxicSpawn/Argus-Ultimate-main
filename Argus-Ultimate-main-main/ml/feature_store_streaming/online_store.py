"""Redis-backed low-latency online feature store with versioning and PIT retrieval."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import redis as _redis  # pyright: ignore[reportMissingImports]

    _redis_available = True
except Exception:
    _redis = None  # type: ignore[assignment]
    _redis_available = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class FeatureRecord:
    entity_id: str
    feature_name: str
    value: Any
    event_ts: datetime
    created_ts: datetime = field(default_factory=_utc_now)
    version: int = 1
    ttl_seconds: int = 300
    source: str = "streaming"

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["event_ts"] = self.event_ts.isoformat()
        payload["created_ts"] = self.created_ts.isoformat()
        return payload


class _InMemoryOnlineStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, List[FeatureRecord]]] = defaultdict(lambda: defaultdict(list))
        self._lock = threading.RLock()

    def write(self, record: FeatureRecord) -> None:
        with self._lock:
            bucket = self._data[record.entity_id][record.feature_name]
            bucket.append(record)
            bucket.sort(key=lambda item: (item.event_ts, item.version))

    def get_latest(self, entity_id: str, feature_name: str) -> Optional[FeatureRecord]:
        with self._lock:
            versions = self._data.get(entity_id, {}).get(feature_name, [])
            if not versions:
                return None
            latest = versions[-1]
            if latest.event_ts + timedelta(seconds=latest.ttl_seconds) < _utc_now():
                return None
            return latest

    def get_as_of(self, entity_id: str, feature_name: str, as_of: datetime) -> Optional[FeatureRecord]:
        with self._lock:
            versions = self._data.get(entity_id, {}).get(feature_name, [])
            candidates = [record for record in versions if record.event_ts <= as_of]
            return candidates[-1] if candidates else None

    def batch_get(self, entity_ids: Sequence[str], feature_names: Sequence[str], *, as_of: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for entity_id in entity_ids:
            result[entity_id] = {}
            for feature_name in feature_names:
                record = self.get_as_of(entity_id, feature_name, as_of) if as_of is not None else self.get_latest(entity_id, feature_name)
                if record is not None:
                    result[entity_id][feature_name] = record.value
        return result


class RedisOnlineFeatureStore:
    """Online store optimized for fast reads with automatic degraded mode."""

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "argus:streaming_features",
        default_ttl_seconds: int = 300,
        max_versions_per_feature: int = 32,
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl_seconds = max(1, int(default_ttl_seconds))
        self.max_versions_per_feature = max(2, int(max_versions_per_feature))
        self._fallback = _InMemoryOnlineStore()
        self._redis: Optional[Any] = None
        self._using_fallback = True
        self._lock = threading.RLock()
        self._version_counters: Dict[Tuple[str, str], int] = defaultdict(int)
        self._latency_samples_ms: List[float] = []
        self._stats: Dict[str, int] = {
            "writes": 0,
            "reads": 0,
            "fallback_reads": 0,
            "fallback_writes": 0,
            "redis_failures": 0,
        }
        self._connect()

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def stats(self) -> Dict[str, Any]:
        avg_latency_ms = sum(self._latency_samples_ms) / len(self._latency_samples_ms) if self._latency_samples_ms else 0.0
        result: Dict[str, Any] = dict(self._stats)
        result["avg_read_latency_ms"] = round(avg_latency_ms, 6)
        return result

    def _connect(self) -> None:
        if not _redis_available:
            logger.warning("Redis not available - online store using in-memory fallback")
            self._using_fallback = True
            return
        try:
            redis_module = _redis
            if redis_module is None:
                raise RuntimeError("redis client unavailable")
            redis_client = redis_module.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=0.01,
                socket_connect_timeout=0.01,
                retry_on_timeout=True,
            )
            redis_client.ping()
            self._redis = redis_client
            self._using_fallback = False
            logger.info("Streaming online store connected to Redis")
        except Exception as exc:
            self._stats["redis_failures"] += 1
            self._redis = None
            self._using_fallback = True
            logger.warning("Redis unavailable (%s) - using fallback", exc)

    def write_features(
        self,
        entity_id: str,
        features: Mapping[str, Any],
        *,
        event_ts: Optional[datetime] = None,
        ttl_seconds: Optional[int] = None,
        source: str = "streaming",
    ) -> List[FeatureRecord]:
        now = event_ts or _utc_now()
        ttl = int(ttl_seconds or self.default_ttl_seconds)
        records: List[FeatureRecord] = []
        with self._lock:
            for feature_name, value in features.items():
                key = (entity_id, feature_name)
                self._version_counters[key] += 1
                record = FeatureRecord(
                    entity_id=entity_id,
                    feature_name=feature_name,
                    value=value,
                    event_ts=now,
                    version=self._version_counters[key],
                    ttl_seconds=ttl,
                    source=source,
                )
                self._fallback.write(record)
                records.append(record)
                self._stats["writes"] += 1
                self._stats["fallback_writes"] += 1
                self._write_redis(record)
        return records

    def _write_redis(self, record: FeatureRecord) -> None:
        if self._using_fallback or self._redis is None:
            return
        try:
            latest_key = self._latest_key(record.entity_id)
            history_key = self._history_key(record.entity_id, record.feature_name)
            pipe = self._redis.pipeline()
            payload = json.dumps(record.to_payload())
            pipe.hset(latest_key, record.feature_name, payload)
            pipe.zadd(history_key, {payload: record.event_ts.timestamp()})
            pipe.zremrangebyrank(history_key, 0, -(self.max_versions_per_feature + 1))
            pipe.expire(latest_key, record.ttl_seconds)
            pipe.expire(history_key, record.ttl_seconds)
            pipe.execute()
        except Exception as exc:
            self._stats["redis_failures"] += 1
            self._using_fallback = True
            logger.warning("Redis write failed for %s/%s: %s", record.entity_id, record.feature_name, exc)

    def get_feature(self, entity_id: str, feature_name: str, *, as_of: Optional[datetime] = None) -> Optional[FeatureRecord]:
        start = time.perf_counter()
        try:
            record = self._read_redis(entity_id, feature_name, as_of=as_of)
            if record is None:
                self._stats["fallback_reads"] += 1
                record = self._fallback.get_as_of(entity_id, feature_name, as_of) if as_of is not None else self._fallback.get_latest(entity_id, feature_name)
            return record
        finally:
            self._stats["reads"] += 1
            self._latency_samples_ms.append((time.perf_counter() - start) * 1000.0)
            self._latency_samples_ms[:] = self._latency_samples_ms[-1000:]

    def _read_redis(self, entity_id: str, feature_name: str, *, as_of: Optional[datetime]) -> Optional[FeatureRecord]:
        if self._using_fallback or self._redis is None:
            return None
        try:
            if as_of is None:
                raw = self._redis.hget(self._latest_key(entity_id), feature_name)
                return self._decode_record(raw)
            history_key = self._history_key(entity_id, feature_name)
            records = self._redis.zrevrangebyscore(history_key, as_of.timestamp(), 0, start=0, num=1)
            return self._decode_record(records[0]) if records else None
        except Exception as exc:
            self._stats["redis_failures"] += 1
            self._using_fallback = True
            logger.warning("Redis read failed for %s/%s: %s", entity_id, feature_name, exc)
            return None

    def batch_get_features(
        self,
        entity_ids: Sequence[str],
        feature_names: Sequence[str],
        *,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for entity_id in entity_ids:
            result[entity_id] = {}
            for feature_name in feature_names:
                record = self.get_feature(entity_id, feature_name, as_of=as_of)
                if record is not None:
                    result[entity_id][feature_name] = record.value
        return result

    def _decode_record(self, raw: Optional[str]) -> Optional[FeatureRecord]:
        if not raw:
            return None
        payload = json.loads(raw)
        return FeatureRecord(
            entity_id=payload["entity_id"],
            feature_name=payload["feature_name"],
            value=payload["value"],
            event_ts=datetime.fromisoformat(payload["event_ts"]),
            created_ts=datetime.fromisoformat(payload["created_ts"]),
            version=int(payload["version"]),
            ttl_seconds=int(payload["ttl_seconds"]),
            source=str(payload.get("source", "streaming")),
        )

    def _latest_key(self, entity_id: str) -> str:
        return f"{self.key_prefix}:{entity_id}:latest"

    def _history_key(self, entity_id: str, feature_name: str) -> str:
        return f"{self.key_prefix}:{entity_id}:{feature_name}:history"

    def close(self) -> None:
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                logger.debug("Failed to close Redis client", exc_info=True)
