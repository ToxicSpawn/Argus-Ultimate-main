"""Kafka/Redpanda ingestion layer for the streaming feature store."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaConsumer as _KafkaConsumer, KafkaProducer as _KafkaProducer  # pyright: ignore[reportMissingImports]

    _kafka_available = True
except Exception:
    _KafkaConsumer = None  # type: ignore[assignment]
    _KafkaProducer = None  # type: ignore[assignment]
    _kafka_available = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class StreamEvent:
    topic: str
    payload: Dict[str, Any]
    key: Optional[str] = None
    partition: Optional[int] = None
    offset: Optional[int] = None
    timestamp: datetime = field(default_factory=_utc_now)
    source: str = "kafka"


@dataclass(slots=True)
class DeadLetterRecord:
    topic: str
    payload: Dict[str, Any]
    error: str
    key: Optional[str] = None
    partition: Optional[int] = None
    offset: Optional[int] = None
    failed_at: datetime = field(default_factory=_utc_now)


class _FallbackTopicBuffer:
    """In-memory queue used when Kafka is unavailable."""

    def __init__(self) -> None:
        self._queues: Dict[str, "queue.Queue[StreamEvent]"] = {}
        self._lock = threading.RLock()

    def put(self, event: StreamEvent) -> None:
        with self._lock:
            if event.topic not in self._queues:
                self._queues[event.topic] = queue.Queue()
            self._queues[event.topic].put(event)

    def get_many(self, topic: str, max_items: int) -> List[StreamEvent]:
        items: List[StreamEvent] = []
        with self._lock:
            topic_queue = self._queues.get(topic)
            if topic_queue is None:
                return items
            for _ in range(max_items):
                try:
                    items.append(topic_queue.get_nowait())
                except queue.Empty:
                    break
        return items


class StreamingIngestionService:
    """Multi-topic streaming ingestion with DLQ and graceful fallback."""

    def __init__(
        self,
        *,
        brokers: Optional[Iterable[str]] = None,
        group_id: str = "argus-feature-store-streaming",
        market_data_topic: str = "argus.market_data",
        trade_events_topic: str = "argus.trade_events",
        order_book_topic: str = "argus.order_book",
        dlq_topic: str = "argus.feature_store.dlq",
        consumer_factory: Optional[Callable[..., Any]] = None,
        producer_factory: Optional[Callable[..., Any]] = None,
        max_poll_records: int = 256,
    ) -> None:
        self.brokers = list(brokers or ["localhost:9092"])
        self.group_id = group_id
        self.market_data_topic = market_data_topic
        self.trade_events_topic = trade_events_topic
        self.order_book_topic = order_book_topic
        self.dlq_topic = dlq_topic
        self.max_poll_records = max(1, int(max_poll_records))

        self._consumer_factory = consumer_factory or _KafkaConsumer
        self._producer_factory = producer_factory or _KafkaProducer
        self._consumer: Optional[Any] = None
        self._producer: Optional[Any] = None
        self._fallback = _FallbackTopicBuffer()
        self._dlq_records: List[DeadLetterRecord] = []
        self._connected = False
        self._stats: Dict[str, int] = {
            "market_events": 0,
            "trade_events": 0,
            "order_book_events": 0,
            "dlq_events": 0,
            "connection_failures": 0,
        }

        self._connect()

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    @property
    def connected(self) -> bool:
        return self._connected

    def _connect(self) -> None:
        if not _kafka_available or self._consumer_factory is None:
            logger.warning("Kafka unavailable - streaming ingestion using in-memory fallback")
            self._connected = False
            return
        try:
            self._consumer = self._consumer_factory(
                self.market_data_topic,
                self.trade_events_topic,
                self.order_book_topic,
                bootstrap_servers=self.brokers,
                group_id=self.group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=250,
                max_poll_records=self.max_poll_records,
                value_deserializer=lambda data: json.loads(data.decode("utf-8")),
                key_deserializer=lambda data: data.decode("utf-8") if data else None,
            )
            if self._producer_factory is not None:
                self._producer = self._producer_factory(
                    bootstrap_servers=self.brokers,
                    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                    key_serializer=lambda value: value.encode("utf-8") if value else None,
                )
            self._connected = True
            logger.info("Streaming ingestion connected to brokers=%s", ",".join(self.brokers))
        except Exception as exc:
            self._stats["connection_failures"] += 1
            self._connected = False
            self._consumer = None
            self._producer = None
            logger.warning("Streaming ingestion connection failed (%s) - using fallback", exc)

    def ingest_fallback(self, topic: str, payload: Dict[str, Any], *, key: Optional[str] = None) -> None:
        """Inject an event into the fallback queue for degraded operation/testing."""
        event = StreamEvent(topic=topic, payload=dict(payload), key=key, source="fallback")
        self._fallback.put(event)

    def consume_market_data(self, max_records: int = 128) -> List[StreamEvent]:
        return self._consume_topic(self.market_data_topic, max_records=max_records, stat_key="market_events")

    def consume_trade_events(self, max_records: int = 128) -> List[StreamEvent]:
        return self._consume_topic(self.trade_events_topic, max_records=max_records, stat_key="trade_events")

    def consume_order_book_updates(self, max_records: int = 128) -> List[StreamEvent]:
        return self._consume_topic(self.order_book_topic, max_records=max_records, stat_key="order_book_events")

    def _consume_topic(self, topic: str, *, max_records: int, stat_key: str) -> List[StreamEvent]:
        limit = max(1, int(max_records))
        if not self._connected or self._consumer is None:
            events = self._fallback.get_many(topic, limit)
            self._stats[stat_key] += len(events)
            return events

        deadline = time.monotonic() + 0.2
        events: List[StreamEvent] = []
        while len(events) < limit and time.monotonic() <= deadline:
            try:
                records = self._consumer.poll(timeout_ms=50, max_records=limit - len(events))
            except Exception as exc:
                logger.warning("Kafka poll failed (%s) - switching to fallback", exc)
                self._stats["connection_failures"] += 1
                self._connected = False
                return self._fallback.get_many(topic, limit)

            for _partition, messages in records.items():
                for message in messages:
                    if message.topic != topic:
                        continue
                    payload = message.value if isinstance(message.value, dict) else {"value": message.value}
                    events.append(
                        StreamEvent(
                            topic=message.topic,
                            payload=payload,
                            key=message.key,
                            partition=getattr(message, "partition", None),
                            offset=getattr(message, "offset", None),
                            timestamp=_utc_now(),
                        )
                    )
                    if len(events) >= limit:
                        break
                if len(events) >= limit:
                    break
        self._stats[stat_key] += len(events)
        return events

    def publish_to_dlq(self, record: DeadLetterRecord) -> None:
        self._dlq_records.append(record)
        self._stats["dlq_events"] += 1
        if not self._connected or self._producer is None:
            logger.error("DLQ fallback topic=%s error=%s", record.topic, record.error)
            return
        try:
            self._producer.send(
                self.dlq_topic,
                key=record.key,
                value={
                    "topic": record.topic,
                    "payload": record.payload,
                    "error": record.error,
                    "partition": record.partition,
                    "offset": record.offset,
                    "failed_at": record.failed_at.isoformat(),
                },
            )
            self._producer.flush(timeout=1.0)
        except Exception as exc:
            logger.exception("Failed to publish DLQ record: %s", exc)

    def handle_processing_error(self, event: StreamEvent, exc: Exception) -> None:
        self.publish_to_dlq(
            DeadLetterRecord(
                topic=event.topic,
                payload=event.payload,
                error=str(exc),
                key=event.key,
                partition=event.partition,
                offset=event.offset,
            )
        )

    def get_dead_letter_records(self) -> List[DeadLetterRecord]:
        return list(self._dlq_records)

    def close(self) -> None:
        for client in (self._consumer, self._producer):
            if client is None:
                continue
            try:
                client.close()
            except Exception:
                logger.debug("Failed to close streaming ingestion client", exc_info=True)
        self._connected = False
