"""
Storage Factory — returns the right database/cache backend based on environment.

Priority:
  1. Explicit URL in environment variable
  2. ARGUS_CONFIG_PROFILE == 'r740' implies R740 backend
  3. Default: SQLite + in-memory dict + asyncio.Queue

Usage
-----
    from core.storage_factory import StorageFactory

    # Get a database connection (SQLite or TimescaleDB)
    conn = StorageFactory.get_db_connection("trades")   # returns sqlite3.Connection OR psycopg2 conn

    # Get a Redis client (or None if not configured)
    redis = StorageFactory.get_redis()

    # Check which backend is active
    if StorageFactory.backend == "r740":
        ...

Environment Variables
---------------------
    ARGUS_DATABASE_URL   — postgresql://user:pass@host:5432/dbname  (TimescaleDB)
    ARGUS_REDIS_URL      — redis://:pass@host:6379/0
    ARGUS_KAFKA_BROKERS  — host:9093,host2:9093
    ARGUS_CONFIG_PROFILE — if 'r740', defaults above are read from env

Design: all methods degrade gracefully. If TimescaleDB is not reachable, SQLite
fallback is used. No code outside this module needs to know which backend is live.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    """Return 'r740' if infrastructure env vars are present, else 'local'."""
    if os.environ.get("ARGUS_DATABASE_URL", "").startswith("postgresql"):
        return "r740"
    if os.environ.get("ARGUS_CONFIG_PROFILE", "").lower() == "r740":
        return "r740"
    return "local"


# Module-level singleton state
_backend: str = _detect_backend()
_redis_client: Optional[Any] = None
_pg_pool: Optional[Any] = None          # psycopg2 / asyncpg pool (lazy)
_kafka_producer: Optional[Any] = None   # confluent_kafka producer (lazy)


class StorageFactory:
    """
    Unified interface for storage backends.

    All methods are classmethods so no instantiation is needed.
    Backends are created lazily on first access.
    """

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @classmethod
    @property
    def backend(cls) -> str:
        return _backend

    @classmethod
    def is_r740(cls) -> bool:
        return _backend == "r740"

    # ------------------------------------------------------------------
    # Relational DB — SQLite (local) or TimescaleDB (r740)
    # ------------------------------------------------------------------

    @classmethod
    def get_db_connection(cls, db_name: str = "argus") -> Any:
        """
        Return a database connection.

        - R740 mode: psycopg2 connection to TimescaleDB (all tables in one DB)
        - Local mode: sqlite3.Connection to data/{db_name}.db
        """
        if cls.is_r740():
            return cls._get_pg_connection()
        return cls._get_sqlite_connection(db_name)

    @classmethod
    def _get_pg_connection(cls) -> Any:
        """Return a fresh psycopg2 connection to TimescaleDB."""
        url = os.environ.get("ARGUS_DATABASE_URL", "")
        if not url:
            logger.warning("StorageFactory: ARGUS_DATABASE_URL not set, falling back to SQLite")
            return cls._get_sqlite_connection("argus")
        try:
            import psycopg2                             # type: ignore[import]
            import psycopg2.extras                      # type: ignore[import]
            conn = psycopg2.connect(url)
            conn.autocommit = False
            return conn
        except ImportError:
            logger.error("StorageFactory: psycopg2 not installed. Run: pip install psycopg2-binary")
            return cls._get_sqlite_connection("argus")
        except Exception as exc:
            logger.error("StorageFactory: TimescaleDB connection failed (%s), falling back to SQLite", exc)
            return cls._get_sqlite_connection("argus")

    @classmethod
    def _get_sqlite_connection(cls, db_name: str) -> sqlite3.Connection:
        """Return a WAL-mode SQLite connection."""
        db_dir = Path("data")
        db_dir.mkdir(parents=True, exist_ok=True)
        path = str(db_dir / f"{db_name}.db")
        conn = sqlite3.connect(path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    # ------------------------------------------------------------------
    # Redis — real-time state store
    # ------------------------------------------------------------------

    @classmethod
    def get_redis(cls) -> Optional[Any]:
        """
        Return a redis.Redis client, or None if not configured.

        Callers must handle the None case (fall back to in-memory dict).
        """
        global _redis_client
        if not cls.is_r740():
            return None
        if _redis_client is not None:
            return _redis_client

        url = os.environ.get("ARGUS_REDIS_URL", "")
        if not url:
            logger.debug("StorageFactory: ARGUS_REDIS_URL not set, Redis disabled")
            return None

        try:
            import redis                                # type: ignore[import]
            client = redis.from_url(
                url,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                decode_responses=True,
            )
            client.ping()
            _redis_client = client
            logger.info("StorageFactory: Redis connected (%s)", url.split("@")[-1])
            return _redis_client
        except ImportError:
            logger.error("StorageFactory: redis package not installed. Run: pip install redis")
            return None
        except Exception as exc:
            logger.error("StorageFactory: Redis connection failed (%s)", exc)
            return None

    # ------------------------------------------------------------------
    # Kafka — event streaming
    # ------------------------------------------------------------------

    @classmethod
    def get_kafka_producer(cls) -> Optional[Any]:
        """
        Return a confluent_kafka.Producer, or None if not configured.

        Callers publish events; if None is returned they use asyncio.Queue.
        """
        global _kafka_producer
        if not cls.is_r740():
            return None
        if _kafka_producer is not None:
            return _kafka_producer

        brokers = os.environ.get("ARGUS_KAFKA_BROKERS", "")
        if not brokers:
            logger.debug("StorageFactory: ARGUS_KAFKA_BROKERS not set, Kafka disabled")
            return None

        try:
            from confluent_kafka import Producer        # type: ignore[import]
            producer = Producer({
                "bootstrap.servers": brokers,
                "acks": "1",
                "retries": 3,
                "batch.size": 16384,
                "linger.ms": 5,
                "compression.type": "lz4",
            })
            _kafka_producer = producer
            logger.info("StorageFactory: Kafka producer connected (%s)", brokers)
            return _kafka_producer
        except ImportError:
            logger.error("StorageFactory: confluent-kafka not installed. Run: pip install confluent-kafka")
            return None
        except Exception as exc:
            logger.error("StorageFactory: Kafka producer init failed (%s)", exc)
            return None

    @classmethod
    def publish_event(cls, topic_env_var: str, key: str, value: Dict) -> bool:
        """
        Publish a JSON event to a Kafka topic.

        topic_env_var: e.g. 'ARGUS_KAFKA_TOPIC_FILLS' → resolves to topic name
        Returns True if published to Kafka, False if fell back to no-op.
        """
        producer = cls.get_kafka_producer()
        if producer is None:
            return False

        import json
        topic = os.environ.get(topic_env_var, topic_env_var.lower().replace("argus_kafka_topic_", "argus."))
        try:
            producer.produce(
                topic,
                key=key.encode(),
                value=json.dumps(value).encode(),
            )
            producer.poll(0)
            return True
        except Exception as exc:
            logger.warning("StorageFactory: Kafka publish failed (%s)", exc)
            return False

    # ------------------------------------------------------------------
    # Loki — structured log shipping
    # ------------------------------------------------------------------

    @classmethod
    def get_loki_handler(cls) -> Optional[Any]:
        """
        Return a python-logging handler that ships to Loki, or None.

        The returned handler can be added to any logger:
            logger.addHandler(StorageFactory.get_loki_handler())
        """
        loki_url = os.environ.get("ARGUS_LOKI_URL", "")
        if not loki_url or not cls.is_r740():
            return None
        try:
            import logging_loki                        # type: ignore[import]
            handler = logging_loki.LokiHandler(
                url=f"{loki_url}/loki/api/v1/push",
                tags={"application": "argus", "env": "live"},
                version="1",
            )
            return handler
        except ImportError:
            logger.debug("StorageFactory: python-logging-loki not installed, Loki disabled")
            return None
        except Exception as exc:
            logger.warning("StorageFactory: Loki handler init failed (%s)", exc)
            return None

    # ------------------------------------------------------------------
    # Health check — verify all backends are reachable
    # ------------------------------------------------------------------

    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """Return connectivity status for all configured backends."""
        result: Dict[str, Any] = {
            "backend": _backend,
            "timescaledb": "not_configured",
            "redis": "not_configured",
            "kafka": "not_configured",
        }

        if os.environ.get("ARGUS_DATABASE_URL", ""):
            try:
                conn = cls._get_pg_connection()
                cur = conn.cursor()
                cur.execute("SELECT version()")
                ver = cur.fetchone()[0]
                conn.close()
                result["timescaledb"] = f"ok ({ver[:30]})"
            except Exception as exc:
                result["timescaledb"] = f"error: {exc}"

        redis = cls.get_redis()
        if redis is not None:
            try:
                redis.ping()
                result["redis"] = "ok"
            except Exception as exc:
                result["redis"] = f"error: {exc}"

        if os.environ.get("ARGUS_KAFKA_BROKERS", ""):
            producer = cls.get_kafka_producer()
            result["kafka"] = "ok" if producer is not None else "error: producer init failed"

        return result

    # ------------------------------------------------------------------
    # Reset (for testing)
    # ------------------------------------------------------------------

    @classmethod
    def _reset(cls) -> None:
        """Reset all cached connections. Used in tests."""
        global _redis_client, _pg_pool, _kafka_producer, _backend
        _redis_client = None
        _pg_pool = None
        _kafka_producer = None
        _backend = _detect_backend()
