"""Redis-backed AppContext persistence — Push 100 StatePersist.

Persists and hydrates the three critical runtime state buckets:
  • alert_config     — per-symbol alert thresholds and notification targets
  • regime_history   — rolling window of recent regime transitions
  • bandit_arms      — bandit arm Q-values, pull counts, and epsilon

All keys are namespaced under ``argus:state:<bucket>`` and serialised
as MessagePack (fast, compact, no schema drift).  Falls back to JSON
if msgpack is unavailable.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

try:
    import msgpack  # type: ignore
    _PACK   = msgpack.packb
    _UNPACK = lambda b: msgpack.unpackb(b, raw=False)  # noqa: E731
    _SER    = "msgpack"
except ImportError:
    _PACK   = lambda o: json.dumps(o).encode()   # noqa: E731
    _UNPACK = lambda b: json.loads(b.decode())   # noqa: E731
    _SER    = "json"
    log.warning("msgpack not installed — falling back to JSON serialisation")

try:
    import redis  # type: ignore
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    log.warning("redis-py not installed — StatePersist disabled")


_NS = "argus:state"
_TTL_SECONDS = 86_400 * 7   # 7-day TTL; refreshed on every write


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class AlertConfig:
    symbol:            str
    price_threshold:   float          = 0.0
    pct_move:          float          = 0.05
    cooldown_secs:     int            = 300
    notification_tags: List[str]      = field(default_factory=list)
    enabled:           bool           = True


@dataclass
class RegimeEntry:
    ts:          float   # Unix epoch seconds
    from_regime: str
    to_regime:   str
    confidence:  float
    symbol:      str     = ""


@dataclass
class BanditState:
    arm:        str
    q_value:    float
    pull_count: int
    epsilon:    float    = 0.1


# ─── Client ──────────────────────────────────────────────────────────────────

class StatePersist:
    """Thread-safe Redis state store for AppContext runtime buckets."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        pool_size: int   = 8,
        socket_timeout: float = 1.0,
        decode_responses: bool = False,
    ) -> None:
        if not _REDIS_AVAILABLE:
            self._client: Optional[Any] = None
            log.warning("StatePersist created but redis-py is missing — all ops are no-ops")
            return

        pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=pool_size,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=decode_responses,
        )
        self._client = redis.Redis(connection_pool=pool)
        log.info("StatePersist connected to %s (serialiser=%s)", redis_url, _SER)

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _key(self, bucket: str, sub: str = "") -> str:
        return f"{_NS}:{bucket}" + (f":{sub}" if sub else "")

    def _set(self, key: str, obj: Any) -> None:
        if self._client is None:
            return
        try:
            self._client.setex(key, _TTL_SECONDS, _PACK(obj))
        except Exception as exc:  # noqa: BLE001
            log.error("StatePersist._set(%s) failed: %s", key, exc)

    def _get(self, key: str) -> Any:
        if self._client is None:
            return None
        try:
            raw = self._client.get(key)
            return _UNPACK(raw) if raw is not None else None
        except Exception as exc:  # noqa: BLE001
            log.error("StatePersist._get(%s) failed: %s", key, exc)
            return None

    # ─── alert_config ────────────────────────────────────────────────────────

    def save_alert_config(self, configs: List[AlertConfig]) -> None:
        """Persist the full alert_config list."""
        self._set(self._key("alert_config"), [asdict(c) for c in configs])

    def load_alert_config(self) -> List[AlertConfig]:
        """Hydrate alert_config from Redis; returns [] on miss."""
        raw = self._get(self._key("alert_config"))
        if not raw:
            return []
        try:
            return [AlertConfig(**r) for r in raw]
        except Exception as exc:  # noqa: BLE001
            log.warning("alert_config deserialise error: %s", exc)
            return []

    # ─── regime_history ──────────────────────────────────────────────────────

    def append_regime_entry(self, entry: RegimeEntry, maxlen: int = 500) -> None:
        """Push a regime transition entry onto the history list (capped at maxlen)."""
        if self._client is None:
            return
        key = self._key("regime_history")
        pipe = self._client.pipeline()
        try:
            pipe.rpush(key, _PACK(asdict(entry)))
            pipe.ltrim(key, -maxlen, -1)
            pipe.expire(key, _TTL_SECONDS)
            pipe.execute()
        except Exception as exc:  # noqa: BLE001
            log.error("append_regime_entry failed: %s", exc)

    def load_regime_history(self, limit: int = 500) -> List[RegimeEntry]:
        """Load up to *limit* most-recent regime entries."""
        if self._client is None:
            return []
        key = self._key("regime_history")
        try:
            raw_list = self._client.lrange(key, -limit, -1)
            return [RegimeEntry(**_UNPACK(r)) for r in raw_list]
        except Exception as exc:  # noqa: BLE001
            log.warning("load_regime_history failed: %s", exc)
            return []

    # ─── bandit_arms ─────────────────────────────────────────────────────────

    def save_bandit_arms(self, arms: List[BanditState]) -> None:
        """Persist all bandit arm states."""
        self._set(self._key("bandit_arms"), [asdict(a) for a in arms])

    def load_bandit_arms(self) -> List[BanditState]:
        """Hydrate bandit arm states; returns [] on miss."""
        raw = self._get(self._key("bandit_arms"))
        if not raw:
            return []
        try:
            return [BanditState(**r) for r in raw]
        except Exception as exc:  # noqa: BLE001
            log.warning("bandit_arms deserialise error: %s", exc)
            return []

    # ─── Generic AppContext snapshot ─────────────────────────────────────────

    def snapshot(self, ctx_dict: Dict[str, Any]) -> None:
        """Persist an arbitrary AppContext snapshot dict."""
        self._set(self._key("snapshot"), ctx_dict)

    def restore(self) -> Optional[Dict[str, Any]]:
        """Restore the latest AppContext snapshot dict."""
        return self._get(self._key("snapshot"))  # type: ignore[return-value]

    # ─── Health ──────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if Redis responds to PING."""
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False
