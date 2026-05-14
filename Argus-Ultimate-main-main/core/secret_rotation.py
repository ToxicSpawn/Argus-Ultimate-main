"""Secret rotation watcher — Push 100 StatePersist.

Watches for in-place secret rotation signals and hot-swaps
exchange API credentials without restarting the bot process.

Rotation triggers supported:
  1. File-based:   mtime change on a secrets file path
  2. Env-var:      ARGUS_SECRET_VERSION env var change
  3. Redis pub/sub: ``argus:secret:rotate`` channel message

When a rotation is detected the watcher calls all registered
``on_rotate`` callbacks with the new secret bundle so connectors
can re-authenticate in-place.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

RotateCallback = Callable[[Dict[str, str]], None]


@dataclass
class SecretBundle:
    """Immutable snapshot of rotated secrets."""
    version:    str
    secrets:    Dict[str, str]  = field(default_factory=dict)
    rotated_at: float           = field(default_factory=time.time)


class SecretRotationWatcher:
    """Background thread that polls for secret rotation and fires callbacks.

    Usage::

        watcher = SecretRotationWatcher(secrets_path="/run/secrets/argus")
        watcher.register(my_connector.on_secret_rotate)
        watcher.start()

    Callbacks receive a ``Dict[str, str]`` mapping secret names to new values.
    """

    def __init__(
        self,
        secrets_path:    Optional[str]   = None,
        poll_interval:   float           = 30.0,
        redis_url:       Optional[str]   = None,
        redis_channel:   str             = "argus:secret:rotate",
    ) -> None:
        self._secrets_path  = Path(secrets_path) if secrets_path else None
        self._poll_interval = poll_interval
        self._redis_url     = redis_url
        self._redis_channel = redis_channel
        self._callbacks:    List[RotateCallback] = []
        self._stop_event    = threading.Event()
        self._last_mtime:   float = 0.0
        self._last_version: str   = os.environ.get("ARGUS_SECRET_VERSION", "")
        self._lock          = threading.Lock()
        self._thread:       Optional[threading.Thread] = None

    # ─── Registration ────────────────────────────────────────────────────────

    def register(self, cb: RotateCallback) -> None:
        """Register a callback to be called on secret rotation."""
        with self._lock:
            self._callbacks.append(cb)
        log.debug("SecretRotationWatcher: registered callback %s", cb)

    def unregister(self, cb: RotateCallback) -> None:
        with self._lock:
            self._callbacks = [c for c in self._callbacks if c is not cb]

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background polling thread and optional Redis subscriber."""
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="secret-rotation-watcher",
            daemon=True,
        )
        self._thread.start()
        log.info("SecretRotationWatcher started (interval=%.0fs)", self._poll_interval)

        if self._redis_url:
            sub_thread = threading.Thread(
                target=self._redis_sub_loop,
                name="secret-rotation-redis-sub",
                daemon=True,
            )
            sub_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ─── Polling loop ────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._check_file()
            self._check_env()
            self._stop_event.wait(self._poll_interval)

    def _check_file(self) -> None:
        if self._secrets_path is None:
            return
        try:
            mtime = self._secrets_path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime > self._last_mtime:
            if self._last_mtime > 0:  # skip initial read
                log.info("Secret file rotation detected: %s", self._secrets_path)
                bundle = self._load_file_secrets()
                self._fire(bundle)
            self._last_mtime = mtime

    def _check_env(self) -> None:
        ver = os.environ.get("ARGUS_SECRET_VERSION", "")
        if ver and ver != self._last_version:
            log.info("ARGUS_SECRET_VERSION changed: %s → %s", self._last_version, ver)
            self._last_version = ver
            bundle = self._build_env_bundle(ver)
            self._fire(bundle)

    # ─── Redis subscriber ────────────────────────────────────────────────────

    def _redis_sub_loop(self) -> None:
        try:
            import redis  # type: ignore
        except ImportError:
            log.warning("redis-py missing — Redis rotation subscription disabled")
            return

        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                r   = redis.from_url(self._redis_url)  # type: ignore[arg-type]
                sub = r.pubsub(ignore_subscribe_messages=True)
                sub.subscribe(self._redis_channel)
                log.info("Subscribed to Redis channel %s", self._redis_channel)
                backoff = 1.0
                for msg in sub.listen():
                    if self._stop_event.is_set():
                        break
                    if msg and msg.get("type") == "message":
                        data = msg.get("data", b"")
                        if isinstance(data, bytes):
                            data = data.decode(errors="replace")
                        log.info("Redis rotation signal received: %s", data)
                        bundle = self._build_env_bundle(data)
                        self._fire(bundle)
            except Exception as exc:  # noqa: BLE001
                log.warning("Redis sub error: %s — retry in %.0fs", exc, backoff)
                self._stop_event.wait(backoff)
                backoff = min(backoff * 2, 60)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _load_file_secrets(self) -> SecretBundle:
        """Read key=value pairs from secrets file."""
        secrets: Dict[str, str] = {}
        try:
            for line in self._secrets_path.read_text().splitlines():  # type: ignore[union-attr]
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    secrets[k.strip()] = v.strip()
        except Exception as exc:  # noqa: BLE001
            log.error("Error reading secrets file: %s", exc)
        return SecretBundle(version=str(self._secrets_path.stat().st_mtime), secrets=secrets)  # type: ignore[union-attr]

    def _build_env_bundle(self, version: str) -> SecretBundle:
        """Snapshot relevant ARGUS_* env vars as the new bundle."""
        secrets = {
            k: v for k, v in os.environ.items()
            if k.startswith("ARGUS_") and "KEY" in k or "SECRET" in k or "TOKEN" in k
        }
        return SecretBundle(version=version, secrets=secrets)

    def _fire(self, bundle: SecretBundle) -> None:
        with self._lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(bundle.secrets)
            except Exception as exc:  # noqa: BLE001
                log.error("Rotation callback %s raised: %s", cb, exc)
