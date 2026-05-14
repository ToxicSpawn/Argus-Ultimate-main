"""
Batch 3 – Signal Subscription Service
======================================
Monetisation via webhook push and internal signal fan-out.
Subscribers (HTTP webhooks or internal callbacks) receive every emitted
trading signal with a JSON payload.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Subscriber:
    subscriber_id: str
    webhook_url: Optional[str] = None
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    hmac_secret: str = ""
    enabled: bool = True
    last_delivered_ts: float = 0.0
    total_delivered: int = 0
    total_failed: int = 0


@dataclass
class SignalEnvelope:
    signal_id: str
    ts: float
    symbol: str
    side: str
    confidence: float
    strategy: str
    source: str = "argus"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalSubscriptionService:
    """
    Fan-out service that delivers trading signals to registered subscribers.

    Subscribers may be:
    * HTTP webhooks (POSTed as signed JSON)
    * In-process callbacks (e.g. for internal consumers)
    """

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._subscribers: Dict[str, Subscriber] = {}
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._total_signals_published = 0
        self._http_session: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop(), name="signal_sub_dispatch")
        logger.info("SignalSubscriptionService started")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._http_session:
            try:
                await self._http_session.close()
            except Exception:
                pass
        logger.info("SignalSubscriptionService stopped")

    # ------------------------------------------------------------------
    # Subscriber management
    # ------------------------------------------------------------------

    def register_webhook(
        self,
        subscriber_id: str,
        url: str,
        hmac_secret: str = "",
    ) -> None:
        self._subscribers[subscriber_id] = Subscriber(
            subscriber_id=subscriber_id,
            webhook_url=url,
            hmac_secret=hmac_secret,
        )
        logger.info("Registered webhook subscriber: %s → %s", subscriber_id, url)

    def register_callback(
        self,
        subscriber_id: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        self._subscribers[subscriber_id] = Subscriber(
            subscriber_id=subscriber_id,
            callback=callback,
        )
        logger.info("Registered callback subscriber: %s", subscriber_id)

    def unregister(self, subscriber_id: str) -> None:
        self._subscribers.pop(subscriber_id, None)

    # ------------------------------------------------------------------
    # Signal publishing
    # ------------------------------------------------------------------

    def publish(self, signal: Any) -> None:
        """Non-blocking enqueue of a signal for fan-out delivery."""
        try:
            envelope = self._to_envelope(signal)
            self._queue.put_nowait(envelope)
            self._total_signals_published += 1
        except asyncio.QueueFull:
            logger.warning("SignalSubscriptionService: queue full, dropping signal")
        except Exception as exc:
            logger.debug("SignalSubscriptionService.publish error: %s", exc)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                envelope: SignalEnvelope = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                await self._fan_out(envelope)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("SignalSubscriptionService dispatch error: %s", exc)

    async def _fan_out(self, envelope: SignalEnvelope) -> None:
        payload = {
            "signal_id": envelope.signal_id,
            "ts": envelope.ts,
            "symbol": envelope.symbol,
            "side": envelope.side,
            "confidence": envelope.confidence,
            "strategy": envelope.strategy,
            "source": envelope.source,
            "metadata": envelope.metadata,
        }
        for sub in list(self._subscribers.values()):
            if not sub.enabled:
                continue
            try:
                if sub.webhook_url:
                    await self._post_webhook(sub, payload)
                elif sub.callback:
                    sub.callback(dict(payload))
                sub.last_delivered_ts = time.time()
                sub.total_delivered += 1
            except Exception as exc:
                sub.total_failed += 1
                logger.debug("SignalSubscriptionService fan-out error (%s): %s", sub.subscriber_id, exc)

    async def _post_webhook(self, sub: Subscriber, payload: Dict[str, Any]) -> None:
        try:
            import aiohttp
        except ImportError:
            logger.debug("aiohttp not installed; skipping webhook delivery for %s", sub.subscriber_id)
            return
        body = json.dumps(payload, ensure_ascii=False)
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if sub.hmac_secret:
            sig = hmac.new(
                sub.hmac_secret.encode(),
                body.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Argus-Signature"] = f"sha256={sig}"
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        async with self._http_session.post(
            sub.webhook_url,
            data=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5.0),
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_envelope(self, signal: Any) -> SignalEnvelope:
        import uuid
        def _g(key: str, default: Any = "") -> Any:
            if isinstance(signal, dict):
                return signal.get(key, default)
            return getattr(signal, key, default)

        return SignalEnvelope(
            signal_id=str(_g("signal_id") or uuid.uuid4().hex[:12]),
            ts=float(_g("timestamp") or time.time()),
            symbol=str(_g("symbol") or ""),
            side=str(_g("side") or _g("action") or "").upper(),
            confidence=float(_g("confidence") or 0.0),
            strategy=str(_g("strategy") or _g("source_strategy") or "unknown"),
            source="argus",
            metadata={},
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "total_published": self._total_signals_published,
            "queue_depth": self._queue.qsize(),
            "subscribers": {
                sid: {
                    "delivered": s.total_delivered,
                    "failed": s.total_failed,
                    "last_ts": s.last_delivered_ts,
                }
                for sid, s in self._subscribers.items()
            },
        }
