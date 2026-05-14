"""AlertManager — multi-channel alert dispatcher — Push 60.

Usage::

    manager = AlertManager()
    manager.register_channel(TelegramChannel.from_env())
    manager.register_channel(DiscordChannel.from_env())

    # Synchronous fire-and-forget (queued)
    manager.enqueue(AlertEvent.warning("Drawdown limit hit", symbol="BTCUSDT"))

    # Async direct send
    await manager.send(AlertEvent.critical("Bot halted"))

Prometheus::

    argus_alerts_sent_total{level, channel}
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from core.alerts.alert_models import AlertEvent, AlertLevel
from core.alerts.base_channel import AbstractAlertChannel

try:
    from prometheus_client import Counter
    _PROM = True
except ImportError:
    _PROM = False

logger = logging.getLogger(__name__)

if _PROM:
    _CTR_ALERTS = Counter(
        "argus_alerts_sent_total",
        "Total alerts dispatched",
        ["level", "channel"],
    )
else:
    _CTR_ALERTS = None


class AlertManager:
    """Fans out AlertEvents to all registered channels.

    Parameters
    ----------
    min_level : AlertLevel
        Global minimum level filter (overrides per-channel filters).
    """

    def __init__(self, min_level: AlertLevel = AlertLevel.INFO) -> None:
        self._channels: Dict[str, AbstractAlertChannel] = {}
        self._min_level = min_level
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._total_sent = 0
        self._total_dropped = 0

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

    def register_channel(self, channel: AbstractAlertChannel) -> None:
        self._channels[channel.name] = channel
        logger.info("AlertManager: registered channel '%s'", channel.name)

    def unregister_channel(self, name: str) -> bool:
        removed = self._channels.pop(name, None) is not None
        if removed:
            logger.info("AlertManager: unregistered channel '%s'", name)
        return removed

    def get_channel(self, name: str) -> Optional[AbstractAlertChannel]:
        return self._channels.get(name)

    @property
    def channels(self) -> List[str]:
        return list(self._channels.keys())

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(self, event: AlertEvent) -> int:
        """Send an alert to all enabled channels concurrently.

        Returns the number of channels that successfully delivered.
        """
        if event.level < self._min_level:
            self._total_dropped += 1
            return 0

        if not self._channels:
            return 0

        tasks = [
            channel.send(event)
            for channel in self._channels.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        sent = sum(1 for r in results if r is True)
        self._total_sent += sent

        # Prometheus
        for channel, result in zip(self._channels.values(), results):
            if result is True and _CTR_ALERTS:
                _CTR_ALERTS.labels(
                    level=event.level.name,
                    channel=channel.name,
                ).inc()

        return sent

    # ------------------------------------------------------------------
    # Fire-and-forget queue
    # ------------------------------------------------------------------

    def enqueue(self, event: AlertEvent) -> None:
        """Non-blocking enqueue for fire-and-forget alerting."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._total_dropped += 1
            logger.warning("AlertManager: queue full, dropping alert")

    async def start_worker(self) -> None:
        """Start the background queue worker coroutine."""
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("AlertManager: worker started")

    async def stop_worker(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("AlertManager: worker stopped")

    async def _worker(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self.send(event)
            except Exception as exc:  # noqa: BLE001
                logger.error("AlertManager: worker error: %s", exc)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    async def info(self, title: str, body: str = "", **kwargs) -> int:
        return await self.send(AlertEvent.info(title, body, **kwargs))

    async def warning(self, title: str, body: str = "", **kwargs) -> int:
        return await self.send(AlertEvent.warning(title, body, **kwargs))

    async def error(self, title: str, body: str = "", **kwargs) -> int:
        return await self.send(AlertEvent.error(title, body, **kwargs))

    async def critical(self, title: str, body: str = "", **kwargs) -> int:
        return await self.send(AlertEvent.critical(title, body, **kwargs))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "channels": [ch.status() for ch in self._channels.values()],
            "total_sent": self._total_sent,
            "total_dropped": self._total_dropped,
            "queue_size": self._queue.qsize(),
        }
