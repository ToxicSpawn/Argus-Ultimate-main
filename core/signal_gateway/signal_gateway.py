"""SignalGateway — main pipeline: ingest → validate → dedup → consensus → emit."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Coroutine, List, Optional

from core.signal_gateway.signal_envelope import SignalEnvelope
from core.signal_gateway.gateway_config import GatewayConfig
from core.signal_gateway.signal_validator import SignalValidator
from core.signal_gateway.signal_deduplicator import SignalDeduplicator
from core.signal_gateway.consensus_engine import ConsensusEngine, ConsensusResult

logger = logging.getLogger(__name__)

# Optional Prometheus metrics — graceful degradation if not installed.
try:
    from prometheus_client import Counter

    _INGESTED = Counter(
        "argus_gateway_signals_ingested_total",
        "Total signals ingested by SignalGateway",
        ["source"],
    )
    _REJECTED = Counter(
        "argus_gateway_signals_rejected_total",
        "Signals rejected by validator or TTL",
        ["source", "reason"],
    )
    _CONSENSUS = Counter(
        "argus_gateway_consensus_fired_total",
        "Consensus decisions fired",
        ["direction"],
    )
    _DEDUP = Counter(
        "argus_gateway_dedup_blocked_total",
        "Signals blocked by deduplicator",
        ["source"],
    )
    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROM_AVAILABLE = False


ConsensusCallback = Callable[[ConsensusResult], Coroutine]


class SignalGateway:
    """Unified signal gateway for Argus.

    Usage
    -----
    gateway = SignalGateway(config)
    gateway.on_consensus(my_async_callback)
    await gateway.start()
    await gateway.ingest(envelope)   # called by each signal source
    await gateway.stop()
    """

    def __init__(
        self,
        config: Optional[GatewayConfig] = None,
        batch_window_ms: int = 50,
    ) -> None:
        self._config = config or GatewayConfig()
        self._batch_window_ms = batch_window_ms
        self._validator = SignalValidator(self._config)
        self._deduplicator = SignalDeduplicator(self._config)
        self._consensus = ConsensusEngine(self._config)
        self._queue: asyncio.Queue[SignalEnvelope] = asyncio.Queue()
        self._callbacks: List[ConsensusCallback] = []
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Internal stats.
        self._ingested: int = 0
        self._rejected: int = 0
        self._dedup_blocked: int = 0
        self._consensus_fired: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background batch-processing worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("SignalGateway started (batch_window=%dms)", self._batch_window_ms)

    async def stop(self) -> None:
        """Gracefully stop the gateway worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("SignalGateway stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest(self, envelope: SignalEnvelope) -> None:
        """Submit a signal envelope into the gateway pipeline."""
        await self._queue.put(envelope)

    def on_consensus(self, callback: ConsensusCallback) -> None:
        """Register an async callback invoked when consensus fires."""
        self._callbacks.append(callback)

    def get_stats(self) -> dict:
        """Return current gateway counters."""
        return {
            "ingested": self._ingested,
            "rejected": self._rejected,
            "dedup_blocked": self._dedup_blocked,
            "consensus_fired": self._consensus_fired,
            "queue_depth": self._queue.qsize(),
        }

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Drain queue in batches every batch_window_ms milliseconds."""
        while self._running:
            batch: List[SignalEnvelope] = []
            try:
                # Wait up to batch_window_ms for the first item.
                deadline = self._batch_window_ms / 1000
                envelope = await asyncio.wait_for(
                    self._queue.get(), timeout=deadline
                )
                batch.append(envelope)
                # Drain any further items already queued without waiting.
                while not self._queue.empty():
                    batch.append(self._queue.get_nowait())
            except asyncio.TimeoutError:
                pass  # Nothing arrived — run consensus on empty batch (no-op).

            if batch:
                await self._process_batch(batch)

    async def _process_batch(
        self, batch: List[SignalEnvelope]
    ) -> None:
        """Validate, dedup, then run consensus on *batch*."""
        valid_envelopes: List[SignalEnvelope] = []

        for env in batch:
            self._ingested += 1
            if _PROM_AVAILABLE:
                _INGESTED.labels(source=env.source.value).inc()

            result = self._validator.validate(env)
            if not result:
                self._rejected += 1
                if _PROM_AVAILABLE:
                    _REJECTED.labels(
                        source=env.source.value,
                        reason=result.reason or "unknown",
                    ).inc()
                logger.debug(
                    "Signal rejected: source=%s reason=%s",
                    env.source.value,
                    result.reason,
                )
                continue

            if await self._deduplicator.is_duplicate(env):
                self._dedup_blocked += 1
                if _PROM_AVAILABLE:
                    _DEDUP.labels(source=env.source.value).inc()
                logger.debug(
                    "Signal deduped: source=%s direction=%s",
                    env.source.value,
                    env.direction,
                )
                continue

            valid_envelopes.append(env)

        if not valid_envelopes:
            return

        consensus = self._consensus.evaluate(valid_envelopes)

        if consensus.fired:
            self._consensus_fired += 1
            if _PROM_AVAILABLE:
                _CONSENSUS.labels(
                    direction=consensus.winning_direction or "flat"
                ).inc()
            logger.info(
                "Consensus fired: direction=%s confidence=%.3f sources=%s",
                consensus.winning_direction,
                consensus.aggregate_confidence,
                [s.value for s in consensus.participating_sources],
            )
            for cb in self._callbacks:
                try:
                    await cb(consensus)
                except Exception:
                    logger.exception("Consensus callback raised an exception")
