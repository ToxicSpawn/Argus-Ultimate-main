"""Tests for core/signal_gateway — Push 35 (32 tests)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.signal_gateway.signal_source import SignalSource, DEFAULT_SOURCE_WEIGHTS
from core.signal_gateway.signal_envelope import SignalEnvelope
from core.signal_gateway.gateway_config import GatewayConfig
from core.signal_gateway.signal_validator import SignalValidator, ValidationResult
from core.signal_gateway.signal_deduplicator import SignalDeduplicator
from core.signal_gateway.consensus_engine import ConsensusEngine, ConsensusResult
from core.signal_gateway.signal_gateway import SignalGateway
from core.signal_gateway.gateway_wiring import wire_gateway_to_argus_bot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(
    source=SignalSource.VOID_BREAKER,
    direction="long",
    confidence=0.8,
    ttl_ms=500,
) -> SignalEnvelope:
    return SignalEnvelope(
        source=source,
        direction=direction,
        confidence=confidence,
        ttl_ms=ttl_ms,
    )


def _cfg(**kwargs) -> GatewayConfig:
    return GatewayConfig(**kwargs)


# ---------------------------------------------------------------------------
# SignalEnvelope
# ---------------------------------------------------------------------------

class TestSignalEnvelope:
    def test_not_expired_fresh(self):
        env = _env(ttl_ms=500)
        assert not env.is_expired()

    def test_expired_old(self):
        old_ns = time.time_ns() - 600 * 1_000_000  # 600ms ago
        env = SignalEnvelope(
            source=SignalSource.RL_AGENT,
            direction="short",
            confidence=0.7,
            timestamp_ns=old_ns,
            ttl_ms=500,
        )
        assert env.is_expired()

    def test_ttl_zero_never_expires(self):
        old_ns = time.time_ns() - 10_000 * 1_000_000  # 10 seconds ago
        env = SignalEnvelope(
            source=SignalSource.DEEPLOB,
            direction="long",
            confidence=0.5,
            timestamp_ns=old_ns,
            ttl_ms=0,
        )
        assert not env.is_expired()

    def test_round_trip_dict(self):
        env = _env()
        assert SignalEnvelope.from_dict(env.to_dict()).direction == env.direction

    def test_age_ms_positive(self):
        env = _env()
        assert env.age_ms() >= 0


# ---------------------------------------------------------------------------
# SignalValidator
# ---------------------------------------------------------------------------

class TestSignalValidator:
    def setup_method(self):
        self.validator = SignalValidator(_cfg())

    def test_valid_long(self):
        assert self.validator.validate(_env(direction="long"))

    def test_valid_short(self):
        assert self.validator.validate(_env(direction="short"))

    def test_valid_flat(self):
        assert self.validator.validate(_env(direction="flat"))

    def test_invalid_direction(self):
        result = self.validator.validate(_env(direction="buy"))
        assert not result
        assert "direction" in result.reason

    def test_confidence_out_of_range_high(self):
        result = self.validator.validate(_env(confidence=1.5))
        assert not result

    def test_confidence_out_of_range_low(self):
        result = self.validator.validate(_env(confidence=-0.1))
        assert not result

    def test_source_not_in_enabled(self):
        cfg = _cfg(enabled_sources=[SignalSource.VOID_BREAKER])
        validator = SignalValidator(cfg)
        result = validator.validate(_env(source=SignalSource.LLM_OVERLAY))
        assert not result
        assert "enabled_sources" in result.reason

    def test_expired_envelope_rejected(self):
        old_ns = time.time_ns() - 600 * 1_000_000
        env = SignalEnvelope(
            source=SignalSource.VOID_BREAKER,
            direction="long",
            confidence=0.8,
            timestamp_ns=old_ns,
            ttl_ms=500,
        )
        result = self.validator.validate(env)
        assert not result
        assert "expired" in result.reason


# ---------------------------------------------------------------------------
# SignalDeduplicator
# ---------------------------------------------------------------------------

class TestSignalDeduplicator:
    def test_first_signal_not_duplicate(self):
        dedup = SignalDeduplicator(_cfg(dedup_window_ms=200))
        env = _env()
        result = asyncio.get_event_loop().run_until_complete(dedup.is_duplicate(env))
        assert not result

    def test_second_identical_is_duplicate(self):
        dedup = SignalDeduplicator(_cfg(dedup_window_ms=200))
        env = _env()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(dedup.is_duplicate(env))
        assert loop.run_until_complete(dedup.is_duplicate(env))

    def test_different_direction_not_duplicate(self):
        dedup = SignalDeduplicator(_cfg(dedup_window_ms=200))
        loop = asyncio.get_event_loop()
        loop.run_until_complete(dedup.is_duplicate(_env(direction="long")))
        result = loop.run_until_complete(dedup.is_duplicate(_env(direction="short")))
        assert not result

    def test_blocked_counter_increments(self):
        dedup = SignalDeduplicator(_cfg(dedup_window_ms=200))
        env = _env()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(dedup.is_duplicate(env))
        loop.run_until_complete(dedup.is_duplicate(env))
        assert dedup.blocked_total == 1

    def test_reset_clears_state(self):
        dedup = SignalDeduplicator(_cfg(dedup_window_ms=200))
        env = _env()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(dedup.is_duplicate(env))
        dedup.reset()
        result = loop.run_until_complete(dedup.is_duplicate(env))
        assert not result


# ---------------------------------------------------------------------------
# ConsensusEngine
# ---------------------------------------------------------------------------

class TestConsensusEngine:
    def setup_method(self):
        self.engine = ConsensusEngine(_cfg(consensus_threshold=0.55, min_sources=2))

    def test_empty_batch_no_consensus(self):
        result = self.engine.evaluate([])
        assert not result.fired

    def test_single_source_below_min_sources(self):
        envs = [_env(source=SignalSource.VOID_BREAKER, direction="long", confidence=0.9)]
        result = self.engine.evaluate(envs)
        assert not result.fired

    def test_two_sources_same_direction_fires(self):
        envs = [
            _env(source=SignalSource.VOID_BREAKER, direction="long", confidence=0.9),
            _env(source=SignalSource.RL_AGENT, direction="long", confidence=0.85),
        ]
        result = self.engine.evaluate(envs)
        assert result.fired
        assert result.winning_direction == "long"

    def test_dissenting_sources_populated(self):
        envs = [
            _env(source=SignalSource.VOID_BREAKER, direction="long", confidence=0.9),
            _env(source=SignalSource.RL_AGENT, direction="long", confidence=0.85),
            _env(source=SignalSource.LLM_OVERLAY, direction="short", confidence=0.6),
        ]
        result = self.engine.evaluate(envs)
        assert result.fired
        assert SignalSource.LLM_OVERLAY in result.dissenting_sources

    def test_threshold_not_met_no_fire(self):
        # Two long at low confidence vs two short at high confidence → short wins
        # but force threshold above what they can achieve
        cfg = _cfg(consensus_threshold=0.99, min_sources=2)
        engine = ConsensusEngine(cfg)
        envs = [
            _env(source=SignalSource.VOID_BREAKER, direction="long", confidence=0.5),
            _env(source=SignalSource.RL_AGENT, direction="short", confidence=0.5),
        ]
        result = engine.evaluate(envs)
        assert not result.fired

    def test_aggregate_confidence_between_0_and_1(self):
        envs = [
            _env(source=SignalSource.VOID_BREAKER, direction="long", confidence=0.9),
            _env(source=SignalSource.RL_AGENT, direction="long", confidence=0.7),
        ]
        result = self.engine.evaluate(envs)
        assert 0.0 <= result.aggregate_confidence <= 1.0


# ---------------------------------------------------------------------------
# SignalGateway integration
# ---------------------------------------------------------------------------

class TestSignalGateway:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_stats_initial_zeroes(self):
        gw = SignalGateway()
        stats = gw.get_stats()
        assert stats["ingested"] == 0
        assert stats["consensus_fired"] == 0

    def test_ingest_queues_envelope(self):
        gw = SignalGateway()
        env = _env()
        self._run(gw.ingest(env))
        assert gw._queue.qsize() == 1

    def test_consensus_callback_invoked(self):
        cfg = _cfg(consensus_threshold=0.1, min_sources=1)
        gw = SignalGateway(config=cfg, batch_window_ms=10)
        fired = []

        async def cb(result: ConsensusResult):
            fired.append(result)

        gw.on_consensus(cb)

        async def run():
            await gw.start()
            await gw.ingest(_env(source=SignalSource.VOID_BREAKER, confidence=0.9))
            await asyncio.sleep(0.1)
            await gw.stop()

        self._run(run())
        assert len(fired) >= 1
        assert fired[0].winning_direction == "long"

    def test_rejected_signal_counted(self):
        gw = SignalGateway()
        env = SignalEnvelope(
            source=SignalSource.VOID_BREAKER,
            direction="invalid",
            confidence=0.8,
        )

        async def run():
            await gw.start()
            await gw.ingest(env)
            await asyncio.sleep(0.1)
            await gw.stop()

        self._run(run())
        assert gw._rejected >= 1


# ---------------------------------------------------------------------------
# Gateway wiring smoke test
# ---------------------------------------------------------------------------

class TestGatewayWiring:
    def test_wire_attaches_gateway(self):
        bot = MagicMock()
        bot._on_consensus_signal = AsyncMock()

        async def run():
            gw = await wire_gateway_to_argus_bot(bot)
            assert hasattr(bot, "signal_gateway")
            await gw.stop()

        asyncio.get_event_loop().run_until_complete(run())
