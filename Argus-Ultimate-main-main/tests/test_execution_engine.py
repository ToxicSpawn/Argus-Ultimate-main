"""
tests/test_execution_engine.py
================================
Property-based + unit tests for ExecutionEngine and ArgusAIAdapter.
Uses pytest + Hypothesis for generative testing.

Run:  pytest tests/test_execution_engine.py -v
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.execution_engine import ExecutionEngine, ExecutionRequest, ExecutionResult
from core.argus_ai_adapter import AISignal, ArgusAIAdapter, build_argus_ai_adapter

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_request(**kwargs) -> ExecutionRequest:
    defaults = dict(
        symbol="BTC/USDT", side="buy", quantity=0.001,
        price=50000.0, strategy_name="test", signal_confidence=0.8,
    )
    return ExecutionRequest(**{**defaults, **kwargs})


def make_mock_router(fill_price: float = 50000.0, fee: float = 1.0) -> AsyncMock:
    router = AsyncMock()
    router.place_order.return_value = {
        "id": "mock_order_1",
        "filled": 0.001,
        "price": fill_price,
        "fee": fee,
    }
    return router


# ---------------------------------------------------------------------------
# ExecutionEngine — unit tests
# ---------------------------------------------------------------------------

class TestExecutionEngineDryRun:

    def test_dry_run_returns_synthetic_fill(self):
        engine = ExecutionEngine(dry_run=True)
        req = make_request(price=50000.0, quantity=0.001)
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is True
        assert result.filled_quantity == pytest.approx(0.001)
        assert result.filled_price == pytest.approx(50000.0)
        assert result.fee == pytest.approx(0.0)

    def test_dry_run_market_order(self):
        engine = ExecutionEngine(dry_run=True)
        req = make_request(price=None)
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is True
        assert result.filled_price == pytest.approx(0.0)

    def test_stats_increment(self):
        engine = ExecutionEngine(dry_run=True)
        req = make_request()
        asyncio.get_event_loop().run_until_complete(engine.execute(req))
        asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert engine.stats["placed"] == 2
        assert engine.stats["total_attempted"] == 2

    def test_stats_reset(self):
        engine = ExecutionEngine(dry_run=True)
        req = make_request()
        asyncio.get_event_loop().run_until_complete(engine.execute(req))
        engine.reset_stats()
        assert engine.stats["placed"] == 0


class TestExecutionEngineRiskFacade:

    def test_risk_rejection_blocks_order(self):
        risk = MagicMock()
        risk.check.return_value = False
        engine = ExecutionEngine(dry_run=True, risk_facade=risk)
        req = make_request()
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is False
        assert result.error == "blocked_by_risk_facade"
        assert engine.stats["rejected"] == 1

    def test_risk_approval_passes_through(self):
        risk = MagicMock()
        risk.check.return_value = True
        engine = ExecutionEngine(dry_run=True, risk_facade=risk)
        req = make_request()
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is True

    def test_risk_exception_blocks_order(self):
        risk = MagicMock()
        risk.check.side_effect = RuntimeError("risk exploded")
        engine = ExecutionEngine(dry_run=True, risk_facade=risk)
        req = make_request()
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is False


class TestExecutionEngineLiveRouter:

    def test_live_router_fill(self):
        router = make_mock_router(fill_price=49999.0)
        engine = ExecutionEngine(order_router=router, dry_run=False)
        req = make_request(price=50000.0)
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is True
        assert result.filled_price == pytest.approx(49999.0)
        assert result.order_id == "mock_order_1"

    def test_router_exception_returns_failed_result(self):
        router = AsyncMock()
        router.place_order.side_effect = ConnectionError("exchange down")
        engine = ExecutionEngine(order_router=router, dry_run=False)
        req = make_request()
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is False
        assert "exchange down" in result.error

    def test_no_router_returns_failed_result(self):
        engine = ExecutionEngine(dry_run=False)
        req = make_request()
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert result.success is False
        assert result.error == "no_order_router_configured"


class TestExecutionEngineBatch:

    def test_batch_execute_all_succeed(self):
        engine = ExecutionEngine(dry_run=True)
        reqs = [make_request(symbol=f"COIN{i}/USDT") for i in range(5)]
        results = asyncio.get_event_loop().run_until_complete(
            engine.execute_batch(reqs)
        )
        assert all(r.success for r in results)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# ArgusAIAdapter — unit tests
# ---------------------------------------------------------------------------

class TestArgusAIAdapter:

    def _make_adapter(self, min_conf=0.55) -> ArgusAIAdapter:
        engine = ExecutionEngine(dry_run=True)
        return ArgusAIAdapter(engine, min_confidence=min_conf)

    def test_dispatch_valid_signal(self):
        adapter = self._make_adapter()
        sig = AISignal(symbol="BTC/USDT", side="buy",
                       quantity=0.001, confidence=0.8)
        result = asyncio.get_event_loop().run_until_complete(adapter.dispatch(sig))
        assert result is not None
        assert result.success is True

    def test_dispatch_below_threshold_returns_none(self):
        adapter = self._make_adapter(min_conf=0.7)
        sig = AISignal(symbol="ETH/USDT", side="sell",
                       quantity=0.01, confidence=0.5)
        result = asyncio.get_event_loop().run_until_complete(adapter.dispatch(sig))
        assert result is None
        assert adapter.stats["signals_gated"] == 1

    def test_dispatch_dict_signal(self):
        adapter = self._make_adapter()
        raw = {"symbol": "ETH/USDT", "side": "buy",
               "quantity": 0.01, "confidence": 0.9}
        result = asyncio.get_event_loop().run_until_complete(adapter.dispatch(raw))
        assert result is not None and result.success

    def test_dispatch_malformed_dict_returns_none(self):
        adapter = self._make_adapter()
        result = asyncio.get_event_loop().run_until_complete(
            adapter.dispatch({"symbol": "BTC/USDT"})  # missing required fields
        )
        assert result is None

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match="side"):
            AISignal.from_dict({"symbol": "BTC/USDT", "side": "long",
                                "quantity": 0.001, "confidence": 0.8})

    def test_confidence_threshold_hot_update(self):
        adapter = self._make_adapter(min_conf=0.5)
        adapter.set_confidence_threshold(0.9)
        sig = AISignal(symbol="BTC/USDT", side="buy",
                       quantity=0.001, confidence=0.8)
        result = asyncio.get_event_loop().run_until_complete(adapter.dispatch(sig))
        assert result is None  # gated by new threshold

    def test_event_bus_called_on_fill(self):
        bus = MagicMock()
        engine = ExecutionEngine(dry_run=True)
        adapter = ArgusAIAdapter(engine, event_bus=bus)
        sig = AISignal(symbol="SOL/USDT", side="buy",
                       quantity=1.0, confidence=0.9)
        asyncio.get_event_loop().run_until_complete(adapter.dispatch(sig))
        bus.publish.assert_called()
        topics = [call.args[0] for call in bus.publish.call_args_list]
        assert "ai.execution.fill" in topics

    def test_batch_dispatch(self):
        adapter = self._make_adapter()
        sigs = [
            AISignal("BTC/USDT", "buy", 0.001, 0.9),
            AISignal("ETH/USDT", "sell", 0.01, 0.3),  # will be gated
            AISignal("SOL/USDT", "buy", 1.0, 0.75),
        ]
        results = asyncio.get_event_loop().run_until_complete(
            adapter.dispatch_batch(sigs)
        )
        assert results[0] is not None  # passed
        assert results[1] is None      # gated
        assert results[2] is not None  # passed

    def test_stats_accuracy(self):
        adapter = self._make_adapter(min_conf=0.6)
        sigs = [
            AISignal("BTC/USDT", "buy", 0.001, 0.9),
            AISignal("ETH/USDT", "buy", 0.01, 0.4),
            AISignal("SOL/USDT", "sell", 1.0, 0.8),
        ]
        for s in sigs:
            asyncio.get_event_loop().run_until_complete(adapter.dispatch(s))
        stats = adapter.stats
        assert stats["signals_received"] == 3
        assert stats["signals_gated"] == 1
        assert stats["signals_dispatched"] == 2


# ---------------------------------------------------------------------------
# Hypothesis property-based tests (if available)
# ---------------------------------------------------------------------------

if HYPOTHESIS_AVAILABLE:

    @given(
        qty=st.floats(min_value=1e-8, max_value=1e6, allow_nan=False, allow_infinity=False),
        price=st.floats(min_value=0.01, max_value=1e7, allow_nan=False, allow_infinity=False),
        side=st.sampled_from(["buy", "sell"]),
        conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_property_dry_run_never_raises(qty, price, side, conf):
        """ExecutionEngine in dry_run must never raise for valid inputs."""
        engine = ExecutionEngine(dry_run=True)
        req = ExecutionRequest(
            symbol="BTC/USDT", side=side, quantity=qty,
            price=price, strategy_name="prop", signal_confidence=conf,
        )
        result = asyncio.get_event_loop().run_until_complete(engine.execute(req))
        assert isinstance(result, ExecutionResult)
        assert result.success is True

    @given(
        conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_property_confidence_gate_monotonic(conf, threshold):
        """Signals below threshold must always be gated, never dispatched."""
        engine = ExecutionEngine(dry_run=True)
        adapter = ArgusAIAdapter(engine, min_confidence=threshold)
        sig = AISignal("BTC/USDT", "buy", 0.001, conf)
        result = asyncio.get_event_loop().run_until_complete(adapter.dispatch(sig))
        if conf < threshold:
            assert result is None
        else:
            assert result is not None
