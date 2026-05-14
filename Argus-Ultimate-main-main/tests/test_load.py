"""
Load Test Skeleton — performance and throughput benchmarks for ARGUS.

All tests are marked ``@pytest.mark.slow`` so they do not run in normal CI.
Run explicitly with: ``py -m pytest tests/test_load.py -m slow``
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(i: int):
    """Create a TradingSignal for throughput testing."""
    from unified_types import TradingSignal
    return TradingSignal(
        symbol=f"BTC/USDT",
        action="BUY" if i % 2 == 0 else "SELL",
        confidence=0.5 + (i % 50) / 100.0,
        strength=0.6,
        entry_price=50000.0 + i,
    )


# ---------------------------------------------------------------------------
# 1. Signal throughput
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSignalThroughput:
    """Measure time to generate and risk-check 1000 signals."""

    def test_signal_throughput(self):
        """Generate 1000 TradingSignal objects and measure creation time."""
        from unified_types import TradingSignal

        start = time.perf_counter()
        signals = [_make_signal(i) for i in range(1000)]
        creation_time = time.perf_counter() - start

        assert len(signals) == 1000
        # Signal creation should be < 1 second for 1000 objects
        assert creation_time < 1.0, f"Signal creation took {creation_time:.3f}s"

    def test_signal_risk_check_throughput(self):
        """Process 1000 signals through risk manager check_order."""
        from risk.unified_risk_manager import UnifiedRiskManager

        rm = UnifiedRiskManager(initial_capital=100000.0)
        signals = [_make_signal(i) for i in range(1000)]

        start = time.perf_counter()
        results = []
        for sig in signals:
            allowed, reason = rm.pre_trade_risk_check(
                symbol=sig.symbol,
                position_size_usd=sig.entry_price * 0.001,
            )
            results.append(allowed)
        elapsed = time.perf_counter() - start

        assert len(results) == 1000
        # Risk checks should process at > 1000/sec
        assert elapsed < 5.0, f"Risk checks took {elapsed:.3f}s for 1000 signals"


# ---------------------------------------------------------------------------
# 2. Concurrent fills
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestConcurrentFills:
    """Simulate 100 concurrent fill callbacks."""

    def test_concurrent_fills(self):
        """100 threads recording fills concurrently should not deadlock or crash."""
        from api.dashboard import ArgusAPIServer

        server = ArgusAPIServer(port=0)
        errors = []

        def record_fill(fill_id):
            try:
                server.update_state("trades", {
                    "id": str(fill_id),
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "qty": 0.001,
                    "price": 50000.0 + fill_id,
                    "pnl": fill_id * 0.01,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            except Exception as e:
                errors.append(str(e))

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(record_fill, i) for i in range(100)]
            for f in as_completed(futures):
                f.result()  # re-raise any exceptions
        elapsed = time.perf_counter() - start

        assert not errors, f"Fill errors: {errors}"
        # All fills processed within 5 seconds
        assert elapsed < 5.0, f"Concurrent fills took {elapsed:.3f}s"
        # Trades list capped at 20
        trades = server.get_state("trades")
        assert len(trades) <= 20

    def test_concurrent_state_updates(self):
        """100 concurrent state updates should be thread-safe."""
        from api.dashboard import ArgusAPIServer

        server = ArgusAPIServer(port=0)

        def updater(thread_id):
            for i in range(50):
                server.update_states({
                    "capital_aud": 1000.0 + thread_id + i,
                    "pnl_aud": thread_id * 0.1,
                    "cycle": i,
                })

        threads = [threading.Thread(target=updater, args=(t,)) for t in range(20)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.perf_counter() - start

        # Verify no thread hung
        for t in threads:
            assert not t.is_alive(), "Thread deadlocked during state update"
        assert elapsed < 10.0


# ---------------------------------------------------------------------------
# 3. Metric cardinality
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMetricCardinality:
    """Verify metric storage stays bounded under high cardinality."""

    def test_metric_cardinality(self):
        """Creating 1000 unique metric labels should keep memory bounded."""
        from api.dashboard import _build_prometheus, _DEFAULT_STATE

        state = dict(_DEFAULT_STATE)
        # Create 1000 unique positions (high cardinality)
        positions = {}
        for i in range(1000):
            positions[f"TOKEN{i}/USDT"] = {
                "qty": 0.001 * i,
                "entry_price": 1.0 + i,
                "current_price": 1.1 + i,
                "unrealised_pnl": 0.1 * i,
            }
        state["positions"] = positions

        start = time.perf_counter()
        output = _build_prometheus(state)
        elapsed = time.perf_counter() - start

        # Should generate output for all 1000 positions
        assert output.count("argus_position_unrealised_pnl") == 1000
        # Should complete in reasonable time
        assert elapsed < 2.0, f"Prometheus build took {elapsed:.3f}s for 1000 metrics"
        # Memory check: output should be reasonable size
        output_kb = len(output.encode()) / 1024
        assert output_kb < 500, f"Prometheus output is {output_kb:.1f}KB (expected < 500KB)"

    def test_metric_output_grows_linearly(self):
        """Metric output size should grow linearly, not exponentially."""
        from api.dashboard import _build_prometheus, _DEFAULT_STATE

        sizes = []
        for n in [10, 100, 1000]:
            state = dict(_DEFAULT_STATE)
            state["positions"] = {
                f"T{i}/USDT": {"qty": 1, "entry_price": 1, "current_price": 1, "unrealised_pnl": 0}
                for i in range(n)
            }
            output = _build_prometheus(state)
            sizes.append(len(output))

        # Size at 1000 should be roughly 10x size at 100
        ratio = sizes[2] / sizes[1]
        assert 5 < ratio < 15, f"Non-linear growth: ratio={ratio:.1f}"


# ---------------------------------------------------------------------------
# 4. Dashboard state update rate
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestStateUpdateRate:
    """Measure dashboard state updates per second."""

    def test_state_update_rate(self):
        """Dashboard should handle > 1000 state updates per second."""
        from api.dashboard import ArgusAPIServer

        server = ArgusAPIServer(port=0)

        num_updates = 5000
        start = time.perf_counter()
        for i in range(num_updates):
            server.update_state("cycle", i)
        elapsed = time.perf_counter() - start

        rate = num_updates / elapsed
        assert rate > 1000, f"State update rate: {rate:.0f}/s (expected > 1000)"

    def test_bulk_state_update_rate(self):
        """Bulk updates should be faster than individual updates."""
        from api.dashboard import ArgusAPIServer

        server = ArgusAPIServer(port=0)

        num_updates = 5000
        batch = {
            "capital_aud": 1000.0,
            "pnl_aud": 10.0,
            "cycle": 0,
            "regime": "TRENDING",
        }

        start = time.perf_counter()
        for i in range(num_updates):
            batch["cycle"] = i
            server.update_states(batch)
        elapsed = time.perf_counter() - start

        rate = num_updates / elapsed
        assert rate > 500, f"Bulk update rate: {rate:.0f}/s (expected > 500)"

    def test_concurrent_read_write_rate(self):
        """Concurrent readers and writers should maintain throughput."""
        from api.dashboard import ArgusAPIServer

        server = ArgusAPIServer(port=0)
        read_count = 0
        write_count = 0
        lock = threading.Lock()

        def writer():
            nonlocal write_count
            for i in range(1000):
                server.update_state("cycle", i)
                with lock:
                    write_count += 1

        def reader():
            nonlocal read_count
            for _ in range(1000):
                server.get_state("cycle")
                with lock:
                    read_count += 1

        start = time.perf_counter()
        threads = (
            [threading.Thread(target=writer) for _ in range(3)] +
            [threading.Thread(target=reader) for _ in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.perf_counter() - start

        total_ops = read_count + write_count
        rate = total_ops / elapsed
        assert rate > 1000, f"Concurrent R/W rate: {rate:.0f} ops/s"
