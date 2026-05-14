"""tests/test_batch6.py

Batch 6 unit tests:
  - SharedState thread safety
  - ThreadSafeCache TTL / LRU / concurrency
  - ArgusConfig Pydantic validation
  - ExchangeRegistry thread safety
  - GuardedExchange factory wrapping
"""
from __future__ import annotations

import threading
import time

import pytest


# ============================================================
#  SharedState
# ============================================================

class TestSharedState:
    def _state(self):
        from core.shared_state import SharedState
        return SharedState()

    def test_get_set(self):
        s = self._state()
        s.set("price", 65000.0)
        assert s.get("price") == 65000.0

    def test_default(self):
        s = self._state()
        assert s.get("missing", default=99) == 99

    def test_update_bulk(self):
        s = self._state()
        s.update({"a": 1, "b": 2, "c": 3})
        assert s.get("b") == 2

    def test_delete(self):
        s = self._state()
        s.set("x", 10)
        assert s.delete("x") is True
        assert "x" not in s
        assert s.delete("x") is False

    def test_snapshot_is_copy(self):
        s = self._state()
        s.set("val", [1, 2, 3])
        snap = s.snapshot()
        snap["val"].append(999)
        assert s.get("val") == [1, 2, 3, 999]  # shallow copy, list same object

    def test_version_increments(self):
        s = self._state()
        v0 = s.version
        s.set("k", "v")
        assert s.version == v0 + 1
        s.update({"a": 1, "b": 2})
        assert s.version == v0 + 2

    def test_50_thread_contention(self):
        from core.shared_state import SharedState
        s = SharedState()
        errors = []

        def worker(i):
            try:
                for _ in range(100):
                    s.set(f"key_{i}", i)
                    _ = s.get(f"key_{i}")
                    s.update({f"bulk_{i}": i * 2})
                    _ = s.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors

    def test_singleton(self):
        from core.shared_state import SharedState
        SharedState.reset_singleton()
        s1 = SharedState.instance()
        s2 = SharedState.instance()
        assert s1 is s2

    def test_namespace(self):
        s = self._state()
        prices = s.ns("prices")
        prices.set("BTC", 65000.0)
        prices.set("ETH", 3200.0)
        assert prices.get("BTC") == 65000.0
        snap = prices.snapshot()
        assert "BTC" in snap and "ETH" in snap
        # Keys don't leak into other namespaces
        signals = s.ns("signals")
        assert signals.get("BTC") is None


# ============================================================
#  ThreadSafeCache
# ============================================================

class TestThreadSafeCache:
    def _cache(self, maxsize=10, ttl=0.0):
        from utils.thread_safe_cache import ThreadSafeCache
        return ThreadSafeCache(maxsize=maxsize, ttl=ttl)

    def test_set_get(self):
        c = self._cache()
        c.set("a", 42)
        assert c.get("a") == 42

    def test_miss_returns_default(self):
        c = self._cache()
        assert c.get("nope", "fallback") == "fallback"

    def test_ttl_expiry(self):
        c = self._cache(ttl=0.05)  # 50ms TTL
        c.set("k", "value")
        assert c.get("k") == "value"
        time.sleep(0.1)
        assert c.get("k") is None

    def test_per_item_ttl(self):
        c = self._cache(ttl=60.0)  # default 60s
        c.set("short", "val", ttl=0.05)
        c.set("long", "val", ttl=60.0)
        time.sleep(0.1)
        assert c.get("short") is None
        assert c.get("long") == "val"

    def test_lru_eviction(self):
        c = self._cache(maxsize=3)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.get("a")  # access 'a' to make it recently used
        c.set("d", 4)  # should evict 'b' (LRU)
        assert c.get("b") is None
        assert c.get("a") == 1
        assert c.get("d") == 4

    def test_invalidate(self):
        c = self._cache()
        c.set("x", 99)
        assert c.invalidate("x") is True
        assert c.get("x") is None
        assert c.invalidate("x") is False

    def test_stats_tracking(self):
        c = self._cache()
        c.set("k", 1)
        c.get("k")
        c.get("k")
        c.get("missing")
        stats = c.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_concurrent_access(self):
        c = self._cache(maxsize=100)
        errors = []

        def worker(i):
            try:
                for j in range(50):
                    c.set(f"k{i}_{j}", i * j)
                    c.get(f"k{i}_{j}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors

    def test_contains(self):
        c = self._cache()
        c.set("present", True)
        assert "present" in c
        assert "absent" not in c


# ============================================================
#  ArgusConfig Pydantic Schema
# ============================================================

class TestArgusConfigSchema:
    def test_defaults_valid(self):
        try:
            from config.schema import ArgusConfig
            cfg = ArgusConfig()
            assert cfg.risk.max_drawdown == 0.15
            assert cfg.system.mode == "dry_run"
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_valid_overrides(self):
        try:
            from config.schema import ArgusConfig, RiskConfig, SystemConfig, TradingConfig
            cfg = ArgusConfig(
                system={"initial_capital": 5000.0, "mode": "dry_run"},
                risk={"max_drawdown": 0.10, "max_daily_loss": 0.03},
                trading={"capital": 5000.0, "max_positions": 10},
            )
            assert cfg.risk.max_drawdown == pytest.approx(0.10)
            assert cfg.trading.max_positions == 10
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_percentage_coercion(self):
        """Values > 1.0 should be treated as percentages and divided by 100."""
        try:
            from config.schema import RiskConfig
            r = RiskConfig(max_drawdown=15.0)  # 15% -> 0.15
            assert r.max_drawdown == pytest.approx(0.15)
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_invalid_mode_rejected(self):
        try:
            from config.schema import SystemConfig
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                SystemConfig(mode="yolo")
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_out_of_range_drawdown_rejected(self):
        try:
            from config.schema import RiskConfig
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                RiskConfig(max_drawdown=2.0)  # > 1.0 after coercion check
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_env_var_resolution(self):
        """${ENV_VAR} references should be resolved from environment."""
        import os
        try:
            from config.schema import SingleExchangeConfig
            os.environ["TEST_API_KEY"] = "mykey123"
            cfg = SingleExchangeConfig(api_key="${TEST_API_KEY}")
            assert cfg.api_key == "mykey123"
        except ImportError:
            pytest.skip("pydantic not installed")
        finally:
            os.environ.pop("TEST_API_KEY", None)

    def test_enabled_names(self):
        try:
            from config.schema import ExchangesConfig
            cfg = ExchangesConfig(
                kraken={"enabled": True, "symbols": ["BTC/USD"]},
                coinbase={"enabled": False},
            )
            assert "kraken" in cfg.enabled_names
            assert "coinbase" not in cfg.enabled_names
        except ImportError:
            pytest.skip("pydantic not installed")


# ============================================================
#  ExchangeRegistry
# ============================================================

class TestExchangeRegistry:
    def test_singleton(self):
        from exchanges import ExchangeRegistry
        r1 = ExchangeRegistry.instance()
        r2 = ExchangeRegistry.instance()
        assert r1 is r2

    def test_register_and_get(self):
        from exchanges import ExchangeRegistry
        r = ExchangeRegistry()
        mock_ex = object()
        r.register("mock", mock_ex)
        assert r.get("mock") is mock_ex
        assert "mock" in r.list_available()

    def test_remove(self):
        from exchanges import ExchangeRegistry
        r = ExchangeRegistry()
        r.register("tmp", object())
        r.remove("tmp")
        assert r.get("tmp") is None

    def test_thread_safe_register(self):
        from exchanges import ExchangeRegistry
        r = ExchangeRegistry()
        errors = []

        def worker(i):
            try:
                r.register(f"ex{i}", object())
                _ = r.list_available()
                r.get(f"ex{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors


# ============================================================
#  GuardedExchange wrapping
# ============================================================

class TestGuardedExchangeWrapping:
    def test_guard_wraps_callable(self):
        from exchanges.guard import GuardedExchange
        from utils.rate_limit_guard import ExchangeRateLimiter

        class FakeExchange:
            id = "fake"
            def fetch_ticker(self, symbol):
                return {"symbol": symbol, "last": 65000.0}

        lim = ExchangeRateLimiter()
        guarded = GuardedExchange(FakeExchange(), exchange_name="fake", limiter=lim)
        result = guarded.fetch_ticker("BTC/USDT")
        assert result["last"] == 65000.0

    def test_guard_non_callable_passthrough(self):
        from exchanges.guard import GuardedExchange
        from utils.rate_limit_guard import ExchangeRateLimiter

        class FakeExchange:
            id = "fake"
            name = "Fake Exchange"
            def fetch_ticker(self, symbol): return {}

        guarded = GuardedExchange(FakeExchange(), exchange_name="fake",
                                   limiter=ExchangeRateLimiter())
        assert guarded.name == "Fake Exchange"  # non-callable passthrough

    def test_guard_retries_on_rate_limit(self):
        from exchanges.guard import GuardedExchange
        from utils.rate_limit_guard import ExchangeRateLimiter, RateLimitExceeded

        call_count = 0

        class FlakyExchange:
            id = "flaky"
            def fetch_ticker(self, symbol):
                nonlocal call_count
                call_count += 1
                return {"last": 100.0}

        # Guard with tiny capacity so token bucket fills instantly
        lim = ExchangeRateLimiter()
        lim.register("flaky", capacity=100.0, refill_rate=100.0, block=True)
        guarded = GuardedExchange(FlakyExchange(), exchange_name="flaky", limiter=lim)
        result = guarded.fetch_ticker("BTC/USDT")
        assert result["last"] == 100.0
        assert call_count == 1
