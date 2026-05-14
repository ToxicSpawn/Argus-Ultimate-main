from __future__ import annotations

import time
import unittest

from services.market_data_service import MarketDataService
from utils.circuit_breaker import CircuitBreaker


class _RetryOHLCVExchange:
    def __init__(self, *, fail_first_n: int = 0) -> None:
        self.fail_first_n = int(max(0, fail_first_n))
        self.calls = 0

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 200):
        _ = symbol, timeframe, limit
        self.calls += 1
        if self.calls <= self.fail_first_n:
            raise TimeoutError("simulated_ohlcv_timeout")
        now_ms = int(time.time() * 1000)
        return [[now_ms, 100.0, 101.0, 99.0, 100.5, 12.0]]


class _AlwaysFailOHLCVExchange:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 200):
        _ = symbol, timeframe, limit
        self.calls += 1
        raise TimeoutError("always_fail")


class TestMarketDataOHLCVRetries(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_retries_then_success(self) -> None:
        ex = _RetryOHLCVExchange(fail_first_n=2)
        svc = MarketDataService(
            exchanges={"kraken": ex},
            primary="kraken",
            ohlcv_ttl_s=0.0,
            ohlcv_poll_interval_s=0.0,
            ohlcv_retry_attempts=2,
            request_timeout_s=0.2,
        )
        df = await svc.fetch_ohlcv_df("BTC/USD", timeframe="1m", limit=2)
        self.assertIsNotNone(df)
        self.assertEqual(ex.calls, 3)
        self.assertEqual(str(svc._cb_ohlcv.state), "closed")

    async def test_poll_interval_reuses_cached_ohlcv(self) -> None:
        ex = _RetryOHLCVExchange(fail_first_n=0)
        svc = MarketDataService(
            exchanges={"kraken": ex},
            primary="kraken",
            ohlcv_ttl_s=0.0,
            ohlcv_poll_interval_s=60.0,
            ohlcv_retry_attempts=0,
            request_timeout_s=0.2,
        )
        d1 = await svc.fetch_ohlcv_df("ETH/USD", timeframe="1m", limit=2)
        d2 = await svc.fetch_ohlcv_df("ETH/USD", timeframe="1m", limit=2)
        self.assertIsNotNone(d1)
        self.assertIsNotNone(d2)
        self.assertEqual(ex.calls, 1)

    async def test_retries_exhausted_count_once_toward_circuit_breaker(self) -> None:
        ex = _AlwaysFailOHLCVExchange()
        svc = MarketDataService(
            exchanges={"kraken": ex},
            primary="kraken",
            ohlcv_ttl_s=0.0,
            ohlcv_poll_interval_s=0.0,
            ohlcv_retry_attempts=2,
            request_timeout_s=0.2,
        )
        svc._cb_ohlcv = CircuitBreaker(failure_threshold=2, cooldown_s=30.0, name="market_data_ohlcv")
        out = await svc.fetch_ohlcv_df("SOL/USD", timeframe="1m", limit=2)
        self.assertIsNone(out)
        self.assertEqual(ex.calls, 3)
        # One failed request call (after retries) should count as one breaker failure.
        self.assertNotEqual(str(svc._cb_ohlcv.state), "open")


if __name__ == "__main__":
    unittest.main()
