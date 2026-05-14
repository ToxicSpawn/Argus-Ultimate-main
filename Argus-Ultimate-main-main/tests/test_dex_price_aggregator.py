"""Tests for data.defi.dex_price_aggregator — DexPriceAggregator."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from data.defi.dex_price_aggregator import (
    ArbOpportunity,
    DexPrice,
    DexPriceAggregator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _make_config(**overrides):
    cfg = MagicMock()
    cfg.dex_aggregator_cache_ttl_s = overrides.get("cache_ttl", 30)
    cfg.dex_aggregator_min_spread_bps = overrides.get("min_spread", 20)
    return cfg


# ---------------------------------------------------------------------------
# DexPrice / ArbOpportunity dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_dex_price_fields(self):
        dp = DexPrice(symbol="ETH/USD", dex_name="uniswap_v3", price=3000.0, liquidity_usd=1e6)
        assert dp.symbol == "ETH/USD"
        assert dp.dex_name == "uniswap_v3"
        assert dp.price == 3000.0
        assert dp.timestamp > 0

    def test_arb_opportunity_fields(self):
        ao = ArbOpportunity(
            symbol="ETH/USD", cex_price=3000.0, dex_price=2950.0,
            spread_bps=166.67, dex_name="uniswap_v3", profitable=True,
        )
        assert ao.profitable is True
        assert ao.spread_bps == 166.67


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------

class TestDexPriceAggregator:
    def test_init_no_rpc(self):
        """Aggregator should init fine with no RPC clients."""
        agg = DexPriceAggregator(config=_make_config())
        assert agg.ethereum_rpc is None
        assert agg.solana_rpc is None

    def test_get_dex_prices_empty_no_rpc(self):
        """With no RPCs, should return empty dict."""
        agg = DexPriceAggregator(config=_make_config())
        result = _run(agg.get_dex_prices())
        assert result == {}

    def test_get_cex_dex_spreads_empty(self):
        """With no DEX prices, should return empty list."""
        agg = DexPriceAggregator(config=_make_config())
        result = _run(agg.get_cex_dex_spreads({"BTC/USD": 60000.0}))
        assert result == []

    def test_cache_works(self):
        """Subsequent calls within TTL should use cache."""
        agg = DexPriceAggregator(config=_make_config(cache_ttl=60))
        # Manually populate cache
        agg._price_cache = {
            "ETH/USD": DexPrice("ETH/USD", "uniswap_v3", 3000.0, 1e6),
        }
        agg._cache_ts = time.time()

        result = _run(agg.get_dex_prices())
        assert "ETH/USD" in result
        assert result["ETH/USD"].price == 3000.0

    def test_spread_calculation(self):
        """Spread should be computed correctly from injected cache."""
        agg = DexPriceAggregator(config=_make_config(min_spread=10))
        agg._price_cache = {
            "BTC/USD": DexPrice("BTC/USD", "uniswap_v3", 59500.0, 1e6),
        }
        agg._cache_ts = time.time()

        result = _run(agg.get_cex_dex_spreads({"BTC/USD": 60000.0}))
        assert len(result) == 1
        opp = result[0]
        assert opp.symbol == "BTC/USD"
        assert opp.cex_price == 60000.0
        assert opp.dex_price == 59500.0
        # spread = |60000 - 59500| / 60000 * 10000 = 83.33 bps
        assert opp.spread_bps == pytest.approx(83.33, rel=0.01)
        # profitable: 83.33 > 50 (30 dex + 20 cex fee) = True
        assert opp.profitable is True

    def test_spread_not_profitable(self):
        """Small spread should not be flagged profitable."""
        agg = DexPriceAggregator(config=_make_config(min_spread=0))
        agg._price_cache = {
            "ETH/USD": DexPrice("ETH/USD", "uniswap_v3", 2999.0, 1e6),
        }
        agg._cache_ts = time.time()

        result = _run(agg.get_cex_dex_spreads({"ETH/USD": 3000.0}))
        assert len(result) == 1
        opp = result[0]
        # spread ~ 3.33 bps — below fee threshold
        assert opp.profitable is False

    def test_spread_filter_min_bps(self):
        """Opportunities below min_spread_bps should be filtered out."""
        agg = DexPriceAggregator(config=_make_config(min_spread=100))
        agg._price_cache = {
            "ETH/USD": DexPrice("ETH/USD", "uniswap_v3", 2999.0, 1e6),
        }
        agg._cache_ts = time.time()

        result = _run(agg.get_cex_dex_spreads({"ETH/USD": 3000.0}))
        # ~3.33 bps < 100 bps threshold
        assert result == []

    def test_multiple_symbols_sorted(self):
        """Opportunities should be sorted by spread descending."""
        agg = DexPriceAggregator(config=_make_config(min_spread=5))
        agg._price_cache = {
            "BTC/USD": DexPrice("BTC/USD", "uniswap_v3", 59000.0, 1e6),
            "ETH/USD": DexPrice("ETH/USD", "uniswap_v3", 2900.0, 1e6),
        }
        agg._cache_ts = time.time()

        result = _run(agg.get_cex_dex_spreads({
            "BTC/USD": 60000.0,
            "ETH/USD": 3000.0,
        }))
        assert len(result) == 2
        # ETH spread ~ 333 bps, BTC spread ~ 166 bps
        assert result[0].symbol == "ETH/USD"
        assert result[1].symbol == "BTC/USD"

    def test_zero_cex_price_skipped(self):
        """Zero CEX price should be skipped."""
        agg = DexPriceAggregator(config=_make_config(min_spread=5))
        agg._price_cache = {
            "ETH/USD": DexPrice("ETH/USD", "uniswap_v3", 3000.0, 1e6),
        }
        agg._cache_ts = time.time()

        result = _run(agg.get_cex_dex_spreads({"ETH/USD": 0.0}))
        assert result == []
