"""Tests for data.onchain.chain_metrics — ChainMetricsProvider."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from data.onchain.chain_metrics import ChainMetricSnapshot, ChainMetricsProvider


class TestChainMetricSnapshot(unittest.TestCase):
    """Tests for ChainMetricSnapshot dataclass."""

    def test_defaults(self) -> None:
        snap = ChainMetricSnapshot()
        self.assertEqual(snap.mvrv_zscore, 0.0)
        self.assertEqual(snap.sopr, 1.0)
        self.assertEqual(snap.net_exchange_flow_btc, 0.0)
        self.assertEqual(snap.signal_bias, 0.0)
        self.assertIsInstance(snap.timestamp, float)

    def test_custom_values(self) -> None:
        snap = ChainMetricSnapshot(
            mvrv_zscore=2.5,
            sopr=0.95,
            net_exchange_flow_btc=-500.0,
            signal_bias=0.35,
            timestamp=1000.0,
        )
        self.assertAlmostEqual(snap.mvrv_zscore, 2.5)
        self.assertAlmostEqual(snap.sopr, 0.95)
        self.assertAlmostEqual(snap.net_exchange_flow_btc, -500.0)
        self.assertAlmostEqual(snap.signal_bias, 0.35)
        self.assertEqual(snap.timestamp, 1000.0)

    def test_signal_bias_range(self) -> None:
        """Signal bias must be in [-1, +1]."""
        snap = ChainMetricSnapshot(signal_bias=0.7)
        self.assertGreaterEqual(snap.signal_bias, -1.0)
        self.assertLessEqual(snap.signal_bias, 1.0)


class TestChainMetricsProvider(unittest.TestCase):
    """Tests for ChainMetricsProvider."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_default_construction(self) -> None:
        provider = ChainMetricsProvider()
        self.assertIsNone(provider._cached_snapshot)
        self.assertEqual(provider._cache_ttl_s, 900.0)

    def test_get_signal_bias_no_data(self) -> None:
        """get_signal_bias returns 0.0 when no data has been fetched."""
        provider = ChainMetricsProvider()
        self.assertEqual(provider.get_signal_bias(), 0.0)

    def test_get_signal_bias_after_fetch(self) -> None:
        """get_signal_bias returns cached value after a fetch."""
        provider = ChainMetricsProvider()
        provider._cached_snapshot = ChainMetricSnapshot(signal_bias=0.42)
        self.assertAlmostEqual(provider.get_signal_bias(), 0.42)

    @patch("data.onchain.chain_metrics.aiohttp")
    def test_get_metrics_api_success(self, mock_aiohttp) -> None:
        """Successful API response produces a valid snapshot."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "market_price_usd": 65000,
            "hash_rate": 6e17,
            "difficulty": 8e13,
            "n_tx": 400000,
            "miners_revenue_usd": 50000000,
            "miners_revenue_btc": 800,
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        provider = ChainMetricsProvider()
        snap = self._run(provider.get_metrics())

        self.assertIsInstance(snap, ChainMetricSnapshot)
        self.assertGreaterEqual(snap.signal_bias, -1.0)
        self.assertLessEqual(snap.signal_bias, 1.0)
        self.assertIsInstance(snap.mvrv_zscore, float)
        self.assertIsInstance(snap.sopr, float)

    @patch("data.onchain.chain_metrics.aiohttp")
    def test_get_metrics_api_failure_returns_neutral(self, mock_aiohttp) -> None:
        """Failed API call returns neutral snapshot."""
        mock_aiohttp.ClientSession = MagicMock(side_effect=Exception("network error"))
        mock_aiohttp.ClientTimeout = MagicMock()

        provider = ChainMetricsProvider()
        snap = self._run(provider.get_metrics())

        self.assertIsInstance(snap, ChainMetricSnapshot)
        self.assertEqual(snap.signal_bias, 0.0)
        self.assertEqual(snap.mvrv_zscore, 0.0)

    def test_cache_ttl_honoured(self) -> None:
        """Second call within TTL returns cached result."""
        provider = ChainMetricsProvider(cache_ttl_s=60)
        cached = ChainMetricSnapshot(signal_bias=0.5)
        provider._cached_snapshot = cached
        provider._cache_ts = time.time()

        snap = self._run(provider.get_metrics())
        self.assertIs(snap, cached)

    def test_cache_expired(self) -> None:
        """Expired cache triggers fresh fetch (which falls back to neutral)."""
        provider = ChainMetricsProvider(cache_ttl_s=1)
        cached = ChainMetricSnapshot(signal_bias=0.5)
        provider._cached_snapshot = cached
        provider._cache_ts = time.time() - 100  # expired

        # Patch _fetch_metrics to avoid real HTTP
        async def fake_fetch():
            return ChainMetricSnapshot(signal_bias=-0.2)

        provider._fetch_metrics = fake_fetch  # type: ignore[assignment]

        snap = self._run(provider.get_metrics())
        self.assertAlmostEqual(snap.signal_bias, -0.2)

    def test_signal_bias_mvrv_extreme_high(self) -> None:
        """MVRV > extreme_high produces negative (bearish) bias."""
        provider = ChainMetricsProvider()
        bias = provider._compute_signal_bias(mvrv_zscore=5.0, sopr=1.0, net_exchange_flow_btc=0.0)
        self.assertLess(bias, 0.0)

    def test_signal_bias_mvrv_extreme_low(self) -> None:
        """MVRV < extreme_low produces positive (bullish) bias."""
        provider = ChainMetricsProvider()
        bias = provider._compute_signal_bias(mvrv_zscore=0.5, sopr=1.0, net_exchange_flow_btc=0.0)
        self.assertGreater(bias, 0.0)

    def test_signal_bias_sopr_capitulation(self) -> None:
        """SOPR < 1.0 (capitulation) produces bullish bias."""
        provider = ChainMetricsProvider()
        bias = provider._compute_signal_bias(mvrv_zscore=2.0, sopr=0.85, net_exchange_flow_btc=0.0)
        self.assertGreater(bias, 0.0)

    def test_signal_bias_clamped(self) -> None:
        """Signal bias is always clamped to [-1, +1]."""
        provider = ChainMetricsProvider()
        # Extreme bullish conditions
        bias = provider._compute_signal_bias(mvrv_zscore=0.0, sopr=0.7, net_exchange_flow_btc=-50000.0)
        self.assertGreaterEqual(bias, -1.0)
        self.assertLessEqual(bias, 1.0)

        # Extreme bearish conditions
        bias = provider._compute_signal_bias(mvrv_zscore=7.0, sopr=1.2, net_exchange_flow_btc=50000.0)
        self.assertGreaterEqual(bias, -1.0)
        self.assertLessEqual(bias, 1.0)


if __name__ == "__main__":
    unittest.main()
