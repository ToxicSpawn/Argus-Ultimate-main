"""Tests for data.sentiment.cryptopanic_feed — CryptoPanicFeed."""

from __future__ import annotations

import asyncio
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from data.sentiment.cryptopanic_feed import CryptoPanicFeed


class TestCryptoPanicFeed(unittest.TestCase):
    """Tests for CryptoPanicFeed."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_default_construction(self) -> None:
        feed = CryptoPanicFeed(api_key="test_key")
        self.assertEqual(feed._api_key, "test_key")
        self.assertEqual(feed._cache_ttl_s, 300.0)
        self.assertEqual(feed._cache, [])

    def test_no_api_key_returns_empty(self) -> None:
        """Returns empty list when no API key is set."""
        feed = CryptoPanicFeed(api_key="")
        result = self._run(feed.get_headlines())
        self.assertEqual(result, [])

    @patch.dict(os.environ, {"CRYPTOPANIC_API_KEY": ""}, clear=False)
    def test_no_env_api_key_returns_empty(self) -> None:
        """Returns empty list when env var is empty."""
        feed = CryptoPanicFeed()
        result = self._run(feed.get_headlines())
        self.assertEqual(result, [])

    def test_get_latest_sentiment_no_cache(self) -> None:
        """Returns 0.0 when no headlines are cached."""
        feed = CryptoPanicFeed(api_key="test")
        self.assertEqual(feed.get_latest_sentiment(), 0.0)

    def test_get_latest_sentiment_bullish(self) -> None:
        """Bullish headlines produce positive sentiment."""
        feed = CryptoPanicFeed(api_key="test")
        feed._cache = [
            {"title": "BTC surges", "source": "test", "symbols": ["BTC"], "published_at": "", "sentiment": "bullish"},
            {"title": "ETH rally", "source": "test", "symbols": ["ETH"], "published_at": "", "sentiment": "bullish"},
        ]
        self.assertGreater(feed.get_latest_sentiment(), 0.0)

    def test_get_latest_sentiment_bearish(self) -> None:
        """Bearish headlines produce negative sentiment."""
        feed = CryptoPanicFeed(api_key="test")
        feed._cache = [
            {"title": "BTC crash", "source": "test", "symbols": ["BTC"], "published_at": "", "sentiment": "bearish"},
            {"title": "ETH dump", "source": "test", "symbols": ["ETH"], "published_at": "", "sentiment": "bearish"},
        ]
        self.assertLess(feed.get_latest_sentiment(), 0.0)

    def test_get_latest_sentiment_mixed(self) -> None:
        """Mixed sentiment averages out."""
        feed = CryptoPanicFeed(api_key="test")
        feed._cache = [
            {"title": "BTC surges", "source": "test", "symbols": ["BTC"], "published_at": "", "sentiment": "bullish"},
            {"title": "ETH crash", "source": "test", "symbols": ["ETH"], "published_at": "", "sentiment": "bearish"},
        ]
        sentiment = feed.get_latest_sentiment()
        self.assertAlmostEqual(sentiment, 0.0)

    def test_cache_ttl_honoured(self) -> None:
        """Second call within TTL returns cached results."""
        feed = CryptoPanicFeed(api_key="test", cache_ttl_s=60)
        feed._cache = [{"title": "cached", "source": "test", "symbols": [], "published_at": "", "sentiment": None}]
        feed._cache_ts = time.time()

        result = self._run(feed.get_headlines())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "cached")

    @patch("data.sentiment.cryptopanic_feed.aiohttp")
    def test_successful_api_call(self, mock_aiohttp) -> None:
        """Parses a successful CryptoPanic API response."""
        fake_api_response = {
            "results": [
                {
                    "title": "Bitcoin hits new ATH",
                    "source": {"title": "CoinDesk"},
                    "currencies": [{"code": "BTC"}],
                    "published_at": "2026-03-20T12:00:00Z",
                    "votes": {"positive": 10, "negative": 2, "important": 1},
                    "kind": "news",
                },
                {
                    "title": "Ethereum upgrade coming soon",
                    "source": {"title": "The Block"},
                    "currencies": [{"code": "ETH"}],
                    "published_at": "2026-03-20T11:00:00Z",
                    "votes": {"positive": 5, "negative": 0, "important": 3},
                    "kind": "news",
                },
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=fake_api_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        feed = CryptoPanicFeed(api_key="test_key")
        result = self._run(feed.get_headlines())

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Bitcoin hits new ATH")
        self.assertEqual(result[0]["source"], "CoinDesk")
        self.assertEqual(result[0]["symbols"], ["BTC"])
        self.assertEqual(result[0]["sentiment"], "bullish")

    @patch("data.sentiment.cryptopanic_feed.aiohttp")
    def test_api_error_returns_empty(self, mock_aiohttp) -> None:
        """API error returns empty list."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        feed = CryptoPanicFeed(api_key="test_key")
        result = self._run(feed.get_headlines())
        self.assertEqual(result, [])

    @patch("data.sentiment.cryptopanic_feed.aiohttp")
    def test_network_exception_returns_empty(self, mock_aiohttp) -> None:
        """Network exception returns empty list."""
        mock_aiohttp.ClientSession = MagicMock(side_effect=Exception("timeout"))
        mock_aiohttp.ClientTimeout = MagicMock()

        feed = CryptoPanicFeed(api_key="test_key")
        result = self._run(feed.get_headlines())
        self.assertEqual(result, [])

    def test_symbol_filtering(self) -> None:
        """get_headlines filters by symbol when requested."""
        feed = CryptoPanicFeed(api_key="test")
        feed._cache = [
            {"title": "BTC news", "source": "test", "symbols": ["BTC"], "published_at": "", "sentiment": None},
            {"title": "ETH news", "source": "test", "symbols": ["ETH"], "published_at": "", "sentiment": None},
            {"title": "General", "source": "test", "symbols": [], "published_at": "", "sentiment": None},
        ]
        feed._cache_ts = time.time()

        result = self._run(feed.get_headlines(symbols=["BTC"]))
        # Should include BTC-tagged and untagged (no symbols) headlines
        titles = [h["title"] for h in result]
        self.assertIn("BTC news", titles)
        self.assertIn("General", titles)
        self.assertNotIn("ETH news", titles)

    def test_parse_results_empty_title_skipped(self) -> None:
        """Items with empty title are skipped in parsing."""
        results = CryptoPanicFeed._parse_results([
            {"title": "", "source": {"title": "test"}, "currencies": [], "published_at": "", "votes": {}},
            {"title": "Good headline", "source": {"title": "test"}, "currencies": [], "published_at": "", "votes": {}},
        ])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Good headline")


if __name__ == "__main__":
    unittest.main()
