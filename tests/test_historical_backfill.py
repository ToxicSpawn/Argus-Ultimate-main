"""Tests for scripts/backfill_historical.py — historical OHLCV backfill."""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_historical import (
    fetch_ohlcv_chunked,
    backfill,
    DEFAULT_PAIRS,
    DEFAULT_TIMEFRAMES,
    TF_MS,
)


class TestTFMS:
    def test_timeframe_constants(self):
        assert TF_MS["1h"] == 3_600_000
        assert TF_MS["15m"] == 900_000
        assert TF_MS["1d"] == 86_400_000

    def test_all_timeframes_positive(self):
        for tf, ms in TF_MS.items():
            assert ms > 0, f"{tf} should be positive"


class TestDefaultPairs:
    def test_eleven_pairs(self):
        assert len(DEFAULT_PAIRS) == 11

    def test_btc_first(self):
        assert DEFAULT_PAIRS[0] == "BTC/USD"

    def test_all_usd_pairs(self):
        for pair in DEFAULT_PAIRS:
            assert pair.endswith("/USD"), f"{pair} should be /USD pair"


class TestFetchOHLCVChunked:
    def test_empty_exchange_returns_empty_df(self):
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = []
        df = fetch_ohlcv_chunked(exchange, "BTC/USD", "1h", 0, 100_000_000, rate_limit_s=0)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_single_chunk(self):
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        candles = [[now_ms + i * 3_600_000, 100 + i, 105 + i, 95 + i, 102 + i, 1000]
                    for i in range(10)]
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = candles
        df = fetch_ohlcv_chunked(
            exchange, "BTC/USD", "1h",
            since_ms=candles[0][0], until_ms=candles[-1][0] + 3_600_000,
            limit=720, rate_limit_s=0,
        )
        assert len(df) == 10
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_deduplicates(self):
        ts = 1_000_000_000_000
        candles = [[ts, 100, 105, 95, 102, 1000], [ts, 100, 105, 95, 102, 1000]]
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = candles
        df = fetch_ohlcv_chunked(exchange, "BTC/USD", "1h", ts, ts + 7_200_000, rate_limit_s=0)
        assert len(df) == 1

    def test_handles_exception(self):
        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = Exception("API error")
        df = fetch_ohlcv_chunked(exchange, "BTC/USD", "1h", 0, 10_000_000_000, limit=1, rate_limit_s=0)
        assert df.empty


class TestBackfill:
    @patch("scripts.backfill_historical.get_ccxt_exchange")
    def test_skips_unavailable_symbol(self, mock_get_ex):
        exchange = MagicMock()
        exchange.markets = {"ETH/USD": {}}
        exchange.load_markets.return_value = None
        mock_get_ex.return_value = exchange

        with patch("scripts.backfill_historical.HistoricalDataIngester") as mock_ingester_cls:
            mock_ingester = MagicMock()
            mock_ingester.load.return_value = None
            mock_ingester_cls.return_value = mock_ingester

            results = backfill(pairs=["FAKE/USD"], timeframes=["1h"], days=1)
            assert results.get("FAKE/USD", 0) == 0

    @patch("scripts.backfill_historical.get_ccxt_exchange")
    def test_resumes_from_existing(self, mock_get_ex):
        exchange = MagicMock()
        exchange.markets = {"BTC/USD": {}}
        exchange.load_markets.return_value = None

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        exchange.fetch_ohlcv.return_value = []
        mock_get_ex.return_value = exchange

        existing_df = pd.DataFrame({
            "timestamp": [now_ms - 7_200_000, now_ms - 3_600_000],
            "open": [100, 101], "high": [105, 106],
            "low": [95, 96], "close": [102, 103], "volume": [1000, 1100],
        })

        with patch("scripts.backfill_historical.HistoricalDataIngester") as mock_ingester_cls:
            mock_ingester = MagicMock()
            mock_ingester.load.return_value = existing_df
            mock_ingester_cls.return_value = mock_ingester

            results = backfill(pairs=["BTC/USD"], timeframes=["1h"], days=1)
            # Should have existing data count
            assert "BTC/USD_1h" in results

    def test_default_timeframes(self):
        assert "1h" in DEFAULT_TIMEFRAMES
        assert "15m" in DEFAULT_TIMEFRAMES
