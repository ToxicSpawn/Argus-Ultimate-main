"""Tests for historical OHLCV data loader."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from core.historical_data_loader import (
    _cache_path, _load_from_cache, _save_to_cache,
    _timeframe_to_ms, load_all_from_cache,
)


class TestTimeframeToMs(unittest.TestCase):
    def test_hourly(self):
        self.assertEqual(_timeframe_to_ms("1h"), 3_600_000)

    def test_daily(self):
        self.assertEqual(_timeframe_to_ms("1d"), 86_400_000)

    def test_minute(self):
        self.assertEqual(_timeframe_to_ms("1m"), 60_000)

    def test_unknown(self):
        self.assertEqual(_timeframe_to_ms("garbage"), 0)


class TestDiskCache(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        # Monkey-patch cache dir
        import core.historical_data_loader as hdl
        self._orig_dir = hdl._CACHE_DIR
        hdl._CACHE_DIR = Path(self._tmpdir)

    def tearDown(self):
        import core.historical_data_loader as hdl
        hdl._CACHE_DIR = self._orig_dir
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        candles = [
            [1000000 + i * 3600000, 100 + i, 105 + i, 95 + i, 102 + i, 1000 + i]
            for i in range(100)
        ]
        _save_to_cache("BTC/USD", "1h", candles)
        result = _load_from_cache("BTC/USD", "1h")
        self.assertIsNotNone(result)
        self.assertEqual(len(result["close"]), 100)
        self.assertAlmostEqual(result["close"][0], 102.0)
        self.assertAlmostEqual(result["high"][0], 105.0)
        self.assertAlmostEqual(result["volume"][0], 1000.0)

    def test_cache_miss(self):
        self.assertIsNone(_load_from_cache("NONEXISTENT/USD", "1h"))

    def test_stale_cache(self):
        import core.historical_data_loader as hdl
        candles = [[1, 100, 105, 95, 102, 1000]]
        _save_to_cache("BTC/USD", "1h", candles)
        # Make cache appear old
        path = _cache_path("BTC/USD", "1h")
        data = json.loads(path.read_text())
        data["cached_at"] = time.time() - 25 * 3600  # 25h ago
        path.write_text(json.dumps(data))
        self.assertIsNone(_load_from_cache("BTC/USD", "1h"))

    def test_load_all_from_cache(self):
        candles = [
            [1000000 + i * 3600000, 100, 105, 95, 102, 1000]
            for i in range(60)
        ]
        _save_to_cache("BTC/USD", "1h", candles)
        _save_to_cache("ETH/USD", "1h", candles)
        result = load_all_from_cache(["BTC/USD", "ETH/USD", "SOL/USD"], "1h")
        self.assertEqual(len(result), 2)  # SOL not cached
        self.assertIn("BTC/USD", result)
        self.assertIn("ETH/USD", result)

    def test_load_all_empty(self):
        result = load_all_from_cache(["BTC/USD"], "1h")
        self.assertEqual(len(result), 0)


class TestCachePathSafety(unittest.TestCase):
    def test_slash_replaced(self):
        path = _cache_path("BTC/USD", "1h")
        self.assertNotIn("/", path.name)
        self.assertIn("BTC_USD", path.name)

    def test_timeframe_in_name(self):
        path = _cache_path("BTC/USD", "4h")
        self.assertIn("4h", path.name)


class TestOHLCVArrayFormat(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import core.historical_data_loader as hdl
        self._orig_dir = hdl._CACHE_DIR
        hdl._CACHE_DIR = Path(self._tmpdir)

    def tearDown(self):
        import core.historical_data_loader as hdl
        hdl._CACHE_DIR = self._orig_dir
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_numpy_arrays_returned(self):
        candles = [
            [1000 + i, 100 + i * 0.1, 105 + i * 0.1, 95 + i * 0.1, 102 + i * 0.1, 5000]
            for i in range(100)
        ]
        _save_to_cache("BTC/USD", "1h", candles)
        result = _load_from_cache("BTC/USD", "1h")
        for key in ("timestamp", "open", "high", "low", "close", "volume"):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], np.ndarray)
            self.assertEqual(len(result[key]), 100)

    def test_high_always_gte_low(self):
        candles = [
            [1000 + i, 100, 110, 90, 105, 5000]
            for i in range(50)
        ]
        _save_to_cache("BTC/USD", "1h", candles)
        result = _load_from_cache("BTC/USD", "1h")
        self.assertTrue(np.all(result["high"] >= result["low"]))


if __name__ == "__main__":
    unittest.main()
