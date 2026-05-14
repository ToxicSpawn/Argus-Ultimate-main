"""
Tests for tick store + replay and LLM signal enhancements.

Covers:
  - TickStore: record, read, OHLCV aggregation, flush, cleanup, stats, partitioning
  - TickReplayer: replay callback, speed=0, OHLCV conversion, empty ranges
  - LLMSignalGenerator: market analysis, news sentiment, caching, timeout, parsing

Run with: py -m pytest tests/test_tick_store_llm.py -v
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# TickStore tests
# ---------------------------------------------------------------------------

import pytest
pytest.importorskip("data.tick_store")
from data.tick_store import TickStore


@pytest.fixture
def tick_dir(tmp_path):
    """Provide a temporary directory for tick storage."""
    d = str(tmp_path / "ticks")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture
def store(tick_dir):
    """Create a TickStore with low flush thresholds for testing."""
    return TickStore(base_dir=tick_dir, flush_count=5, flush_interval_s=0.1)


def _make_ts(year=2026, month=3, day=18, hour=10, minute=0, second=0):
    """Helper: create an epoch timestamp for a specific UTC datetime."""
    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    return dt.timestamp()


class TestTickStoreRecordAndRead:
    """Record ticks and read them back."""

    def test_record_and_flush_single_tick(self, store, tick_dir):
        ts = _make_ts(hour=10, minute=0)
        store.record_tick("BTC/USD", ts, 100.0, 100.1, 100.05, 1.5, "kraken")
        store.flush()

        df = store.read_ticks("BTC/USD", ts - 1, ts + 1)
        assert len(df) == 1
        assert df.iloc[0]["last"] == pytest.approx(100.05)
        assert df.iloc[0]["bid"] == pytest.approx(100.0)
        assert df.iloc[0]["ask"] == pytest.approx(100.1)
        assert df.iloc[0]["volume"] == pytest.approx(1.5)
        assert df.iloc[0]["exchange"] == "kraken"

    def test_record_multiple_ticks(self, store):
        base_ts = _make_ts(hour=10)
        for i in range(10):
            store.record_tick(
                "BTC/USD", base_ts + i, 100.0 + i, 100.1 + i,
                100.05 + i, 0.5, "kraken"
            )
        store.flush()

        df = store.read_ticks("BTC/USD", base_ts - 1, base_ts + 20)
        assert len(df) == 10
        # Verify ordering
        assert list(df["timestamp"]) == sorted(df["timestamp"])

    def test_read_empty_symbol(self, store):
        df = store.read_ticks("NONEXIST/USD", 0, time.time())
        assert df.empty

    def test_read_empty_time_range(self, store):
        ts = _make_ts(hour=10)
        store.record_tick("BTC/USD", ts, 100, 100.1, 100.05, 1, "kraken")
        store.flush()

        # Query a different day
        df = store.read_ticks("BTC/USD", _make_ts(day=1), _make_ts(day=2))
        assert df.empty


class TestTickStoreDatePartitioning:
    """Date-partitioned file creation."""

    def test_ticks_split_across_days(self, store, tick_dir):
        ts_day1 = _make_ts(day=15, hour=12)
        ts_day2 = _make_ts(day=16, hour=12)

        store.record_tick("BTC/USD", ts_day1, 100, 100.1, 100.05, 1, "kraken")
        store.record_tick("BTC/USD", ts_day2, 200, 200.1, 200.05, 2, "kraken")
        store.flush()

        # Check two files created
        sym_dir = os.path.join(tick_dir, "BTC-USD")
        files = sorted(os.listdir(sym_dir))
        assert len(files) == 2
        assert "2026-03-15" in files[0]
        assert "2026-03-16" in files[1]

    def test_symbol_name_sanitization(self, store, tick_dir):
        ts = _make_ts()
        store.record_tick("ETH/USD", ts, 3000, 3001, 3000.5, 10, "coinbase")
        store.flush()

        sym_dir = os.path.join(tick_dir, "ETH-USD")
        assert os.path.isdir(sym_dir)


class TestTickStoreMultipleSymbols:
    """Multiple symbol support."""

    def test_separate_storage_per_symbol(self, store, tick_dir):
        ts = _make_ts()
        store.record_tick("BTC/USD", ts, 100, 100.1, 100.05, 1, "kraken")
        store.record_tick("ETH/USD", ts, 3000, 3001, 3000.5, 10, "kraken")
        store.flush()

        btc = store.read_ticks("BTC/USD", ts - 1, ts + 1)
        eth = store.read_ticks("ETH/USD", ts - 1, ts + 1)
        assert len(btc) == 1
        assert len(eth) == 1
        assert btc.iloc[0]["last"] == pytest.approx(100.05)
        assert eth.iloc[0]["last"] == pytest.approx(3000.5)


class TestTickStoreFlush:
    """Flush on buffer full."""

    def test_auto_flush_on_count(self, tick_dir):
        store = TickStore(base_dir=tick_dir, flush_count=3, flush_interval_s=3600)
        ts = _make_ts()
        for i in range(4):
            store.record_tick("BTC/USD", ts + i, 100, 100.1, 100.05, 1, "kraken")
        # After 4 ticks with flush_count=3, flush should have occurred
        # But the 4th tick triggers flush of all buffered (3+)
        # Force flush for remaining
        store.flush()
        df = store.read_ticks("BTC/USD", ts - 1, ts + 10)
        assert len(df) == 4

    def test_flush_empty_is_noop(self, store):
        # Should not raise
        store.flush()
        stats = store.get_stats()
        assert stats["flushed"] == 0


class TestTickStoreOHLCV:
    """OHLCV aggregation from ticks."""

    def test_ohlcv_1min_bars(self, store):
        base_ts = _make_ts(hour=10, minute=0, second=0)
        # 5 ticks in minute 0, 5 ticks in minute 1
        for i in range(5):
            store.record_tick("BTC/USD", base_ts + i * 10, 100 + i, 101 + i, 100.5 + i, 1.0, "kraken")
        for i in range(5):
            ts = base_ts + 60 + i * 10
            store.record_tick("BTC/USD", ts, 200 + i, 201 + i, 200.5 + i, 2.0, "kraken")
        store.flush()

        ohlcv = store.get_ohlcv("BTC/USD", base_ts - 1, base_ts + 200, interval="1m")
        assert len(ohlcv) >= 2
        # First bar open should be ~100.5
        assert ohlcv.iloc[0]["open"] == pytest.approx(100.5)
        # First bar high should be ~104.5
        assert ohlcv.iloc[0]["high"] == pytest.approx(104.5)
        # First bar low should be ~100.5
        assert ohlcv.iloc[0]["low"] == pytest.approx(100.5)

    def test_ohlcv_correct_volume(self, store):
        base_ts = _make_ts(hour=10)
        for i in range(3):
            store.record_tick("BTC/USD", base_ts + i, 100, 100.1, 100.05, 5.0, "kraken")
        store.flush()

        ohlcv = store.get_ohlcv("BTC/USD", base_ts - 1, base_ts + 60, interval="1m")
        assert len(ohlcv) >= 1
        assert ohlcv.iloc[0]["volume"] == pytest.approx(15.0)

    def test_ohlcv_empty_range(self, store):
        ohlcv = store.get_ohlcv("NONEXIST/USD", 0, 1000, interval="1m")
        assert ohlcv.empty


class TestTickStoreCleanup:
    """Cleanup old files."""

    def test_cleanup_removes_old_files(self, store, tick_dir):
        # Create a tick dated 200 days ago
        old_ts = _make_ts(year=2025, month=8, day=1)
        store.record_tick("BTC/USD", old_ts, 50, 50.1, 50.05, 1, "kraken")
        # And a recent one
        recent_ts = _make_ts()
        store.record_tick("BTC/USD", recent_ts, 100, 100.1, 100.05, 1, "kraken")
        store.flush()

        deleted = store.cleanup(max_days=90)
        assert deleted == 1

        # Recent tick should survive
        df = store.read_ticks("BTC/USD", recent_ts - 1, recent_ts + 1)
        assert len(df) == 1

    def test_cleanup_empty_store(self, store):
        deleted = store.cleanup(max_days=90)
        assert deleted == 0


class TestTickStoreStats:
    """Stats reporting."""

    def test_stats_after_recording(self, store):
        ts = _make_ts()
        for i in range(5):
            store.record_tick("BTC/USD", ts + i, 100, 100.1, 100.05, 1, "kraken")
        store.flush()

        stats = store.get_stats()
        assert "BTC-USD" in stats["symbols"]
        assert stats["total_ticks"] == 5
        assert stats["disk_usage_mb"] >= 0
        assert stats["date_range"] is not None
        assert stats["flushed"] == 5

    def test_stats_empty_store(self, store):
        stats = store.get_stats()
        assert stats["symbols"] == []
        assert stats["total_ticks"] == 0
        assert stats["date_range"] is None


# ---------------------------------------------------------------------------
# TickReplayer tests
# ---------------------------------------------------------------------------

from data.tick_replay import TickReplayer


@pytest.fixture
def replayer(store):
    return TickReplayer(store)


class TestTickReplayer:
    """Replay callback tests."""

    def test_replay_callback_called(self, store, replayer):
        ts = _make_ts(hour=10)
        for i in range(5):
            store.record_tick("BTC/USD", ts + i, 100, 100.1, 100.05 + i, 1, "kraken")
        store.flush()

        received = []

        async def run():
            await replayer.replay(
                "BTC/USD", ts - 1, ts + 10, speed=0,
                callback=lambda t: received.append(t)
            )

        asyncio.run(run())
        assert len(received) == 5
        assert received[0]["last"] == pytest.approx(100.05)
        assert received[4]["last"] == pytest.approx(104.05)

    def test_replay_speed_zero_instant(self, store, replayer):
        ts = _make_ts(hour=10)
        for i in range(20):
            store.record_tick("BTC/USD", ts + i * 60, 100, 100.1, 100.05, 1, "kraken")
        store.flush()

        t0 = time.monotonic()

        async def run():
            return await replayer.replay("BTC/USD", ts - 1, ts + 2000, speed=0)

        count = asyncio.run(run())
        elapsed = time.monotonic() - t0

        assert count == 20
        assert elapsed < 2.0  # Should be near-instant

    def test_replay_empty_range(self, store, replayer):
        async def run():
            return await replayer.replay("BTC/USD", 0, 1, speed=0)

        count = asyncio.run(run())
        assert count == 0

    def test_replay_returns_count(self, store, replayer):
        ts = _make_ts(hour=10)
        for i in range(3):
            store.record_tick("ETH/USD", ts + i, 3000, 3001, 3000.5, 10, "kraken")
        store.flush()

        async def run():
            return await replayer.replay("ETH/USD", ts - 1, ts + 10, speed=0)

        count = asyncio.run(run())
        assert count == 3

    def test_replay_to_ohlcv(self, store, replayer):
        ts = _make_ts(hour=10)
        for i in range(10):
            store.record_tick("BTC/USD", ts + i * 5, 100, 100.1, 100.05 + i, 1, "kraken")
        store.flush()

        ohlcv = replayer.replay_to_ohlcv("BTC/USD", ts - 1, ts + 100, interval="1m")
        assert not ohlcv.empty
        assert "open" in ohlcv.columns
        assert "close" in ohlcv.columns

    def test_replay_to_ohlcv_empty(self, store, replayer):
        ohlcv = replayer.replay_to_ohlcv("NONEXIST/USD", 0, 100, interval="1m")
        assert ohlcv.empty


# ---------------------------------------------------------------------------
# LLM Signal tests
# ---------------------------------------------------------------------------

from ml.llm_signal import (
    LLMSignalGenerator,
    LLMSignal,
    DIRECTION_BULLISH,
    DIRECTION_BEARISH,
    DIRECTION_NEUTRAL,
)


@pytest.fixture
def llm_gen():
    """Create an LLMSignalGenerator with Ollama provider."""
    return LLMSignalGenerator(
        provider="ollama",
        model="test-model",
        timeout=5.0,
        inference_timeout=10.0,
    )


class TestLLMParseResponse:
    """Test response parsing for various LLM outputs."""

    def test_parse_bullish(self, llm_gen):
        text = "Signal: BULLISH\nConfidence: high confidence\nReasoning: Strong momentum."
        sig = llm_gen._parse_response(text)
        assert sig.direction == DIRECTION_BULLISH
        assert sig.confidence == 0.8
        assert "Strong momentum" in sig.reasoning

    def test_parse_bearish(self, llm_gen):
        text = "Signal: BEARISH\nConfidence: low confidence\nReasoning: Breakdown below support."
        sig = llm_gen._parse_response(text)
        assert sig.direction == DIRECTION_BEARISH
        assert sig.confidence == 0.35

    def test_parse_neutral(self, llm_gen):
        text = "Signal: NEUTRAL\nConfidence: moderate confidence\nReasoning: Sideways market."
        sig = llm_gen._parse_response(text)
        assert sig.direction == DIRECTION_NEUTRAL
        assert sig.confidence == 0.6

    def test_parse_empty_response(self, llm_gen):
        sig = llm_gen._parse_response("")
        assert sig.direction == DIRECTION_NEUTRAL
        assert sig.confidence == 0.0
        assert "No response" in sig.reasoning

    def test_parse_invalid_response(self, llm_gen):
        sig = llm_gen._parse_response("I don't know what to say")
        assert sig.direction == DIRECTION_NEUTRAL
        assert sig.confidence == 0.5  # default moderate

    def test_confidence_number_extraction(self, llm_gen):
        text = "BULLISH - very high confidence due to breakout"
        sig = llm_gen._parse_response(text)
        assert sig.confidence == 0.9

    def test_parse_uncertain(self, llm_gen):
        text = "NEUTRAL - uncertain market conditions"
        sig = llm_gen._parse_response(text)
        assert sig.confidence == 0.25


class TestLLMMarketAnalysis:
    """Test generate_market_analysis with mocked HTTP."""

    def test_market_analysis_bullish(self, llm_gen):
        mock_response = (
            "Signal: BULLISH\n"
            "Confidence: high confidence\n"
            "Reasoning: Price is above all moving averages with increasing volume."
        )
        with patch.object(llm_gen, "_query_ollama", return_value=mock_response):
            result = llm_gen.generate_market_analysis(
                symbol="BTC/USD",
                ohlcv_data=[100, 101, 102, 103, 104],
                regime="TRENDING",
                indicators={"rsi": 65.0, "macd": 0.5},
            )
        assert result["direction"] == DIRECTION_BULLISH
        assert result["confidence"] == 0.8
        assert "model_used" in result
        assert "reasoning" in result

    def test_market_analysis_bearish(self, llm_gen):
        mock_response = (
            "Signal: BEARISH\n"
            "Confidence: moderate confidence\n"
            "Reasoning: Breaking below key support."
        )
        with patch.object(llm_gen, "_query_ollama", return_value=mock_response):
            result = llm_gen.generate_market_analysis(
                symbol="ETH/USD",
                ohlcv_data=[200, 195, 190, 185],
                regime="BEARISH",
            )
        assert result["direction"] == DIRECTION_BEARISH
        assert result["confidence"] == 0.6

    def test_market_analysis_error_returns_neutral(self, llm_gen):
        with patch.object(llm_gen, "_query_ollama", side_effect=Exception("Connection refused")):
            result = llm_gen.generate_market_analysis(
                symbol="BTC/USD",
                ohlcv_data=[100, 101],
                regime="UNKNOWN",
            )
        assert result["direction"] == DIRECTION_NEUTRAL
        assert result["confidence"] == 0.0

    def test_market_analysis_caching(self, llm_gen):
        mock_response = "Signal: BULLISH\nConfidence: high confidence\nReasoning: Test."
        call_count = 0

        def counting_query(prompt):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch.object(llm_gen, "_query_ollama", side_effect=counting_query):
            r1 = llm_gen.generate_market_analysis("BTC/USD", [100, 101], "TRENDING")
            r2 = llm_gen.generate_market_analysis("BTC/USD", [100, 101], "TRENDING")

        # Second call should hit cache
        assert call_count == 1
        assert r1 == r2

    def test_market_analysis_cache_expires(self, llm_gen):
        mock_response = "Signal: NEUTRAL\nConfidence: moderate\nReasoning: Sideways."

        with patch.object(llm_gen, "_query_ollama", return_value=mock_response):
            llm_gen.generate_market_analysis("BTC/USD", [100], "UNKNOWN")

        # Expire cache manually
        llm_gen._analysis_cache["BTC/USD"] = (
            time.time() - 400,  # expired
            llm_gen._analysis_cache["BTC/USD"][1],
        )

        call_count = 0
        def counting_query(prompt):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch.object(llm_gen, "_query_ollama", side_effect=counting_query):
            llm_gen.generate_market_analysis("BTC/USD", [100], "UNKNOWN")

        assert call_count == 1


class TestLLMNewsSentiment:
    """Test generate_news_sentiment."""

    def test_sentiment_bullish(self, llm_gen):
        mock_response = (
            "Signal: BULLISH\n"
            "Confidence: high confidence\n"
            "Reasoning: Institutional buying, ETF approval, positive regulation."
        )
        with patch.object(llm_gen, "_query_ollama", return_value=mock_response):
            result = llm_gen.generate_news_sentiment([
                "Bitcoin ETF sees record inflows",
                "Major bank launches crypto custody",
            ])
        assert result["sentiment"] == DIRECTION_BULLISH
        assert result["confidence"] == 0.8
        assert len(result["key_factors"]) > 0

    def test_sentiment_empty_headlines(self, llm_gen):
        result = llm_gen.generate_news_sentiment([])
        assert result["sentiment"] == DIRECTION_NEUTRAL
        assert result["confidence"] == 0.0
        assert result["key_factors"] == []

    def test_sentiment_caching(self, llm_gen):
        mock_response = "Signal: BEARISH\nConfidence: moderate\nReasoning: FUD spreading."
        call_count = 0

        def counting_query(prompt):
            nonlocal call_count
            call_count += 1
            return mock_response

        headlines = ["Market crash fears grow", "Regulatory crackdown"]

        with patch.object(llm_gen, "_query_ollama", side_effect=counting_query):
            r1 = llm_gen.generate_news_sentiment(headlines)
            r2 = llm_gen.generate_news_sentiment(headlines)

        assert call_count == 1
        assert r1 == r2


class TestLLMTimeout:
    """Timeout returns neutral."""

    def test_generate_signal_timeout_returns_neutral(self, llm_gen):
        # Set very short timeout
        llm_gen.inference_timeout = 0.01

        def slow_query(prompt):
            time.sleep(5)
            return "BULLISH"

        with patch.object(llm_gen, "_query_ollama", side_effect=slow_query):
            async def run():
                return await llm_gen.generate_signal(
                    symbol="BTC/USD",
                    regime="TRENDING",
                    price_data=[100, 101, 102],
                )

            sig = asyncio.run(run())
        assert sig.direction == DIRECTION_NEUTRAL
        assert sig.confidence == 0.0


class TestLLMPromptConstruction:
    """Test prompt construction methods."""

    def test_analysis_prompt_contains_indicators(self, llm_gen):
        prompt = llm_gen._build_analysis_prompt(
            "BTC/USD", [100, 101, 102], "TRENDING", {"rsi": 65.0, "macd": 0.5}
        )
        assert "RSI" in prompt
        assert "MACD" in prompt
        assert "BTC/USD" in prompt
        assert "TRENDING" in prompt

    def test_analysis_prompt_empty_prices(self, llm_gen):
        prompt = llm_gen._build_analysis_prompt("BTC/USD", [], "UNKNOWN", {})
        assert "BTC/USD" in prompt

    def test_sentiment_prompt_contains_headlines(self, llm_gen):
        prompt = llm_gen._build_sentiment_prompt(["Bitcoin surges", "ETH breaks out"])
        assert "Bitcoin surges" in prompt
        assert "ETH breaks out" in prompt

    def test_original_prompt_format(self, llm_gen):
        prompt = llm_gen._build_prompt("BTC/USD", "TRENDING", [100, 101], 0.001, None)
        assert "BULLISH" in prompt
        assert "BEARISH" in prompt
        assert "NEUTRAL" in prompt
        assert "BTC/USD" in prompt


class TestLLMIsAvailable:
    """Test availability check."""

    def test_ollama_unavailable(self, llm_gen):
        # Should fail gracefully when Ollama is not running
        result = llm_gen.is_available()
        # We don't assert True/False — just that it doesn't crash
        assert isinstance(result, bool)

    def test_openai_needs_key(self):
        gen = LLMSignalGenerator(provider="openai", model="gpt-4o-mini")
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            assert gen.is_available() is False


class TestLLMSignalDataclass:
    """Test LLMSignal dataclass."""

    def test_as_numeric_bullish(self):
        sig = LLMSignal("BULLISH", 0.8, "test", "model", 100.0)
        assert sig.as_numeric == 1.0

    def test_as_numeric_bearish(self):
        sig = LLMSignal("BEARISH", 0.6, "test", "model", 50.0)
        assert sig.as_numeric == -1.0

    def test_as_numeric_neutral(self):
        sig = LLMSignal("NEUTRAL", 0.5, "test", "model", 25.0)
        assert sig.as_numeric == 0.0
