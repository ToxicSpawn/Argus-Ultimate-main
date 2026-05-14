"""
test_tier1_hft.py — Tests for Tier 1 HFT components.

Covers:
  - core/ws_l2_book_feed.py   : WSL2BookFeed / OrderBook
  - infra/socket_tuner.py     : SocketTuningReport / apply_global_tuning
  - execution/cancel_replace.py : CancelReplaceManager / TokenBucket (CR)
  - execution/quote_throttle.py : QuoteThrottleFilter / TokenBucket (QT)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from core.ws_l2_book_feed import OrderBook, WSL2BookFeed
from infra.socket_tuner import (
    SocketTuningReport,
    apply_global_tuning,
    get_socket_stats,
    tune_socket,
)
from execution.cancel_replace import CancelReplaceManager, TokenBucket as CRTokenBucket
from execution.quote_throttle import QuoteThrottleFilter, TokenBucket as QTTokenBucket


# ===========================================================================
# Helper factories
# ===========================================================================

def _make_feed(exchange: str = "kraken", symbols=None) -> WSL2BookFeed:
    if symbols is None:
        symbols = ["BTC/USD"]
    return WSL2BookFeed(symbols=symbols, exchange=exchange)


def _make_kraken_snapshot(symbol: str = "BTC/USD") -> str:
    """Return a JSON string mimicking a Kraken book snapshot message."""
    return json.dumps([
        42,
        {
            "bs": [
                ["29900.00", "1.5", "1700000000.000000"],
                ["29899.00", "2.0", "1700000000.000001"],
            ],
            "as": [
                ["29901.00", "1.2", "1700000000.000000"],
                ["29902.00", "0.8", "1700000000.000001"],
            ],
        },
        "book-10",
        symbol,
    ])


def _make_kraken_delta(symbol: str = "BTC/USD") -> str:
    """Return a JSON string mimicking a Kraken book delta message."""
    return json.dumps([
        42,
        {
            "b": [["29905.00", "3.0", "1700000001.000000"]],
            "a": [["29901.00", "0.0", "1700000001.000000"]],  # remove ask at 29901
        },
        "book-10",
        symbol,
    ])


# ===========================================================================
# WSL2BookFeed tests
# ===========================================================================

class TestWSL2BookFeedSnapshot:
    """test_ws_l2_book_feed_snapshot — snapshot populates bids and asks."""

    def test_snapshot_populates_book(self):
        feed = _make_feed("kraken", ["BTC/USD"])
        raw = _make_kraken_snapshot("BTC/USD")
        feed._parse_kraken(raw)

        ob = feed.books["BTC/USD"]
        assert ob.is_snapshot is True
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2
        assert 29900.00 in ob.bids
        assert 29901.00 in ob.asks

    def test_snapshot_correct_sizes(self):
        feed = _make_feed("kraken", ["BTC/USD"])
        feed._parse_kraken(_make_kraken_snapshot("BTC/USD"))
        ob = feed.books["BTC/USD"]
        assert ob.bids[29900.00] == pytest.approx(1.5)
        assert ob.asks[29901.00] == pytest.approx(1.2)


class TestWSL2BookFeedDeltaUpdate:
    """test_ws_l2_book_feed_delta_update — incremental update modifies book."""

    def test_delta_adds_bid(self):
        feed = _make_feed("kraken", ["BTC/USD"])
        # Apply snapshot first
        feed._parse_kraken(_make_kraken_snapshot("BTC/USD"))
        # Apply delta
        feed._parse_kraken(_make_kraken_delta("BTC/USD"))

        ob = feed.books["BTC/USD"]
        # New bid at 29905 should be present
        assert 29905.00 in ob.bids
        assert ob.bids[29905.00] == pytest.approx(3.0)

    def test_delta_removes_ask(self):
        feed = _make_feed("kraken", ["BTC/USD"])
        feed._parse_kraken(_make_kraken_snapshot("BTC/USD"))
        feed._parse_kraken(_make_kraken_delta("BTC/USD"))

        ob = feed.books["BTC/USD"]
        # Ask at 29901 was removed (size=0)
        assert 29901.00 not in ob.asks

    def test_delta_before_snapshot_is_ignored(self):
        """Delta arriving before snapshot should not corrupt the book."""
        feed = _make_feed("kraken", ["BTC/USD"])
        feed._parse_kraken(_make_kraken_delta("BTC/USD"))
        ob = feed.books["BTC/USD"]
        assert ob.is_snapshot is False
        assert len(ob.bids) == 0


class TestWSL2BookFeedBestBidAsk:
    """test_ws_l2_book_feed_best_bid_ask — best bid < best ask."""

    def test_best_bid_below_best_ask(self):
        feed = _make_feed("kraken", ["BTC/USD"])
        feed._parse_kraken(_make_kraken_snapshot("BTC/USD"))

        bb = feed.get_best_bid("BTC/USD")
        ba = feed.get_best_ask("BTC/USD")

        assert bb is not None
        assert ba is not None
        assert bb < ba

    def test_best_bid_is_highest_bid(self):
        feed = _make_feed("kraken", ["BTC/USD"])
        feed._parse_kraken(_make_kraken_snapshot("BTC/USD"))
        assert feed.get_best_bid("BTC/USD") == pytest.approx(29900.00)

    def test_best_ask_is_lowest_ask(self):
        feed = _make_feed("kraken", ["BTC/USD"])
        feed._parse_kraken(_make_kraken_snapshot("BTC/USD"))
        assert feed.get_best_ask("BTC/USD") == pytest.approx(29901.00)

    def test_mid_is_average(self):
        feed = _make_feed("kraken", ["BTC/USD"])
        feed._parse_kraken(_make_kraken_snapshot("BTC/USD"))
        mid = feed.get_mid("BTC/USD")
        assert mid == pytest.approx((29900.0 + 29901.0) / 2.0)

    def test_get_book_snapshot_structure(self):
        feed = _make_feed("kraken", ["BTC/USD"])
        feed._parse_kraken(_make_kraken_snapshot("BTC/USD"))
        snap = feed.get_book_snapshot("BTC/USD", levels=2)
        assert "bids" in snap and "asks" in snap
        assert len(snap["bids"]) <= 2
        assert len(snap["asks"]) <= 2
        # Bids sorted desc
        if len(snap["bids"]) > 1:
            assert snap["bids"][0][0] > snap["bids"][1][0]
        # Asks sorted asc
        if len(snap["asks"]) > 1:
            assert snap["asks"][0][0] < snap["asks"][1][0]


# ===========================================================================
# SocketTuner tests
# ===========================================================================

class TestSocketTunerReport:
    """test_socket_tuner_report — SocketTuningReport fields are present."""

    def test_fields_present(self):
        report = SocketTuningReport()
        assert hasattr(report, "tcp_nodelay")
        assert hasattr(report, "so_busy_poll")
        assert hasattr(report, "sndbuf")
        assert hasattr(report, "rcvbuf")
        assert hasattr(report, "ip_tos")
        assert hasattr(report, "platform")

    def test_apply_global_tuning_returns_report(self):
        report = apply_global_tuning()
        assert isinstance(report, SocketTuningReport)
        assert isinstance(report.platform, str)
        assert len(report.platform) > 0

    def test_apply_global_tuning_tcp_nodelay(self):
        report = apply_global_tuning()
        # On any supported platform TCP_NODELAY should be set
        assert report.tcp_nodelay is True

    def test_as_dict(self):
        report = SocketTuningReport(tcp_nodelay=True, sndbuf=65536, rcvbuf=131072, ip_tos=0x10)
        d = report.as_dict()
        assert d["tcp_nodelay"] is True
        assert d["sndbuf"] == 65536
        assert d["rcvbuf"] == 131072
        assert "platform" in d


# ===========================================================================
# CancelReplace tests
# ===========================================================================

def _make_mock_client(supports_native: bool = True) -> MagicMock:
    """Return an async mock exchange client."""
    client = MagicMock()

    async def _amend(order_id, symbol, price, size):
        return {"id": f"amended_{order_id}", "price": price, "status": "amended"}

    async def _cancel(order_id, symbol):
        return {"status": "cancelled"}

    async def _create(symbol, side, price, size, order_type):
        return {"id": "new_order_123", "price": price, "size": size, "status": "created"}

    if supports_native:
        client.amend_order = AsyncMock(side_effect=_amend)
        client.edit_order = AsyncMock(side_effect=_amend)
        client.replace_order = AsyncMock(side_effect=_amend)
    else:
        # Native amend raises NotImplementedError
        client.amend_order = AsyncMock(side_effect=NotImplementedError("not supported"))
        client.edit_order = AsyncMock(side_effect=NotImplementedError("not supported"))
        client.replace_order = AsyncMock(side_effect=NotImplementedError("not supported"))

    client.cancel_order = AsyncMock(side_effect=_cancel)
    client.create_order = AsyncMock(side_effect=_create)
    return client


class TestCancelReplaceFallback:
    """test_cancel_replace_fallback — exchange without native amend uses cancel+create."""

    def test_fallback_calls_cancel_and_create(self):
        client = _make_mock_client(supports_native=False)
        mgr = CancelReplaceManager(
            exchange_clients={"bybit": client},
            capability_overrides={"bybit": True},  # declares native but client raises
        )

        async def _run():
            result = await mgr.amend_order(
                exchange="bybit",
                order_id="orig_order_123",
                symbol="BTC/USDT",
                new_price=30_000.0,
                new_size=0.5,
            )
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        # cancel_order must have been called
        client.cancel_order.assert_called_once()
        # create_order must have been called with the new price
        client.create_order.assert_called_once()
        assert result["status"] == "created"

    def test_no_native_amend_in_registry(self):
        """Exchange not in capability registry → goes straight to cancel+create."""
        client = _make_mock_client(supports_native=False)
        mgr = CancelReplaceManager(
            exchange_clients={"deribit": client},
            capability_overrides={"deribit": False},
        )

        async def _run():
            return await mgr.amend_order("deribit", "o1", "BTC/USD", 29999.0)

        result = asyncio.get_event_loop().run_until_complete(_run())
        client.cancel_order.assert_called_once()
        client.create_order.assert_called_once()


class TestCancelReplaceAmendPair:
    """test_cancel_replace_amend_pair — both sides amended simultaneously."""

    def test_amend_pair_calls_gather(self):
        """Both amend_order coroutines run concurrently via asyncio.gather."""
        client = _make_mock_client(supports_native=True)
        mgr = CancelReplaceManager(exchange_clients={"bybit": client})

        gather_called_with_multiple = []

        original_gather = asyncio.gather

        async def _mock_gather(*coros, **kwargs):
            gather_called_with_multiple.append(len(coros))
            return await original_gather(*coros, **kwargs)

        async def _run():
            with patch("asyncio.gather", side_effect=_mock_gather):
                return await mgr.amend_quote_pair(
                    exchange="bybit",
                    bid_order_id="bid_001",
                    ask_order_id="ask_001",
                    symbol="BTC/USDT",
                    new_bid=29_999.0,
                    new_ask=30_001.0,
                    bid_size=1.0,
                    ask_size=1.0,
                )

        bid_result, ask_result = asyncio.get_event_loop().run_until_complete(_run())
        # asyncio.gather must have been called with 2 coroutines
        assert any(n == 2 for n in gather_called_with_multiple)
        assert bid_result is not None
        assert ask_result is not None

    def test_amend_pair_returns_tuple(self):
        client = _make_mock_client(supports_native=True)
        mgr = CancelReplaceManager(exchange_clients={"bybit": client})

        async def _run():
            return await mgr.amend_quote_pair(
                "bybit", "b1", "a1", "BTC/USDT", 30_000.0, 30_002.0, 0.5, 0.5
            )

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert isinstance(result, tuple)
        assert len(result) == 2


# ===========================================================================
# QuoteThrottle tests
# ===========================================================================

class TestQuoteThrottleSuppressesSmallMove:
    """test_quote_throttle_suppresses_small_move — move < min_tick → False."""

    def test_small_bid_move_suppressed(self):
        filt = QuoteThrottleFilter(min_tick=1.0, min_age_ms=0.0, max_rate_per_sec=1000)
        # Record a refresh so age filter passes
        filt.record_refresh("BTC/USD", bid=100.0, ask=101.0)
        time.sleep(0.001)  # ensure age > 0
        # Move of 0.5 < min_tick of 1.0
        result = filt.should_refresh("BTC/USD", new_bid=100.5, new_ask=101.5,
                                     last_bid=100.0, last_ask=101.0)
        assert result is False

    def test_zero_move_suppressed(self):
        filt = QuoteThrottleFilter(min_tick=0.01, min_age_ms=0.0, max_rate_per_sec=1000)
        filt.record_refresh("ETH/USD", bid=2000.0, ask=2001.0)
        time.sleep(0.001)
        result = filt.should_refresh("ETH/USD", new_bid=2000.0, new_ask=2001.0,
                                     last_bid=2000.0, last_ask=2001.0)
        assert result is False


class TestQuoteThrottleAllowsLargeMove:
    """test_quote_throttle_allows_large_move — move > min_tick + age > min_age → True."""

    def test_large_move_allowed(self):
        filt = QuoteThrottleFilter(min_tick=1.0, min_age_ms=0.0, max_rate_per_sec=1000)
        filt.record_refresh("BTC/USD", bid=100.0, ask=101.0)
        time.sleep(0.001)
        # Move of 5.0 > min_tick of 1.0
        result = filt.should_refresh("BTC/USD", new_bid=105.0, new_ask=106.0,
                                     last_bid=100.0, last_ask=101.0)
        assert result is True

    def test_first_quote_allowed(self):
        """With no previous quote the min-tick filter is bypassed."""
        filt = QuoteThrottleFilter(min_tick=1.0, min_age_ms=0.0, max_rate_per_sec=1000)
        result = filt.should_refresh("NEW/USD", new_bid=50.0, new_ask=51.0)
        assert result is True


class TestQuoteThrottleRateLimit:
    """test_quote_throttle_rate_limit — 25 calls in 1s → some suppressed."""

    def test_excess_calls_suppressed(self):
        filt = QuoteThrottleFilter(min_tick=0.0001, min_age_ms=0.0, max_rate_per_sec=10)
        symbol = "BTC/USD"
        sent = 0
        suppressed = 0

        # Flood with 25 refresh checks; price moves enough each time
        for i in range(25):
            bid = 100.0 + i * 1.0
            ask = bid + 1.0
            last_bid = 100.0 + (i - 1) * 1.0 if i > 0 else None
            last_ask = last_bid + 1.0 if last_bid is not None else None
            if filt.should_refresh(symbol, bid, ask, last_bid, last_ask):
                filt.record_refresh(symbol, bid=bid, ask=ask)
                sent += 1
            else:
                suppressed += 1

        # With rate=10, at most ~10 should pass in a single burst
        assert sent <= 10 + 1  # +1 tolerance for timing
        assert suppressed >= 14  # at least 14 of 25 suppressed

    def test_stats_count_suppressed(self):
        filt = QuoteThrottleFilter(min_tick=0.0001, min_age_ms=0.0, max_rate_per_sec=5)
        symbol = "XRP/USD"
        sent_count = 0
        for i in range(20):
            bid = 0.5 + i * 0.1
            ask = bid + 0.01
            last_bid = 0.5 + (i - 1) * 0.1 if i > 0 else None
            last_ask = last_bid + 0.01 if last_bid is not None else None
            if filt.should_refresh(symbol, bid, ask, last_bid, last_ask):
                filt.record_refresh(symbol, bid=bid, ask=ask)
                sent_count += 1

        stats = filt.get_stats(symbol)
        assert "total_suppressed" in stats
        assert "total_sent" in stats
        assert "suppression_rate" in stats
        # total_suppressed + total_sent should equal total calls minus those that
        # passed should_refresh (which are tracked via record_refresh)
        assert stats["total_sent"] == sent_count
        assert stats["total_suppressed"] >= 0
        # With rate=5, at most 5 should pass in one burst, rest suppressed
        assert stats["total_sent"] <= 6  # cap +1 tolerance


# ===========================================================================
# TokenBucket tests
# ===========================================================================

class TestTokenBucket:
    """test_token_bucket — verify token bucket depletes and refills."""

    def test_bucket_starts_full(self):
        bucket = QTTokenBucket(rate=10.0, capacity=10.0)
        assert bucket.available == pytest.approx(10.0, abs=0.1)

    def test_consume_reduces_tokens(self):
        bucket = QTTokenBucket(rate=10.0, capacity=10.0)
        assert bucket.consume() is True
        assert bucket.available < 10.0

    def test_bucket_depletes(self):
        bucket = QTTokenBucket(rate=10.0, capacity=5.0)
        # Consume all 5 tokens
        for _ in range(5):
            bucket.consume()
        # Next consume should fail
        assert bucket.consume() is False

    def test_bucket_refills_over_time(self):
        bucket = QTTokenBucket(rate=100.0, capacity=5.0)
        # Drain it
        for _ in range(5):
            bucket.consume()
        assert bucket.consume() is False
        # Sleep 100ms → at 100 tok/s we get 10 tokens but cap is 5
        time.sleep(0.1)
        assert bucket.consume() is True

    def test_bucket_respects_capacity(self):
        bucket = QTTokenBucket(rate=100.0, capacity=3.0)
        time.sleep(0.2)  # would add 20 tokens without cap
        bucket._refill()
        assert bucket.available == pytest.approx(3.0, abs=0.01)

    def test_cancel_replace_token_bucket_compatible(self):
        """CancelReplace also has its own TokenBucket; verify it behaves the same."""
        bucket = CRTokenBucket(rate=10.0, capacity=3.0)
        for _ in range(3):
            assert bucket.consume() is True
        assert bucket.consume() is False
