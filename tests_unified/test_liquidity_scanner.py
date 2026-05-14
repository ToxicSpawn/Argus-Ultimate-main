"""
M25 — LiquidityScanner test suite.

Tests for services/liquidity_scanner.py covering:
  - Initialisation defaults and custom params
  - scan() with a fully mocked exchange
  - Score calculation / normalisation
  - Cache behaviour
  - Edge cases (empty markets, single pair, below min_volume)
  - LiquidityResult.to_dict()
  - print_scan_table() output
  - run_scan_sync() sync wrapper
"""
from __future__ import annotations

import asyncio
import io
import sys
import time
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exchange_mock(
    symbols: list = None,
    tickers: dict = None,
    order_books: dict = None,
) -> MagicMock:
    """Build a mock ccxt-style async exchange for LiquidityScanner tests."""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    markets = {
        sym: {"symbol": sym, "base": sym.split("/")[0], "quote": sym.split("/")[1], "active": True}
        for sym in symbols
    }

    if tickers is None:
        tickers = {
            "BTC/USDT": {"quoteVolume": 1_000_000.0, "bid": 64990.0, "ask": 65010.0, "last": 65000.0},
            "ETH/USDT": {"quoteVolume": 500_000.0,   "bid": 3490.0,  "ask": 3510.0,  "last": 3500.0},
            "SOL/USDT": {"quoteVolume": 200_000.0,   "bid": 149.5,   "ask": 150.5,   "last": 150.0},
        }

    def _make_ob(bid: float, ask: float, n: int = 10) -> dict:
        return {
            "bids": [[bid - i, 0.5] for i in range(n)],
            "asks": [[ask + i, 0.5] for i in range(n)],
        }

    if order_books is None:
        order_books = {
            "BTC/USDT": _make_ob(64990.0, 65010.0),
            "ETH/USDT": _make_ob(3490.0, 3510.0),
            "SOL/USDT": _make_ob(149.5, 150.5),
        }

    ex = MagicMock()
    ex.markets = markets
    ex.load_markets = AsyncMock(return_value=markets)
    ex.fetch_tickers = AsyncMock(return_value={k: v for k, v in tickers.items() if k in symbols})
    ex.fetch_order_book = AsyncMock(
        side_effect=lambda sym, limit=10: order_books.get(sym, {"bids": [], "asks": []})
    )
    ex.close = AsyncMock()
    return ex


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------

class TestLiquidityScannerInit:
    """Tests that LiquidityScanner stores constructor params correctly."""

    def test_default_init(self):
        """LiquidityScanner uses sensible defaults."""
        from services.liquidity_scanner import LiquidityScanner

        scanner = LiquidityScanner()
        assert scanner.exchange_id == "kraken"
        assert scanner.max_pairs == 20
        assert scanner.min_volume_usd == 50_000.0
        assert scanner.depth_levels == 10
        assert scanner.batch_size == 5
        assert scanner.cache_ttl_s == 60.0

    def test_custom_init(self):
        """LiquidityScanner stores custom parameters."""
        from services.liquidity_scanner import LiquidityScanner

        scanner = LiquidityScanner(
            exchange_id="bybit",
            quote_currencies=["USDT"],
            max_pairs=5,
            min_volume_usd=10_000.0,
            depth_levels=5,
            batch_size=2,
            batch_delay_s=0.5,
            cache_ttl_s=30.0,
            w_volume=0.5,
            w_spread=0.2,
            w_depth=0.2,
            w_imbalance=0.1,
        )
        assert scanner.exchange_id == "bybit"
        assert scanner.max_pairs == 5
        assert scanner.min_volume_usd == 10_000.0
        assert scanner.depth_levels == 5
        assert scanner.w_volume == 0.5

    def test_quote_currencies_normalised_to_upper(self):
        """Quote currencies are normalised to uppercase."""
        from services.liquidity_scanner import LiquidityScanner

        scanner = LiquidityScanner(quote_currencies=["usdt", "usd"])
        assert "USDT" in scanner.quote_currencies
        assert "USD" in scanner.quote_currencies

    def test_injected_exchange_stored(self):
        """Pre-built exchange is stored for reuse (test injection)."""
        from services.liquidity_scanner import LiquidityScanner

        mock_ex = MagicMock()
        scanner = LiquidityScanner(exchange=mock_ex)
        assert scanner._exchange is mock_ex


# ---------------------------------------------------------------------------
# Tests: scan() with mocked exchange
# ---------------------------------------------------------------------------

class TestLiquidityScannerScan:
    """Tests for LiquidityScanner.scan() with injected mock exchange."""

    @pytest.mark.asyncio
    async def test_scan_returns_list(self):
        """scan() returns a list of LiquidityResult objects."""
        from services.liquidity_scanner import LiquidityScanner, LiquidityResult

        ex = _make_exchange_mock()
        scanner = LiquidityScanner(
            exchange_id="kraken",
            quote_currencies=["USDT"],
            min_volume_usd=100.0,
            exchange=ex,
            batch_delay_s=0.0,
        )
        results = await scanner.scan()

        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, LiquidityResult)

    @pytest.mark.asyncio
    async def test_scan_sorted_by_score_descending(self):
        """Results are sorted by liquidity_score descending."""
        from services.liquidity_scanner import LiquidityScanner

        ex = _make_exchange_mock()
        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=100.0,
            exchange=ex,
            batch_delay_s=0.0,
        )
        results = await scanner.scan()

        scores = [r.liquidity_score for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted highest score first"

    @pytest.mark.asyncio
    async def test_scan_filters_below_min_volume(self):
        """Pairs below min_volume_usd are excluded from results."""
        from services.liquidity_scanner import LiquidityScanner

        ex = _make_exchange_mock()
        # Set min_volume to 600k — only BTC/USDT (1M) should pass
        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=600_000.0,
            exchange=ex,
            batch_delay_s=0.0,
        )
        results = await scanner.scan()

        assert len(results) == 1
        assert results[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_scan_empty_markets(self):
        """scan() returns empty list when no markets are available."""
        from services.liquidity_scanner import LiquidityScanner

        ex = MagicMock()
        ex.load_markets = AsyncMock(return_value={})
        ex.markets = {}
        ex.fetch_tickers = AsyncMock(return_value={})
        ex.close = AsyncMock()

        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            exchange=ex,
            batch_delay_s=0.0,
        )
        results = await scanner.scan()
        assert results == []

    @pytest.mark.asyncio
    async def test_scan_uses_cached_results(self):
        """scan() returns cached results before TTL expires."""
        from services.liquidity_scanner import LiquidityScanner, LiquidityResult

        ex = _make_exchange_mock()
        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=100.0,
            exchange=ex,
            batch_delay_s=0.0,
            cache_ttl_s=60.0,
        )
        # First call populates cache
        first = await scanner.scan()
        # Second call should use cache (load_markets called only once)
        second = await scanner.scan()

        assert first == second
        assert ex.load_markets.call_count == 1  # only first scan touches the exchange

    @pytest.mark.asyncio
    async def test_scan_force_bypasses_cache(self):
        """scan(force=True) re-fetches even within TTL."""
        from services.liquidity_scanner import LiquidityScanner

        ex = _make_exchange_mock()
        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=100.0,
            exchange=ex,
            batch_delay_s=0.0,
            cache_ttl_s=60.0,
        )
        await scanner.scan()
        await scanner.scan(force=True)

        assert ex.load_markets.call_count == 2

    @pytest.mark.asyncio
    async def test_scan_max_pairs_respected(self):
        """scan() returns at most max_pairs results."""
        from services.liquidity_scanner import LiquidityScanner

        ex = _make_exchange_mock()
        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=100.0,
            max_pairs=2,
            exchange=ex,
            batch_delay_s=0.0,
        )
        results = await scanner.scan()
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Tests: score calculation
# ---------------------------------------------------------------------------

class TestLiquidityScoreCalculation:
    """Tests that the composite liquidity score is computed correctly."""

    @pytest.mark.asyncio
    async def test_scores_in_unit_interval(self):
        """All liquidity_scores are in [0, 1]."""
        from services.liquidity_scanner import LiquidityScanner

        ex = _make_exchange_mock()
        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=100.0,
            exchange=ex,
            batch_delay_s=0.0,
        )
        results = await scanner.scan()
        for r in results:
            assert 0.0 <= r.liquidity_score <= 1.0, (
                f"{r.symbol} score {r.liquidity_score} out of [0,1]"
            )

    @pytest.mark.asyncio
    async def test_higher_volume_higher_score(self):
        """Given equal spreads and depths, higher volume yields higher score."""
        from services.liquidity_scanner import LiquidityScanner

        # Arrange two pairs: HIGH volume vs LOW volume, identical spread/depth
        tickers = {
            "BTC/USDT": {"quoteVolume": 5_000_000.0, "bid": 64990.0, "ask": 65010.0, "last": 65000.0},
            "ETH/USDT": {"quoteVolume": 100_000.0,   "bid": 3490.0,  "ask": 3510.0,  "last": 3500.0},
        }
        ob = {"bids": [[100.0, 1.0]] * 10, "asks": [[101.0, 1.0]] * 10}
        order_books = {"BTC/USDT": ob, "ETH/USDT": ob}

        ex = _make_exchange_mock(["BTC/USDT", "ETH/USDT"], tickers, order_books)
        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=50_000.0,
            exchange=ex,
            batch_delay_s=0.0,
            w_volume=0.9,
            w_spread=0.033,
            w_depth=0.033,
            w_imbalance=0.034,
        )
        results = await scanner.scan()

        btc = next(r for r in results if r.symbol == "BTC/USDT")
        eth = next(r for r in results if r.symbol == "ETH/USDT")
        assert btc.liquidity_score > eth.liquidity_score

    @pytest.mark.asyncio
    async def test_spread_pct_calculated_correctly(self):
        """spread_pct = (ask - bid) / mid * 100."""
        from services.liquidity_scanner import LiquidityScanner

        tickers = {
            "BTC/USDT": {"quoteVolume": 1_000_000.0, "bid": 64990.0, "ask": 65010.0, "last": 65000.0},
        }
        ob = {"bids": [[64990.0, 0.5]], "asks": [[65010.0, 0.5]]}
        ex = _make_exchange_mock(["BTC/USDT"], tickers, {"BTC/USDT": ob})

        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=100.0,
            exchange=ex,
            batch_delay_s=0.0,
        )
        results = await scanner.scan()
        assert len(results) == 1
        # Expected: (65010 - 64990) / 65000 * 100 ≈ 0.030769
        expected_spread = (65010.0 - 64990.0) / 65000.0 * 100
        assert abs(results[0].spread_pct - expected_spread) < 0.01

    @pytest.mark.asyncio
    async def test_depth_usd_calculated_correctly(self):
        """depth_usd = sum of price*qty for bids + asks."""
        from services.liquidity_scanner import LiquidityScanner

        tickers = {
            "BTC/USDT": {"quoteVolume": 1_000_000.0, "bid": 100.0, "ask": 101.0, "last": 100.5},
        }
        # 2 bid levels @ 100 * 1 = 100 each, 2 ask levels @ 101 * 1 = 101 each
        ob = {
            "bids": [[100.0, 1.0], [99.0, 1.0]],
            "asks": [[101.0, 1.0], [102.0, 1.0]],
        }
        ex = _make_exchange_mock(["BTC/USDT"], tickers, {"BTC/USDT": ob})
        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=100.0,
            depth_levels=2,
            exchange=ex,
            batch_delay_s=0.0,
        )
        results = await scanner.scan()
        expected_depth = 100.0 + 99.0 + 101.0 + 102.0  # = 402
        assert abs(results[0].depth_usd - expected_depth) < 1.0

    @pytest.mark.asyncio
    async def test_single_pair_score_is_valid(self):
        """Single-pair universe still produces a valid score (normalisation edge case)."""
        from services.liquidity_scanner import LiquidityScanner

        tickers = {
            "BTC/USDT": {"quoteVolume": 1_000_000.0, "bid": 64990.0, "ask": 65010.0, "last": 65000.0},
        }
        ob = {"bids": [[64990.0, 0.5]], "asks": [[65010.0, 0.5]]}
        ex = _make_exchange_mock(["BTC/USDT"], tickers, {"BTC/USDT": ob})

        scanner = LiquidityScanner(
            quote_currencies=["USDT"],
            min_volume_usd=100.0,
            exchange=ex,
            batch_delay_s=0.0,
        )
        results = await scanner.scan()
        assert len(results) == 1
        assert math.isfinite(results[0].liquidity_score)
        assert 0.0 <= results[0].liquidity_score <= 1.0


# ---------------------------------------------------------------------------
# Tests: LiquidityResult data model
# ---------------------------------------------------------------------------

class TestLiquidityResult:
    """Tests for the LiquidityResult dataclass."""

    def test_to_dict_contains_all_fields(self):
        """to_dict() contains all expected keys."""
        from services.liquidity_scanner import LiquidityResult

        r = LiquidityResult(
            symbol="BTC/USDT",
            volume_usd=1_000_000.0,
            spread_pct=0.03,
            depth_usd=50_000.0,
            imbalance=0.05,
            liquidity_score=0.85,
        )
        d = r.to_dict()
        for key in ("symbol", "volume_usd", "spread_pct", "depth_usd", "imbalance", "liquidity_score", "timestamp"):
            assert key in d, f"Missing key: {key}"

    def test_liquidity_result_fields(self):
        """LiquidityResult stores passed values correctly."""
        from services.liquidity_scanner import LiquidityResult

        r = LiquidityResult(
            symbol="ETH/USDT",
            volume_usd=500_000.0,
            spread_pct=0.06,
            depth_usd=20_000.0,
            imbalance=0.1,
            liquidity_score=0.72,
        )
        assert r.symbol == "ETH/USDT"
        assert r.volume_usd == 500_000.0
        assert r.spread_pct == 0.06
        assert r.liquidity_score == 0.72

    def test_timestamp_auto_set(self):
        """timestamp is automatically set to current time if not provided."""
        from services.liquidity_scanner import LiquidityResult

        before = time.time()
        r = LiquidityResult("X/Y", 1.0, 0.01, 1.0, 0.01, 0.5)
        after = time.time()
        assert before <= r.timestamp <= after


# ---------------------------------------------------------------------------
# Tests: print_scan_table
# ---------------------------------------------------------------------------

class TestPrintScanTable:
    """Tests for the ASCII table printer."""

    def test_print_scan_table_outputs_rows(self, capsys):
        """print_scan_table prints one row per result."""
        from services.liquidity_scanner import LiquidityResult, print_scan_table

        results = [
            LiquidityResult("BTC/USDT", 1_000_000.0, 0.03, 50_000.0, 0.05, 0.9),
            LiquidityResult("ETH/USDT", 500_000.0,   0.06, 20_000.0, 0.1,  0.7),
        ]
        print_scan_table(results, top_n=2)
        captured = capsys.readouterr()
        assert "BTC/USDT" in captured.out
        assert "ETH/USDT" in captured.out

    def test_print_scan_table_respects_top_n(self, capsys):
        """print_scan_table prints at most top_n rows."""
        from services.liquidity_scanner import LiquidityResult, print_scan_table

        results = [
            LiquidityResult(f"SYM{i}/USDT", float(i * 1000), 0.01, 1000.0, 0.05, float(i) / 10)
            for i in range(10, 0, -1)
        ]
        print_scan_table(results, top_n=3)
        captured = capsys.readouterr()
        # Only 3 symbols should appear
        count = sum(1 for line in captured.out.splitlines() if "USDT" in line)
        assert count == 3
