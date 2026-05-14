"""
Tests for hft_engine/order_book_processor.py and hft_engine/hft_scalping_engine.py

Covers:
    1.  L2OrderBook basic update (add/update/remove levels)
    2.  L2OrderBook best_bid / best_ask ordering
    3.  L2OrderBook spread_bps calculation
    4.  L2OrderBook imbalance — symmetric book returns 0
    5.  L2OrderBook imbalance — bid-heavy book returns positive value
    6.  L2OrderBook microprice — sanity vs arithmetic mid
    7.  L2OrderBook weighted_mid correctness
    8.  L2OrderBook depth() returns correct number of levels
    9.  L3OrderBook add / cancel / fill / queue_position
    10. OrderBookSignals OBI z-score — extreme imbalance returns |z| > 1
    11. OrderBookSignals VPIN — high informed flow pushes VPIN > 0.6
    12. OrderBookSignals liquidity_sweep_detected on sharp depth drop
    13. OrderBookSignals CVD accumulates correctly
    14. HFTScalpingEngine.analyze_order_book returns signal on strong imbalance
    15. HFTScalpingEngine.analyze_order_book returns None on balanced book
    16. HFTScalpingEngine.analyze_trade_flow returns signal on high VPIN trades
    17. HFTScalpingEngine.scan_for_opportunities returns sorted list
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import pytest

from hft_engine.order_book_processor import (
    L2OrderBook,
    L3OrderBook,
    OrderBookSignals,
    PriceLevel,
    _book_from_dict,
)
from hft_engine.hft_scalping_engine import HFTScalpingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_book(bids: List, asks: List, symbol: str = "TEST") -> L2OrderBook:
    """Build an L2OrderBook from list-of-[price, size] pairs."""
    book = L2OrderBook(symbol=symbol)
    for price, size in bids:
        book.update("bid", price, size)
    for price, size in asks:
        book.update("ask", price, size)
    return book


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. L2OrderBook — basic update (add / update / remove)
# ---------------------------------------------------------------------------


def test_l2_update_add_and_remove():
    book = L2OrderBook("SYM")
    book.update("bid", 100.0, 10.0)
    assert book.best_bid is not None
    assert book.best_bid.price == pytest.approx(100.0)
    # Remove by setting size=0
    book.update("bid", 100.0, 0.0)
    assert book.best_bid is None


# ---------------------------------------------------------------------------
# 2. L2OrderBook — best_bid / best_ask ordering
# ---------------------------------------------------------------------------


def test_l2_best_bid_ask_ordering():
    book = _build_book(
        bids=[(99.0, 5.0), (100.0, 3.0), (98.0, 7.0)],
        asks=[(101.0, 4.0), (102.0, 6.0), (100.5, 2.0)],
    )
    assert book.best_bid.price == pytest.approx(100.0), "Best bid must be highest bid price"
    assert book.best_ask.price == pytest.approx(100.5), "Best ask must be lowest ask price"


# ---------------------------------------------------------------------------
# 3. L2OrderBook — spread_bps
# ---------------------------------------------------------------------------


def test_l2_spread_bps():
    book = _build_book(bids=[(100.0, 1.0)], asks=[(100.1, 1.0)])
    spread = book.spread_bps
    # mid = 100.05, spread = 0.1/100.05*10000 ≈ 9.995 bps
    assert 9.0 < spread < 11.0


def test_l2_spread_bps_empty_returns_zero():
    book = L2OrderBook()
    assert book.spread_bps == 0.0


# ---------------------------------------------------------------------------
# 4. L2OrderBook — imbalance symmetric → 0
# ---------------------------------------------------------------------------


def test_l2_imbalance_symmetric():
    book = _build_book(
        bids=[(100.0, 10.0), (99.5, 10.0)],
        asks=[(100.5, 10.0), (101.0, 10.0)],
    )
    obi = book.imbalance(levels=2)
    assert obi == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. L2OrderBook — imbalance bid-heavy → positive
# ---------------------------------------------------------------------------


def test_l2_imbalance_bid_heavy():
    book = _build_book(
        bids=[(100.0, 90.0)],
        asks=[(101.0, 10.0)],
    )
    obi = book.imbalance(levels=5)
    # (90 - 10) / (90 + 10) = 0.8
    assert obi == pytest.approx(0.8, rel=1e-4)


# ---------------------------------------------------------------------------
# 6. L2OrderBook — microprice vs arithmetic mid
# ---------------------------------------------------------------------------


def test_l2_microprice_direction():
    # Heavy bids → microprice should be above arithmetic mid
    book = _build_book(
        bids=[(100.0, 80.0)],
        asks=[(102.0, 20.0)],
    )
    mid = book.mid_price
    mp = book.microprice()
    assert mp is not None
    # OBI = (80-20)/100 = 0.6, half_spread = 1.0, microprice = 101 + 0.6 = 101.6
    assert mp > mid, "Bid-heavy book: microprice must exceed arithmetic mid"


# ---------------------------------------------------------------------------
# 7. L2OrderBook — weighted_mid sanity
# ---------------------------------------------------------------------------


def test_l2_weighted_mid_between_bid_and_ask():
    book = _build_book(
        bids=[(100.0, 50.0)],
        asks=[(102.0, 50.0)],
    )
    wm = book.weighted_mid()
    assert wm is not None
    assert 100.0 < wm < 102.0


# ---------------------------------------------------------------------------
# 8. L2OrderBook — depth() level count
# ---------------------------------------------------------------------------


def test_l2_depth_respects_n():
    bids = [(float(100 - i), float(i + 1)) for i in range(20)]
    asks = [(float(101 + i), float(i + 1)) for i in range(20)]
    book = _build_book(bids, asks)
    d = book.depth(n=5)
    assert len(d["bids"]) == 5
    assert len(d["asks"]) == 5


# ---------------------------------------------------------------------------
# 9. L3OrderBook — add / cancel / fill / queue_position
# ---------------------------------------------------------------------------


def test_l3_order_lifecycle():
    lb = L3OrderBook("SYM3")

    lb.add_order("O1", "bid", 100.0, 5.0)
    lb.add_order("O2", "bid", 100.0, 3.0)
    lb.add_order("O3", "bid", 100.0, 2.0)

    # O1 first, O2 second, O3 third
    assert lb.queue_position("O1") == 0
    assert lb.queue_position("O2") == 1
    assert lb.queue_position("O3") == 2

    # Fill O1 partially
    lb.fill_order("O1", 2.0)
    assert lb._orders["O1"].size == pytest.approx(3.0)

    # Cancel O1 → O2 moves to front
    lb.cancel_order("O1")
    assert lb.queue_position("O2") == 0
    assert lb.queue_position("O3") == 1

    # Unknown order
    assert lb.queue_position("GHOST") == -1

    assert len(lb) == 2


# ---------------------------------------------------------------------------
# 10. OrderBookSignals — OBI z-score extreme imbalance
# ---------------------------------------------------------------------------


def test_obi_zscore_extreme_imbalance():
    engine = OrderBookSignals("BTC", window=30)

    # Feed 29 balanced ticks then one extreme bid-heavy tick
    for _ in range(29):
        book = _build_book(bids=[(100.0, 10.0)], asks=[(101.0, 10.0)])
        engine.update(book)

    # Extreme bid heavy
    book = _build_book(bids=[(100.0, 95.0)], asks=[(101.0, 5.0)])
    engine.update(book)

    z = engine.obi_signal()
    assert z > 1.5, f"Expected OBI z > 1.5, got {z:.4f}"


# ---------------------------------------------------------------------------
# 11. OrderBookSignals — VPIN high informed flow
# ---------------------------------------------------------------------------


def test_vpin_high_informed_flow():
    engine = OrderBookSignals("ETH", window=100)
    # Pump 500 units of pure buy — should produce very skewed VPIN buckets
    for _ in range(500):
        engine.add_trade(price=1000.0, size=1.0, side="buy")

    v = engine.vpin(bucket_size=50.0)
    # Each bucket should be nearly all buy → VPIN close to 1
    assert v > 0.6, f"Expected VPIN > 0.6, got {v:.4f}"


# ---------------------------------------------------------------------------
# 12. OrderBookSignals — liquidity_sweep_detected
# ---------------------------------------------------------------------------


def test_liquidity_sweep_detected():
    engine = OrderBookSignals("SOL", window=50)

    # Thick book
    book = _build_book(bids=[(100.0, 1000.0)], asks=[(101.0, 1000.0)])
    engine.update(book)

    # After sweep: nearly empty top of book
    book2 = _build_book(bids=[(100.0, 10.0)], asks=[(101.0, 10.0)])
    engine.update(book2)

    assert engine.liquidity_sweep_detected(threshold_pct=0.05) is True


def test_no_sweep_on_stable_book():
    engine = OrderBookSignals("SOL", window=50)
    for _ in range(5):
        book = _build_book(bids=[(100.0, 100.0)], asks=[(101.0, 100.0)])
        engine.update(book)
    assert engine.liquidity_sweep_detected(threshold_pct=0.05) is False


# ---------------------------------------------------------------------------
# 13. OrderBookSignals — CVD accumulation
# ---------------------------------------------------------------------------


def test_cvd_net_buying():
    engine = OrderBookSignals("ADA", window=50)
    for _ in range(10):
        engine.add_trade(1.0, 10.0, "buy")
    for _ in range(3):
        engine.add_trade(1.0, 10.0, "sell")

    cvd = engine.cvd(lookback=50)
    # 10*10 - 3*10 = 70
    assert cvd == pytest.approx(70.0)


def test_cvd_net_selling():
    engine = OrderBookSignals("XRP", window=50)
    for _ in range(8):
        engine.add_trade(0.5, 5.0, "sell")

    cvd = engine.cvd(lookback=50)
    assert cvd < 0


# ---------------------------------------------------------------------------
# 14. HFTScalpingEngine — returns signal on strong imbalance
# ---------------------------------------------------------------------------


def test_hft_engine_signal_on_strong_imbalance():
    """
    Feed 30 balanced ticks with tight spreads, then one extreme bid-heavy tick.
    With |OBI z| > 1.5 and spread_percentile < 0.3 (current spread is tight
    relative to history), a 'long' signal should be returned.

    Strategy:
        - Warm-up: balanced book with gradually widening spread (5 → 34 bps)
          so the spread history has a range.
        - Final tick: extreme bid imbalance with the tightest spread in the
          set (5 bps) → spread_percentile ≈ 0 (very tight vs history).
    """
    engine = HFTScalpingEngine(config={})
    sym = "BTC-USD"

    # Prime with wide spread history (50–100 bps) so any tight tick is < 0.3
    for i in range(30):
        spread = 0.50 + i * 0.05   # 0.50 → 1.95 (wide spread, many bps)
        wide_ob = {
            "bids": [[100.0, 10.0]],
            "asks": [[100.0 + spread, 10.0]],
        }
        _run(engine.analyze_order_book(sym, wide_ob))

    # Now inject extreme bid-heavy tick with a *tight* spread
    # spread = 0.01 → ~1 bps, far below the warm-up history
    imbalanced_ob = {
        "bids": [[100.00, 950.0], [99.99, 50.0]],
        "asks": [[100.01, 50.0], [100.02, 10.0]],
    }
    sig = _run(engine.analyze_order_book(sym, imbalanced_ob))

    assert sig is not None, "Expected a signal on extreme bid imbalance with tight spread"
    assert sig["direction"] == "long"
    assert sig["obi"] > 0
    assert "spread_bps" in sig
    assert "microprice" in sig
    assert 0.0 < sig["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# 15. HFTScalpingEngine — returns None on balanced book
# ---------------------------------------------------------------------------


def test_hft_engine_no_signal_balanced():
    engine = HFTScalpingEngine(config={})
    ob = {
        "bids": [[100.0, 10.0]],
        "asks": [[101.0, 10.0]],
    }
    # Single tick — not enough history for z-score to cross threshold
    sig = _run(engine.analyze_order_book("ETH-USD", ob))
    assert sig is None


# ---------------------------------------------------------------------------
# 16. HFTScalpingEngine — analyze_trade_flow returns signal on high VPIN
# ---------------------------------------------------------------------------


def test_hft_engine_trade_flow_vpin_signal():
    engine = HFTScalpingEngine(config={"vpin_threshold": 0.5})
    sym = "ETH-USD"

    # Push lots of pure buy trades to drive VPIN > 0.5
    buys = [{"price": 2000.0, "size": 5.0, "side": "buy"} for _ in range(200)]
    sig = _run(engine.analyze_trade_flow(sym, buys))

    assert sig is not None, "Expected trade flow signal on heavy buying"
    assert sig["direction"] == "long"
    assert sig["vpin"] > 0.5
    assert "cvd" in sig


# ---------------------------------------------------------------------------
# 17. HFTScalpingEngine — scan_for_opportunities returns sorted list
# ---------------------------------------------------------------------------


def test_scan_for_opportunities_sorted():
    engine = HFTScalpingEngine(config={})

    # Prime BTC history with balanced ticks then inject imbalanced one
    for _ in range(30):
        _run(engine.analyze_order_book("BTC-USD", {
            "bids": [[30000.0, 10.0]], "asks": [[30100.0, 10.0]]
        }))

    request: Dict[str, Any] = {
        "symbols": ["BTC-USD", "ETH-USD"],
        "order_books": {
            "BTC-USD": {
                "bids": [[30000.0, 900.0], [29990.0, 100.0]],
                "asks": [[30100.0, 50.0], [30200.0, 10.0]],
            },
            "ETH-USD": {
                "bids": [[2000.0, 10.0]],
                "asks": [[2010.0, 10.0]],
            },
        },
        "recent_trades": {
            "BTC-USD": [{"price": 30000.0, "size": 5.0, "side": "buy"} for _ in range(20)],
        },
    }

    opps = _run(engine.scan_for_opportunities(request))

    # Should be a list (may be empty if thresholds not met, but must be a list)
    assert isinstance(opps, list)
    # All returned opportunities must have required keys
    for opp in opps:
        assert "symbol" in opp
        assert "direction" in opp
        assert "confidence" in opp
        assert "expected_profit_pct" in opp

    # Must be sorted by confidence descending
    confidences = [o["confidence"] for o in opps]
    assert confidences == sorted(confidences, reverse=True), "Opportunities not sorted by confidence"
