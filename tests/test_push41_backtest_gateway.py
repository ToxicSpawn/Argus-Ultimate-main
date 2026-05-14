"""Tests for Push 41 — backtest_gateway (20 tests)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from scripts.backtest_gateway import (
    _synthetic_ohlcv,
    _bar_to_trades,
    _consensus,
    _forward_return,
    _hit,
    _signal_ofi,
    _signal_vpin,
    load_ohlcv_csv,
    run_backtest,
    write_results,
    BacktestSummary,
    SignalRecord,
)
from alpha.microstructure.live_ofi_stream import LiveOFIStream
from alpha.microstructure.live_vpin_stream import LiveVPINStream


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

class TestSyntheticOHLCV:
    def test_shape(self):
        arr = _synthetic_ohlcv(100)
        assert arr.shape == (100, 6)

    def test_close_positive(self):
        arr = _synthetic_ohlcv(50)
        assert (arr[:, 4] > 0).all()

    def test_reproducible(self):
        a = _synthetic_ohlcv(100, seed=1)
        b = _synthetic_ohlcv(100, seed=1)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Trade decomposition
# ---------------------------------------------------------------------------

class TestBarToTrades:
    def test_returns_list(self):
        arr = _synthetic_ohlcv(10)
        trades = _bar_to_trades(arr[5], n_trades=10)
        assert len(trades) == 10

    def test_trade_keys(self):
        arr = _synthetic_ohlcv(10)
        t = _bar_to_trades(arr[0], n_trades=5)[0]
        assert "price" in t and "amount" in t and "side" in t

    def test_valid_sides(self):
        arr = _synthetic_ohlcv(10)
        trades = _bar_to_trades(arr[0], n_trades=20)
        assert all(t["side"] in ("buy", "sell") for t in trades)


# ---------------------------------------------------------------------------
# Consensus engine
# ---------------------------------------------------------------------------

class TestConsensus:
    def test_majority_long(self):
        signals = [("A", "long", 0.8), ("B", "long", 0.7), ("C", "short", 0.3)]
        result = _consensus(signals, min_confidence=0.5)
        assert result is not None
        assert result[0] == "long"

    def test_returns_none_below_confidence(self):
        signals = [("A", "long", 0.6), ("B", "short", 0.6)]
        result = _consensus(signals, min_confidence=0.9)
        assert result is None

    def test_empty_signals_returns_none(self):
        assert _consensus([], min_confidence=0.5) is None

    def test_short_majority(self):
        signals = [("A", "short", 0.9), ("B", "short", 0.8)]
        result = _consensus(signals, min_confidence=0.5)
        assert result[0] == "short"


# ---------------------------------------------------------------------------
# Forward return / hit
# ---------------------------------------------------------------------------

class TestForwardReturn:
    def test_positive_return(self):
        closes = np.array([100.0, 101.0, 102.0, 103.0])
        r = _forward_return(closes, 0, 2)
        assert abs(r - 0.02) < 1e-9

    def test_out_of_bounds_returns_nan(self):
        closes = np.array([100.0, 101.0])
        r = _forward_return(closes, 1, 5)
        assert math.isnan(r)

    def test_hit_long_positive(self):
        assert _hit("long", 0.01) == 1

    def test_hit_long_negative(self):
        assert _hit("long", -0.01) == 0

    def test_hit_short_negative(self):
        assert _hit("short", -0.01) == 1


# ---------------------------------------------------------------------------
# Full backtest integration (synthetic, no live data)
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_returns_records_and_summary(self):
        candles = _synthetic_ohlcv(200)
        records, summary = run_backtest(candles, quiet=True)
        assert isinstance(records, list)
        assert isinstance(summary, BacktestSummary)

    def test_hit_rates_in_range(self):
        candles = _synthetic_ohlcv(200)
        _, summary = run_backtest(candles, quiet=True)
        assert 0.0 <= summary.hit_rate_1 <= 1.0
        assert 0.0 <= summary.hit_rate_5 <= 1.0

    def test_signal_count_nonnegative(self):
        candles = _synthetic_ohlcv(200)
        _, summary = run_backtest(candles, quiet=True)
        assert summary.n_signals >= 0

    def test_long_short_sum_equals_total(self):
        candles = _synthetic_ohlcv(200)
        _, summary = run_backtest(candles, quiet=True)
        assert summary.long_signals + summary.short_signals == summary.n_signals
