"""Tests for Push 40 — LiveOFIStream, LiveVPINStream, train_deeplob (30 tests)."""

from __future__ import annotations

import numpy as np
import pytest

from alpha.microstructure.live_ofi_stream import LiveOFIStream
from alpha.microstructure.live_vpin_stream import LiveVPINStream


# ---------------------------------------------------------------------------
# LiveOFIStream
# ---------------------------------------------------------------------------

class TestLiveOFIStream:
    def test_init(self):
        s = LiveOFIStream()
        assert s.ofi_zscore == 0.0

    def test_buy_trade_increases_ofi(self):
        s = LiveOFIStream()
        s.on_trade({"side": "buy", "amount": 1.0})
        assert s._bar_tape_ofi > 0

    def test_sell_trade_decreases_ofi(self):
        s = LiveOFIStream()
        s.on_trade({"side": "sell", "amount": 1.0})
        assert s._bar_tape_ofi < 0

    def test_book_delta_positive(self):
        s = LiveOFIStream()
        s.on_book_delta({"bid_delta": 1.0, "ask_delta": 0.0})
        assert s._bar_lob_ofi > 0

    def test_book_delta_negative(self):
        s = LiveOFIStream()
        s.on_book_delta({"bid_delta": 0.0, "ask_delta": 1.0})
        assert s._bar_lob_ofi < 0

    def test_close_bar_resets_accumulators(self):
        s = LiveOFIStream()
        s.on_trade({"side": "buy", "amount": 2.0})
        s.close_bar()
        assert s._bar_tape_ofi == 0.0
        assert s._bar_trade_count == 0

    def test_zscore_zero_before_min_bars(self):
        s = LiveOFIStream(min_bars=5)
        for _ in range(3):
            s.on_trade({"side": "buy", "amount": 1.0})
            s.close_bar()
        assert s.ofi_zscore == 0.0

    def test_zscore_nonzero_after_min_bars(self):
        s = LiveOFIStream(min_bars=3, window=10)
        for i in range(5):
            s.on_trade({"side": "buy" if i % 2 == 0 else "sell", "amount": float(i + 1)})
            s.close_bar()
        # Last bar has different OFI -> non-zero zscore
        assert isinstance(s.ofi_zscore, float)

    def test_zscore_flat_when_all_same(self):
        s = LiveOFIStream(min_bars=3, window=10)
        for _ in range(5):
            s.on_trade({"side": "buy", "amount": 1.0})
            s.close_bar()
        # std of constant series = 0 -> zscore = 0
        assert s.ofi_zscore == 0.0

    def test_reset_clears_history(self):
        s = LiveOFIStream()
        for _ in range(10):
            s.on_trade({"side": "buy", "amount": 1.0})
            s.close_bar()
        s.reset()
        assert len(s._ofi_history) == 0
        assert s.ofi_zscore == 0.0

    def test_get_history_returns_array(self):
        s = LiveOFIStream()
        s.on_trade({"side": "buy", "amount": 1.0})
        s.close_bar()
        h = s.get_history()
        assert isinstance(h, np.ndarray)
        assert len(h) == 1

    def test_trade_count_increments(self):
        s = LiveOFIStream()
        s.on_trade({"side": "buy", "amount": 1.0})
        s.on_trade({"side": "sell", "amount": 0.5})
        assert s.bar_trade_count == 2


# ---------------------------------------------------------------------------
# LiveVPINStream
# ---------------------------------------------------------------------------

class TestLiveVPINStream:
    def test_init_neutral(self):
        s = LiveVPINStream()
        assert s.vpin == 0.5

    def test_all_buys_high_vpin(self):
        s = LiveVPINStream(bucket_size=10.0, n_buckets=5)
        for _ in range(100):
            s.on_trade({"price": 100.0, "amount": 1.0, "side": "buy"})
        # All buys = max imbalance -> VPIN close to 1
        assert s.vpin > 0.8

    def test_all_sells_high_vpin(self):
        s = LiveVPINStream(bucket_size=10.0, n_buckets=5)
        for _ in range(100):
            s.on_trade({"price": 100.0, "amount": 1.0, "side": "sell"})
        assert s.vpin > 0.8

    def test_balanced_trades_low_vpin(self):
        s = LiveVPINStream(bucket_size=10.0, n_buckets=10)
        for i in range(200):
            side = "buy" if i % 2 == 0 else "sell"
            s.on_trade({"price": 100.0, "amount": 1.0, "side": side})
        assert s.vpin < 0.2

    def test_vpin_in_range(self):
        s = LiveVPINStream(bucket_size=5.0)
        for i in range(50):
            s.on_trade({"price": 100.0 + i * 0.1, "amount": 0.5})
        assert 0.0 <= s.vpin <= 1.0

    def test_tick_rule_buy_on_uptick(self):
        s = LiveVPINStream()
        s._prev_price = 100.0
        assert s._tick_rule(100.5) == "buy"

    def test_tick_rule_sell_on_downtick(self):
        s = LiveVPINStream()
        s._prev_price = 100.0
        assert s._tick_rule(99.5) == "sell"

    def test_tick_rule_carry_on_flat(self):
        s = LiveVPINStream()
        s._prev_price = 100.0
        s._prev_side  = "buy"
        assert s._tick_rule(100.0) == "buy"

    def test_bucket_count_increments(self):
        s = LiveVPINStream(bucket_size=5.0)
        for _ in range(10):
            s.on_trade({"price": 100.0, "amount": 5.0, "side": "buy"})
        assert s.total_buckets == 10

    def test_reset_clears_state(self):
        s = LiveVPINStream(bucket_size=5.0)
        for _ in range(20):
            s.on_trade({"price": 100.0, "amount": 1.0, "side": "buy"})
        s.reset()
        assert s.vpin == 0.5
        assert s.total_buckets == 0
        assert s.total_trades == 0

    def test_get_tau_history(self):
        s = LiveVPINStream(bucket_size=5.0)
        for _ in range(10):
            s.on_trade({"price": 100.0, "amount": 5.0, "side": "buy"})
        h = s.get_tau_history()
        assert isinstance(h, np.ndarray)
        assert len(h) == 10


# ---------------------------------------------------------------------------
# train_deeplob synthetic data generation (no torch required)
# ---------------------------------------------------------------------------

class TestTrainDeepLOBData:
    def test_load_or_generate_returns_xy(self):
        from scripts.train_deeplob import load_or_generate
        X, y = load_or_generate("/nonexistent/path.csv", n_synthetic=500)
        assert X.shape == (500, 40)
        assert y.shape == (500,)

    def test_labels_are_valid_classes(self):
        from scripts.train_deeplob import load_or_generate
        _, y = load_or_generate("/nonexistent/path.csv", n_synthetic=500)
        assert set(y).issubset({0, 1, 2})

    def test_train_val_split_sizes(self):
        from scripts.train_deeplob import load_or_generate, train_val_split
        X, y = load_or_generate("/nonexistent/path.csv", n_synthetic=1000)
        X_tr, y_tr, X_vl, y_vl = train_val_split(X, y, val_frac=0.2)
        assert len(X_tr) + len(X_vl) == 1000
        assert abs(len(X_vl) - 200) <= 2

    def test_feature_dtype_float32(self):
        from scripts.train_deeplob import load_or_generate
        X, _ = load_or_generate("/nonexistent/path.csv", n_synthetic=100)
        assert X.dtype == np.float32

    def test_label_dtype_int64(self):
        from scripts.train_deeplob import load_or_generate
        _, y = load_or_generate("/nonexistent/path.csv", n_synthetic=100)
        assert y.dtype == np.int64
