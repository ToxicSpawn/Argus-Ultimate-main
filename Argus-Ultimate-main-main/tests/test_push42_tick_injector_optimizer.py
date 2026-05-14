"""Tests for Push 42 — TickInjector + optimize_gateway (25 tests)."""

from __future__ import annotations

import math
import tempfile
import csv
import os
import pytest

from alpha.microstructure.live_ofi_stream import LiveOFIStream
from alpha.microstructure.live_vpin_stream import LiveVPINStream
from alpha.microstructure.tick_injector import TickInjector
from scripts.optimize_gateway import (
    BacktestRow,
    rescore,
    grid_search,
    load_backtest_csv,
    write_results,
    GridResult,
    _ALL_SOURCES,
)


# ---------------------------------------------------------------------------
# TickInjector unit tests
# ---------------------------------------------------------------------------

class TestTickInjector:
    def _make(self):
        ofi  = LiveOFIStream(window=10)
        vpin = LiveVPINStream(bucket_size=5.0)
        inj  = TickInjector(ofi, vpin, symbol="BTC/USDT", exchange_id="binance")
        return inj, ofi, vpin

    def test_init_not_running(self):
        inj, _, _ = self._make()
        assert not inj.is_running

    def test_tick_count_zero(self):
        inj, _, _ = self._make()
        assert inj.tick_count == 0

    def test_ingest_buy_updates_ofi(self):
        inj, ofi, _ = self._make()
        inj._ingest({"price": 100.0, "amount": 1.0, "side": "buy"})
        assert ofi._bar_tape_ofi > 0

    def test_ingest_sell_updates_ofi(self):
        inj, ofi, _ = self._make()
        inj._ingest({"price": 100.0, "amount": 1.0, "side": "sell"})
        assert ofi._bar_tape_ofi < 0

    def test_ingest_updates_vpin_trade_count(self):
        inj, _, vpin = self._make()
        inj._ingest({"price": 100.0, "amount": 1.0, "side": "buy"})
        assert vpin.total_trades == 1

    def test_ingest_zero_amount_ignored(self):
        inj, ofi, vpin = self._make()
        inj._ingest({"price": 100.0, "amount": 0.0, "side": "buy"})
        assert ofi._bar_tape_ofi == 0.0
        assert vpin.total_trades == 0

    def test_ingest_increments_tick_count(self):
        inj, _, _ = self._make()
        for _ in range(5):
            inj._ingest({"price": 100.0, "amount": 1.0, "side": "buy"})
        assert inj.tick_count == 5

    def test_book_delta_from_price_movement(self):
        inj, ofi, _ = self._make()
        inj._ingest({"price": 100.0, "amount": 1.0, "side": "buy"})
        inj._ingest({"price": 101.0, "amount": 1.0, "side": "buy"})
        # price moved up -> positive book delta accumulated
        assert ofi._bar_lob_ofi > 0

    def test_last_tick_ts_updated(self):
        inj, _, _ = self._make()
        inj._ingest({"price": 100.0, "amount": 1.0, "side": "buy"})
        assert inj.last_tick_ts > 0

    def test_multiple_ingest_vpin_feeds(self):
        inj, _, vpin = self._make()
        for i in range(20):
            inj._ingest({"price": 100.0, "amount": 1.0, "side": "buy"})
        assert vpin.total_trades == 20


# ---------------------------------------------------------------------------
# optimize_gateway unit tests
# ---------------------------------------------------------------------------

def _make_rows(n=100, hit_fraction=0.6):
    """Generate synthetic BacktestRow list."""
    import random
    rng = random.Random(42)
    rows = []
    for i in range(n):
        direction = "long" if rng.random() > 0.5 else "short"
        fwd = 0.001 if rng.random() < hit_fraction else -0.001
        if direction == "short":
            fwd = -fwd
        rows.append(BacktestRow(
            bar_idx=i, direction=direction, confidence=0.6,
            sources=["VOID_BREAKER", "OFI_STREAM"],
            fwd_ret_1=fwd, fwd_ret_5=fwd * 2, fwd_ret_15=fwd * 3,
        ))
    return rows


class TestOptimizeGateway:
    def test_rescore_returns_tuple(self):
        rows = _make_rows(50)
        sharpe, hit, n = rescore(rows, min_confidence=0.5,
                                  source_weights={s: 1.0 for s in _ALL_SOURCES})
        assert isinstance(sharpe, float)
        assert isinstance(hit, float)
        assert isinstance(n, int)

    def test_rescore_hit_rate_in_range(self):
        rows = _make_rows(100, hit_fraction=0.7)
        _, hit, _ = rescore(rows, 0.5, {s: 1.0 for s in _ALL_SOURCES})
        assert 0.0 <= hit <= 1.0

    def test_rescore_zero_below_confidence(self):
        rows = _make_rows(50)
        _, _, n = rescore(rows, min_confidence=0.99,
                           source_weights={s: 1.0 for s in _ALL_SOURCES})
        assert n == 0

    def test_grid_search_returns_list(self):
        rows = _make_rows(80)
        results = grid_search(rows, sources=["OFI_STREAM"], top_n=5, quiet=True)
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_grid_search_ranked(self):
        rows = _make_rows(80)
        results = grid_search(rows, sources=["OFI_STREAM"], top_n=10, quiet=True)
        if len(results) >= 2:
            assert results[0].sharpe >= results[1].sharpe

    def test_grid_result_has_weights(self):
        rows = _make_rows(50)
        results = grid_search(rows, sources=["VOID_BREAKER"], top_n=3, quiet=True)
        assert len(results) > 0
        assert "VOID_BREAKER" in results[0].source_weights

    def test_write_results_creates_json(self):
        rows = _make_rows(50)
        results = grid_search(rows, sources=["OFI_STREAM"], top_n=3, quiet=True)
        with tempfile.TemporaryDirectory() as d:
            path = write_results(results, out_dir=d)
            assert os.path.exists(path)
            import json
            with open(path) as f:
                data = json.load(f)
            assert "best_config" in data
            assert "top_results" in data

    def test_load_backtest_csv_roundtrip(self):
        """Write a minimal CSV and load it back."""
        rows_in = _make_rows(10)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            fname = f.name
            writer = csv.DictWriter(f, fieldnames=[
                "bar_idx","direction","confidence","sources",
                "fwd_ret_1","fwd_ret_5","fwd_ret_15",
                "timestamp","close_price","n_sources",
                "hit_1","hit_5","hit_15",
            ])
            writer.writeheader()
            for r in rows_in:
                writer.writerow({
                    "bar_idx": r.bar_idx, "direction": r.direction,
                    "confidence": r.confidence,
                    "sources": ",".join(r.sources),
                    "fwd_ret_1": r.fwd_ret_1, "fwd_ret_5": r.fwd_ret_5,
                    "fwd_ret_15": r.fwd_ret_15,
                    "timestamp": 0, "close_price": 50000,
                    "n_sources": len(r.sources),
                    "hit_1": 1, "hit_5": 1, "hit_15": 1,
                })
        try:
            rows_out = load_backtest_csv(fname)
            assert len(rows_out) == 10
            assert rows_out[0].direction in ("long", "short")
        finally:
            os.unlink(fname)
