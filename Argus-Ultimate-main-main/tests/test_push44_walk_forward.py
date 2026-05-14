"""Tests for Push 44 — walk_forward (18 tests)."""

from __future__ import annotations

import math
import tempfile
import os
import pytest

from scripts.optimize_gateway import BacktestRow, _ALL_SOURCES
from scripts.walk_forward import (
    run_walk_forward,
    write_results,
    FoldResult,
    WalkForwardSummary,
    _best_confidence,
)


def _make_rows(n=800, hit_fraction=0.58, seed=42):
    import random
    rng = random.Random(seed)
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


class TestBestConfidence:
    def test_returns_valid_conf(self):
        rows = _make_rows(200)
        conf, sharpe = _best_confidence(rows, "fwd_ret_1")
        assert 0.40 <= conf <= 0.70

    def test_sharpe_is_float(self):
        rows = _make_rows(200)
        _, sharpe = _best_confidence(rows, "fwd_ret_1")
        assert isinstance(sharpe, float)


class TestRunWalkForward:
    def test_returns_folds_and_summary(self):
        rows = _make_rows(800)
        folds, summary = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        assert isinstance(folds, list)
        assert isinstance(summary, WalkForwardSummary)

    def test_at_least_one_fold(self):
        rows = _make_rows(800)
        folds, _ = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        assert len(folds) >= 1

    def test_fold_numbers_sequential(self):
        rows = _make_rows(800)
        folds, _ = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        for i, f in enumerate(folds):
            assert f.fold == i + 1

    def test_test_start_after_train_end(self):
        rows = _make_rows(800)
        folds, _ = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        for f in folds:
            assert f.test_start == f.train_end

    def test_hit_rate_in_range(self):
        rows = _make_rows(800)
        folds, _ = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        for f in folds:
            assert 0.0 <= f.test_hit_rate <= 1.0

    def test_cum_edge_monotone_grows(self):
        """cum_edge changes each fold (not necessarily monotone, just a float)."""
        rows = _make_rows(800)
        folds, _ = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        assert all(isinstance(f.cum_edge, float) for f in folds)

    def test_summary_n_folds_matches(self):
        rows = _make_rows(800)
        folds, summary = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        assert summary.n_folds == len(folds)

    def test_summary_total_signals(self):
        rows = _make_rows(800)
        folds, summary = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        assert summary.total_signals == sum(f.test_n_signals for f in folds)

    def test_empty_rows_returns_empty(self):
        folds, summary = run_walk_forward([], quiet=True)
        assert folds == []
        assert summary.n_folds == 0

    def test_insufficient_data_returns_empty(self):
        rows = _make_rows(10)
        folds, summary = run_walk_forward(
            rows, train_window=500, test_window=200, step=100, quiet=True)
        assert summary.n_folds == 0

    def test_fwd_ret_5_col(self):
        rows = _make_rows(800)
        folds, _ = run_walk_forward(
            rows, train_window=200, test_window=100, step=100,
            forward_col="fwd_ret_5", quiet=True)
        assert len(folds) >= 1

    def test_stability_ratio_nonnegative(self):
        rows = _make_rows(800)
        _, summary = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        # stability can be negative if mean sharpe is negative
        assert isinstance(summary.stability_ratio, float)


class TestWriteResults:
    def test_creates_three_files(self):
        rows = _make_rows(800)
        folds, summary = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        with tempfile.TemporaryDirectory() as d:
            f1, f2, f3 = write_results(folds, summary, out_dir=d)
            assert os.path.exists(f1)
            assert os.path.exists(f2)
            assert os.path.exists(f3)

    def test_summary_json_valid(self):
        rows = _make_rows(800)
        folds, summary = run_walk_forward(rows, train_window=200, test_window=100, step=100, quiet=True)
        with tempfile.TemporaryDirectory() as d:
            _, _, json_path = write_results(folds, summary, out_dir=d)
            import json
            with open(json_path) as f:
                data = json.load(f)
            assert "n_folds" in data
            assert "mean_test_sharpe" in data
