"""Tests for backtesting.paper_backtest_reconciler."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time

import pytest

from backtesting.paper_backtest_reconciler import (
    PaperBacktestReconciler,
    ReconciliationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_paper_ledger(path: str, trades: list[dict]) -> None:
    """Create a minimal TradeLedger-compatible SQLite DB with the given trades."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            order_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            exchange TEXT,
            size REAL NOT NULL,
            price REAL NOT NULL,
            status TEXT NOT NULL,
            commission REAL,
            slippage REAL,
            pnl REAL,
            value REAL NOT NULL,
            raw_json TEXT
        )
        """
    )
    for t in trades:
        conn.execute(
            """
            INSERT INTO trades
                (timestamp, order_id, symbol, side, exchange, size, price,
                 status, commission, slippage, pnl, value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t.get("timestamp", time.time()),
                t.get("order_id", "ord-1"),
                t.get("symbol", "BTC/USD"),
                t.get("side", "BUY"),
                t.get("exchange", "kraken"),
                t.get("size", 0.01),
                t.get("price", 50000.0),
                t.get("status", "filled"),
                t.get("commission", 0.5),
                t.get("slippage", 0.0),
                t.get("pnl", 0.0),
                t.get("value", t.get("size", 0.01) * t.get("price", 50000.0)),
            ),
        )
    conn.commit()
    conn.close()


def _make_bt_trade(
    entry_time: float,
    exit_time: float | None,
    side: str = "BUY",
    entry_price: float = 50000.0,
    exit_price: float | None = 51000.0,
    quantity: float = 0.01,
    pnl_usd: float = 10.0,
    impact_cost_usd: float = 0.3,
    symbol: str = "BTC/USD",
    exit_reason: str = "take_profit",
) -> dict:
    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "pnl_usd": pnl_usd,
        "impact_cost_usd": impact_cost_usd,
        "symbol": symbol,
        "exit_reason": exit_reason,
    }


@pytest.fixture
def tmp_ledger(tmp_path):
    """Return a path for a temporary ledger DB."""
    return str(tmp_path / "test_trades.db")


# ---------------------------------------------------------------------------
# ReconciliationReport dataclass
# ---------------------------------------------------------------------------

class TestReconciliationReport:
    def test_default_values(self):
        r = ReconciliationReport()
        assert r.trade_count_paper == 0
        assert r.trade_count_backtest == 0
        assert r.total_pnl_paper == 0.0
        assert r.pnl_divergence_pct == 0.0
        assert r.divergent_trades == []
        assert r.summary == ""

    def test_fields_settable(self):
        r = ReconciliationReport(
            trade_count_paper=10,
            trade_count_backtest=12,
            total_pnl_paper=100.0,
            total_pnl_backtest=120.0,
            pnl_divergence_pct=16.7,
            signal_agreement_pct=85.0,
            slippage_impact_bps=3.5,
        )
        assert r.trade_count_paper == 10
        assert r.total_pnl_backtest == 120.0
        assert r.signal_agreement_pct == 85.0


# ---------------------------------------------------------------------------
# PaperBacktestReconciler — reconcile()
# ---------------------------------------------------------------------------

class TestReconcile:
    def test_empty_ledger_and_empty_backtest(self, tmp_ledger):
        _create_paper_ledger(tmp_ledger, [])
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": []})
        report = rec.reconcile()
        assert report.trade_count_paper == 0
        assert report.trade_count_backtest == 0
        assert report.pnl_divergence_pct == 0.0

    def test_matching_trades(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "commission": 0.5, "order_id": "p1",
             "symbol": "BTC/USD"},
            {"timestamp": base_ts + 3600, "side": "SELL", "price": 51000.0,
             "size": 0.01, "pnl": -5.0, "commission": 0.5, "order_id": "p2",
             "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 1800, side="BUY",
                           entry_price=50000.0, pnl_usd=10.0),
            _make_bt_trade(base_ts + 3600, base_ts + 5400, side="SELL",
                           entry_price=51000.0, pnl_usd=-5.0),
        ]
        bt_results = {
            "all_trades": bt_trades,
            "combined_sharpe": 1.2,
            "combined_max_drawdown_pct": 0.05,
        }

        rec = PaperBacktestReconciler(tmp_ledger, bt_results)
        report = rec.reconcile()

        assert report.trade_count_paper == 2
        assert report.trade_count_backtest == 2
        assert report.total_pnl_paper == pytest.approx(5.0)
        assert report.total_pnl_backtest == pytest.approx(5.0)
        assert report.pnl_divergence_pct == pytest.approx(0.0)
        assert report.signal_agreement_pct == pytest.approx(100.0)
        assert report.sharpe_backtest == pytest.approx(1.2)
        assert report.max_drawdown_backtest == pytest.approx(0.05)

    def test_pnl_divergence(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 100.0, "order_id": "p1", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=80.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        report = rec.reconcile()

        # divergence = |100-80| / max(100,80) * 100 = 20%
        assert report.pnl_divergence_pct == pytest.approx(20.0)

    def test_direction_mismatch_creates_divergent(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="SELL",
                           entry_price=50000.0, pnl_usd=10.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        report = rec.reconcile()

        assert report.signal_agreement_pct == pytest.approx(0.0)
        assert len(report.divergent_trades) >= 1
        assert report.divergent_trades[0]["type"] == "direction_mismatch"

    def test_unmatched_paper_trade(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
            {"timestamp": base_ts + 100_000, "side": "SELL", "price": 52000.0,
             "size": 0.01, "pnl": 20.0, "order_id": "p2", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        # Only one BT trade — the second paper trade is unmatched
        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=10.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        report = rec.reconcile()

        unmatched = [d for d in report.divergent_trades if d["type"] == "unmatched"]
        assert len(unmatched) >= 1

    def test_win_rate_computation(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        # 3 paper trades: 2 wins, 1 loss
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
            {"timestamp": base_ts + 3600, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 5.0, "order_id": "p2", "symbol": "BTC/USD"},
            {"timestamp": base_ts + 7200, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": -3.0, "order_id": "p3", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": []})
        report = rec.reconcile()
        assert report.win_rate_paper == pytest.approx(2 / 3)

    def test_slippage_impact_bps(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        # Paper got filled at 50050, backtest assumed 50000 → 10 bps slip
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50050.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=10.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        report = rec.reconcile()

        expected_bps = abs(50050.0 - 50000.0) / 50000.0 * 10_000
        assert report.slippage_impact_bps == pytest.approx(expected_bps, rel=0.01)

    def test_summary_is_nonempty(self, tmp_ledger):
        _create_paper_ledger(tmp_ledger, [])
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": []})
        report = rec.reconcile()
        assert "Reconciliation" in report.summary

    def test_sharpe_from_paper_pnl(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        paper_trades = [
            {"timestamp": base_ts + i * 3600, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": float(i), "order_id": f"p{i}",
             "symbol": "BTC/USD"}
            for i in range(1, 11)
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": []})
        report = rec.reconcile()
        # All pnls are positive so Sharpe should be positive
        assert report.sharpe_paper > 0

    def test_max_drawdown_paper(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        # Sequence: +10, -20 → cumulative goes 10, -10 → drawdown from peak 10
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
            {"timestamp": base_ts + 3600, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": -20.0, "order_id": "p2", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": []})
        report = rec.reconcile()
        # Peak = 10, trough after = -10 → dd = 20/10 = 200%? No, capped sense:
        # dd = (10 - (-10)) / 10 = 2.0 (200%)
        assert report.max_drawdown_paper > 0

    def test_uses_precomputed_bt_sharpe(self, tmp_ledger):
        _create_paper_ledger(tmp_ledger, [])
        rec = PaperBacktestReconciler(
            tmp_ledger,
            {"all_trades": [], "combined_sharpe": 2.5, "combined_max_drawdown_pct": 0.12},
        )
        report = rec.reconcile()
        assert report.sharpe_backtest == pytest.approx(2.5)
        assert report.max_drawdown_backtest == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# identify_divergence_sources()
# ---------------------------------------------------------------------------

class TestDivergenceSources:
    def test_no_divergence(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "commission": 0.3,
             "order_id": "p1", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=10.0,
                           impact_cost_usd=0.3),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        rec.reconcile()
        sources = rec.identify_divergence_sources()
        assert isinstance(sources, list)
        assert len(sources) >= 1
        # Should report no significant divergence
        assert any("no_significant" in s for s in sources)

    def test_missed_fills_detected(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        _create_paper_ledger(tmp_ledger, [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
        ])

        # Backtest has 3 trades, paper only 1
        bt_trades = [
            _make_bt_trade(base_ts + i * 3600, base_ts + i * 3600 + 1800,
                           pnl_usd=5.0)
            for i in range(3)
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        rec.reconcile()
        sources = rec.identify_divergence_sources()
        assert any("missed_fill" in s for s in sources)

    def test_slippage_detected(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        # Paper price is 200 bps worse than backtest
        _create_paper_ledger(tmp_ledger, [
            {"timestamp": base_ts, "side": "BUY", "price": 50100.0,
             "size": 0.01, "pnl": 5.0, "order_id": "p1", "symbol": "BTC/USD"},
        ])

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=10.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        rec.reconcile()
        sources = rec.identify_divergence_sources()
        assert any("slippage" in s for s in sources)

    def test_timing_delay_detected(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        # Paper fill is 5 min later than BT signal
        _create_paper_ledger(tmp_ledger, [
            {"timestamp": base_ts + 300, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
        ])

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=10.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        rec.reconcile()
        sources = rec.identify_divergence_sources()
        assert any("timing" in s for s in sources)

    def test_direction_mismatch_detected(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        _create_paper_ledger(tmp_ledger, [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
        ])

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="SELL",
                           entry_price=50000.0, pnl_usd=10.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        rec.reconcile()
        sources = rec.identify_divergence_sources()
        assert any("direction" in s for s in sources)

    def test_fee_model_divergence(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        _create_paper_ledger(tmp_ledger, [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "commission": 5.0,
             "order_id": "p1", "symbol": "BTC/USD"},
        ])

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=10.0,
                           impact_cost_usd=0.3),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        rec.reconcile()
        sources = rec.identify_divergence_sources()
        assert any("fee" in s for s in sources)


# ---------------------------------------------------------------------------
# is_backtest_realistic()
# ---------------------------------------------------------------------------

class TestIsBacktestRealistic:
    def test_realistic_when_close(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 100.0, "order_id": "p1", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=95.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        assert rec.is_backtest_realistic(max_divergence_pct=15.0) is True

    def test_unrealistic_when_far(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 100.0, "order_id": "p1", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=50.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        # 50% divergence > 15% threshold
        assert rec.is_backtest_realistic(max_divergence_pct=15.0) is False

    def test_custom_threshold(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        paper_trades = [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 100.0, "order_id": "p1", "symbol": "BTC/USD"},
        ]
        _create_paper_ledger(tmp_ledger, paper_trades)

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=50.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        # With a 60% threshold it should pass
        assert rec.is_backtest_realistic(max_divergence_pct=60.0) is True

    def test_empty_is_realistic(self, tmp_ledger):
        _create_paper_ledger(tmp_ledger, [])
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": []})
        assert rec.is_backtest_realistic() is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_missing_ledger_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.db")
        rec = PaperBacktestReconciler(path, {"all_trades": []})
        report = rec.reconcile()
        assert report.trade_count_paper == 0

    def test_backtest_trades_as_objects(self, tmp_ledger):
        """BacktestResult may contain Trade dataclass instances, not dicts."""
        from dataclasses import dataclass

        @dataclass
        class FakeTrade:
            entry_time: float = 1_700_000_000.0
            exit_time: float = 1_700_003_600.0
            symbol: str = "BTC/USD"
            side: str = "BUY"
            entry_price: float = 50000.0
            exit_price: float = 51000.0
            quantity: float = 0.01
            pnl_usd: float = 10.0
            impact_cost_usd: float = 0.3
            exit_reason: str = "take_profit"

        _create_paper_ledger(tmp_ledger, [
            {"timestamp": 1_700_000_000.0, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
        ])

        rec = PaperBacktestReconciler(
            tmp_ledger, {"all_trades": [FakeTrade()]}
        )
        report = rec.reconcile()
        assert report.trade_count_backtest == 1
        assert report.signal_agreement_pct == pytest.approx(100.0)

    def test_holding_period_computation(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        _create_paper_ledger(tmp_ledger, [])

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 7200, pnl_usd=10.0),  # 2 hours
            _make_bt_trade(base_ts + 10000, base_ts + 10000 + 3600, pnl_usd=5.0),  # 1 hour
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        report = rec.reconcile()
        # avg holding = (7200 + 3600) / 2 = 5400
        assert report.avg_holding_period_backtest == pytest.approx(5400.0)

    def test_different_symbols_not_matched(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        _create_paper_ledger(tmp_ledger, [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": 10.0, "order_id": "p1", "symbol": "BTC/USD"},
        ])

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=2000.0, pnl_usd=5.0, symbol="ETH/USD"),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        report = rec.reconcile()
        # Different symbols should not match
        assert report.signal_agreement_pct == pytest.approx(0.0)
        unmatched = [d for d in report.divergent_trades if d["type"] == "unmatched"]
        assert len(unmatched) == 2  # one paper, one BT

    def test_pnl_divergence_with_negative_values(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        _create_paper_ledger(tmp_ledger, [
            {"timestamp": base_ts, "side": "BUY", "price": 50000.0,
             "size": 0.01, "pnl": -50.0, "order_id": "p1", "symbol": "BTC/USD"},
        ])

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=50000.0, pnl_usd=-30.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        report = rec.reconcile()
        # |(-50)-(-30)| / max(50,30) * 100 = 20/50*100 = 40%
        assert report.pnl_divergence_pct == pytest.approx(40.0)

    def test_zero_price_no_crash(self, tmp_ledger):
        base_ts = 1_700_000_000.0
        _create_paper_ledger(tmp_ledger, [
            {"timestamp": base_ts, "side": "BUY", "price": 0.0,
             "size": 0.0, "pnl": 0.0, "order_id": "p1", "symbol": "BTC/USD"},
        ])

        bt_trades = [
            _make_bt_trade(base_ts, base_ts + 3600, side="BUY",
                           entry_price=0.0, pnl_usd=0.0),
        ]
        rec = PaperBacktestReconciler(tmp_ledger, {"all_trades": bt_trades})
        report = rec.reconcile()
        assert report.slippage_impact_bps == 0.0
