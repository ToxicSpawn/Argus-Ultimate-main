"""
Push 87 — Tests: TradeLedger + LedgerFillObserver
"""
from __future__ import annotations

import tempfile
import time
import os

import pytest

from execution.fill_tracker import FillTracker
from execution.trade_ledger import TradeLedger, LedgerEntry, PositionState
from execution.ledger_fill_observer import LedgerFillObserver, LedgerFillObserverMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ledger(tmp_path) -> TradeLedger:
    return TradeLedger(db_path=str(tmp_path / "ledger.db"))


def make_tracker(tmp_path) -> FillTracker:
    return FillTracker(
        db_path=str(tmp_path / "fills.db"),
        daily_limit_bps=100.0,
        daily_limit_usd=500.0,
    )


# ---------------------------------------------------------------------------
# TradeLedger unit tests
# ---------------------------------------------------------------------------

class TestTradeLedger:

    def test_post_open_leg_zero_pnl(self, tmp_path):
        ledger = make_ledger(tmp_path)
        entry = ledger.post(
            fill_id="f1", strategy="s1", symbol="BTC/USD",
            side="buy", quantity_usd=1000.0, fill_price=60_000.0,
            exchange="kraken", timestamp=time.time(),
        )
        assert entry.realised_pnl_usd == 0.0
        assert entry.running_pnl_usd == 0.0  # no fee supplied → 0

    def test_post_close_leg_positive_pnl(self, tmp_path):
        ledger = make_ledger(tmp_path)
        ledger.post(
            fill_id="f1", strategy="s1", symbol="BTC/USD",
            side="buy", quantity_usd=1000.0, fill_price=60_000.0,
            exchange="kraken", timestamp=time.time(),
        )
        entry = ledger.post(
            fill_id="f2", strategy="s1", symbol="BTC/USD",
            side="sell", quantity_usd=1000.0, fill_price=66_000.0,
            exchange="kraken", timestamp=time.time(),
        )
        # +10 % on $1000 = +$100 approx
        assert entry.realised_pnl_usd == pytest.approx(100.0, rel=1e-6)

    def test_fees_reduce_running_pnl(self, tmp_path):
        ledger = make_ledger(tmp_path)
        entry = ledger.post(
            fill_id="f1", strategy="s1", symbol="ETH/USD",
            side="buy", quantity_usd=500.0, fill_price=3_000.0,
            fee_usd=1.5, exchange="kraken", timestamp=time.time(),
        )
        assert entry.running_pnl_usd == pytest.approx(-1.5, rel=1e-6)

    def test_realised_pnl_accessor(self, tmp_path):
        ledger = make_ledger(tmp_path)
        assert ledger.realised_pnl("nonexistent") == 0.0
        ledger.post(
            fill_id="f1", strategy="alpha", symbol="SOL/USD",
            side="buy", quantity_usd=200.0, fill_price=150.0,
            exchange="kraken", timestamp=time.time(),
        )
        assert ledger.realised_pnl("alpha") == pytest.approx(0.0, abs=1e-8)

    def test_position_tracking(self, tmp_path):
        ledger = make_ledger(tmp_path)
        ledger.post(
            fill_id="f1", strategy="s1", symbol="BTC/USD",
            side="buy", quantity_usd=1000.0, fill_price=60_000.0,
            exchange="kraken", timestamp=time.time(),
        )
        pos = ledger.get_position("s1", "BTC/USD")
        assert pos.net_qty_usd == pytest.approx(1000.0)
        assert pos.avg_entry_price == pytest.approx(60_000.0)

        ledger.post(
            fill_id="f2", strategy="s1", symbol="BTC/USD",
            side="sell", quantity_usd=500.0, fill_price=65_000.0,
            exchange="kraken", timestamp=time.time(),
        )
        pos = ledger.get_position("s1", "BTC/USD")
        assert pos.net_qty_usd == pytest.approx(500.0)

    def test_flat_position_after_full_close(self, tmp_path):
        ledger = make_ledger(tmp_path)
        ledger.post(
            fill_id="f1", strategy="s1", symbol="BTC/USD",
            side="buy", quantity_usd=1000.0, fill_price=60_000.0,
            exchange="kraken", timestamp=time.time(),
        )
        ledger.post(
            fill_id="f2", strategy="s1", symbol="BTC/USD",
            side="sell", quantity_usd=1000.0, fill_price=62_000.0,
            exchange="kraken", timestamp=time.time(),
        )
        pos = ledger.get_position("s1", "BTC/USD")
        assert pos.is_flat()
        assert len(ledger.open_positions()) == 0

    def test_crash_recovery_replays_state(self, tmp_path):
        """A fresh TradeLedger pointed at an existing DB should restore P&L."""
        db = str(tmp_path / "ledger.db")
        ledger = TradeLedger(db_path=db)
        ledger.post(
            fill_id="f1", strategy="s1", symbol="BTC/USD",
            side="buy", quantity_usd=1000.0, fill_price=60_000.0,
            fee_usd=0.3, exchange="kraken", timestamp=time.time(),
        )
        ledger.post(
            fill_id="f2", strategy="s1", symbol="BTC/USD",
            side="sell", quantity_usd=1000.0, fill_price=63_000.0,
            fee_usd=0.3, exchange="kraken", timestamp=time.time(),
        )
        expected_pnl = ledger.realised_pnl("s1")

        # Simulate restart
        ledger2 = TradeLedger(db_path=db)
        assert ledger2.realised_pnl("s1") == pytest.approx(expected_pnl, rel=1e-6)

    def test_export_csv(self, tmp_path):
        ledger = make_ledger(tmp_path)
        ledger.post(
            fill_id="f1", strategy="s1", symbol="BTC/USD",
            side="buy", quantity_usd=500.0, fill_price=60_000.0,
            exchange="kraken", timestamp=time.time(),
        )
        out = str(tmp_path / "export.csv")
        ledger.export_csv(out)
        assert os.path.exists(out)
        with open(out) as fh:
            content = fh.read()
        assert "fill_id" in content
        assert "f1" in content

    def test_export_json(self, tmp_path):
        import json as _json
        ledger = make_ledger(tmp_path)
        ledger.post(
            fill_id="f1", strategy="s1", symbol="BTC/USD",
            side="buy", quantity_usd=500.0, fill_price=60_000.0,
            exchange="kraken", timestamp=time.time(),
        )
        out = str(tmp_path / "export.json")
        ledger.export_json(out)
        data = _json.load(open(out))
        assert len(data) == 1
        assert data[0]["fill_id"] == "f1"


# ---------------------------------------------------------------------------
# LedgerFillObserver (wrapper) tests
# ---------------------------------------------------------------------------

class TestLedgerFillObserver:

    def test_record_fill_posts_to_ledger(self, tmp_path):
        tracker = make_tracker(tmp_path)
        ledger = make_ledger(tmp_path)
        observer = LedgerFillObserver(tracker, ledger)

        observer.record_fill(
            strategy="trend", symbol="BTC/USD", side="buy",
            expected_price=60_000.0, actual_price=60_100.0,
            quantity_usd=1000.0, exchange="kraken",
        )

        pos = ledger.get_position("trend", "BTC/USD")
        assert pos.net_qty_usd == pytest.approx(1000.0)

    def test_slippage_still_tracked_in_fill_tracker(self, tmp_path):
        tracker = make_tracker(tmp_path)
        ledger = make_ledger(tmp_path)
        observer = LedgerFillObserver(tracker, ledger)

        observer.record_fill(
            strategy="arb", symbol="ETH/USD", side="buy",
            expected_price=3_000.0, actual_price=3_010.0,
            quantity_usd=500.0, exchange="coinbase",
        )

        stats = tracker.get_strategy_stats("arb")
        assert stats["fill_count"] == 1
        assert stats["avg_slippage_bps"] > 0

    def test_proxy_budget_methods(self, tmp_path):
        tracker = make_tracker(tmp_path)
        ledger = make_ledger(tmp_path)
        observer = LedgerFillObserver(tracker, ledger)
        assert observer.is_within_budget("new_strategy") is True

    def test_fee_estimation(self, tmp_path):
        tracker = make_tracker(tmp_path)
        ledger = make_ledger(tmp_path)
        observer = LedgerFillObserver(tracker, ledger, default_fee_bps=3.0)

        observer.record_fill(
            strategy="s1", symbol="BTC/USD", side="buy",
            expected_price=60_000.0, actual_price=60_000.0,
            quantity_usd=1000.0,
        )
        # fee = 1000 * 3 / 10000 = 0.30
        pnl = ledger.realised_pnl("s1")
        assert pnl == pytest.approx(-0.30, rel=1e-4)

    def test_ledger_failure_does_not_break_fill_tracker(self, tmp_path):
        """If ledger.post raises, FillRecord is still returned."""
        tracker = make_tracker(tmp_path)

        class BrokenLedger(TradeLedger):
            def post(self, **kwargs):  # type: ignore[override]
                raise RuntimeError("DB offline")

        ledger = BrokenLedger(db_path=str(tmp_path / "broken.db"))
        observer = LedgerFillObserver(tracker, ledger)

        record = observer.record_fill(
            strategy="s", symbol="BTC/USD", side="buy",
            expected_price=60_000.0, actual_price=60_000.0,
            quantity_usd=100.0,
        )
        assert record is not None
        assert record.fill_id is not None


# ---------------------------------------------------------------------------
# LedgerFillObserverMixin tests
# ---------------------------------------------------------------------------

class TestLedgerFillObserverMixin:

    def test_mixin_posts_to_ledger(self, tmp_path):
        class InstrumentedTracker(LedgerFillObserverMixin, FillTracker):
            pass

        ledger = make_ledger(tmp_path)
        tracker = InstrumentedTracker(
            ledger=ledger,
            db_path=str(tmp_path / "fills.db"),
            daily_limit_bps=100.0,
            daily_limit_usd=500.0,
        )
        tracker.record_fill(
            strategy="mix", symbol="SOL/USD", side="buy",
            expected_price=150.0, actual_price=151.0,
            quantity_usd=300.0, exchange="kraken",
        )
        pos = ledger.get_position("mix", "SOL/USD")
        assert pos.net_qty_usd == pytest.approx(300.0)
