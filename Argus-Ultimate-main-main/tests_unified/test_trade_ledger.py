"""
Push 86 — Tests for core/trade_ledger.py
"""

import os
import tempfile
import time
import pytest
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.trade_ledger import TradeLedger, Fill


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger(tmp_path):
    """Fresh in-memory-ish ledger backed by a temp file."""
    db = tmp_path / "test_ledger.db"
    return TradeLedger(db)


def _make_fill(**kwargs) -> Fill:
    defaults = dict(
        symbol="BTC/USDT",
        side="buy",
        qty=0.1,
        price=30_000.0,
        fee=3.0,
        fee_currency="USDT",
        exchange="binance",
        order_id="ord-001",
        strategy="sma_cross",
        tags='{"source": "test"}',
    )
    defaults.update(kwargs)
    return Fill(**defaults)


# ---------------------------------------------------------------------------
# Construction & initialisation
# ---------------------------------------------------------------------------

def test_ledger_initialises(ledger):
    assert ledger.count() == 0


def test_ledger_repr(ledger):
    assert "TradeLedger" in repr(ledger)
    assert "0" in repr(ledger)


# ---------------------------------------------------------------------------
# Record single fill
# ---------------------------------------------------------------------------

def test_record_returns_row_id(ledger):
    fill = _make_fill()
    row_id = ledger.record(fill)
    assert isinstance(row_id, int)
    assert row_id >= 1


def test_record_increments_count(ledger):
    ledger.record(_make_fill())
    assert ledger.count() == 1
    ledger.record(_make_fill(order_id="ord-002"))
    assert ledger.count() == 2


def test_record_many(ledger):
    fills = [_make_fill(order_id=f"ord-{i}") for i in range(10)]
    n = ledger.record_many(fills)
    assert n == 10
    assert ledger.count() == 10


# ---------------------------------------------------------------------------
# Fill validation
# ---------------------------------------------------------------------------

def test_fill_invalid_side_raises():
    with pytest.raises(AssertionError):
        _make_fill(side="hold")


def test_fill_zero_qty_raises():
    with pytest.raises(AssertionError):
        _make_fill(qty=0.0)


def test_fill_zero_price_raises():
    with pytest.raises(AssertionError):
        _make_fill(price=0.0)


# ---------------------------------------------------------------------------
# Fill properties
# ---------------------------------------------------------------------------

def test_fill_notional():
    f = _make_fill(qty=0.5, price=40_000.0)
    assert f.notional == pytest.approx(20_000.0)


def test_fill_net_value_buy():
    f = _make_fill(qty=0.1, price=30_000.0, fee=3.0, side="buy")
    # -(qty*price + fee) = -(3000 + 3) = -3003
    assert f.net_value == pytest.approx(-3003.0)


def test_fill_net_value_sell():
    f = _make_fill(qty=0.1, price=30_000.0, fee=3.0, side="sell")
    # qty*price - fee = 3000 - 3 = 2997
    assert f.net_value == pytest.approx(2997.0)


def test_fill_auto_timestamp(ledger):
    before = int(time.time() * 1000)
    f = _make_fill()
    after = int(time.time() * 1000)
    assert before <= f.timestamp_ms <= after


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def test_get_fills_returns_all(ledger):
    for i in range(5):
        ledger.record(_make_fill(order_id=f"ord-{i}"))
    fills = ledger.get_fills()
    assert len(fills) == 5


def test_get_fills_filter_symbol(ledger):
    ledger.record(_make_fill(symbol="BTC/USDT", order_id="b1"))
    ledger.record(_make_fill(symbol="ETH/USDT", order_id="e1"))
    fills = ledger.get_fills(symbol="ETH/USDT")
    assert len(fills) == 1
    assert fills[0]["symbol"] == "ETH/USDT"


def test_get_fills_filter_strategy(ledger):
    ledger.record(_make_fill(strategy="sma_cross", order_id="s1"))
    ledger.record(_make_fill(strategy="rsi_mean", order_id="s2"))
    fills = ledger.get_fills(strategy="rsi_mean")
    assert len(fills) == 1


def test_get_fills_filter_since_until(ledger):
    ts1 = 1_700_000_000_000
    ts2 = 1_700_000_060_000
    ts3 = 1_700_000_120_000
    ledger.record(_make_fill(timestamp_ms=ts1, order_id="a"))
    ledger.record(_make_fill(timestamp_ms=ts2, order_id="b"))
    ledger.record(_make_fill(timestamp_ms=ts3, order_id="c"))
    fills = ledger.get_fills(since_ms=ts1, until_ms=ts2)
    assert len(fills) == 2


# ---------------------------------------------------------------------------
# PnL report
# ---------------------------------------------------------------------------

def test_pnl_report_empty(ledger):
    report = ledger.pnl_report(symbol="DOGE/USDT")
    assert "error" in report


def test_pnl_report_basic(ledger):
    ledger.record(_make_fill(side="buy",  qty=0.1, price=30_000.0, fee=3.0, order_id="b"))
    ledger.record(_make_fill(side="sell", qty=0.1, price=32_000.0, fee=3.2, order_id="s"))
    report = ledger.pnl_report(symbol="BTC/USDT")
    assert report["total_fills"] == 2
    assert report["buy_count"] == 1
    assert report["sell_count"] == 1
    # net = -(3000+3) + (3200-3.2) = -3003 + 3196.8 = 193.8
    assert report["realised_pnl_usd"] == pytest.approx(193.8)


def test_pnl_report_fees(ledger):
    ledger.record(_make_fill(fee=10.0, order_id="f1"))
    ledger.record(_make_fill(fee=5.0, order_id="f2"))
    report = ledger.pnl_report()
    assert report["total_fees_usd"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def test_equity_curve_empty(ledger):
    curve = ledger.equity_curve()
    assert curve == []


def test_equity_curve_monotonic_sells(ledger):
    for i in range(5):
        ledger.record(_make_fill(
            side="sell", qty=0.1, price=30_000.0, fee=0.0,
            order_id=f"sell-{i}",
            timestamp_ms=1_700_000_000_000 + i * 60_000
        ))
    curve = ledger.equity_curve()
    assert len(curve) == 5
    # each sell adds +3000 (fee=0), so curve should be strictly increasing
    pnls = [c["cumulative_pnl"] for c in curve]
    assert pnls == sorted(pnls)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def test_symbols(ledger):
    ledger.record(_make_fill(symbol="BTC/USDT", order_id="b"))
    ledger.record(_make_fill(symbol="SOL/USDT", order_id="s"))
    syms = ledger.symbols()
    assert set(syms) == {"BTC/USDT", "SOL/USDT"}


def test_strategies(ledger):
    ledger.record(_make_fill(strategy="strat_a", order_id="a"))
    ledger.record(_make_fill(strategy="strat_b", order_id="b"))
    strats = ledger.strategies()
    assert set(strats) == {"strat_a", "strat_b"}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_csv(ledger, tmp_path):
    ledger.record(_make_fill())
    out = tmp_path / "out.csv"
    n = ledger.export_csv(out)
    assert n == 1
    assert out.exists()
    content = out.read_text()
    assert "BTC/USDT" in content


def test_export_json(ledger, tmp_path):
    ledger.record(_make_fill())
    out = tmp_path / "out.json"
    n = ledger.export_json(out)
    assert n == 1
    assert out.exists()
    import json
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert data[0]["symbol"] == "BTC/USDT"


def test_export_empty_csv(ledger, tmp_path):
    out = tmp_path / "empty.csv"
    n = ledger.export_csv(out)
    assert n == 0
