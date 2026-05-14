"""Push 54 — Live P&L tracker + session stats: 26 tests."""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# TradeRecord tests (7)
# ---------------------------------------------------------------------------
from core.pnl.trade_record import TradeRecord


def _trade(
    side="long", entry=100.0, exit_=110.0, qty=1.0,
    fee_bps=2.0, symbol="BTCUSDT"
):
    now = datetime.now(timezone.utc)
    return TradeRecord(
        symbol=symbol, side=side,
        entry_price=entry, exit_price=exit_, qty=qty,
        entry_time=now, exit_time=now + timedelta(minutes=5),
        fee_bps=fee_bps,
    )


class TestTradeRecord:
    def test_long_gross_pnl(self):
        t = _trade(side="long", entry=100, exit_=110, qty=1)
        assert t.gross_pnl == pytest.approx(10.0)

    def test_short_gross_pnl(self):
        t = _trade(side="short", entry=110, exit_=100, qty=1)
        assert t.gross_pnl == pytest.approx(10.0)

    def test_short_loss(self):
        t = _trade(side="short", entry=100, exit_=110, qty=1)
        assert t.gross_pnl == pytest.approx(-10.0)

    def test_fee_cost_positive(self):
        t = _trade(entry=100, exit_=110, qty=1, fee_bps=2)
        assert t.fee_cost > 0

    def test_net_pnl_less_than_gross(self):
        t = _trade(side="long", entry=100, exit_=110, qty=1)
        assert t.net_pnl < t.gross_pnl

    def test_is_winner_true(self):
        t = _trade(side="long", entry=100, exit_=110)
        assert t.is_winner is True

    def test_is_winner_false(self):
        t = _trade(side="long", entry=110, exit_=100)
        assert t.is_winner is False

    def test_duration_seconds(self):
        now = datetime.now(timezone.utc)
        t = TradeRecord(
            "X", "long", 100, 110, 1,
            now, now + timedelta(seconds=300), 2.0
        )
        assert t.duration_seconds == pytest.approx(300.0)

    def test_return_pct_positive(self):
        t = _trade(side="long", entry=100, exit_=110, qty=1)
        assert t.return_pct > 0

    def test_to_dict_keys(self):
        t = _trade()
        d = t.to_dict()
        assert "net_pnl" in d and "is_winner" in d


# ---------------------------------------------------------------------------
# RunningDrawdown tests (5)
# ---------------------------------------------------------------------------
from core.pnl.drawdown import RunningDrawdown


class TestRunningDrawdown:
    def test_no_drawdown_on_rising_equity(self):
        dd = RunningDrawdown()
        for v in [100, 110, 120, 130]:
            dd.update(v)
        assert dd.max_dd == pytest.approx(0.0)

    def test_drawdown_after_peak(self):
        dd = RunningDrawdown()
        dd.update(100)
        dd.update(80)
        assert dd.current_dd == pytest.approx(0.2)

    def test_max_dd_tracks_worst(self):
        dd = RunningDrawdown()
        for v in [100, 80, 90, 60, 70]:
            dd.update(v)
        assert dd.max_dd == pytest.approx(0.4)

    def test_hwm_updates(self):
        dd = RunningDrawdown()
        dd.update(50)
        dd.update(100)
        assert dd.hwm == pytest.approx(100)

    def test_reset_clears_state(self):
        dd = RunningDrawdown()
        dd.update(100)
        dd.update(50)
        dd.reset()
        assert dd.max_dd == 0.0 and dd.hwm == float("-inf")


# ---------------------------------------------------------------------------
# SessionStats tests (6)
# ---------------------------------------------------------------------------
from core.pnl.session_stats import SessionStats


class TestSessionStats:
    def _make_trades(self, n_win=3, n_loss=1):
        trades = []
        for i in range(n_win):
            trades.append(_trade(side="long", entry=100, exit_=110))
        for i in range(n_loss):
            trades.append(_trade(side="long", entry=110, exit_=100))
        return trades

    def test_empty_trades_zero_stats(self):
        s = SessionStats.from_trades([])
        assert s.n_trades == 0 and s.net_pnl == 0.0

    def test_n_trades_correct(self):
        s = SessionStats.from_trades(self._make_trades(3, 1))
        assert s.n_trades == 4

    def test_win_rate(self):
        s = SessionStats.from_trades(self._make_trades(3, 1))
        assert s.win_rate == pytest.approx(0.75)

    def test_net_pnl_positive(self):
        s = SessionStats.from_trades(self._make_trades(3, 1))
        assert s.net_pnl > 0

    def test_max_drawdown_in_range(self):
        s = SessionStats.from_trades(self._make_trades(1, 3))
        assert 0 <= s.max_drawdown <= 1.0

    def test_profit_factor_gt_one_for_profitable(self):
        s = SessionStats.from_trades(self._make_trades(4, 1))
        assert s.profit_factor > 1.0

    def test_pretty_str_contains_win_rate(self):
        s = SessionStats.from_trades(self._make_trades(2, 2))
        assert "Win Rate" in s.pretty_str()

    def test_to_dict_has_all_keys(self):
        s = SessionStats.from_trades(self._make_trades(2, 1))
        d = s.to_dict()
        for k in ["n_trades", "net_pnl", "sharpe_ratio", "max_drawdown"]:
            assert k in d


# ---------------------------------------------------------------------------
# PnLTracker tests (8)
# ---------------------------------------------------------------------------
from core.pnl.pnl_tracker import PnLTracker


class TestPnLTracker:
    def test_open_and_close_returns_record(self):
        t = PnLTracker()
        t.open_position("BTCUSDT", "long", 65000.0, 0.01)
        record = t.close_position("BTCUSDT", 65500.0)
        assert record is not None
        assert record.net_pnl != 0

    def test_close_nonexistent_returns_none(self):
        t = PnLTracker()
        assert t.close_position("MISSING", 100.0) is None

    def test_equity_accumulates(self):
        t = PnLTracker(fee_bps=0)
        t.open_position("BTC", "long", 100.0, 1.0)
        t.close_position("BTC", 110.0)
        assert t.equity == pytest.approx(10.0)

    def test_open_symbols_tracked(self):
        t = PnLTracker()
        t.open_position("BTC", "long", 100, 1)
        assert "BTC" in t.open_symbols

    def test_open_symbols_cleared_on_close(self):
        t = PnLTracker()
        t.open_position("BTC", "long", 100, 1)
        t.close_position("BTC", 110)
        assert "BTC" not in t.open_symbols

    def test_session_stats_after_trade(self):
        t = PnLTracker(fee_bps=0)
        t.open_position("ETH", "long", 3000.0, 1.0)
        t.close_position("ETH", 3100.0)
        stats = t.session_stats()
        assert stats.n_trades == 1
        assert stats.n_winners == 1

    def test_unrealised_pnl(self):
        t = PnLTracker()
        t.open_position("BTC", "long", 65000, 1.0)
        upnl = t.running_unrealised_pnl({"BTC": 65500})
        assert upnl == pytest.approx(500.0)

    def test_reset_clears_all(self):
        t = PnLTracker()
        t.open_position("BTC", "long", 100, 1)
        t.close_position("BTC", 110)
        t.reset()
        assert t.equity == 0.0
        assert len(t.closed_trades) == 0
