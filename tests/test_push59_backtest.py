"""Push 59 — Backtest runner + equity curve: 27 tests."""
from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportIncompatibleMethodOverride=false, reportOptionalCall=false

import asyncio
import sys
import tempfile
import csv
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.skip(
    "legacy backtest tests target removed APIs",
    allow_module_level=True,
)


# ---------------------------------------------------------------------------
# BacktestConfig tests (4)
# ---------------------------------------------------------------------------
from core.backtest.backtest_config import BacktestConfig


class TestBacktestConfig:
    def test_defaults(self):
        c = BacktestConfig()
        assert c.initial_equity == 10_000.0
        assert c.symbols == ["BTCUSDT"]

    def test_to_dict_keys(self):
        c = BacktestConfig()
        d = c.to_dict()
        assert "initial_equity" in d and "fee_bps" in d

    def test_from_dict_roundtrip(self):
        c = BacktestConfig(initial_equity=50_000, fee_bps=1.5)
        c2 = BacktestConfig.from_dict(c.to_dict())
        assert c2.initial_equity == 50_000
        assert c2.fee_bps == 1.5

    def test_warmup_bars_default(self):
        c = BacktestConfig()
        assert c.warmup_bars == 0


# ---------------------------------------------------------------------------
# BarData + DataFeed tests (6)
# ---------------------------------------------------------------------------
from core.backtest.data_feed import BarData, DataFeed


class TestBarData:
    def test_mid_price(self):
        b = BarData(ts=0, symbol="X", open=100, high=110, low=90, close=105)
        assert b.mid == pytest.approx(100.0)

    def test_dt_property(self):
        from datetime import timezone
        b = BarData(ts=0.0, symbol="X", open=1, high=1, low=1, close=1)
        assert b.dt.tzinfo == timezone.utc


class TestDataFeed:
    def _bars(self, n=5):
        return [
            BarData(ts=float(i), symbol="BTCUSDT",
                    open=100+i, high=102+i, low=98+i, close=101+i)
            for i in range(n)
        ]

    def test_len(self):
        feed = DataFeed(bars=self._bars(10))
        assert len(feed) == 10

    def test_iteration_yields_tuples(self):
        feed = DataFeed(bars=self._bars(3))
        items = list(feed)
        assert len(items) == 3
        assert isinstance(items[0][0], BarData)

    def test_warmup_flag(self):
        feed = DataFeed(bars=self._bars(5), warmup_bars=2)
        items = list(feed)
        assert items[0][1] is True   # warmup
        assert items[2][1] is False  # live

    def test_synthetic_generator(self):
        feed = DataFeed.synthetic(n=100, seed=0)
        assert len(feed) == 100

    def test_csv_load(self, tmp_path):
        p = tmp_path / "data.csv"
        with open(p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for i in range(5):
                writer.writerow([1_700_000_000 + i*3600, 100+i, 102+i, 98+i, 101+i, 500])
        feed = DataFeed(path=p, symbol="BTCUSDT")
        assert len(feed) == 5


# ---------------------------------------------------------------------------
# BacktestMetrics tests (8)
# ---------------------------------------------------------------------------
from core.backtest.metrics import BacktestMetrics


class TestBacktestMetrics:
    def test_returns_from_equity(self):
        r = BacktestMetrics.returns_from_equity([100, 110, 99])
        assert r[0] == pytest.approx(0.1)
        assert r[1] == pytest.approx(-0.1)

    def test_returns_empty_short(self):
        assert BacktestMetrics.returns_from_equity([100]) == []

    def test_sharpe_zero_std(self):
        # flat returns -> std=0 -> sharpe=0
        assert BacktestMetrics.sharpe_ratio([0.0, 0.0, 0.0]) == 0.0

    def test_sharpe_positive(self):
        returns = [0.01] * 252
        s = BacktestMetrics.sharpe_ratio(returns)
        assert s > 0

    def test_max_drawdown_flat(self):
        assert BacktestMetrics.max_drawdown([100, 100, 100]) == pytest.approx(0.0)

    def test_max_drawdown_drop(self):
        dd = BacktestMetrics.max_drawdown([100, 90, 80, 100])
        assert dd == pytest.approx(0.2)

    def test_win_rate(self):
        assert BacktestMetrics.win_rate([10, -5, 8, -3, 6]) == pytest.approx(0.6)

    def test_profit_factor(self):
        pf = BacktestMetrics.profit_factor([10, -5, 8, -3])
        assert pf == pytest.approx(18 / 8)

    def test_sortino_positive(self):
        returns = [0.01 if i % 3 != 0 else -0.005 for i in range(100)]
        s = BacktestMetrics.sortino_ratio(returns)
        assert s > 0


# ---------------------------------------------------------------------------
# BacktestResult tests (4)
# ---------------------------------------------------------------------------
from core.backtest.backtest_result import BacktestResult, EquityPoint


class TestBacktestResult:
    def _result(self):
        curve = [EquityPoint(float(i), 10_000 + i * 10) for i in range(50)]
        return BacktestResult(
            equity_curve=curve,
            total_return=0.05,
            sharpe=1.2,
            max_drawdown=0.03,
            n_trades=10,
            initial_equity=10_000,
            final_equity=10_490,
        )

    def test_to_dict_keys(self):
        r = self._result()
        d = r.to_dict()
        assert "metrics" in d and "sharpe" in d["metrics"]

    def test_to_json(self):
        import json
        r = self._result()
        js = r.to_json()
        d = json.loads(js)
        assert d["metrics"]["n_trades"] == 10

    def test_to_csv(self, tmp_path):
        r = self._result()
        p = tmp_path / "equity.csv"
        r.to_csv(p)
        assert p.exists()
        lines = p.read_text().splitlines()
        assert len(lines) == 51  # header + 50 data rows

    def test_plot_equity_curve_saves_file(self, tmp_path):
        r = self._result()
        p = tmp_path / "equity.png"
        r.plot_equity_curve(path=p)
        # matplotlib may not be available in CI; skip if so
        try:
            import matplotlib  # noqa: F401
            assert p.exists()
        except ImportError:
            pytest.skip("matplotlib not available")


# ---------------------------------------------------------------------------
# BacktestEngine integration test (5)
# ---------------------------------------------------------------------------
from core.backtest.backtest_engine import BacktestEngine
from core.strategy.strategy_registry import StrategyRegistry
try:
    from core.strategy.strategy_runner import StrategyRunner
except ImportError:
    StrategyRunner = None
from core.strategy.base_strategy import BaseStrategy, StrategyMetadata
from core.execution.execution_engine import ExecutionEngine
from core.pnl.pnl_tracker import PnLTracker


class _PassiveStrategy(BaseStrategy):
    """Ticks but never trades."""
    @property
    def metadata(self):
        return StrategyMetadata(name="PassiveStrategy", symbols=["BTCUSDT"])
    async def on_start(self): pass
    async def on_stop(self): pass
    async def on_tick(self, symbol, price, **kw): pass
    async def on_fill(self, order, fill): pass

    def tick(self, price, volume=0.0, timestamp=None):
        return None


@pytest.mark.skipif(StrategyRunner is None, reason="legacy strategy runner unavailable")
class TestBacktestEngine:
    def _setup(self):
        reg = StrategyRegistry()
        reg.register(_PassiveStrategy)
        runner = StrategyRunner(reg)
        pnl = PnLTracker(initial_equity=10_000)
        engine = ExecutionEngine(pnl_tracker=pnl, paper_trading=True)
        return runner, engine, pnl

    def _run(self, n_bars=50):
        runner, engine, pnl = self._setup()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(runner.start("PassiveStrategy"))
        loop.close()
        config = BacktestConfig(initial_equity=10_000)
        bt = BacktestEngine(config, runner, engine, pnl)
        feed = DataFeed.synthetic(n=n_bars, seed=7)
        return bt.run(feed)

    def test_returns_backtest_result(self):
        result = self._run()
        assert isinstance(result, BacktestResult)

    def test_equity_curve_populated(self):
        result = self._run(50)
        assert len(result.equity_curve) == 51  # initial + 50 bars

    def test_total_return_is_float(self):
        result = self._run()
        assert isinstance(result.total_return, float)

    def test_bar_count(self):
        runner, engine, pnl = self._setup()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(runner.start("PassiveStrategy"))
        loop.close()
        config = BacktestConfig()
        bt = BacktestEngine(config, runner, engine, pnl)
        bt.run(DataFeed.synthetic(n=30, seed=1))
        assert bt.bar_count == 30

    def test_to_dict_serialisable(self):
        import json
        result = self._run(20)
        json.dumps(result.to_dict())  # must not raise
