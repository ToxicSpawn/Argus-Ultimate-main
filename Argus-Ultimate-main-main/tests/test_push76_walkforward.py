"""Push 76 — Tests: BacktestMetrics, WalkForwardEngine,
MonteCarloSimulator, BacktestRunner. 26 tests.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_prices(n=300):
    from core.backtest.backtest_runner import BacktestRunner
    return BacktestRunner.generate_synthetic_prices(n=n, seed=99)


def _flat_prices(n=100, val=50000.0):
    return [val] * n


def _equity_from_prices(prices, growth=0.001):
    """Simple upward equity curve."""
    eq = [10000.0]
    for _ in prices:
        eq.append(eq[-1] * (1 + growth))
    return eq


# ---------------------------------------------------------------------------
# BacktestMetrics (6)
# ---------------------------------------------------------------------------

class TestBacktestMetrics:
    def test_returns_dataclass(self):
        from core.backtest.metrics import compute_metrics
        eq = _equity_from_prices(range(100))
        m = compute_metrics(eq)
        assert m is not None

    def test_sharpe_positive_on_rising_equity(self):
        from core.backtest.metrics import compute_metrics
        eq = _equity_from_prices(range(252))
        m = compute_metrics(eq)
        assert m.sharpe > 0

    def test_max_drawdown_zero_on_monotone(self):
        from core.backtest.metrics import compute_metrics
        eq = [10000 * (1.001 ** i) for i in range(100)]
        m = compute_metrics(eq)
        assert m.max_drawdown_pct == pytest.approx(0.0, abs=1e-6)

    def test_win_rate_from_trades(self):
        from core.backtest.metrics import compute_metrics
        eq = _equity_from_prices(range(100))
        pnls = [100, -50, 200, -30, 80]
        m = compute_metrics(eq, trade_pnls=pnls)
        assert m.win_rate == pytest.approx(0.6)

    def test_to_dict_has_keys(self):
        from core.backtest.metrics import compute_metrics
        m = compute_metrics(_equity_from_prices(range(50)))
        d = m.to_dict()
        assert "sharpe" in d and "max_drawdown_pct" in d

    def test_insufficient_data(self):
        from core.backtest.metrics import compute_metrics
        m = compute_metrics([10000.0])
        assert m.sharpe == 0.0


# ---------------------------------------------------------------------------
# WalkForwardEngine (8)
# ---------------------------------------------------------------------------

class TestWalkForwardEngine:
    def _factory(self):
        from core.strategy.base_strategy import StrategyConfig
        from core.strategy.momentum_strategy import MomentumStrategy
        cfg = StrategyConfig(
            strategy_id="wf_test", symbol="BTCUSDT",
            params={"fast_period": 3, "slow_period": 7}
        )
        return MomentumStrategy(cfg)

    def test_instantiates(self):
        from core.backtest.walk_forward import WalkForwardEngine
        wf = WalkForwardEngine(n_splits=3)
        assert wf is not None

    def test_make_splits_count(self):
        from core.backtest.walk_forward import WalkForwardEngine
        wf = WalkForwardEngine(n_splits=5, min_oos_bars=5)
        splits = wf._make_splits(500)
        assert len(splits) == 5

    def test_run_returns_result(self):
        from core.backtest.walk_forward import WalkForwardEngine
        prices = _synthetic_prices(300)
        wf = WalkForwardEngine(n_splits=3, min_oos_bars=20)
        result = wf.run(prices, self._factory)
        assert result is not None

    def test_result_has_windows(self):
        from core.backtest.walk_forward import WalkForwardEngine
        prices = _synthetic_prices(300)
        wf = WalkForwardEngine(n_splits=3, min_oos_bars=20)
        result = wf.run(prices, self._factory)
        assert len(result.windows) > 0

    def test_wf_efficiency_computed(self):
        from core.backtest.walk_forward import WalkForwardEngine
        prices = _synthetic_prices(300)
        wf = WalkForwardEngine(n_splits=3, min_oos_bars=20)
        result = wf.run(prices, self._factory)
        assert isinstance(result.wf_efficiency, float)

    def test_to_dict_serialisable(self):
        import json
        from core.backtest.walk_forward import WalkForwardEngine
        prices = _synthetic_prices(300)
        wf = WalkForwardEngine(n_splits=3, min_oos_bars=20)
        result = wf.run(prices, self._factory)
        d = result.to_dict()
        assert json.dumps(d)  # no serialisation error

    def test_anchored_mode(self):
        from core.backtest.walk_forward import WalkForwardEngine
        prices = _synthetic_prices(300)
        wf = WalkForwardEngine(n_splits=3, min_oos_bars=20, anchored=True)
        result = wf.run(prices, self._factory)
        assert result.mode == "anchored"

    def test_too_few_bars_raises(self):
        from core.backtest.walk_forward import WalkForwardEngine
        wf = WalkForwardEngine(n_splits=5, min_oos_bars=100)
        with pytest.raises(ValueError):
            wf.run([1.0] * 10, self._factory)


# ---------------------------------------------------------------------------
# MonteCarloSimulator (7)
# ---------------------------------------------------------------------------

class TestMonteCarloSimulator:
    def test_instantiates(self):
        from core.backtest.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=100, seed=42)
        assert mc is not None

    def test_run_returns_result(self):
        from core.backtest.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=100, seed=42)
        eq = _equity_from_prices(range(100))
        r = mc.run(eq)
        assert r is not None

    def test_ruin_prob_in_range(self):
        from core.backtest.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=200, seed=42)
        eq = _equity_from_prices(range(100))
        r = mc.run(eq)
        assert 0.0 <= r.ruin_probability <= 1.0

    def test_p5_less_than_p95_equity(self):
        from core.backtest.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=500, seed=42)
        eq = _equity_from_prices(range(100))
        r = mc.run(eq)
        assert r.p5_final_equity <= r.p95_final_equity

    def test_to_dict_has_ruin_prob(self):
        from core.backtest.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=100, seed=1)
        r = mc.run(_equity_from_prices(range(50)))
        d = r.to_dict()
        assert "ruin_probability" in d

    def test_percentile_table_keys(self):
        from core.backtest.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=100, seed=1)
        r = mc.run(_equity_from_prices(range(50)))
        assert "final_equity" in r.percentile_table
        assert "sharpe" in r.percentile_table

    def test_insufficient_equity_raises(self):
        from core.backtest.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=10)
        with pytest.raises(ValueError):
            mc.run([10000.0])


# ---------------------------------------------------------------------------
# BacktestRunner (5)
# ---------------------------------------------------------------------------

class TestBacktestRunner:
    def test_instantiates(self):
        from core.backtest.backtest_runner import BacktestRunner
        r = BacktestRunner()
        assert r is not None

    def test_synthetic_prices_generated(self):
        from core.backtest.backtest_runner import BacktestRunner
        p = BacktestRunner.generate_synthetic_prices(n=100, seed=0)
        assert len(p) == 100
        assert all(v > 0 for v in p)

    def test_run_returns_summary(self):
        from core.backtest.backtest_runner import BacktestRunner, BacktestConfig
        cfg = BacktestConfig(
            strategy_name="momentum",
            mc_n_simulations=50,
            wf_n_splits=2,
            output_dir="/tmp/argus_test_reports",
        )
        runner = BacktestRunner(cfg)
        prices = BacktestRunner.generate_synthetic_prices(n=300, seed=7)
        summary = runner.run(prices=prices)
        assert "metrics" in summary

    def test_summary_has_mc(self):
        from core.backtest.backtest_runner import BacktestRunner, BacktestConfig
        cfg = BacktestConfig(mc_n_simulations=50, output_dir="/tmp/argus_test_reports")
        runner = BacktestRunner(cfg)
        prices = BacktestRunner.generate_synthetic_prices(n=300)
        summary = runner.run(prices=prices)
        assert summary["monte_carlo"] is not None

    def test_too_few_bars_raises(self):
        from core.backtest.backtest_runner import BacktestRunner
        runner = BacktestRunner()
        with pytest.raises(ValueError):
            runner.run(prices=[50000.0] * 10)
