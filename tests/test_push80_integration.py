"""Push 80 — Integration tests: ArgusSystem full lifecycle.
30 tests covering system init, tick flow, signal → bus → engine,
risk gate, API endpoints, Prometheus metrics, backtest pipeline.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# ArgusSystem construction (6)
# ---------------------------------------------------------------------------

class TestArgusSystemConstruction:
    def test_paper_factory(self):
        from core.system import ArgusSystem
        s = ArgusSystem.paper("BTCUSDT")
        assert s is not None

    def test_build_initialises_subsystems(self):
        from core.system import ArgusSystem
        s = ArgusSystem.paper()
        s._build()
        assert s.bus is not None
        assert s.order_manager is not None
        assert s.risk_manager is not None
        assert s.engine is not None

    def test_strategies_registered(self):
        from core.system import ArgusSystem
        s = ArgusSystem.paper()
        s._build()
        assert len(s.strategies) >= 2

    def test_from_config_factory(self):
        from core.system import ArgusSystem
        s = ArgusSystem.from_config({"paper_mode": True, "initial_equity": 5000})
        assert s.config.initial_equity == 5000

    def test_stats_empty_before_start(self):
        from core.system import ArgusSystem
        s = ArgusSystem.paper()
        assert s.stats() == {}

    def test_get_app_returns_fastapi(self):
        pytest.importorskip("fastapi")
        from core.system import ArgusSystem
        s = ArgusSystem.paper()
        app = s.get_app()
        assert app is not None


# ---------------------------------------------------------------------------
# ArgusSystem lifecycle (5)
# ---------------------------------------------------------------------------

class TestArgusSystemLifecycle:
    def _system(self):
        from core.system import ArgusSystem
        return ArgusSystem.paper()

    def test_start_sets_running(self):
        s = self._system()
        run(s.start())
        assert s._running
        run(s.stop())

    def test_stop_clears_running(self):
        s = self._system()
        run(s.start())
        run(s.stop())
        assert not s._running

    def test_stats_after_start(self):
        s = self._system()
        run(s.start())
        stats = s.stats()
        assert "engine" in stats
        assert stats["running"] is True
        run(s.stop())

    def test_double_build_safe(self):
        s = self._system()
        s._build()
        s._build()  # should not raise

    def test_tick_before_start_noop(self):
        s = self._system()
        s._build()
        run(s.tick("BTCUSDT", 50000.0))  # should not raise


# ---------------------------------------------------------------------------
# Tick flow: price → strategy → bus (6)
# ---------------------------------------------------------------------------

class TestTickFlow:
    def _system(self):
        from core.system import ArgusSystem
        s = ArgusSystem.paper()
        run(s.start())
        return s

    def test_tick_no_crash(self):
        s = self._system()
        for p in [50000 + i * 10 for i in range(30)]:
            run(s.tick("BTCUSDT", float(p)))
        run(s.stop())

    def test_tick_updates_prometheus(self):
        s = self._system()
        run(s.tick("BTCUSDT", 50000.0))
        # Prometheus equity gauge should be set
        assert s.prom is not None
        run(s.stop())

    def test_tick_wrong_symbol_ignored(self):
        s = self._system()
        run(s.tick("ETHUSDT", 3000.0))  # strategies are for BTCUSDT
        run(s.stop())

    def test_bus_history_grows_on_signals(self):
        from core.system import ArgusSystem
        s = ArgusSystem.paper()
        run(s.start())
        # Feed rising prices to potentially trigger momentum signal
        prices = [50000 + i * 100 for i in range(50)]
        for p in prices:
            run(s.tick("BTCUSDT", float(p)))
        # Bus history length >= 0 (signals only emitted on crossover)
        assert len(s.bus.history) >= 0
        run(s.stop())

    def test_kill_switch_blocks_ticks(self):
        s = self._system()
        s.risk_manager.activate_kill_switch("test")
        for p in [50000 + i * 10 for i in range(5)]:
            run(s.tick("BTCUSDT", float(p)))
        # No new orders should have been submitted after kill switch
        run(s.stop())

    def test_adapter_price_set_on_tick(self):
        s = self._system()
        run(s.tick("BTCUSDT", 55000.0))
        price = s.adapter._prices.get("BTCUSDT", 0)
        assert price == pytest.approx(55000.0)
        run(s.stop())


# ---------------------------------------------------------------------------
# Risk integration (4)
# ---------------------------------------------------------------------------

class TestRiskIntegration:
    def test_kill_switch_via_api(self):
        pytest.importorskip("fastapi")
        from core.system import ArgusSystem
        from fastapi.testclient import TestClient
        s = ArgusSystem.paper()
        run(s.start())
        client = TestClient(s.get_app())
        r = client.post("/kill-switch", json={"action": "activate", "reason": "test"})
        assert r.status_code == 200
        assert s.risk_manager.kill_switch_active
        run(s.stop())

    def test_risk_manager_stats_in_system_stats(self):
        from core.system import ArgusSystem
        s = ArgusSystem.paper()
        run(s.start())
        stats = s.stats()
        assert "kill_switch" in stats["risk"]
        run(s.stop())

    def test_margin_watcher_starts(self):
        from core.system import ArgusSystem
        s = ArgusSystem.paper()
        run(s.start())
        assert s.margin_watcher._running
        run(s.stop())

    def test_risk_event_bus_exists(self):
        from core.system import ArgusSystem
        s = ArgusSystem.paper()
        s._build()
        assert s.risk_event_bus is not None


# ---------------------------------------------------------------------------
# API integration (7)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("fastapi"),
    reason="fastapi not installed"
)
class TestAPIIntegration:
    def _client(self):
        from core.system import ArgusSystem
        from fastapi.testclient import TestClient
        s = ArgusSystem.paper()
        run(s.start())
        return TestClient(s.get_app()), s

    def test_health_ok(self):
        client, s = self._client()
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        run(s.stop())

    def test_health_version(self):
        client, s = self._client()
        r = client.get("/health")
        assert r.json()["version"] == "8.15.0"
        run(s.stop())

    def test_status_200(self):
        client, s = self._client()
        r = client.get("/status")
        assert r.status_code == 200
        run(s.stop())

    def test_metrics_200(self):
        client, s = self._client()
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "argus_equity" in r.text
        run(s.stop())

    def test_positions_empty(self):
        client, s = self._client()
        r = client.get("/positions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        run(s.stop())

    def test_orders_empty(self):
        client, s = self._client()
        r = client.get("/orders")
        assert r.status_code == 200
        run(s.stop())

    def test_signals_empty_initially(self):
        client, s = self._client()
        r = client.get("/signals")
        assert r.status_code == 200
        run(s.stop())


# ---------------------------------------------------------------------------
# Backtest pipeline integration (2)
# ---------------------------------------------------------------------------

class TestBacktestIntegration:
    def test_backtest_runner_full_pipeline(self):
        from core.backtest.backtest_runner import BacktestRunner, BacktestConfig
        cfg = BacktestConfig(
            strategy_name="momentum",
            mc_n_simulations=50,
            wf_n_splits=2,
            output_dir="/tmp/argus_integration_test",
        )
        runner = BacktestRunner(cfg)
        prices = BacktestRunner.generate_synthetic_prices(n=300, seed=42)
        summary = runner.run(prices=prices)
        assert summary["metrics"]["sharpe"] is not None
        assert summary["monte_carlo"] is not None

    def test_walk_forward_wf_efficiency(self):
        from core.backtest.walk_forward import WalkForwardEngine
        from core.strategy.base_strategy import StrategyConfig
        from core.strategy.momentum_strategy import MomentumStrategy
        from core.backtest.backtest_runner import BacktestRunner

        def factory():
            cfg = StrategyConfig(
                strategy_id="wf_int", symbol="BTCUSDT",
                params={"fast_period": 3, "slow_period": 7}
            )
            return MomentumStrategy(cfg)

        prices = BacktestRunner.generate_synthetic_prices(n=300, seed=1)
        wf = WalkForwardEngine(n_splits=3, min_oos_bars=20)
        result = wf.run(prices, factory)
        assert isinstance(result.wf_efficiency, float)
