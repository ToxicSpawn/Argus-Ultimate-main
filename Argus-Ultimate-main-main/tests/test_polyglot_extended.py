"""
Tests for ARGUS polyglot extended language bridges.

All tests exercise FALLBACK paths only — no compiled binaries needed.
Covers: CudaEngine, CppOrderBook, JuliaSolver, LuaStrategyEngine,
        ElixirSupervisor, ZigCompute, and PolyglotEngine integration.

Run:
    py -m pytest tests/test_polyglot_extended.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. CudaEngine Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCudaEngine:
    """Tests for CUDA GPU engine fallback (numpy)."""

    @pytest.fixture
    def engine(self):
        from multilang.cuda_engine.bridge import CudaEngine
        e = CudaEngine()
        assert e.backend == "fallback"
        return e

    def test_backend_is_fallback(self, engine):
        assert engine.available is False
        assert engine.backend == "fallback"

    def test_monte_carlo_var_basic(self, engine):
        returns = np.random.randn(100, 3) * 0.02
        result = engine.monte_carlo_var(returns, n_scenarios=5000, confidence=0.95)
        assert "var" in result
        assert "cvar" in result
        assert result["confidence"] == 0.95
        assert result["n_scenarios"] == 5000
        assert result["var"] >= 0

    def test_monte_carlo_var_single_asset(self, engine):
        returns = np.random.randn(50) * 0.01
        result = engine.monte_carlo_var(returns, n_scenarios=1000)
        assert "var" in result
        assert result["var"] >= 0

    def test_monte_carlo_var_custom_weights(self, engine):
        returns = np.random.randn(80, 2) * 0.02
        result = engine.monte_carlo_var(returns, weights=[0.7, 0.3])
        assert "var" in result

    def test_monte_carlo_var_empty_returns(self, engine):
        result = engine.monte_carlo_var(np.array([]).reshape(0, 1), n_scenarios=100)
        assert result["var"] == 0.0
        assert result["n_scenarios"] == 0

    def test_monte_carlo_var_cvar_gte_var(self, engine):
        returns = np.random.randn(200, 3) * 0.05
        result = engine.monte_carlo_var(returns, n_scenarios=10000, confidence=0.95)
        assert result["cvar"] >= result["var"] - 1e-10

    def test_batch_ema_basic(self, engine):
        prices = np.random.randn(3, 50).cumsum(axis=1) + 100
        result = engine.batch_ema(prices, periods=[10, 20, 30])
        assert "emas" in result
        assert len(result["emas"]) == 3
        assert len(result["emas"][0]) == 50

    def test_batch_ema_single_period(self, engine):
        prices = np.random.randn(2, 30).cumsum(axis=1) + 100
        result = engine.batch_ema(prices, periods=15)
        assert len(result["emas"]) == 2

    def test_batch_ema_1d_input(self, engine):
        prices = np.random.randn(40).cumsum() + 100
        result = engine.batch_ema(prices, periods=10)
        assert len(result["emas"]) == 1

    def test_correlation_matrix_basic(self, engine):
        matrix = np.random.randn(4, 100)
        result = engine.correlation_matrix(matrix)
        assert "matrix" in result
        corr = np.array(result["matrix"])
        assert corr.shape == (4, 4)
        # Diagonal should be ~1
        for i in range(4):
            assert abs(corr[i, i] - 1.0) < 0.01

    def test_correlation_matrix_single_asset(self, engine):
        matrix = np.random.randn(1, 50)
        result = engine.correlation_matrix(matrix)
        assert result["matrix"] == [[1.0]]

    def test_batch_signal_score(self, engine):
        signals = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        features = np.array([1.0, 2.0, 3.0])
        result = engine.batch_signal_score(signals, features)
        assert "scores" in result
        assert len(result["scores"]) == 3
        expected = signals @ features
        np.testing.assert_allclose(result["scores"], expected.tolist(), atol=1e-10)

    def test_latency_tracking(self, engine):
        assert engine.avg_latency_ms == 0.0
        engine.monte_carlo_var(np.random.randn(10, 2) * 0.01, n_scenarios=100)
        assert engine.avg_latency_ms > 0


# ═══════════════════════════════════════════════════════════════════════
# 2. CppOrderBook Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCppOrderBook:
    """Tests for C++ order book fallback (Python dict)."""

    @pytest.fixture
    def book(self):
        from multilang.workers.cpp_orderbook.bridge import CppOrderBook
        b = CppOrderBook()
        assert b.backend == "fallback"
        return b

    def test_backend_is_fallback(self, book):
        assert book.available is False
        assert book.backend == "fallback"

    def test_update_and_mid_price(self, book):
        book.update("bid", 65000.0, 1.0)
        book.update("ask", 65010.0, 1.0)
        state = book.get_state()
        assert state["mid_price"] == pytest.approx(65005.0)

    def test_spread_bps(self, book):
        book.update("bid", 50000.0, 1.0)
        book.update("ask", 50050.0, 1.0)
        state = book.get_state()
        expected_spread = (50050.0 - 50000.0) / 50025.0 * 10000.0
        assert state["spread_bps"] == pytest.approx(expected_spread, rel=1e-6)

    def test_empty_book_state(self, book):
        state = book.get_state()
        assert state["mid_price"] == 0.0
        assert state["spread_bps"] == 0.0
        assert state["imbalance"] == 0.0
        assert state["bid_levels"] == 0
        assert state["ask_levels"] == 0

    def test_update_remove_level(self, book):
        book.update("bid", 65000.0, 1.0)
        book.update("bid", 65000.0, 0.0)  # Remove
        state = book.get_state()
        assert state["bid_levels"] == 0

    def test_imbalance_bid_heavy(self, book):
        book.update("bid", 65000.0, 10.0)
        book.update("bid", 64999.0, 10.0)
        book.update("ask", 65010.0, 1.0)
        book.update("ask", 65011.0, 1.0)
        state = book.get_state()
        assert state["imbalance"] > 0  # More bid volume

    def test_imbalance_ask_heavy(self, book):
        book.update("bid", 65000.0, 1.0)
        book.update("ask", 65010.0, 10.0)
        book.update("ask", 65011.0, 10.0)
        state = book.get_state()
        assert state["imbalance"] < 0  # More ask volume

    def test_detect_walls(self, book):
        # Normal levels
        for i in range(10):
            book.update("bid", 65000.0 - i, 1.0)
            book.update("ask", 65010.0 + i, 1.0)
        # Add a wall (large order)
        book.update("bid", 64995.0, 50.0)
        state = book.get_state()
        walls = state["walls"]
        assert len(walls) > 0
        bid_walls = [w for w in walls if w["side"] == "bid"]
        assert any(w["price"] == 64995.0 for w in bid_walls)

    def test_vwap_basic(self, book):
        book.update("ask", 65000.0, 1.0)
        book.update("ask", 65010.0, 1.0)
        vwap = book.get_vwap(65000.0)  # Fill exactly one level
        assert vwap == pytest.approx(65000.0)

    def test_vwap_partial_fill(self, book):
        book.update("ask", 100.0, 10.0)
        book.update("ask", 110.0, 10.0)
        # Fill 1500 USD: 10 units at 100 = 1000, then 500/110 units at 110
        vwap = book.get_vwap(1500.0)
        assert vwap > 100.0
        assert vwap < 110.0

    def test_vwap_empty_book(self, book):
        assert book.get_vwap(10000.0) == 0.0

    def test_clear(self, book):
        book.update("bid", 65000.0, 1.0)
        book.update("ask", 65010.0, 1.0)
        book.clear()
        state = book.get_state()
        assert state["bid_levels"] == 0
        assert state["ask_levels"] == 0

    def test_best_bid_ask(self, book):
        book.update("bid", 65000.0, 1.0)
        book.update("bid", 64990.0, 1.0)
        book.update("ask", 65010.0, 1.0)
        book.update("ask", 65020.0, 1.0)
        state = book.get_state()
        assert state["best_bid"] == 65000.0
        assert state["best_ask"] == 65010.0


# ═══════════════════════════════════════════════════════════════════════
# 3. JuliaSolver Tests
# ═══════════════════════════════════════════════════════════════════════

class TestJuliaSolver:
    """Tests for Julia optimization solver fallback (numpy/scipy)."""

    @pytest.fixture
    def solver(self):
        from multilang.workers.julia_solver.bridge import JuliaSolver
        s = JuliaSolver()
        # Julia may or may not be installed; either backend is acceptable
        assert s.backend in ("fallback", "native")
        return s

    def test_backend_detected(self, solver):
        assert solver.backend in ("fallback", "native")
        assert isinstance(solver.available, bool)

    def test_optimize_portfolio_basic(self, solver):
        np.random.seed(42)
        returns = np.random.randn(100, 3) * 0.02
        result = solver.optimize_portfolio(returns, risk_aversion=2.0)
        assert "weights" in result
        assert "expected_return" in result
        assert "expected_risk" in result
        weights = result["weights"]
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.01  # Weights sum to ~1
        assert all(w >= -0.01 for w in weights)  # Weights non-negative

    def test_optimize_portfolio_high_risk_aversion(self, solver):
        np.random.seed(42)
        returns = np.random.randn(100, 3) * 0.02
        result_low = solver.optimize_portfolio(returns, risk_aversion=0.5)
        result_high = solver.optimize_portfolio(returns, risk_aversion=10.0)
        # Higher risk aversion should give lower expected risk
        assert result_high["expected_risk"] <= result_low["expected_risk"] + 0.01

    def test_risk_parity_basic(self, solver):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = solver.risk_parity(cov)
        assert "weights" in result
        assert "risk_contributions" in result
        weights = result["weights"]
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 0.01

    def test_risk_parity_equal_var(self, solver):
        # Equal variance → should give equal weights
        cov = np.eye(3) * 0.04
        result = solver.risk_parity(cov)
        weights = result["weights"]
        # All weights should be approximately equal
        for w in weights:
            assert abs(w - 1.0 / 3) < 0.05

    def test_kelly_optimal_basic(self, solver):
        win_rates = [0.6, 0.55, 0.5]
        payoff_ratios = [1.5, 2.0, 1.0]
        result = solver.kelly_optimal(win_rates, payoff_ratios)
        assert "fractions" in result
        assert "total_allocation" in result
        fractions = result["fractions"]
        assert len(fractions) == 3
        assert all(f >= 0 for f in fractions)
        assert result["total_allocation"] <= 1.0 + 0.01

    def test_kelly_optimal_with_correlation(self, solver):
        win_rates = [0.6, 0.6]
        payoff_ratios = [2.0, 2.0]
        corr_high = [[1.0, 0.9], [0.9, 1.0]]
        corr_low = [[1.0, 0.0], [0.0, 1.0]]
        result_corr = solver.kelly_optimal(win_rates, payoff_ratios, corr_high)
        result_uncorr = solver.kelly_optimal(win_rates, payoff_ratios, corr_low)
        # High correlation should reduce allocations
        total_corr = sum(result_corr["fractions"])
        total_uncorr = sum(result_uncorr["fractions"])
        assert total_corr <= total_uncorr + 0.01

    def test_kelly_zero_payoff(self, solver):
        win_rates = [0.5]
        payoff_ratios = [0.0]
        result = solver.kelly_optimal(win_rates, payoff_ratios)
        assert result["fractions"][0] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 4. LuaStrategyEngine Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLuaStrategyEngine:
    """Tests for Lua strategy scripting engine fallback (Python)."""

    @pytest.fixture
    def engine(self):
        from multilang.workers.lua_scripts.bridge import LuaStrategyEngine
        e = LuaStrategyEngine()
        return e

    def test_backend_is_fallback(self, engine):
        assert engine.backend == "fallback"

    def test_evaluate_no_strategy(self, engine):
        result = engine.evaluate({"close": [100, 101, 102]})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
        assert "no strategy" in result["reason"]

    def test_load_strategy(self, engine):
        strategy_path = Path(__file__).resolve().parent.parent / "multilang" / "workers" / "lua_scripts" / "example_strategy.lua"
        name = engine.load_strategy(str(strategy_path))
        assert name == "example_strategy"
        assert "example_strategy" in engine.list_strategies()

    def test_evaluate_buy_signal(self, engine):
        strategy_path = Path(__file__).resolve().parent.parent / "multilang" / "workers" / "lua_scripts" / "example_strategy.lua"
        engine.load_strategy(str(strategy_path))
        # Create data where fast SMA > slow SMA (uptrend)
        prices = list(range(50, 110))  # Steadily rising
        result = engine.evaluate({"close": prices})
        assert result["action"] == "BUY"
        assert result["confidence"] > 0

    def test_evaluate_sell_signal(self, engine):
        strategy_path = Path(__file__).resolve().parent.parent / "multilang" / "workers" / "lua_scripts" / "example_strategy.lua"
        engine.load_strategy(str(strategy_path))
        # Create data where fast SMA < slow SMA (downtrend)
        prices = list(range(110, 50, -1))  # Steadily falling
        result = engine.evaluate({"close": prices})
        assert result["action"] == "SELL"
        assert result["confidence"] > 0

    def test_evaluate_insufficient_data(self, engine):
        strategy_path = Path(__file__).resolve().parent.parent / "multilang" / "workers" / "lua_scripts" / "example_strategy.lua"
        engine.load_strategy(str(strategy_path))
        result = engine.evaluate({"close": [100]})
        assert result["action"] == "HOLD"

    def test_list_strategies_empty(self, engine):
        assert engine.list_strategies() == []

    def test_hot_reload(self, engine):
        strategy_path = Path(__file__).resolve().parent.parent / "multilang" / "workers" / "lua_scripts" / "example_strategy.lua"
        engine.load_strategy(str(strategy_path))
        assert engine.hot_reload("example_strategy") is True

    def test_hot_reload_unknown(self, engine):
        assert engine.hot_reload("nonexistent") is False

    def test_load_nonexistent_strategy(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.load_strategy("/nonexistent/strategy.lua")


# ═══════════════════════════════════════════════════════════════════════
# 5. ElixirSupervisor Tests
# ═══════════════════════════════════════════════════════════════════════

class TestElixirSupervisor:
    """Tests for Elixir connection supervisor fallback (Python)."""

    @pytest.fixture
    def supervisor(self):
        from multilang.workers.elixir_supervisor.bridge import ElixirSupervisor
        s = ElixirSupervisor()
        return s

    def test_backend_is_fallback(self, supervisor):
        assert supervisor.backend == "fallback"

    def test_start(self, supervisor):
        result = supervisor.start()
        assert result["started"] is True
        assert len(result["exchanges"]) == 3
        assert "kraken" in result["exchanges"]

    def test_health_before_start(self, supervisor):
        health = supervisor.get_health()
        for exchange in ["kraken", "coinbase", "bybit"]:
            assert health[exchange]["status"] == "not_started"

    def test_health_after_start(self, supervisor):
        supervisor.start()
        health = supervisor.get_health()
        for exchange in ["kraken", "coinbase", "bybit"]:
            assert health[exchange]["status"] == "connected"
            assert health[exchange]["uptime_pct"] > 0
            assert health[exchange]["restarts"] == 0

    def test_restart_connection(self, supervisor):
        supervisor.start()
        result = supervisor.restart_connection("kraken")
        assert result["restarted"] is True
        assert result["exchange"] == "kraken"
        assert result["restarts"] == 1

    def test_restart_unknown_exchange(self, supervisor):
        supervisor.start()
        result = supervisor.restart_connection("unknown_exchange")
        assert result["restarted"] is False

    def test_uptime_before_start(self, supervisor):
        uptime = supervisor.get_uptime()
        for exchange in ["kraken", "coinbase", "bybit"]:
            assert uptime[exchange]["uptime_pct"] == 0.0

    def test_uptime_after_start(self, supervisor):
        supervisor.start()
        uptime = supervisor.get_uptime()
        for exchange in ["kraken", "coinbase", "bybit"]:
            assert uptime[exchange]["uptime_pct"] > 0
            assert uptime[exchange]["total_seconds"] >= 0

    def test_custom_exchanges(self):
        from multilang.workers.elixir_supervisor.bridge import ElixirSupervisor
        s = ElixirSupervisor(exchanges=["binance", "okx"])
        result = s.start()
        assert set(result["exchanges"]) == {"binance", "okx"}

    def test_multiple_restarts(self, supervisor):
        supervisor.start()
        supervisor.restart_connection("kraken")
        supervisor.restart_connection("kraken")
        result = supervisor.restart_connection("kraken")
        assert result["restarts"] == 3


# ═══════════════════════════════════════════════════════════════════════
# 6. ZigCompute Tests
# ═══════════════════════════════════════════════════════════════════════

class TestZigCompute:
    """Tests for Zig WASM compute fallback (numpy)."""

    @pytest.fixture
    def zc(self):
        from multilang.workers.zig_wasm.bridge import ZigCompute
        z = ZigCompute()
        assert z.backend == "fallback"
        return z

    def test_backend_is_fallback(self, zc):
        assert zc.available is False
        assert zc.backend == "fallback"

    def test_drawdown_basic(self, zc):
        equity = [100, 110, 105, 115, 90, 100]
        dd = zc.drawdown(equity)
        # Max drawdown: peak=115, trough=90 → (115-90)/115 ≈ 0.2174
        assert dd == pytest.approx(25.0 / 115.0, rel=1e-6)

    def test_drawdown_no_drawdown(self, zc):
        equity = [100, 110, 120, 130]
        assert zc.drawdown(equity) == 0.0

    def test_drawdown_single_point(self, zc):
        assert zc.drawdown([100]) == 0.0

    def test_drawdown_empty(self, zc):
        assert zc.drawdown([]) == 0.0

    def test_sharpe_basic(self, zc):
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001  # Slight positive drift
        sr = zc.sharpe(returns)
        assert isinstance(sr, float)
        # Should be modestly positive with positive drift
        assert sr > 0

    def test_sharpe_zero_returns(self, zc):
        returns = [0.0] * 10
        assert zc.sharpe(returns) == 0.0

    def test_sharpe_with_risk_free(self, zc):
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        sr_no_rf = zc.sharpe(returns, risk_free_rate=0.0)
        sr_with_rf = zc.sharpe(returns, risk_free_rate=0.01)
        assert sr_with_rf < sr_no_rf

    def test_sharpe_short_array(self, zc):
        assert zc.sharpe([0.01]) == 0.0

    def test_sortino_basic(self, zc):
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01 + 0.001
        so = zc.sortino(returns)
        assert isinstance(so, float)

    def test_sortino_all_positive(self, zc):
        returns = [0.01, 0.02, 0.015, 0.025]
        so = zc.sortino(returns)
        assert so == 99.99  # No downside

    def test_sortino_vs_sharpe(self, zc):
        # Sortino should generally be >= Sharpe for same data with positive skew
        np.random.seed(42)
        returns = np.abs(np.random.randn(100)) * 0.01 + 0.001
        sr = zc.sharpe(returns)
        so = zc.sortino(returns)
        # Both should be positive for positive returns
        assert sr > 0
        assert so > 0

    def test_calmar_basic(self, zc):
        cr = zc.calmar(0.20, 0.10)
        assert cr == pytest.approx(2.0)

    def test_calmar_zero_drawdown(self, zc):
        cr = zc.calmar(0.15, 0.0)
        assert cr == 99.99

    def test_calmar_negative_return(self, zc):
        cr = zc.calmar(-0.10, 0.20)
        assert cr == pytest.approx(-0.5)

    def test_calmar_zero_return_zero_dd(self, zc):
        cr = zc.calmar(0.0, 0.0)
        assert cr == 0.0

    def test_latency_tracking(self, zc):
        assert zc.avg_latency_ms == 0.0
        zc.drawdown([100, 90, 110])
        assert zc.avg_latency_ms > 0


# ═══════════════════════════════════════════════════════════════════════
# 7. PolyglotEngine Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPolyglotEngineExtended:
    """Integration tests for PolyglotEngine with all 10 languages."""

    @pytest.fixture
    def engine(self):
        from core.polyglot_engine import PolyglotEngine
        e = PolyglotEngine()
        e.initialize()
        return e

    def test_all_10_languages_in_status(self, engine):
        status = engine.get_status()
        expected_keys = {
            "rust", "c", "javascript", "go",
            "cuda", "cpp", "julia", "lua", "elixir", "zig",
        }
        assert expected_keys.issubset(set(status.keys()))

    def test_new_engines_loaded(self, engine):
        assert engine.cuda is not None
        assert engine.cpp_orderbook is not None
        assert engine.julia is not None
        assert engine.lua is not None
        assert engine.elixir is not None
        assert engine.zig is not None

    def test_all_backends_valid(self, engine):
        status = engine.get_status()
        for lang in ["cuda", "cpp", "julia", "lua", "elixir", "zig"]:
            assert status[lang]["backend"] in ("fallback", "native", "not_loaded")

    def test_benchmark_includes_new_engines(self, engine):
        bench = engine.benchmark(n=50)
        # Should have entries for the new engines
        new_keys = {"cuda_mc_var", "cpp_orderbook", "julia_optimize",
                     "lua_evaluate", "elixir_health", "zig_sharpe"}
        assert new_keys.issubset(set(bench.keys()))

    def test_benchmark_results_format(self, engine):
        bench = engine.benchmark(n=50)
        for key, result in bench.items():
            assert "native_ms" in result
            assert "fallback_ms" in result
            assert "speedup_factor" in result
            assert result["fallback_ms"] >= 0

    def test_initialize_returns_count(self):
        from core.polyglot_engine import PolyglotEngine
        e = PolyglotEngine()
        count = e.initialize()
        # At least the 6 new engines should load (they always load in fallback)
        assert count >= 6
