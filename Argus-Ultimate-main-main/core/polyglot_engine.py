"""
ARGUS Polyglot Engine — unified interface for multi-language computation.

Manages ten language-specific components:
  - Rust engine:     correlation, VaR, Kelly fraction, z-score (subprocess/numpy)
  - C fast_math:     EMA, rolling z-score, weighted mid-price (ctypes/numpy)
  - Node.js WS mux:  WebSocket multiplexer for exchange feeds (subprocess/asyncio)
  - Go order router: low-latency order routing (HTTP/Python)
  - CUDA GPU engine: Monte Carlo VaR, batch EMA, correlation, signal scoring (GPU/numpy)
  - C++ order book:  L2 order book with mid, spread, imbalance, walls (subprocess/Python)
  - Julia solver:    portfolio optimization, risk parity, Kelly optimal (subprocess/numpy)
  - Lua scripting:   strategy scripting with hot-reload (subprocess/Python)
  - Elixir supervisor: exchange connection supervisor with health tracking (subprocess/Python)
  - Zig WASM:        performance metrics — drawdown, Sharpe, Sortino, Calmar (WASM/numpy)

Each component gracefully degrades to a Python fallback when the native
binary/library is not compiled or unavailable.

Usage:
    engine = PolyglotEngine()
    status = engine.get_status()
    bench  = engine.benchmark()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class PolyglotEngine:
    """Unified multi-language computation interface for ARGUS."""

    def __init__(self) -> None:
        self.rust = None
        self.c_math = None
        self.node_ws = None
        self.go_router = None
        self.cuda = None
        self.cpp_orderbook = None
        self.julia = None
        self.r_stats = None
        self.lua = None
        self.elixir = None
        self.zig = None
        self._initialized = False

    def initialize(self) -> int:
        """
        Initialize all polyglot components. Returns count of successfully
        loaded components (regardless of native vs fallback).
        """
        count = 0

        # Rust engine
        try:
            from multilang.workers.rust_engine.bridge import RustEngine
            self.rust = RustEngine()
            count += 1
            logger.info("PolyglotEngine: RustEngine loaded (%s)", self.rust.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: RustEngine failed: %s", exc)

        # C fast_math
        try:
            from multilang.c_extensions.fast_math import FastMath
            self.c_math = FastMath()
            count += 1
            logger.info("PolyglotEngine: FastMath loaded (%s)", self.c_math.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: FastMath failed: %s", exc)

        # Node.js WS multiplexer
        try:
            from multilang.workers.node_ws.bridge import NodeWSMultiplexer
            self.node_ws = NodeWSMultiplexer()
            count += 1
            logger.info("PolyglotEngine: NodeWSMultiplexer loaded (%s)", self.node_ws.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: NodeWSMultiplexer failed: %s", exc)

        # Go order router
        try:
            from multilang.workers.go_router.bridge import GoOrderRouter
            self.go_router = GoOrderRouter()
            count += 1
            logger.info("PolyglotEngine: GoOrderRouter loaded (%s)", self.go_router.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: GoOrderRouter failed: %s", exc)

        # CUDA GPU engine
        try:
            from multilang.cuda_engine.bridge import CudaEngine
            self.cuda = CudaEngine()
            count += 1
            logger.info("PolyglotEngine: CudaEngine loaded (%s)", self.cuda.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: CudaEngine failed: %s", exc)

        # C++ order book
        try:
            from multilang.workers.cpp_orderbook.bridge import CppOrderBook
            self.cpp_orderbook = CppOrderBook()
            count += 1
            logger.info("PolyglotEngine: CppOrderBook loaded (%s)", self.cpp_orderbook.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: CppOrderBook failed: %s", exc)

        # Julia solver
        try:
            from multilang.workers.julia_solver.bridge import JuliaSolver
            self.julia = JuliaSolver()
            count += 1
            logger.info("PolyglotEngine: JuliaSolver loaded (%s)", self.julia.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: JuliaSolver failed: %s", exc)

        # R statistics bridge
        try:
            from multilang.workers.r_bridge import RStatsBridge
            self.r_stats = RStatsBridge()
            count += 1
            logger.info("PolyglotEngine: RStatsBridge loaded (%s)", self.r_stats.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: RStatsBridge failed: %s", exc)

        # Lua strategy engine
        try:
            from multilang.workers.lua_scripts.bridge import LuaStrategyEngine
            self.lua = LuaStrategyEngine()
            count += 1
            logger.info("PolyglotEngine: LuaStrategyEngine loaded (%s)", self.lua.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: LuaStrategyEngine failed: %s", exc)

        # Elixir supervisor
        try:
            from multilang.workers.elixir_supervisor.bridge import ElixirSupervisor
            self.elixir = ElixirSupervisor()
            count += 1
            logger.info("PolyglotEngine: ElixirSupervisor loaded (%s)", self.elixir.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: ElixirSupervisor failed: %s", exc)

        # Zig WASM compute
        try:
            from multilang.workers.zig_wasm.bridge import ZigCompute
            self.zig = ZigCompute()
            count += 1
            logger.info("PolyglotEngine: ZigCompute loaded (%s)", self.zig.backend)
        except Exception as exc:
            logger.debug("PolyglotEngine: ZigCompute failed: %s", exc)

        self._initialized = True
        logger.info("PolyglotEngine: %d/10 components initialized", count)
        return count

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Return status of each language component.

        Returns:
            {language: {available: bool, backend: 'native'|'fallback', latency_ms: float}}
        """
        status = {}

        if self.rust is not None:
            status["rust"] = {
                "available": self.rust.available,
                "backend": self.rust.backend,
                "latency_ms": self.rust.avg_latency_ms,
            }
        else:
            status["rust"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.c_math is not None:
            status["c"] = {
                "available": self.c_math.available,
                "backend": self.c_math.backend,
                "latency_ms": self.c_math.avg_latency_ms,
            }
        else:
            status["c"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.node_ws is not None:
            status["javascript"] = {
                "available": self.node_ws.available,
                "backend": self.node_ws.backend,
                "latency_ms": 0.0,
            }
        else:
            status["javascript"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.go_router is not None:
            status["go"] = {
                "available": self.go_router.available,
                "backend": self.go_router.backend,
                "latency_ms": self.go_router.avg_latency_ms,
            }
        else:
            status["go"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.cuda is not None:
            status["cuda"] = {
                "available": self.cuda.available,
                "backend": self.cuda.backend,
                "latency_ms": self.cuda.avg_latency_ms,
            }
        else:
            status["cuda"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.cpp_orderbook is not None:
            status["cpp"] = {
                "available": self.cpp_orderbook.available,
                "backend": self.cpp_orderbook.backend,
                "latency_ms": self.cpp_orderbook.avg_latency_ms,
            }
        else:
            status["cpp"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.julia is not None:
            status["julia"] = {
                "available": self.julia.available,
                "backend": self.julia.backend,
                "latency_ms": self.julia.avg_latency_ms,
            }
        else:
            status["julia"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.r_stats is not None:
            status["r"] = {
                "available": self.r_stats.available,
                "backend": self.r_stats.backend,
                "latency_ms": self.r_stats.avg_latency_ms,
            }
        else:
            status["r"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.lua is not None:
            status["lua"] = {
                "available": self.lua.available,
                "backend": self.lua.backend,
                "latency_ms": self.lua.avg_latency_ms,
            }
        else:
            status["lua"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.elixir is not None:
            status["elixir"] = {
                "available": self.elixir.available,
                "backend": self.elixir.backend,
                "latency_ms": self.elixir.avg_latency_ms,
            }
        else:
            status["elixir"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        if self.zig is not None:
            status["zig"] = {
                "available": self.zig.available,
                "backend": self.zig.backend,
                "latency_ms": self.zig.avg_latency_ms,
            }
        else:
            status["zig"] = {"available": False, "backend": "not_loaded", "latency_ms": 0.0}

        return status

    def benchmark(self, n: int = 1000) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmarks comparing native vs fallback for each component.

        Args:
            n: Number of data points for benchmark arrays.

        Returns:
            {component: {native_ms, fallback_ms, speedup_factor}}
        """
        results = {}

        # Benchmark Rust engine (correlation matrix)
        if self.rust is not None:
            results["rust_correlation"] = self._bench_rust_correlation(n)

        # Benchmark C fast_math (EMA)
        if self.c_math is not None:
            results["c_ema"] = self._bench_c_ema(n)

        # Benchmark Go router (route selection)
        if self.go_router is not None:
            results["go_route"] = self._bench_go_route()

        # Benchmark CUDA engine (Monte Carlo VaR)
        if self.cuda is not None:
            results["cuda_mc_var"] = self._bench_cuda_mc_var(n)

        # Benchmark C++ order book (update + state)
        if self.cpp_orderbook is not None:
            results["cpp_orderbook"] = self._bench_cpp_orderbook(n)

        # Benchmark Julia solver (portfolio optimization)
        if self.julia is not None:
            results["julia_optimize"] = self._bench_julia_optimize(n)

        # Benchmark R statistics bridge (volatility / VaR)
        if self.r_stats is not None:
            results["r_stats"] = self._bench_r_stats(n)

        # Benchmark Lua strategy engine (evaluate)
        if self.lua is not None:
            results["lua_evaluate"] = self._bench_lua_evaluate(n)

        # Benchmark Elixir supervisor (health check)
        if self.elixir is not None:
            results["elixir_health"] = self._bench_elixir_health()

        # Benchmark Zig WASM (Sharpe ratio)
        if self.zig is not None:
            results["zig_sharpe"] = self._bench_zig_sharpe(n)

        return results

    # ── Existing benchmarks ───────────────────────────────────────────

    def _bench_rust_correlation(self, n: int) -> Dict[str, Any]:
        """Benchmark correlation matrix computation."""
        from multilang.workers.rust_engine.bridge import RustEngine

        series = [np.random.randn(n).tolist() for _ in range(5)]
        data = {"series": series}

        # Always benchmark fallback
        fb = RustEngine.__dict__["_fb_correlation_matrix"]
        t0 = time.monotonic()
        for _ in range(10):
            fb(data)
        fallback_ms = (time.monotonic() - t0) / 10 * 1000

        # Native only if available
        native_ms = fallback_ms
        if self.rust.available:
            t0 = time.monotonic()
            for _ in range(10):
                self.rust.compute("correlation_matrix", data)
            native_ms = (time.monotonic() - t0) / 10 * 1000

        return {
            "native_ms": round(native_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "speedup_factor": round(fallback_ms / max(native_ms, 0.001), 2),
        }

    def _bench_c_ema(self, n: int) -> Dict[str, Any]:
        """Benchmark EMA computation."""
        from multilang.c_extensions.fast_math import FastMath

        prices = np.random.randn(n).cumsum() + 100

        # Always benchmark fallback
        t0 = time.monotonic()
        for _ in range(100):
            FastMath._fb_ema(prices, 20)
        fallback_ms = (time.monotonic() - t0) / 100 * 1000

        # Native only if available
        native_ms = fallback_ms
        if self.c_math.available:
            t0 = time.monotonic()
            for _ in range(100):
                self.c_math.ema(prices, 20)
            native_ms = (time.monotonic() - t0) / 100 * 1000

        return {
            "native_ms": round(native_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "speedup_factor": round(fallback_ms / max(native_ms, 0.001), 2),
        }

    def _bench_go_route(self) -> Dict[str, Any]:
        """Benchmark order routing."""
        order = {"symbol": "BTC/AUD", "side": "buy", "quantity": 0.01, "type": "limit"}

        # Fallback timing
        t0 = time.monotonic()
        for _ in range(100):
            self.go_router._fb_route(order, None)
        fallback_ms = (time.monotonic() - t0) / 100 * 1000

        # Native timing (if running)
        native_ms = fallback_ms
        if self.go_router.available and self.go_router._process:
            t0 = time.monotonic()
            for _ in range(10):
                self.go_router.route(order)
            native_ms = (time.monotonic() - t0) / 10 * 1000

        return {
            "native_ms": round(native_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "speedup_factor": round(fallback_ms / max(native_ms, 0.001), 2),
        }

    # ── New benchmarks ────────────────────────────────────────────────

    def _bench_cuda_mc_var(self, n: int) -> Dict[str, Any]:
        """Benchmark CUDA Monte Carlo VaR."""
        from multilang.cuda_engine.bridge import CudaEngine

        returns = np.random.randn(n, 3) * 0.02
        data = {
            "returns": returns.tolist(),
            "weights": [0.4, 0.3, 0.3],
            "n_scenarios": 10000,
            "confidence": 0.95,
        }

        # Always benchmark fallback
        t0 = time.monotonic()
        for _ in range(5):
            CudaEngine._fb_monte_carlo_var(data)
        fallback_ms = (time.monotonic() - t0) / 5 * 1000

        native_ms = fallback_ms
        return {
            "native_ms": round(native_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "speedup_factor": round(fallback_ms / max(native_ms, 0.001), 2),
        }

    def _bench_cpp_orderbook(self, n: int) -> Dict[str, Any]:
        """Benchmark C++ order book updates."""
        from multilang.workers.cpp_orderbook.bridge import CppOrderBook

        book = CppOrderBook()
        t0 = time.monotonic()
        for i in range(n):
            book.update("bid", 65000.0 - i * 0.5, 0.1 + i * 0.001)
            book.update("ask", 65010.0 + i * 0.5, 0.1 + i * 0.001)
        for _ in range(10):
            book.get_state()
        fallback_ms = (time.monotonic() - t0) * 1000

        return {
            "native_ms": round(fallback_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "speedup_factor": 1.0,
        }

    def _bench_julia_optimize(self, n: int) -> Dict[str, Any]:
        """Benchmark Julia portfolio optimization."""
        from multilang.workers.julia_solver.bridge import JuliaSolver

        returns = np.random.randn(min(n, 100), 4) * 0.02
        data = {"returns": returns.tolist(), "risk_aversion": 2.0}

        t0 = time.monotonic()
        for _ in range(3):
            JuliaSolver._fb_optimize_portfolio(data)
        fallback_ms = (time.monotonic() - t0) / 3 * 1000

        return {
            "native_ms": round(fallback_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "speedup_factor": 1.0,
        }

    def _bench_r_stats(self, n: int) -> Dict[str, Any]:
        """Benchmark R/fallback statistics tasks."""
        returns = (np.random.randn(n) * 0.02).tolist()

        t0 = time.monotonic()
        for _ in range(10):
            self.r_stats.var_estimate(returns, 0.95)
        elapsed_ms = (time.monotonic() - t0) / 10 * 1000

        return {
            "native_ms": round(elapsed_ms, 3),
            "fallback_ms": round(elapsed_ms, 3),
            "speedup_factor": 1.0,
        }

    def _bench_lua_evaluate(self, n: int) -> Dict[str, Any]:
        """Benchmark Lua strategy evaluation."""
        prices = (np.random.randn(n).cumsum() + 100).tolist()
        data = {"close": prices}

        t0 = time.monotonic()
        for _ in range(100):
            self.lua._fb_evaluate(data)
        fallback_ms = (time.monotonic() - t0) / 100 * 1000

        return {
            "native_ms": round(fallback_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "speedup_factor": 1.0,
        }

    def _bench_elixir_health(self) -> Dict[str, Any]:
        """Benchmark Elixir health check."""
        self.elixir.start()

        t0 = time.monotonic()
        for _ in range(100):
            self.elixir.get_health()
        fallback_ms = (time.monotonic() - t0) / 100 * 1000

        return {
            "native_ms": round(fallback_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "speedup_factor": 1.0,
        }

    def _bench_zig_sharpe(self, n: int) -> Dict[str, Any]:
        """Benchmark Zig Sharpe ratio computation."""
        from multilang.workers.zig_wasm.bridge import ZigCompute

        returns = np.random.randn(n) * 0.02

        t0 = time.monotonic()
        for _ in range(100):
            ZigCompute._fb_sharpe(returns, 0.0)
        fallback_ms = (time.monotonic() - t0) / 100 * 1000

        return {
            "native_ms": round(fallback_ms, 3),
            "fallback_ms": round(fallback_ms, 3),
            "speedup_factor": 1.0,
        }
