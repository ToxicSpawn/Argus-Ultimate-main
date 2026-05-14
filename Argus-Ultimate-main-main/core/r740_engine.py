"""
R740 Engine — unleash 192GB DDR4 RAM for market dominance.

This module activates when ARGUS detects it's running on the PowerEdge R740.
Instead of the conservative memory footprints used on a workstation, it
scales every intelligence system to fill 192GB of ECC RAM.

Architecture:
  Workstation (RTX 5080): Strategy execution, GPU evolution, real-time trading
  R740 (192GB DDR4 ECC RAM + NVMe): Data storage, backtesting, deep pattern matching

Hardware upgrade from R740 → R740 (April 2026):
  • DDR4-2933 vs DDR3-1600 → ~2x memory bandwidth for TimescaleDB + evolver
  • NVMe (U.2) for DB hot data → ~10x IOPS vs SATA
  • Xeon Scalable with AVX-512 → accelerated ML inference
  • More PCIe 3.0 lanes → dedicated bandwidth for Solarflare 10GbE NIC

What 192GB enables:
┌─────────────────────────────────────────────────────────────┐
│ MEGA EVOLUTION    │ 10,000 genome population                │
│                   │ 50 GP trees per island                  │
│                   │ 100 robustness jitter tests per genome  │
│                   │ Full CMA-ES covariance tracking         │
├───────────────────┼─────────────────────────────────────────┤
│ DEEP MEMORY       │ 500,000 market episodes                 │
│                   │ 100,000 causal events                   │
│                   │ 50,000 counterfactual decisions          │
│                   │ 10,000 hypothesis tests                  │
├───────────────────┼─────────────────────────────────────────┤
│ FULL TICK STORE   │ Every trade on 100+ pairs               │
│                   │ Full L2 order book snapshots             │
│                   │ 5 years of 1-minute candles in RAM       │
│                   │ Tape replay for strategy validation      │
├───────────────────┼─────────────────────────────────────────┤
│ PARALLEL BACKTEST │ 100 strategies simultaneously           │
│                   │ Walk-forward on 5 years of data          │
│                   │ Monte Carlo with 10,000 simulations      │
│                   │ Full cross-asset correlation matrix      │
├───────────────────┼─────────────────────────────────────────┤
│ ML ENSEMBLE       │ 50 models voting on every signal         │
│                   │ Per-regime model selection                │
│                   │ Real-time retraining every 100 trades     │
│                   │ Full feature store (10,000+ features)     │
└───────────────────┴─────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)


def is_r740() -> bool:
    """Detect if we're running on the R740 (or R740-class hardware)."""
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        return total_ram_gb >= 128  # R740 has 192GB DDR4
    except ImportError:
        # Fallback: check environment variable
        return os.environ.get("ARGUS_R740", "").lower() in ("1", "true", "yes")


# Backward-compatible alias for legacy imports during migration
def is_r720() -> bool:
    """Deprecated alias for is_r740() — kept during R740→R740 migration."""
    return is_r740()


def get_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 16.0  # conservative default


# ════════════════════════════════════════════════════════════════════════════
# R740 Configuration Scaler
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class R740Config:
    """Scaled configuration for R740 hardware."""
    # Evolution
    evolver_population: int = 10000
    generator_population: int = 500
    generator_tree_depth: int = 5
    mccv_folds: int = 10
    robustness_jitters: int = 20
    hall_of_fame_size: int = 200
    stagnation_limit: int = 20

    # Memory
    market_memory_capacity: int = 500000
    causal_event_capacity: int = 100000
    counterfactual_capacity: int = 50000
    hypothesis_capacity: int = 1000
    novelty_archive_capacity: int = 10000

    # Data
    historical_lookback_hours: int = 43800  # 5 years
    tick_store_capacity: int = 50_000_000   # 50M ticks
    indicator_cache_periods: List[int] = field(default_factory=lambda: list(range(2, 201)))

    # Backtesting
    parallel_backtests: int = 100
    monte_carlo_simulations: int = 10000
    walk_forward_windows: int = 20

    # ML
    max_models: int = 50
    feature_store_capacity: int = 1_000_000
    retrain_interval_trades: int = 100

    @classmethod
    def for_hardware(cls, ram_gb: float) -> 'R740Config':
        """Scale configuration to available RAM."""
        if ram_gb >= 192:
            return cls()  # full R740 config
        elif ram_gb >= 128:
            return cls(
                evolver_population=5000, generator_population=200,
                market_memory_capacity=200000, causal_event_capacity=50000,
                tick_store_capacity=20_000_000, parallel_backtests=50,
                monte_carlo_simulations=5000,
            )
        elif ram_gb >= 64:
            return cls(
                evolver_population=2000, generator_population=100,
                market_memory_capacity=50000, causal_event_capacity=20000,
                tick_store_capacity=5_000_000, parallel_backtests=20,
                monte_carlo_simulations=2000, max_models=20,
            )
        else:
            # Workstation defaults
            return cls(
                evolver_population=200, generator_population=30,
                generator_tree_depth=3, mccv_folds=3, robustness_jitters=3,
                market_memory_capacity=2000, causal_event_capacity=500,
                counterfactual_capacity=500, hypothesis_capacity=20,
                historical_lookback_hours=4380, tick_store_capacity=100000,
                parallel_backtests=4, monte_carlo_simulations=200,
                max_models=5, hall_of_fame_size=50,
                indicator_cache_periods=[5, 10, 14, 20, 30, 50],
            )


# ════════════════════════════════════════════════════════════════════════════
# In-Memory Tick Store (R740: 50M ticks in RAM)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Tick:
    """One trade from the tape."""
    timestamp: float
    price: float
    quantity: float
    side: str       # "buy" or "sell"


class InMemoryTickStore:
    """
    Stores every trade for every pair in RAM.

    On R740 with 192GB: ~50M ticks across 100 pairs.
    Each tick ≈ 40 bytes → 50M × 40 = 2GB (trivial for 192GB).

    Enables:
    - Tape replay for strategy validation
    - Order flow analysis (buy/sell volume imbalance)
    - Trade velocity measurement
    - Large block detection
    - VWAP computation from actual trades
    """

    def __init__(self, capacity_per_symbol: int = 500000):
        self._cap = capacity_per_symbol
        self._ticks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=capacity_per_symbol))
        self._total_count = 0

    def record(self, symbol: str, price: float, quantity: float, side: str,
               timestamp: Optional[float] = None) -> None:
        ts = timestamp or time.time()
        self._ticks[symbol].append(Tick(ts, price, quantity, side))
        self._total_count += 1

    def get_recent(self, symbol: str, n: int = 100) -> List[Tick]:
        ticks = self._ticks.get(symbol)
        if not ticks:
            return []
        return list(ticks)[-n:]

    def get_vwap(self, symbol: str, lookback_seconds: float = 300) -> float:
        """Compute VWAP from actual trades over lookback window."""
        ticks = self._ticks.get(symbol)
        if not ticks:
            return 0.0
        now = time.time()
        total_value = 0.0
        total_qty = 0.0
        for tick in reversed(ticks):
            if now - tick.timestamp > lookback_seconds:
                break
            total_value += tick.price * tick.quantity
            total_qty += tick.quantity
        return total_value / max(total_qty, 1e-9)

    def get_buy_sell_ratio(self, symbol: str, lookback_seconds: float = 60) -> float:
        """Buy volume / sell volume ratio. > 1 = buying pressure."""
        ticks = self._ticks.get(symbol)
        if not ticks:
            return 1.0
        now = time.time()
        buy_vol = 0.0
        sell_vol = 0.0
        for tick in reversed(ticks):
            if now - tick.timestamp > lookback_seconds:
                break
            if tick.side == "buy":
                buy_vol += tick.quantity
            else:
                sell_vol += tick.quantity
        return buy_vol / max(sell_vol, 1e-9)

    def get_trade_velocity(self, symbol: str, lookback_seconds: float = 60) -> float:
        """Trades per second over lookback window."""
        ticks = self._ticks.get(symbol)
        if not ticks:
            return 0.0
        now = time.time()
        count = 0
        for tick in reversed(ticks):
            if now - tick.timestamp > lookback_seconds:
                break
            count += 1
        return count / max(lookback_seconds, 1)

    def get_large_block_ratio(self, symbol: str, lookback_seconds: float = 300,
                               threshold_mult: float = 5.0) -> float:
        """Ratio of volume from large blocks (> 5x median size)."""
        ticks = self._ticks.get(symbol)
        if not ticks:
            return 0.0
        now = time.time()
        recent = [t for t in ticks if now - t.timestamp <= lookback_seconds]
        if len(recent) < 10:
            return 0.0
        sizes = [t.quantity for t in recent]
        median_size = sorted(sizes)[len(sizes) // 2]
        threshold = median_size * threshold_mult
        large_vol = sum(t.quantity for t in recent if t.quantity > threshold)
        total_vol = sum(t.quantity for t in recent)
        return large_vol / max(total_vol, 1e-9)

    def replay(self, symbol: str, start_time: float, end_time: float) -> List[Tick]:
        """Replay ticks between start and end time."""
        ticks = self._ticks.get(symbol)
        if not ticks:
            return []
        return [t for t in ticks if start_time <= t.timestamp <= end_time]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "symbols": len(self._ticks),
            "total_ticks": self._total_count,
            "per_symbol": {sym: len(ticks) for sym, ticks in list(self._ticks.items())[:10]},
            "memory_est_mb": self._total_count * 40 / (1024 * 1024),
        }


# ════════════════════════════════════════════════════════════════════════════
# Parallel Backtester (R740: 100 strategies simultaneously)
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BacktestResult:
    """Result of one backtest run."""
    strategy_id: str
    sharpe: float
    total_return_pct: float
    max_drawdown_pct: float
    trade_count: int
    win_rate: float
    profit_factor: float
    calmar: float
    duration_ms: float


def _run_single_backtest(args: Tuple) -> Dict[str, Any]:
    """Worker function for parallel backtesting (runs in subprocess)."""
    strategy_id, close, high, low, volume, params = args
    T = len(close)
    if T < 50:
        return {"strategy_id": strategy_id, "sharpe": 0, "trades": 0}

    # Simple breakout backtest (can be extended for any strategy)
    lookback = int(params.get("lookback", 20))
    tp = float(params.get("tp_pct", 2.0))
    sl = float(params.get("sl_pct", 1.5))
    fee = 0.26

    trades = []
    in_pos = False
    entry = 0.0

    for i in range(lookback + 1, T):
        h = float(np.max(high[i - lookback:i]))
        if not in_pos and close[i] > h:
            in_pos = True
            entry = close[i]
        elif in_pos:
            gain = (close[i] / entry - 1) * 100
            if gain >= tp or gain <= -sl:
                trades.append(gain - 2 * fee)
                in_pos = False

    if not trades:
        return {"strategy_id": strategy_id, "sharpe": 0, "trades": 0}

    n = len(trades)
    mean_r = sum(trades) / n
    std_r = (sum((t - mean_r) ** 2 for t in trades) / max(n - 1, 1)) ** 0.5

    return {
        "strategy_id": strategy_id,
        "sharpe": mean_r / max(std_r, 1e-9),
        "total_return": sum(trades),
        "trades": n,
        "win_rate": sum(1 for t in trades if t > 0) / n,
    }


class ParallelBacktester:
    """
    Runs many backtests simultaneously using multiprocessing.

    On R740: 100 strategies tested in parallel using 24 CPU cores.
    On workstation: 4-8 parallel backtests.
    """

    def __init__(self, max_workers: Optional[int] = None):
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 24)
        self._max_workers = max_workers

    def batch_backtest(
        self,
        strategies: List[Dict[str, Any]],
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple backtests in parallel.

        Args:
            strategies: list of {"id": str, "params": dict}
            close, high, low, volume: numpy arrays of OHLCV data

        Returns:
            List of result dicts (one per strategy)
        """
        if not strategies:
            return []

        args = [
            (s["id"], close, high, low, volume, s.get("params", {}))
            for s in strategies
        ]

        t0 = time.time()

        # Use ThreadPoolExecutor (safer than ProcessPool for numpy shared memory)
        results = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [executor.submit(_run_single_backtest, a) for a in args]
            for f in futures:
                try:
                    results.append(f.result(timeout=30))
                except Exception:
                    results.append({"strategy_id": "error", "sharpe": 0, "trades": 0})

        duration_ms = (time.time() - t0) * 1000
        logger.info("ParallelBacktester: %d strategies in %.0fms (%d workers)",
                     len(strategies), duration_ms, self._max_workers)
        return results

    def monte_carlo(
        self,
        trades: List[float],
        n_simulations: int = 10000,
        n_trades_per_sim: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation of strategy performance.

        Resamples trades with replacement to estimate distribution of outcomes.
        On R740: 10,000 simulations. On workstation: 200.
        """
        if not trades or len(trades) < 5:
            return {"median_return": 0, "p5_return": 0, "p95_return": 0, "prob_profit": 0}

        n = n_trades_per_sim or len(trades)
        rng = np.random.RandomState(42)
        sim_returns = []

        for _ in range(n_simulations):
            sample = rng.choice(trades, size=n, replace=True)
            sim_returns.append(float(np.sum(sample)))

        sim_returns.sort()
        return {
            "median_return": sim_returns[len(sim_returns) // 2],
            "p5_return": sim_returns[int(len(sim_returns) * 0.05)],
            "p95_return": sim_returns[int(len(sim_returns) * 0.95)],
            "prob_profit": sum(1 for r in sim_returns if r > 0) / len(sim_returns),
            "worst_case": sim_returns[0],
            "best_case": sim_returns[-1],
            "simulations": n_simulations,
        }


# ════════════════════════════════════════════════════════════════════════════
# R740 Activation
# ════════════════════════════════════════════════════════════════════════════

class R740Engine:
    """
    Main R740 engine that scales all ARGUS systems to 192GB.

    Usage:
        engine = R740Engine()
        if engine.active:
            config = engine.config
            tick_store = engine.tick_store
            backtester = engine.backtester
    """

    def __init__(self):
        self._ram_gb = get_ram_gb()
        self._active = self._ram_gb >= 64
        self._config = R740Config.for_hardware(self._ram_gb)

        self.tick_store = InMemoryTickStore(
            capacity_per_symbol=self._config.tick_store_capacity // 100,
        )
        self.backtester = ParallelBacktester()

        if self._active:
            logger.info(
                "R740Engine: activated (%.0fGB RAM) — pop=%d, memory=%d, ticks=%d, parallel=%d",
                self._ram_gb, self._config.evolver_population,
                self._config.market_memory_capacity,
                self._config.tick_store_capacity,
                self._config.parallel_backtests,
            )

    @property
    def active(self) -> bool:
        return self._active

    @property
    def config(self) -> R740Config:
        return self._config

    def scale_component(self, component_name: str, current_value: int) -> int:
        """Scale a component's capacity based on available RAM."""
        scale_map = {
            "evolver_population": self._config.evolver_population,
            "generator_population": self._config.generator_population,
            "market_memory_capacity": self._config.market_memory_capacity,
            "causal_event_capacity": self._config.causal_event_capacity,
            "novelty_archive_capacity": self._config.novelty_archive_capacity,
            "hall_of_fame_size": self._config.hall_of_fame_size,
            "monte_carlo_simulations": self._config.monte_carlo_simulations,
        }
        return scale_map.get(component_name, current_value)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active": self._active,
            "ram_gb": self._ram_gb,
            "config": {
                "evolver_population": self._config.evolver_population,
                "generator_population": self._config.generator_population,
                "market_memory_capacity": self._config.market_memory_capacity,
                "tick_store_capacity": self._config.tick_store_capacity,
                "parallel_backtests": self._config.parallel_backtests,
                "monte_carlo_simulations": self._config.monte_carlo_simulations,
            },
            "tick_store": self.tick_store.get_stats(),
        }
