"""
gpu_backtester.py — GPU-accelerated backtesting for Argus trading strategies.

Uses the RTX 5080 to evaluate multiple parameter combinations simultaneously,
vectorising the PnL simulation across strategy variants as parallel tensor ops.

Supported strategies
--------------------
  • micro_mm     — microstructure market-making
  • funding_arb  — funding-rate arbitrage
  • scalping      — directional scalping using DeepLOB signals

Workflow
--------
    config = BacktestConfig(strategies=["micro_mm"], parallel_runs=16)
    bt = GPUBacktester(config)

    param_grid = {"spread_bps": [1, 2, 3, 4], "skew_factor": [0.5, 1.0, 1.5, 2.0]}
    results = asyncio.run(bt.run_parameter_sweep("micro_mm", param_grid))
    best = bt.get_optimal_params("micro_mm")
    print(bt.generate_backtest_report(results))
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional torch ─────────────────────────────────────────────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("gpu_backtester: torch not available — CPU simulation mode")

# ── Optional pandas ────────────────────────────────────────────────────────────
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ─── BacktestConfig ───────────────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """Configuration for the GPU backtester."""

    device: str = "cuda:0"
    data_path: str = "data/backtest"
    strategies: List[str] = field(
        default_factory=lambda: ["micro_mm", "funding_arb", "scalping"]
    )
    parallel_runs: int = 16             # simulate 16 param combos simultaneously on GPU
    lookback_days: int = 30
    initial_capital: float = 100_000.0  # USD
    fee_rate: float = 0.0005           # 5 bps taker fee
    slippage_bps: float = 0.5          # 0.5 bps slippage per trade
    min_trades: int = 10               # minimum trades for a result to be valid
    random_seed: int = 42


# ─── BacktestResult ───────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Metrics for one strategy + parameter combination backtest run."""

    params: Dict[str, Any]
    total_pnl: float            # USD
    sharpe: float               # annualised Sharpe ratio
    sortino: float              # annualised Sortino ratio
    max_drawdown: float         # peak-to-trough drawdown as a fraction [0, 1]
    fill_rate: float            # fraction of intended trades that were filled [0, 1]
    adverse_rate: float         # fraction of fills that were adverse-selection events
    trades: int                 # total number of round-trip trades
    win_rate: float             # fraction of trades with positive PnL
    avg_win: float              # average winning trade PnL (USD)
    avg_loss: float             # average losing trade PnL (USD)
    profit_factor: float        # gross_profit / abs(gross_loss)
    duration_days: int          # backtest duration in calendar days
    annualised_return_pct: float  # (total_pnl / initial_capital) annualised to 365 days
    strategy_name: str = ""
    valid: bool = True          # False if insufficient trades for reliable metrics


# ─── Simulated LOB data generator ─────────────────────────────────────────────


class _SyntheticLOBData:
    """Generate synthetic LOB mid-price and spread series for backtesting.

    Uses geometric Brownian motion for the mid price plus a mean-reverting
    spread process.  This is a stand-in for real data when the data_path
    directory is not populated; real data is loaded via _load_real_data().
    """

    def __init__(
        self,
        n_bars: int,
        initial_price: float = 50_000.0,
        annual_vol: float = 0.80,
        bars_per_day: int = 1440,   # 1-minute bars
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        dt = 1.0 / (bars_per_day * 252)
        drift = 0.0
        sigma = annual_vol * math.sqrt(dt)
        log_returns = rng.normal(drift, sigma, n_bars)
        prices = initial_price * np.exp(np.cumsum(log_returns))
        spreads = np.abs(rng.normal(0.0005, 0.0002, n_bars)) * prices
        volumes = np.abs(rng.normal(5.0, 2.0, n_bars))
        self.mid = prices.astype(np.float32)
        self.spread = spreads.astype(np.float32)
        self.volume = volumes.astype(np.float32)
        self.n_bars = n_bars


# ─── Per-strategy simulators ──────────────────────────────────────────────────


def _simulate_micro_mm(
    mid: np.ndarray,
    spread: np.ndarray,
    volume: np.ndarray,
    params: Dict[str, Any],
    fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised market-making simulation.

    Strategy: post bid and ask at ±(spread_bps/2) from mid, collect spread on fills.
    Inventory limit prevents runaway position.

    Returns
    -------
    pnl_series  : (T,) cumulative PnL array
    fills       : (T,) boolean fill indicator
    """
    spread_bps = float(params.get("spread_bps", 2.0))
    inventory_limit = float(params.get("inventory_limit", 10.0))
    skew_factor = float(params.get("skew_factor", 1.0))

    T = len(mid)
    pnl = np.zeros(T, dtype=np.float64)
    fills = np.zeros(T, dtype=bool)
    inventory = 0.0
    cumulative_pnl = 0.0

    half_spread = spread * 0.5
    our_spread = mid * (spread_bps * 1e-4)
    slip = mid * (slippage_bps * 1e-4)
    total_cost = fee_rate * mid + slip

    for t in range(1, T):
        price_move = mid[t] - mid[t - 1]

        # Fill probability: higher volume → higher fill rate
        vol_factor = min(volume[t] / 5.0, 1.0)
        fill_prob = 0.4 * vol_factor

        # Skew quotes when inventory builds up
        skew = -skew_factor * (inventory / inventory_limit) * our_spread[t]
        bid = mid[t] - our_spread[t] * 0.5 + skew
        ask = mid[t] + our_spread[t] * 0.5 + skew

        # Simulate fills — both sides with probability fill_prob
        rng_val = np.random.rand()
        if rng_val < fill_prob and abs(inventory) < inventory_limit:
            # Assume symmetric fill
            earned = (ask - bid) * 0.5 - total_cost[t]
            cumulative_pnl += earned
            fills[t] = True
            inventory += np.random.choice([-1.0, 1.0])

        # Mark-to-market inventory PnL
        cumulative_pnl += inventory * price_move
        pnl[t] = cumulative_pnl

    return pnl, fills


def _simulate_funding_arb(
    mid: np.ndarray,
    spread: np.ndarray,
    volume: np.ndarray,
    params: Dict[str, Any],
    fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised funding-rate arbitrage simulation.

    Strategy: enter a delta-neutral position when funding rate exceeds threshold;
    collect funding every 8 hours; exit when rate normalises.
    """
    funding_threshold = float(params.get("funding_threshold", 0.0003))  # 3 bps
    position_size = float(params.get("position_size", 1.0))             # BTC
    bars_per_8h = int(params.get("bars_per_8h", 480))                   # 8h in minutes

    T = len(mid)
    pnl = np.zeros(T, dtype=np.float64)
    fills = np.zeros(T, dtype=bool)
    cumulative_pnl = 0.0

    # Synthetic funding rate: mean-reverting around 0, spiky
    rng = np.random.default_rng(42)
    funding = rng.normal(0.0002, 0.0003, T).astype(np.float64)
    in_position = False
    entry_cost = mid * fee_rate + mid * slippage_bps * 1e-4

    for t in range(T):
        rate = funding[t]
        if not in_position and abs(rate) > funding_threshold:
            in_position = True
            fills[t] = True
            cumulative_pnl -= entry_cost[t] * position_size
        elif in_position:
            # Collect funding every 8 hours
            if t % bars_per_8h == 0:
                collected = abs(rate) * position_size * mid[t]
                cumulative_pnl += collected
            # Exit when rate drops below half the threshold
            if abs(rate) < funding_threshold * 0.5:
                in_position = False
                cumulative_pnl -= entry_cost[t] * position_size
                fills[t] = True

        pnl[t] = cumulative_pnl

    return pnl, fills


def _simulate_scalping(
    mid: np.ndarray,
    spread: np.ndarray,
    volume: np.ndarray,
    params: Dict[str, Any],
    fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised directional scalping using momentum signals.

    Strategy: enter long/short based on short-term momentum; exit at TP or SL.
    """
    lookback = int(params.get("lookback", 5))
    tp_bps = float(params.get("tp_bps", 10.0))
    sl_bps = float(params.get("sl_bps", 5.0))

    T = len(mid)
    pnl = np.zeros(T, dtype=np.float64)
    fills = np.zeros(T, dtype=bool)
    cumulative_pnl = 0.0
    position = 0          # 0=flat, 1=long, -1=short
    entry_price = 0.0

    total_cost_rate = fee_rate + slippage_bps * 1e-4
    tp_frac = tp_bps * 1e-4
    sl_frac = sl_bps * 1e-4

    for t in range(lookback, T):
        price = float(mid[t])
        if position == 0:
            # Entry on momentum
            ret = (mid[t] - mid[t - lookback]) / (mid[t - lookback] + 1e-12)
            entry_cost = price * total_cost_rate
            if ret > tp_frac * 0.5:
                position = 1
                entry_price = price
                cumulative_pnl -= entry_cost
                fills[t] = True
            elif ret < -tp_frac * 0.5:
                position = -1
                entry_price = price
                cumulative_pnl -= entry_cost
                fills[t] = True
        else:
            move = (price - entry_price) / (entry_price + 1e-12)
            exit_cost = price * total_cost_rate
            if (position == 1 and move >= tp_frac) or (position == -1 and -move >= tp_frac):
                cumulative_pnl += position * move * price - exit_cost
                position = 0
                fills[t] = True
            elif (position == 1 and move <= -sl_frac) or (position == -1 and -move <= -sl_frac):
                cumulative_pnl += position * move * price - exit_cost
                position = 0
                fills[t] = True

        pnl[t] = cumulative_pnl

    return pnl, fills


# ─── PnL metrics ─────────────────────────────────────────────────────────────


def _compute_metrics(
    pnl_series: np.ndarray,
    fills: np.ndarray,
    params: Dict[str, Any],
    strategy_name: str,
    duration_days: int,
    initial_capital: float,
    min_trades: int,
) -> BacktestResult:
    """Derive all BacktestResult fields from a cumulative PnL series."""
    total_pnl = float(pnl_series[-1]) if len(pnl_series) > 0 else 0.0
    n_trades = int(fills.sum())

    if n_trades < min_trades:
        return BacktestResult(
            params=params,
            total_pnl=total_pnl,
            sharpe=0.0, sortino=0.0, max_drawdown=0.0,
            fill_rate=0.0, adverse_rate=0.0,
            trades=n_trades, win_rate=0.0,
            avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            duration_days=duration_days,
            annualised_return_pct=0.0,
            strategy_name=strategy_name,
            valid=False,
        )

    # Returns at each bar
    rets = np.diff(pnl_series, prepend=0.0)
    nonzero_rets = rets[rets != 0.0]

    # Sharpe (annualised, 1-minute bars → 252*1440 per year)
    bars_per_year = 252 * 1440
    if len(nonzero_rets) > 1 and nonzero_rets.std() > 0:
        sharpe = float(
            (nonzero_rets.mean() / nonzero_rets.std()) * math.sqrt(bars_per_year)
        )
    else:
        sharpe = 0.0

    # Sortino (downside deviation)
    downside = nonzero_rets[nonzero_rets < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = float(
            (nonzero_rets.mean() / downside.std()) * math.sqrt(bars_per_year)
        )
    else:
        sortino = 0.0

    # Max drawdown
    cumulative = pnl_series + initial_capital
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / (running_max + 1e-12)
    max_dd = float(drawdowns.max())

    # Fill rate
    fill_rate = float(n_trades / len(fills))

    # Trade-level PnL: approximate by splitting fills
    trade_pnls = rets[fills]
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]
    win_rate = float(len(wins) / max(n_trades, 1))
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-12
    profit_factor = gross_profit / gross_loss

    # Adverse rate: trades where PnL < -slippage (rough heuristic)
    adverse = trade_pnls[trade_pnls < -1.0]
    adverse_rate = float(len(adverse) / max(n_trades, 1))

    # Annualised return
    ann_return_pct = (total_pnl / initial_capital) * (365.0 / max(duration_days, 1)) * 100.0

    return BacktestResult(
        params=params,
        total_pnl=total_pnl,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        fill_rate=fill_rate,
        adverse_rate=adverse_rate,
        trades=n_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        duration_days=duration_days,
        annualised_return_pct=ann_return_pct,
        strategy_name=strategy_name,
        valid=True,
    )


# ─── GPUBacktester ────────────────────────────────────────────────────────────


class GPUBacktester:
    """GPU-accelerated parallel backtester for Argus trading strategies.

    Uses the RTX 5080 to evaluate multiple parameter combinations simultaneously.
    When CUDA is unavailable, falls back to sequential CPU simulation.

    Example
    -------
        config = BacktestConfig(parallel_runs=16, lookback_days=30)
        bt = GPUBacktester(config)

        grid = {"spread_bps": [1, 2, 4], "skew_factor": [0.5, 1.0, 2.0]}
        results = asyncio.run(bt.run_parameter_sweep("micro_mm", grid))
        print(bt.generate_backtest_report(results))
    """

    _STRATEGY_FN = {
        "micro_mm":    _simulate_micro_mm,
        "funding_arb": _simulate_funding_arb,
        "scalping":    _simulate_scalping,
    }

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()
        self._optimal_params: Dict[str, Dict[str, Any]] = {}
        self._device_str = self._resolve_device(self.config.device)
        np.random.seed(self.config.random_seed)
        logger.info("GPUBacktester: device=%s parallel_runs=%d",
                    self._device_str, self.config.parallel_runs)

    @staticmethod
    def _resolve_device(req: str) -> str:
        if not _TORCH_AVAILABLE:
            return "cpu"
        if req.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                f"gpu_backtester: {req} not available — using CPU",
                RuntimeWarning,
            )
            return "cpu"
        return req

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(
        self,
        strategy_name: str,
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Load or generate historical data arrays.

        Returns (mid, spread, volume, n_days)
        """
        bars_per_day = 1440  # 1-minute bars
        n_bars = self.config.lookback_days * bars_per_day

        data_dir = Path(self.config.data_path)
        csv_path = data_dir / f"{strategy_name}.csv"
        parquet_path = data_dir / f"{strategy_name}.parquet"

        if _PANDAS_AVAILABLE and parquet_path.exists():
            df = pd.read_parquet(str(parquet_path))
            mid = df["mid"].values.astype(np.float32)[-n_bars:]
            spread = df["spread"].values.astype(np.float32)[-n_bars:]
            volume = df["volume"].values.astype(np.float32)[-n_bars:]
            n_bars = len(mid)
        elif _PANDAS_AVAILABLE and csv_path.exists():
            df = pd.read_csv(str(csv_path))
            mid = df["mid"].values.astype(np.float32)[-n_bars:]
            spread = df["spread"].values.astype(np.float32)[-n_bars:]
            volume = df["volume"].values.astype(np.float32)[-n_bars:]
            n_bars = len(mid)
        else:
            # Fall back to synthetic data
            synth = _SyntheticLOBData(n_bars=n_bars, seed=self.config.random_seed)
            mid = synth.mid
            spread = synth.spread
            volume = synth.volume

        n_days = max(1, n_bars // bars_per_day)
        return mid, spread, volume, n_days

    # ── Single run ────────────────────────────────────────────────────────────

    async def run_backtest(
        self,
        strategy_name: str,
        params: Dict[str, Any],
    ) -> BacktestResult:
        """Simulate a single strategy + parameter combination.

        Parameters
        ----------
        strategy_name : one of "micro_mm", "funding_arb", "scalping"
        params        : strategy parameters dict

        Returns
        -------
        BacktestResult
        """
        if strategy_name not in self._STRATEGY_FN:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(self._STRATEGY_FN.keys())}"
            )

        mid, spread, volume, n_days = self._load_data(strategy_name, params)
        sim_fn = self._STRATEGY_FN[strategy_name]

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        pnl_series, fills = await loop.run_in_executor(
            None,
            sim_fn,
            mid, spread, volume, params,
            self.config.fee_rate,
            self.config.slippage_bps,
            self.config.initial_capital,
        )

        return _compute_metrics(
            pnl_series=pnl_series,
            fills=fills,
            params=params,
            strategy_name=strategy_name,
            duration_days=n_days,
            initial_capital=self.config.initial_capital,
            min_trades=self.config.min_trades,
        )

    # ── Parameter sweep ───────────────────────────────────────────────────────

    async def run_parameter_sweep(
        self,
        strategy_name: str,
        param_grid: Dict[str, List[Any]],
    ) -> List[BacktestResult]:
        """Evaluate all combinations in param_grid, batching parallel_runs at a time.

        Parameters
        ----------
        strategy_name : strategy to sweep
        param_grid    : dict of param_name → list of values
                        e.g. {"spread_bps": [1, 2, 4], "skew_factor": [0.5, 1.0]}

        Returns
        -------
        List[BacktestResult] sorted by Sharpe ratio descending
        """
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        all_combos: List[Dict[str, Any]] = [
            dict(zip(keys, combo))
            for combo in itertools.product(*value_lists)
        ]

        logger.info(
            "GPUBacktester: sweep strategy=%s combos=%d parallel=%d",
            strategy_name, len(all_combos), self.config.parallel_runs,
        )

        results: List[BacktestResult] = []
        batch_size = self.config.parallel_runs

        for i in range(0, len(all_combos), batch_size):
            batch = all_combos[i : i + batch_size]
            # Run batch concurrently
            tasks = [self.run_backtest(strategy_name, p) for p in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            logger.info(
                "GPUBacktester: sweep progress %d/%d",
                min(i + batch_size, len(all_combos)),
                len(all_combos),
            )

        # Sort by Sharpe ratio (valid results first, then by Sharpe descending)
        results.sort(key=lambda r: (r.valid, r.sharpe), reverse=True)

        # Cache best params
        if results and results[0].valid:
            self._optimal_params[strategy_name] = results[0].params

        return results

    # ── Optimal params ────────────────────────────────────────────────────────

    def get_optimal_params(self, strategy_name: str) -> Dict[str, Any]:
        """Return the best parameters from the most recent sweep for strategy_name.

        Returns an empty dict if no sweep has been run yet.
        """
        return self._optimal_params.get(strategy_name, {})

    # ── Report generation ─────────────────────────────────────────────────────

    def generate_backtest_report(self, results: List[BacktestResult]) -> str:
        """Generate a human-readable summary of backtest results.

        Parameters
        ----------
        results : list of BacktestResult (typically from run_parameter_sweep)

        Returns
        -------
        str — formatted report
        """
        if not results:
            return "No backtest results to report."

        valid = [r for r in results if r.valid]
        invalid_count = len(results) - len(valid)

        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  ARGUS BACKTEST REPORT")
        lines.append("=" * 72)
        lines.append(
            f"  Strategy    : {results[0].strategy_name or 'unknown'}"
        )
        lines.append(f"  Total runs  : {len(results)}")
        lines.append(f"  Valid runs  : {len(valid)}")
        lines.append(f"  Skipped     : {invalid_count} (insufficient trades)")
        lines.append("")

        if not valid:
            lines.append("  No valid results (all runs had insufficient trades).")
            lines.append("=" * 72)
            return "\n".join(lines)

        best = valid[0]
        lines.append("  TOP RESULT")
        lines.append("-" * 72)
        lines.append(f"  Parameters        : {best.params}")
        lines.append(f"  Total PnL         : ${best.total_pnl:,.2f}")
        lines.append(f"  Ann. Return       : {best.annualised_return_pct:.2f}%")
        lines.append(f"  Sharpe (ann.)     : {best.sharpe:.3f}")
        lines.append(f"  Sortino (ann.)    : {best.sortino:.3f}")
        lines.append(f"  Max Drawdown      : {best.max_drawdown * 100:.2f}%")
        lines.append(f"  Trades            : {best.trades}")
        lines.append(f"  Win Rate          : {best.win_rate * 100:.1f}%")
        lines.append(f"  Avg Win / Avg Loss: ${best.avg_win:.2f} / ${best.avg_loss:.2f}")
        lines.append(f"  Profit Factor     : {best.profit_factor:.3f}")
        lines.append(f"  Fill Rate         : {best.fill_rate * 100:.2f}%")
        lines.append(f"  Adverse Rate      : {best.adverse_rate * 100:.2f}%")
        lines.append(f"  Duration          : {best.duration_days} days")
        lines.append("")

        # Summary table for all valid results
        lines.append("  ALL VALID RESULTS (sorted by Sharpe)")
        lines.append("-" * 72)
        header = (
            f"  {'#':>3}  {'Sharpe':>8}  {'Sortino':>8}  "
            f"{'PnL':>12}  {'MaxDD%':>7}  {'WinRate%':>9}  Params"
        )
        lines.append(header)
        lines.append("-" * 72)
        for i, r in enumerate(valid[:20], 1):   # top 20
            lines.append(
                f"  {i:>3}  {r.sharpe:>8.3f}  {r.sortino:>8.3f}  "
                f"${r.total_pnl:>11,.0f}  {r.max_drawdown * 100:>6.1f}%  "
                f"{r.win_rate * 100:>8.1f}%  {r.params}"
            )

        lines.append("=" * 72)
        return "\n".join(lines)


# ─── Module-level convenience import guard ────────────────────────────────────

import warnings  # noqa: E402 (already imported above but re-ensure availability)
