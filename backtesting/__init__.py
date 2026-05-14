"""Argus backtesting package — canonical backtesting implementation.

This is the single source of truth for all backtesting.
The backtest/ directory is deprecated — all imports should use backtesting/.
"""
from backtesting.unified_event_backtester import (
    UnifiedEventBacktester,
    BacktestResult,
    HistoricalMarketDataService,
    apply_slippage_bps,
    run_backtest_sync,
    run_backtest_oos,
)
from backtesting.walk_forward_optimizer import WalkForwardOptimizer
from backtesting.monte_carlo import MonteCarlo as MonteCarloSimulator
from backtesting.continuous_backtester import ContinuousBacktester
from backtesting.ab_test_framework import ABTestFramework
from backtesting.regime_backtest import RegimeBacktester

__all__ = [
    "UnifiedEventBacktester",
    "BacktestResult",
    "HistoricalMarketDataService",
    "apply_slippage_bps",
    "run_backtest_sync",
    "run_backtest_oos",
    "WalkForwardOptimizer",
    "MonteCarloSimulator",
    "ContinuousBacktester",
    "ABTestFramework",
    "RegimeBacktester",
]
