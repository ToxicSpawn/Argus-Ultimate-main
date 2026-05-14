"""
Earnings Predictor — ARGUS forecasts its own future performance.

This is NOT a crystal ball. It's a statistical model that estimates
expected returns based on:
1. Measured edge from actual trades (not backtests)
2. Strategy-specific win rates and average P&L
3. Current market regime and volatility
4. Historical performance in similar conditions
5. Monte Carlo simulation of possible outcomes

The prediction is a DISTRIBUTION, not a single number:
"With 90% confidence, next month's return will be between -2% and +8%"

This gives the user a realistic expectation and helps ARGUS
decide how aggressively to allocate capital.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EarningsForecast:
    """Projected earnings over a time horizon."""
    horizon_days: int
    starting_capital: float
    # Point estimates
    expected_return_pct: float      # best estimate
    expected_pnl: float             # in currency
    # Distribution
    p5_return_pct: float            # 5th percentile (bad case)
    p25_return_pct: float           # 25th percentile
    p50_return_pct: float           # median
    p75_return_pct: float           # 75th percentile
    p95_return_pct: float           # 95th percentile (good case)
    # P&L equivalents
    p5_pnl: float
    p50_pnl: float
    p95_pnl: float
    # Risk metrics
    prob_profit: float              # probability of positive return
    prob_loss_gt_5pct: float        # probability of losing >5%
    max_drawdown_expected_pct: float
    # Confidence
    data_quality: float             # 0-1: how much data backs this forecast
    model_confidence: float         # 0-1: how reliable is the model
    regime: str
    # Per-strategy breakdown
    strategy_contributions: Dict[str, float]
    # Caveats
    warnings: List[str]


class EarningsPredictor:
    """
    Predicts ARGUS future earnings from measured trading performance.

    Uses actual trade history to build a statistical model, then
    simulates thousands of possible futures via Monte Carlo.

    IMPORTANT: Predictions are only as good as the data. With <50 trades,
    forecasts have wide confidence intervals. With 500+ trades across
    multiple regimes, forecasts become statistically meaningful.
    """

    def __init__(self, n_simulations: int = 5000):
        self._n_sims = n_simulations
        self._trade_history: List[Dict[str, Any]] = []
        self._strategy_trades: Dict[str, List[float]] = defaultdict(list)
        self._regime_trades: Dict[str, List[float]] = defaultdict(list)
        self._daily_returns: List[float] = []

    def record_trade(self, pnl_pct: float, strategy: str = "",
                     regime: str = "normal") -> None:
        """Record a completed trade."""
        self._trade_history.append({
            "pnl_pct": pnl_pct, "strategy": strategy,
            "regime": regime, "timestamp": time.time(),
        })
        self._strategy_trades[strategy].append(pnl_pct)
        self._regime_trades[regime].append(pnl_pct)
        if len(self._trade_history) > 5000:
            self._trade_history = self._trade_history[-5000:]

    def record_daily_return(self, return_pct: float) -> None:
        """Record end-of-day portfolio return."""
        self._daily_returns.append(return_pct)
        if len(self._daily_returns) > 500:
            self._daily_returns = self._daily_returns[-500:]

    def predict(
        self,
        horizon_days: int = 30,
        capital: float = 1000.0,
        current_regime: str = "normal",
        trades_per_day: Optional[float] = None,
        active_strategies: Optional[List[str]] = None,
    ) -> EarningsForecast:
        """
        Predict earnings over the given horizon.

        Returns a probabilistic forecast based on historical performance.
        """
        warnings = []

        # Determine data quality
        n_trades = len(self._trade_history)
        data_quality = min(1.0, n_trades / 200)  # 200+ trades = full confidence

        if n_trades < 10:
            warnings.append(f"Only {n_trades} trades recorded — forecast is highly uncertain")
            return self._insufficient_data_forecast(horizon_days, capital, warnings)

        # Estimate trades per day
        if trades_per_day is None:
            if len(self._trade_history) >= 2:
                time_span = self._trade_history[-1]["timestamp"] - self._trade_history[0]["timestamp"]
                days_span = max(time_span / 86400, 0.1)
                trades_per_day = n_trades / days_span
            else:
                trades_per_day = 2.0
        trades_per_day = max(0.5, min(50, trades_per_day))

        # Get trade returns for active strategies in current regime
        relevant_trades = self._get_relevant_trades(current_regime, active_strategies)
        if len(relevant_trades) < 5:
            relevant_trades = [t["pnl_pct"] for t in self._trade_history]
            warnings.append("Using all trades (insufficient regime-specific data)")

        if not relevant_trades:
            return self._insufficient_data_forecast(horizon_days, capital, warnings)

        # Core statistics
        mean_ret = sum(relevant_trades) / len(relevant_trades)
        std_ret = (sum((r - mean_ret) ** 2 for r in relevant_trades) / max(len(relevant_trades) - 1, 1)) ** 0.5
        win_rate = sum(1 for r in relevant_trades if r > 0) / len(relevant_trades)

        # Per-strategy contribution
        strategy_contrib = {}
        if active_strategies:
            for strat in active_strategies:
                strat_trades = self._strategy_trades.get(strat, [])
                if strat_trades:
                    strategy_contrib[strat] = sum(strat_trades) / len(strat_trades) * len(strat_trades)

        # Monte Carlo simulation
        total_trades = int(trades_per_day * horizon_days)
        rng = np.random.RandomState(42)

        sim_returns = []
        sim_max_dd = []
        trade_returns = np.array(relevant_trades)

        for _ in range(self._n_sims):
            # Resample trades with replacement
            sim_trades = rng.choice(trade_returns, size=total_trades, replace=True)

            # Compound returns
            equity = [capital]
            peak = capital
            max_dd = 0.0
            for ret in sim_trades:
                new_eq = equity[-1] * (1 + ret / 100)
                equity.append(new_eq)
                peak = max(peak, new_eq)
                dd = (peak - new_eq) / peak
                max_dd = max(max_dd, dd)

            final_return = (equity[-1] / capital - 1) * 100
            sim_returns.append(final_return)
            sim_max_dd.append(max_dd * 100)

        sim_returns.sort()
        sim_max_dd.sort()

        # Percentiles
        def pct(arr, p):
            idx = int(len(arr) * p / 100)
            return arr[min(idx, len(arr) - 1)]

        expected = sum(sim_returns) / len(sim_returns)
        prob_profit = sum(1 for r in sim_returns if r > 0) / len(sim_returns)
        prob_loss_5 = sum(1 for r in sim_returns if r < -5) / len(sim_returns)

        # Model confidence: based on data quality + prediction stability
        # If p5 and p95 are very far apart, confidence is low
        spread = pct(sim_returns, 95) - pct(sim_returns, 5)
        model_confidence = data_quality * max(0.2, 1.0 - spread / 100)

        if n_trades < 50:
            warnings.append("Fewer than 50 trades — increase sample for reliable forecasts")
        if std_ret > abs(mean_ret) * 3:
            warnings.append("High variance — returns are very inconsistent")
        if win_rate < 0.45:
            warnings.append(f"Win rate is {win_rate:.0%} — below 45% breakeven threshold")

        return EarningsForecast(
            horizon_days=horizon_days,
            starting_capital=capital,
            expected_return_pct=expected,
            expected_pnl=capital * expected / 100,
            p5_return_pct=pct(sim_returns, 5),
            p25_return_pct=pct(sim_returns, 25),
            p50_return_pct=pct(sim_returns, 50),
            p75_return_pct=pct(sim_returns, 75),
            p95_return_pct=pct(sim_returns, 95),
            p5_pnl=capital * pct(sim_returns, 5) / 100,
            p50_pnl=capital * pct(sim_returns, 50) / 100,
            p95_pnl=capital * pct(sim_returns, 95) / 100,
            prob_profit=prob_profit,
            prob_loss_gt_5pct=prob_loss_5,
            max_drawdown_expected_pct=pct(sim_max_dd, 50),
            data_quality=data_quality,
            model_confidence=model_confidence,
            regime=current_regime,
            strategy_contributions=strategy_contrib,
            warnings=warnings,
        )

    def _get_relevant_trades(self, regime: str, strategies: Optional[List[str]]) -> List[float]:
        """Get trades relevant to current conditions."""
        relevant = []
        for t in self._trade_history:
            regime_match = not regime or t.get("regime", "") == regime
            strat_match = not strategies or t.get("strategy", "") in strategies
            if regime_match and strat_match:
                relevant.append(t["pnl_pct"])
        return relevant

    def _insufficient_data_forecast(self, horizon_days: int, capital: float,
                                     warnings: List[str]) -> EarningsForecast:
        """Return a highly uncertain forecast when insufficient data."""
        warnings.append("INSUFFICIENT DATA — this forecast is not reliable")
        return EarningsForecast(
            horizon_days=horizon_days, starting_capital=capital,
            expected_return_pct=0.0, expected_pnl=0.0,
            p5_return_pct=-8.0, p25_return_pct=-3.0, p50_return_pct=0.0,
            p75_return_pct=3.0, p95_return_pct=8.0,
            p5_pnl=-80, p50_pnl=0, p95_pnl=80,
            prob_profit=0.5, prob_loss_gt_5pct=0.1,
            max_drawdown_expected_pct=5.0,
            data_quality=0.0, model_confidence=0.0,
            regime="unknown", strategy_contributions={},
            warnings=warnings,
        )

    def project_compounding(
        self,
        months: int = 12,
        capital: float = 1000.0,
        current_regime: str = "normal",
    ) -> List[Dict[str, Any]]:
        """Project month-by-month compounding with confidence intervals."""
        projections = []
        running_capital = capital

        for month in range(1, months + 1):
            forecast = self.predict(
                horizon_days=30, capital=running_capital,
                current_regime=current_regime,
            )
            projections.append({
                "month": month,
                "starting_capital": running_capital,
                "expected_pnl": forecast.expected_pnl,
                "expected_capital": running_capital + forecast.expected_pnl,
                "p5_capital": running_capital + forecast.p5_pnl,
                "p50_capital": running_capital + forecast.p50_pnl,
                "p95_capital": running_capital + forecast.p95_pnl,
                "prob_profit": forecast.prob_profit,
                "model_confidence": forecast.model_confidence,
            })
            # Compound with median return for next month's starting capital
            running_capital = running_capital + forecast.p50_pnl

        return projections

    def get_stats(self) -> Dict[str, Any]:
        trades = [t["pnl_pct"] for t in self._trade_history]
        return {
            "total_trades": len(trades),
            "avg_return_pct": sum(trades) / len(trades) if trades else 0,
            "win_rate": sum(1 for t in trades if t > 0) / len(trades) if trades else 0,
            "strategies_tracked": len(self._strategy_trades),
            "regimes_tracked": len(self._regime_trades),
            "daily_returns_recorded": len(self._daily_returns),
        }
