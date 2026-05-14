"""Optuna objective function for Argus hyperopt — Push 51.

Runs a fast vectorised single-pass backtest simulation over synthetic
or provided OHLCV data and returns annualised Sharpe ratio.
Prunes unpromising trials that produce fewer than MIN_TRADES trades.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

MIN_TRADES = 20
ANNUALISATION_FACTOR = 252  # trading days
DEFAULT_N_BARS = 2000


class ArgusObjective:
    """Callable Optuna objective that simulates Argus P&L for a trial's params.

    Parameters
    ----------
    returns : array-like, optional
        Pre-computed daily log-returns. If None, synthetic data is generated.
    n_bars : int
        Number of bars for the synthetic simulation.
    """

    def __init__(
        self,
        returns: Optional[np.ndarray] = None,
        n_bars: int = DEFAULT_N_BARS,
    ) -> None:
        if returns is not None:
            self._returns = np.asarray(returns, dtype=float)
        else:
            rng = np.random.default_rng(42)
            self._returns = rng.normal(0.0002, 0.018, size=n_bars)
        self._n_bars = len(self._returns)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _regime_scalar_series(
        self,
        returns: np.ndarray,
        bull_scalar: float,
        bear_scalar: float,
        refit_bars: int,
    ) -> np.ndarray:
        """Produce a scalar series using a simple rolling-window heuristic."""
        scalars = np.ones(len(returns))
        for i in range(refit_bars, len(returns)):
            window = returns[i - refit_bars : i]
            mean_ret = float(np.mean(window))
            if mean_ret > 0.0001:
                scalars[i] = bull_scalar
            elif mean_ret < -0.0001:
                scalars[i] = bear_scalar
            # else sideways → 1.0
        return scalars

    def _simulate(
        self,
        gateway_confidence: float,
        bull_scalar: float,
        bear_scalar: float,
        spread_bps: float,
        refit_bars: int,
    ) -> Dict[str, Any]:
        """Vectorised single-pass P&L simulation."""
        r = self._returns
        scalars = self._regime_scalar_series(r, bull_scalar, bear_scalar, refit_bars)
        spread_cost = spread_bps * 1e-4

        # Signal: trade when |return| exceeds a confidence-derived threshold
        threshold = gateway_confidence * 0.01
        signal = np.where(np.abs(r) > threshold, np.sign(r), 0.0)

        # P&L = signal * next_bar_return - spread_cost * |signal|
        pnl = signal[:-1] * r[1:] - spread_cost * np.abs(signal[:-1])
        pnl *= scalars[:-1]

        trades = int(np.sum(np.abs(signal[:-1]) > 0))
        return {"pnl": pnl, "trades": trades}

    # ------------------------------------------------------------------
    # Optuna interface
    # ------------------------------------------------------------------

    def __call__(self, trial: Any) -> float:  # trial: optuna.Trial
        if not _OPTUNA_AVAILABLE:
            raise RuntimeError("optuna is not installed")

        gateway_confidence = trial.suggest_float("gateway_confidence", 0.30, 0.90)
        bull_scalar = trial.suggest_float("hmm_bull_scalar", 1.00, 1.80)
        bear_scalar = trial.suggest_float("hmm_bear_scalar", 0.30, 0.70)
        spread_bps = trial.suggest_float("spread_bps", 1.0, 20.0)
        refit_bars = trial.suggest_int("regime_refit_bars", 50, 300)

        result = self._simulate(
            gateway_confidence=gateway_confidence,
            bull_scalar=bull_scalar,
            bear_scalar=bear_scalar,
            spread_bps=spread_bps,
            refit_bars=refit_bars,
        )

        trades = result["trades"]
        pnl = result["pnl"]

        if trades < MIN_TRADES:
            if _OPTUNA_AVAILABLE:
                raise optuna.exceptions.TrialPruned()
            return float("-inf")

        mean_pnl = float(np.mean(pnl))
        std_pnl = float(np.std(pnl)) + 1e-9
        sharpe = (mean_pnl / std_pnl) * math.sqrt(ANNUALISATION_FACTOR)
        return sharpe
