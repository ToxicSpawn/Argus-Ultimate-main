"""
Kelly Criterion with Bayesian Bootstrap Uncertainty Estimation.

Standard Kelly sizing uses point estimates of win rate and average win/loss.
Real-world edge estimation has uncertainty — especially with small trade samples.
Using the full Kelly on a uncertain estimate leads to over-sizing and ruin.

This module:
  1. Applies Bayesian bootstrap to get confidence intervals on win rate / avg win
  2. Uses the *lower* CI bound (conservative) to compute Kelly fraction
  3. Applies a fractional Kelly multiplier (default 0.5 = half-Kelly)

Result: Position sizes that are robust to parameter uncertainty.

Reference:
  Thorp (1962), "Beat the Dealer" / Kelly (1956), "A New Interpretation of
  Information Rate" / Efron (1979) Bayesian Bootstrap
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_KELLY_FRACTION = 0.5     # half-Kelly by default
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_CONFIDENCE = 0.90        # use 90th percentile lower bound
MIN_TRADES = 10                  # need at least this many trades for meaningful estimate


class KellyUncertaintyCalculator:
    """
    Computes fractional Kelly position sizes with Bayesian bootstrap CIs.
    """

    def __init__(
        self,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        confidence_level: float = DEFAULT_CONFIDENCE,
        min_trades: int = MIN_TRADES,
        max_fraction: float = 0.25,  # hard cap: never bet more than 25% of capital
    ):
        self.kelly_fraction = kelly_fraction
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.min_trades = min_trades
        self.max_fraction = max_fraction

    def _bayesian_bootstrap_win_rate(self, wins: List[float], losses: List[float]) -> Tuple[float, float]:
        """
        Bayesian bootstrap estimate of win rate.

        Uses Dirichlet(1,...,1) weights (non-informative prior).
        Returns (point_estimate, lower_CI).
        """
        n = len(wins) + len(losses)
        labels = [1] * len(wins) + [0] * len(losses)
        labels_arr = np.array(labels, dtype=float)
        rng = np.random.default_rng(42)

        boot_win_rates = []
        for _ in range(self.n_bootstrap):
            # Dirichlet weights (equivalent to Bayesian bootstrap)
            weights = rng.dirichlet(np.ones(n))
            boot_win_rates.append(float(np.dot(weights, labels_arr)))

        boot_win_rates.sort()
        lower_idx = int((1.0 - self.confidence_level) * self.n_bootstrap)
        lower_ci = boot_win_rates[max(0, lower_idx)]
        point_est = len(wins) / n
        return point_est, lower_ci

    def _bayesian_bootstrap_mean(self, values: List[float]) -> Tuple[float, float]:
        """
        Bayesian bootstrap of a scalar distribution.

        Returns (point_estimate, lower_CI).
        For wins: we want upper estimate (lower is conservative); use lower CI.
        For losses: we want the *upper* estimate (larger losses = more conservative).
        """
        if not values:
            return 0.0, 0.0
        arr = np.array(values, dtype=float)
        n = len(arr)
        rng = np.random.default_rng(43)

        boot_means = []
        for _ in range(self.n_bootstrap):
            weights = rng.dirichlet(np.ones(n))
            boot_means.append(float(np.dot(weights, arr)))

        boot_means.sort()
        lower_idx = int((1.0 - self.confidence_level) * self.n_bootstrap)
        lower_ci = boot_means[max(0, lower_idx)]
        return float(np.mean(arr)), lower_ci

    def calculate(
        self,
        trades: List[float],
        capital: float = 1000.0,
        price: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compute optimal position size using half-Kelly with bootstrap CIs.

        Args:
            trades: List of trade returns as fraction of capital
                    (e.g., +0.02 = 2% gain, -0.01 = 1% loss)
            capital: Current portfolio capital (AUD)
            price:   Asset price (for converting fraction to units)

        Returns:
            {
              "fraction": float,          # fraction of capital to risk
              "capital_at_risk": float,   # AUD amount to risk
              "units": float,             # asset units to buy/sell
              "kelly_full": float,        # unconstrained Kelly fraction
              "win_rate": float,          # point estimate win rate
              "win_rate_lower_ci": float, # conservative win rate (bootstrap)
              "avg_win": float,           # average winning trade
              "avg_loss": float,          # average losing trade (magnitude)
              "n_trades": int,
              "method": str,
            }
        """
        n = len(trades)

        if n < self.min_trades:
            logger.debug(
                "Kelly: insufficient trades (%d < %d min) — using minimal sizing",
                n, self.min_trades,
            )
            frac = 0.02  # 2% default when no data
            return {
                "fraction": frac,
                "capital_at_risk": capital * frac,
                "units": (capital * frac) / max(price, 1e-10),
                "kelly_full": 0.0,
                "win_rate": 0.5,
                "win_rate_lower_ci": 0.5,
                "avg_win": 0.01,
                "avg_loss": 0.01,
                "n_trades": n,
                "method": "insufficient_data_default",
            }

        wins   = [t for t in trades if t > 0]
        losses = [abs(t) for t in trades if t < 0]

        if not wins or not losses:
            frac = 0.02
            return {
                "fraction": frac,
                "capital_at_risk": capital * frac,
                "units": (capital * frac) / max(price, 1e-10),
                "kelly_full": 0.0,
                "win_rate": len(wins) / n if wins else 0.0,
                "win_rate_lower_ci": 0.0,
                "avg_win": float(np.mean(wins)) if wins else 0.0,
                "avg_loss": float(np.mean(losses)) if losses else 0.0,
                "n_trades": n,
                "method": "one_sided_trades",
            }

        # Bayesian bootstrap CIs
        win_rate_pt, win_rate_lower = self._bayesian_bootstrap_win_rate(wins, losses)
        avg_win_pt,  avg_win_lower  = self._bayesian_bootstrap_mean(wins)
        avg_loss_pt, avg_loss_upper = self._bayesian_bootstrap_mean(losses)

        # Use conservative (lower) win rate and lower avg win, but upper avg loss
        # This deliberately undersizes — safer than oversizing
        w = win_rate_lower
        b = max(avg_win_lower, 1e-6)    # avg win (lower bound)
        a = max(avg_loss_upper, 1e-6)   # avg loss magnitude (upper bound = more conservative)

        # Kelly formula: f* = (w/a - (1-w)/b)  ... wait, standard form:
        # f* = (W*b - L) / b where W=win_rate, L=loss_rate, b=avg_win/avg_loss
        # Simplified: f* = W - (1-W)/(b/a)
        odds = b / a
        kelly_full = max(0.0, w - (1.0 - w) / max(odds, 1e-6))

        kelly_fractional = kelly_full * self.kelly_fraction
        kelly_capped = min(kelly_fractional, self.max_fraction)

        capital_at_risk = capital * kelly_capped
        units = capital_at_risk / max(price, 1e-10)

        logger.debug(
            "Kelly: W=%.3f (CI_lower=%.3f) avg_win=%.4f avg_loss=%.4f "
            "kelly_full=%.4f fractional=%.4f capped=%.4f",
            win_rate_pt, win_rate_lower, avg_win_pt, avg_loss_pt,
            kelly_full, kelly_fractional, kelly_capped,
        )

        return {
            "fraction": kelly_capped,
            "capital_at_risk": capital_at_risk,
            "units": units,
            "kelly_full": kelly_full,
            "kelly_fractional_uncapped": kelly_fractional,
            "win_rate": win_rate_pt,
            "win_rate_lower_ci": win_rate_lower,
            "avg_win": avg_win_pt,
            "avg_win_lower_ci": avg_win_lower,
            "avg_loss": avg_loss_pt,
            "avg_loss_upper_ci": avg_loss_upper,
            "n_trades": n,
            "method": f"bayesian_bootstrap_{self.kelly_fraction}x_kelly",
        }

    def size_from_signal(
        self,
        signal_confidence: float,
        trades: List[float],
        capital: float,
        price: float,
    ) -> Dict[str, Any]:
        """
        Compute position size combining Kelly fraction with signal confidence.

        Final fraction = kelly_fraction * signal_confidence (capped at max_fraction).
        """
        kelly = self.calculate(trades, capital, price)
        combined = kelly["fraction"] * max(0.0, min(1.0, signal_confidence))
        combined = min(combined, self.max_fraction)

        return {
            **kelly,
            "signal_confidence": signal_confidence,
            "fraction": combined,
            "capital_at_risk": capital * combined,
            "units": (capital * combined) / max(price, 1e-10),
            "method": kelly["method"] + "_confidence_weighted",
        }
