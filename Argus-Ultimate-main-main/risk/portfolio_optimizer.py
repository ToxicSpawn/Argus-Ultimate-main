"""
Black-Litterman Portfolio Optimizer — dynamically allocates capital across strategies.

Combines a prior (equal-weight or market-cap-weight) with strategy signal
"views" to produce an optimal capital allocation that respects uncertainty.

The Black-Litterman formula:
    μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q]
    Σ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}

Where:
    π    = implied equilibrium returns (from prior weights)
    Σ    = covariance matrix of strategy returns
    τ    = scalar (uncertainty in prior, typically 0.025)
    P    = pick matrix (which strategies each view applies to)
    Q    = view vector (expected outperformance)
    Ω    = view uncertainty matrix (diagonal)

Usage:
    optimizer = BlackLittermanOptimizer(strategies=["trend_follow", "mean_revert", "stat_arb"])

    # Record strategy returns:
    optimizer.record_return("trend_follow", 0.012)
    optimizer.record_return("mean_revert", -0.003)

    # Add a view: "trend_follow will outperform mean_revert by 1.5% next period"
    optimizer.add_view(
        strategy_a="trend_follow",
        strategy_b="mean_revert",
        expected_outperformance=0.015,
        confidence=0.7
    )

    # Compute optimal weights:
    weights = optimizer.optimize()
    # {"trend_follow": 0.45, "mean_revert": 0.20, "stat_arb": 0.35}
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class View:
    """A single analyst/signal view on expected strategy performance."""

    strategy_a: str
    """Long strategy (the subject of this view)."""

    strategy_b: Optional[str]
    """Short strategy; if None this is an absolute view on strategy_a."""

    expected_return: float
    """
    Expected return for strategy_a.
    If strategy_b is set this is the expected outperformance of a over b.
    """

    confidence: float
    """
    Confidence in this view, in [0, 1].
    Scales the diagonal entry of Ω:  ω = variance_of_view / confidence.
    """

    timestamp: float
    """Unix timestamp when the view was created."""

    source: str = "signal"
    """Origin of the view: 'signal', 'manual', or 'backtest'."""


@dataclass
class OptimizationResult:
    """Output of a single optimizer.optimize() call."""

    weights: Dict[str, float]
    """Normalized strategy weights that sum to 1.0."""

    expected_returns: Dict[str, float]
    """Posterior (or prior) expected returns per strategy."""

    covariance: Optional[np.ndarray]
    """Full covariance matrix, or None when using equal-weight fallback."""

    sharpe_estimate: float
    """Portfolio-level Sharpe estimate (annualised, rf=0)."""

    max_weight: float
    """Largest weight in the result."""

    min_weight: float
    """Smallest weight in the result."""

    timestamp: float
    """Unix timestamp of this result."""

    method: str
    """
    How weights were computed:
    'black_litterman' | 'equal_weight' | 'sharpe_optimal' | 'min_variance' | 'hrp'
    """

    n_active_views: int
    """Number of views that fed into the BL update."""


# ---------------------------------------------------------------------------
# Black-Litterman optimizer
# ---------------------------------------------------------------------------

class BlackLittermanOptimizer:
    """
    Optimal strategy weight allocation using the Black-Litterman framework.

    Falls back gracefully to equal-weight if insufficient return history
    or if scipy is unavailable.
    """

    MIN_HISTORY: int = 20   # minimum observations before enabling BL
    MAX_HISTORY: int = 500  # rolling window cap

    def __init__(
        self,
        strategies: List[str],
        tau: float = 0.025,
        risk_aversion: float = 2.5,
        min_weight: float = 0.05,
        max_weight: float = 0.50,
        prior_mode: str = "equal",
        rebalance_threshold: float = 0.05,
    ) -> None:
        """
        Parameters
        ----------
        strategies:
            Names of the strategies to allocate across.
        tau:
            Scalar expressing uncertainty in the prior (typically 0.01-0.05).
        risk_aversion:
            Coefficient λ in the utility function U = w'μ - (λ/2) w'Σw.
        min_weight:
            Lower bound on any individual strategy weight.
        max_weight:
            Upper bound on any individual strategy weight.
        prior_mode:
            'equal'  – equal-weight prior.
            'sharpe' – weight proportional to Sharpe from history (when available).
        rebalance_threshold:
            Minimum absolute weight change before signalling a rebalance.
        """
        if not strategies:
            raise ValueError("strategies must be a non-empty list")
        if not (0.0 < min_weight < max_weight <= 1.0):
            raise ValueError(
                f"Require 0 < min_weight ({min_weight}) < max_weight ({max_weight}) <= 1"
            )
        n = len(strategies)
        if min_weight * n > 1.0:
            raise ValueError(
                f"min_weight={min_weight} × {n} strategies exceeds 1.0 – infeasible"
            )

        self.strategies: List[str] = list(strategies)
        self.tau: float = tau
        self.risk_aversion: float = risk_aversion
        self.min_weight: float = min_weight
        self.max_weight: float = max_weight
        self.prior_mode: str = prior_mode
        self.rebalance_threshold: float = rebalance_threshold

        # Return history: strategy → deque of floats
        self._returns: Dict[str, deque] = {
            s: deque(maxlen=self.MAX_HISTORY) for s in strategies
        }

        # Pending views for the next optimize() call
        self._views: List[View] = []

        # Last computed result (for should_rebalance / get_current_weights)
        self._last_result: Optional[OptimizationResult] = None

        # Strategy index for quick lookup
        self._idx: Dict[str, int] = {s: i for i, s in enumerate(strategies)}

        logger.info(
            "BlackLittermanOptimizer initialised: strategies=%s tau=%.4f "
            "risk_aversion=%.2f weight_bounds=[%.2f, %.2f]",
            strategies,
            tau,
            risk_aversion,
            min_weight,
            max_weight,
        )

    # ------------------------------------------------------------------
    # Public data-intake API
    # ------------------------------------------------------------------

    def record_return(
        self,
        strategy: str,
        return_pct: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a realised return for a strategy.

        Parameters
        ----------
        strategy:
            Strategy name (must be in self.strategies).
        return_pct:
            Realised return as a decimal (e.g. 0.012 for +1.2 %).
        timestamp:
            Unix timestamp of the observation; defaults to now.
        """
        if strategy not in self._returns:
            logger.warning("record_return: unknown strategy '%s' – ignored", strategy)
            return
        if not np.isfinite(return_pct):
            logger.warning(
                "record_return: non-finite return %.6f for '%s' – ignored",
                return_pct,
                strategy,
            )
            return
        self._returns[strategy].append(float(return_pct))

    def add_view(
        self,
        strategy_a: str,
        expected_return: Optional[float] = None,
        strategy_b: Optional[str] = None,
        expected_outperformance: Optional[float] = None,
        confidence: float = 0.5,
        source: str = "signal",
    ) -> None:
        """
        Add a view about expected strategy performance.

        Use *one* of:
          - ``expected_return``:         absolute view on strategy_a.
          - ``expected_outperformance``: relative view (a outperforms b by X).

        Parameters
        ----------
        strategy_a:
            Primary strategy for this view.
        expected_return:
            Absolute expected return for strategy_a.
        strategy_b:
            Comparison strategy (required when using expected_outperformance).
        expected_outperformance:
            Expected outperformance of a over b.
        confidence:
            View confidence in [0, 1].
        source:
            Origin label: 'signal', 'manual', or 'backtest'.
        """
        if strategy_a not in self._idx:
            logger.warning("add_view: unknown strategy_a '%s' – ignored", strategy_a)
            return
        if strategy_b is not None and strategy_b not in self._idx:
            logger.warning("add_view: unknown strategy_b '%s' – ignored", strategy_b)
            return
        if expected_return is None and expected_outperformance is None:
            logger.warning(
                "add_view: must supply expected_return or expected_outperformance – ignored"
            )
            return
        if expected_return is not None and expected_outperformance is not None:
            logger.warning(
                "add_view: supply exactly one of expected_return / expected_outperformance – "
                "using expected_outperformance"
            )
            expected_return = None

        confidence = float(np.clip(confidence, 1e-6, 1.0))

        if expected_outperformance is not None:
            ret = float(expected_outperformance)
            b = strategy_b
        else:
            ret = float(expected_return)  # type: ignore[arg-type]
            b = None

        view = View(
            strategy_a=strategy_a,
            strategy_b=b,
            expected_return=ret,
            confidence=confidence,
            timestamp=time.time(),
            source=source,
        )
        self._views.append(view)
        logger.debug(
            "View added: %s %s %s ret=%.4f conf=%.2f src=%s",
            strategy_a,
            ("vs " + b) if b else "(absolute)",
            "",
            ret,
            confidence,
            source,
        )

    def clear_views(self) -> None:
        """Discard all pending views. Call after each optimization cycle."""
        n = len(self._views)
        self._views.clear()
        if n:
            logger.debug("Cleared %d view(s)", n)

    # ------------------------------------------------------------------
    # Core optimization
    # ------------------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """
        Compute optimal strategy weights using the Black-Litterman framework.

        Steps
        -----
        1. If insufficient history (< MIN_HISTORY per strategy): return equal weights.
        2. Compute Σ (covariance matrix) from return history.
        3. Compute prior weights and implied equilibrium returns π = λΣw_prior.
        4. If views are available: apply Black-Litterman update to get μ_BL.
        5. Run mean-variance optimisation with weight bounds.
        6. Return normalised weights as OptimizationResult.
        """
        n = len(self.strategies)
        now = time.time()

        # Step 1: check for sufficient history
        min_obs = min(len(self._returns[s]) for s in self.strategies)
        if min_obs < self.MIN_HISTORY:
            logger.info(
                "Insufficient history (min=%d < %d) – returning equal weights",
                min_obs,
                self.MIN_HISTORY,
            )
            result = self._equal_weight_result(method="equal_weight", n_views=0)
            self._last_result = result
            return result

        # Step 2: covariance matrix
        sigma = self._compute_covariance()
        if sigma is None:
            logger.warning("Covariance computation failed – returning equal weights")
            result = self._equal_weight_result(method="equal_weight", n_views=0)
            self._last_result = result
            return result

        # Step 3: prior weights and implied equilibrium returns
        w_prior = self._compute_prior_weights(sigma)
        pi = self.risk_aversion * sigma @ w_prior  # shape (n,)

        # Step 4: Black-Litterman update (if views present)
        active_views = list(self._views)
        if active_views:
            try:
                mu_bl = self._black_litterman_update(sigma, pi, active_views)
                method = "black_litterman"
            except Exception as exc:
                logger.warning(
                    "Black-Litterman update failed (%s) – using prior returns", exc
                )
                mu_bl = pi
                method = "sharpe_optimal"
        else:
            mu_bl = pi
            method = "sharpe_optimal"

        # Step 5: mean-variance optimisation
        try:
            w_opt = self._mean_variance_optimize(mu_bl, sigma)
        except Exception as exc:
            logger.warning(
                "Mean-variance optimisation failed (%s) – using prior weights", exc
            )
            w_opt = w_prior

        # Normalise
        w_opt = np.clip(w_opt, self.min_weight, self.max_weight)
        total = w_opt.sum()
        if total <= 0.0:
            w_opt = np.full(n, 1.0 / n)
        else:
            w_opt = w_opt / total

        weights = {s: float(w_opt[i]) for i, s in enumerate(self.strategies)}
        expected_returns = {s: float(mu_bl[i]) for i, s in enumerate(self.strategies)}

        # Sharpe estimate (annualised by sqrt(252), rf=0)
        port_return = float(w_opt @ mu_bl)
        port_variance = float(w_opt @ sigma @ w_opt)
        if port_variance > 0.0:
            sharpe = (port_return / np.sqrt(port_variance)) * np.sqrt(252)
        else:
            sharpe = 0.0

        result = OptimizationResult(
            weights=weights,
            expected_returns=expected_returns,
            covariance=sigma,
            sharpe_estimate=sharpe,
            max_weight=float(w_opt.max()),
            min_weight=float(w_opt.min()),
            timestamp=now,
            method=method,
            n_active_views=len(active_views),
        )
        self._last_result = result
        logger.info(
            "optimize() → method=%s views=%d sharpe=%.3f weights=%s",
            method,
            len(active_views),
            sharpe,
            {s: f"{v:.3f}" for s, v in weights.items()},
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_covariance(self) -> Optional[np.ndarray]:
        """
        Build covariance matrix Σ from return history.

        Returns None if computation fails (e.g. all-zero variance).
        """
        n = len(self.strategies)
        min_len = min(len(self._returns[s]) for s in self.strategies)

        # Align all series to the same length (most recent observations)
        matrix = np.zeros((min_len, n))
        for j, s in enumerate(self.strategies):
            arr = list(self._returns[s])
            arr = arr[-min_len:]
            matrix[:, j] = arr

        # Replace any NaN/Inf with column mean, then 0
        for j in range(n):
            col = matrix[:, j]
            finite_mask = np.isfinite(col)
            if not finite_mask.all():
                col_mean = col[finite_mask].mean() if finite_mask.any() else 0.0
                col[~finite_mask] = col_mean
                matrix[:, j] = col

        try:
            sigma = np.cov(matrix, rowvar=False)
        except Exception as exc:
            logger.warning("np.cov failed: %s", exc)
            return None

        if sigma.ndim == 0:
            sigma = np.array([[float(sigma)]])

        # Regularise: add small ridge to ensure positive-definiteness
        sigma = sigma + 1e-8 * np.eye(n)

        # Sanity check
        if not np.all(np.isfinite(sigma)):
            logger.warning("Covariance matrix contains non-finite values after regularisation")
            return None

        return sigma

    def _compute_prior_weights(self, sigma: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute prior weight vector.

        'equal'  → uniform 1/n (clamped to bounds, then normalised).
        'sharpe' → weight proportional to per-strategy Sharpe (falls back to equal
                   if any Sharpe is non-positive or sigma is unavailable).
        """
        n = len(self.strategies)

        if self.prior_mode == "sharpe" and sigma is not None:
            try:
                sharpes = np.zeros(n)
                for i, s in enumerate(self.strategies):
                    arr = np.array(list(self._returns[s]))
                    mean_r = arr.mean()
                    std_r = arr.std(ddof=1) if len(arr) > 1 else 0.0
                    if std_r > 0.0:
                        sharpes[i] = mean_r / std_r
                    else:
                        sharpes[i] = 0.0

                # Use only positive Sharpes; fall back if all non-positive
                pos_sharpes = np.maximum(sharpes, 0.0)
                total = pos_sharpes.sum()
                if total > 0.0:
                    w = pos_sharpes / total
                    w = np.clip(w, self.min_weight, self.max_weight)
                    w = w / w.sum()
                    return w
            except Exception as exc:
                logger.warning("Sharpe prior computation failed (%s) – using equal", exc)

        # Equal-weight prior, clamped to bounds
        w = np.full(n, 1.0 / n)
        w = np.clip(w, self.min_weight, self.max_weight)
        return w / w.sum()

    def _black_litterman_update(
        self,
        sigma: np.ndarray,
        pi: np.ndarray,
        views: List[View],
    ) -> np.ndarray:
        """
        Apply the Black-Litterman formula to obtain posterior expected returns.

        BL formula:
            μ_BL = M^{-1} @ (inv(τΣ) @ π + P' @ inv(Ω) @ Q)
            where M = inv(τΣ) + P' @ inv(Ω) @ P

        Returns μ_BL of shape (n_strategies,).
        """
        n = len(self.strategies)
        k = len(views)

        tau_sigma = self.tau * sigma                 # (n, n)
        tau_sigma_inv = np.linalg.inv(tau_sigma)     # (n, n)

        # Build P (k × n), Q (k,), Ω diagonal entries (k,)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for row, view in enumerate(views):
            ia = self._idx[view.strategy_a]
            P[row, ia] = 1.0
            if view.strategy_b is not None:
                ib = self._idx[view.strategy_b]
                P[row, ib] = -1.0

            Q[row] = view.expected_return

            # Uncertainty: variance of the view, scaled by inverse confidence.
            # We estimate view variance from the diagonal of τΣ for strategy_a.
            view_var = tau_sigma[ia, ia]
            if view.strategy_b is not None:
                ib = self._idx[view.strategy_b]
                # Relative view variance (long a, short b)
                view_var = (
                    tau_sigma[ia, ia]
                    + tau_sigma[ib, ib]
                    - 2.0 * tau_sigma[ia, ib]
                )
            # Larger uncertainty → smaller confidence
            omega_diag[row] = view_var / max(view.confidence, 1e-6)

        # Protect against zero/negative omega entries
        omega_diag = np.maximum(omega_diag, 1e-10)
        omega_inv = np.diag(1.0 / omega_diag)       # (k, k)

        Pt = P.T                                      # (n, k)

        # M = inv(τΣ) + P' Ω^{-1} P
        M = tau_sigma_inv + Pt @ omega_inv @ P       # (n, n)

        # rhs = inv(τΣ) π + P' Ω^{-1} Q
        rhs = tau_sigma_inv @ pi + Pt @ omega_inv @ Q  # (n,)

        # μ_BL = M^{-1} rhs
        mu_bl = np.linalg.solve(M, rhs)              # (n,)

        if not np.all(np.isfinite(mu_bl)):
            logger.warning(
                "BL posterior contains non-finite values; falling back to prior returns"
            )
            return pi.copy()

        return mu_bl

    def _mean_variance_optimize(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        """
        Maximise the Sharpe ratio subject to weight bounds and a sum-to-one constraint.

        Objective:  maximize  w'μ / sqrt(w'Σw)   (rf = 0)
        Subject to: sum(w) = 1
                    min_weight ≤ w[i] ≤ max_weight  ∀i

        Uses scipy.optimize.minimize (SLSQP).  Falls back to proportional-to-mu
        allocation if scipy is unavailable or the solver fails.
        """
        n = len(self.strategies)
        w0 = self._compute_prior_weights(sigma)

        try:
            from scipy.optimize import minimize  # optional dependency

            def neg_sharpe(w: np.ndarray) -> float:
                port_ret = float(w @ mu)
                port_var = float(w @ sigma @ w)
                if port_var <= 0.0:
                    return 0.0
                return -(port_ret / np.sqrt(port_var))

            def neg_sharpe_grad(w: np.ndarray):
                """Analytical gradient of -Sharpe for SLSQP."""
                port_ret = float(w @ mu)
                sigma_w = sigma @ w
                port_var = float(w @ sigma_w)
                if port_var <= 1e-12:
                    return np.zeros(n)
                sqrt_var = np.sqrt(port_var)
                grad = -(
                    (mu * sqrt_var - port_ret * sigma_w / sqrt_var)
                    / port_var
                )
                return grad

            bounds = [(self.min_weight, self.max_weight)] * n
            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

            result = minimize(
                neg_sharpe,
                w0,
                jac=neg_sharpe_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )

            if result.success or result.fun < neg_sharpe(w0):
                return np.array(result.x)

            logger.warning(
                "SLSQP did not converge cleanly (message: %s) – using prior weights",
                result.message,
            )
            return w0

        except ImportError:
            logger.warning(
                "scipy not available – using proportional-to-mu weight allocation"
            )
            return self._proportional_fallback(mu)
        except Exception as exc:
            logger.warning("_mean_variance_optimize failed: %s – using prior weights", exc)
            return w0

    def _proportional_fallback(self, mu: np.ndarray) -> np.ndarray:
        """
        Fallback when scipy is unavailable: weight proportional to positive expected return.
        """
        n = len(self.strategies)
        pos_mu = np.maximum(mu, 0.0)
        total = pos_mu.sum()
        if total <= 0.0:
            return np.full(n, 1.0 / n)
        w = pos_mu / total
        w = np.clip(w, self.min_weight, self.max_weight)
        w = w / w.sum()
        return w

    def _equal_weight_result(self, method: str, n_views: int) -> OptimizationResult:
        """Return an equal-weight OptimizationResult with no covariance."""
        n = len(self.strategies)
        w = max(self.min_weight, min(self.max_weight, 1.0 / n))
        raw = {s: w for s in self.strategies}
        total = sum(raw.values())
        weights = {s: v / total for s, v in raw.items()}
        return OptimizationResult(
            weights=weights,
            expected_returns={s: 0.0 for s in self.strategies},
            covariance=None,
            sharpe_estimate=0.0,
            max_weight=max(weights.values()),
            min_weight=min(weights.values()),
            timestamp=time.time(),
            method=method,
            n_active_views=n_views,
        )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_current_weights(self) -> Dict[str, float]:
        """
        Return the weights from the last optimize() call.

        Returns equal weights if optimize() has never been called.
        """
        if self._last_result is not None:
            return dict(self._last_result.weights)
        n = len(self.strategies)
        w = 1.0 / n
        return {s: w for s in self.strategies}

    def get_return_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Return per-strategy statistics computed from recorded return history.

        Keys per strategy: 'mean', 'std', 'sharpe', 'n_obs'.
        """
        stats: Dict[str, Dict[str, float]] = {}
        for s in self.strategies:
            arr = np.array(list(self._returns[s]))
            if len(arr) == 0:
                stats[s] = {"mean": 0.0, "std": 0.0, "sharpe": 0.0, "n_obs": 0.0}
                continue
            mean_r = float(arr.mean())
            std_r = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0.0 else 0.0
            stats[s] = {
                "mean": mean_r,
                "std": std_r,
                "sharpe": float(sharpe),
                "n_obs": float(len(arr)),
            }
        return stats

    def should_rebalance(self, current_weights: Dict[str, float]) -> bool:
        """
        Return True if any weight in current_weights deviates from the last
        computed target by more than rebalance_threshold.

        Returns False (no rebalance) if optimize() has never been called.
        """
        if self._last_result is None:
            return False
        target = self._last_result.weights
        for s in self.strategies:
            if s not in current_weights:
                return True
            diff = abs(current_weights[s] - target.get(s, 0.0))
            if diff > self.rebalance_threshold:
                logger.debug(
                    "Rebalance triggered: strategy=%s diff=%.4f > threshold=%.4f",
                    s,
                    diff,
                    self.rebalance_threshold,
                )
                return True
        return False

    def to_capital_allocation(
        self,
        total_capital: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Convert a weight dict to dollar amounts.

        Parameters
        ----------
        total_capital:
            Total portfolio capital in account currency.
        weights:
            Weight dict; defaults to last computed weights.

        Returns
        -------
        Dict mapping strategy name → dollar amount.
        """
        if weights is None:
            weights = self.get_current_weights()
        return {s: w * total_capital for s, w in weights.items()}


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (HRP) optimizer
# ---------------------------------------------------------------------------

class HRPOptimizer:
    """
    Hierarchical Risk Parity (HRP) allocator for strategy allocation.
    
    HRP uses hierarchical clustering to allocate weights based on the
    correlation structure of strategy returns. It's more robust to
    estimation errors in the covariance matrix than traditional mean-variance.
    
    Based on: De Prado, M. L. (2016). "Building Diversified Portfolios that
    Outperform Out-of-Sample."
    """

    def __init__(
        self,
        strategies: List[str],
        min_weight: float = 0.05,
        max_weight: float = 0.50,
        linkage_method: str = "single",
    ) -> None:
        if not strategies:
            raise ValueError("strategies must be a non-empty list")
        self.strategies: List[str] = list(strategies)
        self.min_weight: float = min_weight
        self.max_weight: float = max_weight
        self.linkage_method: str = linkage_method
        
        # Return history: strategy → deque of floats
        self._returns: Dict[str, deque] = {
            s: deque(maxlen=500) for s in strategies
        }
        
        # Last computed result
        self._last_result: Optional[OptimizationResult] = None
        
        logger.info(
            "HRPOptimizer initialised: strategies=%s linkage=%s weight_bounds=[%.2f, %.2f]",
            strategies, linkage_method, min_weight, max_weight,
        )
    
    def record_return(
        self,
        strategy: str,
        return_pct: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a realised return for a strategy."""
        if strategy not in self._returns:
            logger.warning("record_return: unknown strategy '%s' – ignored", strategy)
            return
        if not np.isfinite(return_pct):
            logger.warning("record_return: non-finite return %.6f for '%s' – ignored", return_pct, strategy)
            return
        self._returns[strategy].append(float(return_pct))
    
    def add_view(self, *args, **kwargs) -> None:
        """No-op: HRP does not use views like Black-Litterman."""
        pass
    
    def clear_views(self) -> None:
        """No-op: HRP does not use views."""
        pass
    
    def optimize(self) -> OptimizationResult:
        """
        Compute optimal strategy weights using HRP.
        
        Steps:
        1. Build returns matrix from history
        2. Compute correlation and covariance matrices
        3. Hierarchical clustering on correlation distance
        4. Quasi-diagonalization
        5. Recursive bisection for weight allocation
        """
        n = len(self.strategies)
        now = time.time()
        
        # Check for sufficient history
        min_obs = min(len(self._returns[s]) for s in self.strategies)
        if min_obs < 30:
            logger.info("Insufficient history (min=%d < 30) – returning equal weights", min_obs)
            result = self._equal_weight_result(method="equal_weight", n_views=0)
            self._last_result = result
            return result
        
        try:
            # Build returns matrix
            min_len = min(len(self._returns[s]) for s in self.strategies)
            matrix = np.zeros((min_len, n))
            for j, s in enumerate(self.strategies):
                arr = list(self._returns[s])[-min_len:]
                matrix[:, j] = arr
            
            # Compute correlation and covariance
            corr = np.corrcoef(matrix, rowvar=False)
            cov = np.cov(matrix, rowvar=False)
            
            if corr.ndim == 0:
                corr = np.array([[1.0]])
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            
            # Distance matrix from correlation
            dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0, 1))
            
            # Hierarchical clustering
            from scipy.cluster.hierarchy import linkage, leaves_list
            condensed = dist[np.triu_indices(n, k=1)]
            Z = linkage(condensed, method=self.linkage_method)
            order = leaves_list(Z)
            
            # Quasi-diagonal covariance
            cov_sorted = cov[np.ix_(order, order)]
            
            # Recursive bisection
            weights = np.ones(n)
            clusters: list[list[int]] = [list(range(n))]
            while clusters:
                new_clusters = []
                for cluster in clusters:
                    if len(cluster) < 2:
                        continue
                    split = len(cluster) // 2
                    left_idx = cluster[:split]
                    right_idx = cluster[split:]
                    
                    def _cluster_var(idxs: list[int]) -> float:
                        sub_cov = cov_sorted[np.ix_(idxs, idxs)]
                        inv_var = 1.0 / np.maximum(np.diag(sub_cov), 1e-12)
                        w_ = inv_var / inv_var.sum()
                        return float(w_ @ sub_cov @ w_)
                    
                    var_l = _cluster_var(left_idx)
                    var_r = _cluster_var(right_idx)
                    total_var = var_l + var_r
                    alpha = 1.0 - var_l / total_var if total_var > 0 else 0.5
                    
                    weights[left_idx] *= (1 - alpha)
                    weights[right_idx] *= alpha
                    
                    if len(left_idx) > 1:
                        new_clusters.append(left_idx)
                    if len(right_idx) > 1:
                        new_clusters.append(right_idx)
                clusters = new_clusters
            
            # Re-order back to original column order
            final_weights = np.zeros(n)
            for i, orig_idx in enumerate(order):
                final_weights[orig_idx] = weights[i]
            
            # Normalize and apply bounds
            final_weights = np.clip(final_weights, self.min_weight, self.max_weight)
            total = final_weights.sum()
            if total <= 0.0:
                final_weights = np.full(n, 1.0 / n)
            else:
                final_weights = final_weights / total
            
            weights_dict = {s: float(final_weights[i]) for i, s in enumerate(self.strategies)}
            
            # Compute expected returns (from historical mean)
            expected_returns = {}
            for i, s in enumerate(self.strategies):
                arr = np.array(list(self._returns[s]))
                expected_returns[s] = float(arr.mean()) if len(arr) > 0 else 0.0
            
            # Sharpe estimate
            port_return = float(final_weights @ np.array([expected_returns[s] for s in self.strategies]))
            port_variance = float(final_weights @ cov @ final_weights)
            sharpe = (port_return / np.sqrt(port_variance)) * np.sqrt(252) if port_variance > 0.0 else 0.0
            
            result = OptimizationResult(
                weights=weights_dict,
                expected_returns=expected_returns,
                covariance=cov,
                sharpe_estimate=sharpe,
                max_weight=float(final_weights.max()),
                min_weight=float(final_weights.min()),
                timestamp=now,
                method="hrp",
                n_active_views=0,
            )
            self._last_result = result
            logger.info(
                "HRP optimize() → sharpe=%.3f weights=%s",
                sharpe, {s: f"{v:.3f}" for s, v in weights_dict.items()},
            )
            return result
            
        except ImportError:
            logger.warning("scipy not available for HRP clustering – using equal weights")
            return self._equal_weight_result(method="equal_weight", n_views=0)
        except Exception as exc:
            logger.warning("HRP optimization failed (%s) – using equal weights", exc)
            return self._equal_weight_result(method="equal_weight", n_views=0)
    
    def _equal_weight_result(self, method: str, n_views: int) -> OptimizationResult:
        """Return an equal-weight OptimizationResult."""
        n = len(self.strategies)
        w = max(self.min_weight, min(self.max_weight, 1.0 / n))
        weights = {s: w for s in self.strategies}
        total = sum(weights.values())
        weights = {s: v / total for s, v in weights.items()}
        return OptimizationResult(
            weights=weights,
            expected_returns={s: 0.0 for s in self.strategies},
            covariance=None,
            sharpe_estimate=0.0,
            max_weight=max(weights.values()),
            min_weight=min(weights.values()),
            timestamp=time.time(),
            method=method,
            n_active_views=n_views,
        )
    
    def get_current_weights(self) -> Dict[str, float]:
        """Return weights from last optimize() call."""
        if self._last_result is not None:
            return dict(self._last_result.weights)
        n = len(self.strategies)
        w = 1.0 / n
        return {s: w for s in self.strategies}
    
    def get_return_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-strategy statistics."""
        stats: Dict[str, Dict[str, float]] = {}
        for s in self.strategies:
            arr = np.array(list(self._returns[s]))
            if len(arr) == 0:
                stats[s] = {"mean": 0.0, "std": 0.0, "sharpe": 0.0, "n_obs": 0.0}
                continue
            mean_r = float(arr.mean())
            std_r = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0.0 else 0.0
            stats[s] = {"mean": mean_r, "std": std_r, "sharpe": float(sharpe), "n_obs": float(len(arr))}
        return stats
    
    def should_rebalance(self, current_weights: Dict[str, float]) -> bool:
        """Check if rebalance is needed."""
        if self._last_result is None:
            return False
        target = self._last_result.weights
        for s in self.strategies:
            if s not in current_weights:
                return True
            diff = abs(current_weights[s] - target.get(s, 0.0))
            if diff > 0.05:  # 5% threshold
                return True
        return False
    
    def to_capital_allocation(
        self,
        total_capital: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Convert weights to dollar amounts."""
        if weights is None:
            weights = self.get_current_weights()
        return {s: w * total_capital for s, w in weights.items()}


# ---------------------------------------------------------------------------
# Equal-weight fallback optimizer
# ---------------------------------------------------------------------------

class EqualWeightOptimizer:
    """
    Simple equal-weight allocator.

    Used as a drop-in fallback when BlackLittermanOptimizer has insufficient
    return history, or as a baseline for performance comparison.
    """

    def __init__(
        self,
        strategies: List[str],
        min_weight: float = 0.05,
        max_weight: float = 0.50,
    ) -> None:
        if not strategies:
            raise ValueError("strategies must be a non-empty list")
        self.strategies: List[str] = list(strategies)
        self.min_weight: float = min_weight
        self.max_weight: float = max_weight

    # ------------------------------------------------------------------
    # Core API (mirrors BlackLittermanOptimizer interface)
    # ------------------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """Return uniformly equal weights, clamped to [min_weight, max_weight]."""
        n = len(self.strategies)
        w = max(self.min_weight, min(self.max_weight, 1.0 / n))
        weights = {s: w for s in self.strategies}
        total = sum(weights.values())
        weights = {s: v / total for s, v in weights.items()}

        w_arr = np.array(list(weights.values()))

        return OptimizationResult(
            weights=weights,
            expected_returns={s: 0.0 for s in self.strategies},
            covariance=None,
            sharpe_estimate=0.0,
            max_weight=float(w_arr.max()),
            min_weight=float(w_arr.min()),
            timestamp=time.time(),
            method="equal_weight",
            n_active_views=0,
        )

    def get_current_weights(self) -> Dict[str, float]:
        """Return equal weights without running the full optimize() path."""
        return self.optimize().weights

    def get_return_stats(self) -> Dict[str, Dict[str, float]]:
        """No history tracked; returns zeroed stats for each strategy."""
        return {s: {"mean": 0.0, "std": 0.0, "sharpe": 0.0, "n_obs": 0.0}
                for s in self.strategies}

    def should_rebalance(self, current_weights: Dict[str, float]) -> bool:
        """Always returns False: equal weights never require rebalancing."""
        return False

    def to_capital_allocation(
        self,
        total_capital: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Convert weights to dollar amounts."""
        if weights is None:
            weights = self.get_current_weights()
        return {s: w * total_capital for s, w in weights.items()}

    # ------------------------------------------------------------------
    # No-op methods for interface compatibility
    # ------------------------------------------------------------------

    def record_return(self, *args, **kwargs) -> None:  # noqa: D401
        """No-op: EqualWeightOptimizer does not track returns."""

    def add_view(self, *args, **kwargs) -> None:  # noqa: D401
        """No-op: EqualWeightOptimizer does not accept views."""

    def clear_views(self) -> None:
        """No-op: no views to clear."""
