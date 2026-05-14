"""Gaussian Process-based Bayesian optimization for ARGUS parameter tuning.

Upgrade (2026-04 peak-potential):
- Matern-5/2 kernel option alongside RBF: better suited for non-smooth
  financial objective functions where RBF over-smooths.
- Lower Confidence Bound (LCB) acquisition in addition to EI: LCB is more
  aggressive in exploitation, useful when sample budgets are tight.
- Multi-restart suggest: runs acquisition maximisation from N random seeds
  and returns the global best, reducing sensitivity to local optima.
- Warm-start: when a new param is registered with existing observations
  (e.g. from a checkpoint), the GP is immediately fitted.
- Marginal likelihood hyperparameter fitting: lengthscale and noise are
  optimised via grid search over log-marginal-likelihood each time the GP
  is fitted, using pure numpy.
- Observation windowing: keeps only the most recent max_obs observations
  per parameter to prevent GP scaling issues as data accumulates.
- Per-parameter best_n tracking: returns top-N candidates rather than
  just the single best, for ensemble probing.

All operations remain pure numpy - no scipy or torch required.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _rbf_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    lengthscale: float,
    variance: float,
) -> np.ndarray:
    """Radial Basis Function (squared exponential) kernel."""
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 1)
    diff = x1 - x2.T
    sq = diff * diff
    ls = max(1e-6, float(lengthscale))
    var = max(1e-12, float(variance))
    return var * np.exp(-0.5 * sq / (ls * ls))


def _matern52_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    lengthscale: float,
    variance: float,
) -> np.ndarray:
    """Matern 5/2 kernel — less smooth than RBF, often better for finance."""
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 1)
    ls = max(1e-6, float(lengthscale))
    var = max(1e-12, float(variance))
    r = np.abs(x1 - x2.T) / ls
    sqrt5_r = math.sqrt(5.0) * r
    return var * (1.0 + sqrt5_r + (5.0 / 3.0) * r ** 2) * np.exp(-sqrt5_r)


# ---------------------------------------------------------------------------
# Gaussian Process regressor
# ---------------------------------------------------------------------------


@dataclass
class GaussianProcessRegressor:
    """Zero-mean GP regressor with RBF or Matern-5/2 kernel.

    When ``fit_hyperparams=True`` a small grid search over lengthscale and
    noise is run on each :meth:`fit` call to maximise the log marginal
    likelihood. This keeps hyperparameters data-driven without requiring
    scipy.
    """

    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 0.05
    jitter: float = 1e-6
    kernel: Literal["rbf", "matern52"] = "matern52"
    fit_hyperparams: bool = True

    _X: np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    _y: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    _L: Optional[np.ndarray] = None
    _alpha: Optional[np.ndarray] = None
    _y_mean: float = 0.0

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.kernel == "matern52":
            return _matern52_kernel(x1, x2, self.lengthscale, self.variance)
        return _rbf_kernel(x1, x2, self.lengthscale, self.variance)

    def _k_diag(self) -> float:
        """k(x, x) for the chosen kernel (always == variance)."""
        return float(self.variance)

    def _log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute log marginal likelihood for current hyperparams."""
        try:
            K = self._kernel(X, X) + (self.noise + self.jitter) * np.eye(len(X))
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = (
            -0.5 * float(y @ alpha)
            - float(np.sum(np.log(np.diag(L))))
            - 0.5 * len(y) * math.log(2 * math.pi)
        )
        return float(lml)

    def _optimise_hyperparams(self, X: np.ndarray, y: np.ndarray) -> None:
        """Grid search over lengthscale and noise for best log marginal likelihood."""
        span = float(X.max() - X.min()) if len(X) > 1 else 1.0
        ls_candidates = [span * f for f in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)]
        noise_candidates = [1e-3, 0.01, 0.05, 0.1, 0.5]
        best_lml = -math.inf
        best_ls, best_noise = self.lengthscale, self.noise
        for ls in ls_candidates:
            for ns in noise_candidates:
                orig_ls, orig_noise = self.lengthscale, self.noise
                self.lengthscale, self.noise = ls, ns
                lml = self._log_marginal_likelihood(X, y)
                if lml > best_lml:
                    best_lml = lml
                    best_ls, best_noise = ls, ns
                self.lengthscale, self.noise = orig_ls, orig_noise
        self.lengthscale = best_ls
        self.noise = best_noise

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.shape[0] == 0:
            self._X = X
            self._y = y
            self._L = None
            self._alpha = None
            self._y_mean = 0.0
            return

        # Adapt variance to output scale.
        y_std = float(np.std(y)) if y.size > 1 else 1.0
        self.variance = max(y_std * y_std, 1e-6)

        if self.fit_hyperparams and len(X) >= 4:
            self._optimise_hyperparams(X.flatten(), y)
        elif not self.fit_hyperparams:
            span = float(X.max() - X.min())
            if span > 0:
                self.lengthscale = max(span / 4.0, 1e-3)

        self._y_mean = float(np.mean(y))
        y_centered = y - self._y_mean

        K = self._kernel(X, X) + (self.noise + self.jitter) * np.eye(X.shape[0])
        try:
            self._L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K = K + 1e-3 * np.eye(K.shape[0])
            self._L = np.linalg.cholesky(K)

        z = np.linalg.solve(self._L, y_centered)
        self._alpha = np.linalg.solve(self._L.T, z)
        self._X = X
        self._y = y

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_star = np.asarray(X_star, dtype=np.float64).reshape(-1, 1)
        n_star = X_star.shape[0]
        if self._L is None or self._alpha is None or self._X.shape[0] == 0:
            mean = np.full(n_star, self._y_mean)
            var = np.full(n_star, self.variance)
            return mean, var
        K_s = self._kernel(self._X, X_star)
        mean = self._y_mean + K_s.T @ self._alpha
        v = np.linalg.solve(self._L, K_s)
        K_ss_diag = np.full(n_star, self._k_diag())
        var = K_ss_diag - np.sum(v * v, axis=0)
        var = np.clip(var, 1e-12, None)
        return mean, var

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_obs": int(self._X.shape[0]),
            "lengthscale": float(self.lengthscale),
            "variance": float(self.variance),
            "noise": float(self.noise),
            "y_mean": float(self._y_mean),
            "kernel": self.kernel,
        }


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def _standard_normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _standard_normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class AcquisitionFunction:
    """Expected Improvement and Lower Confidence Bound acquisitions."""

    xi: float = 0.01
    kappa: float = 2.0  # LCB exploration weight

    def expected_improvement(
        self,
        mean: np.ndarray,
        variance: np.ndarray,
        y_best: float,
    ) -> np.ndarray:
        sigma = np.sqrt(np.clip(variance, 1e-12, None))
        improvement = mean - y_best - self.xi
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sigma > 0, improvement / sigma, 0.0)
            ei = improvement * _standard_normal_cdf(z) + sigma * _standard_normal_pdf(z)
            ei = np.where(sigma <= 0, 0.0, ei)
        return np.clip(ei, 0.0, None)

    def lower_confidence_bound(
        self,
        mean: np.ndarray,
        variance: np.ndarray,
    ) -> np.ndarray:
        """LCB: maximise mean + kappa * std (UCB variant for maximisation)."""
        sigma = np.sqrt(np.clip(variance, 1e-12, None))
        return mean + self.kappa * sigma


# ---------------------------------------------------------------------------
# BayesianOptimizer
# ---------------------------------------------------------------------------


@dataclass
class _ParamState:
    name: str
    min_value: float
    max_value: float
    x_obs: List[float] = field(default_factory=list)
    y_obs: List[float] = field(default_factory=list)
    gp: GaussianProcessRegressor = field(default_factory=GaussianProcessRegressor)
    best_x: Optional[float] = None
    best_y: float = -math.inf


class BayesianOptimizer:
    """Per-parameter Bayesian optimizer with multi-restart and LCB/EI switching.

    Parameters
    ----------
    n_candidates
        Number of random candidates per acquisition evaluation.
    n_restarts
        Number of random restarts for acquisition maximisation. Higher values
        reduce the chance of returning a local optimum at the cost of compute.
    xi
        EI exploration boost (larger -> more exploration).
    kappa
        LCB exploration weight.
    acquisition
        Which acquisition function to use: ``'ei'`` or ``'lcb'``.
    max_obs
        Maximum observations to keep per parameter (rolling window).
    kernel
        GP kernel type: ``'matern52'`` (default) or ``'rbf'``.
    seed
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_candidates: int = 256,
        n_restarts: int = 5,
        xi: float = 0.01,
        kappa: float = 2.0,
        acquisition: Literal["ei", "lcb"] = "ei",
        max_obs: int = 200,
        kernel: Literal["rbf", "matern52"] = "matern52",
        seed: Optional[int] = None,
    ) -> None:
        self.n_candidates = int(max(16, n_candidates))
        self.n_restarts = int(max(1, n_restarts))
        self.acq_fn = AcquisitionFunction(xi=xi, kappa=kappa)
        self.acquisition = acquisition
        self.max_obs = int(max_obs)
        self.kernel = kernel
        self._rng = np.random.default_rng(seed)
        self._params: Dict[str, _ParamState] = {}

    # -- registration --------------------------------------------------------

    def register_param(
        self,
        name: str,
        min_value: float,
        max_value: float,
        warm_x: Optional[List[float]] = None,
        warm_y: Optional[List[float]] = None,
    ) -> None:
        """Register a parameter, optionally warm-starting with existing observations."""
        if min_value >= max_value:
            raise ValueError(f"min_value ({min_value}) must be < max_value ({max_value})")
        gp = GaussianProcessRegressor(kernel=self.kernel)
        state = _ParamState(
            name=name,
            min_value=float(min_value),
            max_value=float(max_value),
            gp=gp,
        )
        if warm_x and warm_y and len(warm_x) == len(warm_y):
            for x, y in zip(warm_x, warm_y):
                state.x_obs.append(float(x))
                state.y_obs.append(float(y))
                if float(y) > state.best_y:
                    state.best_y = float(y)
                    state.best_x = float(x)
            # Immediately fit the GP on warm-start data.
            if len(state.x_obs) >= 3:
                state.gp.fit(np.array(state.x_obs), np.array(state.y_obs))
        self._params[name] = state
        logger.debug("Registered BO param %s in [%s, %s]", name, min_value, max_value)

    def _get(self, name: str) -> _ParamState:
        if name not in self._params:
            raise KeyError(f"Unknown BO parameter: {name}")
        return self._params[name]

    # -- suggestion ----------------------------------------------------------

    def _maximise_acquisition(self, state: _ParamState, mean: np.ndarray, var: np.ndarray, candidates: np.ndarray) -> int:
        """Return index of candidate maximising the selected acquisition."""
        if self.acquisition == "lcb":
            scores = self.acq_fn.lower_confidence_bound(mean, var)
        else:
            scores = self.acq_fn.expected_improvement(mean, var, state.best_y)
        if float(np.max(scores)) <= 0.0:
            return int(np.argmax(mean))
        return int(np.argmax(scores))

    def suggest(self, name: str) -> float:
        state = self._get(name)
        n_obs = len(state.x_obs)

        # Cold start: low, high, centre.
        if n_obs < 3:
            order = [
                state.min_value,
                state.max_value,
                0.5 * (state.min_value + state.max_value),
            ]
            return float(order[n_obs])

        X = np.array(state.x_obs, dtype=np.float64)
        y = np.array(state.y_obs, dtype=np.float64)
        state.gp.fit(X, y)

        best_val: Optional[float] = None
        best_score = -math.inf

        # Multi-restart: evaluate acquisition from n_restarts random candidate sets.
        for _ in range(self.n_restarts):
            candidates = self._rng.uniform(
                state.min_value, state.max_value, size=self.n_candidates
            )
            anchors = [
                state.best_x if state.best_x is not None else 0.5 * (state.min_value + state.max_value),
                0.5 * (state.min_value + state.max_value),
            ]
            candidates = np.concatenate([candidates, np.array(anchors, dtype=np.float64)])
            mean, var = state.gp.predict(candidates)

            if self.acquisition == "lcb":
                scores = self.acq_fn.lower_confidence_bound(mean, var)
            else:
                scores = self.acq_fn.expected_improvement(mean, var, state.best_y)

            idx = int(np.argmax(scores)) if float(np.max(scores)) > 0.0 else int(np.argmax(mean))
            if float(scores[idx]) > best_score:
                best_score = float(scores[idx])
                best_val = float(candidates[idx])

        return float(best_val) if best_val is not None else 0.5 * (state.min_value + state.max_value)

    def suggest_top_n(self, name: str, n: int = 3) -> List[float]:
        """Return the top-N distinct candidate values by acquisition score."""
        state = self._get(name)
        if len(state.x_obs) < 3:
            return [self.suggest(name)]
        X = np.array(state.x_obs, dtype=np.float64)
        y = np.array(state.y_obs, dtype=np.float64)
        state.gp.fit(X, y)
        candidates = self._rng.uniform(state.min_value, state.max_value, size=self.n_candidates * self.n_restarts)
        mean, var = state.gp.predict(candidates)
        if self.acquisition == "lcb":
            scores = self.acq_fn.lower_confidence_bound(mean, var)
        else:
            scores = self.acq_fn.expected_improvement(mean, var, state.best_y)
        top_idx = np.argsort(scores)[::-1][:n]
        return [float(candidates[i]) for i in top_idx]

    # -- observation update --------------------------------------------------

    def update(self, name: str, value: float, outcome_pnl: float) -> None:
        state = self._get(name)
        value = float(np.clip(value, state.min_value, state.max_value))
        outcome_pnl = float(outcome_pnl)
        state.x_obs.append(value)
        state.y_obs.append(outcome_pnl)
        # Rolling window.
        if len(state.x_obs) > self.max_obs:
            state.x_obs = state.x_obs[-self.max_obs:]
            state.y_obs = state.y_obs[-self.max_obs:]
        if outcome_pnl > state.best_y:
            state.best_y = outcome_pnl
            state.best_x = value
        logger.debug(
            "BO update %s: value=%.6f pnl=%.4f best=%.4f",
            name, value, outcome_pnl, state.best_y,
        )

    # -- inspection ----------------------------------------------------------

    def get_best(self, name: str) -> Optional[float]:
        return self._get(name).best_x

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n_params": len(self._params),
            "xi": float(self.acq_fn.xi),
            "kappa": float(self.acq_fn.kappa),
            "n_candidates": int(self.n_candidates),
            "n_restarts": int(self.n_restarts),
            "acquisition": self.acquisition,
            "kernel": self.kernel,
            "params": {},
        }
        for name, state in self._params.items():
            out["params"][name] = {
                "n_obs": len(state.x_obs),
                "best_x": state.best_x,
                "best_y": None if not math.isfinite(state.best_y) else float(state.best_y),
                "min_value": state.min_value,
                "max_value": state.max_value,
                "gp": state.gp.snapshot(),
            }
        return out


__all__ = [
    "GaussianProcessRegressor",
    "AcquisitionFunction",
    "BayesianOptimizer",
]
