from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau, norm, rankdata, t

logger = logging.getLogger(__name__)

_EPS = 1e-10
_TRADING_DAYS = 252.0


class CopulaType(str, Enum):
    """Supported copula families for dependency modelling."""

    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"
    VINE = "vine"


@dataclass(slots=True)
class _ViewSpec:
    assets: list[str]
    coeffs: list[float]
    target_return: float
    confidence: float


def _regularize_covariance(cov_matrix: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance matrix must be square")

    cov = (cov + cov.T) / 2.0
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance matrix contains non-finite values")

    min_eigenvalue = float(np.min(np.linalg.eigvalsh(cov)))
    if min_eigenvalue < _EPS:
        cov = cov + np.eye(cov.shape[0], dtype=float) * (_EPS - min_eigenvalue + 1e-8)
    return cov


def _normalise_weights(weights: np.ndarray) -> np.ndarray:
    vector = np.asarray(weights, dtype=float).reshape(-1)
    total = float(np.sum(vector))
    if vector.size == 0:
        return vector
    if total <= _EPS:
        return np.full(vector.shape[0], 1.0 / vector.shape[0], dtype=float)
    return vector / total


def _coerce_returns_matrix(
    returns: np.ndarray | Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, list[str]]:
    if isinstance(returns, Mapping):
        symbols = [str(symbol) for symbol in returns.keys()]
        if not symbols:
            raise ValueError("returns mapping cannot be empty")
        series = [np.asarray(returns[symbol], dtype=float).reshape(-1) for symbol in symbols]
        min_length = min(arr.shape[0] for arr in series)
        if min_length < 2:
            raise ValueError("returns must contain at least two observations")
        matrix = np.column_stack([arr[-min_length:] for arr in series])
    else:
        matrix = np.asarray(returns, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            raise ValueError("returns must be a two-dimensional matrix with at least two rows")
        symbols = [f"asset_{idx}" for idx in range(matrix.shape[1])]

    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite = np.isfinite(values)
        if finite.all():
            continue
        fill_value = float(np.mean(values[finite])) if finite.any() else 0.0
        values[~finite] = fill_value
        matrix[:, column] = values

    return matrix, symbols


def _empirical_tail_dependence(
    uniforms: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    n_assets = uniforms.shape[1]
    lower = np.zeros((n_assets, n_assets), dtype=float)
    upper = np.zeros((n_assets, n_assets), dtype=float)

    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                lower[i, j] = 1.0
                upper[i, j] = 1.0
                continue

            lower_joint = float(np.mean((uniforms[:, i] <= alpha) & (uniforms[:, j] <= alpha)))
            upper_joint = float(np.mean((uniforms[:, i] >= 1.0 - alpha) & (uniforms[:, j] >= 1.0 - alpha)))
            lower[i, j] = lower_joint / max(alpha, _EPS)
            upper[i, j] = upper_joint / max(alpha, _EPS)

    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)


class CopulaFitter:
    """Fit copula-inspired dependency structures from asset returns."""

    def __init__(self) -> None:
        self._copula_type: CopulaType | None = None
        self._correlation_matrix: np.ndarray | None = None
        self._tail_dependence: dict[str, np.ndarray] | None = None
        self._degrees_of_freedom: float = 8.0
        self._uniform_returns: np.ndarray | None = None

    def fit_gaussian_copula(self, returns: np.ndarray) -> np.ndarray:
        """Fit a Gaussian copula from pseudo-observations."""
        uniforms = self._pseudo_observations(returns)
        gaussian_scores = norm.ppf(np.clip(uniforms, 1e-6, 1.0 - 1e-6))
        correlation = _regularize_covariance(np.corrcoef(gaussian_scores, rowvar=False))
        n_assets = correlation.shape[0]
        self._copula_type = CopulaType.GAUSSIAN
        self._correlation_matrix = correlation
        self._uniform_returns = uniforms
        self._tail_dependence = {
            "lower": np.eye(n_assets, dtype=float),
            "upper": np.eye(n_assets, dtype=float),
        }
        return correlation

    def fit_student_t_copula(self, returns: np.ndarray) -> np.ndarray:
        """Fit a Student-t copula and estimate symmetric tail dependence."""
        uniforms = self._pseudo_observations(returns)
        nu = self._estimate_degrees_of_freedom(np.asarray(returns, dtype=float))
        t_scores = t.ppf(np.clip(uniforms, 1e-6, 1.0 - 1e-6), df=nu)
        correlation = _regularize_covariance(np.corrcoef(t_scores, rowvar=False))
        lambda_matrix = self._student_t_tail_dependence(correlation, nu)

        self._copula_type = CopulaType.STUDENT_T
        self._degrees_of_freedom = nu
        self._correlation_matrix = correlation
        self._uniform_returns = uniforms
        self._tail_dependence = {"lower": lambda_matrix, "upper": lambda_matrix.copy()}
        return correlation

    def fit_vine_copula(self, returns: np.ndarray) -> np.ndarray:
        """Fit a vine-style pairwise dependency surface for higher dimensions."""
        matrix = np.asarray(returns, dtype=float)
        uniforms = self._pseudo_observations(matrix)
        n_assets = matrix.shape[1]
        correlation = np.eye(n_assets, dtype=float)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                tau_value, _ = kendalltau(matrix[:, i], matrix[:, j])
                tau_value = 0.0 if not np.isfinite(tau_value) else float(tau_value)
                correlation[i, j] = np.sin(np.pi * tau_value / 2.0)
                correlation[j, i] = correlation[i, j]

        empirical_corr = np.corrcoef(norm.ppf(np.clip(uniforms, 1e-6, 1.0 - 1e-6)), rowvar=False)
        correlation = _regularize_covariance(0.65 * correlation + 0.35 * empirical_corr)
        lower_tail, upper_tail = _empirical_tail_dependence(uniforms)

        self._copula_type = CopulaType.VINE
        self._correlation_matrix = correlation
        self._uniform_returns = uniforms
        self._tail_dependence = {"lower": lower_tail, "upper": upper_tail}
        return correlation

    def get_correlation_matrix(self) -> np.ndarray:
        """Return the fitted copula correlation matrix."""
        if self._correlation_matrix is None:
            raise ValueError("no copula has been fitted yet")
        return self._correlation_matrix.copy()

    def get_tail_dependence(self) -> dict[str, np.ndarray]:
        """Return lower and upper tail dependence coefficient matrices."""
        if self._tail_dependence is None:
            raise ValueError("no copula has been fitted yet")
        return {
            "lower": self._tail_dependence["lower"].copy(),
            "upper": self._tail_dependence["upper"].copy(),
        }

    @staticmethod
    def _pseudo_observations(returns: np.ndarray) -> np.ndarray:
        matrix = np.asarray(returns, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("returns must be two-dimensional")
        n_obs = matrix.shape[0]
        ranks = np.column_stack([rankdata(matrix[:, idx], method="average") for idx in range(matrix.shape[1])])
        return ranks / (n_obs + 1.0)

    @staticmethod
    def _estimate_degrees_of_freedom(returns: np.ndarray) -> float:
        centered = returns - np.mean(returns, axis=0, keepdims=True)
        variance = np.mean(centered ** 2, axis=0)
        variance = np.where(variance < _EPS, _EPS, variance)
        kurtosis = np.mean(centered ** 4, axis=0) / (variance ** 2)
        excess_kurtosis = float(np.nanmean(np.maximum(kurtosis - 3.0, 0.0)))
        if excess_kurtosis <= _EPS:
            return 30.0
        nu = 6.0 / excess_kurtosis + 4.0
        return float(np.clip(nu, 4.5, 30.0))

    @staticmethod
    def _student_t_tail_dependence(correlation: np.ndarray, nu: float) -> np.ndarray:
        n_assets = correlation.shape[0]
        lambda_matrix = np.eye(n_assets, dtype=float)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                rho = float(np.clip(correlation[i, j], -0.999, 0.999))
                value = 2.0 * t.cdf(
                    -np.sqrt((nu + 1.0) * (1.0 - rho) / max(1.0 + rho, _EPS)),
                    df=nu + 1.0,
                )
                lambda_matrix[i, j] = value
                lambda_matrix[j, i] = value
        return np.clip(lambda_matrix, 0.0, 1.0)


class CopulaBlackLitterman:
    """Black-Litterman optimizer extended with copula-based dependency modelling."""

    def __init__(self, market_caps: Sequence[float] | Mapping[str, float], risk_aversion: float, tau: float) -> None:
        self.risk_aversion = float(risk_aversion)
        self.tau = float(tau)
        if self.risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")

        if isinstance(market_caps, Mapping):
            self.symbols = [str(symbol) for symbol in market_caps.keys()]
            weights = np.array([float(weight) for weight in market_caps.values()], dtype=float)
        else:
            weights = np.asarray(market_caps, dtype=float).reshape(-1)
            self.symbols = [f"asset_{idx}" for idx in range(weights.shape[0])]

        if weights.size == 0:
            raise ValueError("market_caps cannot be empty")

        self.market_weights = _normalise_weights(np.maximum(weights, 0.0))
        self.copula_type = CopulaType.GAUSSIAN
        self.copula_fitter = CopulaFitter()
        self._returns: np.ndarray | None = None
        self._prior_covariance: np.ndarray | None = None
        self._copula_covariance: np.ndarray | None = None
        self._posterior_returns: np.ndarray | None = None
        self._posterior_covariance: np.ndarray | None = None
        self._optimal_weights: np.ndarray | None = None
        self._views: list[_ViewSpec] = []

    def set_views(self, views: Sequence[Any], confidence: float | Sequence[float]) -> None:
        """Set Black-Litterman views and associated confidence levels."""
        confidences = self._expand_confidences(confidence, len(views))
        parsed: list[_ViewSpec] = []
        for idx, view in enumerate(views):
            parsed.append(self._parse_view(view, confidences[idx]))
        self._views = parsed

    def set_copula(self, copula_type: CopulaType | str) -> None:
        """Select the copula family used to model dependencies."""
        self.copula_type = copula_type if isinstance(copula_type, CopulaType) else CopulaType(str(copula_type).lower())

    def optimize(self) -> dict[str, float]:
        """Compute optimal portfolio weights using copula-adjusted Black-Litterman."""
        returns = self._require_returns()
        self._fit_copula(returns)
        equilibrium = self.risk_aversion * self._require_copula_covariance() @ self.market_weights
        posterior_returns = self._build_posterior_returns(equilibrium)
        posterior_covariance = self._build_posterior_covariance()
        weights = self._solve_weights(posterior_returns, posterior_covariance)

        self._posterior_returns = posterior_returns
        self._posterior_covariance = posterior_covariance
        self._optimal_weights = weights
        return {symbol: float(weights[idx]) for idx, symbol in enumerate(self.symbols)}

    def get_efficient_frontier(self, n_points: int) -> list[dict[str, Any]]:
        """Generate a long-only efficient frontier under the current copula model."""
        if n_points <= 1:
            raise ValueError("n_points must be greater than 1")
        if self._optimal_weights is None:
            self.optimize()

        expected_returns = self._require_posterior_returns()
        covariance = self._require_posterior_covariance()
        target_min = float(np.min(expected_returns))
        target_max = float(np.max(expected_returns))
        targets = np.linspace(target_min, target_max, n_points)
        frontier: list[dict[str, Any]] = []

        for target in targets:
            weights = self._solve_target_return_weights(expected_returns, covariance, target)
            portfolio_return = float(weights @ expected_returns)
            portfolio_volatility = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
            frontier.append(
                {
                    "target_return": portfolio_return,
                    "volatility": portfolio_volatility,
                    "weights": {symbol: float(weights[idx]) for idx, symbol in enumerate(self.symbols)},
                }
            )

        return frontier

    def compute_cvar(self, alpha: float = 0.05) -> float:
        """Compute portfolio CVaR using copula-adjusted historical scenarios."""
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie between 0 and 1")
        returns = self._require_returns()
        weights = self._optimal_weights if self._optimal_weights is not None else np.array(list(self.optimize().values()))
        scenario_returns = self._copula_adjusted_scenarios(returns)
        portfolio_returns = scenario_returns @ weights
        cutoff = float(np.quantile(portfolio_returns, alpha))
        tail_losses = portfolio_returns[portfolio_returns <= cutoff]
        if tail_losses.size == 0:
            return float(-cutoff)
        return float(-np.mean(tail_losses))

    def _set_returns(self, returns: np.ndarray, symbols: Sequence[str] | None = None) -> None:
        matrix = np.asarray(returns, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            raise ValueError("returns must be a two-dimensional matrix with at least two rows")
        self._returns = matrix
        if symbols is not None:
            self.symbols = [str(symbol) for symbol in symbols]
        if len(self.symbols) != matrix.shape[1]:
            raise ValueError("market_caps dimension must match returns columns")

    def _fit_copula(self, returns: np.ndarray) -> None:
        sample_covariance = _regularize_covariance(np.cov(returns, rowvar=False) * _TRADING_DAYS)
        if self.copula_type == CopulaType.STUDENT_T:
            correlation = self.copula_fitter.fit_student_t_copula(returns)
        elif self.copula_type == CopulaType.VINE:
            correlation = self.copula_fitter.fit_vine_copula(returns)
        else:
            correlation = self.copula_fitter.fit_gaussian_copula(returns)
            if self.copula_type in {CopulaType.CLAYTON, CopulaType.GUMBEL, CopulaType.FRANK}:
                self._override_archimedean_tail_dependence(returns, correlation)

        vol = np.sqrt(np.maximum(np.diag(sample_covariance), _EPS))
        copula_covariance = correlation * np.outer(vol, vol)
        tail = self.copula_fitter.get_tail_dependence()
        tail_intensity = 0.5 * (tail["lower"] + tail["upper"])
        self._prior_covariance = sample_covariance
        self._copula_covariance = _regularize_covariance(copula_covariance * (1.0 + 0.25 * tail_intensity))

    def _override_archimedean_tail_dependence(self, returns: np.ndarray, correlation: np.ndarray) -> None:
        uniforms = self.copula_fitter._pseudo_observations(returns)
        n_assets = correlation.shape[0]
        lower = np.eye(n_assets, dtype=float)
        upper = np.eye(n_assets, dtype=float)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                tau_value, _ = kendalltau(returns[:, i], returns[:, j])
                tau_value = 0.0 if not np.isfinite(tau_value) else float(np.clip(tau_value, -0.95, 0.95))
                if self.copula_type == CopulaType.CLAYTON:
                    theta = max(2.0 * tau_value / max(1.0 - tau_value, _EPS), 0.0)
                    lower_value = 2.0 ** (-1.0 / max(theta, _EPS)) if theta > 0.0 else 0.0
                    upper_value = 0.0
                elif self.copula_type == CopulaType.GUMBEL:
                    theta = max(1.0 / max(1.0 - tau_value, _EPS), 1.0)
                    lower_value = 0.0
                    upper_value = 2.0 - 2.0 ** (1.0 / theta)
                else:
                    theta = float(np.sign(tau_value) * np.sqrt(max(abs(tau_value), 0.0)) * 5.0)
                    lower_value = 0.0
                    upper_value = 0.0

                lower[i, j] = lower[j, i] = lower_value
                upper[i, j] = upper[j, i] = upper_value

        empirical_lower, empirical_upper = _empirical_tail_dependence(uniforms)
        self.copula_fitter._tail_dependence = {
            "lower": np.clip(0.7 * lower + 0.3 * empirical_lower, 0.0, 1.0),
            "upper": np.clip(0.7 * upper + 0.3 * empirical_upper, 0.0, 1.0),
        }

    def _build_posterior_returns(self, equilibrium: np.ndarray) -> np.ndarray:
        if not self._views:
            return equilibrium.copy()

        covariance = self._require_copula_covariance()
        p_matrix = np.zeros((len(self._views), len(self.symbols)), dtype=float)
        q_vector = np.zeros(len(self._views), dtype=float)
        omega = np.zeros((len(self._views), len(self._views)), dtype=float)
        index = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        tau_cov = self.tau * covariance

        for row_idx, view in enumerate(self._views):
            row = np.zeros(len(self.symbols), dtype=float)
            for asset, coeff in zip(view.assets, view.coeffs):
                if asset in index:
                    row[index[asset]] = coeff
            p_matrix[row_idx] = row
            q_vector[row_idx] = view.target_return
            projected_variance = float(row @ tau_cov @ row.T)
            omega[row_idx, row_idx] = max(projected_variance * ((1.0 - view.confidence) / view.confidence), _EPS)

        tau_cov_inv = np.linalg.inv(tau_cov)
        omega_inv = np.linalg.inv(omega)
        posterior_precision = tau_cov_inv + p_matrix.T @ omega_inv @ p_matrix
        rhs = tau_cov_inv @ equilibrium + p_matrix.T @ omega_inv @ q_vector
        return np.linalg.solve(posterior_precision, rhs)

    def _build_posterior_covariance(self) -> np.ndarray:
        covariance = self._require_copula_covariance()
        if not self._views:
            return covariance.copy()
        tail = self.copula_fitter.get_tail_dependence()
        tail_penalty = 0.5 * (tail["lower"] + tail["upper"])
        return _regularize_covariance(covariance * (1.0 + self.tau * tail_penalty))

    def _solve_weights(self, expected_returns: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        n_assets = covariance.shape[0]
        initial = np.full(n_assets, 1.0 / n_assets, dtype=float)

        def objective(weights: np.ndarray) -> float:
            portfolio_return = float(weights @ expected_returns)
            portfolio_risk = float(weights @ covariance @ weights)
            return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_risk)

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            logger.warning("Copula BL optimisation fell back to unconstrained weights: %s", result.message)
            raw = np.linalg.solve(self.risk_aversion * covariance, expected_returns)
            return _normalise_weights(np.maximum(raw, 0.0))
        return _normalise_weights(np.maximum(result.x, 0.0))

    def _solve_target_return_weights(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        target_return: float,
    ) -> np.ndarray:
        n_assets = covariance.shape[0]
        initial = self._optimal_weights.copy() if self._optimal_weights is not None else np.full(n_assets, 1.0 / n_assets)

        def objective(weights: np.ndarray) -> float:
            return float(weights @ covariance @ weights)

        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: float(w @ expected_returns) - target_return},
        )
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            return _normalise_weights(initial)
        return _normalise_weights(np.maximum(result.x, 0.0))

    def _copula_adjusted_scenarios(self, returns: np.ndarray) -> np.ndarray:
        covariance = self._require_copula_covariance()
        sample_covariance = self._prior_covariance if self._prior_covariance is not None else covariance
        vol_base = np.sqrt(np.maximum(np.diag(sample_covariance), _EPS))
        vol_target = np.sqrt(np.maximum(np.diag(covariance), _EPS))
        scaled = returns * (vol_target / vol_base)
        tail = self.copula_fitter.get_tail_dependence()
        tail_multiplier = 1.0 + np.mean(0.5 * (tail["lower"] + tail["upper"]), axis=1)
        downside = np.minimum(scaled, 0.0) * tail_multiplier
        upside = np.maximum(scaled, 0.0)
        return downside + upside

    def _expand_confidences(self, confidence: float | Sequence[float], n_views: int) -> np.ndarray:
        if isinstance(confidence, Sequence) and not isinstance(confidence, (str, bytes)):
            values = np.asarray(confidence, dtype=float).reshape(-1)
        else:
            values = np.full(n_views, float(confidence), dtype=float)
        if values.shape[0] != n_views:
            raise ValueError("confidence length must match number of views")
        return np.clip(values, 1e-3, 1.0)

    def _parse_view(self, view: Any, confidence: float) -> _ViewSpec:
        if isinstance(view, Mapping):
            raw_assets = view.get("assets", [])
            assets = [str(asset) for asset in raw_assets] if isinstance(raw_assets, Sequence) else []
            raw_coeffs = view.get("coeffs", [1.0] * len(assets))
            raw_target = view.get("return", view.get("expected_return", 0.0))
            coeffs = [self._safe_float(coeff, 0.0) for coeff in raw_coeffs] if isinstance(raw_coeffs, Sequence) else []
            target_return = self._safe_float(raw_target, 0.0)
        else:
            row = np.asarray(view, dtype=float).reshape(-1)
            if row.shape[0] != len(self.symbols) + 1:
                raise ValueError("array views must provide N coefficients followed by target return")
            active = np.where(np.abs(row[:-1]) > _EPS)[0]
            assets = [self.symbols[idx] for idx in active]
            coeffs = [float(row[idx]) for idx in active]
            target_return = float(row[-1])

        if not assets:
            raise ValueError("each view must reference at least one asset")
        if len(coeffs) != len(assets):
            raise ValueError("view coeffs length must match assets length")
        return _ViewSpec(assets=assets, coeffs=coeffs, target_return=target_return, confidence=float(confidence))

    def _require_returns(self) -> np.ndarray:
        if self._returns is None:
            raise ValueError("returns have not been set for optimisation")
        return self._returns

    def _require_copula_covariance(self) -> np.ndarray:
        if self._copula_covariance is None:
            raise ValueError("copula covariance has not been estimated yet")
        return self._copula_covariance

    def _require_posterior_returns(self) -> np.ndarray:
        if self._posterior_returns is None:
            raise ValueError("posterior returns have not been computed yet")
        return self._posterior_returns

    def _require_posterior_covariance(self) -> np.ndarray:
        if self._posterior_covariance is None:
            raise ValueError("posterior covariance has not been computed yet")
        return self._posterior_covariance

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


class MacroeconomicViews:
    """Build Black-Litterman views from macroeconomic assumptions."""

    def __init__(self) -> None:
        self._macro_views: list[dict[str, Any]] = []

    def add_cycle_view(self, cycle_state: str, asset_impacts: Mapping[str, float]) -> None:
        """Add a macro-cycle view such as expansion, slowdown, or recession."""
        scale = {
            "expansion": 1.0,
            "recovery": 0.8,
            "slowdown": 0.6,
            "recession": -1.0,
            "stagflation": -0.8,
        }.get(str(cycle_state).strip().lower(), 0.5)
        for asset, impact in asset_impacts.items():
            self._macro_views.append(
                {
                    "assets": [str(asset)],
                    "coeffs": [1.0],
                    "return": float(scale * impact),
                    "confidence": 0.65,
                }
            )

    def add_inflation_view(self, expected_inflation: Mapping[str, float] | float) -> None:
        """Add inflation-linked return adjustments."""
        if isinstance(expected_inflation, Mapping):
            mapping = expected_inflation
        else:
            mapping = {"inflation_sensitive_assets": float(expected_inflation)}

        for asset, inflation in mapping.items():
            self._macro_views.append(
                {
                    "assets": [str(asset)],
                    "coeffs": [1.0],
                    "return": float(0.4 * inflation),
                    "confidence": 0.55,
                }
            )

    def add_rate_view(self, expected_rates: Mapping[str, float] | float) -> None:
        """Add rate-sensitive views from expected policy or market rates."""
        if isinstance(expected_rates, Mapping):
            mapping = expected_rates
        else:
            mapping = {"rate_sensitive_assets": float(expected_rates)}

        for asset, rate in mapping.items():
            self._macro_views.append(
                {
                    "assets": [str(asset)],
                    "coeffs": [1.0],
                    "return": float(-0.3 * rate),
                    "confidence": 0.60,
                }
            )

    def generate_bl_views(self) -> list[dict[str, Any]]:
        """Generate Black-Litterman-compatible view dictionaries."""
        return [dict(view) for view in self._macro_views]


class PortfolioOptimizer:
    """Top-level copula-enhanced portfolio optimizer with backtesting helpers."""

    def __init__(
        self,
        market_caps: Sequence[float] | Mapping[str, float],
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ) -> None:
        self.market_caps = market_caps
        self.risk_aversion = float(risk_aversion)
        self.tau = float(tau)
        self._optimizer = CopulaBlackLitterman(market_caps, risk_aversion=self.risk_aversion, tau=self.tau)
        self._returns: np.ndarray | None = None
        self._symbols: list[str] = list(self._optimizer.symbols)
        self._latest_weights: dict[str, float] | None = None
        self._backtest_equity_curve: np.ndarray | None = None
        self._backtest_returns: np.ndarray | None = None

    def optimize_with_copula(
        self,
        returns: np.ndarray | Mapping[str, Sequence[float]],
        views: Sequence[Any] | None,
        copula_type: CopulaType | str,
    ) -> dict[str, float]:
        """Run the full copula-based Black-Litterman optimisation workflow."""
        matrix, symbols = _coerce_returns_matrix(returns)
        self._returns = matrix
        self._symbols = symbols
        self._optimizer._set_returns(matrix, symbols=symbols)
        self._optimizer.set_copula(copula_type)
        if views:
            confidences = []
            for view in views:
                if isinstance(view, Mapping):
                    raw_confidence = view.get("confidence", 0.5)
                    confidences.append(self._safe_float(raw_confidence, 0.5))
                else:
                    confidences.append(0.5)
            self._optimizer.set_views(views, confidences)
        else:
            self._optimizer.set_views([], [])
        self._latest_weights = self._optimizer.optimize()
        return dict(self._latest_weights)

    def backtest(self, start_date: Any, end_date: Any) -> dict[str, Any]:
        """Backtest the latest copula-optimised weights over the stored return history."""
        if self._returns is None:
            raise ValueError("run optimize_with_copula before backtesting")
        if self._latest_weights is None:
            raise ValueError("no portfolio weights are available for backtesting")

        start_idx, end_idx = self._resolve_backtest_window(start_date, end_date, self._returns.shape[0])
        window = self._returns[start_idx:end_idx]
        weights = np.array([self._latest_weights[symbol] for symbol in self._symbols], dtype=float)
        portfolio_returns = window @ weights
        equity_curve = np.cumprod(1.0 + portfolio_returns)

        self._backtest_returns = portfolio_returns
        self._backtest_equity_curve = equity_curve
        return {
            "start_index": start_idx,
            "end_index": end_idx,
            "total_return": float(equity_curve[-1] - 1.0) if equity_curve.size else 0.0,
            "equity_curve": equity_curve,
            "portfolio_returns": portfolio_returns,
        }

    def compute_risk_metrics(self) -> dict[str, float]:
        """Compute VaR, CVaR, max drawdown, and Sharpe ratio."""
        if self._returns is None or self._latest_weights is None:
            raise ValueError("run optimize_with_copula before computing risk metrics")

        if self._backtest_returns is not None:
            portfolio_returns = self._backtest_returns
            equity_curve = self._backtest_equity_curve
        else:
            weights = np.array([self._latest_weights[symbol] for symbol in self._symbols], dtype=float)
            portfolio_returns = self._returns @ weights
            equity_curve = np.cumprod(1.0 + portfolio_returns)

        var_95 = float(-np.quantile(portfolio_returns, 0.05))
        cvar_95 = float(self._optimizer.compute_cvar(alpha=0.05))
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / np.maximum(running_max, _EPS)
        sharpe = float(np.mean(portfolio_returns) / max(np.std(portfolio_returns, ddof=1), _EPS) * np.sqrt(_TRADING_DAYS))

        return {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": float(np.min(drawdowns)),
            "sharpe": sharpe,
        }

    @staticmethod
    def _resolve_backtest_window(start_date: Any, end_date: Any, n_obs: int) -> tuple[int, int]:
        def _to_index(value: Any, default: int) -> int:
            if value is None:
                return default
            if isinstance(value, (int, np.integer)):
                return int(np.clip(int(value), 0, n_obs))
            return default

        start_idx = _to_index(start_date, 0)
        end_idx = _to_index(end_date, n_obs)
        if end_idx <= start_idx:
            raise ValueError("end_date must be greater than start_date")
        return start_idx, end_idx

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
