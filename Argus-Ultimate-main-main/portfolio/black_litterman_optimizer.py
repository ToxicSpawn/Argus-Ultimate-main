from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-10


@dataclass(slots=True)
class MarketView:
    symbol: str
    expected_return: float
    confidence: float
    view_type: str = "absolute"

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol or "").strip()
        self.expected_return = float(self.expected_return)
        self.confidence = float(self.confidence)
        self.view_type = str(self.view_type or "absolute").strip().lower()

        if not self.symbol:
            raise ValueError("MarketView.symbol must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("MarketView.confidence must be between 0.0 and 1.0")
        if self.view_type not in {"absolute", "relative"}:
            raise ValueError("MarketView.view_type must be 'absolute' or 'relative'")


@dataclass(slots=True)
class BlackLittermanConfig:
    risk_aversion: float = 2.5
    tau: float = 0.05
    market_cap_weights: Dict[str, float] = field(default_factory=dict)
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        self.risk_aversion = float(self.risk_aversion)
        self.tau = float(self.tau)
        self.risk_free_rate = float(self.risk_free_rate)
        self.market_cap_weights = {
            str(symbol): float(weight)
            for symbol, weight in (self.market_cap_weights or {}).items()
        }

        if self.risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.market_cap_weights:
            total = sum(max(weight, 0.0) for weight in self.market_cap_weights.values())
            if total <= 0.0:
                raise ValueError("market_cap_weights must contain positive values")
            self.market_cap_weights = {
                symbol: max(weight, 0.0) / total
                for symbol, weight in self.market_cap_weights.items()
            }


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
    total = float(np.sum(weights))
    if total <= _EPS:
        return np.full(weights.shape[0], 1.0 / max(weights.shape[0], 1), dtype=float)
    return weights / total


class BlackLittermanOptimizer:
    """Asset-level Black-Litterman optimizer with Argus integration hooks."""

    def __init__(
        self,
        symbols: Optional[Sequence[str]] = None,
        config: Optional[BlackLittermanConfig] = None,
        min_weight: float = 0.0,
        max_weight: float = 0.35,
        allow_short: bool = False,
    ) -> None:
        self.config = config or BlackLittermanConfig()
        inferred_symbols = list(symbols or self.config.market_cap_weights.keys())
        self.symbols: List[str] = [str(symbol) for symbol in inferred_symbols]
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.allow_short = bool(allow_short)

        if self.symbols:
            if not (0.0 <= self.min_weight <= self.max_weight <= 1.0):
                raise ValueError("Require 0 <= min_weight <= max_weight <= 1")
            if not self.allow_short and self.min_weight * len(self.symbols) > 1.0 + 1e-9:
                raise ValueError("Weight bounds are infeasible for the supplied symbols")

        self._covariance_matrix: Optional[np.ndarray] = None
        self._equilibrium_returns: Optional[np.ndarray] = None
        self._posterior_returns: Optional[np.ndarray] = None
        self._posterior_covariance: Optional[np.ndarray] = None
        self._latest_p: Optional[np.ndarray] = None
        self._latest_q: Optional[np.ndarray] = None
        self._latest_omega: Optional[np.ndarray] = None
        self._latest_weights: Optional[Dict[str, float]] = None

        logger.info(
            "BlackLittermanOptimizer initialised for %d symbol(s) with bounds [%.3f, %.3f]",
            len(self.symbols),
            self.min_weight,
            self.max_weight,
        )

    @property
    def latest_weights(self) -> Dict[str, float]:
        return dict(self._latest_weights or {})

    def set_symbols(self, symbols: Sequence[str]) -> None:
        self.symbols = [str(symbol) for symbol in symbols]

    def estimate_covariance_from_market_data(
        self,
        market_data: Mapping[str, Sequence[float]] | np.ndarray,
        annualization_factor: float = 365.0,
    ) -> np.ndarray:
        """Estimate covariance from symbol->returns or a returns matrix."""
        if isinstance(market_data, np.ndarray):
            matrix = np.asarray(market_data, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            if not self.symbols:
                self.symbols = [f"asset_{idx}" for idx in range(matrix.shape[1])]
        else:
            if not self.symbols:
                self.symbols = list(market_data.keys())
            matrix = self._market_data_to_matrix(market_data)

        if matrix.shape[0] < 2:
            raise ValueError("At least two return observations are required for covariance")

        covariance = np.cov(matrix, rowvar=False) * float(annualization_factor)
        covariance = _regularize_covariance(covariance)
        self._covariance_matrix = covariance
        logger.debug("Estimated covariance matrix with shape %s", covariance.shape)
        return covariance

    def calculate_equilibrium_returns(
        self,
        cov_matrix: np.ndarray,
        market_weights: Mapping[str, float] | Sequence[float],
    ) -> np.ndarray:
        implied_excess = self.calculate_implied_returns(market_weights, cov_matrix)
        equilibrium = implied_excess + self.config.risk_free_rate
        self._equilibrium_returns = equilibrium
        logger.debug("Calculated equilibrium returns for %d symbols", equilibrium.shape[0])
        return equilibrium

    def calculate_implied_returns(
        self,
        market_weights: Mapping[str, float] | Sequence[float],
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        cov = _regularize_covariance(cov_matrix)
        weights = self._weights_to_vector(market_weights)
        implied = self.config.risk_aversion * cov @ weights
        self._covariance_matrix = cov
        return implied

    def incorporate_views(
        self,
        prior_returns: np.ndarray,
        views: List[MarketView],
        tau: float,
    ) -> np.ndarray:
        return self.calculate_posterior_returns(prior_returns, views, tau)

    def calculate_posterior_returns(
        self,
        prior: np.ndarray,
        views: List[MarketView],
        tau: float,
    ) -> np.ndarray:
        if self._covariance_matrix is None:
            raise ValueError("Covariance matrix is required before incorporating views")

        prior_vector = np.asarray(prior, dtype=float).reshape(-1)
        cov = _regularize_covariance(self._covariance_matrix)
        if prior_vector.shape[0] != cov.shape[0]:
            raise ValueError("prior vector length must match covariance dimensions")

        if not views:
            self._posterior_returns = prior_vector.copy()
            self._latest_p = np.zeros((0, cov.shape[0]), dtype=float)
            self._latest_q = np.zeros(0, dtype=float)
            self._latest_omega = np.zeros((0, 0), dtype=float)
            return self._posterior_returns

        P, Q, omega = self._build_view_matrices(views, cov, float(tau))
        tau_cov = float(tau) * cov
        tau_cov_inv = np.linalg.inv(tau_cov)
        omega_inv = np.linalg.inv(omega)

        posterior_precision = tau_cov_inv + P.T @ omega_inv @ P
        rhs = tau_cov_inv @ prior_vector + P.T @ omega_inv @ Q
        posterior_returns = np.linalg.solve(posterior_precision, rhs)

        self._posterior_returns = posterior_returns
        self._latest_p = P
        self._latest_q = Q
        self._latest_omega = omega
        logger.debug("Calculated posterior returns from %d view(s)", len(views))
        return posterior_returns

    def calculate_posterior_covariance(
        self,
        prior_cov: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        cov = _regularize_covariance(prior_cov)
        p_matrix = np.asarray(P, dtype=float)
        q_vector = np.asarray(Q, dtype=float).reshape(-1)
        if p_matrix.ndim != 2:
            raise ValueError("P must be a two-dimensional matrix")
        if p_matrix.shape[0] != q_vector.shape[0]:
            raise ValueError("P and Q dimensions must align")
        if p_matrix.shape[1] != cov.shape[0]:
            raise ValueError("P column count must match covariance dimensions")

        tau_cov = float(tau) * cov
        if p_matrix.shape[0] == 0:
            posterior_cov = cov + tau_cov
        else:
            omega = self._latest_omega
            if omega is None or omega.shape[0] != p_matrix.shape[0]:
                projected = p_matrix @ tau_cov @ p_matrix.T
                omega = np.diag(np.maximum(np.diag(projected), _EPS))
            posterior_uncertainty = np.linalg.inv(
                np.linalg.inv(tau_cov) + p_matrix.T @ np.linalg.inv(omega) @ p_matrix
            )
            posterior_cov = cov + posterior_uncertainty

        self._posterior_covariance = _regularize_covariance(posterior_cov)
        logger.debug("Calculated posterior covariance with shape %s", posterior_cov.shape)
        return self._posterior_covariance

    def optimize_portfolio(
        self,
        prior_returns: np.ndarray,
        posterior_cov: np.ndarray,
        risk_aversion: float,
    ) -> Dict[str, float]:
        expected_returns = np.asarray(prior_returns, dtype=float).reshape(-1)
        cov = _regularize_covariance(posterior_cov)
        if expected_returns.shape[0] != cov.shape[0]:
            raise ValueError("return vector length must match covariance dimensions")

        symbols = self._resolve_symbols(cov.shape[0])
        raw_weights = np.linalg.solve(max(float(risk_aversion), _EPS) * cov, expected_returns)
        if not self.allow_short:
            raw_weights = np.maximum(raw_weights, 0.0)
        weights = self._apply_weight_constraints(raw_weights)
        allocation = {symbol: float(weights[idx]) for idx, symbol in enumerate(symbols)}
        self._latest_weights = allocation
        logger.info("Optimized Black-Litterman portfolio for %d symbols", len(symbols))
        return allocation

    def blend_with_equilibrium(self, blend_weight: float) -> np.ndarray:
        if self._equilibrium_returns is None:
            raise ValueError("Equilibrium returns have not been calculated yet")
        if self._posterior_returns is None:
            raise ValueError("Posterior returns have not been calculated yet")

        weight = float(np.clip(blend_weight, 0.0, 1.0))
        blended = weight * self._equilibrium_returns + (1.0 - weight) * self._posterior_returns
        self._posterior_returns = blended
        logger.debug("Blended posterior and equilibrium returns with weight %.3f", weight)
        return blended

    def generate_views_from_ml_signals(
        self,
        ml_signals: Mapping[str, Any] | Iterable[Mapping[str, Any] | Any],
    ) -> List[MarketView]:
        """Create MarketView objects from common Argus ML signal shapes."""
        views: List[MarketView] = []

        if isinstance(ml_signals, Mapping) and not any(
            key in ml_signals for key in ("symbol", "expected_return", "predicted_return", "confidence")
        ):
            iterable: Iterable[Any] = [
                {"symbol": symbol, **(payload if isinstance(payload, Mapping) else {"signal": payload})}
                for symbol, payload in ml_signals.items()
            ]
        elif isinstance(ml_signals, Mapping):
            iterable = [ml_signals]
        else:
            iterable = ml_signals

        for item in iterable:
            payload = item if isinstance(item, Mapping) else self._object_to_mapping(item)
            symbol = str(payload.get("symbol", "") or "").strip()
            if not symbol:
                continue

            expected_return = payload.get("expected_return")
            if expected_return is None:
                expected_return = payload.get("predicted_return")
            if expected_return is None:
                expected_return = payload.get("return_forecast")
            if expected_return is None:
                expected_return = payload.get("alpha")
            if expected_return is None:
                signal_value = float(payload.get("signal", payload.get("combined_value", 0.0)) or 0.0)
                expected_return = signal_value * 0.05

            confidence = float(payload.get("confidence", 0.5) or 0.5)
            view_type = str(payload.get("view_type", "absolute") or "absolute")

            try:
                views.append(
                    MarketView(
                        symbol=symbol,
                        expected_return=float(expected_return),
                        confidence=confidence,
                        view_type=view_type,
                    )
                )
            except ValueError:
                logger.debug("Skipping invalid ML view payload for symbol '%s'", symbol)

        logger.debug("Generated %d market view(s) from ML signals", len(views))
        return views

    def optimize_from_market_data(
        self,
        market_data: Mapping[str, Sequence[float]] | np.ndarray,
        views: Optional[List[MarketView]] = None,
        ml_signals: Optional[Mapping[str, Any] | Iterable[Mapping[str, Any] | Any]] = None,
        blend_weight: Optional[float] = None,
    ) -> Dict[str, float]:
        covariance = self.estimate_covariance_from_market_data(market_data)
        market_weights_source: Mapping[str, float] | Sequence[float]
        if self.config.market_cap_weights:
            market_weights_source = self.config.market_cap_weights
        else:
            market_weights_source = np.full(covariance.shape[0], 1.0 / covariance.shape[0], dtype=float)

        equilibrium = self.calculate_equilibrium_returns(covariance, market_weights_source)
        all_views = list(views or [])
        if ml_signals is not None:
            all_views.extend(self.generate_views_from_ml_signals(ml_signals))

        posterior_returns = self.calculate_posterior_returns(equilibrium, all_views, self.config.tau)
        if blend_weight is not None and all_views:
            posterior_returns = self.blend_with_equilibrium(blend_weight)

        posterior_covariance = self.calculate_posterior_covariance(
            covariance,
            self._latest_p if self._latest_p is not None else np.zeros((0, covariance.shape[0]), dtype=float),
            self._latest_q if self._latest_q is not None else np.zeros(0, dtype=float),
            self.config.tau,
        )
        return self.optimize_portfolio(posterior_returns, posterior_covariance, self.config.risk_aversion)

    def integrate_with_existing_optimizer(
        self,
        existing_optimizer: Any,
        weights: Optional[Mapping[str, float]] = None,
        total_capital: Optional[float] = None,
    ) -> Any:
        """Bridge BL weights into existing Argus optimizer/allocation interfaces."""
        resolved_weights = dict(weights or self.latest_weights)
        if not resolved_weights:
            raise ValueError("No weights available to integrate with existing optimizer")

        if total_capital is not None and hasattr(existing_optimizer, "to_capital_allocation"):
            return existing_optimizer.to_capital_allocation(float(total_capital), resolved_weights)
        if hasattr(existing_optimizer, "apply_target_weights"):
            return existing_optimizer.apply_target_weights(resolved_weights)
        if hasattr(existing_optimizer, "set_target_weights"):
            return existing_optimizer.set_target_weights(resolved_weights)
        return resolved_weights

    def _market_data_to_matrix(self, market_data: Mapping[str, Sequence[float]]) -> np.ndarray:
        symbols = self._resolve_symbols_from_market_data(market_data)
        series = [np.asarray(market_data[symbol], dtype=float).reshape(-1) for symbol in symbols]
        min_length = min(arr.shape[0] for arr in series)
        if min_length <= 0:
            raise ValueError("market_data must contain at least one observation per symbol")

        matrix = np.column_stack([arr[-min_length:] for arr in series])
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            finite = np.isfinite(values)
            if finite.all():
                continue
            fill_value = float(np.mean(values[finite])) if finite.any() else 0.0
            values[~finite] = fill_value
            matrix[:, column] = values
        return matrix

    def _build_view_matrices(
        self,
        views: List[MarketView],
        covariance: np.ndarray,
        tau: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        symbols = self._resolve_symbols(covariance.shape[0])
        index = {symbol: idx for idx, symbol in enumerate(symbols)}
        p_rows: List[np.ndarray] = []
        q_values: List[float] = []
        omega_values: List[float] = []
        tau_cov = tau * covariance

        for view in views:
            row = np.zeros(len(symbols), dtype=float)
            if view.view_type == "absolute":
                if view.symbol not in index:
                    logger.debug("Skipping view for unknown symbol '%s'", view.symbol)
                    continue
                row[index[view.symbol]] = 1.0
            else:
                long_symbol, short_symbol = self._parse_relative_symbol(view.symbol)
                if long_symbol not in index or short_symbol not in index:
                    logger.debug("Skipping relative view with unknown symbols '%s'", view.symbol)
                    continue
                row[index[long_symbol]] = 1.0
                row[index[short_symbol]] = -1.0

            projected_variance = float(row @ tau_cov @ row.T)
            confidence = float(np.clip(view.confidence, 1e-6, 1.0))
            omega = max(projected_variance * ((1.0 - confidence) / confidence), _EPS)

            p_rows.append(row)
            q_values.append(float(view.expected_return))
            omega_values.append(omega)

        if not p_rows:
            return (
                np.zeros((0, covariance.shape[0]), dtype=float),
                np.zeros(0, dtype=float),
                np.zeros((0, 0), dtype=float),
            )

        return (
            np.vstack(p_rows),
            np.asarray(q_values, dtype=float),
            np.diag(np.asarray(omega_values, dtype=float)),
        )

    def _weights_to_vector(self, market_weights: Mapping[str, float] | Sequence[float]) -> np.ndarray:
        if isinstance(market_weights, Mapping):
            symbols = self._resolve_symbols_from_market_data(market_weights)
            weights = np.array([float(market_weights.get(symbol, 0.0)) for symbol in symbols], dtype=float)
        else:
            weights = np.asarray(market_weights, dtype=float).reshape(-1)
            symbols = self._resolve_symbols(weights.shape[0])

        if len(symbols) != weights.shape[0]:
            raise ValueError("market weights length does not match symbol universe")
        self.symbols = symbols
        return _normalise_weights(np.maximum(weights, 0.0))

    def _apply_weight_constraints(self, raw_weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(raw_weights, dtype=float).reshape(-1)
        n_assets = weights.shape[0]
        if n_assets == 0:
            return weights
        if not self.allow_short:
            weights = np.maximum(weights, 0.0)

        weights = _normalise_weights(weights)
        for _ in range(100):
            updated = weights.copy()
            if not self.allow_short:
                updated = np.clip(updated, self.min_weight, self.max_weight)
            else:
                updated = np.minimum(updated, self.max_weight)
            updated = _normalise_weights(updated)
            if np.max(np.abs(updated - weights)) < 1e-8:
                weights = updated
                break
            weights = updated
        return weights

    def _resolve_symbols(self, dimension: int) -> List[str]:
        if self.symbols:
            if len(self.symbols) != dimension:
                raise ValueError("symbol universe length does not match matrix dimensions")
            return list(self.symbols)
        self.symbols = [f"asset_{idx}" for idx in range(dimension)]
        return list(self.symbols)

    def _resolve_symbols_from_market_data(self, data: Mapping[str, Any]) -> List[str]:
        if self.symbols:
            return list(self.symbols)
        self.symbols = [str(symbol) for symbol in data.keys()]
        return list(self.symbols)

    @staticmethod
    def _parse_relative_symbol(symbol: str) -> Tuple[str, str]:
        for delimiter in ("|", ">", ":", ","):
            if delimiter in symbol:
                long_symbol, short_symbol = [part.strip() for part in symbol.split(delimiter, 1)]
                if long_symbol and short_symbol:
                    return long_symbol, short_symbol
        raise ValueError(
            "Relative views must encode both symbols, e.g. 'BTC/USD|ETH/USD'"
        )

    @staticmethod
    def _object_to_mapping(obj: Any) -> Dict[str, Any]:
        keys = (
            "symbol",
            "expected_return",
            "predicted_return",
            "return_forecast",
            "alpha",
            "signal",
            "combined_value",
            "confidence",
            "view_type",
        )
        return {key: getattr(obj, key) for key in keys if hasattr(obj, key)}
