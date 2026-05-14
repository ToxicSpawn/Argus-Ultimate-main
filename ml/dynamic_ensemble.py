"""
Dynamic Ensemble Learning with Adaptive Weighting.

Combines multiple model predictions using adaptive weighting strategies
that respond to market conditions, model performance, and regime shifts.

Weighting methods:
- Equal: uniform weights across all models
- Inverse Volatility: lower-volatility models get higher weights
- Momentum: models with recent strong performance get higher weights
- Sharpe-based: risk-adjusted performance determines weights
- Exponential Decay: recent performance weighted more heavily

Regime awareness:
- Detects bull/bear/sideways regimes from market data
- Maintains per-regime weight profiles for each model
- Switches weighting strategy based on current regime

Example::

    ensemble = DynamicEnsemble()
    ensemble.register_model("xgboost", xgb_model, predict_fn=xgb_predict)
    ensemble.register_model("lstm", lstm_model, predict_fn=lstm_predict)
    ensemble.update_weights(returns_history, method="exponential_decay")
    prediction = ensemble.predict(current_features)

    regime_ensemble = RegimeAwareEnsemble(ensemble)
    regime = regime_ensemble.detect_regime(market_data)
    prediction = regime_ensemble.get_regime_specific_prediction(regime, features)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelPerformance:
    """Performance record for a single model over a time window."""

    model_name: str
    returns: np.ndarray
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.returns is not None and len(self.returns) > 0:
            self._compute_metrics()

    def _compute_metrics(self) -> None:
        returns = np.asarray(self.returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret > 1e-12:
            self.sharpe = float(mean_ret / std_ret * np.sqrt(252))
        else:
            self.sharpe = 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        self.max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        self.win_rate = float(np.mean(returns > 0))


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    calmar_ratio: float
    predictions: np.ndarray
    targets: np.ndarray
    equity_curve: np.ndarray


@dataclass
class ComparisonResult:
    """Comparison of multiple models."""

    model_names: List[str]
    sharpe_ratios: List[float]
    max_drawdowns: List[float]
    win_rates: List[float]
    total_returns: List[float]
    best_model: str
    rankings: List[Tuple[str, float]]


# ---------------------------------------------------------------------------
# EnsembleWeighting
# ---------------------------------------------------------------------------


class EnsembleWeighting:
    """
    Collection of weighting strategies for ensemble models.

    Each method takes a list of model names and their returns history,
    and returns a numpy array of weights that sum to 1.0.
    """

    @staticmethod
    def equal_weight(models: List[str]) -> np.ndarray:
        """Assign equal weights to all models."""
        n = len(models)
        if n == 0:
            return np.array([])
        weights = np.ones(n) / n
        logger.debug("Equal weights for %d models: %s", n, weights)
        return weights

    @staticmethod
    def inverse_volatility_weight(
        models: List[str],
        returns: np.ndarray,
    ) -> np.ndarray:
        """
        Weight models inversely proportional to their return volatility.

        Models with more stable (lower variance) predictions get higher weights.

        Parameters
        ----------
        models : list[str]
            Model names.
        returns : np.ndarray
            2D array of shape (n_models, n_periods) with per-model returns.
        """
        n = len(models)
        if n == 0:
            return np.array([])

        if returns.ndim == 1:
            returns = returns.reshape(1, -1)

        volatilities = np.std(returns, axis=1)
        inv_vol = 1.0 / np.maximum(volatilities, 1e-10)
        weights = inv_vol / np.sum(inv_vol)

        logger.debug(
            "Inverse volatility weights: %s",
            {m: round(w, 4) for m, w in zip(models, weights)},
        )
        return weights

    @staticmethod
    def momentum_weight(
        models: List[str],
        returns: np.ndarray,
        lookback: int = 20,
    ) -> np.ndarray:
        """
        Weight models based on recent momentum (lookback-period returns).

        Models with stronger recent performance get exponentially higher weights.

        Parameters
        ----------
        models : list[str]
            Model names.
        returns : np.ndarray
            2D array of shape (n_models, n_periods).
        lookback : int
            Number of recent periods to consider for momentum.
        """
        n = len(models)
        if n == 0:
            return np.array([])

        if returns.ndim == 1:
            returns = returns.reshape(1, -1)

        recent = returns[:, -lookback:] if returns.shape[1] >= lookback else returns
        momentum = np.sum(recent, axis=1)

        exp_momentum = np.exp(momentum)
        weights = exp_momentum / np.sum(exp_momentum)

        logger.debug(
            "Momentum weights (lookback=%d): %s",
            lookback,
            {m: round(w, 4) for m, w in zip(models, weights)},
        )
        return weights

    @staticmethod
    def adaptive_weight(
        models: List[str],
        returns: np.ndarray,
        method: str = "sharpe",
    ) -> np.ndarray:
        """
        Compute adaptive weights based on a performance metric.

        Parameters
        ----------
        models : list[str]
            Model names.
        returns : np.ndarray
            2D array of shape (n_models, n_periods).
        method : str
            Weighting method: "sharpe", "sortino", "calmar", "omega".
        """
        n = len(models)
        if n == 0:
            return np.array([])

        if returns.ndim == 1:
            returns = returns.reshape(1, -1)

        scores = np.zeros(n)
        for i in range(n):
            model_returns = returns[i]
            model_returns = model_returns[~np.isnan(model_returns)]
            if len(model_returns) == 0:
                scores[i] = 0.0
                continue

            mean_ret = np.mean(model_returns)
            std_ret = np.std(model_returns)

            if method == "sharpe":
                scores[i] = mean_ret / max(std_ret, 1e-10) * np.sqrt(252)
            elif method == "sortino":
                downside = model_returns[model_returns < 0]
                downside_std = np.std(downside) if len(downside) > 0 else 1e-10
                scores[i] = mean_ret / max(downside_std, 1e-10) * np.sqrt(252)
            elif method == "calmar":
                cumulative = np.cumsum(model_returns)
                running_max = np.maximum.accumulate(cumulative)
                max_dd = np.min(cumulative - running_max)
                scores[i] = np.sum(model_returns) / max(abs(max_dd), 1e-10)
            elif method == "omega":
                threshold = 0.0
                gains = model_returns[model_returns > threshold]
                losses = model_returns[model_returns <= threshold]
                total_gain = np.sum(gains) if len(gains) > 0 else 0.0
                total_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-10
                scores[i] = total_gain / total_loss
            else:
                scores[i] = mean_ret / max(std_ret, 1e-10) * np.sqrt(252)

        softmax_scores = np.exp(np.clip(scores, -10, 10))
        weights = softmax_scores / np.sum(softmax_scores)

        logger.debug(
            "Adaptive weights (method=%s): %s",
            method,
            {m: round(w, 4) for m, w in zip(models, weights)},
        )
        return weights


# ---------------------------------------------------------------------------
# DynamicEnsemble
# ---------------------------------------------------------------------------


class DynamicEnsemble:
    """
    Dynamic ensemble that combines multiple models with adaptive weighting.

    Models are registered with a name, the model object, and a predict
    function. Weights are updated periodically based on performance history.

    Parameters
    ----------
    weighting_method : str
        Default weighting method for update_weights().
    min_weight : float
        Minimum weight any model can have (prevents zeroing out).
    """

    def __init__(
        self,
        weighting_method: str = "exponential_decay",
        min_weight: float = 0.01,
    ) -> None:
        self.models: Dict[str, Any] = {}
        self.predict_fns: Dict[str, Callable] = {}
        self.weights: Dict[str, float] = {}
        self.weighting_method = weighting_method
        self.min_weight = min_weight
        self.performance_history: Dict[str, List[float]] = {}
        self._weighter = EnsembleWeighting()

        logger.info(
            "DynamicEnsemble initialised (method=%s, min_weight=%.4f)",
            weighting_method,
            min_weight,
        )

    def register_model(
        self,
        name: str,
        model: Any,
        predict_fn: Callable,
    ) -> None:
        """
        Register a model in the ensemble.

        Parameters
        ----------
        name : str
            Unique identifier for the model.
        model : Any
            The model object (e.g. trained sklearn/xgboost model).
        predict_fn : callable
            Function that takes features and returns predictions.
        """
        self.models[name] = model
        self.predict_fns[name] = predict_fn
        self.performance_history[name] = []

        if len(self.weights) == 0:
            self.weights[name] = 1.0
        else:
            n = len(self.weights) + 1
            for k in self.weights:
                self.weights[k] *= (n - 1) / n
            self.weights[name] = 1.0 / n

        logger.info("Registered model '%s' in ensemble", name)

    def unregister_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        self.models.pop(name, None)
        self.predict_fns.pop(name, None)
        self.weights.pop(name, None)
        self.performance_history.pop(name, None)
        self._normalise_weights()
        logger.info("Unregistered model '%s' from ensemble", name)

    def update_weights(
        self,
        returns_history: Dict[str, np.ndarray],
        method: Optional[str] = None,
    ) -> None:
        """
        Update model weights based on recent returns history.

        Parameters
        ----------
        returns_history : dict[str, np.ndarray]
            Mapping of model name to array of recent returns.
        method : str, optional
            Weighting method override. Defaults to instance weighting_method.
        """
        method = method or self.weighting_method
        model_names = [n for n in self.models if n in returns_history]

        if not model_names:
            logger.warning("No models with returns history for weight update")
            return

        returns_array = np.array([returns_history[n] for n in model_names])

        if method == "equal":
            new_weights = self._weighter.equal_weight(model_names)
        elif method == "inverse_volatility":
            new_weights = self._weighter.inverse_volatility_weight(
                model_names, returns_array
            )
        elif method == "momentum":
            new_weights = self._weighter.momentum_weight(
                model_names, returns_array
            )
        elif method == "adaptive":
            new_weights = self._weighter.adaptive_weight(
                model_names, returns_array
            )
        elif method == "exponential_decay":
            new_weights = self._exponential_decay_weights(
                model_names, returns_array
            )
        elif method == "sharpe":
            new_weights = self._weighter.adaptive_weight(
                model_names, returns_array, method="sharpe"
            )
        else:
            new_weights = self._exponential_decay_weights(
                model_names, returns_array
            )

        for i, name in enumerate(model_names):
            self.weights[name] = max(new_weights[i], self.min_weight)

        self._normalise_weights()
        logger.info(
            "Updated ensemble weights (method=%s): %s",
            method,
            {k: round(v, 4) for k, v in self.weights.items()},
        )

    def predict(self, features: Any) -> Dict[str, Any]:
        """
        Generate weighted ensemble prediction.

        Parameters
        ----------
        features : Any
            Input features for prediction (passed to each model's predict_fn).

        Returns
        -------
        dict
            Contains 'prediction' (weighted average), 'individual' (per-model),
            and 'weights' (current weights used).
        """
        if not self.models:
            logger.warning("No models registered for prediction")
            return {"prediction": 0.0, "individual": {}, "weights": {}}

        individual_predictions: Dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for name, predict_fn in self.predict_fns.items():
            try:
                pred = predict_fn(features)
                pred_value = float(pred) if not hasattr(pred, "__len__") else float(np.mean(pred))
                individual_predictions[name] = pred_value
                w = self.weights.get(name, 0.0)
                weighted_sum += pred_value * w
                total_weight += w
            except Exception:
                logger.exception("Prediction failed for model '%s'", name)

        if total_weight > 0:
            ensemble_prediction = weighted_sum / total_weight
        else:
            ensemble_prediction = 0.0

        return {
            "prediction": ensemble_prediction,
            "individual": individual_predictions,
            "weights": dict(self.weights),
        }

    def get_model_contributions(self) -> Dict[str, float]:
        """
        Calculate each model's contribution to the ensemble.

        Contribution = weight * average_absolute_prediction.
        Normalised to sum to 1.0.

        Returns
        -------
        dict[str, float]
            Model contribution percentages.
        """
        contributions: Dict[str, float] = {}

        for name in self.models:
            history = self.performance_history.get(name, [])
            if history:
                avg_abs = np.mean(np.abs(history))
            else:
                avg_abs = 0.0
            contributions[name] = self.weights.get(name, 0.0) * max(avg_abs, 1e-10)

        total = sum(contributions.values())
        if total > 0:
            for name in contributions:
                contributions[name] /= total
        else:
            n = len(contributions)
            if n > 0:
                for name in contributions:
                    contributions[name] = 1.0 / n

        return contributions

    def get_model_performances(
        self,
        lookback: Optional[int] = None,
    ) -> Dict[str, ModelPerformance]:
        """
        Get performance records for all models.

        Parameters
        ----------
        lookback : int, optional
            Number of most recent returns to include.

        Returns
        -------
        dict[str, ModelPerformance]
        """
        performances: Dict[str, ModelPerformance] = {}

        for name, history in self.performance_history.items():
            if not history:
                continue

            returns = np.array(history)
            if lookback is not None:
                returns = returns[-lookback:]

            performances[name] = ModelPerformance(
                model_name=name,
                returns=returns,
            )

        return performances

    def record_prediction_outcome(
        self,
        model_name: str,
        prediction: float,
        actual: float,
    ) -> None:
        """
        Record the outcome of a prediction for weight updates.

        Parameters
        ----------
        model_name : str
            Which model made the prediction.
        prediction : float
            The predicted value.
        actual : float
            The actual observed value.
        """
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        return_value = actual - prediction
        self.performance_history[model_name].append(return_value)

        logger.debug(
            "Recorded outcome for '%s': pred=%.4f actual=%.4f return=%.4f",
            model_name, prediction, actual, return_value,
        )

    def _normalise_weights(self) -> None:
        """Normalise weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    def _exponential_decay_weights(
        self,
        model_names: List[str],
        returns_array: np.ndarray,
        decay_factor: float = 0.95,
    ) -> np.ndarray:
        """
        Compute weights using exponentially decayed cumulative returns.

        More recent returns are weighted more heavily.
        """
        n_models = len(model_names)
        scores = np.zeros(n_models)

        for i in range(n_models):
            model_returns = returns_array[i]
            model_returns = model_returns[~np.isnan(model_returns)]
            if len(model_returns) == 0:
                scores[i] = 0.0
                continue

            n = len(model_returns)
            time_weights = np.array([decay_factor ** (n - 1 - j) for j in range(n)])
            scores[i] = np.sum(model_returns * time_weights)

        softmax_scores = np.exp(np.clip(scores, -10, 10))
        weights = softmax_scores / np.sum(softmax_scores)
        return weights

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the ensemble state."""
        performances = self.get_model_performances()
        contributions = self.get_model_contributions()

        return {
            "n_models": len(self.models),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "contributions": {k: round(v, 4) for k, v in contributions.items()},
            "performances": {
                k: {
                    "sharpe": round(p.sharpe, 4),
                    "max_drawdown": round(p.max_drawdown, 4),
                    "win_rate": round(p.win_rate, 4),
                }
                for k, p in performances.items()
            },
        }


# ---------------------------------------------------------------------------
# RegimeAwareEnsemble
# ---------------------------------------------------------------------------


class RegimeAwareEnsemble:
    """
    Ensemble that adapts to market regimes.

    Detects bull/bear/sideways regimes and maintains separate weight
    profiles for each regime. Models that perform well in a specific
    regime get higher weights when that regime is active.

    Parameters
    ----------
    base_ensemble : DynamicEnsemble
        The underlying ensemble to wrap.
    lookback : int
        Number of periods for regime detection.
    """

    def __init__(
        self,
        base_ensemble: DynamicEnsemble,
        lookback: int = 60,
    ) -> None:
        self.base_ensemble = base_ensemble
        self.lookback = lookback
        self.regime_weights: Dict[str, Dict[str, float]] = {
            "bull": {},
            "bear": {},
            "sideways": {},
        }
        self.regime_performance: Dict[str, Dict[str, List[float]]] = {
            "bull": {},
            "bear": {},
            "sideways": {},
        }
        self._current_regime: str = "sideways"
        self._regime_history: List[str] = []

        logger.info(
            "RegimeAwareEnsemble initialised (lookback=%d)",
            lookback,
        )

    def detect_regime(self, market_data: np.ndarray) -> str:
        """
        Detect current market regime from price data.

        Uses a combination of trend, volatility, and momentum signals.

        Parameters
        ----------
        market_data : np.ndarray
            Array of recent prices (newest last).

        Returns
        -------
        str
            "bull", "bear", or "sideways".
        """
        prices = np.asarray(market_data, dtype=np.float64)
        if len(prices) < 2:
            return "sideways"

        data = prices[-self.lookback:] if len(prices) >= self.lookback else prices

        trend = self._compute_trend(data)
        volatility = self._compute_volatility(data)
        momentum = self._compute_momentum(data)

        if trend > 0.02 and momentum > 0:
            regime = "bull"
        elif trend < -0.02 and momentum < 0:
            regime = "bear"
        else:
            regime = "sideways"

        if volatility > 0.05:
            if regime == "bull":
                regime = "bull"
            elif regime == "bear":
                regime = "bear"
            else:
                regime = "sideways"

        self._current_regime = regime
        self._regime_history.append(regime)

        logger.debug(
            "Regime detected: %s (trend=%.4f, vol=%.4f, momentum=%.4f)",
            regime, trend, volatility, momentum,
        )
        return regime

    def update_regime_weights(
        self,
        regime: str,
        returns_history: Dict[str, np.ndarray],
    ) -> None:
        """
        Update weights for a specific regime.

        Parameters
        ----------
        regime : str
            Regime label ("bull", "bear", "sideways").
        returns_history : dict[str, np.ndarray]
            Model returns for this regime.
        """
        if regime not in self.regime_weights:
            self.regime_weights[regime] = {}
            self.regime_performance[regime] = {}

        model_names = list(returns_history.keys())
        if not model_names:
            return

        returns_array = np.array([returns_history[n] for n in model_names])

        weights = EnsembleWeighting().adaptive_weight(
            model_names, returns_array, method="sharpe"
        )

        for i, name in enumerate(model_names):
            self.regime_weights[regime][name] = weights[i]
            if name not in self.regime_performance[regime]:
                self.regime_performance[regime][name] = []
            self.regime_performance[regime][name].extend(
                returns_array[i].tolist()
            )

        logger.info(
            "Updated regime weights for '%s': %s",
            regime,
            {k: round(v, 4) for k, v in self.regime_weights[regime].items()},
        )

    def get_regime_specific_prediction(
        self,
        regime: str,
        features: Any,
    ) -> Dict[str, Any]:
        """
        Get prediction using regime-specific weights.

        Parameters
        ----------
        regime : str
            Current regime ("bull", "bear", "sideways").
        features : Any
            Input features for prediction.

        Returns
        -------
        dict
            Prediction with regime-specific weighting.
        """
        regime_w = self.regime_weights.get(regime, {})
        if not regime_w:
            logger.warning(
                "No regime-specific weights for '%s', using base ensemble",
                regime,
            )
            return self.base_ensemble.predict(features)

        original_weights = dict(self.base_ensemble.weights)

        for name in self.base_ensemble.models:
            if name in regime_w:
                self.base_ensemble.weights[name] = regime_w[name]
            else:
                self.base_ensemble.weights[name] = self.base_ensemble.min_weight

        self.base_ensemble._normalise_weights()

        try:
            result = self.base_ensemble.predict(features)
            result["regime"] = regime
            result["regime_weights"] = dict(regime_w)
            return result
        finally:
            self.base_ensemble.weights = original_weights

    def predict(
        self,
        market_data: np.ndarray,
        features: Any,
    ) -> Dict[str, Any]:
        """
        Full prediction pipeline: detect regime then predict.

        Parameters
        ----------
        market_data : np.ndarray
            Recent price data for regime detection.
        features : Any
            Input features for model prediction.

        Returns
        -------
        dict
            Prediction with regime info.
        """
        regime = self.detect_regime(market_data)
        return self.get_regime_specific_prediction(regime, features)

    def get_regime_summary(self) -> Dict[str, Any]:
        """Return summary of regime detection and weights."""
        return {
            "current_regime": self._current_regime,
            "regime_history": self._regime_history[-20:],
            "regime_weights": {
                r: {k: round(v, 4) for k, v in w.items()}
                for r, w in self.regime_weights.items()
                if w
            },
            "regime_model_count": {
                r: len(w) for r, w in self.regime_weights.items()
            },
        }

    @staticmethod
    def _compute_trend(prices: np.ndarray) -> float:
        """Compute linear trend slope normalised by price level."""
        if len(prices) < 2:
            return 0.0
        n = len(prices)
        x = np.arange(n, dtype=np.float64)
        x_mean = np.mean(x)
        y_mean = np.mean(prices)
        numerator = np.sum((x - x_mean) * (prices - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator < 1e-12:
            return 0.0
        slope = numerator / denominator
        return slope / max(y_mean, 1e-10)

    @staticmethod
    def _compute_volatility(prices: np.ndarray) -> float:
        """Compute annualised volatility of returns."""
        if len(prices) < 2:
            return 0.0
        returns = np.diff(np.log(prices))
        return float(np.std(returns) * np.sqrt(252))

    @staticmethod
    def _compute_momentum(prices: np.ndarray) -> float:
        """Compute momentum as recent return."""
        if len(prices) < 2:
            return 0.0
        return float((prices[-1] - prices[0]) / max(prices[0], 1e-10))


# ---------------------------------------------------------------------------
# EnsembleBacktester
# ---------------------------------------------------------------------------


class EnsembleBacktester:
    """
    Backtesting framework for dynamic ensembles.

    Evaluates ensemble performance against individual models and
    alternative weighting strategies.

    Parameters
    ----------
    initial_capital : float
        Starting capital for backtest simulation.
    """

    def __init__(self, initial_capital: float = 100000.0) -> None:
        self.initial_capital = initial_capital
        logger.info(
            "EnsembleBacktester initialised (initial_capital=%.2f)",
            initial_capital,
        )

    def backtest(
        self,
        ensemble: DynamicEnsemble,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> BacktestResult:
        """
        Run backtest on an ensemble.

        Parameters
        ----------
        ensemble : DynamicEnsemble
            The ensemble to backtest.
        features : np.ndarray
            2D array of shape (n_samples, n_features).
        targets : np.ndarray
            1D array of actual values (n_samples,).

        Returns
        -------
        BacktestResult
        """
        n_samples = len(targets)
        predictions = np.zeros(n_samples)
        equity_curve = np.zeros(n_samples)
        capital = self.initial_capital

        for i in range(n_samples):
            feat = features[i] if features.ndim > 1 else features
            result = ensemble.predict(feat)
            pred = result["prediction"]
            predictions[i] = pred

            return_value = targets[i] - pred
            capital *= (1 + return_value)
            equity_curve[i] = capital

            for name in ensemble.models:
                try:
                    individual_pred = ensemble.predict_fns[name](feat)
                    individual_val = (
                        float(individual_pred)
                        if not hasattr(individual_pred, "__len__")
                        else float(np.mean(individual_pred))
                    )
                    ensemble.record_prediction_outcome(
                        name, individual_val, targets[i]
                    )
                except Exception:
                    pass

        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = np.insert(returns, 0, 0.0)

        total_return = (equity_curve[-1] / equity_curve[0]) - 1 if equity_curve[0] > 0 else 0.0
        sharpe = self._compute_sharpe(returns)
        max_dd = self._compute_max_drawdown(equity_curve)
        win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.0
        calmar = total_return / max(abs(max_dd), 1e-10) if max_dd != 0 else 0.0

        logger.info(
            "Backtest complete: total_return=%.4f sharpe=%.4f max_dd=%.4f win_rate=%.4f",
            total_return, sharpe, max_dd, win_rate,
        )

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            calmar_ratio=calmar,
            predictions=predictions,
            targets=targets,
            equity_curve=equity_curve,
        )

    def compare_models(
        self,
        models: Dict[str, Tuple[Any, Callable]],
        features: np.ndarray,
        targets: np.ndarray,
    ) -> ComparisonResult:
        """
        Compare multiple models individually.

        Parameters
        ----------
        models : dict[str, tuple]
            Mapping of name to (model_object, predict_fn).
        features : np.ndarray
            2D array of shape (n_samples, n_features).
        targets : np.ndarray
            1D array of actual values.

        Returns
        -------
        ComparisonResult
        """
        results: Dict[str, Dict[str, float]] = {}

        for name, (model, predict_fn) in models.items():
            predictions = np.zeros(len(targets))
            for i in range(len(targets)):
                try:
                    feat = features[i] if features.ndim > 1 else features
                    pred = predict_fn(feat)
                    predictions[i] = (
                        float(pred)
                        if not hasattr(pred, "__len__")
                        else float(np.mean(pred))
                    )
                except Exception:
                    predictions[i] = 0.0

            returns = targets - predictions
            total_return = float(np.sum(returns))
            sharpe = self._compute_sharpe(returns)
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            max_dd = float(np.min(cumulative - running_max))
            win_rate = float(np.mean(returns > 0))

            results[name] = {
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
            }

        model_names = list(results.keys())
        sharpe_ratios = [results[n]["sharpe"] for n in model_names]
        max_drawdowns = [results[n]["max_drawdown"] for n in model_names]
        win_rates = [results[n]["win_rate"] for n in model_names]
        total_returns = [results[n]["total_return"] for n in model_names]

        rankings = sorted(
            [(n, results[n]["sharpe"]) for n in model_names],
            key=lambda x: x[1],
            reverse=True,
        )
        best_model = rankings[0][0] if rankings else ""

        logger.info(
            "Model comparison complete. Best: '%s' (sharpe=%.4f)",
            best_model,
            rankings[0][1] if rankings else 0.0,
        )

        return ComparisonResult(
            model_names=model_names,
            sharpe_ratios=sharpe_ratios,
            max_drawdowns=max_drawdowns,
            win_rates=win_rates,
            total_returns=total_returns,
            best_model=best_model,
            rankings=rankings,
        )

    def compare_weighting_methods(
        self,
        ensemble: DynamicEnsemble,
        features: np.ndarray,
        targets: np.ndarray,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, BacktestResult]:
        """
        Compare different weighting methods on the same data.

        Parameters
        ----------
        ensemble : DynamicEnsemble
            Base ensemble (weights will be reset per method).
        features : np.ndarray
            Feature matrix.
        targets : np.ndarray
            Target values.
        methods : list[str], optional
            Weighting methods to compare.

        Returns
        -------
        dict[str, BacktestResult]
        """
        if methods is None:
            methods = ["equal", "inverse_volatility", "momentum", "sharpe", "exponential_decay"]

        results: Dict[str, BacktestResult] = {}

        for method in methods:
            test_ensemble = DynamicEnsemble(weighting_method=method)
            for name, model in ensemble.models.items():
                test_ensemble.register_model(name, model, ensemble.predict_fns[name])

            returns_history: Dict[str, np.ndarray] = {}
            for name in ensemble.models:
                history = ensemble.performance_history.get(name, [])
                if history:
                    returns_history[name] = np.array(history)
                else:
                    returns_history[name] = np.array([0.0])

            test_ensemble.update_weights(returns_history, method=method)
            result = self.backtest(test_ensemble, features, targets)
            results[method] = result

        logger.info(
            "Weighting method comparison: %s",
            {m: round(r.sharpe_ratio, 4) for m, r in results.items()},
        )
        return results

    @staticmethod
    def _compute_sharpe(returns: np.ndarray) -> float:
        """Compute annualised Sharpe ratio."""
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret < 1e-12:
            return 0.0
        return float(mean_ret / std_ret * np.sqrt(252))

    @staticmethod
    def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
        """Compute maximum drawdown from equity curve."""
        if len(equity_curve) < 2:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / np.maximum(running_max, 1e-10)
        return float(np.min(drawdowns))
