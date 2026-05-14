"""
Hyperparameter Optimizer — Bayesian optimization of strategy parameters.

Uses Optuna (if available) or random search to optimize strategy parameters
against a performance objective (Sharpe ratio, Sortino, profit factor).

Supported parameters:
  - signal_confidence threshold (0.3 - 0.8)
  - position_size_pct (0.05 - 0.25)
  - stop_loss_pct (0.005 - 0.03)
  - take_profit_pct (0.01 - 0.06)
  - regime_filter_strength (0.0 - 1.0)

Usage:
    optimizer = HyperOptimizer(n_trials=50, objective="sharpe")
    best_params = optimizer.optimize(backtest_fn)
    # backtest_fn(params) → float (objective value)
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


@dataclass
class TrialResult:
    trial_id: int
    params: Dict
    objective_value: float
    duration_seconds: float
    status: str  # "ok" | "failed" | "pruned"


class HyperOptimizer:
    """Bayesian hyperparameter optimization for strategy parameters."""

    PARAM_SPACE: Dict[str, Tuple] = {
        "signal_confidence":     (0.30, 0.80),
        "position_size_pct":     (0.05, 0.25),
        "stop_loss_pct":         (0.005, 0.030),
        "take_profit_pct":       (0.010, 0.060),
        "regime_filter_strength": (0.0, 1.0),
    }

    def __init__(
        self,
        n_trials: int = 50,
        objective: str = "sharpe",
        param_space: Optional[Dict[str, Tuple]] = None,
        random_seed: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.objective = objective
        self.param_space = param_space if param_space is not None else self.PARAM_SPACE.copy()
        self.random_seed = random_seed
        self._results: List[TrialResult] = []
        self._best_params: Optional[Dict] = None
        self._rng = random.Random(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, backtest_fn: Callable[[Dict], float]) -> Dict:
        """
        Run optimization and return best parameters.

        Parameters
        ----------
        backtest_fn:
            Function that takes a params dict and returns a float score
            (higher = better). Exceptions are caught and assigned -999.0.
        """
        if _OPTUNA_AVAILABLE:
            logger.info("HyperOptimizer: using Optuna (Bayesian search), n_trials=%d", self.n_trials)
            return self._run_optuna(backtest_fn)
        logger.info("HyperOptimizer: using random search (optuna not available), n_trials=%d", self.n_trials)
        return self._run_random_search(backtest_fn)

    @property
    def best_params(self) -> Optional[Dict]:
        return self._best_params

    def get_results(self) -> List[TrialResult]:
        return list(self._results)

    def plot_importance(self) -> None:
        """Log parameter importance (requires Optuna with completed study)."""
        if not _OPTUNA_AVAILABLE or not hasattr(self, "_study"):
            logger.info("HyperOptimizer: parameter importance requires optuna + completed study")
            return
        try:
            importance = optuna.importance.get_param_importances(self._study)
            logger.info("Parameter importance: %s", importance)
        except Exception:
            logger.debug("plot_importance failed", exc_info=True)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_optuna(self, backtest_fn: Callable[[Dict], float]) -> Dict:
        """Bayesian search via Optuna."""
        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        self._study = study

        def objective_fn(trial: optuna.Trial) -> float:
            params = self._sample_params(trial)
            t0 = time.perf_counter()
            try:
                score = float(backtest_fn(params))
                if score != score:  # NaN check
                    score = -999.0
            except Exception:
                logger.debug("Trial failed", exc_info=True)
                score = -999.0
            dur = time.perf_counter() - t0
            status = "ok" if score > -999.0 else "failed"
            self._results.append(TrialResult(
                trial_id=trial.number,
                params=params,
                objective_value=score,
                duration_seconds=dur,
                status=status,
            ))
            return score

        study.optimize(objective_fn, n_trials=self.n_trials, show_progress_bar=False)
        self._best_params = study.best_params
        logger.info(
            "HyperOptimizer: best score=%.4f params=%s",
            study.best_value, self._best_params,
        )
        self.plot_importance()
        return self._best_params

    def _run_random_search(self, backtest_fn: Callable[[Dict], float]) -> Dict:
        """Uniform random search fallback."""
        best_score = float("-inf")
        best_params: Dict = {}

        for trial_id in range(self.n_trials):
            params = self._sample_params()
            t0 = time.perf_counter()
            try:
                score = float(backtest_fn(params))
                if score != score:
                    score = -999.0
                status = "ok"
            except Exception:
                logger.debug("Random trial %d failed", trial_id, exc_info=True)
                score = -999.0
                status = "failed"
            dur = time.perf_counter() - t0

            self._results.append(TrialResult(
                trial_id=trial_id,
                params=params,
                objective_value=score,
                duration_seconds=dur,
                status=status,
            ))

            if score > best_score:
                best_score = score
                best_params = params.copy()

        self._best_params = best_params
        logger.info(
            "HyperOptimizer: random search best score=%.4f params=%s",
            best_score, best_params,
        )
        return best_params

    def _sample_params(self, trial=None) -> Dict:
        """Sample a parameter set from the defined space."""
        params = {}
        for name, (lo, hi) in self.param_space.items():
            if trial is not None and _OPTUNA_AVAILABLE:
                # Optuna trial
                params[name] = trial.suggest_float(name, lo, hi)
            else:
                params[name] = self._rng.uniform(lo, hi)
        return params
