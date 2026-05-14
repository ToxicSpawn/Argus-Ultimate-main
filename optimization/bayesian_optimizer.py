"""
Bayesian Hyperparameter Optimization for Argus-Ultimate v5.0.0 (HFT-Pinnacle)
Optuna-based tuning calibrated specifically for $1k capital accounts.
Supports multi-objective optimization (ROI + Sharpe + max drawdown).
"""

import logging
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("optuna is required: pip install optuna")

logger = logging.getLogger(__name__)


DEFAULT_PARAM_SPACE: Dict[str, Any] = {
    # Risk params
    "min_trade_usd": {"type": "float", "low": 8.0, "high": 25.0},
    "max_position_pct": {"type": "float", "low": 0.05, "high": 0.20},
    "kelly_fraction": {"type": "float", "low": 0.25, "high": 0.75},
    "drawdown_guard_threshold": {"type": "float", "low": 0.02, "high": 0.08},
    "drawdown_guard_scale": {"type": "float", "low": 0.3, "high": 0.8},
    # Bandit params
    "bandit_alpha": {"type": "float", "low": 0.5, "high": 3.0},
    "bandit_min_samples": {"type": "int", "low": 5, "high": 30},
    "bandit_convergence_scale": {"type": "float", "low": 0.3, "high": 0.9},
    # Strategy params
    "lookback_window": {"type": "int", "low": 20, "high": 100},
    "signal_threshold": {"type": "float", "low": 0.3, "high": 0.8},
    "fee_buffer_multiplier": {"type": "float", "low": 1.0, "high": 3.0},
}


class BayesianOptimizer:
    """
    Wraps Optuna TPE sampler for Bayesian hyperparameter optimization.
    Calibrated for $1k capital: penalizes high drawdown and fee drag.
    """

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        param_space: Optional[Dict[str, Any]] = None,
        n_trials: int = 100,
        n_startup_trials: int = 15,
        study_name: str = "argus_hpt_1k",
        direction: str = "maximize",
        results_dir: str = "optimization/results",
        seed: int = 42,
        pruning: bool = True,
    ):
        """
        Args:
            objective_fn: Function that accepts a dict of params and returns a scalar score.
            param_space: Dict defining the search space. Uses DEFAULT_PARAM_SPACE if None.
            n_trials: Total number of Optuna trials.
            n_startup_trials: Random exploration trials before TPE kicks in.
            study_name: Name of the Optuna study.
            direction: 'maximize' or 'minimize'.
            results_dir: Directory to save results JSON.
            seed: Random seed for reproducibility.
            pruning: Enable MedianPruner for early stopping of bad trials.
        """
        self.objective_fn = objective_fn
        self.param_space = param_space or DEFAULT_PARAM_SPACE
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.study_name = study_name
        self.direction = direction
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.pruning = pruning
        self.study: Optional[optuna.Study] = None

    def _build_sampler(self) -> TPESampler:
        return TPESampler(
            n_startup_trials=self.n_startup_trials,
            seed=self.seed,
            multivariate=True,  # Captures param correlations
        )

    def _build_pruner(self):
        if self.pruning:
            return MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=5)
        return optuna.pruners.NopPruner()

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters from the defined search space."""
        params = {}
        for name, spec in self.param_space.items():
            ptype = spec["type"]
            if ptype == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif ptype == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown param type: {ptype}")
        return params

    def _wrapped_objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        try:
            score = self.objective_fn(params)
            if not np.isfinite(score):
                logger.warning(f"Trial {trial.number}: non-finite score {score}, pruning.")
                raise optuna.exceptions.TrialPruned()
            return score
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()

    def run(self) -> optuna.Study:
        """Run the Bayesian optimization study."""
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=self._build_sampler(),
            pruner=self._build_pruner(),
        )
        logger.info(f"Starting Bayesian optimization: {self.n_trials} trials, study='{self.study_name}'")
        self.study.optimize(
            self._wrapped_objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )
        self._save_results()
        return self.study

    def _save_results(self) -> None:
        """Save best params and trial summary to JSON."""
        if self.study is None:
            return
        best = self.study.best_trial
        results = {
            "study_name": self.study_name,
            "n_trials": self.n_trials,
            "best_trial_number": best.number,
            "best_value": best.value,
            "best_params": best.params,
            "n_completed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }
        out_path = self.results_dir / f"{self.study_name}_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")
        logger.info(f"Best value: {best.value:.6f} | Params: {best.params}")

    def best_params(self) -> Optional[Dict[str, Any]]:
        """Return best params from completed study."""
        if self.study is None:
            return None
        return self.study.best_params

    def importance(self) -> Optional[Dict[str, float]]:
        """Return parameter importances (requires completed study)."""
        if self.study is None or len(self.study.trials) < 2:
            return None
        try:
            return optuna.importance.get_param_importances(self.study)
        except Exception as e:
            logger.warning(f"Could not compute importances: {e}")
            return None


class SmallCapObjective:
    """
    Example objective function factory calibrated for $1k capital.
    Combines ROI, Sharpe ratio, and drawdown penalty.
    Replace `simulate_strategy` with your actual backtester call.
    """

    def __init__(
        self,
        capital: float = 1000.0,
        roi_weight: float = 0.5,
        sharpe_weight: float = 0.3,
        drawdown_weight: float = 0.2,
    ):
        self.capital = capital
        self.roi_weight = roi_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight

    def __call__(self, params: Dict[str, Any]) -> float:
        """
        Evaluate params and return composite score.
        Higher = better.
        """
        roi, sharpe, max_dd = self.simulate_strategy(params)
        # Penalise drawdown heavily for small capital
        dd_penalty = max(0.0, max_dd - 0.03) * 10.0
        score = (
            self.roi_weight * roi
            + self.sharpe_weight * sharpe
            - self.drawdown_weight * dd_penalty
        )
        return float(score)

    def simulate_strategy(self, params: Dict[str, Any]):
        """
        Placeholder: replace with real backtester.
        Returns (roi, sharpe, max_drawdown) as floats.
        """
        # Stub: random values for testing scaffolding
        np.random.seed(abs(hash(str(params))) % (2**31))
        roi = np.random.normal(0.02, 0.01)
        sharpe = np.random.normal(1.2, 0.3)
        max_dd = abs(np.random.normal(0.03, 0.01))
        return roi, sharpe, max_dd
