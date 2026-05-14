"""Batch 3 — Optuna hyperparameter optimiser wrapper.

Provides a drop-in optimiser that integrates with the walk-forward backtester
and exposes a clean interface for defining parameter search spaces.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Type for the objective factory:
# Takes a dict of params, returns a float score (maximise)
ObjectiveFactory = Callable[[Dict[str, Any], pd.DataFrame], float]


class OptunaOptimiser:
    """Wrap Optuna to optimise strategy hyperparameters in-sample."""

    def __init__(
        self,
        n_trials: int = 100,
        n_jobs: int = 1,
        direction: str = "maximize",
        pruner: Optional[str] = "median",
        sampler: Optional[str] = "tpe",
        study_name: str = "argus_optim",
        storage: Optional[str] = None,
    ) -> None:
        if optuna is None:
            raise ImportError("optuna is not installed. Run: pip install optuna")
        self._n_trials = n_trials
        self._n_jobs = n_jobs
        self._direction = direction
        self._pruner_name = pruner
        self._sampler_name = sampler
        self._study_name = study_name
        self._storage = storage
        self._study: Optional[optuna.Study] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimise(
        self,
        param_space: Callable[[optuna.Trial], Dict[str, Any]],
        objective_factory: ObjectiveFactory,
        train_df: pd.DataFrame,
        callbacks: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Run Optuna optimisation; return best params dict.

        Parameters
        ----------
        param_space : function(trial) → dict of suggested params.
        objective_factory : function(params, train_df) → float score.
        train_df : in-sample data passed to objective.
        """
        sampler = self._build_sampler()
        pruner = self._build_pruner()

        self._study = optuna.create_study(
            study_name=self._study_name,
            direction=self._direction,
            sampler=sampler,
            pruner=pruner,
            storage=self._storage,
            load_if_exists=self._storage is not None,
        )

        def _objective(trial: optuna.Trial) -> float:
            params = param_space(trial)
            try:
                return objective_factory(params, train_df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Trial %d failed: %s", trial.number, exc)
                raise optuna.exceptions.TrialPruned() from exc

        self._study.optimize(
            _objective,
            n_trials=self._n_trials,
            n_jobs=self._n_jobs,
            callbacks=callbacks,
            show_progress_bar=False,
        )

        best = self._study.best_params
        logger.info("Optuna best value=%.6f params=%s", self._study.best_value, best)
        return best

    def get_importance(self) -> Dict[str, float]:
        """Return feature importances from the completed study."""
        if self._study is None:
            return {}
        return optuna.importance.get_param_importances(self._study)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sampler(self):
        mapping = {
            "tpe": optuna.samplers.TPESampler,
            "cmaes": optuna.samplers.CmaEsSampler,
            "random": optuna.samplers.RandomSampler,
            "nsga2": optuna.samplers.NSGAIISampler,
        }
        cls = mapping.get(self._sampler_name or "tpe", optuna.samplers.TPESampler)
        return cls()

    def _build_pruner(self):
        if self._pruner_name is None:
            return optuna.pruners.NopPruner()
        mapping = {
            "median": optuna.pruners.MedianPruner,
            "hyperband": optuna.pruners.HyperbandPruner,
            "successive_halving": optuna.pruners.SuccessiveHalvingPruner,
        }
        cls = mapping.get(self._pruner_name, optuna.pruners.MedianPruner)
        return cls()
