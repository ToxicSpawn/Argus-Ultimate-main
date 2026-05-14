"""
ml/optuna_tuner.py — Bayesian Hyperparameter Optimization via Optuna TPE

Replaces the random GA search with Tree-structured Parzen Estimator (TPE)
sampling for strategy hyperparameter optimization.  Persists studies to SQLite
so results survive restarts.

Falls back gracefully when ``optuna`` is not installed: all public methods
return empty/default values and log a warning once.

Usage::

    tuner = OptunaTuner()
    best = tuner.optimize(
        strategy_name="momentum",
        objective_fn=my_objective,   # fn(trial) -> float (higher = better)
        n_trials=50,
    )
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Optuna import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import optuna  # type: ignore[import-untyped]

    _HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    _HAS_OPTUNA = False

_OPTUNA_WARN_ONCE = False

_DEFAULT_DB_PATH = os.path.join("data", "optuna_studies.db")


def _warn_no_optuna() -> None:
    """Emit a single warning when optuna is unavailable."""
    global _OPTUNA_WARN_ONCE
    if not _OPTUNA_WARN_ONCE:
        log.warning(
            "optuna is not installed — OptunaTuner is disabled. "
            "Install with: pip install optuna"
        )
        _OPTUNA_WARN_ONCE = True


class OptunaTuner:
    """Bayesian hyperparameter optimiser backed by Optuna TPE sampler.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file used for Optuna study persistence.
        Defaults to ``data/optuna_studies.db``.
    direction : str
        ``"maximize"`` or ``"minimize"`` — optimization direction.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        direction: str = "maximize",
    ) -> None:
        self._db_path = db_path
        self._direction = direction
        self._studies: Dict[str, Any] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        if _HAS_OPTUNA:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self._storage_url = f"sqlite:///{db_path}"
            log.info("OptunaTuner initialised — storage %s", self._storage_url)
        else:
            self._storage_url = None
            _warn_no_optuna()

    # ------------------------------------------------------------------
    # Study lifecycle
    # ------------------------------------------------------------------

    def create_study(
        self,
        strategy_name: str,
        params_spec: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create (or load) an Optuna study for *strategy_name*.

        Parameters
        ----------
        strategy_name : str
            Unique name used as the Optuna study name.
        params_spec : dict, optional
            Stored for later reference but not used to create the study itself.

        Returns
        -------
        optuna.Study or None
            The study object, or ``None`` when optuna is not available.
        """
        if not _HAS_OPTUNA:
            _warn_no_optuna()
            return None

        with self._lock:
            if strategy_name in self._studies:
                return self._studies[strategy_name]

            study = optuna.create_study(
                study_name=strategy_name,
                storage=self._storage_url,
                sampler=optuna.samplers.TPESampler(),
                direction=self._direction,
                load_if_exists=True,
            )
            self._studies[strategy_name] = study
            log.info(
                "Optuna study '%s' created/loaded (%d existing trials)",
                strategy_name,
                len(study.trials),
            )
            return study

    # ------------------------------------------------------------------
    # Parameter suggestion helper
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_params(
        trial: Any,
        params_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Suggest parameters for a single *trial* based on *params_spec*.

        ``params_spec`` maps parameter names to dicts describing the search
        space.  Supported keys:

        * ``{"type": "float", "low": 0.0, "high": 1.0}``
        * ``{"type": "float", "low": 0.0, "high": 1.0, "log": True}``
        * ``{"type": "int", "low": 1, "high": 100}``
        * ``{"type": "categorical", "choices": ["a", "b", "c"]}``

        Returns
        -------
        dict
            Mapping of parameter name to suggested value.
        """
        if not _HAS_OPTUNA:
            _warn_no_optuna()
            return {}

        params: Dict[str, Any] = {}
        for name, spec in params_spec.items():
            ptype = spec.get("type", "float")
            if ptype == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif ptype == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                )
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    spec["choices"],
                )
            else:
                log.warning("OptunaTuner: unknown param type '%s' for '%s'", ptype, name)
        return params

    # ------------------------------------------------------------------
    # Optimise
    # ------------------------------------------------------------------

    def optimize(
        self,
        strategy_name: str,
        objective_fn: Callable[..., float],
        n_trials: int = 50,
        timeout_s: Optional[int] = 300,
        params_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run *n_trials* of Bayesian optimization.

        Parameters
        ----------
        strategy_name : str
            Study identifier (created if it doesn't exist).
        objective_fn : callable
            Called as ``objective_fn(trial)`` and must return a scalar.
            The *trial* object supports ``suggest_float``, ``suggest_int``,
            ``suggest_categorical`` (standard Optuna API).  If *params_spec*
            is provided, you may also call
            ``OptunaTuner.suggest_params(trial, params_spec)`` inside the
            objective for convenience.
        n_trials : int
            Number of optimisation trials.
        timeout_s : int or None
            Wall-clock timeout in seconds.
        params_spec : dict, optional
            Saved alongside results for reproducibility.

        Returns
        -------
        dict
            Best parameter dict found, or ``{}`` if optuna unavailable.
        """
        if not _HAS_OPTUNA:
            _warn_no_optuna()
            return {}

        study = self.create_study(strategy_name, params_spec)
        if study is None:
            return {}

        log.info(
            "OptunaTuner: starting %d trials for '%s' (timeout=%ss)",
            n_trials,
            strategy_name,
            timeout_s,
        )

        study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout_s,
            show_progress_bar=False,
        )

        best = study.best_params
        best_value = study.best_value
        log.info(
            "OptunaTuner: '%s' best value=%.6f params=%s",
            strategy_name,
            best_value,
            best,
        )

        with self._lock:
            self._results[strategy_name] = {
                "best_params": best,
                "best_value": best_value,
                "n_trials": len(study.trials),
                "params_spec": params_spec,
            }

        return best

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_best_params(self, strategy_name: str) -> Dict[str, Any]:
        """Return the best parameters for *strategy_name*, or ``{}``."""
        if not _HAS_OPTUNA:
            _warn_no_optuna()
            return {}

        with self._lock:
            cached = self._results.get(strategy_name)
            if cached:
                return cached["best_params"]

        # Try loading from storage
        study = self.create_study(strategy_name)
        if study is None or len(study.trials) == 0:
            return {}
        return study.best_params

    # ------------------------------------------------------------------
    # Persistence (JSON export / import)
    # ------------------------------------------------------------------

    def save_results(self, path: str) -> None:
        """Persist current best-results cache to a JSON file."""
        with self._lock:
            data = dict(self._results)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        log.info("OptunaTuner: saved results to %s", path)

    def load_results(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Load previously saved results from JSON.

        Returns
        -------
        dict
            Mapping of strategy name to result dict.
        """
        if not os.path.exists(path):
            log.warning("OptunaTuner: results file not found: %s", path)
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        with self._lock:
            self._results.update(data)
        log.info("OptunaTuner: loaded results from %s (%d entries)", path, len(data))
        return data
