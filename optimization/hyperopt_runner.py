"""Argus Optuna HyperoptRunner — Push 51.

Runs an Optuna TPE study to optimise Argus trading parameters.
Best params are saved to optimization/best_params.json and can
be injected into the live config via StudyStore.apply_best_params().

Usage::

    python -m optimization.hyperopt_runner --trials 100
    python -m optimization.hyperopt_runner --trials 50 --out my_params.json

Environment variables::

    ARGUS_HYPEROPT_TRIALS   — number of Optuna trials (default 100)
    ARGUS_HYPEROPT_STORAGE  — SQLAlchemy DB URL for distributed study
                              (default: in-memory)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from optimization.objective import ArgusObjective
from optimization.param_space import ARGUS_DEFAULT_PARAM_SPACE
from optimization.study_store import StudyStore

logger = logging.getLogger(__name__)

DEFAULT_TRIALS = 100
STUDY_NAME = "argus-hyperopt"


class HyperoptRunner:
    """Orchestrates an Optuna study for Argus parameter optimisation.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials to run.
    storage : str, optional
        SQLAlchemy connection string. If None, uses in-memory study.
    out_path : Path, optional
        Where to write best_params.json.
    returns : array-like, optional
        Historical log-returns for the objective. If None, synthetic data used.
    """

    def __init__(
        self,
        n_trials: int = DEFAULT_TRIALS,
        storage: Optional[str] = None,
        out_path: Optional[Path] = None,
        returns: Optional[np.ndarray] = None,
    ) -> None:
        if not _OPTUNA_AVAILABLE:
            raise RuntimeError(
                "optuna is not installed. Run: pip install 'optuna>=3.6'"
            )
        self.n_trials = n_trials
        self.storage = storage
        self.store = StudyStore(path=out_path)
        self.objective = ArgusObjective(returns=returns)
        self._study: Optional[optuna.Study] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> optuna.Study:
        """Create and optimise the study. Returns the completed study."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study = optuna.create_study(
            study_name=STUDY_NAME,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            storage=self.storage,
            load_if_exists=True,
        )

        logger.info(
            "HyperoptRunner: starting study '%s' — %d trials",
            STUDY_NAME,
            self.n_trials,
        )

        self._study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        best = self._study.best_trial
        logger.info(
            "HyperoptRunner: best trial #%d — Sharpe=%.4f params=%s",
            best.number,
            best.value,
            best.params,
        )

        self.store.save(params=best.params, best_value=best.value)
        return self._study

    @property
    def best_params(self) -> dict:
        """Return best params from the completed study."""
        if self._study is None:
            return self.store.load()
        return self._study.best_trial.params

    @property
    def best_value(self) -> float:
        """Return best Sharpe from the completed study."""
        if self._study is None:
            return float("nan")
        return float(self._study.best_value)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Argus Optuna hyperopt runner")
    p.add_argument(
        "--trials",
        type=int,
        default=int(os.getenv("ARGUS_HYPEROPT_TRIALS", DEFAULT_TRIALS)),
        help="Number of Optuna trials (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for best_params.json (default: optimization/best_params.json)",
    )
    p.add_argument(
        "--storage",
        type=str,
        default=os.getenv("ARGUS_HYPEROPT_STORAGE"),
        help="SQLAlchemy storage URL for distributed study (optional)",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    runner = HyperoptRunner(
        n_trials=args.trials,
        storage=args.storage,
        out_path=args.out,
    )
    runner.run()
    print(f"Best Sharpe: {runner.best_value:.4f}")
    print(f"Best params: {runner.best_params}")
