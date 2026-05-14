"""Persist and apply Optuna best params — Push 51."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_BEST_PARAMS_PATH = Path("optimization/best_params.json")


class StudyStore:
    """Load and save the best hyperparameters produced by Optuna.

    Parameters
    ----------
    path : Path or str, optional
        File path for persisting best_params JSON.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_BEST_PARAMS_PATH
        self._best_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, params: Dict[str, Any], best_value: float) -> None:
        """Persist params + best_value to JSON."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_value": best_value,
            "params": params,
        }
        with open(self.path, "w") as fh:
            json.dump(payload, fh, indent=2)
        self._best_params = params
        logger.info("StudyStore saved best params → %s (Sharpe=%.4f)", self.path, best_value)

    def load(self) -> Dict[str, Any]:
        """Load best params from disk. Returns empty dict if file missing."""
        if not self.path.exists():
            logger.warning("StudyStore: no best_params file at %s", self.path)
            return {}
        with open(self.path) as fh:
            payload = json.load(fh)
        self._best_params = payload.get("params", {})
        logger.info(
            "StudyStore loaded best params from %s (Sharpe=%.4f)",
            self.path,
            payload.get("best_value", float("nan")),
        )
        return self._best_params

    # ------------------------------------------------------------------
    # Runtime injection
    # ------------------------------------------------------------------

    def apply_best_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge best_params into a live config dict and return it.

        Mapping:
            gateway_confidence   → config["gateway"]["min_confidence"]
            hmm_bull_scalar      → config["regime"]["bull_scalar"]
            hmm_bear_scalar      → config["regime"]["bear_scalar"]
            spread_bps           → config["spread"]["base_bps"]
            regime_refit_bars    → config["regime"]["refit_bars"]
        """
        params = self._best_params or self.load()
        if not params:
            logger.warning("StudyStore.apply_best_params: no params loaded, skipping")
            return config

        mapping = {
            "gateway_confidence": ("gateway", "min_confidence"),
            "hmm_bull_scalar": ("regime", "bull_scalar"),
            "hmm_bear_scalar": ("regime", "bear_scalar"),
            "spread_bps": ("spread", "base_bps"),
            "regime_refit_bars": ("regime", "refit_bars"),
        }

        for param_key, (section, key) in mapping.items():
            if param_key in params:
                config.setdefault(section, {})[key] = params[param_key]
                logger.debug("Applied %s=%s → config[%s][%s]", param_key, params[param_key], section, key)

        return config

    @property
    def best_params(self) -> Dict[str, Any]:
        return dict(self._best_params)
