"""
matrix_evaluator.py — OctoBot-style Matrix Evaluator.  (Push 29b)

Aggregates signals from all registered tentacles into a single
weighted consensus signal. Mirrors OctoBot's EvaluatorMatrix which
combines TA, social, and real-time evaluators with configurable weights.

New in Push 29b
---------------
  MatrixEvaluator._resolve_mode() reads config/matrix_mode.json written by
  compare_matrix_modes.py --auto-apply, so the empirically best aggregation
  mode is always used without manual edits.  Falls back to WEIGHTED_MEAN if
  the file is missing, corrupt, or contains an unrecognised mode value.

Aggregation modes
-----------------
  WEIGHTED_MEAN   : sum(signal_i * weight_i * confidence_i) / sum(weight_i * confidence_i)
  MAJORITY_VOTE   : sign of sum of signals weighted by confidence
  MIN_AGREEMENT   : only emit signal if >= N tentacles agree on direction

Output
------
  MatrixResult
    .signal       : float [-1, 1]  final blended signal
    .conviction   : float [0, 1]   agreement strength across tentacles
    .action       : str  BUY | SELL | HOLD
    .breakdown    : list of per-tentacle EvalResult
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_tentacle import (
    BaseTentacle, EvalResult, TentacleType,
    TENTACLE_REGISTRY,
)

logger = logging.getLogger(__name__)

# Repo-root-relative path to the mode config written by compare_matrix_modes.py
_MODE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "matrix_mode.json"


class AggregationMode(str, Enum):
    WEIGHTED_MEAN = "WEIGHTED_MEAN"
    MAJORITY_VOTE = "MAJORITY_VOTE"
    MIN_AGREEMENT = "MIN_AGREEMENT"


class Action(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MatrixResult:
    signal: float
    conviction: float
    action: str
    breakdown: List[EvalResult] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.conviction >= 0.5 and self.action != Action.HOLD


class MatrixEvaluator:
    """
    Aggregates tentacle signals into a single blended matrix signal.

    Parameters
    ----------
    tentacles        : list of instantiated BaseTentacle objects.
                       If None, auto-loads all TA_EVALUATOR tentacles
                       from TENTACLE_REGISTRY.
    mode             : AggregationMode.  If None, resolved from
                       config/matrix_mode.json (Push 29b); falls back
                       to WEIGHTED_MEAN if file is absent or corrupt.
    buy_threshold    : signal must exceed this to emit BUY (default: 0.15)
    sell_threshold   : signal must be below -this to emit SELL (default: 0.15)
    min_agreement    : for MIN_AGREEMENT mode — minimum tentacles agreeing
    type_weights     : optional dict of TentacleType -> weight multiplier
    exclude_types    : list of TentacleType to exclude from aggregation
    """

    _DEFAULT_MODE = AggregationMode.WEIGHTED_MEAN

    def __init__(
        self,
        tentacles: Optional[List[BaseTentacle]] = None,
        mode: Optional[AggregationMode] = None,
        buy_threshold: float = 0.15,
        sell_threshold: float = 0.15,
        min_agreement: int = 2,
        type_weights: Optional[Dict[TentacleType, float]] = None,
        exclude_types: Optional[List[TentacleType]] = None,
    ) -> None:
        # Resolve mode: explicit arg > config file > default
        self._mode           = mode if mode is not None else self._resolve_mode()
        self._buy_threshold  = buy_threshold
        self._sell_threshold = sell_threshold
        self._min_agreement  = min_agreement
        self._type_weights   = type_weights or {}
        self._exclude_types  = set(exclude_types or [])

        self._tentacles = tentacles if tentacles is not None else self._auto_load_tentacles()

        logger.info(
            "MatrixEvaluator initialised with %d tentacles [mode=%s]",
            len(self._tentacles), self._mode.value,
        )

    # ------------------------------------------------------------------
    # Mode resolution (Push 29b)
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_mode(cls) -> AggregationMode:
        """
        Read the winning aggregation mode from config/matrix_mode.json.

        Written by ``compare_matrix_modes.py --auto-apply`` (Push 29a).
        Falls back silently to WEIGHTED_MEAN if:
          - file does not exist (first run / fresh clone)
          - file is malformed JSON
          - "mode" key is missing or not a valid AggregationMode value
        """
        try:
            if not _MODE_CONFIG_PATH.exists():
                logger.debug(
                    "config/matrix_mode.json not found — using default mode %s",
                    cls._DEFAULT_MODE.value,
                )
                return cls._DEFAULT_MODE

            with open(_MODE_CONFIG_PATH, "r") as fh:
                data = json.load(fh)

            raw_mode = data.get("mode", "")
            resolved = AggregationMode(raw_mode)

            sharpe      = data.get("mean_sharpe", "n/a")
            generated   = data.get("generated_at", "unknown")
            logger.info(
                "[Push 29b] Matrix mode loaded from config: %s "
                "(Sharpe=%.4f, generated=%s)",
                resolved.value,
                float(sharpe) if sharpe != "n/a" else 0.0,
                generated,
            )
            return resolved

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "config/matrix_mode.json unreadable (%s) — falling back to %s",
                exc, cls._DEFAULT_MODE.value,
            )
            return cls._DEFAULT_MODE
        except Exception as exc:
            logger.warning(
                "Unexpected error reading matrix_mode.json (%s) — using %s",
                exc, cls._DEFAULT_MODE.value,
            )
            return cls._DEFAULT_MODE

    @classmethod
    def get_configured_mode(cls) -> AggregationMode:
        """
        Public helper — returns whichever mode would be used if no explicit
        mode is passed to __init__.  Useful for logging / dashboards.
        """
        return cls._resolve_mode()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, candles: np.ndarray, **kwargs: Any) -> MatrixResult:
        results: List[EvalResult] = []
        for tentacle in self._tentacles:
            if tentacle.tentacle_type in self._exclude_types:
                continue
            result = tentacle.safe_evaluate(candles, **kwargs)
            results.append(result)

        if not results:
            return MatrixResult(
                signal=0.0, conviction=0.0, action=Action.HOLD,
                metadata={"reason": "no_tentacles"},
            )

        signal, conviction = self._aggregate(results)
        action = self._classify(signal, conviction)

        return MatrixResult(
            signal=round(signal, 4),
            conviction=round(conviction, 4),
            action=action,
            breakdown=results,
            metadata={
                "mode"           : self._mode.value,
                "n_tentacles"    : len(results),
                "buy_threshold"  : self._buy_threshold,
                "sell_threshold" : self._sell_threshold,
            },
        )

    def add_tentacle(self, tentacle: BaseTentacle) -> None:
        self._tentacles.append(tentacle)
        logger.info("MatrixEvaluator: added tentacle %s", tentacle.name)

    def remove_tentacle(self, name: str) -> bool:
        before = len(self._tentacles)
        self._tentacles = [t for t in self._tentacles if t.name != name]
        removed = len(self._tentacles) < before
        if removed:
            logger.info("MatrixEvaluator: removed tentacle %s", name)
        return removed

    def reset_all(self) -> None:
        for t in self._tentacles:
            t.reset()

    @property
    def tentacle_names(self) -> List[str]:
        return [t.name for t in self._tentacles]

    @property
    def active_mode(self) -> AggregationMode:
        """The AggregationMode currently in use."""
        return self._mode

    # ------------------------------------------------------------------
    # Aggregation modes
    # ------------------------------------------------------------------

    def _aggregate(self, results: List[EvalResult]) -> tuple:
        if self._mode == AggregationMode.WEIGHTED_MEAN:
            return self._weighted_mean(results)
        elif self._mode == AggregationMode.MAJORITY_VOTE:
            return self._majority_vote(results)
        elif self._mode == AggregationMode.MIN_AGREEMENT:
            return self._min_agreement_agg(results)
        return self._weighted_mean(results)

    def _effective_weight(self, result: EvalResult) -> float:
        tentacle  = next((t for t in self._tentacles if t.name == result.tentacle_name), None)
        base_w    = tentacle.weight if tentacle else 1.0
        t_type    = tentacle.tentacle_type if tentacle else TentacleType.TA_EVALUATOR
        type_mult = self._type_weights.get(t_type, 1.0)
        return base_w * type_mult

    def _weighted_mean(self, results: List[EvalResult]) -> tuple:
        total_weight = 0.0
        weighted_sum = 0.0
        for r in results:
            w = self._effective_weight(r) * r.confidence
            weighted_sum += r.signal * w
            total_weight += w
        if total_weight == 0:
            return 0.0, 0.0
        signal     = weighted_sum / total_weight
        conviction = min(1.0, total_weight / len(results))
        return signal, conviction

    def _majority_vote(self, results: List[EvalResult]) -> tuple:
        pos   = sum(r.confidence for r in results if r.signal > 0)
        neg   = sum(r.confidence for r in results if r.signal < 0)
        total = pos + neg
        if total == 0:
            return 0.0, 0.0
        if pos > neg:
            return pos / total, pos / total
        elif neg > pos:
            return -(neg / total), neg / total
        return 0.0, 0.0

    def _min_agreement_agg(self, results: List[EvalResult]) -> tuple:
        buyers  = [r for r in results if r.signal > 0]
        sellers = [r for r in results if r.signal < 0]
        if len(buyers) >= self._min_agreement and len(buyers) > len(sellers):
            return self._weighted_mean(buyers)
        if len(sellers) >= self._min_agreement and len(sellers) > len(buyers):
            return self._weighted_mean(sellers)
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, signal: float, conviction: float) -> str:
        if signal >= self._buy_threshold:
            return Action.BUY
        if signal <= -self._sell_threshold:
            return Action.SELL
        return Action.HOLD

    # ------------------------------------------------------------------
    # Auto-load from registry
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_load_tentacles() -> List[BaseTentacle]:
        try:
            from . import ta_evaluators  # noqa: F401
            from . import dip_analyser   # noqa: F401
            from . import llm_signal     # noqa: F401
        except ImportError as exc:
            logger.warning("MatrixEvaluator auto-load import warning: %s", exc)

        instances = []
        for name, cls in TENTACLE_REGISTRY.items():
            try:
                instances.append(cls())
            except Exception as exc:
                logger.warning("Could not instantiate tentacle %s: %s", name, exc)
        return instances

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summary(self, result: MatrixResult) -> Dict[str, Any]:
        return {
            "action"    : result.action,
            "signal"    : result.signal,
            "conviction": result.conviction,
            "actionable": result.is_actionable,
            "mode"      : self._mode.value,
            "breakdown" : [
                {
                    "tentacle"  : r.tentacle_name,
                    "signal"    : r.signal,
                    "confidence": r.confidence,
                    "weighted"  : round(r.weighted_signal, 4),
                    "metadata"  : r.metadata,
                }
                for r in result.breakdown
            ],
        }
