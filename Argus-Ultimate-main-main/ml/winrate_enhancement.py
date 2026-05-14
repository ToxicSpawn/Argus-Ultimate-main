"""Win-rate enhancement utilities for Argus ML.

These helpers keep high-impact improvements modular:

* soft/probabilistic labels for training targets
* dynamic model weighting by recent outcomes and regime
* calibrated confidence adaptation
* conflict/anomaly abstention gates

They are intentionally dependency-light and safe to use in paper/live paths.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


SIGNAL_SELL = 0
SIGNAL_HOLD = 1
SIGNAL_BUY = 2


@dataclass
class ModelVote:
    """A single model vote in the shared 0=sell, 1=hold, 2=buy space."""

    name: str
    signal: int
    confidence: float
    probabilities: Optional[List[float]] = None


@dataclass
class CombinedVote:
    """Weighted vote result."""

    signal: int
    confidence: float
    class_scores: Dict[int, float]
    disagreement: float
    model_agreement: bool
    weights_used: Dict[str, float] = field(default_factory=dict)


class SoftLabelGenerator:
    """Create smooth 3-class labels from future returns.

    Hard labels discard useful information around the decision boundary. This
    generator returns probability targets where larger future returns sharpen the
    buy/sell class and small returns remain mostly hold.
    """

    def __init__(self, horizon: int = 4, neutral_threshold: float = 0.01, temperature: float = 0.35) -> None:
        self.horizon = max(1, int(horizon))
        self.neutral_threshold = float(neutral_threshold)
        self.temperature = max(1e-6, float(temperature))

    def future_returns(self, close: Iterable[float]) -> np.ndarray:
        prices = np.asarray(list(close), dtype=np.float64)
        out = np.zeros(len(prices), dtype=np.float64)
        if len(prices) <= self.horizon:
            return out
        denom = np.maximum(np.abs(prices[:-self.horizon]), 1e-12)
        out[:-self.horizon] = (prices[self.horizon:] - prices[:-self.horizon]) / denom
        return out

    def transform_returns(self, returns: Iterable[float]) -> np.ndarray:
        r = np.asarray(list(returns), dtype=np.float64)
        y = np.zeros((len(r), 3), dtype=np.float64)

        if len(r) == 0:
            return y

        scaled = np.clip(r / max(self.neutral_threshold, 1e-9), -8.0, 8.0)
        directional = 1.0 / (1.0 + np.exp(-scaled / self.temperature))
        trend_strength = np.clip(np.abs(r) / max(self.neutral_threshold, 1e-9), 0.0, 1.0)

        y[:, SIGNAL_HOLD] = 1.0 - 0.75 * trend_strength
        y[:, SIGNAL_BUY] = trend_strength * directional
        y[:, SIGNAL_SELL] = trend_strength * (1.0 - directional)
        y = y / np.maximum(y.sum(axis=1, keepdims=True), 1e-12)
        return y

    def from_close(self, close: Iterable[float]) -> np.ndarray:
        return self.transform_returns(self.future_returns(close))


class DynamicModelWeightManager:
    """Accuracy-weighted model voting with optional regime overlays."""

    def __init__(
        self,
        base_weights: Optional[Mapping[str, float]] = None,
        regime_weights: Optional[Mapping[int, Mapping[str, float]]] = None,
        state_path: str | Path = "data/ml_dynamic_weights.json",
        alpha: float = 0.12,
    ) -> None:
        self.base_weights = dict(base_weights or {
            "gradient_boosting": 0.50,
            "lstm": 0.30,
            "bayesian": 0.25,
            "rl": 0.20,
        })
        self.regime_weights = {int(k): dict(v) for k, v in (regime_weights or {}).items()}
        self.state_path = Path(state_path)
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self.reliability: Dict[str, float] = {name: 0.5 for name in self.base_weights}
        self._load()

    def get_weight(self, model_name: str, regime: Optional[int] = None) -> float:
        base = self.base_weights.get(model_name, 0.1)
        if regime is not None:
            base *= self.regime_weights.get(int(regime), {}).get(model_name, 1.0)
        reliability = self.reliability.get(model_name, 0.5)
        return max(0.0, base * (0.5 + reliability))

    def combine(self, votes: Iterable[ModelVote], regime: Optional[int] = None) -> CombinedVote:
        score = {SIGNAL_SELL: 0.0, SIGNAL_HOLD: 0.0, SIGNAL_BUY: 0.0}
        weights_used: Dict[str, float] = {}
        vote_list = list(votes)

        for vote in vote_list:
            if vote.signal not in score:
                continue
            confidence = min(1.0, max(0.0, float(vote.confidence)))
            weight = self.get_weight(vote.name, regime) * max(0.05, confidence)
            score[vote.signal] += weight
            weights_used[vote.name] = weight

        total = sum(score.values())
        if total <= 0:
            return CombinedVote(SIGNAL_HOLD, 0.0, score, 1.0, False, weights_used)

        signal = max(score.items(), key=lambda item: item[1])[0]
        confidence = score[signal] / total
        disagreement = 1.0 - confidence
        model_agreement = len({v.signal for v in vote_list if v.signal in score}) <= 1
        return CombinedVote(int(signal), float(confidence), score, float(disagreement), model_agreement, weights_used)

    def update_outcome(self, model_name: str, was_correct: bool) -> None:
        prev = self.reliability.get(model_name, 0.5)
        obs = 1.0 if was_correct else 0.0
        self.reliability[model_name] = self.alpha * obs + (1.0 - self.alpha) * prev
        self._save()

    def _load(self) -> None:
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                if isinstance(data, dict):
                    self.reliability.update({str(k): float(v) for k, v in data.get("reliability", {}).items()})
        except Exception as exc:
            logger.debug("Failed to load dynamic model weights: %s", exc)

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({"reliability": self.reliability}, indent=2))
        except Exception as exc:
            logger.debug("Failed to save dynamic model weights: %s", exc)


class ConflictAbstentionGate:
    """Turn low-quality consensus into wait/hold before risk sizing."""

    def __init__(
        self,
        min_confidence: float = 0.38,
        max_disagreement: float = 0.48,
        anomaly_confidence_multiplier: float = 0.50,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.max_disagreement = float(max_disagreement)
        self.anomaly_confidence_multiplier = float(anomaly_confidence_multiplier)

    def evaluate(
        self,
        vote: CombinedVote,
        *,
        anomaly_detected: bool = False,
        model_agreement: Optional[bool] = None,
    ) -> Tuple[bool, str, float]:
        agreement = vote.model_agreement if model_agreement is None else model_agreement
        adjusted_confidence = vote.confidence

        if anomaly_detected:
            adjusted_confidence *= self.anomaly_confidence_multiplier
            if not agreement:
                return True, "anomaly_and_model_conflict", adjusted_confidence

        if adjusted_confidence < self.min_confidence:
            return True, "low_calibrated_confidence", adjusted_confidence
        if vote.disagreement > self.max_disagreement and not agreement:
            return True, "high_model_disagreement", adjusted_confidence
        return False, "pass", adjusted_confidence


class CalibratedConfidenceAdapter:
    """Thin adapter around ``ConfidenceCalibrator`` with safe fallback."""

    def __init__(self, db_path: str | Path = "data/confidence_calibration.db") -> None:
        try:
            from ml.confidence_calibrator import ConfidenceCalibrator

            self._calibrator = ConfidenceCalibrator(db_path=db_path)
        except Exception as exc:
            logger.debug("Confidence calibrator unavailable: %s", exc)
            self._calibrator = None

    def calibrate(self, model_name: str, confidence: float) -> float:
        confidence = min(1.0, max(0.0, float(confidence)))
        if self._calibrator is None:
            return confidence
        try:
            return float(self._calibrator.calibrate(model_name, confidence))
        except Exception:
            return confidence

    def record(self, model_name: str, confidence: float, was_correct: bool) -> None:
        if self._calibrator is None:
            return
        try:
            self._calibrator.record_prediction(model_name, confidence, was_correct)
        except Exception as exc:
            logger.debug("Failed to record calibration outcome: %s", exc)


def compact_gb_features(df) -> np.ndarray:
    """Build the 9-feature vector used by the current gradient boosting artifacts."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / loss.clip(lower=1e-8)))
    price_pos = (close - low.rolling(24).min()) / (high.rolling(24).max() - low.rolling(24).min()).clip(lower=1e-8)
    volume_ratio = volume / volume.rolling(24).mean().clip(lower=1e-8)

    row = np.array([
        close.pct_change(1).iloc[-1],
        close.pct_change(4).iloc[-1],
        close.pct_change(12).iloc[-1],
        close.pct_change(24).iloc[-1],
        close.pct_change(1).rolling(12).std().iloc[-1],
        close.pct_change(1).rolling(24).std().iloc[-1],
        rsi.iloc[-1],
        price_pos.iloc[-1],
        volume_ratio.iloc[-1],
    ], dtype=np.float64)
    return np.nan_to_num(row.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)


def compact_deep_features(df) -> np.ndarray:
    """Build the 7-feature vector used by the current deep-learning artifacts."""
    gb = compact_gb_features(df)[0]
    return gb[[0, 1, 3, 4, 6, 7, 8]].reshape(1, -1)
