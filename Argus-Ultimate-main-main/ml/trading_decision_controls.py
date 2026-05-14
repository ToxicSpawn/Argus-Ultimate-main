"""Trading ML decision controls: cost-aware scoring and confidence gates.

These utilities are deliberately dependency-light. They sit between model
outputs and trading actions so ML improvements can reduce overtrading and avoid
selecting high-accuracy models that are poor after fees/slippage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class CostModel:
    """Convert trading frictions into a normalized score penalty."""

    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    market_impact_bps: float = 0.0
    turnover: float = 1.0
    bps_to_score: float = 0.0001
    max_penalty: float = 1.0

    @property
    def total_cost_bps(self) -> float:
        raw_cost = self.fee_bps + self.slippage_bps + self.spread_bps + self.market_impact_bps
        return float(max(raw_cost, 0.0) * max(self.turnover, 0.0))

    def penalty(self) -> float:
        return float(np.clip(self.total_cost_bps * self.bps_to_score, 0.0, self.max_penalty))

    @classmethod
    def from_tca_report(
        cls,
        report: Dict[str, Any],
        *,
        turnover: float = 1.0,
        bps_to_score: float = 0.0001,
    ) -> "CostModel":
        average_costs = report.get("average_costs_bps", {}) if isinstance(report, dict) else {}
        return cls(
            fee_bps=float(report.get("fee_bps", 0.0)) if isinstance(report, dict) else 0.0,
            slippage_bps=float(average_costs.get("slippage", 0.0)),
            spread_bps=float(average_costs.get("spread", 0.0)),
            market_impact_bps=float(average_costs.get("market_impact", 0.0)),
            turnover=turnover,
            bps_to_score=bps_to_score,
        )


@dataclass
class CostAdjustedScore:
    """Raw model score plus normalized trading cost adjustment."""

    raw_score: float
    cost_penalty: float
    net_score: float
    total_cost_bps: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_score": round(self.raw_score, 6),
            "cost_penalty": round(self.cost_penalty, 6),
            "net_score": round(self.net_score, 6),
            "total_cost_bps": round(self.total_cost_bps, 4),
            "metadata": self.metadata,
        }


class CostAwareScorer:
    """Rank ML models by net score after estimated trading costs."""

    def __init__(self, default_cost_model: Optional[CostModel] = None):
        self.default_cost_model = default_cost_model or CostModel()

    def score(
        self,
        raw_score: float,
        cost_model: Optional[CostModel] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostAdjustedScore:
        model = cost_model or self.default_cost_model
        penalty = model.penalty()
        return CostAdjustedScore(
            raw_score=float(raw_score),
            cost_penalty=penalty,
            net_score=float(raw_score) - penalty,
            total_cost_bps=model.total_cost_bps,
            metadata=metadata or {},
        )

    def rank(self, candidates: Dict[str, CostAdjustedScore]) -> list[tuple[str, CostAdjustedScore]]:
        return sorted(candidates.items(), key=lambda item: item[1].net_score, reverse=True)


@dataclass
class CalibrationBucket:
    """Empirical accuracy for one confidence bucket."""

    lower: float
    upper: float
    count: int
    empirical_accuracy: float

    def contains(self, confidence: float) -> bool:
        return self.lower <= confidence <= self.upper


@dataclass
class CalibrationResult:
    """Output from calibrating a confidence estimate."""

    raw_confidence: float
    calibrated_confidence: float
    uncertainty: float
    bucket_count: int
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_confidence": round(self.raw_confidence, 6),
            "calibrated_confidence": round(self.calibrated_confidence, 6),
            "uncertainty": round(self.uncertainty, 6),
            "bucket_count": self.bucket_count,
            "method": self.method,
        }


class PredictionCalibrator:
    """Empirical confidence calibrator using historical correctness bins."""

    def __init__(self, n_bins: int = 10, min_bucket_samples: int = 5, shrinkage: float = 10.0):
        self.n_bins = max(int(n_bins), 2)
        self.min_bucket_samples = max(int(min_bucket_samples), 1)
        self.shrinkage = max(float(shrinkage), 0.0)
        self._buckets: list[CalibrationBucket] = []

    def fit(self, confidences: Iterable[float], outcomes: Iterable[bool]) -> "PredictionCalibrator":
        conf = np.clip(np.asarray(list(confidences), dtype=float), 0.0, 1.0)
        correct = np.asarray(list(outcomes), dtype=float)
        if conf.shape[0] != correct.shape[0]:
            raise ValueError("confidences and outcomes must have the same length")
        if conf.shape[0] == 0:
            raise ValueError("calibration data must not be empty")

        self._buckets = []
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        global_accuracy = float(np.mean(correct))
        for idx in range(self.n_bins):
            lower = float(edges[idx])
            upper = float(edges[idx + 1])
            if idx == self.n_bins - 1:
                mask = (conf >= lower) & (conf <= upper)
            else:
                mask = (conf >= lower) & (conf < upper)
            count = int(np.sum(mask))
            empirical = float(np.mean(correct[mask])) if count else global_accuracy
            weight = count / (count + self.shrinkage) if self.shrinkage > 0 else 1.0
            accuracy = empirical * weight + global_accuracy * (1.0 - weight)
            self._buckets.append(CalibrationBucket(lower, upper, count, float(np.clip(accuracy, 0.0, 1.0))))
        return self

    def calibrate(self, confidence: float) -> CalibrationResult:
        raw = float(np.clip(confidence, 0.0, 1.0))
        if not self._buckets:
            return CalibrationResult(raw, raw, 1.0 - raw, 0, "identity")

        bucket = self._find_bucket(raw)
        if bucket.count < self.min_bucket_samples:
            calibrated = (raw + bucket.empirical_accuracy) / 2.0
            method = "shrunk_bucket"
        else:
            calibrated = bucket.empirical_accuracy
            method = "bucket"
        calibrated = float(np.clip(calibrated, 0.0, 1.0))
        return CalibrationResult(
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            uncertainty=1.0 - calibrated,
            bucket_count=bucket.count,
            method=method,
        )

    def _find_bucket(self, confidence: float) -> CalibrationBucket:
        for bucket in self._buckets:
            if bucket.contains(confidence):
                return bucket
        return self._buckets[-1]


@dataclass
class TradeGateDecision:
    """Decision from confidence/uncertainty trade gating."""

    should_trade: bool
    action: str
    confidence: float
    uncertainty: float
    size_multiplier: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_trade": self.should_trade,
            "action": self.action,
            "confidence": round(self.confidence, 6),
            "uncertainty": round(self.uncertainty, 6),
            "size_multiplier": round(self.size_multiplier, 6),
            "reason": self.reason,
        }


class ConfidenceTradeGate:
    """Convert calibrated confidence into trade/no-trade and size scaling."""

    def __init__(
        self,
        min_confidence: float = 0.55,
        max_uncertainty: float = 0.55,
        min_size_multiplier: float = 0.0,
    ):
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.max_uncertainty = float(np.clip(max_uncertainty, 0.0, 1.0))
        self.min_size_multiplier = float(np.clip(min_size_multiplier, 0.0, 1.0))

    def evaluate(self, action: str, confidence: float, uncertainty: Optional[float] = None) -> TradeGateDecision:
        conf = float(np.clip(confidence, 0.0, 1.0))
        unc = float(np.clip(1.0 - conf if uncertainty is None else uncertainty, 0.0, 1.0))
        if action in {"hold", "reduce"}:
            return TradeGateDecision(False, action, conf, unc, 0.0, "non_entry_action")
        if conf < self.min_confidence:
            return TradeGateDecision(False, "hold", conf, unc, 0.0, "confidence_below_threshold")
        if unc > self.max_uncertainty:
            return TradeGateDecision(False, "hold", conf, unc, 0.0, "uncertainty_above_threshold")

        confidence_span = max(1.0 - self.min_confidence, 1e-9)
        confidence_scale = (conf - self.min_confidence) / confidence_span
        uncertainty_scale = 1.0 - min(unc / max(self.max_uncertainty, 1e-9), 1.0) * 0.5
        multiplier = self.min_size_multiplier + (1.0 - self.min_size_multiplier) * confidence_scale * uncertainty_scale
        return TradeGateDecision(True, action, conf, unc, float(np.clip(multiplier, 0.0, 1.0)), "trade_allowed")
