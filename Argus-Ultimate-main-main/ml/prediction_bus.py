"""Canonical ML prediction contract and decision bus.

The project has many ML producers: ensemble signals, fused signals, model
ensembles, regime routers, and strategy bridges. This module gives them one
small, serializable contract so risk, execution, monitoring, and backtests can
consume ML decisions consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import numpy as np

from core.strategy.signal import Signal, SignalSide
from ml.trading_decision_controls import ConfidenceTradeGate, TradeGateDecision


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), lower, upper))


def _action_from_direction(direction: float, strength: float, threshold: float = 0.15) -> str:
    if abs(direction) < threshold or strength < 0.2:
        return "hold"
    return "buy" if direction > 0 else "sell"


@dataclass
class PredictionBundle:
    """Canonical ML output for trading decisions."""

    symbol: str
    action: str
    direction: float
    strength: float
    confidence: float
    regime: str = "UNKNOWN"
    regime_confidence: float = 0.0
    size_multiplier: float = 1.0
    sources: Dict[str, Any] = field(default_factory=dict)
    ml_outputs: Dict[str, Any] = field(default_factory=dict)
    model_prediction_vector: Optional[list[float]] = None
    regime_weights: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.direction = _clip(self.direction, -1.0, 1.0)
        self.strength = _clip(self.strength, 0.0, 1.0)
        self.confidence = _clip(self.confidence, 0.0, 1.0)
        self.regime_confidence = _clip(self.regime_confidence, 0.0, 1.0)
        self.size_multiplier = _clip(self.size_multiplier, 0.0, 2.0)
        self.action = str(self.action).lower()
        if self.action not in {"buy", "sell", "hold", "reduce"}:
            self.action = _action_from_direction(self.direction, self.strength)

    @classmethod
    def from_ensemble_signal(cls, symbol: str, ensemble_signal: Any, *, regime: str = "UNKNOWN") -> "PredictionBundle":
        direction = float(getattr(ensemble_signal, "composite", 0.0))
        confidence = float(getattr(ensemble_signal, "confidence", 0.0))
        size_multiplier = float(getattr(ensemble_signal, "size_multiplier", 1.0))
        timestamp = datetime.fromtimestamp(
            float(getattr(ensemble_signal, "timestamp", datetime.now(timezone.utc).timestamp())),
            tz=timezone.utc,
        )
        strength = min(abs(direction) * size_multiplier, 1.0)
        return cls(
            symbol=symbol,
            action=_action_from_direction(direction, strength),
            direction=direction,
            strength=strength,
            confidence=confidence,
            regime=str(getattr(ensemble_signal, "regime_bias", regime) or regime),
            regime_confidence=confidence,
            size_multiplier=size_multiplier,
            sources=dict(getattr(ensemble_signal, "sources", {}) or {}),
            ml_outputs={"ensemble_signal": cls._safe_to_dict(ensemble_signal)},
            timestamp=timestamp,
        )

    @classmethod
    def from_fused_signal(cls, symbol: str, fused_signal: Any, *, regime: str = "UNKNOWN") -> "PredictionBundle":
        direction = float(getattr(fused_signal, "direction", 0.0))
        strength = float(getattr(fused_signal, "strength", 0.0))
        confidence = float(getattr(fused_signal, "confidence", 0.0))
        return cls(
            symbol=symbol,
            action=str(getattr(fused_signal, "action", _action_from_direction(direction, strength))),
            direction=direction,
            strength=strength,
            confidence=confidence,
            regime=regime,
            regime_confidence=confidence,
            sources=dict(getattr(fused_signal, "source_contributions", {}) or {}),
            ml_outputs={"fused_signal": cls._safe_to_dict(fused_signal)},
            metadata=dict(getattr(fused_signal, "metadata", {}) or {}),
            timestamp=getattr(fused_signal, "timestamp", datetime.now(timezone.utc)),
        )

    @classmethod
    def from_ensemble_prediction(
        cls,
        symbol: str,
        ensemble_prediction: Any,
        *,
        regime: str = "UNKNOWN",
        regime_confidence: float = 0.0,
    ) -> "PredictionBundle":
        predictions = np.asarray(getattr(ensemble_prediction, "predictions", []), dtype=float).reshape(-1)
        confidence_values = np.asarray(getattr(ensemble_prediction, "confidence", []), dtype=float).reshape(-1)
        mean_prediction = float(np.mean(predictions)) if predictions.size else 0.0
        mean_confidence = float(np.mean(confidence_values)) if confidence_values.size else 0.0
        direction = _clip(mean_prediction, -1.0, 1.0)
        strength = min(abs(direction), 1.0)
        return cls(
            symbol=symbol,
            action=_action_from_direction(direction, strength),
            direction=direction,
            strength=strength,
            confidence=mean_confidence,
            regime=regime,
            regime_confidence=regime_confidence,
            sources=dict(getattr(ensemble_prediction, "model_weights", {}) or {}),
            ml_outputs={"ensemble_prediction": cls._safe_to_dict(ensemble_prediction)},
            model_prediction_vector=[float(value) for value in predictions.tolist()],
            timestamp=getattr(ensemble_prediction, "timestamp", datetime.now(timezone.utc)),
        )

    def apply_gate(self, gate: ConfidenceTradeGate) -> TradeGateDecision:
        decision = gate.evaluate(self.action, self.confidence, 1.0 - self.confidence)
        self.metadata["gate"] = decision.to_dict()
        if not decision.should_trade:
            self.action = decision.action
            self.size_multiplier = 0.0
        else:
            self.size_multiplier *= decision.size_multiplier
        return decision

    def to_signal(self, strategy_id: str = "ml_prediction_bus") -> Signal:
        side = SignalSide.FLAT
        if self.action == "buy":
            side = SignalSide.LONG
        elif self.action == "sell":
            side = SignalSide.SHORT
        return Signal(
            symbol=self.symbol,
            side=side,
            strength=float(np.clip(self.strength * self.confidence * max(self.size_multiplier, 0.0), 0.0, 1.0)),
            strategy_id=strategy_id,
            metadata=self.to_dict(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "direction": round(self.direction, 6),
            "strength": round(self.strength, 6),
            "confidence": round(self.confidence, 6),
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 6),
            "size_multiplier": round(self.size_multiplier, 6),
            "sources": self.sources,
            "ml_outputs": self.ml_outputs,
            "model_prediction_vector": self.model_prediction_vector,
            "regime_weights": self.regime_weights,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @staticmethod
    def _safe_to_dict(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            out: Dict[str, Any] = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, np.ndarray):
                    out[key] = value.tolist()
                elif isinstance(value, datetime):
                    out[key] = value.isoformat()
                else:
                    out[key] = value
            return out
        return {"value": str(obj)}


@dataclass
class MLDecision:
    """Final decision emitted by the ML decision bus."""

    bundle: PredictionBundle
    signal: Signal
    gate_decision: Optional[TradeGateDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "signal": {
                "symbol": self.signal.symbol,
                "side": self.signal.side.value,
                "strength": round(self.signal.strength, 6),
                "strategy_id": self.signal.strategy_id,
            },
            "gate_decision": self.gate_decision.to_dict() if self.gate_decision else None,
        }


class MLDecisionBus:
    """Assemble canonical ML decisions from one or more prediction bundles."""

    def __init__(self, gate: Optional[ConfidenceTradeGate] = None, strategy_id: str = "ml_decision_bus"):
        self.gate = gate
        self.strategy_id = strategy_id

    def decide(self, bundles: Iterable[PredictionBundle]) -> MLDecision:
        bundle_list = list(bundles)
        if not bundle_list:
            neutral = PredictionBundle(
                symbol="UNKNOWN",
                action="hold",
                direction=0.0,
                strength=0.0,
                confidence=0.0,
                metadata={"reason": "no_bundles"},
            )
            return MLDecision(neutral, neutral.to_signal(self.strategy_id))

        final_bundle = self._merge(bundle_list)
        gate_decision = final_bundle.apply_gate(self.gate) if self.gate else None
        return MLDecision(final_bundle, final_bundle.to_signal(self.strategy_id), gate_decision)

    def _merge(self, bundles: list[PredictionBundle]) -> PredictionBundle:
        weights = np.asarray([max(bundle.confidence, 1e-6) for bundle in bundles], dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-9)
        direction = float(np.sum([bundle.direction * weight for bundle, weight in zip(bundles, weights)]))
        strength = float(np.sum([bundle.strength * weight for bundle, weight in zip(bundles, weights)]))
        confidence = float(np.sum([bundle.confidence * weight for bundle, weight in zip(bundles, weights)]))
        size_multiplier = float(np.sum([bundle.size_multiplier * weight for bundle, weight in zip(bundles, weights)]))
        action = _action_from_direction(direction, strength)
        primary = bundles[0]
        sources = {f"bundle_{idx}": bundle.sources for idx, bundle in enumerate(bundles)}
        ml_outputs = {f"bundle_{idx}": bundle.to_dict() for idx, bundle in enumerate(bundles)}
        return PredictionBundle(
            symbol=primary.symbol,
            action=action,
            direction=direction,
            strength=strength,
            confidence=confidence,
            regime=primary.regime,
            regime_confidence=primary.regime_confidence,
            size_multiplier=size_multiplier,
            sources=sources,
            ml_outputs=ml_outputs,
            regime_weights=primary.regime_weights,
            metadata={"merged_bundles": len(bundles)},
        )
