"""Production inference utilities for the microstructure foundation model."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .feature_engineering import MicrostructureFeatureEngineer
from .feature_extractor import FoundationModelFeatureExtractor
from .model import TradeFoundationModel
from .tokenizer import UniversalMicrostructureTokenizer

logger = logging.getLogger(__name__)

try:
    from core.strategy.signal import Signal, SignalSide
except Exception:  # pragma: no cover - optional integration
    Signal = None
    SignalSide = None


@dataclass(slots=True)
class InferenceConfig:
    """Settings for live and batch inference."""

    device: str = "cpu"
    max_batch_size: int = 64
    confidence_temperature: float = 1.0
    strategy_id: str = "tradefm_foundation"
    min_sequence_length: int = 1


@dataclass(slots=True)
class PredictionResult:
    """Prediction payload for downstream systems."""

    next_token_id: int
    confidence: float
    probabilities: Sequence[float]
    embedding: Sequence[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FoundationModelInference:
    """Real-time and batch inference wrapper around the foundation model."""

    def __init__(
        self,
        model: TradeFoundationModel,
        tokenizer: Optional[UniversalMicrostructureTokenizer] = None,
        feature_engineer: Optional[MicrostructureFeatureEngineer] = None,
        extractor: Optional[FoundationModelFeatureExtractor] = None,
        config: Optional[InferenceConfig] = None,
    ) -> None:
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer or UniversalMicrostructureTokenizer()
        self.feature_engineer = feature_engineer or MicrostructureFeatureEngineer()
        self.extractor = extractor or FoundationModelFeatureExtractor(self.model, tokenizer=self.tokenizer, device=self.config.device)

    @staticmethod
    def _entropy_confidence(probabilities: torch.Tensor) -> float:
        entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-8)), dim=-1)
        max_entropy = math.log(probabilities.size(-1)) if probabilities.size(-1) > 1 else 1.0
        confidence = 1.0 - (entropy / max(max_entropy, 1e-8))
        return float(confidence.mean().detach().cpu().item())

    def _event_to_model_input(self, events: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        if len(events) < self.config.min_sequence_length:
            raise ValueError("events sequence is too short for inference")
        token_ids = self.tokenizer.to_token_ids(events)
        if not token_ids:
            raise ValueError("tokenization produced an empty sequence")
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)

    def real_time_feature_extraction(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        features = self.feature_engineer.transform(event)
        return features.as_dict()

    def extract_representation(self, events: Sequence[Mapping[str, Any]]) -> Sequence[float]:
        token_ids = self.tokenizer.to_token_ids(events)
        representation = self.extractor.get_market_state_embeddings(token_ids)
        return representation.pooled_embedding.detach().cpu().tolist()

    @torch.no_grad()
    def predict_next_event(self, events: Sequence[Mapping[str, Any]]) -> PredictionResult:
        model_input = self._event_to_model_input(events)
        output = self.model(model_input)
        logits = output["logits"][:, -1, :] / max(self.config.confidence_temperature, 1e-6)
        probabilities = F.softmax(logits, dim=-1)
        next_token_id = int(torch.argmax(probabilities, dim=-1).item())
        confidence = self._entropy_confidence(probabilities)
        representation = self.extractor.get_market_state_embeddings(model_input.squeeze(0).tolist())
        return PredictionResult(
            next_token_id=next_token_id,
            confidence=confidence,
            probabilities=probabilities.squeeze(0).detach().cpu().tolist(),
            embedding=representation.pooled_embedding.detach().cpu().tolist(),
            metadata={"sequence_length": model_input.size(1)},
        )

    @torch.no_grad()
    def batch_predict(self, batch_events: Sequence[Sequence[Mapping[str, Any]]]) -> Sequence[PredictionResult]:
        results = []
        for events in batch_events[: self.config.max_batch_size]:
            results.append(self.predict_next_event(events))
        return results

    def publish_to_signal_bus(self, symbol: str, prediction: PredictionResult, signal_bus: Any) -> None:
        if signal_bus is None:
            logger.debug("Signal bus unavailable; skipping publish for %s", symbol)
            return
        if Signal is None or SignalSide is None:
            logger.debug("Signal integration unavailable; skipping publish for %s", symbol)
            return
        if prediction.next_token_id % 3 == 0:
            side = SignalSide.LONG
        elif prediction.next_token_id % 3 == 1:
            side = SignalSide.SHORT
        else:
            side = SignalSide.FLAT
        signal = Signal(
            symbol=symbol,
            side=side,
            strength=max(0.0, min(1.0, prediction.confidence)),
            strategy_id=self.config.strategy_id,
            metadata={
                "next_token_id": prediction.next_token_id,
                "top_probability": max(prediction.probabilities) if prediction.probabilities else 0.0,
            },
        )
        try:
            publish_sync = getattr(signal_bus, "publish_sync", None)
            if callable(publish_sync):
                publish_sync(signal)
            else:
                publish = getattr(signal_bus, "publish", None)
                if callable(publish):
                    result = publish(signal)
                    if hasattr(result, "__await__"):
                        logger.debug("Async signal bus publish returned awaitable for %s", symbol)
                else:
                    raise AttributeError("signal_bus does not expose publish_sync or publish")
        except Exception:
            logger.exception("Failed to publish foundation-model signal for %s", symbol)

    def predict_and_publish(self, symbol: str, events: Sequence[Mapping[str, Any]], signal_bus: Any) -> PredictionResult:
        prediction = self.predict_next_event(events)
        self.publish_to_signal_bus(symbol, prediction, signal_bus)
        return prediction
