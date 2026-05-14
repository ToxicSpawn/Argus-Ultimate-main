"""Production inference helpers for dynamic GNN market predictions."""

# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .dynamic_graph import GraphSnapshot
from .hybrid_model import HybridSpatialTemporalModel

logger = logging.getLogger(__name__)

try:
    from core.strategy.signal import Signal, SignalSide
except Exception:  # pragma: no cover - optional integration fallback
    Signal = None  # type: ignore[assignment]
    SignalSide = None  # type: ignore[assignment]


@dataclass(slots=True)
class InferenceResult:
    asset_predictions: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    attention_weights: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_importance: Dict[str, Dict[str, float]] = field(default_factory=dict)


class DynamicGNNInferenceEngine:
    def __init__(
        self,
        model: HybridSpatialTemporalModel,
        confidence_zscore: float = 1.96,
        strategy_id: str = "dynamic_gnn",
    ) -> None:
        self.model = model
        self.confidence_zscore = float(confidence_zscore)
        self.strategy_id = str(strategy_id)

    def predict_batch(
        self,
        graph_sequence: Sequence[GraphSnapshot],
        regime_features: Optional[np.ndarray] = None,
    ) -> InferenceResult:
        predictions, attention = self.model.forward(graph_sequence, regime_features=regime_features, return_attention=True)
        latest_graph = graph_sequence[-1].graph
        if predictions.shape[0] != len(latest_graph.nodes):
            raise ValueError("Prediction rows must match asset count")

        asset_predictions: Dict[str, Dict[str, float]] = {}
        confidence_intervals: Dict[str, Dict[str, float]] = {}
        feature_importance = self.feature_importance(attention, latest_graph.nodes)

        returns_pred = predictions[:, 0]
        vol_pred = predictions[:, 1] if predictions.shape[1] > 1 else np.zeros_like(returns_pred)
        ret_std = float(np.std(returns_pred) + 1e-6)
        vol_std = float(np.std(vol_pred) + 1e-6)

        for idx, asset in enumerate(latest_graph.nodes):
            asset_predictions[asset] = {
                "predicted_return": float(returns_pred[idx]),
                "predicted_volatility": float(vol_pred[idx]),
                "confidence": float(np.exp(-abs(vol_pred[idx]))),
            }
            confidence_intervals[asset] = {
                "return_lower": float(returns_pred[idx] - self.confidence_zscore * ret_std),
                "return_upper": float(returns_pred[idx] + self.confidence_zscore * ret_std),
                "vol_lower": float(max(0.0, vol_pred[idx] - self.confidence_zscore * vol_std)),
                "vol_upper": float(vol_pred[idx] + self.confidence_zscore * vol_std),
            }

        return InferenceResult(
            asset_predictions=asset_predictions,
            confidence_intervals=confidence_intervals,
            attention_weights=attention,
            feature_importance=feature_importance,
        )

    @staticmethod
    def feature_importance(attention: Mapping[str, np.ndarray], assets: Sequence[str]) -> Dict[str, Dict[str, float]]:
        importance: Dict[str, Dict[str, float]] = {str(asset): {} for asset in assets}
        for key, weights in attention.items():
            arr = np.asarray(weights, dtype=np.float32)
            if arr.ndim >= 2 and arr.shape[-1] == len(assets):
                score = np.mean(np.abs(arr), axis=tuple(range(arr.ndim - 1)))
                for idx, asset in enumerate(assets):
                    importance[str(asset)][key] = float(score[idx])
            else:
                scalar_score = float(np.mean(np.abs(arr))) if arr.size else 0.0
                for asset in assets:
                    importance[str(asset)][key] = scalar_score
        return importance

    async def publish_to_signal_bus(
        self,
        result: InferenceResult,
        signal_bus: Any,
        min_strength: float = 0.05,
    ) -> int:
        if signal_bus is None or Signal is None or SignalSide is None:
            logger.warning("Signal bus integration unavailable for dynamic GNN inference")
            return 0

        published = 0
        for asset, payload in result.asset_predictions.items():
            strength = float(min(1.0, abs(payload["predicted_return"])))
            if strength < min_strength:
                continue
            if payload["predicted_return"] > 0:
                side = SignalSide.LONG
            elif payload["predicted_return"] < 0:
                side = SignalSide.SHORT
            else:
                side = SignalSide.FLAT

            signal = Signal(
                symbol=asset,
                side=side,
                strength=strength,
                strategy_id=self.strategy_id,
                metadata={
                    "predicted_return": payload["predicted_return"],
                    "predicted_volatility": payload["predicted_volatility"],
                    "confidence": payload["confidence"],
                },
            )
            try:
                await signal_bus.publish(signal)
                published += 1
            except Exception:
                logger.exception("Failed to publish dynamic GNN signal for %s", asset)
        return published

    def publish_to_signal_bus_sync(
        self,
        result: InferenceResult,
        signal_bus: Any,
        min_strength: float = 0.05,
    ) -> int:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.publish_to_signal_bus(result, signal_bus, min_strength=min_strength))
                return 0
            return loop.run_until_complete(self.publish_to_signal_bus(result, signal_bus, min_strength=min_strength))
        except RuntimeError:
            return asyncio.run(self.publish_to_signal_bus(result, signal_bus, min_strength=min_strength))
