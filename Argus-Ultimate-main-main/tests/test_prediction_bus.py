"""Tests for canonical ML prediction bundles and decision bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from core.strategy.signal import SignalSide
from ml.ensemble_signal_hub import EnsembleSignal
from ml.ensemble_predictor import EnsemblePrediction
from ml.prediction_bus import MLDecisionBus, PredictionBundle
from ml.signal_fusion import FusedSignal
from ml.trading_decision_controls import ConfidenceTradeGate


class TestPredictionBundle:
    def test_construct_clamps_fields_and_normalizes_action(self):
        bundle = PredictionBundle(
            symbol="BTCUSDT",
            action="unknown",
            direction=2.0,
            strength=1.5,
            confidence=1.2,
            regime_confidence=-1.0,
            size_multiplier=3.0,
        )

        assert bundle.direction == 1.0
        assert bundle.strength == 1.0
        assert bundle.confidence == 1.0
        assert bundle.regime_confidence == 0.0
        assert bundle.size_multiplier == 2.0
        assert bundle.action == "buy"

    def test_from_ensemble_signal_maps_existing_shape(self):
        signal = EnsembleSignal(
            composite=0.7,
            confidence=0.8,
            size_multiplier=1.2,
            regime_bias="BULLISH",
            sources={"alpha": 0.7},
        )

        bundle = PredictionBundle.from_ensemble_signal("BTCUSDT", signal)

        assert bundle.symbol == "BTCUSDT"
        assert bundle.action == "buy"
        assert bundle.direction == 0.7
        assert bundle.confidence == 0.8
        assert bundle.sources == {"alpha": 0.7}
        assert "ensemble_signal" in bundle.ml_outputs

    def test_from_fused_signal_maps_action_and_metadata(self):
        fused = FusedSignal(
            direction=-0.6,
            strength=0.7,
            confidence=0.9,
            action="sell",
            sources_used=2,
            source_contributions={"technical": 0.6, "regime": 0.4},
            timestamp=datetime.now(timezone.utc),
            metadata={"has_conflict": False},
        )

        bundle = PredictionBundle.from_fused_signal("ETHUSDT", fused, regime="TREND_DOWN")

        assert bundle.action == "sell"
        assert bundle.direction == -0.6
        assert bundle.regime == "TREND_DOWN"
        assert bundle.metadata["has_conflict"] is False

    def test_from_ensemble_prediction_uses_mean_prediction_and_confidence(self):
        prediction = EnsemblePrediction(
            predictions=np.array([0.5, 0.7, 0.9]),
            confidence=np.array([0.6, 0.8, 1.0]),
            model_predictions={},
            model_weights={"a": 0.4, "b": 0.6},
            combination_method="weighted_average",
            n_models_used=2,
            timestamp=datetime.now(timezone.utc),
        )

        bundle = PredictionBundle.from_ensemble_prediction("BTCUSDT", prediction, regime="TREND_UP")

        assert bundle.action == "buy"
        assert bundle.direction == np.mean([0.5, 0.7, 0.9])
        assert bundle.confidence == np.mean([0.6, 0.8, 1.0])
        assert bundle.model_prediction_vector == [0.5, 0.7, 0.9]

    def test_to_signal_maps_action_to_core_signal(self):
        bundle = PredictionBundle(
            symbol="BTCUSDT",
            action="sell",
            direction=-0.8,
            strength=0.75,
            confidence=0.8,
        )

        signal = bundle.to_signal(strategy_id="test_bus")

        assert signal.side == SignalSide.SHORT
        assert signal.strategy_id == "test_bus"
        assert 0.0 < signal.strength <= 1.0
        assert signal.metadata["symbol"] == "BTCUSDT"


class TestMLDecisionBus:
    def test_empty_decision_returns_neutral_signal(self):
        bus = MLDecisionBus()

        decision = bus.decide([])

        assert decision.bundle.action == "hold"
        assert decision.signal.side == SignalSide.FLAT
        assert decision.bundle.metadata["reason"] == "no_bundles"

    def test_decision_bus_merges_bundles_by_confidence(self):
        bus = MLDecisionBus()
        bullish = PredictionBundle("BTCUSDT", "buy", 0.9, 0.8, 0.9, sources={"a": 1})
        bearish = PredictionBundle("BTCUSDT", "sell", -0.4, 0.8, 0.2, sources={"b": 1})

        decision = bus.decide([bullish, bearish])

        assert decision.bundle.action == "buy"
        assert decision.bundle.direction > 0.0
        assert decision.signal.side == SignalSide.LONG
        assert decision.bundle.metadata["merged_bundles"] == 2

    def test_decision_bus_applies_confidence_gate(self):
        bus = MLDecisionBus(gate=ConfidenceTradeGate(min_confidence=0.7))
        weak = PredictionBundle("BTCUSDT", "buy", 0.8, 0.8, 0.4)

        decision = bus.decide([weak])

        assert decision.bundle.action == "hold"
        assert decision.signal.side == SignalSide.FLAT
        assert decision.gate_decision is not None
        assert decision.gate_decision.reason == "confidence_below_threshold"

    def test_decision_to_dict_is_serializable_shape(self):
        bus = MLDecisionBus(gate=ConfidenceTradeGate(min_confidence=0.2))
        bundle = PredictionBundle("BTCUSDT", "buy", 0.5, 0.7, 0.8)

        data = bus.decide([bundle]).to_dict()

        assert data["bundle"]["symbol"] == "BTCUSDT"
        assert data["signal"]["side"] == "LONG"
        assert data["gate_decision"]["should_trade"] is True
