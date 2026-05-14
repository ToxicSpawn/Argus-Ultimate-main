"""Tests for local quantum strategy advisory integration."""

from __future__ import annotations

from core.strategy.signal import SignalSide
from ml.quantum_strategy_advisor import (
    LocalQuantumStrategyAdvisor,
    QuantumTailRiskConsensus,
    QuantumStrategyAdvice,
    QuantumWalkFeatureSummary,
)


class FakeQuantumFacade:
    def __init__(self, weights):
        self.weights = weights
        self.calls = []

    def run_quantum_walk(self, returns, *, correlation_threshold=0.3, max_steps=50, strategy="centrality"):
        self.calls.append({
            "returns": returns,
            "correlation_threshold": correlation_threshold,
            "max_steps": max_steps,
            "strategy": strategy,
        })
        return {
            "weights": self.weights,
            "walk_entropy": 0.42,
            "mixing_time": 7,
            "method": "szegedy_quantum_walk",
            "quantum_metadata": {
                "execution_mode": "classical_statevector_simulation",
                "hardware_enabled": False,
                "honest_claim": "Fake local quantum walk; no hardware quantum advantage is claimed.",
            },
        }


class FailingQuantumFacade:
    def run_quantum_walk(self, returns, *, correlation_threshold=0.3, max_steps=50, strategy="centrality"):
        raise RuntimeError("quantum unavailable")

    def estimate_tail_risk_qmc(self, returns, *, n_samples=10000, confidence=0.95):
        raise RuntimeError("qmc unavailable")

    def estimate_tail_risk_mlqae(self, returns, *, confidence=0.95, n_samples=10000, n_qubits=4):
        raise RuntimeError("mlqae unavailable")


class TailRiskFacade(FakeQuantumFacade):
    def __init__(self):
        super().__init__({"BTCUSDT": 0.6, "ETHUSDT": 0.25, "SOLUSDT": 0.15})
        self.tail_calls = []

    def estimate_tail_risk_qmc(self, returns, *, n_samples=10000, confidence=0.95):
        self.tail_calls.append(("qmc", n_samples, confidence))
        return {
            "var": -0.035,
            "cvar": -0.052,
            "method": "sobol_qmc",
            "quantum_metadata": {"capability": "qmc_var_cvar"},
        }

    def estimate_tail_risk_mlqae(self, returns, *, confidence=0.95, n_samples=10000, n_qubits=4):
        self.tail_calls.append(("mlqae", n_samples, confidence, n_qubits))
        return {
            "var_95": -0.041,
            "cvar_95": -0.049,
            "method": "mlqae_in_repo",
            "quantum_metadata": {"capability": "mlqae_var"},
        }


def _returns():
    return {
        "BTCUSDT": [0.01, 0.02, -0.01, 0.03],
        "ETHUSDT": [0.0, -0.01, 0.02, 0.01],
        "SOLUSDT": [-0.02, 0.01, 0.0, 0.01],
    }


class TestLocalQuantumStrategyAdvisor:
    def test_disabled_advisor_returns_neutral_hold(self):
        advisor = LocalQuantumStrategyAdvisor(enabled=False)

        advice = advisor.advise("BTCUSDT", _returns())

        assert advice.action == "hold"
        assert advice.confidence == 0.0
        assert advice.metadata["reason"] == "quantum_advisor_disabled"

    def test_overweight_asset_generates_buy_advice(self):
        facade = FakeQuantumFacade({"BTCUSDT": 0.6, "ETHUSDT": 0.25, "SOLUSDT": 0.15})
        advisor = LocalQuantumStrategyAdvisor(facade=facade, neutral_band=0.01)

        advice = advisor.advise("BTCUSDT", _returns())

        assert isinstance(advice, QuantumStrategyAdvice)
        assert advice.action == "buy"
        assert advice.direction > 0.0
        assert advice.strength > 0.0
        assert advice.size_multiplier > 1.0
        assert advice.execution_mode == "classical_statevector_simulation"
        assert isinstance(advice.feature_summary, QuantumWalkFeatureSummary)
        assert advice.feature_summary.directional_bias > 0.0
        assert advice.feature_summary.conviction > 0.0
        assert advice.feature_summary.entropy == 0.42
        assert advice.feature_summary.mixing_time == 7.0
        assert facade.calls[0]["correlation_threshold"] == 0.3

    def test_feature_summary_is_bounded_and_serialized(self):
        facade = FakeQuantumFacade({"BTCUSDT": 0.6, "ETHUSDT": 0.25, "SOLUSDT": 0.15})
        advisor = LocalQuantumStrategyAdvisor(facade=facade, neutral_band=0.01)

        advice = advisor.advise("BTCUSDT", _returns())
        payload = advice.to_dict()

        features = payload["feature_summary"]
        assert 0.0 <= features["concentration"] <= 1.0
        assert 0.0 <= features["dispersion"] <= 1.0
        assert 0.0 <= features["conviction"] <= 1.0
        assert features["directional_bias"] == round(advice.direction, 6)
        assert features["entropy"] == 0.42
        assert features["mixing_time"] == 7.0

    def test_underweight_asset_generates_sell_advice(self):
        facade = FakeQuantumFacade({"BTCUSDT": 0.1, "ETHUSDT": 0.55, "SOLUSDT": 0.35})
        advisor = LocalQuantumStrategyAdvisor(facade=facade, neutral_band=0.01)

        advice = advisor.advise("BTCUSDT", _returns())

        assert advice.action == "sell"
        assert advice.direction < 0.0
        assert advice.size_multiplier < 1.0

    def test_equal_weight_stays_neutral(self):
        facade = FakeQuantumFacade({"BTCUSDT": 1 / 3, "ETHUSDT": 1 / 3, "SOLUSDT": 1 / 3})
        advisor = LocalQuantumStrategyAdvisor(facade=facade, neutral_band=0.05)

        advice = advisor.advise("BTCUSDT", _returns())

        assert advice.action == "hold"
        assert advice.strength == 0.0
        assert advice.size_multiplier == 1.0

    def test_quantum_failure_returns_neutral_fallback(self):
        advisor = LocalQuantumStrategyAdvisor(facade=FailingQuantumFacade())

        advice = advisor.advise("BTCUSDT", _returns())

        assert advice.action == "hold"
        assert advice.metadata["reason"] == "quantum_walk_failed"
        assert "quantum unavailable" in advice.metadata["error"]
        assert advice.feature_summary is not None
        assert advice.feature_summary.conviction == 0.0
        assert advice.feature_summary.directional_bias == 0.0

    def test_missing_or_short_returns_are_neutral(self):
        advisor = LocalQuantumStrategyAdvisor(facade=FakeQuantumFacade({"BTCUSDT": 1.0}), min_history=3)

        advice = advisor.advise("BTCUSDT", {"BTCUSDT": [0.01], "ETHUSDT": [0.02]})

        assert advice.action == "hold"
        assert advice.metadata["reason"] == "symbol_missing_from_returns"

    def test_advice_bundle_maps_to_prediction_bundle_and_signal(self):
        facade = FakeQuantumFacade({"BTCUSDT": 0.6, "ETHUSDT": 0.25, "SOLUSDT": 0.15})
        advisor = LocalQuantumStrategyAdvisor(facade=facade, neutral_band=0.01)

        bundle = advisor.advise_bundle("BTCUSDT", _returns(), regime="TREND_UP", regime_confidence=0.8)
        signal = bundle.to_signal(strategy_id="quantum_advisory")

        assert bundle.action == "buy"
        assert bundle.regime == "TREND_UP"
        assert bundle.sources["quantum_walk"]["BTCUSDT"] == 0.6
        assert bundle.metadata["quantum_features"]["directional_bias"] > 0.0
        assert bundle.metadata["quantum_features"]["conviction"] > 0.0
        assert signal.side == SignalSide.LONG
        assert signal.strategy_id == "quantum_advisory"

    def test_tail_risk_consensus_uses_conservative_estimates(self):
        facade = TailRiskFacade()
        advisor = LocalQuantumStrategyAdvisor(facade=facade, min_history=3)

        consensus = advisor.assess_tail_risk([0.01, -0.02, 0.005, -0.04], n_samples=128, n_qubits=2)

        assert isinstance(consensus, QuantumTailRiskConsensus)
        assert consensus.var == -0.041
        assert consensus.cvar == -0.052
        assert consensus.estimator_count == 2
        assert consensus.risk_level == "high"
        assert consensus.disagreement > 0.0
        assert consensus.qmc["method"] == "sobol_qmc"
        assert consensus.mlqae["method"] == "mlqae_in_repo"
        assert ("qmc", 128, 0.95) in facade.tail_calls
        assert ("mlqae", 128, 0.95, 2) in facade.tail_calls

    def test_tail_risk_consensus_serializes_honest_metadata(self):
        advisor = LocalQuantumStrategyAdvisor(facade=TailRiskFacade(), min_history=3)

        payload = advisor.assess_tail_risk([0.01, -0.02, 0.005, -0.04]).to_dict()

        assert payload["var"] == -0.041
        assert payload["cvar"] == -0.052
        assert payload["estimator_count"] == 2
        assert payload["risk_level"] == "high"
        assert "no hardware quantum advantage" in payload["honest_claim"]

    def test_tail_risk_failure_returns_unknown_fallback(self):
        advisor = LocalQuantumStrategyAdvisor(facade=FailingQuantumFacade(), min_history=3)

        consensus = advisor.assess_tail_risk([0.01, -0.02, 0.005, -0.04])

        assert consensus.risk_level == "unknown"
        assert consensus.estimator_count == 0
        assert consensus.var == 0.0
        assert consensus.cvar == 0.0
        assert consensus.qmc["error"] == "qmc unavailable"
        assert consensus.mlqae["error"] == "mlqae unavailable"
