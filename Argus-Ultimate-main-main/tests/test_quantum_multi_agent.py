"""Tests for quantum multi-agent coordinator."""

from __future__ import annotations


def test_coordinator_initialization():
    from ml.quantum_multi_agent import QuantumMultiAgentCoordinator

    coord = QuantumMultiAgentCoordinator(n_iterations=10, seed=42)

    assert coord.n_iterations == 10
    assert coord.coupling_strength > 0


def test_coordinate_single_agent():
    from ml.quantum_multi_agent import (
        QuantumMultiAgentCoordinator,
        AgentDecision,
        CoordinatedDecision,
    )

    coord = QuantumMultiAgentCoordinator()

    decisions = [
        AgentDecision(agent_id="agent_1", signal="buy", confidence=0.8, position_size=0.5)
    ]

    result = coord.coordinate(decisions)

    assert isinstance(result, CoordinatedDecision)
    assert result.consensus_signal == "buy"
    assert len(result.agent_decisions) == 1


def test_coordinate_multiple_agents():
    from ml.quantum_multi_agent import (
        QuantumMultiAgentCoordinator,
        AgentDecision,
    )

    coord = QuantumMultiAgentCoordinator(n_iterations=10, seed=42)

    decisions = [
        AgentDecision(agent_id="trend", signal="buy", confidence=0.7, position_size=0.3),
        AgentDecision(agent_id="mean_rev", signal="sell", confidence=0.6, position_size=0.2),
        AgentDecision(agent_id="vol", signal="hold", confidence=0.5, position_size=0.1),
    ]

    result = coord.coordinate(decisions)

    assert result.consensus_signal in ("buy", "sell", "hold")
    assert 0.0 <= result.consensus_confidence <= 1.0
    assert result.convergence_iterations == 10
    assert result.entanglement_score >= 0.0


def test_coordinate_empty_agents():
    from ml.quantum_multi_agent import QuantumMultiAgentCoordinator

    coord = QuantumMultiAgentCoordinator()

    result = coord.coordinate([])

    assert result.consensus_signal == "hold"
    assert result.consensus_confidence == 0.0
    assert len(result.agent_decisions) == 0


def test_coordinate_trading_agents():
    from ml.quantum_multi_agent import coordinate_trading_agents

    signals = [
        ("agent_1", "buy", 0.8),
        ("agent_2", "buy", 0.7),
        ("agent_3", "sell", 0.6),
    ]

    result = coordinate_trading_agents(signals)

    assert result.consensus_signal in ("buy", "sell", "hold")
    assert len(result.agent_decisions) == 3


def test_coordinated_decision_serialization():
    from ml.quantum_multi_agent import (
        CoordinatedDecision,
        AgentDecision,
    )

    decision = CoordinatedDecision(
        consensus_signal="buy",
        consensus_confidence=0.75,
        agent_decisions=[
            AgentDecision(agent_id="a1", signal="buy", confidence=0.8, position_size=0.5),
            AgentDecision(agent_id="a2", signal="sell", confidence=0.6, position_size=0.3),
        ],
        entanglement_score=0.3,
        convergence_iterations=15,
    )

    payload = decision.to_dict()

    assert payload["consensus_signal"] == "buy"
    assert payload["consensus_confidence"] == 0.75
    assert len(payload["agent_decisions"]) == 2
    assert payload["entanglement_score"] == 0.3
    assert "no quantum advantage claimed" in payload["honest_claim"]


def test_strong_consensus():
    from ml.quantum_multi_agent import QuantumMultiAgentCoordinator, AgentDecision

    coord = QuantumMultiAgentCoordinator()

    # All agents agree on buy
    decisions = [
        AgentDecision(agent_id=f"agent_{i}", signal="buy", confidence=0.9, position_size=0.5)
        for i in range(4)
    ]

    result = coord.coordinate(decisions)

    assert result.consensus_signal == "buy"
    assert result.consensus_confidence > 0.5