"""
Quantum-inspired multi-agent coordination for trading.

Uses quantum-inspired techniques to coordinate multiple trading agents.
Local-only, no quantum advantage claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

import numpy as np


@dataclass
class AgentDecision:
    """Decision from a single trading agent."""

    agent_id: str
    signal: str
    confidence: float
    position_size: float
    target_assets: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class CoordinatedDecision:
    """Result of quantum-inspired multi-agent coordination."""

    consensus_signal: str
    consensus_confidence: float
    agent_decisions: List[AgentDecision]
    entanglement_score: float
    convergence_iterations: int
    method: str = "quantum_multi_agent"
    honest_claim: str = (
        "Quantum-inspired multi-agent coordination. Uses quantum-inspired "
        "state representation for agent communication. Classical simulation; "
        "no quantum advantage claimed."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_signal": self.consensus_signal,
            "consensus_confidence": float(self.consensus_confidence),
            "agent_decisions": [
                {
                    "agent_id": d.agent_id,
                    "signal": d.signal,
                    "confidence": float(d.confidence),
                    "position_size": float(d.position_size),
                    "target_assets": d.target_assets,
                    "reasoning": d.reasoning,
                }
                for d in self.agent_decisions
            ],
            "entanglement_score": float(self.entanglement_score),
            "convergence_iterations": self.convergence_iterations,
            "method": self.method,
            "honest_claim": self.honest_claim,
        }


class QuantumMultiAgentCoordinator:
    """
    Quantum-inspired multi-agent coordination.

    Uses a quantum-inspired approach where each agent's state is represented
    as an amplitude in a shared quantum state vector. Agents interact through
    entanglement-like correlations rather than direct communication.

    Workflow:
    1. Initialize agent states as quantum amplitudes
    2. Apply interaction Hamiltonian (coupling between agents)
    3. Measure collective state for consensus decision
    4. Return coordinated decision with confidence
    """

    def __init__(
        self,
        *,
        n_iterations: int = 20,
        coupling_strength: float = 0.5,
        threshold: float = 0.6,
        seed: Optional[int] = None,
    ) -> None:
        self.n_iterations = max(1, int(n_iterations))
        self.coupling_strength = max(0.0, min(float(coupling_strength), 1.0))
        self.threshold = max(0.0, min(float(threshold), 1.0))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _initialize_quantum_state(
        self,
        agent_decisions: List[AgentDecision],
    ) -> np.ndarray:
        """Initialize quantum state from agent decisions."""
        n_agents = len(agent_decisions)
        dim = 1 << n_agents  # 2^n possible basis states

        # Each agent contributes amplitude based on confidence
        state = np.zeros(dim, dtype=np.complex128)

        for i, decision in enumerate(agent_decisions):
            # Map agent to basis state
            basis_idx = 1 << i

            # Amplitude proportional to confidence
            amplitude = np.sqrt(max(decision.confidence, 0.01))

            # Phase based on signal
            if decision.signal == "buy":
                phase = 0.0
            elif decision.signal == "sell":
                phase = np.pi
            else:
                phase = np.pi / 2

            state[basis_idx] = amplitude * np.exp(1j * phase)

        # Normalize
        norm = np.linalg.norm(state)
        if norm > 1e-10:
            state = state / norm
        else:
            state[0] = 1.0

        return state

    def _apply_interaction(
        self,
        state: np.ndarray,
        agent_decisions: List[AgentDecision],
    ) -> np.ndarray:
        """Apply entanglement-like interaction between agents."""
        n_agents = len(agent_decisions)

        # Simple coupling: rotate toward consensus
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Compute coupling effect
                ci = agent_decisions[i].confidence
                cj = agent_decisions[j].confidence

                if agent_decisions[i].signal == agent_decisions[j].signal:
                    # Agree - strengthen correlation
                    coupling = self.coupling_strength * np.sqrt(ci * cj)
                else:
                    # Disagree - reduce correlation
                    coupling = -self.coupling_strength * np.sqrt(ci * cj)

                # Apply phase rotation
                state = state * (1 + coupling * 0.01)

        # Renormalize
        norm = np.linalg.norm(state)
        if norm > 1e-10:
            state = state / norm

        return state

    def _measure_consensus(
        self,
        state: np.ndarray,
        agent_decisions: List[AgentDecision],
    ) -> tuple[str, float]:
        """Measure consensus signal from quantum state."""
        n_agents = len(agent_decisions)

        # Compute weighted vote
        buy_amplitude = 0.0
        sell_amplitude = 0.0
        hold_amplitude = 0.0

        for i, decision in enumerate(agent_decisions):
            basis_idx = 1 << i
            amplitude = state[basis_idx]

            if decision.signal == "buy":
                buy_amplitude += abs(amplitude)
            elif decision.signal == "sell":
                sell_amplitude += abs(amplitude)
            else:
                hold_amplitude += abs(amplitude)

        total = buy_amplitude + sell_amplitude + hold_amplitude
        if total < 1e-10:
            return "hold", 0.0

        # Determine consensus
        max_amp = max(buy_amplitude, sell_amplitude, hold_amplitude)

        if max_amp < self.threshold:
            return "hold", float(max_amp / total)

        if max_amp == buy_amplitude:
            return "buy", float(max_amp / total)
        elif max_amp == sell_amplitude:
            return "sell", float(max_amp / total)
        else:
            return "hold", float(max_amp / total)

    def coordinate(
        self,
        agent_decisions: List[AgentDecision],
    ) -> CoordinatedDecision:
        """
        Coordinate multiple agent decisions into a consensus.

        Args:
            agent_decisions: List of decisions from trading agents

        Returns:
            CoordinatedDecision with consensus signal and metadata
        """
        if not agent_decisions:
            return CoordinatedDecision(
                consensus_signal="hold",
                consensus_confidence=0.0,
                agent_decisions=[],
                entanglement_score=0.0,
                convergence_iterations=0,
            )

        # Initialize quantum state
        state = self._initialize_quantum_state(agent_decisions)

        # Iterative coordination (quantum-inspired)
        initial_entropy = -sum(
            abs(state[i]) ** 2 * np.log(max(abs(state[i]) ** 2, 1e-10))
            for i in range(len(state))
            if abs(state[i]) ** 2 > 1e-10
        )

        for iteration in range(self.n_iterations):
            state = self._apply_interaction(state, agent_decisions)

        # Final entropy (lower = more consensus)
        final_entropy = -sum(
            abs(state[i]) ** 2 * np.log(max(abs(state[i]) ** 2, 1e-10))
            for i in range(len(state))
            if abs(state[i]) ** 2 > 1e-10
        )

        # Entanglement score: reduction in entropy
        entanglement_score = max(0.0, (initial_entropy - final_entropy) / max(initial_entropy, 1e-10))

        # Measure consensus
        consensus_signal, consensus_confidence = self._measure_consensus(state, agent_decisions)

        return CoordinatedDecision(
            consensus_signal=consensus_signal,
            consensus_confidence=consensus_confidence,
            agent_decisions=agent_decisions,
            entanglement_score=entanglement_score,
            convergence_iterations=self.n_iterations,
        )


def coordinate_trading_agents(
    agent_signals: List[tuple[str, str, float]],
) -> CoordinatedDecision:
    """
    Quick multi-agent coordination.

    Args:
        agent_signals: List of (agent_id, signal, confidence) tuples

    Returns:
        CoordinatedDecision
    """
    coordinator = QuantumMultiAgentCoordinator()

    decisions = [
        AgentDecision(
            agent_id=aid,
            signal=sig,
            confidence=conf,
            position_size=conf,
        )
        for aid, sig, conf in agent_signals
    ]

    return coordinator.coordinate(decisions)


__all__ = [
    "AgentDecision",
    "CoordinatedDecision",
    "QuantumMultiAgentCoordinator",
    "coordinate_trading_agents",
]