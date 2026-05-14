# pyright: reportMissingImports=false
"""
Advanced Learning Integration Module for Argus Trading.

This module integrates all learning systems into a cohesive framework that can be
used by the main trading loop for continuous learning and adaptation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for the integrated system."""
    FULL = auto()  # All systems active
    LIGHTWEIGHT = auto()  # Only essential systems
    QUANTUM = auto()  # Quantum-enhanced learning
    CLASSICAL = auto()  # Classical learning only
    TRAINING = auto()  # Training mode


@dataclass
class LearningConfig:
    """Configuration for integrated learning system."""
    mode: LearningMode = LearningMode.FULL
    enable_quantum_rl: bool = True
    enable_multi_agent: bool = True
    enable_knowledge_distillation: bool = True
    enable_rlhf: bool = True
    enable_uncertainty: bool = True
    enable_adversarial: bool = True
    enable_active_learning: bool = True
    enable_transfer_learning: bool = True
    enable_dashboard: bool = True
    uncertainty_threshold: float = 0.6
    min_confidence: float = 0.5
    learning_rate: float = 0.001


@dataclass
class LearningDecision:
    """A decision from the integrated learning system."""
    action: int  # 0: hold, 1: buy, 2: sell, 3: hedge
    confidence: float
    uncertainty: float
    position_size: float
    reasoning: str
    agent_decisions: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Metrics for learning system performance."""
    total_decisions: int = 0
    avg_confidence: float = 0.0
    avg_uncertainty: float = 0.0
    learning_rate: float = 0.0
    active_systems: int = 0
    quantum_advantage: float = 0.0


class AdvancedLearningOrchestrator:
    """
    Main orchestrator that coordinates all learning systems.
    
    This class integrates:
    - Quantum Reinforcement Learning (QQL, QDQN, QPG, QAC)
    - Multi-Agent RL
    - Knowledge Distillation
    - RLHF (Human Feedback)
    - Uncertainty Quantification
    - Adversarial Training
    - Active Learning
    - Transfer Learning
    - Learning Health Dashboard
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        """Initialize the advanced learning orchestrator."""
        self.config = config or LearningConfig()
        self.systems: Dict[str, Any] = {}
        self.is_initialized = False
        self.metrics = LearningMetrics()
        self.decision_history: List[LearningDecision] = []
        
        logger.info("Initializing Advanced Learning Orchestrator")
        self._initialize_systems()

    def _initialize_systems(self) -> None:
        """Initialize all learning systems."""
        # 1. Knowledge Distillation
        if self.config.enable_knowledge_distillation:
            try:
                from ml.knowledge_distillation import KnowledgeDistillationSystem, DistillationConfig
                self.systems["knowledge_distillation"] = KnowledgeDistillationSystem(
                    DistillationConfig(epochs=50)
                )
                logger.info("✓ Knowledge Distillation initialized")
            except ImportError as e:
                logger.warning(f"Knowledge Distillation not available: {e}")

        # 2. Multi-Agent RL
        if self.config.enable_multi_agent:
            try:
                from ml.multi_agent_rl import MultiAgentSystem, EnsembleMethod
                self.systems["multi_agent_rl"] = MultiAgentSystem(
                    EnsembleMethod.PERFORMANCE_WEIGHTED
                )
                logger.info("✓ Multi-Agent RL initialized")
            except ImportError as e:
                logger.warning(f"Multi-Agent RL not available: {e}")

        # 3. RLHF System
        if self.config.enable_rlhf:
            try:
                from ml.rlhf_system import RLHFSystem, OnlineRLHF
                self.systems["rlhf"] = OnlineRLHF(update_frequency=50)
                logger.info("✓ RLHF System initialized")
            except ImportError as e:
                logger.warning(f"RLHF System not available: {e}")

        # 4. Uncertainty Quantification
        if self.config.enable_uncertainty:
            try:
                from ml.uncertainty_quantification import (
                    UncertaintyQuantifier, UncertaintyConfig, UncertaintyMethod, RiskAwareTradingSystem
                )
                quantifier = UncertaintyQuantifier(
                    UncertaintyConfig(method=UncertaintyMethod.ENSEMBLE, min_confidence=self.config.min_confidence)
                )
                self.systems["uncertainty"] = RiskAwareTradingSystem(quantifier)
                logger.info("✓ Uncertainty Quantification initialized")
            except ImportError as e:
                logger.warning(f"Uncertainty Quantification not available: {e}")

        # 5. Adversarial Training
        if self.config.enable_adversarial:
            try:
                from ml.adversarial_training import AdversarialGenerator, RobustTradingSystem
                self.systems["adversarial"] = RobustTradingSystem()
                logger.info("✓ Adversarial Training initialized")
            except ImportError as e:
                logger.warning(f"Adversarial Training not available: {e}")

        # 6. Active Learning
        if self.config.enable_active_learning:
            try:
                from ml.active_learning import ActiveLearner, ActiveLearningOrchestrator
                self.systems["active_learning"] = ActiveLearningOrchestrator()
                logger.info("✓ Active Learning initialized")
            except ImportError as e:
                logger.warning(f"Active Learning not available: {e}")

        # 7. Transfer Learning
        if self.config.enable_transfer_learning:
            try:
                from ml.transfer_learning import TransferLearner, TransferLearningOrchestrator
                self.systems["transfer_learning"] = TransferLearningOrchestrator()
                logger.info("✓ Transfer Learning initialized")
            except ImportError as e:
                logger.warning(f"Transfer Learning not available: {e}")

        # 8. Learning Health Dashboard
        if self.config.enable_dashboard:
            try:
                from ml.learning_health_dashboard import SystemIntegrator
                self.systems["dashboard"] = SystemIntegrator()
                logger.info("✓ Learning Health Dashboard initialized")
            except ImportError as e:
                logger.warning(f"Learning Health Dashboard not available: {e}")

        # 9. Quantum RL (if enabled)
        if self.config.enable_quantum_rl:
            try:
                from quantum.advanced.quantum_reinforcement_learning import (
                    QuantumReinforcementLearning, QuantumRLParameters
                )
                self.systems["quantum_rl"] = QuantumReinforcementLearning(
                    QuantumRLParameters(episodes=100, qubits=8)
                )
                logger.info("✓ Quantum RL initialized")
            except ImportError as e:
                logger.warning(f"Quantum RL not available: {e}")

        self.is_initialized = True
        self.metrics.active_systems = len(self.systems)
        
        logger.info(f"Advanced Learning Orchestrator initialized with {len(self.systems)} systems")

    def make_decision(self, 
                     market_state: NDArray[np.float64],
                     base_model_prediction: Optional[int] = None) -> LearningDecision:
        """
        Make a trading decision using all integrated learning systems.
        
        This is the main entry point for the trading loop.
        """
        if not self.is_initialized:
            logger.warning("Orchestrator not initialized, returning default decision")
            return LearningDecision(
                action=0,  # Hold
                confidence=0.5,
                uncertainty=0.5,
                position_size=1.0,
                reasoning="System not initialized"
            )

        agent_decisions: Dict[str, int] = {}
        confidences: List[float] = []
        uncertainties: List[float] = []

        # 1. Multi-Agent RL Decision
        if "multi_agent_rl" in self.systems:
            try:
                decision, metadata = self.systems["multi_agent_rl"].make_decision(market_state)
                agent_decisions["multi_agent"] = decision
                confidences.append(metadata.get("confidence", 0.5))
            except Exception as e:
                logger.warning(f"Multi-Agent RL error: {e}")

        # 2. Quantum RL Decision (if enabled)
        if "quantum_rl" in self.systems and self.config.enable_quantum_rl:
            try:
                quantum_agent = self.systems["quantum_rl"]
                action, metadata = quantum_agent.select_action(market_state)
                agent_decisions["quantum_rl"] = action
                confidences.append(metadata.get("selected_q_value", 0.5))
            except Exception as e:
                logger.warning(f"Quantum RL error: {e}")

        # 3. Base model prediction
        if base_model_prediction is not None:
            agent_decisions["base_model"] = base_model_prediction
            confidences.append(0.7)

        # 4. Ensemble decision
        final_action = self._ensemble_decision(agent_decisions)
        
        # 5. Uncertainty estimation
        uncertainty = 0.5
        if "uncertainty" in self.systems:
            try:
                # Simulate model predictions for uncertainty estimation
                predictions = [np.random.randn(4) * 0.5 for _ in range(5)]
                estimate = self.systems["uncertainty"].uncertainty_quantifier.estimate_uncertainty(
                    predictions, market_state
                )
                uncertainty = estimate.total_uncertainty
                uncertainties.append(uncertainty)
                
                # Risk-adjust position size
                position_size = estimate.risk_adjusted_position(1.0, 0.5)
            except Exception as e:
                logger.warning(f"Uncertainty estimation error: {e}")
                position_size = 1.0
        else:
            position_size = 1.0

        # 6. Calculate confidence
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        # 7. Build reasoning
        reasoning = self._build_reasoning(agent_decisions, avg_confidence, uncertainty)

        # 8. Create decision
        decision = LearningDecision(
            action=final_action,
            confidence=avg_confidence,
            uncertainty=uncertainty,
            position_size=position_size,
            reasoning=reasoning,
            agent_decisions=agent_decisions,
            metadata={
                "systems_used": list(self.systems.keys()),
                "num_agent_decisions": len(agent_decisions)
            }
        )

        # 9. Record decision
        self.decision_history.append(decision)
        self.metrics.total_decisions += 1
        self.metrics.avg_confidence = (
            self.metrics.avg_confidence * (self.metrics.total_decisions - 1) + avg_confidence
        ) / self.metrics.total_decisions
        self.metrics.avg_uncertainty = (
            self.metrics.avg_uncertainty * (self.metrics.total_decisions - 1) + uncertainty
        ) / self.metrics.total_decisions

        return decision

    def _ensemble_decision(self, agent_decisions: Dict[str, int]) -> int:
        """Combine decisions from multiple agents."""
        if not agent_decisions:
            return 0  # Hold

        # Weighted voting
        action_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        # Different weights for different agents
        weights = {
            "multi_agent": 1.5,
            "quantum_rl": 1.2,
            "base_model": 1.0
        }
        
        for agent_name, action in agent_decisions.items():
            weight = weights.get(agent_name, 1.0)
            action_scores[action] += weight

        # Return action with highest score
        return max(action_scores, key=action_scores.get)

    def _build_reasoning(self, 
                        agent_decisions: Dict[str, int],
                        confidence: float,
                        uncertainty: float) -> str:
        """Build human-readable reasoning for the decision."""
        action_names = {0: "hold", 1: "buy", 2: "sell", 3: "hedge"}
        
        parts = []
        
        for agent, action in agent_decisions.items():
            parts.append(f"{agent}: {action_names.get(action, 'unknown')}")
        
        reasoning = f"Ensemble decision based on {len(agent_decisions)} agents. "
        reasoning += f"Votes: {', '.join(parts)}. "
        reasoning += f"Confidence: {confidence:.2%}, Uncertainty: {uncertainty:.2%}. "
        
        if uncertainty > 0.4:
            reasoning += "High uncertainty - position size reduced."
        elif confidence > 0.7:
            reasoning += "High confidence - standard position size."
        
        return reasoning

    def update_with_feedback(self, 
                            market_state: NDArray[np.float64],
                            decision: LearningDecision,
                            reward: float,
                            human_feedback: Optional[float] = None) -> None:
        """
        Update learning systems with feedback from a trade.
        
        This is called after each trade to enable continuous learning.
        """
        # 1. Update Multi-Agent RL
        if "multi_agent_rl" in self.systems:
            try:
                self.systems["multi_agent_rl"].update_agents(reward)
            except Exception as e:
                logger.warning(f"Multi-Agent RL update error: {e}")

        # 2. Update RLHF if human feedback provided
        if "rlhf" in self.systems and human_feedback is not None:
            try:
                self.systems["rlhf"].collect_and_update(
                    market_state, decision.action, human_feedback
                )
            except Exception as e:
                logger.warning(f"RLHF update error: {e}")

        # 3. Update uncertainty calibration
        if "uncertainty" in self.systems:
            try:
                # Check if decision was correct (reward > 0)
                self.systems["uncertainty"].uncertainty_quantifier.calibrate(
                    decision.confidence, reward > 0
                )
            except Exception as e:
                logger.warning(f"Uncertainty calibration error: {e}")

        # 4. Update dashboard
        if "dashboard" in self.systems:
            try:
                self.systems["dashboard"].update_all_systems()
            except Exception as e:
                logger.warning(f"Dashboard update error: {e}")

        logger.debug(f"Updated learning systems with reward: {reward:.4f}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all learning systems."""
        status = {
            "initialized": self.is_initialized,
            "config": {
                "mode": self.config.mode.name,
                "quantum_enabled": self.config.enable_quantum_rl
            },
            "metrics": {
                "total_decisions": self.metrics.total_decisions,
                "avg_confidence": self.metrics.avg_confidence,
                "avg_uncertainty": self.metrics.avg_uncertainty,
                "active_systems": self.metrics.active_systems
            },
            "systems": {}
        }

        # Get individual system statuses
        for name, system in self.systems.items():
            try:
                if name == "dashboard":
                    status["systems"][name] = system.dashboard.get_health_summary()
                elif name == "multi_agent_rl":
                    status["systems"][name] = system.get_performance_summary()
                else:
                    status["systems"][name] = {"status": "active"}
            except Exception as e:
                status["systems"][name] = {"status": "error", "error": str(e)}

        return status

    def enable_learning_mode(self, mode: LearningMode) -> None:
        """Change learning mode."""
        self.config.mode = mode
        
        # Adjust system settings based on mode
        if mode == LearningMode.LIGHTWEIGHT:
            self.config.enable_quantum_rl = False
            self.config.enable_adversarial = False
        elif mode == LearningMode.QUANTUM:
            self.config.enable_quantum_rl = True
        elif mode == LearningMode.CLASSICAL:
            self.config.enable_quantum_rl = False
        
        logger.info(f"Learning mode changed to: {mode.name}")


class IntegratedTradingLoop:
    """
    Enhanced trading loop with integrated learning systems.
    
    This wraps the standard Argus trading loop with advanced learning capabilities.
    """

    def __init__(self, 
                 learning_orchestrator: Optional[AdvancedLearningOrchestrator] = None,
                 config: Optional[LearningConfig] = None):
        """Initialize the integrated trading loop."""
        self.orchestrator = learning_orchestrator or AdvancedLearningOrchestrator(config)
        self.trade_count = 0
        self.total_pnl = 0.0
        
        logger.info("Integrated Trading Loop initialized")

    def process_market_data(self, 
                           market_data: Dict[str, Any],
                           base_model_prediction: Optional[int] = None) -> Dict[str, Any]:
        """
        Process market data and make a trading decision.
        
        This is the main entry point for the trading system.
        """
        # Extract market state
        market_state = self._extract_market_state(market_data)
        
        # Get learning-based decision
        decision = self.orchestrator.make_decision(market_state, base_model_prediction)
        
        # Build response
        response = {
            "action": decision.action,
            "action_name": {0: "hold", 1: "buy", 2: "sell", 3: "hedge"}.get(decision.action, "unknown"),
            "confidence": decision.confidence,
            "uncertainty": decision.uncertainty,
            "position_size": decision.position_size,
            "reasoning": decision.reasoning,
            "agent_decisions": decision.agent_decisions,
            "metadata": decision.metadata
        }
        
        self.trade_count += 1
        
        logger.info(f"Trade #{self.trade_count}: {response['action_name']} "
                   f"(confidence: {decision.confidence:.2%}, "
                   f"position: {decision.position_size:.2f})")
        
        return response

    def _extract_market_state(self, market_data: Dict[str, Any]) -> NDArray[np.float64]:
        """Extract market state vector from market data."""
        # Simplified extraction - in reality would use proper feature engineering
        features = []
        
        # Price features
        if "close" in market_data:
            features.append(float(market_data["close"]))
        if "open" in market_data:
            features.append(float(market_data["open"]))
        if "high" in market_data:
            features.append(float(market_data["high"]))
        if "low" in market_data:
            features.append(float(market_data["low"]))
        
        # Volume features
        if "volume" in market_data:
            features.append(float(market_data["volume"]))
        
        # Technical indicators (simplified)
        features.extend([0.0] * (8 - len(features)))  # Pad to 8 features
        
        return np.array(features[:8])  # Ensure exactly 8 features

    def record_trade_outcome(self, 
                            market_data: Dict[str, Any],
                            decision: Dict[str, Any],
                            actual_reward: float,
                            human_rating: Optional[float] = None) -> None:
        """Record the outcome of a trade for learning."""
        market_state = self._extract_market_state(market_data)
        
        # Create a LearningDecision from the stored decision
        learning_decision = LearningDecision(
            action=decision["action"],
            confidence=decision["confidence"],
            uncertainty=decision["uncertainty"],
            position_size=decision["position_size"],
            reasoning=decision["reasoning"],
            agent_decisions=decision.get("agent_decisions", {}),
            metadata=decision.get("metadata", {})
        )
        
        # Update orchestrator with feedback
        self.orchestrator.update_with_feedback(
            market_state, learning_decision, actual_reward, human_rating
        )
        
        self.total_pnl += actual_reward
        
        logger.info(f"Trade outcome recorded: reward={actual_reward:.4f}, "
                   f"total_pnl={self.total_pnl:.4f}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary."""
        system_status = self.orchestrator.get_system_status()
        
        return {
            "trading": {
                "total_trades": self.trade_count,
                "total_pnl": self.total_pnl,
                "avg_pnl_per_trade": self.total_pnl / max(1, self.trade_count)
            },
            "learning": system_status
        }


__all__ = [
    "AdvancedLearningOrchestrator",
    "IntegratedTradingLoop",
    "LearningConfig",
    "LearningMode",
    "LearningDecision",
    "LearningMetrics"
]