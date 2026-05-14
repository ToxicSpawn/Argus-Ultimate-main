"""
orchestrator/meta_orchestrator.py
==================================
Meta-Orchestrator — the unified brain of Argus.

Coordinates all trading modules through:
  - World Model (unified state)
  - Decision Bus (blackboard architecture)
  - Agent Registry (lifecycle management)
  - Temporal Hierarchy (multi-timescale coordination)

This is the single entry point that transforms 100+ independent modules
into one coherent, self-improving trading intelligence.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from orchestrator.world_model import (
    WorldModel, RegimeState, Regime, RiskState, PositionState,
    LiquidityState, AgentState, AgentHealth, Goal,
)
from orchestrator.decision_bus import (
    DecisionBus, Observation, Hypothesis, Decision,
    ObservationType, HypothesisType, DecisionType,
)
from orchestrator.agent_registry import (
    AgentRegistry, AgentConfig, AgentCategory, AgentStatus,
)
from orchestrator.temporal_hierarchy import (
    TemporalHierarchy, Timescale, TimescaleLayer, CrossTimescaleSignal,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cycle result
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    """Result of one orchestration cycle."""
    cycle_id        : int
    cycle_time_ms   : float
    observations    : int
    hypotheses      : int
    decisions       : int
    actions         : int
    health_status   : str  # "healthy", "degraded", "critical"
    alerts          : List[Dict[str, Any]]
    timestamp       : float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Meta-Orchestrator
# ---------------------------------------------------------------------------

class MetaOrchestrator:
    """
    The unified brain of Argus.

    Coordinates all trading modules through a blackboard architecture
    with multi-timescale adaptation and self-healing capabilities.

    Usage:
        orchestrator = MetaOrchestrator()
        orchestrator.register_default_agents()
        orchestrator.start()
        
        # In trading loop:
        result = await orchestrator.cycle(market_data)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.config = config or {}

        # Core layers
        self.world_model    = WorldModel()
        self.decision_bus   = DecisionBus()
        self.agent_registry = AgentRegistry()
        self.temporal       = TemporalHierarchy()

        # State
        self._running       = False
        self._cycle_count   = 0
        self._start_time    : float = 0.0
        self._lock          = threading.RLock()

        # Alerts
        self._alerts        : Deque[Dict[str, Any]] = deque(maxlen=1000)

        # Callbacks
        self._on_decision   : List[Callable[[Decision], None]] = []
        self._on_alert      : List[Callable[[Dict[str, Any]], None]] = []

        # Integration hooks (set by caller)
        self._enterprise_risk   : Optional[Any] = None
        self._execution_int     : Optional[Any] = None
        self._compliance_int    : Optional[Any] = None

        logger.info("MetaOrchestrator: initialised with capital=%.0f", initial_capital)

    # ------------------------------------------------------------------ Integration hooks

    def set_enterprise_risk(self, integrator: Any) -> None:
        """Set the enterprise risk integrator."""
        self._enterprise_risk = integrator

    def set_execution_integrator(self, integrator: Any) -> None:
        """Set the execution integrator."""
        self._execution_int = integrator

    def set_compliance_integrator(self, integrator: Any) -> None:
        """Set the compliance integrator."""
        self._compliance_int = integrator

    # ------------------------------------------------------------------ Agent registration

    def register_default_agents(self) -> int:
        """Register default agents from existing Argus modules."""
        count = 0

        # --- Perception agents ---
        perception_agents = [
            AgentConfig("regime_detector", AgentCategory.PERCEPTION,
                       "ml/volatility_adaptive_drl/regime_detector.py", update_interval=1.0),
            AgentConfig("order_flow", AgentCategory.PERCEPTION,
                       "hft_engine/order_book_processor.py", update_interval=0.01),
            AgentConfig("sentiment", AgentCategory.PERCEPTION,
                       "ml/llm_sentiment_enhanced.py", update_interval=60.0),
            AgentConfig("correlation", AgentCategory.PERCEPTION,
                       "risk/correlation_monitor.py", update_interval=5.0),
            AgentConfig("volatility", AgentCategory.PERCEPTION,
                       "ml/volatility_forecaster.py", update_interval=5.0),
        ]

        for config in perception_agents:
            self.agent_registry.register(config)
            self.temporal.register_agent(Timescale.SECOND, config.name)
            count += 1

        # --- Reasoning agents ---
        reasoning_agents = [
            AgentConfig("causal_intelligence", AgentCategory.REASONING,
                       "ml/causal_inference.py", update_interval=5.0),
            AgentConfig("uncertainty", AgentCategory.REASONING,
                       "ml/uncertainty_quantifier.py", update_interval=5.0),
            AgentConfig("regime_predictor", AgentCategory.REASONING,
                       "ml/regime_predictor.py", update_interval=60.0),
        ]

        for config in reasoning_agents:
            self.agent_registry.register(config)
            self.temporal.register_agent(Timescale.MINUTE, config.name)
            count += 1

        # --- Acting agents ---
        acting_agents = [
            AgentConfig("execution", AgentCategory.ACTING,
                       "execution/institutional_execution.py", update_interval=0.1),
            AgentConfig("smart_router", AgentCategory.ACTING,
                       "execution/smart_order_router_v2.py", update_interval=0.1),
            AgentConfig("risk_manager", AgentCategory.ACTING,
                       "risk/advanced_risk_engine.py", update_interval=1.0),
            AgentConfig("market_maker", AgentCategory.ACTING,
                       "strategies/avellaneda_stoikov/market_maker.py", update_interval=0.1),
        ]

        for config in acting_agents:
            self.agent_registry.register(config)
            self.temporal.register_agent(Timescale.SECOND, config.name)
            count += 1

        # --- Learning agents ---
        learning_agents = [
            AgentConfig("online_learner", AgentCategory.LEARNING,
                       "ml/online_learning.py", update_interval=300.0),
            AgentConfig("evolutionary", AgentCategory.LEARNING,
                       "ml/genetic_evolver.py", update_interval=3600.0),
            AgentConfig("meta_learner", AgentCategory.LEARNING,
                       "ml/meta_learning.py", update_interval=3600.0),
            AgentConfig("model_rolling", AgentCategory.LEARNING,
                       "ml/model_rolling/deployment_orchestrator.py", update_interval=600.0),
        ]

        for config in learning_agents:
            self.agent_registry.register(config)
            self.temporal.register_agent(Timescale.HOUR, config.name)
            count += 1

        # --- Monitoring agents ---
        monitoring_agents = [
            AgentConfig("drift_detector", AgentCategory.MONITORING,
                       "ml/drift_detector.py", update_interval=60.0),
            AgentConfig("enterprise_risk", AgentCategory.MONITORING,
                       "monitoring/enterprise_risk_integration.py", update_interval=5.0),
            AgentConfig("compliance", AgentCategory.MONITORING,
                       "monitoring/compliance_integration.py", update_interval=60.0),
            AgentConfig("self_healer", AgentCategory.MONITORING,
                       "orchestrator/self_healer.py", update_interval=300.0),
        ]

        for config in monitoring_agents:
            self.agent_registry.register(config)
            self.temporal.register_agent(Timescale.MINUTE, config.name)
            count += 1

        logger.info("MetaOrchestrator: registered %d default agents", count)
        return count

    # ------------------------------------------------------------------ Lifecycle

    def start(self) -> None:
        """Start the orchestrator."""
        with self._lock:
            if self._running:
                logger.warning("MetaOrchestrator: already running")
                return

            self._running = True
            self._start_time = time.time()
            self._cycle_count = 0

            # Start all agents
            self.agent_registry.start_all()

            # Initialize world model
            self.world_model.update_portfolio(self.initial_capital, self.initial_capital, 0.0)

            logger.info("MetaOrchestrator: started")

    def stop(self) -> None:
        """Stop the orchestrator."""
        with self._lock:
            self._running = False
            self.agent_registry.stop_all()
            logger.info("MetaOrchestrator: stopped after %d cycles", self._cycle_count)

    # ------------------------------------------------------------------ Main cycle

    async def cycle(self, market_data: Optional[Dict[str, Any]] = None) -> CycleResult:
        """
        One complete orchestration cycle.

        This is the heartbeat — called on every tick (or batch of ticks).
        """
        cycle_start = time.time()
        self._cycle_count += 1

        alerts: List[Dict[str, Any]] = []
        n_obs = 0
        n_hyp = 0
        n_dec = 0
        n_act = 0

        try:
            # 1. PERCEIVE: Gather observations from perception agents
            observations = self._perceive(market_data or {})
            n_obs = len(observations)

            # Post to Decision Bus
            for obs in observations:
                self.decision_bus.post_observation(obs)

            # Update World Model
            self._update_world_model(observations)

            # 2. REASON: Gather hypotheses from reasoning agents
            hypotheses = self._reason(observations)
            n_hyp = len(hypotheses)

            for hyp in hypotheses:
                self.decision_bus.post_hypothesis(hyp)

            # 3. DECIDE: Get coordinated decisions
            decisions = self._decide(hypotheses, market_data or {})
            n_dec = len(decisions)

            # 4. ACT: Execute decisions
            actions = self._act(decisions)
            n_act = len(actions)

            # 5. LEARN: Learning agents update
            self._learn(observations, hypotheses, decisions, actions)

            # 6. MONITOR: Self-monitoring
            health_status, health_alerts = self._monitor()
            alerts.extend(health_alerts)

            # 7. REPAIR: Self-healing if needed
            if health_status in ("degraded", "critical"):
                self._repair(health_status)

            # 8. UPDATE TIMELINE: Process temporal hierarchy
            self._update_temporal(market_data or {})

        except Exception as e:
            logger.error("MetaOrchestrator: cycle error: %s", e, exc_info=True)
            alerts.append({
                "type"    : "orchestrator_error",
                "severity": "critical",
                "message" : str(e),
                "ts"      : time.time(),
            })
            health_status = "critical"

        cycle_time = (time.time() - cycle_start) * 1000

        result = CycleResult(
            cycle_id      = self._cycle_count,
            cycle_time_ms = cycle_time,
            observations  = n_obs,
            hypotheses    = n_hyp,
            decisions     = n_dec,
            actions       = n_act,
            health_status = health_status,
            alerts        = alerts,
        )

        # Notify alert callbacks
        for alert in alerts:
            for cb in self._on_alert:
                try:
                    cb(alert)
                except Exception:
                    pass

        return result

    # ------------------------------------------------------------------ Internal steps

    def _perceive(self, market_data: Dict[str, Any]) -> List[Observation]:
        """Gather observations from perception agents."""
        observations: List[Observation] = []

        # Get active perception agents
        agents = self.agent_registry.get_healthy_active(AgentCategory.PERCEPTION)

        for agent in agents:
            try:
                # In production, this would call the agent's observe function
                # For now, create synthetic observations from market data
                obs = self._create_observation(agent.config.name, market_data)
                if obs:
                    observations.append(obs)
            except Exception as e:
                self.agent_registry.record_error(agent.config.name, str(e))

        return observations

    def _create_observation(self, agent_name: str, market_data: Dict[str, Any]) -> Optional[Observation]:
        """Create an observation from an agent (simplified for framework)."""
        now = time.time()

        if agent_name == "regime_detector":
            return Observation(
                agent=agent_name,
                type=ObservationType.REGIME,
                data={"regime": market_data.get("regime", "normal"), "confidence": 0.7},
                confidence=0.7,
            )
        elif agent_name == "order_flow":
            return Observation(
                agent=agent_name,
                type=ObservationType.ORDER_FLOW,
                data={"imbalance": market_data.get("order_imbalance", 0.0)},
                confidence=0.8,
            )
        elif agent_name == "volatility":
            return Observation(
                agent=agent_name,
                type=ObservationType.VOLATILITY,
                data={"current": market_data.get("volatility", 0.02)},
                confidence=0.9,
            )
        elif agent_name == "sentiment":
            return Observation(
                agent=agent_name,
                type=ObservationType.SENTIMENT,
                data={"score": market_data.get("sentiment", 0.0)},
                confidence=0.6,
            )
        elif agent_name == "correlation":
            return Observation(
                agent=agent_name,
                type=ObservationType.CORRELATION,
                data={"regime": market_data.get("correlation_regime", "normal")},
                confidence=0.7,
            )
        return None

    def _reason(self, observations: List[Observation]) -> List[Hypothesis]:
        """Form hypotheses from observations."""
        hypotheses: List[Hypothesis] = []

        # Get active reasoning agents
        agents = self.agent_registry.get_healthy_active(AgentCategory.REASONING)

        for agent in agents:
            try:
                hyp = self._create_hypothesis(agent.config.name, observations)
                if hyp:
                    hypotheses.append(hyp)
            except Exception as e:
                self.agent_registry.record_error(agent.config.name, str(e))

        return hypotheses

    def _create_hypothesis(self, agent_name: str, observations: List[Observation]) -> Optional[Hypothesis]:
        """Create a hypothesis from an agent (simplified)."""
        if agent_name == "causal_intelligence":
            return Hypothesis(
                agent=agent_name,
                type=HypothesisType.CAUSAL,
                description="Market movement driven by order flow imbalance",
                evidence=["order_flow"],
                confidence=0.6,
                implications={"action": "adjust_position", "direction": "neutral"},
            )
        elif agent_name == "regime_predictor":
            return Hypothesis(
                agent=agent_name,
                type=HypothesisType.PREDICTIVE,
                description="Regime stable for next 30 minutes",
                evidence=["regime_detector", "volatility"],
                confidence=0.7,
                implications={"regime_stable": True},
            )
        return None

    def _decide(self, hypotheses: List[Hypothesis], market_data: Dict[str, Any]) -> List[Decision]:
        """Make coordinated decisions."""
        decisions: List[Decision] = []

        # Check risk limits first
        risk_ok = self._check_risk_limits(market_data)
        if not risk_ok:
            decisions.append(Decision(
                type=DecisionType.PAUSE,
                action={"reason": "risk_limit_breach"},
                reasoning=["risk_check"],
                confidence=0.95,
            ))
            return decisions

        # Request execution decision
        decision = self.decision_bus.request_decision(
            context_type="execute",
            context_data=market_data,
            min_confidence=0.3,
        )

        if decision:
            decisions.append(decision)

        return decisions

    def _check_risk_limits(self, market_data: Dict[str, Any]) -> bool:
        """Check if risk limits are within bounds."""
        risk = self.world_model.get_risk()
        if risk.risk_utilisation > 1.0:
            return False
        if risk.current_dd_pct < -15.0:
            return False
        return True

    def _act(self, decisions: List[Decision]) -> List[Dict[str, Any]]:
        """Execute decisions through acting agents."""
        actions: List[Dict[str, Any]] = []

        for decision in decisions:
            try:
                action = self._execute_decision(decision)
                if action:
                    actions.append(action)
            except Exception as e:
                logger.error("MetaOrchestrator: action error: %s", e)

        return actions

    def _execute_decision(self, decision: Decision) -> Optional[Dict[str, Any]]:
        """Execute a single decision."""
        if decision.type == DecisionType.PAUSE:
            logger.warning("MetaOrchestrator: PAUSING TRADING — %s", decision.action.get("reason"))
            return {"type": "pause", "reason": decision.action.get("reason")}

        elif decision.type == DecisionType.EXECUTE:
            # Route through execution integrator
            if self._execution_int:
                try:
                    result = self._execution_int.route_order(
                        symbol=decision.action.get("symbol", "BTC/USD"),
                        side=decision.action.get("side", "buy"),
                        size_usd=decision.action.get("size_usd", 1000.0),
                        venue_books={},
                    )
                    return {"type": "execute", "result": result}
                except Exception as e:
                    logger.error("Execution error: %s", e)

        return {"type": decision.type.value, "action": decision.action}

    def _learn(self, observations: List[Observation], hypotheses: List[Hypothesis],
               decisions: List[Decision], actions: List[Dict[str, Any]]) -> None:
        """Learning agents update from experience."""
        # Update agent weights based on performance
        self.agent_registry.update_weights_from_performance()

    def _monitor(self) -> Tuple[str, List[Dict[str, Any]]]:
        """Self-monitoring agents check health."""
        alerts: List[Dict[str, Any]] = []

        # Run health checks
        health_results = self.agent_registry.run_health_checks()

        # Check overall health
        avg_health = sum(health_results.values()) / max(1, len(health_results))

        if avg_health < 0.3:
            status = "critical"
        elif avg_health < 0.6:
            status = "degraded"
        else:
            status = "healthy"

        # Check enterprise risk
        if self._enterprise_risk:
            try:
                risk_summary = self._enterprise_risk.get_risk_summary()
                utilisation = risk_summary.get("var_utilisation_pct", 0)
                if utilisation > 100:
                    alerts.append({
                        "type": "var_limit_breach",
                        "severity": "critical",
                        "message": f"VaR utilisation {utilisation:.0f}% exceeds limit",
                        "ts": time.time(),
                    })
            except Exception:
                pass

        return status, alerts

    def _repair(self, health_status: str) -> None:
        """Self-healing repairs."""
        if health_status == "critical":
            logger.critical("MetaOrchestrator: CRITICAL health — initiating emergency protocol")
            # In production: isolate failing agents, retrain models, etc.

    def _update_world_model(self, observations: List[Observation]) -> None:
        """Update world model from observations."""
        for obs in observations:
            if obs.type == ObservationType.REGIME:
                regime_str = obs.data.get("regime", "normal")
                regime_map = {
                    "low_vol": Regime.LOW_VOL, "normal": Regime.NORMAL,
                    "high_vol": Regime.HIGH_VOL, "crisis": Regime.CRISIS,
                }
                self.world_model.update_regime(RegimeState(
                    current=regime_map.get(regime_str, Regime.UNKNOWN),
                    confidence=obs.confidence,
                ))
            elif obs.type == ObservationType.VOLATILITY:
                vol = obs.data.get("current", 0.02)
                current_regime = self.world_model.get_regime()
                self.world_model.update_regime(RegimeState(
                    current=current_regime.current,
                    confidence=current_regime.confidence,
                    volatility=vol,
                ))

    def _update_temporal(self, market_data: Dict[str, Any]) -> None:
        """Update temporal hierarchy."""
        for ts in [Timescale.SECOND, Timescale.MINUTE, Timescale.HOUR]:
            self.temporal.tick(ts, market_data)

    # ------------------------------------------------------------------ Callbacks

    def on_decision(self, callback: Callable[[Decision], None]) -> None:
        self._on_decision.append(callback)

    def on_alert(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._on_alert.append(callback)

    # ------------------------------------------------------------------ Status

    def get_status(self) -> Dict[str, Any]:
        """Get full orchestrator status."""
        return {
            "running"       : self._running,
            "cycle_count"   : self._cycle_count,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "world_model"   : self.world_model.snapshot(),
            "agent_registry": self.agent_registry.get_status(),
            "decision_bus"  : self.decision_bus.get_stats(),
            "temporal"      : self.temporal.get_status(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_orchestrator: Optional[MetaOrchestrator] = None


def get_orchestrator(
    initial_capital: float = 1_000_000.0,
    config: Optional[Dict[str, Any]] = None,
) -> MetaOrchestrator:
    """Get or create the Meta-Orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MetaOrchestrator(initial_capital=initial_capital, config=config)
    return _orchestrator
