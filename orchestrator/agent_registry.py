"""
orchestrator/agent_registry.py
===============================
Registry for all agents in the Meta-Orchestrator.

Manages:
  - Agent registration and discovery
  - Agent lifecycle (start, stop, pause)
  - Agent health monitoring
  - Agent weight management
  - Agent performance tracking
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentCategory(Enum):
    """Agent categories."""
    PERCEPTION      = "perception"      # Market observation
    REASONING       = "reasoning"       # Hypothesis formation
    ACTING          = "acting"          # Decision execution
    LEARNING        = "learning"        # Continuous improvement
    MONITORING      = "monitoring"      # Self-monitoring


class AgentStatus(Enum):
    """Agent lifecycle status."""
    REGISTERED      = "registered"
    STARTING        = "starting"
    ACTIVE          = "active"
    PAUSED          = "paused"
    DEGRADED        = "degraded"
    STOPPING        = "stopping"
    STOPPED         = "stopped"
    ERROR           = "error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name            : str
    category        : AgentCategory
    module_path     : str
    update_interval : float = 1.0  # seconds between updates
    min_confidence  : float = 0.1
    max_confidence  : float = 1.0
    priority        : int = 5  # 1-10
    tags            : List[str] = field(default_factory=list)


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    decisions_made  : int = 0
    successes       : int = 0
    failures        : int = 0
    avg_confidence  : float = 0.0
    avg_latency_ms  : float = 0.0
    last_active     : float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0


@dataclass
class RegisteredAgent:
    """A registered agent in the system."""
    config          : AgentConfig
    status          : AgentStatus = AgentStatus.REGISTERED
    metrics         : AgentMetrics = field(default_factory=AgentMetrics)
    weight          : float = 1.0
    health          : float = 1.0  # 0-1
    last_health_check: float = 0.0
    error_count     : int = 0
    last_error      : Optional[str] = None

    # Callbacks
    observe_fn      : Optional[Callable] = None
    reason_fn       : Optional[Callable] = None
    act_fn          : Optional[Callable] = None
    learn_fn        : Optional[Callable] = None
    monitor_fn      : Optional[Callable] = None


# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """
    Central registry for all agents in the Meta-Orchestrator.

    Handles registration, lifecycle, health monitoring, and weight management.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._agents: Dict[str, RegisteredAgent] = {}
        self._category_index: Dict[AgentCategory, List[str]] = {
            cat: [] for cat in AgentCategory
        }
        self._health_check_interval: float = 30.0
        self._last_health_check: float = 0.0

        logger.info("AgentRegistry: initialised")

    # ------------------------------------------------------------------ Registration

    def register(
        self,
        config: AgentConfig,
        observe_fn: Optional[Callable] = None,
        reason_fn: Optional[Callable] = None,
        act_fn: Optional[Callable] = None,
        learn_fn: Optional[Callable] = None,
        monitor_fn: Optional[Callable] = None,
    ) -> bool:
        """
        Register a new agent.

        Returns True if registered successfully.
        """
        with self._lock:
            if config.name in self._agents:
                logger.warning("AgentRegistry: agent '%s' already registered", config.name)
                return False

            agent = RegisteredAgent(
                config=config,
                observe_fn=observe_fn,
                reason_fn=reason_fn,
                act_fn=act_fn,
                learn_fn=learn_fn,
                monitor_fn=monitor_fn,
            )

            self._agents[config.name] = agent
            self._category_index[config.category].append(config.name)

            logger.info(
                "AgentRegistry: registered '%s' [%s] priority=%d",
                config.name, config.category.value, config.priority,
            )
            return True

    def unregister(self, name: str) -> bool:
        """Unregister an agent."""
        with self._lock:
            agent = self._agents.pop(name, None)
            if agent is None:
                return False

            cat_list = self._category_index.get(agent.config.category, [])
            if name in cat_list:
                cat_list.remove(name)

            logger.info("AgentRegistry: unregistered '%s'", name)
            return True

    # ------------------------------------------------------------------ Lifecycle

    def start(self, name: str) -> bool:
        """Start an agent."""
        with self._lock:
            agent = self._agents.get(name)
            if agent is None:
                return False
            agent.status = AgentStatus.ACTIVE
            agent.metrics.last_active = time.time()
            logger.info("AgentRegistry: started '%s'", name)
            return True

    def stop(self, name: str) -> bool:
        """Stop an agent."""
        with self._lock:
            agent = self._agents.get(name)
            if agent is None:
                return False
            agent.status = AgentStatus.STOPPED
            logger.info("AgentRegistry: stopped '%s'", name)
            return True

    def pause(self, name: str) -> bool:
        """Pause an agent."""
        with self._lock:
            agent = self._agents.get(name)
            if agent is None:
                return False
            agent.status = AgentStatus.PAUSED
            logger.info("AgentRegistry: paused '%s'", name)
            return True

    def resume(self, name: str) -> bool:
        """Resume a paused agent."""
        with self._lock:
            agent = self._agents.get(name)
            if agent is None:
                return False
            if agent.status == AgentStatus.PAUSED:
                agent.status = AgentStatus.ACTIVE
                logger.info("AgentRegistry: resumed '%s'", name)
                return True
            return False

    def start_all(self) -> int:
        """Start all registered agents."""
        count = 0
        with self._lock:
            for name, agent in self._agents.items():
                if agent.status == AgentStatus.REGISTERED:
                    agent.status = AgentStatus.ACTIVE
                    agent.metrics.last_active = time.time()
                    count += 1
        logger.info("AgentRegistry: started %d agents", count)
        return count

    def stop_all(self) -> int:
        """Stop all agents."""
        count = 0
        with self._lock:
            for agent in self._agents.values():
                if agent.status == AgentStatus.ACTIVE:
                    agent.status = AgentStatus.STOPPED
                    count += 1
        logger.info("AgentRegistry: stopped %d agents", count)
        return count

    # ------------------------------------------------------------------ Queries

    def get(self, name: str) -> Optional[RegisteredAgent]:
        """Get a registered agent by name."""
        with self._lock:
            return self._agents.get(name)

    def get_by_category(self, category: AgentCategory) -> List[RegisteredAgent]:
        """Get all agents in a category."""
        with self._lock:
            names = self._category_index.get(category, [])
            return [self._agents[n] for n in names if n in self._agents]

    def get_active_by_category(self, category: AgentCategory) -> List[RegisteredAgent]:
        """Get active agents in a category, sorted by weight."""
        agents = self.get_by_category(category)
        return [
            a for a in agents
            if a.status == AgentStatus.ACTIVE
        ]

    def get_all_active(self) -> List[RegisteredAgent]:
        """Get all active agents."""
        with self._lock:
            return [
                a for a in self._agents.values()
                if a.status == AgentStatus.ACTIVE
            ]

    def get_healthy_active(self, category: Optional[AgentCategory] = None) -> List[RegisteredAgent]:
        """Get healthy, active agents, optionally filtered by category."""
        with self._lock:
            agents = self._agents.values()
            if category:
                agents = [a for a in agents if a.config.category == category]
            return [
                a for a in agents
                if a.status == AgentStatus.ACTIVE and a.health > 0.5
            ]

    # ------------------------------------------------------------------ Weight management

    def set_weight(self, name: str, weight: float) -> None:
        """Set agent weight (for meta-learner)."""
        with self._lock:
            agent = self._agents.get(name)
            if agent:
                agent.weight = max(0.0, min(1.0, weight))

    def update_weights_from_performance(self) -> None:
        """Update weights based on recent performance."""
        with self._lock:
            for agent in self._agents.values():
                if agent.metrics.decisions_made > 0:
                    # Weight = success_rate * health
                    new_weight = agent.metrics.success_rate * agent.health
                    agent.weight = 0.9 * agent.weight + 0.1 * new_weight

    # ------------------------------------------------------------------ Metrics

    def record_decision(self, name: str, success: bool, confidence: float, latency_ms: float) -> None:
        """Record a decision outcome for an agent."""
        with self._lock:
            agent = self._agents.get(name)
            if agent is None:
                return

            agent.metrics.decisions_made += 1
            if success:
                agent.metrics.successes += 1
            else:
                agent.metrics.failures += 1

            # Update running averages
            n = agent.metrics.decisions_made
            agent.metrics.avg_confidence = (
                (agent.metrics.avg_confidence * (n - 1) + confidence) / n
            )
            agent.metrics.avg_latency_ms = (
                (agent.metrics.avg_latency_ms * (n - 1) + latency_ms) / n
            )
            agent.metrics.last_active = time.time()

    def record_error(self, name: str, error: str) -> None:
        """Record an error for an agent."""
        with self._lock:
            agent = self._agents.get(name)
            if agent is None:
                return

            agent.error_count += 1
            agent.last_error = error

            # Degrade if too many errors
            if agent.error_count > 10:
                agent.status = AgentStatus.DEGRADED
                agent.health = max(0.0, agent.health - 0.2)
                logger.warning("AgentRegistry: agent '%s' degraded (errors=%d)", name, agent.error_count)

    # ------------------------------------------------------------------ Health checks

    def run_health_checks(self) -> Dict[str, float]:
        """Run health checks on all active agents."""
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return {}

        results: Dict[str, float] = {}

        with self._lock:
            for name, agent in self._agents.items():
                if agent.status not in (AgentStatus.ACTIVE, AgentStatus.DEGRADED):
                    continue

                # Check if agent has been active recently
                if agent.metrics.last_active > 0:
                    inactive_seconds = now - agent.metrics.last_active
                    if inactive_seconds > 300:  # 5 minutes inactive
                        agent.health = max(0.0, agent.health - 0.1)

                # Check error rate
                if agent.metrics.decisions_made > 0:
                    error_rate = agent.metrics.failures / agent.metrics.decisions_made
                    if error_rate > 0.5:
                        agent.health = max(0.0, agent.health - 0.1)

                results[name] = agent.health
                agent.last_health_check = now

        self._last_health_check = now
        return results

    # ------------------------------------------------------------------ Status

    def get_status(self) -> Dict[str, Any]:
        """Get registry status summary."""
        with self._lock:
            status_counts = {}
            for agent in self._agents.values():
                s = agent.status.value
                status_counts[s] = status_counts.get(s, 0) + 1

            category_counts = {}
            for cat, names in self._category_index.items():
                category_counts[cat.value] = len(names)

            return {
                "total_agents"   : len(self._agents),
                "by_status"      : status_counts,
                "by_category"    : category_counts,
                "avg_health"     : sum(a.health for a in self._agents.values()) / max(1, len(self._agents)),
                "avg_weight"     : sum(a.weight for a in self._agents.values()) / max(1, len(self._agents)),
            }

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents with their status."""
        with self._lock:
            return [
                {
                    "name"          : a.config.name,
                    "category"      : a.config.category.value,
                    "status"        : a.status.value,
                    "health"        : a.health,
                    "weight"        : a.weight,
                    "decisions"     : a.metrics.decisions_made,
                    "success_rate"  : a.metrics.success_rate,
                    "avg_confidence": a.metrics.avg_confidence,
                    "error_count"   : a.error_count,
                }
                for a in self._agents.values()
            ]
