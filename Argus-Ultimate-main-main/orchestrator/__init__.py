"""
orchestrator/__init__.py
=========================
Meta-Orchestrator — the unified brain of Argus.

Coordinates all trading modules through a blackboard architecture,
enabling emergent behavior, self-healing, and multi-timescale adaptation.
"""

from orchestrator.meta_orchestrator import MetaOrchestrator, get_orchestrator
from orchestrator.world_model import WorldModel, RegimeState
from orchestrator.decision_bus import DecisionBus, Observation, Hypothesis, Decision
from orchestrator.agent_registry import AgentRegistry, AgentCategory, AgentStatus
from orchestrator.temporal_hierarchy import TemporalHierarchy, TimescaleLayer

__all__ = [
    "MetaOrchestrator",
    "get_orchestrator",
    "WorldModel",
    "RegimeState",
    "DecisionBus",
    "Observation",
    "Hypothesis",
    "Decision",
    "AgentRegistry",
    "AgentCategory",
    "AgentStatus",
    "TemporalHierarchy",
    "TimescaleLayer",
]
