"""Governance layer — incident engine, execution alpha tuning, bounded auto-actions."""

# Re-export all public names from the split modules
from .types import (
    utc_now_iso,
    clamp,
    Severity,
    IncidentStatus,
    IncidentClass,
    Thresholds,
    AggressionConfig,
    SlicingConfig,
    RoutingConfig,
    AbandonConfig,
    ExecutionAlphaConfig,
    TradeRecord,
    PositionRecord,
    ReplayAuditRecord,
    RuntimeSnapshot,
    Incident,
    ActionSet,
    ExecutionContext,
    ExecutionDecision,
    GovernanceOutcome,
)

from .incidents import (
    INCIDENTS_DDL,
    IncidentRepository,
    IncidentFactory,
    IncidentEngine,
)

from .execution_alpha import ExecutionAlphaTuningPack

from .coordinator import ArgusGovernanceCoordinator, decide_order

__all__ = [
    # Utilities
    "utc_now_iso",
    "clamp",
    # Enums
    "Severity",
    "IncidentStatus",
    "IncidentClass",
    # Configs
    "Thresholds",
    "AggressionConfig",
    "SlicingConfig",
    "RoutingConfig",
    "AbandonConfig",
    "ExecutionAlphaConfig",
    # Records
    "TradeRecord",
    "PositionRecord",
    "ReplayAuditRecord",
    "RuntimeSnapshot",
    "Incident",
    "ActionSet",
    "ExecutionContext",
    "ExecutionDecision",
    "GovernanceOutcome",
    # Persistence
    "INCIDENTS_DDL",
    "IncidentRepository",
    "IncidentFactory",
    "IncidentEngine",
    # Execution
    "ExecutionAlphaTuningPack",
    # Coordination
    "ArgusGovernanceCoordinator",
    "decide_order",
]
