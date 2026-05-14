"""Governance coordinator — orchestration glue and decide_order entry point."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from .types import (
    ExecutionAlphaConfig,
    ExecutionContext,
    GovernanceOutcome,
    RuntimeSnapshot,
    Thresholds,
)
from .incidents import IncidentEngine, IncidentRepository
from .execution_alpha import ExecutionAlphaTuningPack


class ArgusGovernanceCoordinator:
    def __init__(
        self,
        db_path: str,
        thresholds: Optional[Thresholds] = None,
        exec_cfg: Optional[ExecutionAlphaConfig] = None,
    ) -> None:
        self.repo = IncidentRepository(db_path)
        self.repo.init_schema()
        self.incident_engine = IncidentEngine(thresholds or Thresholds())
        self.base_tuning_pack = ExecutionAlphaTuningPack(exec_cfg or ExecutionAlphaConfig())

    def evaluate_snapshot(self, snapshot: RuntimeSnapshot) -> GovernanceOutcome:
        incidents, actions = self.incident_engine.evaluate(snapshot)
        self.repo.insert_many(incidents)
        tuned_pack = self.base_tuning_pack.apply_overrides(actions.execution_overrides)
        return GovernanceOutcome(
            incidents=incidents,
            actions=actions,
            execution_tuning_pack=tuned_pack,
        )


def decide_order(
    coordinator: ArgusGovernanceCoordinator,
    snapshot: RuntimeSnapshot,
    exec_ctx: ExecutionContext,
) -> Dict[str, Any]:
    """
    Example proving-phase flow:
    1. Evaluate incidents and bounded auto-actions.
    2. If critical stop is active, refuse order.
    3. Run execution tuning pack to decide maker/taker/cancel/slice.
    4. Return decision payload for the execution engine.
    """
    outcome = coordinator.evaluate_snapshot(snapshot)

    if outcome.actions.stop_trading:
        return {
            "allowed": False,
            "reason": "critical incident stop_trading asserted",
            "incidents": [asdict(i) for i in outcome.incidents],
        }

    decision = outcome.execution_tuning_pack.decide(exec_ctx)

    return {
        "allowed": not decision.cancel,
        "decision": asdict(decision),
        "actions": asdict(outcome.actions),
        "incidents": [asdict(i) for i in outcome.incidents],
    }
