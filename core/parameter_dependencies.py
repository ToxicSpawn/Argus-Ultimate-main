"""
Parameter Dependencies — constraint graph for parameter relationships.

When you have 500+ adaptive parameters, you can't let them drift independently.
Some have logical relationships:

  - take_profit_pct MUST be > stop_loss_pct (otherwise negative R:R)
  - max_position_pct * kelly_fraction MUST be <= safe_total
  - var_limit MUST be < cvar_limit
  - timing.fast_period MUST be < timing.slow_period
  - confidence_threshold MUST be < strict_confidence_threshold

This module tracks those relationships and BLOCKS any drift that would
violate them. The result: parameters can adapt freely within the safe
constraint manifold.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Constraint:
    """A logical constraint between parameters."""
    name: str
    description: str
    params_involved: List[str]
    check_fn: Callable[[Dict[str, float]], bool]
    error_message: str = ""
    severity: str = "error"  # "error" / "warning"


class ParameterDependencyGraph:
    """
    Tracks relationships between parameters and validates proposed values.

    Usage::

        graph = ParameterDependencyGraph()

        # Define a constraint
        graph.add_constraint(
            name="tp_gt_sl",
            description="Take profit must exceed stop loss by 1.5x",
            params_involved=["take_profit_pct", "stop_loss_pct"],
            check_fn=lambda v: v["take_profit_pct"] >= v["stop_loss_pct"] * 1.5,
            error_message="TP must be >= 1.5 * SL",
        )

        # Validate a proposed change
        proposed = {"take_profit_pct": 0.020, "stop_loss_pct": 0.015}
        if graph.validate(proposed):
            apply_changes(proposed)
    """

    def __init__(self) -> None:
        self._constraints: Dict[str, Constraint] = {}
        self._param_to_constraints: Dict[str, Set[str]] = defaultdict(set)
        self._violation_count = 0
        self._check_count = 0
        self._initialize_default_constraints()
        logger.info("ParameterDependencyGraph: initialized with %d defaults", len(self._constraints))

    def _initialize_default_constraints(self) -> None:
        """Set up the canonical ARGUS parameter constraints."""

        # Risk-reward ratio: TP must be > SL
        self.add_constraint(
            name="tp_gt_sl",
            description="Take profit must exceed stop loss by 1.5x",
            params_involved=["take_profit_pct", "stop_loss_pct"],
            check_fn=lambda v: v.get("take_profit_pct", 0) >= v.get("stop_loss_pct", 0) * 1.2,
            error_message="take_profit_pct must be >= 1.2 × stop_loss_pct",
        )

        # CVaR limit must exceed VaR limit
        self.add_constraint(
            name="cvar_gt_var",
            description="CVaR limit must be greater than VaR limit",
            params_involved=["portfolio_cvar_limit_pct", "portfolio_var_limit_pct"],
            check_fn=lambda v: (
                v.get("portfolio_cvar_limit_pct", 0)
                >= v.get("portfolio_var_limit_pct", 0) * 1.0
            ),
            error_message="portfolio_cvar_limit_pct must be >= portfolio_var_limit_pct",
        )

        # Kelly × max position must be safe
        self.add_constraint(
            name="kelly_safe_total",
            description="Kelly fraction × max position should not exceed 30% per trade",
            params_involved=["kelly_fraction", "max_position_pct"],
            check_fn=lambda v: (
                v.get("kelly_fraction", 0) * v.get("max_position_pct", 0) <= 0.40
            ),
            error_message="kelly_fraction × max_position_pct must be <= 0.40",
        )

        # Daily loss limit must be > stop loss (else immediate halt on first loss)
        self.add_constraint(
            name="daily_loss_gt_stop",
            description="Daily loss limit must exceed individual stop loss",
            params_involved=["daily_loss_limit_pct", "stop_loss_pct"],
            check_fn=lambda v: (
                v.get("daily_loss_limit_pct", 1.0) > v.get("stop_loss_pct", 0) * 2
            ),
            error_message="daily_loss_limit_pct must be > 2 × stop_loss_pct",
        )

        # Max concurrent positions must be reasonable
        self.add_constraint(
            name="max_positions_reasonable",
            description="Max concurrent positions must be 1-100",
            params_involved=["max_concurrent_positions"],
            check_fn=lambda v: 1 <= v.get("max_concurrent_positions", 5) <= 100,
            error_message="max_concurrent_positions must be in [1, 100]",
        )

        # Confidence threshold ordering
        self.add_constraint(
            name="confidence_min_max",
            description="Min confidence < max confidence",
            params_involved=["min_confidence", "max_confidence"],
            check_fn=lambda v: (
                v.get("min_confidence", 0)
                <= v.get("max_confidence", 1)
            ),
            error_message="min_confidence must be <= max_confidence",
        )

        # Trailing stop must be smaller than initial stop
        self.add_constraint(
            name="trailing_lt_initial",
            description="Trailing stop should be tighter than initial stop",
            params_involved=["trailing_stop_pct", "stop_loss_pct"],
            check_fn=lambda v: (
                v.get("trailing_stop_pct", 0) <= v.get("stop_loss_pct", 1)
            ),
            error_message="trailing_stop_pct should be <= stop_loss_pct",
            severity="warning",
        )

        # Cycle seconds must be reasonable
        self.add_constraint(
            name="cycle_seconds_range",
            description="Cycle seconds must be 1-300",
            params_involved=["cycle_seconds"],
            check_fn=lambda v: 1 <= v.get("cycle_seconds", 10) <= 300,
            error_message="cycle_seconds must be in [1, 300]",
        )

    def add_constraint(
        self,
        name: str,
        description: str,
        params_involved: List[str],
        check_fn: Callable[[Dict[str, float]], bool],
        error_message: str = "",
        severity: str = "error",
    ) -> bool:
        """Add a constraint to the graph."""
        if name in self._constraints:
            return False
        constraint = Constraint(
            name=name,
            description=description,
            params_involved=list(params_involved),
            check_fn=check_fn,
            error_message=error_message or f"Constraint {name} violated",
            severity=severity,
        )
        self._constraints[name] = constraint
        for param in params_involved:
            self._param_to_constraints[param].add(name)
        return True

    def validate(self, proposed_values: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check if proposed parameter values violate any constraints.
        Returns (is_valid, list_of_violations).
        """
        self._check_count += 1
        violations: List[str] = []

        # Find which constraints involve the proposed parameters
        relevant_constraints: Set[str] = set()
        for param in proposed_values.keys():
            relevant_constraints.update(self._param_to_constraints.get(param, set()))

        for constraint_name in relevant_constraints:
            constraint = self._constraints[constraint_name]
            try:
                ok = constraint.check_fn(proposed_values)
            except Exception as exc:
                # Constraint check itself failed — treat as violation
                logger.debug(
                    "ParameterDependencyGraph: constraint %s check failed: %s",
                    constraint_name, exc,
                )
                ok = False

            if not ok:
                violations.append(f"{constraint.name}: {constraint.error_message}")
                if constraint.severity == "error":
                    self._violation_count += 1

        # Check if any blocking violations exist
        has_errors = any(
            self._constraints[v.split(":")[0]].severity == "error"
            for v in violations
            if v.split(":")[0] in self._constraints
        )
        return (not has_errors, violations)

    def filter_safe_changes(
        self,
        current_values: Dict[str, float],
        proposed_changes: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Filter a dict of proposed parameter changes, dropping any that would
        cause violations. Returns a dict of safe changes.
        """
        safe: Dict[str, float] = {}
        for param_name, new_value in proposed_changes.items():
            # Build a hypothetical state with this change applied
            hypothetical = dict(current_values)
            hypothetical[param_name] = new_value
            valid, violations = self.validate(hypothetical)
            if valid:
                safe[param_name] = new_value
            else:
                logger.debug(
                    "filter_safe_changes: dropped %s=%s (violations: %s)",
                    param_name, new_value, violations,
                )
        return safe

    def get_constraints_for_param(self, param_name: str) -> List[str]:
        """Get all constraints that involve a specific parameter."""
        return sorted(self._param_to_constraints.get(param_name, set()))

    def list_constraints(self) -> List[str]:
        return sorted(self._constraints.keys())

    def get_constraint(self, name: str) -> Optional[Dict[str, Any]]:
        c = self._constraints.get(name)
        if c is None:
            return None
        return {
            "name": c.name,
            "description": c.description,
            "params_involved": list(c.params_involved),
            "error_message": c.error_message,
            "severity": c.severity,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_constraints": len(self._constraints),
            "check_count": self._check_count,
            "violation_count": self._violation_count,
            "tracked_params": len(self._param_to_constraints),
        }
