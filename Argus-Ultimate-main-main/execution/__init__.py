"""Execution package — lazy imports to avoid circular dependency."""

from .algo_orders import SlicePlan, build_twap_plan, build_vwap_style_plan
from .contingency_orders import ContingencyExecutor, OTOGroup, OTOStatus
from .conditional_orders import (
    ConditionalOrderManager,
    ConditionalGroup,
    GroupType,
    GroupStatus,
    LegType,
    LegStatus,
    OrderLeg,
)
from .order_types_advanced import AdvancedOrderSpec, TimeInForce, normalize_to_venue

__all__ = [
    # algo orders
    "SlicePlan", "build_twap_plan", "build_vwap_style_plan",
    # contingency execution layer
    "ContingencyExecutor", "OTOGroup", "OTOStatus",
    # conditional order manager
    "ConditionalOrderManager", "ConditionalGroup",
    "GroupType", "GroupStatus", "LegType", "LegStatus", "OrderLeg",
    # advanced order types
    "AdvancedOrderSpec", "TimeInForce", "normalize_to_venue",
    # legacy
    "DeltaNeutralExecutor", "HedgeSuggestion",
    "SmartExecutionCore", "SmartExecutionInput",
    "SmartOrderExecution", "SmartOrderExecutionRequest",
]
