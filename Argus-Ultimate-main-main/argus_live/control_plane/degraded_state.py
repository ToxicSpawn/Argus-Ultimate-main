from __future__ import annotations

from enum import Enum


class RuntimeHealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED_DATA = "DEGRADED_DATA"
    DEGRADED_VENUE = "DEGRADED_VENUE"
    DEGRADED_RECONCILIATION = "DEGRADED_RECONCILIATION"
    FROZEN = "FROZEN"
    HALTED = "HALTED"
    PROMOTION_BLOCKED = "PROMOTION_BLOCKED"
