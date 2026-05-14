from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["PROPOSED", "RISK_REJECTED", "TARGET_APPROVED", "ROUTING_SELECTED", "SUBMITTED", "VENUE_ACKED", "PARTIAL_FILL", "FILLED", "CANCELLED", "REJECTED", "RECON_PENDING", "RECONCILED", "ATTRIBUTED"]
_ALLOWED: dict[Status, set[Status]] = {
    "PROPOSED": {"RISK_REJECTED", "TARGET_APPROVED"},
    "RISK_REJECTED": set(),
    "TARGET_APPROVED": {"ROUTING_SELECTED", "RISK_REJECTED"},
    "ROUTING_SELECTED": {"SUBMITTED", "REJECTED"},
    "SUBMITTED": {"VENUE_ACKED", "REJECTED", "CANCELLED"},
    "VENUE_ACKED": {"PARTIAL_FILL", "FILLED", "CANCELLED"},
    "PARTIAL_FILL": {"FILLED", "CANCELLED"},
    "FILLED": {"RECON_PENDING"},
    "CANCELLED": set(),
    "REJECTED": set(),
    "RECON_PENDING": {"RECONCILED"},
    "RECONCILED": {"ATTRIBUTED"},
    "ATTRIBUTED": set(),
}


@dataclass(frozen=True)
class TransitionResult:
    ok: bool
    reason: str


def validate_transition(current: Status, nxt: Status) -> TransitionResult:
    if nxt not in _ALLOWED[current]:
        return TransitionResult(False, f"illegal transition {current} -> {nxt}")
    return TransitionResult(True, "ok")
