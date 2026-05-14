from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveGuardResult:
    allowed: bool
    reason: str


def check_live_allowed(
    *,
    requested_mode: str,
    node_role: str,
    operator_ack_present: bool,
    soak_ok: bool,
    reconciliation_ok: bool,
    operator_halted: bool,
    operator_frozen: bool,
) -> LiveGuardResult:
    if requested_mode != "live":
        return LiveGuardResult(True, "non-live mode")
    if node_role == "strategy-node":
        return LiveGuardResult(False, "strategy-node cannot run live execution")
    if operator_halted:
        return LiveGuardResult(False, "operator halt is active")
    if operator_frozen:
        return LiveGuardResult(False, "operator freeze is active")
    if not soak_ok:
        return LiveGuardResult(False, "soak evidence failed")
    if not reconciliation_ok:
        return LiveGuardResult(False, "reconciliation guard failed")
    if not operator_ack_present:
        return LiveGuardResult(False, "operator acknowledgement missing")
    return LiveGuardResult(True, "live allowed")
