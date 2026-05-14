from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ReconciliationGuardResult:
    ok: bool
    reason: str


def check_reconciliation_guard(
    state_path: str | Path,
    operator_ack_present: bool,
) -> ReconciliationGuardResult:
    path = Path(state_path)
    if not path.exists():
        return ReconciliationGuardResult(True, "no reconciliation freeze state")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ReconciliationGuardResult(False, f"invalid reconciliation state: {exc}")

    freeze_active = bool(data.get("freeze_active", False))
    if freeze_active and not operator_ack_present:
        return ReconciliationGuardResult(False, "reconciliation freeze active and not acknowledged")
    return ReconciliationGuardResult(True, "reconciliation ok")
