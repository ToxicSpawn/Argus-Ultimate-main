from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SoakGateResult:
    ok: bool
    reason: str


def check_soak_gate(report_path: str | Path, required_profile: str) -> SoakGateResult:
    path = Path(report_path)
    if not path.exists():
        return SoakGateResult(False, "soak report missing")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return SoakGateResult(False, f"invalid soak report: {exc}")

    if data.get("profile") != required_profile:
        return SoakGateResult(False, "soak profile mismatch")
    if not bool(data.get("passed", False)):
        return SoakGateResult(False, "soak thresholds failed")
    return SoakGateResult(True, "soak passed")
