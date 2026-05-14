from __future__ import annotations

from dataclasses import asdict
from typing import Any

from argus_live.constitution.models import Constitution, Limits, SafetyRules, VenueRules


def build_constitution(cfg: dict[str, Any]) -> Constitution:
    c = cfg["constitution"]
    return Constitution(
        version=c["version"],
        profile=c["profile"],
        limits=Limits(**c["limits"]),
        safety=SafetyRules(**c["safety"]),
        venues=VenueRules(**c["venues"]),
    )


def constitution_to_dict(c: Constitution) -> dict[str, Any]:
    return asdict(c)
