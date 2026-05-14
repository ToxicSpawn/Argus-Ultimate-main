from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PromotionBundle:
    strategy_id: str
    feature_hash: str
    training_window: str
    evaluation_window: str
    walk_forward_score: float
    stress_score: float
    replay_passed: bool
    approved_by: str | None
    approved_at_utc: str | None
    signature: str | None

    def to_dict(self) -> dict:
        return asdict(self)
