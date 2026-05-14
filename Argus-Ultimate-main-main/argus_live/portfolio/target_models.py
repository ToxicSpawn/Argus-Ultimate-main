from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid


@dataclass(frozen=True)
class TargetProposal:
    proposal_id: str
    strategy_id: str
    symbol: str
    target_weight: float
    current_weight: float
    reference_price: float
    manifest_hash: str
    created_at_utc: str

    @staticmethod
    def new(*, strategy_id: str, symbol: str, target_weight: float, current_weight: float, reference_price: float, manifest_hash: str) -> "TargetProposal":
        return TargetProposal(str(uuid.uuid4()), strategy_id, symbol, target_weight, current_weight, reference_price, manifest_hash, datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class TargetDelta:
    proposal_id: str
    strategy_id: str
    symbol: str
    delta_weight: float
    delta_notional: float
    side: str
    reference_price: float
    manifest_hash: str
