from __future__ import annotations

import json
from pathlib import Path

from argus_live.promotion.promotion_bundle import PromotionBundle


class ApprovalStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, bundle: PromotionBundle) -> None:
        existing = self.load_all()
        existing.append(bundle)
        self.path.write_text(json.dumps([b.to_dict() for b in existing], indent=2), encoding="utf-8")

    def load_all(self) -> list[PromotionBundle]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [PromotionBundle(**item) for item in raw]
