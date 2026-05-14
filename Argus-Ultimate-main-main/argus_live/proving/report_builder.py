from __future__ import annotations

import json
from pathlib import Path

from argus_live.proving.day_review import DayReview


def write_review_report(review: DayReview, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(review.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
