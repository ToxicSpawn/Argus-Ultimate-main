from __future__ import annotations

import json
from pathlib import Path


def summarize_jsonl(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "lines": 0}
    count = 0
    event_types: dict[str, int] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                evt = json.loads(line)
                et = evt.get("event_type", "unknown")
                event_types[et] = event_types.get(et, 0) + 1
    return {"exists": True, "lines": count, "event_types": event_types}
