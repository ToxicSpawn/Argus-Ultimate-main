from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JournalEvent:
    event_type: str
    entity_id: str
    payload: dict[str, Any]
    created_at_utc: str

    @staticmethod
    def new(event_type: str, entity_id: str, payload: dict[str, Any]) -> "JournalEvent":
        return JournalEvent(event_type, entity_id, payload, datetime.now(timezone.utc).isoformat())


class EventJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: JournalEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), sort_keys=True) + "\n")
