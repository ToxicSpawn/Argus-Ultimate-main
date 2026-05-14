from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditRecord:
    actor: str
    action: str
    reason_code: str
    payload: dict[str, Any]
    created_at_utc: str

    @staticmethod
    def new(actor: str, action: str, reason_code: str, payload: dict[str, Any]) -> "AuditRecord":
        return AuditRecord(actor, action, reason_code, payload, datetime.now(timezone.utc).isoformat())


class AuditLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
