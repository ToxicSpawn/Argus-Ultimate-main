from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class JSONLLogger:
    """Simple append-only JSONL logger for structured ops events."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: Dict[str, Any]) -> None:
        row = dict(event or {})
        row.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
