from __future__ import annotations

import json
from pathlib import Path


class RuntimeStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict) -> None:
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))
