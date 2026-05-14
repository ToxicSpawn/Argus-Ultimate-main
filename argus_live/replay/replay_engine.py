from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReplayEngine:
    def __init__(self) -> None:
        self.state: dict[str, str] = {}
        self.events: list[dict[str, Any]] = []

    def load_journal(self, path: str | Path) -> None:
        self.state.clear()
        self.events.clear()
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.events.append(json.loads(line))

    def replay(self) -> dict[str, str]:
        for event in self.events:
            event_type = event["event_type"]
            entity_id = event["entity_id"]
            payload = event["payload"]
            if event_type == "INTENT_CREATED":
                self.state[entity_id] = "PROPOSED"
            elif event_type == "STATE_TRANSITION":
                self.state[entity_id] = payload["to"]
        return dict(self.state)
