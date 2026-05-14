from __future__ import annotations

import json
from pathlib import Path

from argus_live.execution.state_machine import validate_transition


def validate_journal(path: str | Path) -> list[str]:
    errors: list[str] = []
    state: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            evt = json.loads(line)
            if evt["event_type"] == "INTENT_CREATED":
                state[evt["entity_id"]] = "PROPOSED"
            elif evt["event_type"] == "STATE_TRANSITION":
                entity_id = evt["entity_id"]
                current = state.get(entity_id)
                if current is None:
                    errors.append(f"line {line_no}: transition before intent creation")
                    continue
                nxt = evt["payload"]["to"]
                result = validate_transition(current, nxt)
                if not result.ok:
                    errors.append(f"line {line_no}: {result.reason}")
                else:
                    state[entity_id] = nxt
    return errors
