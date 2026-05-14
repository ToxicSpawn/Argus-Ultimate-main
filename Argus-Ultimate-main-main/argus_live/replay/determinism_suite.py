from __future__ import annotations

from pathlib import Path

from argus_live.replay.journal_validator import validate_journal
from argus_live.replay.replay_engine import ReplayEngine


def replay_terminal_state(path: str | Path) -> dict[str, str]:
    errors = validate_journal(path)
    if errors:
        raise RuntimeError("; ".join(errors))
    engine = ReplayEngine()
    engine.load_journal(path)
    return engine.replay()
