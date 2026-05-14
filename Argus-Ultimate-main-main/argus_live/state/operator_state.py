from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path("reports/operator_state.json")


class OperatorMode(enum.Enum):
    NORMAL = "NORMAL"
    HALTED = "HALTED"
    FROZEN = "FROZEN"


@dataclass(frozen=True)
class OperatorState:
    mode: OperatorMode = OperatorMode.NORMAL
    last_updated_ts: str = ""

    def to_dict(self) -> dict:
        return {"mode": self.mode.value, "last_updated_ts": self.last_updated_ts}

    @classmethod
    def from_dict(cls, d: dict) -> OperatorState:
        return cls(
            mode=OperatorMode(d.get("mode", "NORMAL")),
            last_updated_ts=d.get("last_updated_ts", ""),
        )


class OperatorStateStore:
    """Persistent operator state store — wraps load/save for bootstrap."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else STATE_FILE

    def load(self) -> OperatorState:
        return load_state(self._path)

    def save(self, state: OperatorState) -> None:
        save_state(state, self._path)

    def set_mode(self, mode: OperatorMode) -> OperatorState:
        return set_mode(mode, self._path)

    def assert_trading_allowed(self) -> None:
        assert_trading_allowed(self._path)


def load_state(path: Path | None = None) -> OperatorState:
    p = path or STATE_FILE
    if not p.exists():
        return OperatorState()
    raw = json.loads(p.read_text(encoding="utf-8"))
    return OperatorState.from_dict(raw)


def save_state(state: OperatorState, path: Path | None = None) -> None:
    p = path or STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def set_mode(mode: OperatorMode, path: Path | None = None) -> OperatorState:
    state = OperatorState(
        mode=mode,
        last_updated_ts=datetime.now(timezone.utc).isoformat(),
    )
    save_state(state, path)
    return state


def assert_trading_allowed(path: Path | None = None) -> None:
    state = load_state(path)
    if state.mode == OperatorMode.HALTED:
        raise RuntimeError("Trading halted by operator kill-switch")
    if state.mode == OperatorMode.FROZEN:
        raise RuntimeError("System frozen — no trading permitted")
