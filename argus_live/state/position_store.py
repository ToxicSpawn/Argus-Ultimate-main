from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    quantity: float
    avg_cost: float
    realized_pnl: float
    updated_at_utc: str


class PositionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, positions: dict[str, PositionSnapshot]) -> None:
        payload = {k: asdict(v) for k, v in positions.items()}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> dict[str, PositionSnapshot]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {k: PositionSnapshot(**v) for k, v in raw.items()}

    @staticmethod
    def now_snapshot(symbol: str, quantity: float, avg_cost: float, realized_pnl: float) -> PositionSnapshot:
        return PositionSnapshot(symbol, quantity, avg_cost, realized_pnl, datetime.now(timezone.utc).isoformat())
