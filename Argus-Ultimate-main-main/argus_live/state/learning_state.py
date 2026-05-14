from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LearningState:
    """Persistent state for the live learning system."""

    strategy_edge_bps: dict[str, float] = field(default_factory=dict)
    strategy_trade_count: dict[str, int] = field(default_factory=dict)
    venue_slippage_bps: dict[str, float] = field(default_factory=dict)
    venue_trade_count: dict[str, int] = field(default_factory=dict)
    lifecycle_state: dict[str, str] = field(default_factory=dict)


class LearningStateStore:
    """Load and save LearningState to a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> LearningState:
        if not self._path.exists():
            logger.info("No learning state file at %s, returning defaults", self._path)
            return LearningState()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return LearningState(
                strategy_edge_bps=data.get("strategy_edge_bps", {}),
                strategy_trade_count=data.get("strategy_trade_count", {}),
                venue_slippage_bps=data.get("venue_slippage_bps", {}),
                venue_trade_count=data.get("venue_trade_count", {}),
                lifecycle_state=data.get("lifecycle_state", {}),
            )
        except Exception:
            logger.exception("Failed to load learning state from %s", self._path)
            return LearningState()

    def save(self, state: LearningState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            logger.exception("Failed to save learning state to %s", self._path)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
