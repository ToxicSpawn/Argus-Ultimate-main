from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LifecycleStore:
    """Persist strategy lifecycle map (strategy_id -> state name) as JSON."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, str]:
        if not self._path.exists():
            logger.info("No lifecycle store at %s, returning empty map", self._path)
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("Lifecycle store is not a dict, returning empty map")
                return {}
            return {str(k): str(v) for k, v in data.items()}
        except Exception:
            logger.exception("Failed to load lifecycle store from %s", self._path)
            return {}

    def save(self, lifecycle_map: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(lifecycle_map, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            logger.exception("Failed to save lifecycle store to %s", self._path)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
