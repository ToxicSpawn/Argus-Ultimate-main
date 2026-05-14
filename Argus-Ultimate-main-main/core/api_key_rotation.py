"""
API Key Rotation Policy — tracks exchange API key age and enforces rotation.

Warns after 90 days, blocks live trading after 180 days without rotation.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyStatus:
    """Status of a single API key."""
    exchange: str
    key_prefix: str     # first 8 chars of key (for identification without exposing full key)
    created_at: float
    last_rotated_at: float
    age_days: float
    status: str         # "ok", "warn", "expired"
    reason: str


class APIKeyRotationPolicy:
    """
    Tracks API key age and enforces rotation policy.

    Policy:
        - age < warn_days: OK
        - warn_days <= age < block_days: WARNING (log, alert)
        - age >= block_days: BLOCKED (refuse live trading)
    """

    def __init__(
        self,
        state_path: str = "data/api_key_rotation.json",
        warn_days: int = 90,
        block_days: int = 180,
    ):
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._warn_days = warn_days
        self._block_days = block_days
        self._keys: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self._state_path.write_text(
            json.dumps(self._keys, indent=2, default=str),
            encoding="utf-8",
        )

    def register_key(self, exchange: str, api_key: str) -> None:
        """Register or re-register an API key (call on startup or after rotation)."""
        prefix = api_key[:8] if len(api_key) >= 8 else api_key
        existing = self._keys.get(exchange, {})

        # If same key prefix, don't reset rotation time
        if existing.get("key_prefix") == prefix:
            return

        # New key — record rotation
        self._keys[exchange] = {
            "key_prefix": prefix,
            "created_at": existing.get("created_at", time.time()),
            "last_rotated_at": time.time(),
        }
        self._save()
        logger.info("API key registered/rotated for %s (prefix=%s)", exchange, prefix)

    def check(self, exchange: str) -> KeyStatus:
        """Check the rotation status of a key."""
        info = self._keys.get(exchange)
        if not info:
            return KeyStatus(
                exchange=exchange, key_prefix="unknown",
                created_at=0, last_rotated_at=0,
                age_days=999, status="expired",
                reason="no key registered",
            )

        last_rotated = float(info.get("last_rotated_at", 0))
        age_days = (time.time() - last_rotated) / 86400.0

        if age_days >= self._block_days:
            status = "expired"
            reason = f"key age {age_days:.0f} days >= {self._block_days} day limit — rotate immediately"
        elif age_days >= self._warn_days:
            status = "warn"
            reason = f"key age {age_days:.0f} days >= {self._warn_days} day warning — rotate soon"
        else:
            status = "ok"
            reason = f"key age {age_days:.0f} days — within policy"

        return KeyStatus(
            exchange=exchange,
            key_prefix=str(info.get("key_prefix", "?")),
            created_at=float(info.get("created_at", 0)),
            last_rotated_at=last_rotated,
            age_days=round(age_days, 1),
            status=status,
            reason=reason,
        )

    def check_all(self) -> List[KeyStatus]:
        """Check all registered keys."""
        return [self.check(exchange) for exchange in self._keys]

    def assert_live_allowed(self, exchange: str) -> None:
        """Raise if key is too old for live trading."""
        status = self.check(exchange)
        if status.status == "expired":
            raise RuntimeError(f"API key rotation required for {exchange}: {status.reason}")
        if status.status == "warn":
            logger.warning("API key rotation warning for %s: %s", exchange, status.reason)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "registered_exchanges": list(self._keys.keys()),
            "warn_days": self._warn_days,
            "block_days": self._block_days,
            "statuses": {ex: self.check(ex).status for ex in self._keys},
        }
