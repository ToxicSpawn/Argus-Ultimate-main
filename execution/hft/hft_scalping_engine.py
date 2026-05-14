"""HFT scalping engine — moved from hft/."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HFTScalpingEngine:
    """High-frequency scalping engine."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        logger.info('HFTScalpingEngine initialised')

    async def run_tick(self, orderbook: dict) -> dict | None:
        """Process a single tick and return an order signal or None."""
        raise NotImplementedError
