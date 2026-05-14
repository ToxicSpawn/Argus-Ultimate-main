"""
BrainService (safe stub).

The previous version was a placeholder ML microservice with corrupted syntax and
heavy optional dependencies. This stub keeps the import surface stable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AlphaSignal:
    symbol: str
    action: str  # BUY/SELL/HOLD
    confidence: float
    timestamp: float
    meta: Dict[str, Any]


class BrainService:
    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self._running = False

    async def start(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(1.0)

    async def stop(self) -> None:
        self._running = False

    def score_tick(self, symbol: str, price: float, timestamp: float) -> Optional[AlphaSignal]:
        # Placeholder: emit nothing by default
        return None

