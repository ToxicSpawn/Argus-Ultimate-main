"""
DataService (safe stub).

The previous version was a placeholder microservice with corrupted syntax.
This module remains as a minimal, import-safe scaffold.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MarketTick:
    symbol: str
    price: float
    timestamp: float
    meta: Dict[str, Any]


class DataService:
    def __init__(self, symbols: Optional[List[str]] = None, interval_s: float = 1.0) -> None:
        self.symbols = symbols or ["BTC/USD", "ETH/USD"]
        self.interval_s = float(interval_s)
        self._running = False

    async def start(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval_s)

    async def stop(self) -> None:
        self._running = False

