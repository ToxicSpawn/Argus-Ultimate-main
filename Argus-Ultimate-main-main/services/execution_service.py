"""
ExecutionService (safe stub).

The previous version was a placeholder microservice with corrupted syntax.
This module remains as a minimal, import-safe scaffold.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExecutionRequest:
    symbol: str
    side: str
    size: float
    timestamp: float
    meta: Dict[str, Any]


class ExecutionService:
    def __init__(self) -> None:
        self._running = False

    async def start(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(1.0)

    async def stop(self) -> None:
        self._running = False

    async def submit(self, req: ExecutionRequest) -> Dict[str, Any]:
        return {"accepted": False, "reason": "ExecutionService is a stub in this build", "request": req.__dict__}

