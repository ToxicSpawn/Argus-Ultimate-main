from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Deque, Dict, List, Optional


@dataclass
class FeatureRow:
    price: float
    spread_bps: float
    depth: float
    volume: float


class RollingFeatureStore:
    """Lightweight rolling feature store for deterministic regime classification."""

    def __init__(self, window: int = 120) -> None:
        self.window = max(10, int(window))
        self._rows: Dict[str, Deque[FeatureRow]] = {}

    def update(
        self,
        *,
        symbol: str,
        price: float,
        spread_bps: float = 0.0,
        depth: float = 0.0,
        volume: float = 0.0,
    ) -> None:
        sym = str(symbol or "")
        if not sym:
            return
        q = self._rows.setdefault(sym, deque(maxlen=self.window))
        q.append(
            FeatureRow(
                price=float(price or 0.0),
                spread_bps=float(spread_bps or 0.0),
                depth=float(depth or 0.0),
                volume=float(volume or 0.0),
            )
        )

    def snapshot(self) -> Dict[str, float]:
        rows: List[FeatureRow] = []
        for q in self._rows.values():
            rows.extend(list(q))
        if len(rows) < 3:
            return {
                "trend_slope": 0.0,
                "volatility_pct": 0.0,
                "spread_bps": 0.0,
                "depth": 0.0,
                "volume": 0.0,
            }
        prices = [r.price for r in rows if r.price > 0]
        if len(prices) < 3:
            return {
                "trend_slope": 0.0,
                "volatility_pct": 0.0,
                "spread_bps": mean([r.spread_bps for r in rows]),
                "depth": mean([r.depth for r in rows]),
                "volume": mean([r.volume for r in rows]),
            }
        rets = []
        for i in range(1, len(prices)):
            prev = max(prices[i - 1], 1e-9)
            rets.append((prices[i] - prices[i - 1]) / prev)
        trend = (prices[-1] - prices[0]) / max(prices[0], 1e-9)
        return {
            "trend_slope": float(trend),
            "volatility_pct": float(pstdev(rets) * 100.0) if len(rets) >= 2 else 0.0,
            "spread_bps": float(mean([r.spread_bps for r in rows])),
            "depth": float(mean([r.depth for r in rows])),
            "volume": float(mean([r.volume for r in rows])),
        }

    def count(self) -> int:
        return sum(len(q) for q in self._rows.values())
