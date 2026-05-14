"""Batch 2 — Real-time sentiment feed aggregator.

Pulls sentiment scores from multiple sources (CryptoPanic, Santiment,
Fear & Greed Index) and exposes a normalised [-1, 1] composite score.
Designed to be polled asynchronously by the strategy layer.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"


@dataclass
class SentimentSnapshot:
    timestamp: float
    fear_greed: float  # 0-100 → mapped to [-1, 1]
    cryptopanic: Optional[float]  # [-1, 1]
    santiment: Optional[float]  # [-1, 1]
    composite: float
    sources: List[str] = field(default_factory=list)


class SentimentFeed:
    """Aggregate crypto sentiment from public APIs."""

    def __init__(
        self,
        cryptopanic_token: str = "",
        santiment_api_key: str = "",
        refresh_interval_s: float = 300.0,
        fear_greed_weight: float = 0.5,
        cryptopanic_weight: float = 0.3,
        santiment_weight: float = 0.2,
    ) -> None:
        self._cp_token = cryptopanic_token
        self._sa_key = santiment_api_key
        self._refresh_s = refresh_interval_s
        self._weights: Dict[str, float] = {
            "fear_greed": fear_greed_weight,
            "cryptopanic": cryptopanic_weight,
            "santiment": santiment_weight,
        }
        self._snapshot: Optional[SentimentSnapshot] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_sentiment(self, symbol: str = "BTC") -> SentimentSnapshot:
        """Return the latest sentiment snapshot, refreshing if stale."""
        async with self._lock:
            if self._snapshot is None or self._is_stale():
                self._snapshot = await self._refresh(symbol)
        return self._snapshot  # type: ignore[return-value]

    def get_sentiment_sync(self, symbol: str = "BTC") -> SentimentSnapshot:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.get_sentiment(symbol))
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_stale(self) -> bool:
        if self._snapshot is None:
            return True
        return time.time() - self._snapshot.timestamp > self._refresh_s

    async def _refresh(self, symbol: str) -> SentimentSnapshot:
        fg = await self._fetch_fear_greed()
        cp = await self._fetch_cryptopanic(symbol) if self._cp_token else None
        sa = None  # Santiment requires GraphQL; placeholder

        sources = ["fear_greed"]
        composite_num = self._weights["fear_greed"] * fg
        composite_den = self._weights["fear_greed"]

        if cp is not None:
            composite_num += self._weights["cryptopanic"] * cp
            composite_den += self._weights["cryptopanic"]
            sources.append("cryptopanic")

        composite = composite_num / composite_den if composite_den > 0 else 0.0
        return SentimentSnapshot(
            timestamp=time.time(),
            fear_greed=fg,
            cryptopanic=cp,
            santiment=sa,
            composite=float(composite),
            sources=sources,
        )

    async def _fetch_fear_greed(self) -> float:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(FEAR_GREED_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
            value = int(data["data"][0]["value"])  # 0-100
            return (value - 50) / 50.0  # map to [-1, 1]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fear-greed fetch failed: %s", exc)
            return 0.0

    async def _fetch_cryptopanic(
        self, symbol: str
    ) -> Optional[float]:
        url = (
            f"https://cryptopanic.com/api/v1/posts/"
            f"?auth_token={self._cp_token}&currencies={symbol}&public=true&kind=news"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
            results = data.get("results", [])
            if not results:
                return None
            # Map votes to sentiment: positive-negative/total
            scores = []
            for item in results[:20]:
                v = item.get("votes", {})
                pos = v.get("positive", 0)
                neg = v.get("negative", 0)
                total = pos + neg
                if total > 0:
                    scores.append((pos - neg) / total)
            return float(sum(scores) / len(scores)) if scores else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("CryptoPanic fetch failed: %s", exc)
            return None
