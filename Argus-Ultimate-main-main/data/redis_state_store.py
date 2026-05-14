"""
Redis-Backed Position & State Store
=====================================
Persists bot state (positions, trades, capital, RL episode data) to Redis
so the bot resumes correctly after a crash or restart.

Usage:
    store = RedisStateStore()
    await store.connect()
    await store.save_position(symbol, pos_dict)
    pos = await store.load_position(symbol)
    await store.save_capital(capital)
    capital = await store.load_capital(default=1000.0)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REDIS_URL_ENV = "REDIS_URL"   # e.g. redis://localhost:6379/0
KEY_PREFIX    = "argus:"


class RedisStateStore:
    """
    Async Redis state store using aioredis.
    Falls back to in-memory dict if Redis is unavailable.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        prefix: str = KEY_PREFIX,
        ttl: int = 86400 * 7,    # keys expire in 7 days
    ):
        self._url    = url or os.environ.get(REDIS_URL_ENV, "redis://localhost:6379/0")
        self._prefix = prefix
        self._ttl    = ttl
        self._client: Any = None
        self._fallback: Dict[str, str] = {}   # in-memory fallback
        self._using_fallback = False

    async def connect(self) -> bool:
        """Connect to Redis. Returns True on success, False on fallback."""
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=3,
            )
            await self._client.ping()
            logger.info("RedisStateStore connected to %s", self._url)
            self._using_fallback = False
            return True
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — using in-memory fallback", exc)
            self._using_fallback = True
            return False

    async def close(self) -> None:
        if self._client and not self._using_fallback:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Generic get / set
    # ------------------------------------------------------------------

    async def _set(self, key: str, value: Any) -> None:
        raw = json.dumps(value)
        k   = self._prefix + key
        if self._using_fallback:
            self._fallback[k] = raw
            return
        try:
            await self._client.set(k, raw, ex=self._ttl)
        except Exception as exc:
            logger.debug("Redis set failed for %s: %s", k, exc)
            self._fallback[k] = raw

    async def _get(self, key: str, default: Any = None) -> Any:
        k = self._prefix + key
        if self._using_fallback:
            raw = self._fallback.get(k)
            return json.loads(raw) if raw is not None else default
        try:
            raw = await self._client.get(k)
            return json.loads(raw) if raw is not None else default
        except Exception as exc:
            logger.debug("Redis get failed for %s: %s", k, exc)
            return default

    async def _delete(self, key: str) -> None:
        k = self._prefix + key
        if self._using_fallback:
            self._fallback.pop(k, None)
            return
        try:
            await self._client.delete(k)
        except Exception as exc:
            logger.debug("Redis delete failed for %s: %s", k, exc)

    # ------------------------------------------------------------------
    # Position API
    # ------------------------------------------------------------------

    async def save_position(self, symbol: str, position: Dict[str, Any]) -> None:
        await self._set(f"position:{symbol}", position)

    async def load_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        return await self._get(f"position:{symbol}")

    async def delete_position(self, symbol: str) -> None:
        await self._delete(f"position:{symbol}")

    async def load_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Load all open positions."""
        pattern = self._prefix + "position:*"
        positions = {}
        if self._using_fallback:
            for k, v in self._fallback.items():
                if k.startswith(self._prefix + "position:"):
                    sym = k.replace(self._prefix + "position:", "")
                    positions[sym] = json.loads(v)
            return positions
        try:
            keys = await self._client.keys(pattern)
            for k in keys:
                sym = k.replace(self._prefix + "position:", "")
                raw = await self._client.get(k)
                if raw:
                    positions[sym] = json.loads(raw)
        except Exception as exc:
            logger.warning("load_all_positions failed: %s", exc)
        return positions

    # ------------------------------------------------------------------
    # Capital
    # ------------------------------------------------------------------

    async def save_capital(self, capital: float) -> None:
        await self._set("capital", capital)

    async def load_capital(self, default: float = 1000.0) -> float:
        val = await self._get("capital", default)
        return float(val)

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    async def append_trade(self, trade: Dict[str, Any]) -> None:
        """Append a trade record (stored as JSON list)."""
        trades = await self._get("trades", [])
        trades.append(trade)
        # Keep last 1000 trades
        if len(trades) > 1000:
            trades = trades[-1000:]
        await self._set("trades", trades)

    async def load_trades(self) -> List[Dict[str, Any]]:
        return await self._get("trades", [])

    # ------------------------------------------------------------------
    # RL episode slippage
    # ------------------------------------------------------------------

    async def save_rl_slippage(self, episode_slippage: float) -> None:
        await self._set("rl_episode_slippage", episode_slippage)

    async def load_rl_slippage(self) -> float:
        return float(await self._get("rl_episode_slippage", 0.0))

    # ------------------------------------------------------------------
    # Generic KV (for dashboards / monitoring)
    # ------------------------------------------------------------------

    async def set(self, key: str, value: Any) -> None:
        await self._set(key, value)

    async def get(self, key: str, default: Any = None) -> Any:
        return await self._get(key, default)
