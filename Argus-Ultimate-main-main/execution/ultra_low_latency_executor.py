"""Ultra-Low Latency Execution Engine with Microsecond-Level Optimization.

Features:
- Batched order submission
- Connection pooling
- Parallel exchange communication
- Price cache with TTL
- Order flow optimization
- Smart order routing
"""

from __future__ import annotations

import asyncio
import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    submission_us: float = 0.0
    confirmation_us: float = 0.0
    total_round_trip_us: float = 0.0
    queue_wait_us: float = 0.0


@dataclass
class OptimizedOrder:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    submitted_at: float
    priority: int = 0
    batch_id: Optional[str] = None


class PriceCache:
    def __init__(self, ttl_ms: int = 100):
        self._cache: Dict[str, Tuple[float, float]] = {}
        self._ttl_ms = ttl_ms

    def get(self, symbol: str) -> Optional[float]:
        if symbol in self._cache:
            price, ts = self._cache[symbol]
            if (time.time() - ts) * 1000 < self._ttl_ms:
                return price
        return None

    def set(self, symbol: str, price: float) -> None:
        self._cache[symbol] = (price, time.time())

    def prefetch(self, symbols: List[str], prices: Dict[str, float]) -> None:
        for sym in symbols:
            if sym in prices:
                self.set(sym, prices[sym])


class OrderBatcher:
    def __init__(self, max_batch_size: int = 10, max_wait_us: int = 500):
        self._max_batch_size = max_batch_size
        self._max_wait_us = max_wait_us
        self._pending: deque[OptimizedOrder] = deque()
        self._last_flush = time.time()

    def add(self, order: OptimizedOrder) -> List[List[OptimizedOrder]]:
        self._pending.append(order)
        
        now = time.time()
        should_flush = (
            len(self._pending) >= self._max_batch_size or
            (now - self._last_flush) * 1_000_000 >= self._max_wait_us
        )
        
        if should_flush:
            self._last_flush = now
            batch = list(self._pending)
            self._pending.clear()
            return [batch]
        return []

    def flush(self) -> List[List[OptimizedOrder]]:
        if self._pending:
            batch = list(self._pending)
            self._pending.clear()
            self._last_flush = time.time()
            return [batch]
        return []


class ConnectionPool:
    def __init__(self, pool_size: int = 5):
        self._pool_size = pool_size
        self._semaphore = asyncio.Semaphore(pool_size)
        self._active = 0
        self._total_connections = 0

    async def acquire(self):
        await self._semaphore.acquire()
        self._active += 1
        self._total_connections += 1
        return self._active

    def release(self):
        self._active = max(0, self._active - 1)
        self._semaphore.release()


class UltraLowLatencyExecutor:
    def __init__(
        self,
        adapter: Any = None,
        max_batch_size: int = 10,
        max_wait_us: int = 500,
        price_cache_ttl_ms: int = 100,
        connection_pool_size: int = 5,
    ):
        self._adapter = adapter
        self._price_cache = PriceCache(ttl_ms=price_cache_ttl_ms)
        self._batcher = OrderBatcher(max_batch_size, max_wait_us)
        self._connection_pool = ConnectionPool(pool_size=connection_pool_size)
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        self._latency_history: deque = deque(maxlen=10000)
        self._order_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
        self._stats = {
            "orders_submitted": 0,
            "orders_batched": 0,
            "avg_latency_us": 0.0,
            "p50_latency_us": 0.0,
            "p99_latency_us": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def start(self):
        self._running = True
        asyncio.create_task(self._order_processor())
        asyncio.create_task(self._batch_scheduler())
        logger.info("Ultra-low latency executor started")

    async def stop(self):
        self._running = False
        for order in self._batcher.flush():
            await self._submit_batch(order)
        self._executor.shutdown(wait=False)

    async def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: Optional[float] = None,
        priority: int = 0,
    ) -> Tuple[str, LatencyMetrics]:
        t0 = time.perf_counter()
        
        cached_price = self._price_cache.get(symbol)
        if cached_price and not price:
            price = cached_price
            self._stats["cache_hits"] += 1
        else:
            self._stats["cache_misses"] += 1
        
        if not price:
            price = cached_price or 0.0
        
        order_id = hashlib.sha256(f"{symbol}{time.time()}".encode()).hexdigest()[:12]
        
        order = OptimizedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            submitted_at=t0,
            priority=priority,
        )
        
        await self._order_queue.put(order)
        
        metrics = LatencyMetrics()
        return order_id, metrics

    async def _order_processor(self):
        while self._running:
            try:
                order = await asyncio.wait_for(
                    self._order_queue.get(),
                    timeout=0.001
                )
                batches = self._batcher.add(order)
                for batch in batches:
                    await self._submit_batch(batch)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Order processor error: {e}")

    async def _batch_scheduler(self):
        while self._running:
            await asyncio.sleep(0.0005)
            batches = self._batcher.flush()
            for batch in batches:
                await self._submit_batch(batch)

    async def _submit_batch(self, batch: List[OptimizedOrder]) -> None:
        if not self._adapter or not batch:
            return
        
        conn_id = await self._connection_pool.acquire()
        try:
            t0 = time.perf_counter()
            
            if hasattr(self._adapter, 'submit_batch'):
                results = await self._adapter.submit_batch(batch)
            else:
                results = []
                for order in batch:
                    try:
                        result = await self._adapter.place_order(order)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Order {order.order_id} failed: {e}")
            
            latency_us = (time.perf_counter() - t0) * 1_000_000
            
            for order, result in zip(batch, results):
                total_latency = (time.perf_counter() - order.submitted_at) * 1_000_000
                self._latency_history.append(total_latency)
            
            self._stats["orders_submitted"] += len(batch)
            self._stats["orders_batched"] += len(batch) - 1
            self._update_latency_stats()
            
        finally:
            self._connection_pool.release()

    def _update_latency_stats(self):
        if not self._latency_history:
            return
        
        latencies = sorted(self._latency_history)
        n = len(latencies)
        
        self._stats["avg_latency_us"] = sum(latencies) / n
        self._stats["p50_latency_us"] = latencies[int(n * 0.5)]
        self._stats["p99_latency_us"] = latencies[int(n * 0.99)]

    def update_prices(self, prices: Dict[str, float]) -> None:
        self._price_cache.prefetch(list(prices.keys()), prices)

    def get_stats(self) -> Dict[str, Any]:
        total_cache = self._stats["cache_hits"] + self._stats["cache_misses"]
        cache_hit_rate = (
            self._stats["cache_hits"] / total_cache 
            if total_cache > 0 else 0.0
        )
        return {
            **self._stats,
            "cache_hit_rate": round(cache_hit_rate, 4),
        }


class SmartOrderRouter:
    def __init__(self, exchanges: List[Any] = None):
        self._exchanges = exchanges or []
        self._best_exchange: Dict[str, str] = {}

    def add_exchange(self, exchange: Any, priority: int = 0):
        self._exchanges.append((exchange, priority))
        self._exchanges.sort(key=lambda x: x[1])

    async def route_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "limit",
    ) -> Tuple[Any, float]:
        best_latency = float("inf")
        best_exchange = None
        
        for exchange, _ in self._exchanges:
            try:
                latency = getattr(exchange, '_avg_latency_us', float("inf"))
                if latency < best_latency:
                    best_latency = latency
                    best_exchange = exchange
            except Exception:
                continue
        
        if not best_exchange and self._exchanges:
            best_exchange = self._exchanges[0][0]
        
        return best_exchange, best_latency
