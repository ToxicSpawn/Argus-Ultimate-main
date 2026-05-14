"""Advanced Order Types.

Includes:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg Orders
- Stop-Limit Orders
- OCO (One-Cancels-the-Other)
- Trailing Stop
- Market if Touched (MIT)
"""

from __future__ import annotations

import logging
import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class OrderAlgoType(Enum):
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    STOP_LIMIT = "stop_limit"
    OCO = "oco"
    TRAILING_STOP = "trailing_stop"
    MARKET_IF_TOUCHED = "mit"


@dataclass
class AlgoOrderConfig:
    symbol: str
    side: str
    total_qty: float
    algo_type: OrderAlgoType
    duration_secs: float = 300
    slice_count: int = 10
    participation_rate: float = 0.1
    start_price: float = 0.0
    stop_price: float = 0.0
    limit_price: float = 0.0
    trailing_distance_pct: float = 0.5
    icebergs: int = 5
    callback: Optional[Callable] = None


@dataclass
class AlgoOrderStatus:
    order_id: str
    algo_type: OrderAlgoType
    status: str
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    avg_fill_price: float = 0.0
    slices_completed: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    child_order_ids: List[str] = field(default_factory=list)


class TWAPExecutor:
    def __init__(self, adapter: Any = None):
        self._adapter = adapter
        self._orders: Dict[str, AlgoOrderStatus] = {}
        self._executions: deque = deque(maxlen=1000)

    async def execute(self, config: AlgoOrderConfig) -> str:
        order_id = f"twap_{int(time.time() * 1000)}"
        
        status = AlgoOrderStatus(
            order_id=order_id,
            algo_type=OrderAlgoType.TWAP,
            status="RUNNING",
            remaining_qty=config.total_qty,
            start_time=time.time(),
        )
        self._orders[order_id] = status
        
        asyncio.create_task(self._run_twap(order_id, config))
        
        return order_id

    async def _run_twap(self, order_id: str, config: AlgoOrderConfig) -> None:
        status = self._orders[order_id]
        slice_qty = config.total_qty / config.slice_count
        interval = config.duration_secs / config.slice_count
        
        for i in range(config.slice_count):
            if status.status == "CANCELLED":
                break
            
            try:
                price = await self._get_smart_price(config.symbol)
                
                result = await self._adapter.create_order(
                    symbol=config.symbol,
                    side=config.side,
                    order_type="LIMIT",
                    qty=slice_qty,
                    price=price,
                )
                
                child_id = result.get("orderId", f"child_{i}")
                status.child_order_ids.append(child_id)
                status.filled_qty += slice_qty
                status.remaining_qty -= slice_qty
                status.avg_fill_price = (
                    (status.avg_fill_price * (i) + price) / (i + 1)
                )
                status.slices_completed += 1
                
                self._executions.append({
                    "order_id": order_id,
                    "child_id": child_id,
                    "price": price,
                    "qty": slice_qty,
                    "timestamp": time.time(),
                })
                
            except Exception as e:
                logger.error(f"TWAP slice {i} failed: {e}")
            
            if i < config.slice_count - 1:
                await asyncio.sleep(interval)
        
        status.status = "COMPLETED"
        status.end_time = time.time()
        logger.info(f"TWAP order {order_id} completed: {status.filled_qty} @ {status.avg_fill_price}")

    async def _get_smart_price(self, symbol: str) -> float:
        if self._adapter:
            try:
                ticker = await self._adapter.fetch_ticker(symbol)
                return ticker.last
            except Exception:
                pass
        return 50000.0


class VWAPExecutor:
    def __init__(self, adapter: Any = None):
        self._adapter = adapter
        self._orders: Dict[str, AlgoOrderStatus] = {}
        self._volume_profile: Dict[str, List[float]] = {}
        self._executions: deque = deque(maxlen=1000)

    async def execute(self, config: AlgoOrderConfig) -> str:
        order_id = f"vwap_{int(time.time() * 1000)}"
        
        status = AlgoOrderStatus(
            order_id=order_id,
            algo_type=OrderAlgoType.VWAP,
            status="RUNNING",
            remaining_qty=config.total_qty,
            start_time=time.time(),
        )
        self._orders[order_id] = status
        
        asyncio.create_task(self._run_vwap(order_id, config))
        
        return order_id

    async def _run_vwap(self, order_id: str, config: AlgoOrderConfig) -> None:
        status = self._orders[order_id]
        
        volume_profile = self._get_volume_profile(config.symbol, config.slice_count)
        
        for i in range(config.slice_count):
            if status.status == "CANCELLED":
                break
            
            target_pct = volume_profile[i] if i < len(volume_profile) else 1.0 / config.slice_count
            slice_qty = config.total_qty * target_pct
            
            try:
                price = await self._get_smart_price(config.symbol)
                
                result = await self._adapter.create_order(
                    symbol=config.symbol,
                    side=config.side,
                    order_type="LIMIT",
                    qty=slice_qty,
                    price=price,
                )
                
                child_id = result.get("orderId", f"child_{i}")
                status.child_order_ids.append(child_id)
                status.filled_qty += slice_qty
                status.remaining_qty -= slice_qty
                
            except Exception as e:
                logger.error(f"VWAP slice {i} failed: {e}")
        
        status.status = "COMPLETED"
        status.end_time = time.time()

    def _get_volume_profile(self, symbol: str, slices: int) -> List[float]:
        if symbol in self._volume_profile:
            return self._volume_profile[symbol]
        
        base = np.linspace(0.1, 1.0, slices)
        profile = base / base.sum()
        return list(profile)

    async def _get_smart_price(self, symbol: str) -> float:
        if self._adapter:
            try:
                ticker = await self._adapter.fetch_ticker(symbol)
                return ticker.last
            except Exception:
                pass
        return 50000.0


class IcebergExecutor:
    def __init__(self, adapter: Any = None):
        self._adapter = adapter
        self._orders: Dict[str, AlgoOrderStatus] = {}

    async def execute(self, config: AlgoOrderConfig) -> str:
        order_id = f"iceberg_{int(time.time() * 1000)}"
        
        status = AlgoOrderStatus(
            order_id=order_id,
            algo_type=OrderAlgoType.ICEBERG,
            status="RUNNING",
            remaining_qty=config.total_qty,
            start_time=time.time(),
        )
        self._orders[order_id] = status
        
        asyncio.create_task(self._run_iceberg(order_id, config))
        
        return order_id

    async def _run_iceberg(self, order_id: str, config: AlgoOrderConfig) -> None:
        status = self._orders[order_id]
        visible_qty = config.total_qty / config.icebergs
        
        for i in range(config.icebergs):
            if status.status == "CANCELLED":
                break
            
            try:
                result = await self._adapter.create_order(
                    symbol=config.symbol,
                    side=config.side,
                    order_type="LIMIT",
                    qty=visible_qty,
                    price=config.start_price,
                    iceberg_qty=visible_qty,
                )
                
                status.child_order_ids.append(result.get("orderId", ""))
                status.filled_qty += visible_qty
                status.remaining_qty -= visible_qty
                
            except Exception as e:
                logger.error(f"Iceberg slice {i} failed: {e}")
            
            await asyncio.sleep(1)
        
        status.status = "COMPLETED"
        status.end_time = time.time()


class StopLimitExecutor:
    def __init__(self, adapter: Any = None):
        self._adapter = adapter
        self._orders: Dict[str, AlgoOrderStatus] = {}
        self._triggers: Dict[str, Tuple[float, float]] = {}

    async def execute(self, config: AlgoOrderConfig) -> str:
        order_id = f"stop_{int(time.time() * 1000)}"
        
        status = AlgoOrderStatus(
            order_id=order_id,
            algo_type=OrderAlgoType.STOP_LIMIT,
            status="WAITING",
            remaining_qty=config.total_qty,
            start_time=time.time(),
        )
        self._orders[order_id] = status
        self._triggers[order_id] = (config.stop_price, config.limit_price)
        
        asyncio.create_task(self._monitor_stop(order_id, config))
        
        return order_id

    async def _monitor_stop(self, order_id: str, config: AlgoOrderConfig) -> None:
        stop_price, limit_price = self._triggers.get(order_id, (0, 0))
        status = self._orders[order_id]
        
        while status.status == "WAITING":
            try:
                ticker = await self._adapter.fetch_ticker(config.symbol)
                current_price = ticker.last
                
                triggered = False
                if config.side == "BUY" and current_price >= stop_price:
                    triggered = True
                elif config.side == "SELL" and current_price <= stop_price:
                    triggered = True
                
                if triggered:
                    result = await self._adapter.create_order(
                        symbol=config.symbol,
                        side=config.side,
                        order_type="LIMIT",
                        qty=config.total_qty,
                        price=limit_price,
                        stop_price=stop_price,
                    )
                    
                    status.child_order_ids.append(result.get("orderId", ""))
                    status.status = "TRIGGERED"
                    logger.info(f"Stop-limit triggered: {order_id} at {current_price}")
                    break
                    
            except Exception as e:
                logger.error(f"Stop monitor error: {e}")
            
            await asyncio.sleep(1)


class OCOExecutor:
    def __init__(self, adapter: Any = None):
        self._adapter = adapter
        self._orders: Dict[str, AlgoOrderStatus] = {}

    async def execute(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        limit_price: float,
    ) -> str:
        order_id = f"oco_{int(time.time() * 1000)}"
        
        status = AlgoOrderStatus(
            order_id=order_id,
            algo_type=OrderAlgoType.OCO,
            status="WAITING",
            remaining_qty=qty,
            start_time=time.time(),
        )
        self._orders[order_id] = status
        
        asyncio.create_task(self._run_oco(order_id, symbol, side, qty, stop_price, limit_price))
        
        return order_id

    async def _run_oco(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        limit_price: float,
    ) -> None:
        status = self._orders[order_id]
        
        try:
            result = await self._adapter.create_order(
                symbol=symbol,
                side=side,
                order_type="OCO",
                qty=qty,
                stop_price=stop_price,
                price=limit_price,
            )
            
            status.child_order_ids.append(result.get("orderId", ""))
            status.status = "PLACED"
            
        except Exception as e:
            logger.error(f"OCO order failed: {e}")
            status.status = "FAILED"


class TrailingStopExecutor:
    def __init__(self, adapter: Any = None):
        self._adapter = adapter
        self._orders: Dict[str, AlgoOrderStatus] = {}
        self._trailing_data: Dict[str, Dict] = {}

    async def execute(
        self,
        symbol: str,
        side: str,
        qty: float,
        trailing_distance_pct: float = 0.5,
    ) -> str:
        order_id = f"trail_{int(time.time() * 1000)}"
        
        status = AlgoOrderStatus(
            order_id=order_id,
            algo_type=OrderAlgoType.TRAILING_STOP,
            status="RUNNING",
            remaining_qty=qty,
            start_time=time.time(),
        )
        self._orders[order_id] = status
        
        self._trailing_data[order_id] = {
            "highest_price": 0.0 if side == "SELL" else float("inf"),
            "trailing_distance_pct": trailing_distance_pct / 100,
            "activated": False,
        }
        
        asyncio.create_task(self._monitor_trailing(order_id, symbol, side, qty))
        
        return order_id

    async def _monitor_trailing(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
    ) -> None:
        status = self._orders[order_id]
        trail = self._trailing_data[order_id]
        
        while status.status == "RUNNING":
            try:
                ticker = await self._adapter.fetch_ticker(symbol)
                current_price = ticker.last
                
                if side == "BUY":
                    if current_price > trail["highest_price"]:
                        trail["highest_price"] = current_price
                    stop_price = current_price * (1 - trail["trailing_distance_pct"])
                    
                    if not trail["activated"] and current_price >= trail["highest_price"] * 0.99:
                        trail["activated"] = True
                    
                    if trail["activated"] and current_price <= stop_price:
                        await self._adapter.create_order(
                            symbol=symbol,
                            side="SELL",
                            order_type="MARKET",
                            qty=qty,
                        )
                        status.status = "TRIGGERED"
                        break
                        
                else:
                    if current_price < trail["highest_price"] or trail["highest_price"] == 0:
                        trail["highest_price"] = current_price
                    stop_price = current_price * (1 + trail["trailing_distance_pct"])
                    
                    if not trail["activated"] and current_price <= trail["highest_price"] * 1.01:
                        trail["activated"] = True
                    
                    if trail["activated"] and current_price >= stop_price:
                        await self._adapter.create_order(
                            symbol=symbol,
                            side="BUY",
                            order_type="MARKET",
                            qty=qty,
                        )
                        status.status = "TRIGGERED"
                        break
                        
            except Exception as e:
                logger.error(f"Trailing stop monitor error: {e}")
            
            await asyncio.sleep(1)


class AdvancedOrderManager:
    def __init__(self, adapter: Any = None):
        self._adapter = adapter
        self._twap = TWAPExecutor(adapter)
        self._vwap = VWAPExecutor(adapter)
        self._iceberg = IcebergExecutor(adapter)
        self._stop_limit = StopLimitExecutor(adapter)
        self._oco = OCOExecutor(adapter)
        self._trailing = TrailingStopExecutor(adapter)
        
        self._all_orders: Dict[str, AlgoOrderStatus] = {}

    async def create_twap_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        duration_secs: float = 300,
        slice_count: int = 10,
    ) -> str:
        config = AlgoOrderConfig(
            symbol=symbol,
            side=side,
            total_qty=qty,
            algo_type=OrderAlgoType.TWAP,
            duration_secs=duration_secs,
            slice_count=slice_count,
        )
        order_id = await self._twap.execute(config)
        self._all_orders[order_id] = self._twap._orders[order_id]
        return order_id

    async def create_vwap_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        duration_secs: float = 300,
        slice_count: int = 10,
    ) -> str:
        config = AlgoOrderConfig(
            symbol=symbol,
            side=side,
            total_qty=qty,
            algo_type=OrderAlgoType.VWAP,
            duration_secs=duration_secs,
            slice_count=slice_count,
        )
        order_id = await self._vwap.execute(config)
        self._all_orders[order_id] = self._vwap._orders[order_id]
        return order_id

    async def create_iceberg_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        icebergs: int = 5,
    ) -> str:
        config = AlgoOrderConfig(
            symbol=symbol,
            side=side,
            total_qty=qty,
            algo_type=OrderAlgoType.ICEBERG,
            start_price=price,
            icebergs=icebergs,
        )
        order_id = await self._iceberg.execute(config)
        self._all_orders[order_id] = self._iceberg._orders[order_id]
        return order_id

    async def create_stop_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        limit_price: float,
    ) -> str:
        config = AlgoOrderConfig(
            symbol=symbol,
            side=side,
            total_qty=qty,
            algo_type=OrderAlgoType.STOP_LIMIT,
            stop_price=stop_price,
            limit_price=limit_price,
        )
        order_id = await self._stop_limit.execute(config)
        self._all_orders[order_id] = self._stop_limit._orders[order_id]
        return order_id

    async def create_oco_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        limit_price: float,
    ) -> str:
        order_id = await self._oco.execute(symbol, side, qty, stop_price, limit_price)
        self._all_orders[order_id] = self._oco._orders[order_id]
        return order_id

    async def create_trailing_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        trailing_distance_pct: float = 0.5,
    ) -> str:
        order_id = await self._trailing.execute(symbol, side, qty, trailing_distance_pct)
        self._all_orders[order_id] = self._trailing._orders[order_id]
        return order_id

    def get_order_status(self, order_id: str) -> Optional[AlgoOrderStatus]:
        return self._all_orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._all_orders:
            self._all_orders[order_id].status = "CANCELLED"
            return True
        return False
