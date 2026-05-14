#!/usr/bin/env python3
"""
Unified Execution Engine (legacy compatibility).

Some legacy modules import `core.execution.unified_execution.UnifiedExecutionEngine`.
This implementation adapts the existing `core.execution_unified.UnifiedExecutionEngine`
to the older `execute(...)` API used by `core.execution.unified_execution_facade`.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict
from typing import Any, Dict, Optional

try:
    from core.execution_unified import ExecutionRequest, UnifiedExecutionEngine as _Engine
except ImportError:
    # core.execution_unified may not exist in all configurations
    from dataclasses import dataclass

    @dataclass
    class ExecutionRequest:
        symbol: str = ""
        side: str = ""
        quantity: float = 0.0
        price: float = 0.0
        order_type: str = "limit"

    class _Engine:
        def __init__(self, *args, **kwargs):
            pass


class _ExchangeManagerAdapter:
    def __init__(self, exchange_connector: Any):
        self._ex = exchange_connector

    async def execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Adapter for `core.execution_unified.UnifiedExecutionEngine` which expects an
        exchange_manager with `execute_order(dict) -> dict`.
        """

        ex = self._ex
        if ex is None:
            return None

        symbol = str(order.get("symbol") or "")
        side = str(order.get("side") or "")
        order_type = str(order.get("order_type") or "market")
        amount = float(order.get("amount") or 0.0)
        price = order.get("price")
        params = order.get("params") or {}

        # Preferred: connectors that implement `execute_order(dict)`
        fn = getattr(ex, "execute_order", None)
        if callable(fn):
            res = fn(dict(order))
            if inspect.isawaitable(res):
                res = await res
            if isinstance(res, dict):
                return res

        # Next: `create_order(...)` (core.exchange_connector)
        fn2 = getattr(ex, "create_order", None)
        if not callable(fn2):
            return None

        # Try keyword style.
        try:
            res2 = fn2(
                symbol=symbol,
                side=side,
                amount=amount,
                order_type=order_type,
                price=price if price is None else float(price),
                params=params,
            )
            if inspect.isawaitable(res2):
                res2 = await res2
            if isinstance(res2, dict):
                return res2
        except TypeError:
            pass

        # Dataclass style.
        try:
            from core.exchange_connector import OrderRequest

            req = OrderRequest(
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=float(amount),
                price=price if price is None else float(price),
                params=params if isinstance(params, dict) else None,
            )
            res3 = fn2(req)
            if inspect.isawaitable(res3):
                res3 = await res3
            if res3 is None:
                return None
            if isinstance(res3, dict):
                return res3
            return getattr(res3, "__dict__", None) or None
        except Exception:
            return None


class UnifiedExecutionEngine:
    def __init__(self, exchange_connector: Any = None, data_feed: Any = None):
        self.exchange_connector = exchange_connector
        self.data_feed = data_feed
        self._engine = _Engine(exchange_manager=_ExchangeManagerAdapter(exchange_connector))

    async def execute(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        execution_algo: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        algo = (execution_algo or kwargs.get("algo") or "direct").lower()
        if algo in {"market", "limit"}:
            algo = "direct"

        req = ExecutionRequest(
            symbol=str(symbol),
            side=str(side),
            amount=float(quantity),
            order_type=str(order_type),
            price=price if price is None else float(price),
            algo=str(algo),
            params=dict(kwargs) if kwargs else None,
        )
        res = await self._engine.execute_order(req)
        # core.execution_unified.ExecutionResult is a dataclass
        out = asdict(res)
        out["request_id"] = getattr(res, "request_id", out.get("request_id", ""))
        return out

    async def cancel_order(self, order_id: str) -> bool:
        ex = self.exchange_connector
        fn = getattr(ex, "cancel_order", None)
        if callable(fn):
            res = fn(order_id)
            if inspect.isawaitable(res):
                res = await res
            return bool(res)
        return False

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        ex = self.exchange_connector
        fn = getattr(ex, "get_order_status", None)
        if callable(fn):
            res = fn(order_id)
            if inspect.isawaitable(res):
                res = await res
            if res is None:
                return None
            if isinstance(res, dict):
                return res
            od = getattr(res, "__dict__", None)
            return dict(od) if isinstance(od, dict) else None
        return None

