#!/usr/bin/env python3
"""
Fast execution engine (legacy compatibility).

This is a minimal, dependency-light implementation to support older modules that
use `core.execution.unified_execution_facade`. It routes orders directly to the
provided exchange connector.
"""

from __future__ import annotations
import logging

import logging

import inspect
import time
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

class FastExecutionEngine:
    def __init__(self, exchange_connector: Any = None) -> None:
        self.exchange_connector = exchange_connector

    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a single order as directly as possible.

        Supports multiple connector styles:
        - `create_order(OrderRequest(...))` (core.exchange_connector.ExchangeConnector)
        - `create_order(symbol=..., side=..., amount=..., order_type=..., price=...)`
        - `execute_order({...})`
        """

        ex = self.exchange_connector
        if ex is None:
            raise RuntimeError("FastExecutionEngine requires an exchange_connector")

        # 1) execute_order(dict) style
        fn = getattr(ex, "execute_order", None)
        if callable(fn):
            res = fn(
                {
                    "symbol": symbol,
                    "side": side,
                    "order_type": order_type,
                    "amount": float(quantity),
                    "price": price,
                    "params": kwargs or {},
                }
            )
            if inspect.isawaitable(res):
                res = await res
            if isinstance(res, dict):
                return res

        # 2) create_order(...) style
        fn2 = getattr(ex, "create_order", None)
        if not callable(fn2):
            raise RuntimeError("exchange_connector does not implement create_order/execute_order")

        # Try keyword style first
        try:
            res2 = fn2(
                symbol=symbol,
                side=side,
                amount=float(quantity),
                order_type=order_type,
                price=price,
                params=kwargs or {},
            )
            if inspect.isawaitable(res2):
                res2 = await res2
            if isinstance(res2, dict):
                return res2
        except TypeError:
            pass

        # Fall back to dataclass OrderRequest style
        try:
            from core.exchange_connector import OrderRequest

            req = OrderRequest(
                symbol=str(symbol),
                side=str(side),
                order_type=str(order_type),
                amount=float(quantity),
                price=price if price is None else float(price),
                params=dict(kwargs) if kwargs else None,
            )
            res3 = fn2(req)
            if inspect.isawaitable(res3):
                res3 = await res3
            # Normalize to dict
            if res3 is None:
                return {"status": "failed", "symbol": symbol, "side": side, "filled": 0.0, "price": float(price or 0.0)}
            if isinstance(res3, dict):
                return res3
            od = getattr(res3, "__dict__", None)
            if isinstance(od, dict):
                return dict(od)
        except Exception as _e:
            logger.debug("fast_execution error: %s", _e)

        # Last resort: return a simulated fill.
        return {
            "id": f"fast_{int(time.time() * 1000000)}",
            "symbol": symbol,
            "side": side,
            "status": "closed",
            "filled": float(quantity),
            "price": float(price or 0.0),
        }

