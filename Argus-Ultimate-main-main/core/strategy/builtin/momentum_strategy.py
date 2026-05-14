"""MomentumStrategy — built-in example strategy — Push 58.

Logic:
  - Maintains a rolling price window of length `window` (default 20).
  - Computes SMA of the window.
  - Emits BUY signal when price > SMA * (1 + threshold).
  - Emits SELL signal when price < SMA * (1 - threshold).
  - Submits orders via ExecutionEngine if attached.

Parameters (set via set_param or constructor kwargs)::

    window      int     rolling window length (default 20)
    threshold   float   signal threshold fraction (default 0.002)
    qty         float   order quantity (default 0.01)
    symbol      str     primary symbol to trade (default 'BTCUSDT')
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Deque, Optional

from core.strategy.base_strategy import AbstractStrategy, StrategyMetadata, StrategyState


class MomentumStrategy(AbstractStrategy):
    """Simple SMA momentum strategy."""

    def __init__(
        self,
        window: int = 20,
        threshold: float = 0.002,
        qty: float = 0.01,
        symbol: str = "BTCUSDT",
        engine: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.set_param("window", window)
        self.set_param("threshold", threshold)
        self.set_param("qty", qty)
        self.set_param("symbol", symbol)
        self._engine = engine
        self._prices: Deque[float] = deque(maxlen=window)
        self._position: int = 0  # +1 long, -1 short, 0 flat
        self._signals_emitted = 0

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="MomentumStrategy",
            version="1.0.0",
            symbols=[self.get_param("symbol", "BTCUSDT")],
            description="SMA momentum strategy with configurable window and threshold",
            author="Argus",
            tags=["momentum", "sma", "trend-following"],
        )

    async def on_start(self) -> None:
        self._prices.clear()
        self._position = 0
        self._signals_emitted = 0

    async def on_stop(self) -> None:
        pass

    async def on_tick(
        self,
        symbol: str,
        price: float,
        bid: float = 0.0,
        ask: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if symbol != self.get_param("symbol"):
            return

        self._prices.append(price)
        window = self.get_param("window", 20)
        if len(self._prices) < window:
            return

        sma = sum(self._prices) / len(self._prices)
        threshold = self.get_param("threshold", 0.002)
        qty = self.get_param("qty", 0.01)

        if price > sma * (1 + threshold) and self._position <= 0:
            self._position = 1
            self._signals_emitted += 1
            if self._engine is not None:
                from core.execution.order_models import Order, OrderSide, OrderType
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    qty=qty,
                )
                await self._engine.submit_order(order, mid_price=price, confidence=0.7)

        elif price < sma * (1 - threshold) and self._position >= 0:
            self._position = -1
            self._signals_emitted += 1
            if self._engine is not None:
                from core.execution.order_models import Order, OrderSide, OrderType
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    qty=qty,
                )
                await self._engine.submit_order(order, mid_price=price, confidence=0.7)

    async def on_fill(self, order: Any, fill: Any) -> None:
        pass

    @property
    def signals_emitted(self) -> int:
        return self._signals_emitted

    @property
    def current_sma(self) -> Optional[float]:
        if not self._prices:
            return None
        return sum(self._prices) / len(self._prices)
