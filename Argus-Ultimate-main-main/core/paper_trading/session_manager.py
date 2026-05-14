"""Push 70 — PaperTradingSession: full session orchestrator.

Wires together:
  AsyncWebSocketFeed   (price feed)
  PaperTrader          (simulated fills)
  RealTimePnLTracker   (PnL + drawdown)
  ArgusMetrics         (Prometheus export, optional)
  AlertManager         (alerts, optional)

Lifecycle:
  session.start()   — connects feed, starts eval loop
  session.stop()    — graceful shutdown
  session.snapshot() — returns current state dict
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from core.paper_trading.paper_trader import PaperTrader, OrderSide, FillEvent
from core.paper_trading.pnl_tracker import RealTimePnLTracker
from core.paper_trading.ws_feed import AsyncWebSocketFeed, FeedConfig
from core.paper_trading.reconnect import ReconnectPolicy


@dataclass
class SessionConfig:
    symbol: str = "BTCUSDT"
    initial_equity: float = 10_000.0
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    snapshot_interval_secs: float = 60.0
    feed: FeedConfig = field(default_factory=FeedConfig)
    reconnect: ReconnectPolicy = field(default_factory=ReconnectPolicy)


class PaperTradingSession:
    """Full paper trading session orchestrator.

    Args:
        config:          SessionConfig
        strategy_fn:     async callable(price: float) -> Optional[signal]
                         signal: +1=buy, -1=sell, 0=flat, None=no action
        on_fill:         Optional callback on each fill
        metrics:         Optional ArgusMetrics instance
        alert_manager:   Optional AlertManager instance
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        strategy_fn: Optional[Callable] = None,
        on_fill: Optional[Callable[[FillEvent], None]] = None,
        metrics=None,
        alert_manager=None,
    ):
        self.cfg = config or SessionConfig()
        self.strategy_fn = strategy_fn
        self._metrics = metrics
        self._alert_manager = alert_manager

        self.trader = PaperTrader(
            initial_cash=self.cfg.initial_equity,
            commission_bps=self.cfg.commission_bps,
            slippage_bps=self.cfg.slippage_bps,
            on_fill=self._on_fill_internal,
        )
        self.pnl = RealTimePnLTracker(
            initial_equity=self.cfg.initial_equity,
            snapshot_interval_secs=self.cfg.snapshot_interval_secs,
        )
        self.feed = AsyncWebSocketFeed(
            config=self.cfg.feed,
            on_message=self._on_raw_message,
            reconnect=self.cfg.reconnect,
        )

        self._user_on_fill = on_fill
        self._running = False
        self._price_callbacks: Dict[str, list] = {}
        self._last_price: float = 0.0
        self._bar_count: int = 0
        self._session_start: float = 0.0

    async def start(self) -> None:
        self._running = True
        self._session_start = time.time()
        await self.feed.start()

    async def stop(self) -> None:
        self._running = False
        await self.feed.stop()

    def inject_price(self, symbol: str, price: float) -> None:
        """Manually inject a price tick (used in tests / replay mode)."""
        self._last_price = price
        self._bar_count += 1
        fills = self.trader.on_price_tick(symbol, price)
        equity = self.trader.equity({symbol: price})
        self.pnl.update_equity(equity)

        # Update unrealised PnL for open positions
        pos = self.trader.positions.get(symbol)
        if pos:
            self.pnl.update_mark_price(
                symbol, price, pos.qty, pos.avg_entry, pos.side
            )

        # Push to Prometheus if available
        if self._metrics:
            self._metrics.update_risk_snapshot(
                halted=False,
                drawdown_pct=self.pnl.current_drawdown_pct,
                cvar_95=0.0, cvar_99=0.0,
                daily_pnl=self.pnl.daily_pnl,
                equity=equity,
                position_count=len(self.trader.positions),
            )

    async def _on_raw_message(self, raw: str) -> None:
        """Parse raw WS message and extract price."""
        try:
            import json
            data = json.loads(raw)
            # Bybit linear format: data.data[0].lastPrice
            price = None
            if "data" in data and isinstance(data["data"], list):
                price = float(data["data"][0].get("lastPrice", 0))
            elif "p" in data:  # Binance trade format
                price = float(data["p"])
            if price and price > 0:
                self.inject_price(self.cfg.symbol, price)
                if self.strategy_fn:
                    if asyncio.iscoroutinefunction(self.strategy_fn):
                        signal = await self.strategy_fn(price)
                    else:
                        signal = self.strategy_fn(price)
                    await self._act_on_signal(signal, price)
        except Exception:
            pass

    async def _act_on_signal(self, signal: Optional[float], price: float) -> None:
        if signal is None or signal == 0:
            return
        symbol = self.cfg.symbol
        equity = self.trader.equity({symbol: price})
        # Size: 1% of equity per trade
        notional = equity * 0.01
        qty = notional / max(price, 1e-9)
        if signal > 0:
            self.trader.place_market_order(symbol, OrderSide.BUY, qty, price)
        elif signal < 0:
            self.trader.place_market_order(symbol, OrderSide.SELL, qty, price)

    def _on_fill_internal(self, event: FillEvent) -> None:
        self.pnl.record_fill(
            symbol=event.symbol,
            side=event.side.value,
            qty=event.qty,
            fill_price=event.fill_price,
            pnl=event.pnl,
            commission=event.commission,
        )
        if self._user_on_fill:
            self._user_on_fill(event)

    def snapshot(self) -> dict:
        """Return current session state as a dict."""
        return {
            "symbol": self.cfg.symbol,
            "session_uptime_secs": time.time() - self._session_start if self._session_start else 0,
            "bar_count": self._bar_count,
            "equity": self.trader.equity(),
            "cash": self.trader.cash,
            "n_positions": len(self.trader.positions),
            "n_fills": self.trader.n_fills,
            "realised_pnl": self.pnl.realised_pnl,
            "unrealised_pnl": self.pnl.unrealised_pnl,
            "total_pnl": self.pnl.total_pnl,
            "daily_pnl": self.pnl.daily_pnl,
            "current_drawdown_pct": self.pnl.current_drawdown_pct,
            "max_drawdown_pct": self.pnl.max_drawdown_pct,
            "win_rate": self.pnl.win_rate,
            "feed_state": self.feed.state.value,
            "feed_reconnects": self.feed.total_reconnects,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_price(self) -> float:
        return self._last_price
