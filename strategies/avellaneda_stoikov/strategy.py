"""Argus integration for Avellaneda-Stoikov market making."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from core.strategy.base_strategy import BaseStrategy, StrategyConfig
from core.strategy.signal import Signal, SignalSide

from .market_maker import AvellanedaStoikovMarketMaker, MarketMakerConfig, MarketSnapshot, OptimalQuote
from .order_scheduler import ManagedOrder, OrderScheduler, OrderSchedulerConfig
from .profit_calculator import FillRecord, ProfitCalculator

logger = logging.getLogger(__name__)


class AvellanedaStoikovStrategy(BaseStrategy):
    """Canonical Argus strategy wrapper around Avellaneda-Stoikov quoting."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        params = config.params
        self.market_maker = AvellanedaStoikovMarketMaker(
            MarketMakerConfig(
                symbol=config.symbol,
                gamma=float(params.get("gamma", 0.1)),
                kappa=float(params.get("kappa", 1.5)),
                sigma_floor=float(params.get("sigma_floor", 1e-4)),
                horizon_seconds=float(params.get("horizon_seconds", 300.0)),
                min_spread_bps=float(params.get("min_spread_bps", 4.0)),
                max_spread_bps=float(params.get("max_spread_bps", 80.0)),
                base_order_size=float(params.get("base_order_size", 0.01)),
                min_order_size=float(params.get("min_order_size", 0.001)),
                max_order_size=float(params.get("max_order_size", 1.0)),
                inventory_target=float(params.get("inventory_target", 0.0)),
                max_inventory=float(params.get("max_inventory", 5.0)),
                order_size_decay=float(params.get("order_size_decay", 0.75)),
                tick_size=float(params.get("tick_size", 0.01)),
                use_injective_reservation_adjustment=bool(params.get("use_injective_reservation_adjustment", True)),
                volatility_window=int(params.get("volatility_window", 120)),
                ewma_lambda=float(params.get("ewma_lambda", 0.94)),
                inventory_soft_limit_fraction=float(params.get("inventory_soft_limit_fraction", 0.7)),
                max_skew_bps=float(params.get("max_skew_bps", 20.0)),
            )
        )
        self.scheduler = OrderScheduler(
            OrderSchedulerConfig(
                refresh_interval_seconds=float(params.get("refresh_interval_seconds", 1.0)),
                stale_after_seconds=float(params.get("stale_after_seconds", 5.0)),
                replace_price_tolerance_bps=float(params.get("replace_price_tolerance_bps", 2.0)),
            ),
            self.market_maker.inventory_manager,
        )
        self.profit_calculator = ProfitCalculator()
        self.latest_quote: Optional[OptimalQuote] = None

    def _coerce_snapshot(self, data: Any) -> MarketSnapshot:
        if isinstance(data, dict):
            best_bid = float(data.get("bid", data.get("best_bid", data.get("price", 0.0))))
            best_ask = float(data.get("ask", data.get("best_ask", data.get("price", 0.0))))
            mid = float(data.get("mid_price", (best_bid + best_ask) / 2.0 if best_bid and best_ask else data.get("price", 0.0)))
            return MarketSnapshot(
                mid_price=mid,
                best_bid=best_bid or mid,
                best_ask=best_ask or mid,
                timestamp=float(data.get("timestamp", time.time())),
                market_depth=float(data.get("market_depth", 1.0)),
                order_book_imbalance=float(data.get("order_book_imbalance", 0.0)),
                last_trade_price=float(data.get("price", mid)),
                volume=float(data.get("volume", 0.0)),
            )

        close = float(getattr(data, "close", getattr(data, "price", 0.0)))
        bid = float(getattr(data, "bid", close))
        ask = float(getattr(data, "ask", close))
        return MarketSnapshot(
            mid_price=float(getattr(data, "mid_price", (bid + ask) / 2.0 if bid and ask else close)),
            best_bid=bid,
            best_ask=ask,
            timestamp=float(getattr(data, "timestamp", time.time())),
            market_depth=float(getattr(data, "market_depth", 1.0)),
            order_book_imbalance=float(getattr(data, "order_book_imbalance", 0.0)),
            last_trade_price=close,
            volume=float(getattr(data, "volume", 0.0)),
        )

    def _build_inventory_signal(self, quote: OptimalQuote) -> Optional[Signal]:
        ratio = self.market_maker.inventory_manager.inventory_ratio()
        if abs(ratio) < self.market_maker.config.inventory_soft_limit_fraction:
            return None

        if ratio > 0:
            side = SignalSide.SHORT
            price = quote.ask_price
            strength = min(1.0, abs(ratio))
        else:
            side = SignalSide.LONG
            price = quote.bid_price
            strength = min(1.0, abs(ratio))

        return self._make_signal(
            side=side,
            strength=strength,
            price=price,
            order_type="Limit",
            bid_price=quote.bid_price,
            ask_price=quote.ask_price,
            reservation_price=quote.reservation_price,
            optimal_spread=quote.optimal_spread,
            inventory=quote.inventory,
            inventory_penalty=quote.inventory_penalty,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
        )

    def tick(self, price: float, volume: float = 0.0, timestamp: Optional[float] = None) -> Optional[Signal]:
        snapshot = MarketSnapshot(
            mid_price=price,
            best_bid=price,
            best_ask=price,
            timestamp=timestamp or time.time(),
            last_trade_price=price,
            volume=volume,
        )
        self.latest_quote = self.market_maker.generate_quotes(snapshot)
        return self._build_inventory_signal(self.latest_quote)

    def on_bar(self, bar: Any) -> Optional[Signal]:
        snapshot = self._coerce_snapshot(bar)
        self.latest_quote = self.market_maker.generate_quotes(snapshot)

        bid_order = ManagedOrder(side="buy", price=self.latest_quote.bid_price, quantity=self.latest_quote.bid_size, created_at=snapshot.timestamp)
        ask_order = ManagedOrder(side="sell", price=self.latest_quote.ask_price, quantity=self.latest_quote.ask_size, created_at=snapshot.timestamp)

        if self.scheduler.should_refresh(snapshot.timestamp):
            self.scheduler.register_orders([bid_order, ask_order])
        else:
            for action in self.scheduler.replacement_actions(bid_order, ask_order):
                if action.action in {"create", "replace"}:
                    self.scheduler.register_orders([action.order])

        return self._build_inventory_signal(self.latest_quote)

    def on_order(self, order_event: Any) -> None:
        if isinstance(order_event, dict):
            order_id = str(order_event.get("order_id", ""))
            status = str(order_event.get("status", "open"))
        else:
            order_id = str(getattr(order_event, "order_id", ""))
            status = str(getattr(order_event, "status", "open"))
        if order_id:
            self.scheduler.on_order_update(order_id, status)

    def on_fill(self, fill_event: Any) -> None:
        side = str(getattr(fill_event, "side", fill_event.get("side", "buy") if isinstance(fill_event, dict) else "buy")).lower()
        price = float(getattr(fill_event, "price", fill_event.get("price", 0.0) if isinstance(fill_event, dict) else 0.0))
        quantity = float(getattr(fill_event, "quantity", fill_event.get("quantity", fill_event.get("qty", 0.0)) if isinstance(fill_event, dict) else 0.0))
        order_id = str(getattr(fill_event, "order_id", fill_event.get("order_id", "") if isinstance(fill_event, dict) else ""))
        reference_mid = float(getattr(fill_event, "reference_mid", fill_event.get("reference_mid", price) if isinstance(fill_event, dict) else price))
        fee = float(getattr(fill_event, "fee", fill_event.get("fee", 0.0) if isinstance(fill_event, dict) else 0.0))

        if price <= 0 or quantity <= 0:
            logger.warning("Ignoring invalid fill event: %s", fill_event)
            return

        if order_id:
            self.scheduler.process_fill(order_id, side, quantity, price)
        else:
            self.market_maker.inventory_manager.update_fill(side, quantity, price)

        self.profit_calculator.process_fill(
            FillRecord(
                side=side,
                price=price,
                quantity=quantity,
                reference_mid=reference_mid,
                fee=fee,
            )
        )
        super().on_fill(fill_event)
