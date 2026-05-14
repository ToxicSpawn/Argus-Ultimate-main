"""Event-driven HFT backtest engine with queue-aware execution."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from core.backtest.hft_simulator.fill_simulation import FillResult, FillSimulator
from core.backtest.hft_simulator.latency_model import LatencyModel
from core.backtest.hft_simulator.market_impact import MarketImpactModel
from core.backtest.hft_simulator.metrics import HFTMetrics, compute_hft_metrics
from core.backtest.hft_simulator.order_book_l3 import L3Order, L3OrderBook, Side
from core.backtest.hft_simulator.queue_position import QueuePositionModel

logger = logging.getLogger(__name__)

EventType = Literal["add", "cancel", "modify", "trade"]
OrderType = Literal["limit", "market"]


@dataclass(slots=True)
class MarketEvent:
    event_type: EventType
    timestamp_ns: int
    side: Side
    price: float
    quantity: float
    order_id: str = ""
    owner: str = "market"


@dataclass(slots=True)
class OrderRequest:
    side: Side
    quantity: float
    order_type: OrderType = "limit"
    price: float | None = None
    time_in_force_s: float = 1.0


@dataclass(slots=True)
class LiveOrder:
    order_id: str
    side: Side
    price: float
    quantity: float
    remaining_quantity: float
    submit_timestamp_ns: int
    expiry_timestamp_ns: int


@dataclass
class HFTBacktestConfig:
    symbol: str
    initial_cash: float = 100_000.0
    maker_fee_bps: float = -0.10
    taker_fee_bps: float = 0.50
    tick_size: float = 0.01
    max_position: float = 25.0
    daily_volume: float = 1_000_000.0
    latency_model: LatencyModel = field(default_factory=LatencyModel)
    impact_model: MarketImpactModel = field(default_factory=MarketImpactModel)
    queue_lookback: int = 2048


@dataclass(slots=True)
class HFTBacktestResult:
    equity_curve: list[float]
    fill_records: list[dict[str, float | str | bool]]
    metrics: HFTMetrics
    final_cash: float
    final_position: float


StrategyCallback = Callable[[L3OrderBook, MarketEvent, "HFTBacktestEngine"], Sequence[OrderRequest]]


class HFTBacktestEngine:
    def __init__(self, config: HFTBacktestConfig) -> None:
        self.config: HFTBacktestConfig = config
        self.book: L3OrderBook = L3OrderBook(symbol=config.symbol)
        self.queue_model: QueuePositionModel = QueuePositionModel(lookback=config.queue_lookback)
        self.fill_simulator: FillSimulator = FillSimulator(
            queue_model=self.queue_model,
            latency_model=config.latency_model,
            impact_model=config.impact_model,
        )
        self.reset()

    def reset(self) -> None:
        self.cash: float = float(self.config.initial_cash)
        self.position: float = 0.0
        self.order_counter: int = 0
        self.live_orders: dict[str, LiveOrder] = {}
        self.fill_records: list[dict[str, float | str | bool]] = []
        self.equity_curve: list[float] = [self.config.initial_cash]

    def run(self, market_events: Sequence[MarketEvent], strategy: StrategyCallback | None = None) -> HFTBacktestResult:
        for event in market_events:
            self.process_market_event(event)
            if strategy is not None:
                requests = strategy(self.book, event, self)
                for request in requests:
                    _ = self.submit_order(request, event.timestamp_ns)
            self._process_live_orders(event)
            self._mark_to_market(self.book.mid_price() or self.book.last_trade_price or event.price)
        metrics = compute_hft_metrics(self.fill_records, self.equity_curve)
        return HFTBacktestResult(
            equity_curve=self.equity_curve,
            fill_records=self.fill_records,
            metrics=metrics,
            final_cash=float(self.cash),
            final_position=float(self.position),
        )

    def process_market_event(self, event: MarketEvent) -> None:
        if event.event_type == "add":
            self.book.add_order(
                L3Order(
                    order_id=event.order_id,
                    side=event.side,
                    price=event.price,
                    quantity=event.quantity,
                    timestamp_ns=event.timestamp_ns,
                    owner=event.owner,
                )
            )
        elif event.event_type == "cancel":
            _ = self.book.cancel_order(event.order_id, event.quantity if event.quantity > 0 else None)
        elif event.event_type == "modify":
            _ = self.book.modify_order(event.order_id, event.quantity)
        elif event.event_type == "trade":
            _ = self.book.apply_trade(event.side, event.price, event.quantity, event.timestamp_ns)

    def submit_order(self, request: OrderRequest, timestamp_ns: int) -> FillResult:
        signed_quantity = request.quantity if request.side == "buy" else -request.quantity
        if abs(self.position + signed_quantity) > self.config.max_position:
            raise ValueError("max_position exceeded")
        self.order_counter += 1
        order_id = f"hft-{self.order_counter}"
        if request.order_type == "market":
            fill = self.fill_simulator.simulate_market_fill(
                self.book,
                order_id,
                request.side,
                request.quantity,
                timestamp_ns,
                self.config.daily_volume,
            )
            self._apply_fill(fill)
            return fill
        price = float(request.price if request.price is not None else self._default_limit_price(request.side))
        arrival_ns = self.config.latency_model.estimate_order_arrival_ns(timestamp_ns, self.book.queue_ahead(request.side, price))
        order = L3Order(order_id=order_id, side=request.side, price=price, quantity=request.quantity, timestamp_ns=arrival_ns, owner="strategy")
        queue_ahead = self.book.queue_ahead(request.side, price)
        self.book.add_order(order)
        _ = self.queue_model.register_order(order_id, request.side, price, queue_ahead, request.quantity, arrival_ns)
        live_order = LiveOrder(
            order_id=order_id,
            side=request.side,
            price=price,
            quantity=request.quantity,
            remaining_quantity=request.quantity,
            submit_timestamp_ns=arrival_ns,
            expiry_timestamp_ns=int(arrival_ns + request.time_in_force_s * 1_000_000_000),
        )
        self.live_orders[order_id] = live_order
        return FillResult(order_id, request.side, request.quantity, 0.0, request.quantity, price, "live", True, 0.0, queue_ahead, 0.0, 0.0)

    def cancel_order(self, order_id: str) -> None:
        live_order = self.live_orders.pop(order_id, None)
        if live_order is None:
            return
        _ = self.book.cancel_order(order_id)
        self.queue_model.cancel_order(order_id)

    def metrics(self) -> HFTMetrics:
        return compute_hft_metrics(self.fill_records, self.equity_curve)

    def _process_live_orders(self, event: MarketEvent) -> None:
        for order_id, live_order in list(self.live_orders.items()):
            if event.timestamp_ns < live_order.submit_timestamp_ns:
                continue
            if event.timestamp_ns >= live_order.expiry_timestamp_ns:
                self.cancel_order(order_id)
                continue
            if event.event_type != "trade":
                _ = self.queue_model.sync_with_book(self.book, order_id, event.timestamp_ns)
                continue
            if not self._trade_can_reach_order(live_order, event):
                continue
            fill = self.fill_simulator.simulate_limit_fill(
                self.book,
                order_id,
                live_order.side,
                live_order.price,
                live_order.remaining_quantity,
                event.quantity,
                (live_order.expiry_timestamp_ns - event.timestamp_ns) / 1_000_000_000.0,
                event.timestamp_ns,
            )
            if fill.filled_quantity <= 0:
                continue
            live_order.remaining_quantity = fill.remaining_quantity
            self._apply_fill(fill)
            if live_order.remaining_quantity <= 1e-12:
                _ = self.live_orders.pop(order_id, None)

    def _trade_can_reach_order(self, live_order: LiveOrder, event: MarketEvent) -> bool:
        if live_order.side == "buy":
            return event.side == "sell" and event.price <= live_order.price + 1e-12
        return event.side == "buy" and event.price >= live_order.price - 1e-12

    def _apply_fill(self, fill: FillResult) -> None:
        if fill.filled_quantity <= 0:
            return
        signed_qty = fill.filled_quantity if fill.side == "buy" else -fill.filled_quantity
        fee_bps = self.config.maker_fee_bps if fill.is_maker else self.config.taker_fee_bps
        fee = fill.average_price * fill.filled_quantity * fee_bps / 10_000.0
        self.position += signed_qty
        self.cash -= signed_qty * fill.average_price + fee
        post_trade_mid = self.book.mid_price() or self.book.last_trade_price or fill.average_price
        self.fill_records.append(
            {
                "order_id": fill.order_id,
                "side": fill.side,
                "requested_quantity": fill.requested_quantity,
                "filled_quantity": fill.filled_quantity,
                "average_price": fill.average_price,
                "latency_us": fill.latency_us,
                "slippage_bps": fill.slippage_bps,
                "queue_ahead": fill.queue_ahead,
                "is_maker": fill.is_maker,
                "post_trade_mid": post_trade_mid,
                "temporary_impact_bps": fill.temporary_impact_bps,
                "permanent_impact_bps": fill.permanent_impact_bps,
            }
        )

    def _default_limit_price(self, side: Side) -> float:
        if side == "buy":
            return self.book.best_bid() or self.book.mid_price() or self.config.tick_size
        return self.book.best_ask() or self.book.mid_price() or self.config.tick_size

    def _mark_to_market(self, reference_price: float) -> None:
        equity = self.cash + self.position * reference_price
        self.equity_curve.append(float(equity))
