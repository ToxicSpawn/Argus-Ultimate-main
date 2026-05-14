"""Realistic fill simulation using queue position and market impact."""
# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.backtest.hft_simulator.latency_model import LatencyModel, LatencySample
from core.backtest.hft_simulator.market_impact import MarketImpactEstimate, MarketImpactModel
from core.backtest.hft_simulator.order_book_l3 import BookExecution, L3OrderBook, Side
from core.backtest.hft_simulator.queue_position import FillProbabilityEstimate, QueuePositionModel


@dataclass(slots=True)
class FillResult:
    order_id: str
    side: Side
    requested_quantity: float
    filled_quantity: float
    remaining_quantity: float
    average_price: float
    status: str
    is_maker: bool
    fill_probability: float
    queue_ahead: float
    slippage_bps: float
    latency_us: float
    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0
    executions: list[dict[str, float | str]] = field(default_factory=list)


@dataclass
class FillSimulator:
    queue_model: QueuePositionModel
    latency_model: LatencyModel
    impact_model: MarketImpactModel

    def simulate_limit_fill(
        self,
        book: L3OrderBook,
        order_id: str,
        side: Side,
        price: float,
        quantity: float,
        traded_quantity: float,
        time_remaining_s: float,
        timestamp_ns: int,
        volatility: float = 0.0,
    ) -> FillResult:
        position = self.queue_model.sync_with_book(book, order_id, timestamp_ns)
        if position is None:
            return FillResult(order_id, side, quantity, 0.0, quantity, price, "unknown", True, 0.0, 0.0, 0.0, 0.0)
        estimate: FillProbabilityEstimate = self.queue_model.estimate_fill(order_id, traded_quantity, time_remaining_s, volatility)
        fill_qty = float(min(quantity, estimate.expected_fill_quantity))
        latency: LatencySample = self.latency_model.sample(estimate.queue_ahead)
        execution = None if fill_qty <= 0 else book.execute_order(order_id, fill_qty, timestamp_ns)
        actual_filled = 0.0 if execution is None else float(execution.quantity)
        if actual_filled > 0:
            self.queue_model.record_fill(order_id, actual_filled, timestamp_ns)
            if traded_quantity > 0:
                self.queue_model.record_level_depletion(min(traded_quantity, estimate.queue_ahead + actual_filled))
        remaining = float(max(quantity - actual_filled, 0.0))
        status = "filled" if remaining <= 1e-12 else ("partial" if actual_filled > 0 else "live")
        return FillResult(
            order_id=order_id,
            side=side,
            requested_quantity=float(quantity),
            filled_quantity=actual_filled,
            remaining_quantity=remaining,
            average_price=float(price if actual_filled > 0 else 0.0),
            status=status,
            is_maker=True,
            fill_probability=estimate.probability,
            queue_ahead=estimate.queue_ahead,
            slippage_bps=0.0,
            latency_us=latency.total_us,
            executions=[] if execution is None else [self._execution_dict(execution)],
        )

    def simulate_market_fill(
        self,
        book: L3OrderBook,
        order_id: str,
        side: Side,
        quantity: float,
        timestamp_ns: int,
        daily_volume: float,
        volatility: float = 0.0,
    ) -> FillResult:
        reference_price = book.mid_price() or book.last_trade_price or 0.0
        visible_depth = book.total_depth("sell" if side == "buy" else "buy", max_levels=5)
        executions = book.sweep_market_order(side, quantity, timestamp_ns)
        raw_filled = float(np.sum([execution.quantity for execution in executions], dtype=float)) if executions else 0.0
        raw_notional = float(np.sum([execution.price * execution.quantity for execution in executions], dtype=float)) if executions else 0.0
        raw_average = raw_notional / raw_filled if raw_filled > 0 else 0.0
        impact: MarketImpactEstimate
        impacted_price, impact = self.impact_model.apply_to_price(
            raw_average if raw_average > 0 else reference_price,
            side,
            raw_filled,
            visible_depth,
            daily_volume,
            volatility,
        )
        latency = self.latency_model.sample(0.0)
        remaining = float(max(quantity - raw_filled, 0.0))
        status = "filled" if remaining <= 1e-12 else ("partial" if raw_filled > 0 else "rejected")
        slippage_bps = 0.0
        if reference_price > 0 and impacted_price > 0:
            direction = 1.0 if side == "buy" else -1.0
            slippage_bps = float(direction * (impacted_price - reference_price) / reference_price * 10_000.0)
        return FillResult(
            order_id=order_id,
            side=side,
            requested_quantity=float(quantity),
            filled_quantity=raw_filled,
            remaining_quantity=remaining,
            average_price=float(impacted_price if raw_filled > 0 else 0.0),
            status=status,
            is_maker=False,
            fill_probability=1.0 if raw_filled > 0 else 0.0,
            queue_ahead=0.0,
            slippage_bps=slippage_bps,
            latency_us=latency.total_us,
            temporary_impact_bps=impact.temporary_bps,
            permanent_impact_bps=impact.permanent_bps,
            executions=[self._execution_dict(execution) for execution in executions],
        )

    @staticmethod
    def _execution_dict(execution: BookExecution) -> dict[str, float | str]:
        return {
            "resting_order_id": execution.resting_order_id,
            "resting_owner": execution.resting_owner,
            "price": execution.price,
            "quantity": execution.quantity,
            "timestamp_ns": float(execution.timestamp_ns),
        }
