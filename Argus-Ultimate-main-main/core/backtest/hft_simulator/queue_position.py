"""Queue position tracking and practical fill estimation."""
# pyright: reportMissingImports=false

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

import numpy as np

from core.backtest.hft_simulator.order_book_l3 import L3OrderBook, Side


@dataclass(slots=True)
class QueuePosition:
    order_id: str
    side: Side
    price: float
    entered_at_ns: int
    initial_queue_ahead: float
    current_queue_ahead: float
    remaining_quantity: float
    realized_filled: float = 0.0
    queue_progress: float = 0.0
    last_update_ns: int = 0

    def update_queue(self, queue_ahead: float, timestamp_ns: int) -> None:
        self.current_queue_ahead = float(max(queue_ahead, 0.0))
        self.last_update_ns = timestamp_ns
        baseline = max(self.initial_queue_ahead, 1e-12)
        self.queue_progress = float(np.clip(1.0 - (self.current_queue_ahead / baseline), 0.0, 1.0))


@dataclass(slots=True)
class FillProbabilityEstimate:
    probability: float
    expected_fill_quantity: float
    queue_ahead: float


@dataclass
class QueuePositionModel:
    lookback: int = 4096
    positions: Dict[str, QueuePosition] = field(default_factory=dict)
    depletion_history: Deque[float] = field(init=False)
    fill_ratio_history: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.depletion_history = deque(maxlen=self.lookback)
        self.fill_ratio_history = deque(maxlen=self.lookback)

    def register_order(
        self,
        order_id: str,
        side: Side,
        price: float,
        queue_ahead: float,
        quantity: float,
        timestamp_ns: int,
    ) -> QueuePosition:
        position = QueuePosition(
            order_id=order_id,
            side=side,
            price=price,
            entered_at_ns=timestamp_ns,
            initial_queue_ahead=float(max(queue_ahead, 0.0)),
            current_queue_ahead=float(max(queue_ahead, 0.0)),
            remaining_quantity=float(max(quantity, 0.0)),
            last_update_ns=timestamp_ns,
        )
        self.positions[order_id] = position
        return position

    def sync_with_book(self, book: L3OrderBook, order_id: str, timestamp_ns: int) -> QueuePosition | None:
        position = self.positions.get(order_id)
        if position is None:
            return None
        queue_ahead = book.queue_ahead(position.side, position.price, order_id)
        position.update_queue(queue_ahead, timestamp_ns)
        return position

    def record_level_depletion(self, depleted_quantity: float) -> None:
        if depleted_quantity > 0:
            self.depletion_history.append(float(depleted_quantity))

    def record_fill(self, order_id: str, filled_quantity: float, timestamp_ns: int) -> None:
        position = self.positions.get(order_id)
        if position is None or filled_quantity <= 0:
            return
        position.realized_filled += float(filled_quantity)
        position.remaining_quantity = float(max(position.remaining_quantity - filled_quantity, 0.0))
        ratio = filled_quantity / max(position.realized_filled + position.remaining_quantity, 1e-12)
        self.fill_ratio_history.append(float(np.clip(ratio, 0.0, 1.0)))
        position.last_update_ns = timestamp_ns
        if position.remaining_quantity <= 1e-12:
            self.positions.pop(order_id, None)

    def cancel_order(self, order_id: str) -> None:
        self.positions.pop(order_id, None)

    def estimate_fill(
        self,
        order_id: str,
        trade_quantity: float,
        time_remaining_s: float,
        volatility: float = 0.0,
    ) -> FillProbabilityEstimate:
        position = self.positions.get(order_id)
        if position is None:
            return FillProbabilityEstimate(0.0, 0.0, 0.0)
        historical_depletion = float(np.mean(np.asarray(self.depletion_history, dtype=float))) if self.depletion_history else 0.0
        queue_ahead = max(position.current_queue_ahead - historical_depletion, 0.0)
        executable = max(trade_quantity - queue_ahead, 0.0)
        raw_fill_ratio = executable / max(position.remaining_quantity, 1e-12)
        pace_factor = 1.0 - np.exp(-max(time_remaining_s, 0.0))
        progress_factor = 0.35 + 0.65 * position.queue_progress
        volatility_penalty = 1.0 / (1.0 + max(volatility, 0.0))
        historical_ratio = float(np.mean(np.asarray(self.fill_ratio_history, dtype=float))) if self.fill_ratio_history else 0.5
        probability = float(np.clip((0.55 * raw_fill_ratio + 0.45 * historical_ratio) * pace_factor * progress_factor * volatility_penalty, 0.0, 1.0))
        expected_fill = float(min(position.remaining_quantity, max(executable, 0.0) * probability))
        return FillProbabilityEstimate(probability=probability, expected_fill_quantity=expected_fill, queue_ahead=queue_ahead)

    def queue_advantage(self, order_id: str) -> float:
        position = self.positions.get(order_id)
        return 0.0 if position is None else float(position.queue_progress)
