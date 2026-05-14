"""
core/backtest/hft_backtest.py
===============================
Event-driven HFT backtester with L3 queue-position modeling.

Provides:
  - L3 order-book simulation (tick-by-tick)
  - Queue-position-aware fill probability
  - Adverse selection and market-impact modeling
  - Realistic latency modeling
  - Full HFT metrics (PnL, queue-utilisation, adverse selection ratio)

Designed to be run from the same entry points as the live HFT engine,
but driven by historical L3 message logs or synthetic order-book replay.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT  = "limit"
    MARKET = "market"
    CANCEL = "cancel"


class FillStatus(Enum):
    FILLED        = "filled"
    PARTIAL       = "partial"
    CANCELLED     = "cancelled"
    REJECTED      = "rejected"
    LIVE          = "live"


@dataclass
class HFTOrder:
    order_id       : str
    symbol         : str
    side           : OrderSide
    order_type     : OrderType
    price          : float
    size           : float
    filled        : float = 0.0
    avg_fill_price : float = 0.0
    status         : FillStatus = FillStatus.LIVE
    submitted_at   : float = field(default_factory=time.time)
    filled_at      : float = 0.0
    latency_ms     : float = 0.0


@dataclass
class QueuedOrder:
    """One order in the exchange's L3 queue."""
    order_id    : str
    side        : OrderSide
    price       : float
    size        : float
    position    : int          # 0 = first in queue
    submitted_at: float
    is_ours     : bool = False


@dataclass
class L3Tick:
    """One L3 message."""
    timestamp   : float
    type_       : str           # "trade", "cancel", "modify", "add"
    side        : OrderSide
    price       : float
    size        : float
    order_id    : str
    maker_order_id: Optional[str] = None   # for trades


@dataclass
class BacktestStats:
    """Aggregate statistics from an HFT backtest run."""
    total_pnl         : float
    pnl_std           : float
    max_drawdown      : float
    sharpe            : float
    sortino           : float
    total_trades      : int
    filled_trades     : int
    cancelled_trades  : int
    avg_queue_position: float
    avg_fill_latency_ms: float
    adverse_selection_ratio: float
    market_impact_bps : float
    queue_utilisation : float
    win_rate          : float
    profit_factor     : float
    avg_trade_pnl     : float
    total_volume      : float


@dataclass
class TradeRecord:
    """One backtest trade."""
    order_id       : str
    side           : OrderSide
    entry_price    : float
    exit_price     : float
    size           : float
    pnl            : float
    queue_position : int
    fill_latency_ms: float
    adverse_sel_bps: float
    timestamp      : float


# ---------------------------------------------------------------------------
# Fill probability model
# ---------------------------------------------------------------------------

class FillProbabilityModel:
    """
    Models fill probability as a function of:
      - queue position (more ahead = lower fill probability)
      - order book pressure (imbalance)
      - spread (wider spread = harder to cross)
      - volatility (high vol = more queue depletion)
    """

    def __init__(
        self,
        base_fill_rate : float = 0.05,   # fraction of visible size filled per tick
        queue_decay    : float = 0.90,   # each position reduces fill prob by 10 %
        min_fill_prob  : float = 0.001,
    ) -> None:
        self._base_fill_rate = base_fill_rate
        self._queue_decay    = queue_decay
        self._min_fill_prob  = min_fill_prob

    def fill_probability(
        self,
        queue_position: int,
        book_imbalance: float,   # -1 (heavy sell) to +1 (heavy buy)
        spread_bps    : float,
        volatility    : float,
        side          : OrderSide,
    ) -> float:
        """
        Return probability [0, 1] of being filled in the next tick.
        """
        # Base prob decays exponentially with queue depth
        base_prob = self._base_fill_rate * (self._queue_decay ** max(0, queue_position))

        # Imbalance adjustment: if book is heavily one-sided,
        # the contra side has lower fill probability
        imbalance_penalty = 1.0 - (abs(book_imbalance) * 0.3)
        if side == OrderSide.BUY and book_imbalance < -0.3:
            imbalance_penalty *= 0.5
        elif side == OrderSide.SELL and book_imbalance > 0.3:
            imbalance_penalty *= 0.5

        # Spread penalty: wide spreads make crossing harder
        spread_penalty = 1.0 / (1.0 + spread_bps / 100.0)

        # Volatility adjustment: high vol = more queue depletion = harder to fill
        vol_penalty = 1.0 / (1.0 + volatility)

        prob = base_prob * imbalance_penalty * spread_penalty * vol_penalty
        return max(self._min_fill_prob, min(1.0, prob))


# ---------------------------------------------------------------------------
# Market impact model
# ---------------------------------------------------------------------------

class MarketImpactModel:
    """
    Kyle (1985) lambda-based market impact model.

    impact_bps = lambda * |order_flow| / (2 * avg_daily_volume / ticks_per_day)
    """

    def __init__(
        self,
        kyle_lambda : float = 0.5,
        half_spread_bps: float = 2.0,
    ) -> None:
        self._kyle_lambda    = kyle_lambda
        self._half_spread    = half_spread_bps / 10000.0

    def impact_bps(
        self,
        order_flow  : float,   # signed: +buy volume, -sell volume
        daily_volume: float,
        ticks_per_day: int = 86400,
    ) -> float:
        """Return expected market impact in basis points."""
        normalized_flow = order_flow / (daily_volume / ticks_per_day)
        return float(self._kyle_lambda * abs(normalized_flow) * 10000)

    def spread_cost_bps(self, side: OrderSide) -> float:
        """Half-spread cost in bps."""
        return self._half_spread * 10000


# ---------------------------------------------------------------------------
# HFT Backtest Engine
# ---------------------------------------------------------------------------

class HFTBacktestEngine:
    """
    Event-driven HFT backtester.

    Parameters
    ----------
    symbol          : str
    initial_equity  : float   — starting capital in quote currency
    maker_fee_bps   : float   — maker fee (rebate) in bps  (e.g. 2.0)
    taker_fee_bps   : float   — taker fee in bps              (e.g. 7.0)
    max_position    : float   — max notional position
    fill_model      : FillProbabilityModel
    impact_model    : MarketImpactModel
    latency_ms      : float   — simulated round-trip latency
    """

    def __init__(
        self,
        symbol         : str,
        initial_equity : float = 1_000_000.0,
        maker_fee_bps  : float = 2.0,
        taker_fee_bps  : float = 7.0,
        max_position   : float = 50_000.0,
        fill_model     : Optional[FillProbabilityModel] = None,
        impact_model   : Optional[MarketImpactModel]   = None,
        latency_ms     : float = 50.0,
    ) -> None:
        self.symbol        = symbol
        self.equity        = initial_equity
        self.initial_equity= initial_equity
        self.maker_fee     = maker_fee_bps / 10000.0
        self.taker_fee    = taker_fee_bps / 10000.0
        self.max_position  = max_position
        self.fill_model    = fill_model   or FillProbabilityModel()
        self.impact_model  = impact_model or MarketImpactModel()
        self.latency_ms    = latency_ms

        # Order tracking
        self._orders      : Dict[str, HFTOrder]  = {}
        self._order_id_ctr: int = 0

        # L3 queue simulation
        self._queue_bid: Deque[QueuedOrder] = deque()   # sorted: best bid first
        self._queue_ask: Deque[QueuedOrder] = deque()

        # Market state
        self._mid_price   : float = 0.0
        self._best_bid    : float = 0.0
        self._best_ask    : float = 0.0
        self._book_imbal   : float = 0.0
        self._spread_bps   : float = 0.0
        self._volatility   : float = 0.0
        self._daily_volume : float = 0.0
        self._tick_count   : int   = 0

        # PnL tracking
        self._equity_curve: Deque[float] = deque(maxlen=100_000)
        self._trades      : List[TradeRecord] = []
        self._fill_latencies: Deque[float] = deque(maxlen=10_000)

        # Position
        self._position   : float = 0.0    # positive = long
        self._entry_price: float = 0.0
        self._position_pnl: float = 0.0

        # Callbacks for strategy
        self._on_market_update: Optional[Callable[[float, float, float, float], None]] = None
        self._on_fill        : Optional[Callable[[HFTOrder, float], None]] = None

        logger.info(
            "HFTBacktestEngine initialised | equity=%.0f | latency=%.1fms",
            initial_equity, latency_ms,
        )

    # ------------------------------------------------------------------ Config

    def set_market_state(
        self,
        bid    : float,
        ask    : float,
        imbal  : float,
        vol    : float,
        daily_vol: float,
    ) -> None:
        """Called by the driver to update the simulated market state."""
        self._best_bid     = bid
        self._best_ask     = ask
        self._mid_price    = (bid + ask) / 2.0
        self._book_imbal   = max(-1.0, min(1.0, imbal))
        self._spread_bps   = ((ask - bid) / self._mid_price * 10000) if self._mid_price > 0 else 0.0
        self._volatility   = vol
        self._daily_volume = daily_vol
        self._tick_count  += 1

    def set_callbacks(
        self,
        on_market_update: Optional[Callable[[float, float, float, float], None]] = None,
        on_fill         : Optional[Callable[[HFTOrder, float], None]] = None,
    ) -> None:
        """Set strategy event callbacks."""
        self._on_market_update = on_market_update
        self._on_fill         = on_fill

    # ------------------------------------------------------------------ Order entry

    def submit_limit(
        self,
        side  : OrderSide,
        price  : float,
        size   : float,
    ) -> HFTOrder:
        """Submit a limit order into the simulated L3 queue."""
        self._order_id_ctr += 1
        order_id = f"bt_{self._order_id_ctr:08d}"
        ts = time.time()

        order = HFTOrder(
            order_id   = order_id,
            symbol     = self.symbol,
            side       = side,
            order_type = OrderType.LIMIT,
            price      = price,
            size       = size,
            submitted_at = ts,
            latency_ms = self.latency_ms,
        )
        self._orders[order_id] = order
        self._add_to_queue(order)
        logger.debug("HFT BT: submitted limit %s %s @ %.6f sz=%.4f", side.value, order_id, price, size)
        return order

    def submit_market(
        self,
        side: OrderSide,
        size: float,
    ) -> HFTOrder:
        """Submit a market order (always crosses the spread)."""
        self._order_id_ctr += 1
        order_id = f"bt_{self._order_id_ctr:08d}"
        ts = time.time()

        fill_price = self._best_ask if side == OrderSide.BUY else self._best_bid
        impact = self.impact_model.impact_bps(
            order_flow=size * (1 if side == OrderSide.BUY else -1),
            daily_volume=self._daily_volume,
        )
        fill_price = fill_price * (1 + impact / 10000) if side == OrderSide.BUY else fill_price * (1 - impact / 10000)
        fee = fill_price * size * self.taker_fee

        order = HFTOrder(
            order_id    = order_id,
            symbol      = self.symbol,
            side        = side,
            order_type  = OrderType.MARKET,
            price       = fill_price,
            size        = size,
            filled      = size,
            avg_fill_price = fill_price,
            status      = FillStatus.FILLED,
            submitted_at= ts,
            filled_at   = ts + self.latency_ms / 1000.0,
            latency_ms  = self.latency_ms,
        )
        self._orders[order_id] = order
        self._apply_fill(order, fill_price, fee)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Attempt to cancel a live order."""
        order = self._orders.get(order_id)
        if order is None or order.status != FillStatus.LIVE:
            return False
        # Remove from queue
        queue = self._queue_bid if order.side == OrderSide.BUY else self._queue_ask
        for i, qo in enumerate(queue):
            if qo.order_id == order_id:
                queue.remove(qo)
                order.status = FillStatus.CANCELLED
                self._reindex_queue(queue)
                logger.debug("HFT BT: cancelled %s", order_id)
                return True
        return False

    # ------------------------------------------------------------------ Queue simulation

    def _add_to_queue(self, order: HFTOrder) -> None:
        """Add a limit order to the L3 queue."""
        qo = QueuedOrder(
            order_id    = order.order_id,
            side        = order.side,
            price       = order.price,
            size        = order.size,
            position    = len(self._queue_bid) if order.side == OrderSide.BUY else len(self._queue_ask),
            submitted_at= order.submitted_at,
            is_ours     = True,
        )
        if order.side == OrderSide.BUY:
            self._queue_bid.append(qo)
            self._sort_queue(self._queue_bid, ascending=False)  # best bid first
        else:
            self._queue_ask.append(qo)
            self._sort_queue(self._queue_ask, ascending=True)   # best ask first

    def _sort_queue(self, queue: Deque[QueuedOrder], ascending: bool) -> None:
        """Sort queue by price, maintaining position ordering within same price."""
        sorted_list = sorted(
            list(queue),
            key=lambda qo: (qo.price if ascending else -qo.price, qo.submitted_at),
        )
        queue.clear()
        for qo in sorted_list:
            queue.append(qo)

    def _reindex_queue(self, queue: Deque[QueuedOrder]) -> None:
        """Reassign position indices after a cancel/fill."""
        for i, qo in enumerate(queue):
            qo.position = i

    def _queue_position_of(self, order_id: str, side: OrderSide) -> int:
        queue = self._queue_bid if side == OrderSide.BUY else self._queue_ask
        for qo in queue:
            if qo.order_id == order_id:
                return qo.position
        return 999_999

    def _process_tick(self) -> None:
        """Process one simulation tick — attempt fills on all live orders."""
        if self._mid_price <= 0:
            return

        queue_bid = self._queue_bid
        queue_ask = self._queue_ask

        # Trade processing for bids
        to_remove_bid: List[QueuedOrder] = []
        for qo in queue_bid:
            if qo.size <= 0:
                to_remove_bid.append(qo)
                continue

            queue_pos  = qo.position
            fill_prob  = self.fill_model.fill_probability(
                queue_position = queue_pos,
                book_imbalance= self._book_imbal,
                spread_bps    = self._spread_bps,
                volatility    = self._volatility,
                side          = OrderSide.BUY,
            )

            if np.random.random() < fill_prob:
                # Fill
                order = self._orders.get(qo.order_id)
                if order is None:
                    to_remove_bid.append(qo)
                    continue

                impact_bps = self.impact_model.impact_bps(
                    order_flow=qo.size,
                    daily_volume=self._daily_volume,
                )
                fill_price  = self._best_bid * (1 - impact_bps / 10000)
                fee         = fill_price * qo.size * self.maker_fee
                self._apply_fill(order, fill_price, fee)
                to_remove_bid.append(qo)

        for qo in to_remove_bid:
            if qo in queue_bid:
                queue_bid.remove(qo)

        # Trade processing for asks
        to_remove_ask: List[QueuedOrder] = []
        for qo in queue_ask:
            if qo.size <= 0:
                to_remove_ask.append(qo)
                continue

            fill_prob  = self.fill_model.fill_probability(
                queue_position = qo.position,
                book_imbalance= self._book_imbal,
                spread_bps    = self._spread_bps,
                volatility    = self._volatility,
                side          = OrderSide.SELL,
            )

            if np.random.random() < fill_prob:
                order = self._orders.get(qo.order_id)
                if order is None:
                    to_remove_ask.append(qo)
                    continue

                impact_bps = self.impact_model.impact_bps(
                    order_flow=qo.size,
                    daily_volume=self._daily_volume,
                )
                fill_price  = self._best_ask * (1 + impact_bps / 10000)
                fee         = fill_price * qo.size * self.maker_fee
                self._apply_fill(order, fill_price, fee)
                to_remove_ask.append(qo)

        for qo in to_remove_ask:
            if qo in queue_ask:
                queue_ask.remove(qo)

        self._reindex_queue(queue_bid)
        self._reindex_queue(queue_ask)

    def _apply_fill(self, order: HFTOrder, fill_price: float, fee: float) -> None:
        """Apply a fill to an order and update equity."""
        ts = time.time()
        order.filled       = order.size
        order.avg_fill_price = fill_price
        order.status       = FillStatus.FILLED
        order.filled_at    = ts

        latency = (order.filled_at - order.submitted_at) * 1000.0
        self._fill_latencies.append(latency)

        # Update position
        if order.side == OrderSide.BUY:
            self._position     += order.size
            self._entry_price  = fill_price if self._position == order.size \
                else (self._entry_price * (self._position - order.size) + fill_price * order.size) / self._position
            self.equity        -= (fill_price * order.size + fee)
        else:
            self._position     -= order.size
            if self._position == 0:
                self._entry_price = 0.0
            self.equity        += (fill_price * order.size - fee)

        # Unrealised PnL
        if self._position != 0:
            mid = self._mid_price
            if self._position > 0:
                self._position_pnl = (mid - self._entry_price) * self._position
            else:
                self._position_pnl = (self._entry_price - mid) * abs(self._position)
        else:
            self._position_pnl = 0.0

        self._equity_curve.append(self.equity + self._position_pnl)

        if self._on_fill:
            self._on_fill(order, fill_price)

        logger.debug(
            "HFT BT: filled %s %s @ %.6f fee=%.4f equity=%.2f",
            order.order_id, order.side.value, fill_price, fee, self.equity,
        )

    # ------------------------------------------------------------------ Close position

    def close_position(self, reason: str = "manual") -> Optional[TradeRecord]:
        """Close the current open position at mid-price."""
        if abs(self._position) < 1e-9:
            return None

        side   = OrderSide.BUY if self._position < 0 else OrderSide.SELL
        exit_p = self._mid_price
        impact = self.impact_model.impact_bps(
            order_flow=abs(self._position),
            daily_volume=self._daily_volume,
        )
        exit_p = exit_p * (1 + impact / 10000) if side == OrderSide.SELL else exit_p * (1 - impact / 10000)
        fee    = exit_p * abs(self._position) * self.taker_fee

        pnl    = (exit_p - self._entry_price) * abs(self._position) - fee
        self.equity += pnl

        avg_q_pos = self._queue_position_of(
            list(self._orders.values())[-1].order_id, side
        )

        rec = TradeRecord(
            order_id       = f"close_{self._tick_count}",
            side           = side,
            entry_price    = self._entry_price,
            exit_price     = exit_p,
            size           = abs(self._position),
            pnl            = pnl,
            queue_position = avg_q_pos,
            fill_latency_ms= self.latency_ms,
            adverse_sel_bps= impact,
            timestamp      = time.time(),
        )
        self._trades.append(rec)
        self._position    = 0.0
        self._entry_price = 0.0
        self._position_pnl= 0.0
        self._equity_curve.append(self.equity)

        logger.info("HFT BT: closed position pnl=%.2f reason=%s", pnl, reason)
        return rec

    # ------------------------------------------------------------------ Step

    def step(self) -> None:
        """Advance simulation by one tick."""
        self._process_tick()
        if self._on_market_update:
            self._on_market_update(
                self._best_bid, self._best_ask, self._book_imbal, self._volatility,
            )

    # ------------------------------------------------------------------ Stats

    def compute_stats(self) -> BacktestStats:
        """Compute final backtest statistics."""
        equity_arr = np.array(self._equity_curve) if self._equity_curve else np.array([self.initial_equity])
        returns    = np.diff(equity_arr) / equity_arr[:-1] if len(equity_arr) > 1 else np.array([0.0])

        total_pnl   = float(self.equity - self.initial_equity)
        pnl_std     = float(np.std(returns) * 100) if len(returns) > 1 else 0.0

        # Drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns   = (equity_arr - running_max) / running_max
        max_dd      = float(np.min(drawdowns) * 100) if len(drawdowns) > 0 else 0.0

        # Sharpe / Sortino
        mean_ret  = float(np.mean(returns)) if len(returns) > 0 else 0.0
        std_ret   = float(np.std(returns, ddof=1)) if len(returns) > 1 else 1e-12
        neg_ret   = returns[returns < 0]
        down_std  = float(np.std(neg_ret, ddof=1)) if len(neg_ret) > 1 else 1e-12
        sharpe    = float(mean_ret / std_ret * np.sqrt(252 * 24)) if std_ret > 0 else 0.0
        sortino   = float(mean_ret / down_std * np.sqrt(252 * 24)) if down_std > 0 else 0.0

        # Trade stats
        trades     = self._trades
        n_trades   = len(trades)
        wins       = [t for t in trades if t.pnl > 0]
        losses     = [t for t in trades if t.pnl <= 0]
        win_rate   = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win    = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss   = abs(np.mean([t.pnl for t in losses])) if losses else 1e-12
        profit_fac = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Fill stats
        filled    = sum(1 for o in self._orders.values() if o.status == FillStatus.FILLED)
        cancelled = sum(1 for o in self._orders.values() if o.status == FillStatus.CANCELLED)
        avg_q_pos = float(np.mean([
            self._queue_position_of(o.order_id, o.side)
            for o in self._orders.values()
            if o.status == FillStatus.FILLED
        ])) if filled > 0 else 0.0
        avg_lat   = float(np.mean(self._fill_latencies)) if self._fill_latencies else 0.0

        # Adverse selection
        asr = float(np.mean([t.adverse_sel_bps for t in trades])) if trades else 0.0

        # Queue utilisation
        total_vol = sum(t.size for t in trades)
        queue_util = float(total_vol / self._daily_volume * 100) if self._daily_volume > 0 else 0.0

        return BacktestStats(
            total_pnl             = total_pnl,
            pnl_std               = pnl_std,
            max_drawdown          = max_dd,
            sharpe                = sharpe,
            sortino               = sortino,
            total_trades          = n_trades,
            filled_trades         = filled,
            cancelled_trades      = cancelled,
            avg_queue_position    = avg_q_pos,
            avg_fill_latency_ms   = avg_lat,
            adverse_selection_ratio= asr,
            market_impact_bps     = float(np.mean([
                t.adverse_sel_bps for t in trades
            ])) if trades else 0.0,
            queue_utilisation      = queue_util,
            win_rate               = win_rate,
            profit_factor          = profit_fac,
            avg_trade_pnl          = float(np.mean([t.pnl for t in trades])) if trades else 0.0,
            total_volume           = total_vol,
        )

    @property
    def position(self) -> float:
        return self._position

    @property
    def equity_curve(self) -> List[float]:
        return list(self._equity_curve)

    @property
    def trades(self) -> List[TradeRecord]:
        return list(self._trades)
