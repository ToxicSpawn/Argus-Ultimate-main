#!/usr/bin/env python3
"""
core/paper_trade_engine.py — Argus v6.4.0
==========================================
Paper-trading simulation engine for the PC node (PAPER_PC).

Provides realistic order simulation including:
  - Configurable fill probability (market microstructure realism)
  - Bid-ask slippage in basis points
  - Full position and P&L tracking
  - Equity curve history
  - Side-by-side comparison against live node P&L

Usage:
    from core.paper_trade_engine import PaperTradeEngine, PaperConfig

    engine = PaperTradeEngine(PaperConfig(initial_capital_usd=620.0))
    order = asyncio.run(engine.create_order(
        symbol="BTC/USDT", side="buy", order_type="market",
        quantity=0.001, price=65000.0, exchange="binance"
    ))
    print(engine.get_session_pnl())
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderStatus(str, Enum):
    PENDING  = "pending"
    FILLED   = "filled"
    PARTIAL  = "partial"
    EXPIRED  = "expired"
    REJECTED = "rejected"


class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"
    STOP   = "stop"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PaperOrder:
    """
    Represents a simulated order in the paper-trading engine.

    Attributes
    ----------
    order_id:
        Unique identifier (UUID4).
    symbol:
        Trading pair, e.g. ``"BTC/USDT"``.
    side:
        ``"buy"`` or ``"sell"``.
    order_type:
        ``"market"``, ``"limit"``, or ``"stop"``.
    quantity:
        Amount of base asset to trade.
    requested_price:
        The price specified by the strategy (None for market orders).
    fill_price:
        Actual simulated fill price (after slippage).  None until filled.
    fill_quantity:
        How much was filled (may be < quantity for partial fills).
    status:
        :class:`OrderStatus` of this order.
    exchange:
        Exchange name (e.g. ``"binance"``).
    commission_usd:
        Estimated commission in USD at time of fill.
    created_at:
        Unix timestamp when the order was created.
    filled_at:
        Unix timestamp when the order was filled (None if not filled).
    note:
        Optional human-readable note (e.g. ``"slippage=2.0bps"``).
    node_id:
        Node that generated this order (for dedup guard compatibility).
    """
    order_id:        str
    symbol:          str
    side:            str
    order_type:      str
    quantity:        float
    requested_price: Optional[float]
    fill_price:      Optional[float]   = None
    fill_quantity:   float             = 0.0
    status:          OrderStatus       = OrderStatus.PENDING
    exchange:        str               = "paper"
    commission_usd:  float             = 0.0
    created_at:      float             = field(default_factory=time.time)
    filled_at:       Optional[float]   = None
    note:            str               = ""
    node_id:         str               = "pc-gpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id":        self.order_id,
            "symbol":          self.symbol,
            "side":            self.side,
            "order_type":      self.order_type,
            "quantity":        self.quantity,
            "requested_price": self.requested_price,
            "fill_price":      self.fill_price,
            "fill_quantity":   self.fill_quantity,
            "status":          self.status.value,
            "exchange":        self.exchange,
            "commission_usd":  self.commission_usd,
            "created_at":      self.created_at,
            "filled_at":       self.filled_at,
            "note":            self.note,
            "node_id":         self.node_id,
        }


@dataclass
class PaperPosition:
    """
    A currently open paper-trading position.

    Attributes
    ----------
    symbol:
        Trading pair.
    side:
        ``"long"`` or ``"short"``.
    quantity:
        Current open size in base asset.
    avg_entry_price:
        Volume-weighted average entry price.
    unrealised_pnl:
        Estimated P&L using the last known mark price.
    realised_pnl:
        Locked-in P&L from partial closes.
    exchange:
        Exchange this position is on.
    opened_at:
        Unix timestamp of first fill that opened this position.
    last_updated:
        Unix timestamp of the most recent update.
    """
    symbol:           str
    side:             str
    quantity:         float
    avg_entry_price:  float
    unrealised_pnl:   float = 0.0
    realised_pnl:     float = 0.0
    exchange:         str   = "paper"
    opened_at:        float = field(default_factory=time.time)
    last_updated:     float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol":          self.symbol,
            "side":            self.side,
            "quantity":        self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "unrealised_pnl":  self.unrealised_pnl,
            "realised_pnl":    self.realised_pnl,
            "exchange":        self.exchange,
            "opened_at":       self.opened_at,
            "last_updated":    self.last_updated,
        }


@dataclass
class ComparisonResult:
    """
    Side-by-side comparison of paper vs live P&L.

    Attributes
    ----------
    paper_pnl:
        Total session P&L from the paper-trading engine.
    live_pnl:
        Total session P&L reported by the live node (from state sync).
    difference:
        ``paper_pnl - live_pnl``.
    correlation:
        Pearson correlation coefficient between paper and live equity curves
        (NaN if insufficient data).
    paper_win_rate:
        Fraction of paper trades that were profitable (0–1).
    live_win_rate:
        Fraction of live trades that were profitable (0–1).
    paper_trade_count:
        Number of paper trades in the comparison window.
    live_trade_count:
        Number of live trades in the comparison window.
    recommendation:
        Human-readable string advising whether paper strategy is viable.
    generated_at:
        UTC timestamp string when this comparison was generated.
    """
    paper_pnl:         float
    live_pnl:          float
    difference:        float
    correlation:       float
    paper_win_rate:    float = 0.0
    live_win_rate:     float = 0.0
    paper_trade_count: int   = 0
    live_trade_count:  int   = 0
    recommendation:    str   = ""
    generated_at:      str   = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_pnl":         self.paper_pnl,
            "live_pnl":          self.live_pnl,
            "difference":        self.difference,
            "correlation":       self.correlation,
            "paper_win_rate":    self.paper_win_rate,
            "live_win_rate":     self.live_win_rate,
            "paper_trade_count": self.paper_trade_count,
            "live_trade_count":  self.live_trade_count,
            "recommendation":    self.recommendation,
            "generated_at":      self.generated_at,
        }


# ---------------------------------------------------------------------------
# PaperConfig
# ---------------------------------------------------------------------------

@dataclass
class PaperConfig:
    """
    Configuration for :class:`PaperTradeEngine`.

    Attributes
    ----------
    initial_capital_usd:
        Starting balance in USD (default 620.0 matching the bot's live capital).
    slippage_bps:
        Simulated bid-ask spread slippage in basis points applied to each fill.
        e.g. 2.0 bps on a $65,000 BTC/USDT order → $13 slippage.
    fill_probability:
        Probability in [0, 1] that a market/limit order is filled on the first
        attempt.  Use 1.0 for deterministic back-tests; 0.6 for realism.
    commission_rate_bps:
        Exchange commission in basis points (default 7 bps ≈ 0.07%).
    track_live_comparison:
        If *True*, equity curve data is kept for comparison with live node.
    state_dir:
        Directory where paper state JSON files are persisted.
    node_id:
        Node ID stamped into paper orders (for dedup guard compatibility).
    max_position_size_usd:
        Hard cap on any single position (0 = unlimited).
    """
    initial_capital_usd:     float = 620.0
    slippage_bps:            float = 2.0
    fill_probability:        float = 0.6
    commission_rate_bps:     float = 7.0
    track_live_comparison:   bool  = True
    state_dir:               str   = "data/node_state"
    node_id:                 str   = "pc-gpu"
    max_position_size_usd:   float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_slippage(price: float, side: str, slippage_bps: float) -> float:
    """
    Return the fill price after applying slippage.

    A *buy* order is filled at a slightly *higher* price (worse for buyer).
    A *sell* order is filled at a slightly *lower* price (worse for seller).
    """
    slip_frac = slippage_bps / 10_000.0
    if side.lower() == "buy":
        return price * (1.0 + slip_frac)
    return price * (1.0 - slip_frac)


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """
    Compute Pearson correlation coefficient between two equal-length lists.
    Returns NaN if fewer than 2 data points or zero variance.
    """
    n = min(len(xs), len(ys))
    if n < 2:
        return float("nan")
    xs = xs[:n]
    ys = ys[:n]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if std_x == 0.0 or std_y == 0.0:
        return float("nan")
    return cov / (std_x * std_y)


def _win_rate(pnl_list: List[float]) -> float:
    """Return fraction of trades with positive P&L."""
    if not pnl_list:
        return 0.0
    return sum(1 for p in pnl_list if p > 0) / len(pnl_list)


# ---------------------------------------------------------------------------
# PaperTradeEngine
# ---------------------------------------------------------------------------

class PaperTradeEngine:
    """
    Simulates order execution for the PC node without touching any exchange.

    All accounting is done in USD.  Positions are tracked per-symbol
    per-exchange.  Equity curve data is stored for comparison with the live
    node's P&L history.

    Thread safety
    -------------
    All public ``async`` methods are safe to call from a single asyncio event
    loop.  The internal state is protected by :class:`asyncio.Lock`.

    Parameters
    ----------
    config:
        :class:`PaperConfig` instance.
    """

    def __init__(self, config: PaperConfig) -> None:
        self.config = config
        self._lock  = asyncio.Lock()

        # Accounting
        self._balance_usd:   float                       = config.initial_capital_usd
        self._locked_usd:    float                       = 0.0       # margin in open positions
        self._positions:     Dict[str, PaperPosition]   = {}         # key: "symbol:exchange"
        self._orders:        Dict[str, PaperOrder]       = {}         # key: order_id
        self._session_start: float                       = time.time()

        # History
        self._equity_curve:  List[Tuple[float, float]]  = []         # (timestamp, equity_usd)
        self._trade_pnls:    List[float]                 = []         # realised P&L per closed trade
        self._pnl_history:   List[Dict[str, Any]]        = []         # richer per-trade records

        # Mark price cache (updated by update_mark_price)
        self._mark_prices:   Dict[str, float]            = {}

        # Record initial equity
        self._snapshot_equity()

        logger.info(
            "PaperTradeEngine initialised — capital=%.2f USD, slippage=%.1f bps, "
            "fill_prob=%.2f",
            config.initial_capital_usd,
            config.slippage_bps,
            config.fill_probability,
        )

    # ------------------------------------------------------------------
    # Mark price management
    # ------------------------------------------------------------------

    def update_mark_price(self, symbol: str, price: float) -> None:
        """
        Update the mark price for *symbol* and recompute unrealised P&L for
        all open positions in that symbol.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTC/USDT"``.
        price:
            Current mid-market price.
        """
        self._mark_prices[symbol] = price
        for key, pos in self._positions.items():
            if pos.symbol == symbol:
                if pos.side == "long":
                    pos.unrealised_pnl = (price - pos.avg_entry_price) * pos.quantity
                else:
                    pos.unrealised_pnl = (pos.avg_entry_price - price) * pos.quantity
                pos.last_updated = time.time()

    # ------------------------------------------------------------------
    # Order creation
    # ------------------------------------------------------------------

    async def create_order(
        self,
        symbol:     str,
        side:       str,
        order_type: str,
        quantity:   float,
        price:      Optional[float],
        exchange:   str = "paper",
    ) -> PaperOrder:
        """
        Simulate submitting an order.  No exchange API is called.

        Fill logic
        ----------
        1. Roll a uniform random number against ``fill_probability``.
        2. If filled: apply slippage, compute commission, update balance
           and positions.
        3. If not filled: mark order as ``EXPIRED``.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).
        side:
            ``"buy"`` or ``"sell"``.
        order_type:
            ``"market"``, ``"limit"``, or ``"stop"``.
        quantity:
            Amount of base asset.
        price:
            Reference price. For market orders this is the last known
            mid-market price.  Pass ``None`` to use the cached mark price.
        exchange:
            Exchange label (for accounting/display only).

        Returns
        -------
        PaperOrder
            The created (and possibly filled) order.
        """
        async with self._lock:
            # Resolve price
            ref_price = price
            if ref_price is None:
                ref_price = self._mark_prices.get(symbol)
            if ref_price is None or ref_price <= 0:
                order = PaperOrder(
                    order_id=str(uuid.uuid4()),
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    requested_price=price,
                    status=OrderStatus.REJECTED,
                    exchange=exchange,
                    note="no price available",
                    node_id=self.config.node_id,
                )
                self._orders[order.order_id] = order
                logger.warning(
                    "Paper order REJECTED (no price): %s %s %s", side, quantity, symbol
                )
                return order

            # Validate quantity
            if quantity <= 0:
                order = PaperOrder(
                    order_id=str(uuid.uuid4()),
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    requested_price=ref_price,
                    status=OrderStatus.REJECTED,
                    exchange=exchange,
                    note="quantity <= 0",
                    node_id=self.config.node_id,
                )
                self._orders[order.order_id] = order
                return order

            # Position size guard
            notional = quantity * ref_price
            if (
                self.config.max_position_size_usd > 0
                and notional > self.config.max_position_size_usd
            ):
                logger.warning(
                    "Paper order: notional %.2f > max_position_size_usd %.2f — capping quantity.",
                    notional, self.config.max_position_size_usd,
                )
                quantity = self.config.max_position_size_usd / ref_price
                notional = self.config.max_position_size_usd

            # Simulate fill
            filled = random.random() < self.config.fill_probability
            order_id = str(uuid.uuid4())

            if filled:
                fill_price = _apply_slippage(ref_price, side, self.config.slippage_bps)
                commission = (notional * self.config.commission_rate_bps) / 10_000.0
                order = PaperOrder(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    requested_price=ref_price,
                    fill_price=fill_price,
                    fill_quantity=quantity,
                    status=OrderStatus.FILLED,
                    exchange=exchange,
                    commission_usd=commission,
                    filled_at=time.time(),
                    note=f"slippage={self.config.slippage_bps:.1f}bps",
                    node_id=self.config.node_id,
                )
                self._apply_fill(order)
            else:
                order = PaperOrder(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    requested_price=ref_price,
                    status=OrderStatus.EXPIRED,
                    exchange=exchange,
                    note="fill_probability miss",
                    node_id=self.config.node_id,
                )
                logger.debug(
                    "Paper order EXPIRED (fill_prob miss): %s %s %s @ %.4f",
                    side, quantity, symbol, ref_price,
                )

            self._orders[order.order_id] = order
            self._snapshot_equity()
            return order

    # ------------------------------------------------------------------
    # Internal fill application
    # ------------------------------------------------------------------

    def _apply_fill(self, order: PaperOrder) -> None:
        """
        Update balance, positions, and P&L records after a successful fill.
        """
        assert order.fill_price is not None
        fill_notional = order.fill_quantity * order.fill_price
        commission    = order.commission_usd

        pos_key = f"{order.symbol}:{order.exchange}"

        if order.side.lower() == "buy":
            # Debit cash
            cost = fill_notional + commission
            self._balance_usd -= cost

            # Update or open long position
            if pos_key in self._positions:
                pos = self._positions[pos_key]
                if pos.side == "long":
                    # Average up / add to long
                    total_qty   = pos.quantity + order.fill_quantity
                    pos.avg_entry_price = (
                        (pos.avg_entry_price * pos.quantity + order.fill_price * order.fill_quantity)
                        / total_qty
                    )
                    pos.quantity    = total_qty
                    pos.last_updated = time.time()
                else:
                    # Closing a short position
                    realised = (pos.avg_entry_price - order.fill_price) * min(
                        pos.quantity, order.fill_quantity
                    )
                    realised -= commission
                    pos.realised_pnl += realised
                    self._balance_usd += realised
                    self._trade_pnls.append(realised)
                    self._record_pnl_history(order, realised)

                    remaining = pos.quantity - order.fill_quantity
                    if remaining <= 1e-10:
                        del self._positions[pos_key]
                    else:
                        pos.quantity     = remaining
                        pos.last_updated = time.time()
            else:
                self._positions[pos_key] = PaperPosition(
                    symbol=order.symbol,
                    side="long",
                    quantity=order.fill_quantity,
                    avg_entry_price=order.fill_price,
                    exchange=order.exchange,
                )

        else:  # sell
            # Credit cash
            self._balance_usd += fill_notional - commission

            if pos_key in self._positions:
                pos = self._positions[pos_key]
                if pos.side == "long":
                    # Closing a long position
                    realised = (order.fill_price - pos.avg_entry_price) * min(
                        pos.quantity, order.fill_quantity
                    )
                    realised -= commission
                    pos.realised_pnl += realised
                    self._trade_pnls.append(realised)
                    self._record_pnl_history(order, realised)

                    remaining = pos.quantity - order.fill_quantity
                    if remaining <= 1e-10:
                        del self._positions[pos_key]
                    else:
                        pos.quantity     = remaining
                        pos.last_updated = time.time()
                else:
                    # Adding to short
                    total_qty = pos.quantity + order.fill_quantity
                    pos.avg_entry_price = (
                        (pos.avg_entry_price * pos.quantity + order.fill_price * order.fill_quantity)
                        / total_qty
                    )
                    pos.quantity     = total_qty
                    pos.last_updated = time.time()
            else:
                # Open new short
                self._positions[pos_key] = PaperPosition(
                    symbol=order.symbol,
                    side="short",
                    quantity=order.fill_quantity,
                    avg_entry_price=order.fill_price,
                    exchange=order.exchange,
                )

        logger.debug(
            "Paper fill applied: %s %s %.6f %s @ %.4f  balance=%.2f USD",
            order.side, order.symbol, order.fill_quantity,
            order.exchange, order.fill_price, self._balance_usd,
        )

    def _record_pnl_history(self, order: PaperOrder, realised_pnl: float) -> None:
        """Append a realised trade record to _pnl_history."""
        self._pnl_history.append({
            "order_id":    order.order_id,
            "symbol":      order.symbol,
            "side":        order.side,
            "quantity":    order.fill_quantity,
            "fill_price":  order.fill_price,
            "realised_pnl": realised_pnl,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def _snapshot_equity(self) -> None:
        """Record a point on the equity curve."""
        if not self.config.track_live_comparison:
            return
        total_equity = self._balance_usd + sum(
            p.unrealised_pnl for p in self._positions.values()
        )
        self._equity_curve.append((time.time(), total_equity))

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def get_balance(self) -> Dict[str, float]:
        """
        Return current balance breakdown.

        Keys: ``available_usd``, ``locked_usd``, ``total_equity_usd``,
        ``unrealised_pnl_usd``, ``initial_capital_usd``.
        """
        unrealised = sum(p.unrealised_pnl for p in self._positions.values())
        return {
            "available_usd":       round(self._balance_usd, 6),
            "locked_usd":          round(self._locked_usd, 6),
            "total_equity_usd":    round(self._balance_usd + unrealised, 6),
            "unrealised_pnl_usd":  round(unrealised, 6),
            "initial_capital_usd": self.config.initial_capital_usd,
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return a list of open position dicts."""
        return [p.to_dict() for p in self._positions.values()]

    def get_orders(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return order dicts, optionally filtered by status string.

        Parameters
        ----------
        status_filter:
            If provided, only orders with this status value are returned.
        """
        orders = list(self._orders.values())
        if status_filter:
            orders = [o for o in orders if o.status.value == status_filter]
        return [o.to_dict() for o in orders]

    def get_session_pnl(self) -> float:
        """
        Total realised P&L for this session in USD.

        Computed as: ``current_equity - initial_capital``.
        """
        balance = self.get_balance()
        return round(balance["total_equity_usd"] - self.config.initial_capital_usd, 6)

    def get_equity_curve(self) -> List[Dict[str, float]]:
        """
        Return the equity curve as a list of ``{"timestamp": float, "equity_usd": float}``
        dicts.
        """
        return [{"timestamp": ts, "equity_usd": eq} for ts, eq in self._equity_curve]

    def get_pnl_history(self) -> List[Dict[str, Any]]:
        """Return the list of realised trade P&L records."""
        return list(self._pnl_history)

    # ------------------------------------------------------------------
    # Live comparison
    # ------------------------------------------------------------------

    def compare_to_live(
        self,
        live_pnl_history: List[Dict[str, Any]],
    ) -> ComparisonResult:
        """
        Compare paper-trading performance against the live node's P&L history.

        Parameters
        ----------
        live_pnl_history:
            List of trade records from the live node.  Each record should
            have at minimum a ``"realised_pnl"`` key.

        Returns
        -------
        ComparisonResult
            Comparison metrics and a recommendation string.
        """
        # Paper series
        paper_pnl_vals = [r["realised_pnl"] for r in self._pnl_history]
        paper_total    = self.get_session_pnl()
        paper_wr       = _win_rate(paper_pnl_vals)

        # Live series
        live_pnl_vals  = [r.get("realised_pnl", 0.0) for r in live_pnl_history]
        live_total     = sum(live_pnl_vals)
        live_wr        = _win_rate(live_pnl_vals)

        # Equity curve correlation (align by trade index)
        n = min(len(paper_pnl_vals), len(live_pnl_vals))
        if n >= 2:
            paper_cum = [sum(paper_pnl_vals[:i+1]) for i in range(n)]
            live_cum  = [sum(live_pnl_vals[:i+1])  for i in range(n)]
            corr = _pearson_correlation(paper_cum, live_cum)
        else:
            corr = float("nan")

        difference = paper_total - live_total

        # Recommendation
        recommendation = self._build_recommendation(
            paper_total, live_total, corr, paper_wr, live_wr,
            len(paper_pnl_vals), len(live_pnl_vals),
        )

        return ComparisonResult(
            paper_pnl=round(paper_total, 4),
            live_pnl=round(live_total, 4),
            difference=round(difference, 4),
            correlation=round(corr, 4) if not math.isnan(corr) else corr,
            paper_win_rate=round(paper_wr, 4),
            live_win_rate=round(live_wr, 4),
            paper_trade_count=len(paper_pnl_vals),
            live_trade_count=len(live_pnl_vals),
            recommendation=recommendation,
        )

    def _build_recommendation(
        self,
        paper_pnl: float,
        live_pnl: float,
        corr: float,
        paper_wr: float,
        live_wr: float,
        n_paper: int,
        n_live: int,
    ) -> str:
        """
        Generate a plain-language recommendation based on the comparison metrics.
        """
        if n_paper < 5 or n_live < 5:
            return (
                "Insufficient trade history for a reliable recommendation "
                f"(paper={n_paper} trades, live={n_live} trades). "
                "Continue accumulating data."
            )

        lines = []

        # PnL comparison
        if paper_pnl > live_pnl * 1.10:
            lines.append(
                "Paper P&L is >10% higher than live — possible slippage/execution drag "
                "on the R7525; review order routing."
            )
        elif live_pnl > paper_pnl * 1.10:
            lines.append(
                "Live P&L is >10% higher than paper — strategy may be under-filling "
                "simulated orders; consider lowering fill_probability."
            )
        else:
            lines.append("Paper and live P&L are within 10% — simulation fidelity is acceptable.")

        # Correlation
        if not math.isnan(corr):
            if corr >= 0.85:
                lines.append(f"Equity curve correlation is strong ({corr:.2f}) — strategy is consistent.")
            elif corr >= 0.5:
                lines.append(
                    f"Equity curve correlation is moderate ({corr:.2f}) — "
                    "some divergence; investigate market-hours mismatches."
                )
            else:
                lines.append(
                    f"Equity curve correlation is weak ({corr:.2f}) — "
                    "paper and live are diverging; re-tune paper simulation parameters."
                )

        # Win rates
        if abs(paper_wr - live_wr) > 0.15:
            lines.append(
                f"Win rate gap: paper={paper_wr:.0%} vs live={live_wr:.0%} — "
                "slippage or fill-prob miscalibration suspected."
            )

        # Overall
        if paper_pnl > 0 and live_pnl > 0:
            lines.append("Both nodes are profitable — strategy viable for continued operation.")
        elif paper_pnl <= 0 and live_pnl <= 0:
            lines.append("ALERT: Both nodes are losing money — consider pausing strategy.")
        elif live_pnl < 0:
            lines.append("ALERT: Live node is losing money while paper is positive — check R7525 execution.")

        return "  ".join(lines)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_paper_report(self) -> str:
        """
        Return a formatted multi-line text report summarising the paper session.

        Suitable for logging, Slack notifications, or dashboard display.
        """
        bal        = self.get_balance()
        session_s  = time.time() - self._session_start
        hours, rem = divmod(int(session_s), 3600)
        mins       = rem // 60
        n_trades   = len([o for o in self._orders.values() if o.status == OrderStatus.FILLED])
        n_expired  = len([o for o in self._orders.values() if o.status == OrderStatus.EXPIRED])

        lines = [
            "=" * 60,
            "  ARGUS PAPER TRADE ENGINE — SESSION REPORT",
            "=" * 60,
            f"  Session duration    : {hours}h {mins}m",
            f"  Initial capital     : ${self.config.initial_capital_usd:,.2f}",
            f"  Current equity      : ${bal['total_equity_usd']:,.2f}",
            f"  Session P&L         : ${self.get_session_pnl():+,.4f}",
            f"  Unrealised P&L      : ${bal['unrealised_pnl_usd']:+,.4f}",
            f"  Available balance   : ${bal['available_usd']:,.2f}",
            f"  Open positions      : {len(self._positions)}",
            f"  Filled orders       : {n_trades}",
            f"  Expired orders      : {n_expired}",
            f"  Win rate            : {_win_rate(self._trade_pnls):.1%}",
            f"  Slippage config     : {self.config.slippage_bps:.1f} bps",
            f"  Fill probability    : {self.config.fill_probability:.0%}",
            f"  Commission rate     : {self.config.commission_rate_bps:.1f} bps",
            "-" * 60,
        ]

        if self._positions:
            lines.append("  OPEN POSITIONS:")
            for key, pos in self._positions.items():
                lines.append(
                    f"    {pos.symbol} {pos.side.upper():5s} "
                    f"qty={pos.quantity:.6f}  "
                    f"entry={pos.avg_entry_price:.4f}  "
                    f"upnl={pos.unrealised_pnl:+.4f}"
                )
            lines.append("-" * 60)

        if self._pnl_history:
            lines.append("  LAST 5 CLOSED TRADES:")
            for rec in self._pnl_history[-5:]:
                lines.append(
                    f"    {rec['symbol']} {rec['side']:4s} "
                    f"pnl={rec['realised_pnl']:+.4f}"
                )
            lines.append("-" * 60)

        lines.append(f"  Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """
        Persist paper engine state to the state_dir for Git sync pickup.

        Writes:
        - ``active_positions.json``
        - ``active_orders.json``   (FILLED orders only)
        - ``pnl_history.json``
        - ``capital_state.json``
        """
        base = Path(self.config.state_dir)
        base.mkdir(parents=True, exist_ok=True)

        bal = self.get_balance()

        def write(fname: str, data: Any) -> None:
            (base / fname).write_text(json.dumps(data, indent=2, default=str))

        write("active_positions.json", self.get_positions())
        write("active_orders.json", [
            o.to_dict() for o in self._orders.values()
            if o.status == OrderStatus.FILLED
        ])
        write("pnl_history.json", self._pnl_history)
        write("capital_state.json", {
            "total_usd":      bal["total_equity_usd"],
            "available_usd":  bal["available_usd"],
            "locked_usd":     bal["locked_usd"],
            "equity_usd":     bal["total_equity_usd"],
            "session_pnl":    self.get_session_pnl(),
            "timestamp_utc":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        logger.debug("Paper engine state saved to %s", base)

    def load_state(self) -> None:
        """
        Restore paper engine state from state_dir (e.g. after a restart).

        Silently skips missing files.
        """
        base = Path(self.config.state_dir)

        def load(fname: str, default: Any) -> Any:
            p = base / fname
            if not p.exists():
                return default
            try:
                return json.loads(p.read_text())
            except Exception:
                return default

        pnl_history = load("pnl_history.json", [])
        self._pnl_history = pnl_history
        self._trade_pnls  = [r["realised_pnl"] for r in pnl_history]

        capital = load("capital_state.json", {})
        if "available_usd" in capital:
            self._balance_usd = capital["available_usd"]

        logger.info(
            "Paper engine state restored — balance=%.2f USD, trades=%d",
            self._balance_usd, len(self._pnl_history),
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PaperTradeEngine(capital={self.config.initial_capital_usd}, "
            f"balance={self._balance_usd:.2f}, "
            f"positions={len(self._positions)}, "
            f"orders={len(self._orders)})"
        )
