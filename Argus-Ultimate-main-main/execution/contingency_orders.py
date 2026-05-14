"""
execution/contingency_orders.py
================================
Async live-exchange execution layer for contingency order groups.

Sits on top of ConditionalOrderManager (conditional_orders.py) and adds:
  - Async submit / cancel via any ccxt-compatible connector
  - OTO (One-Triggers-Other) support: submitting dependent legs on entry fill
  - Full bracket order lifecycle: entry → async OCO placement on fill
  - Trailing stop ratchet with async amend via cancel-replace
  - post_only / reduce_only flag pass-through to venue
  - Prometheus metric hooks (inc_error on cancel failure)
  - Structured logging on every state transition

Usage
-----
    from execution.contingency_orders import ContingencyExecutor

    executor = ContingencyExecutor(connector=ccxt_bybit, metrics=emitter)
    gid = await executor.submit_bracket(
        symbol="BTC/USDT",
        entry_price=65_000.0,
        tp_price=68_000.0,
        sl_price=63_000.0,
        quantity=0.01,
        exchange="bybit",
        post_only=True,
    )
    # On receiving a fill WebSocket message:
    await executor.on_fill(order_id="<exchange-order-id>", filled_qty=0.01, fill_price=65_010.0)
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .conditional_orders import (
    ConditionalGroup,
    ConditionalOrderManager,
    GroupStatus,
    GroupType,
    LegStatus,
    LegType,
    OrderLeg,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OTO group type (extends existing GroupType concept — stored in metadata)
# ---------------------------------------------------------------------------

class OTOStatus(Enum):
    WAITING  = "waiting"   # trigger leg not yet filled
    ACTIVE   = "active"    # trigger filled; dependent legs submitted
    FILLED   = "filled"
    CANCELLED = "cancelled"


@dataclass
class OTOGroup:
    """
    One-Triggers-Other: when the trigger leg fills, all dependent legs are
    submitted to the exchange automatically.
    """
    group_id: str
    symbol: str
    exchange: str
    trigger_order_id: str          # synthetic ID until exchange ack
    trigger_side: str
    trigger_price: float
    trigger_qty: float
    dependent_specs: List[Dict[str, Any]]   # raw params for submit_order
    status: OTOStatus = OTOStatus.WAITING
    submitted_dependent_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ContingencyExecutor
# ---------------------------------------------------------------------------

class ContingencyExecutor:
    """
    Async execution layer for OCO, bracket, trailing-stop, and OTO orders.

    Parameters
    ----------
    connector : object
        Any object exposing:
          - ``create_order(symbol, type, side, amount, price, params) -> dict``  (async or sync)
          - ``cancel_order(order_id, symbol) -> dict``                           (async or sync)
          Both may be coroutines; the executor will await them automatically.
    metrics : optional PrometheusEmitter
        If provided, ``inc_error(kind=...)`` is called on cancel/submit failures.
    """

    def __init__(
        self,
        connector: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        self._connector = connector
        self._metrics   = metrics
        self._cond_mgr  = ConditionalOrderManager(connector=connector)
        self._oto_groups: Dict[str, OTOGroup] = {}          # group_id → OTOGroup
        self._oto_trigger_index: Dict[str, str] = {}        # trigger_order_id → group_id
        # exchange_order_id → synthetic_order_id mapping (populated after submit)
        self._id_map: Dict[str, str] = {}                   # exchange_id → synthetic_id
        self._id_map_rev: Dict[str, str] = {}               # synthetic_id → exchange_id
        mode = "LIVE" if connector is not None else "SIMULATION"
        logger.info("ContingencyExecutor initialised in %s mode", mode)

    # ------------------------------------------------------------------
    # Public factory methods
    # ------------------------------------------------------------------

    async def submit_oco(
        self,
        symbol: str,
        tp_price: float,
        sl_price: float,
        quantity: float,
        exchange: str,
        side: str = "sell",
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> str:
        """
        Submit an OCO (One-Cancels-Other) pair directly to the exchange.

        Returns the ConditionalOrderManager group_id.
        The two legs (TP limit + SL stop-limit) are placed simultaneously.
        When either fills, on_fill() will cancel the surviving leg.
        """
        gid = self._cond_mgr.create_oco(
            symbol=symbol,
            tp_price=tp_price,
            sl_price=sl_price,
            quantity=quantity,
            exchange=exchange,
            side=side,
        )
        group = self._cond_mgr._groups[gid]
        tp_leg = self._leg_by_type(group, LegType.TP)
        sl_leg = self._leg_by_type(group, LegType.SL)

        params: Dict[str, Any] = {}
        if post_only:
            params["postOnly"] = True
        if reduce_only:
            params["reduceOnly"] = True

        await asyncio.gather(
            self._submit_leg(tp_leg, symbol, "limit", params=dict(params)),
            self._submit_leg(sl_leg, symbol, "stop",  params=dict(params)),
        )
        logger.info(
            "ContingencyExecutor: OCO submitted group=%s symbol=%s tp=%.4f sl=%.4f qty=%.6f",
            gid, symbol, tp_price, sl_price, quantity,
        )
        return gid

    async def submit_bracket(
        self,
        symbol: str,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        quantity: float,
        exchange: str,
        entry_side: str = "buy",
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> str:
        """
        Submit a bracket order: entry limit → OCO (TP + SL) on fill.

        The TP and SL legs are held in PENDING state until the entry fills,
        then placed atomically via on_fill().
        Returns the ConditionalOrderManager group_id.
        """
        gid = self._cond_mgr.create_bracket(
            symbol=symbol,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            quantity=quantity,
            exchange=exchange,
            entry_side=entry_side,
        )
        group = self._cond_mgr._groups[gid]
        entry_leg = self._leg_by_type(group, LegType.ENTRY)

        params: Dict[str, Any] = {}
        if post_only:
            params["postOnly"] = True
        # Store flags on group for later use when OCO legs are submitted
        group.__dict__["_post_only"]    = post_only
        group.__dict__["_reduce_only"]  = reduce_only

        await self._submit_leg(entry_leg, symbol, "limit", params=params)
        logger.info(
            "ContingencyExecutor: BRACKET entry submitted group=%s symbol=%s "
            "entry=%.4f tp=%.4f sl=%.4f qty=%.6f",
            gid, symbol, entry_price, tp_price, sl_price, quantity,
        )
        return gid

    async def submit_trailing_stop(
        self,
        symbol: str,
        trail_pct: float,
        quantity: float,
        exchange: str,
        initial_price: float = 0.0,
        side: str = "sell",
    ) -> str:
        """
        Register a trailing stop.  The initial stop order is placed on the
        exchange; on_price_update() will amend it via cancel-replace as the
        market moves.
        Returns the ConditionalOrderManager group_id.
        """
        gid = self._cond_mgr.create_trailing_stop(
            symbol=symbol,
            trail_pct=trail_pct,
            quantity=quantity,
            exchange=exchange,
            initial_price=initial_price,
            side=side,
        )
        group = self._cond_mgr._groups[gid]
        trail_leg = group.legs[0]
        if initial_price > 0:
            await self._submit_leg(trail_leg, symbol, "stop", params={})
        logger.info(
            "ContingencyExecutor: TRAILING registered group=%s symbol=%s "
            "trail_pct=%.2f%% initial_price=%.4f",
            gid, symbol, trail_pct * 100, initial_price,
        )
        return gid

    async def submit_oto(
        self,
        symbol: str,
        trigger_price: float,
        trigger_side: str,
        trigger_qty: float,
        dependent_specs: List[Dict[str, Any]],
        exchange: str,
        post_only: bool = False,
    ) -> str:
        """
        Submit an OTO (One-Triggers-Other) order.

        The trigger leg is placed immediately.  When it fills, all
        ``dependent_specs`` are submitted in parallel.  Each spec is a dict
        of kwargs passed directly to ``create_order``:
            [
              {"type": "limit", "side": "sell", "price": 68000, "amount": 0.01},
              {"type": "stop",  "side": "sell", "price": 62000, "amount": 0.01},
            ]
        Returns the OTOGroup group_id.
        """
        group_id = str(uuid.uuid4())
        trigger_id = f"oto-trigger-{uuid.uuid4().hex[:12]}"

        oto = OTOGroup(
            group_id=group_id,
            symbol=symbol,
            exchange=exchange,
            trigger_order_id=trigger_id,
            trigger_side=trigger_side,
            trigger_price=trigger_price,
            trigger_qty=trigger_qty,
            dependent_specs=dependent_specs,
        )
        self._oto_groups[group_id] = oto
        self._oto_trigger_index[trigger_id] = group_id

        params: Dict[str, Any] = {"postOnly": True} if post_only else {}
        exchange_id = await self._raw_submit(
            symbol=symbol,
            order_type="limit",
            side=trigger_side,
            amount=trigger_qty,
            price=trigger_price,
            params=params,
        )
        if exchange_id:
            self._id_map[exchange_id]    = trigger_id
            self._id_map_rev[trigger_id] = exchange_id

        logger.info(
            "ContingencyExecutor: OTO trigger submitted group=%s symbol=%s "
            "side=%s price=%.4f qty=%.6f dependents=%d",
            group_id, symbol, trigger_side, trigger_price, trigger_qty,
            len(dependent_specs),
        )
        return group_id

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_fill(
        self,
        order_id: str,
        filled_qty: float,
        fill_price: float,
    ) -> None:
        """
        Call when a fill WebSocket/REST message is received.

        Handles:
          - ConditionalOrderManager fill routing (OCO/bracket/trailing)
          - Bracket OCO leg activation on entry fill
          - OTO dependent leg submission on trigger fill

        ``order_id`` may be either the exchange-assigned ID or a synthetic ID.
        """
        # Resolve to synthetic ID for ConditionalOrderManager
        synthetic_id = self._id_map.get(order_id, order_id)

        # Route through conditional manager (OCO / bracket / trailing)
        self._cond_mgr.on_fill(
            order_id=synthetic_id,
            filled_qty=filled_qty,
            fill_price=fill_price,
        )

        # Check if bracket entry just filled → submit OCO legs
        bracket_gid = self._cond_mgr._order_to_group.get(synthetic_id)
        if bracket_gid:
            group = self._cond_mgr._groups.get(bracket_gid)
            if group and group.status == GroupStatus.PARTIAL and group.group_type == GroupType.BRACKET:
                await self._activate_bracket_oco(group)

        # Check if OTO trigger filled → submit dependents
        oto_gid = self._oto_trigger_index.get(synthetic_id)
        if oto_gid:
            oto = self._oto_groups.get(oto_gid)
            if oto and oto.status == OTOStatus.WAITING:
                await self._activate_oto_dependents(oto)

    async def on_price_update(self, symbol: str, price: float) -> None:
        """
        Feed market price ticks.  Routes to ConditionalOrderManager for
        trailing stop ratchet; if the stop price advances, performs
        cancel-replace on the exchange.
        """
        # Snapshot stop prices before update
        old_stops: Dict[str, float] = {}
        for group in self._cond_mgr._groups.values():
            if group.symbol == symbol and group.group_type == GroupType.TRAILING:
                old_stops[group.group_id] = group.__dict__.get("_stop_price", 0.0)

        self._cond_mgr.on_price_update(symbol=symbol, price=price)

        # Detect stop price changes → cancel-replace
        for group in self._cond_mgr._groups.values():
            if group.symbol != symbol or group.group_type != GroupType.TRAILING:
                continue
            new_stop = group.__dict__.get("_stop_price", 0.0)
            old_stop = old_stops.get(group.group_id, 0.0)
            if new_stop != old_stop and new_stop > 0 and group.legs:
                trail_leg = group.legs[0]
                await self._amend_trailing_stop(
                    leg=trail_leg,
                    symbol=symbol,
                    new_stop_price=new_stop,
                )

    async def cancel_group(self, group_id: str) -> None:
        """
        Cancel all active legs in a ConditionalOrderManager group and
        remove from the exchange.
        """
        group = self._cond_mgr._groups.get(group_id)
        if group is None:
            logger.warning(
                "ContingencyExecutor.cancel_group: group_id=%s not found", group_id
            )
            return
        for leg in group.legs:
            if leg.status == LegStatus.PENDING:
                exchange_id = self._id_map_rev.get(leg.order_id)
                if exchange_id:
                    await self._raw_cancel(exchange_id, group.symbol)
        self._cond_mgr.cancel_group(group_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _submit_leg(
        self,
        leg: OrderLeg,
        symbol: str,
        order_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Submit a single OrderLeg to the exchange and record the mapping."""
        exchange_id = await self._raw_submit(
            symbol=symbol,
            order_type=order_type,
            side=leg.side,
            amount=leg.quantity,
            price=leg.price,
            params=params or {},
        )
        if exchange_id:
            self._id_map[exchange_id]      = leg.order_id
            self._id_map_rev[leg.order_id] = exchange_id

    async def _activate_bracket_oco(
        self,
        group: ConditionalGroup,
    ) -> None:
        """Place TP and SL legs after entry fills in a bracket."""
        post_only   = group.__dict__.get("_post_only", False)
        reduce_only = group.__dict__.get("_reduce_only", False)
        params: Dict[str, Any] = {}
        if post_only:
            params["postOnly"] = True
        if reduce_only:
            params["reduceOnly"] = True

        tp_leg = self._leg_by_type(group, LegType.TP)
        sl_leg = self._leg_by_type(group, LegType.SL)
        if tp_leg and sl_leg:
            await asyncio.gather(
                self._submit_leg(tp_leg, group.symbol, "limit", params=dict(params)),
                self._submit_leg(sl_leg, group.symbol, "stop",  params=dict(params)),
            )
            logger.info(
                "ContingencyExecutor: BRACKET OCO legs activated group=%s",
                group.group_id,
            )

    async def _activate_oto_dependents(self, oto: OTOGroup) -> None:
        """Submit all dependent legs for an OTO group."""
        oto.status = OTOStatus.ACTIVE
        tasks = [
            self._raw_submit(
                symbol=oto.symbol,
                order_type=spec.get("type", "limit"),
                side=spec.get("side", "sell"),
                amount=spec.get("amount", oto.trigger_qty),
                price=spec.get("price", 0.0),
                params=spec.get("params", {}),
            )
            for spec in oto.dependent_specs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(
                    "ContingencyExecutor: OTO dependent %d submit failed group=%s: %s",
                    i, oto.group_id, res,
                )
                if self._metrics:
                    self._metrics.inc_error(kind="execution")
            else:
                if res:
                    oto.submitted_dependent_ids.append(str(res))
        logger.info(
            "ContingencyExecutor: OTO dependents activated group=%s submitted=%d",
            oto.group_id, len(oto.submitted_dependent_ids),
        )

    async def _amend_trailing_stop(
        self,
        leg: OrderLeg,
        symbol: str,
        new_stop_price: float,
    ) -> None:
        """Cancel-replace a trailing stop leg with the new stop price."""
        old_exchange_id = self._id_map_rev.get(leg.order_id)
        if old_exchange_id:
            await self._raw_cancel(old_exchange_id, symbol)
        new_exchange_id = await self._raw_submit(
            symbol=symbol,
            order_type="stop",
            side=leg.side,
            amount=leg.quantity,
            price=new_stop_price,
            params={},
        )
        if new_exchange_id:
            # Update mappings
            if old_exchange_id:
                del self._id_map[old_exchange_id]
            self._id_map[new_exchange_id]      = leg.order_id
            self._id_map_rev[leg.order_id]     = new_exchange_id
        logger.debug(
            "ContingencyExecutor: trailing stop amended leg=%s new_stop=%.4f",
            leg.order_id, new_stop_price,
        )

    async def _raw_submit(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Call connector.create_order(); return exchange order ID or None."""
        if self._connector is None:
            synth_id = f"sim-{uuid.uuid4().hex[:8]}"
            logger.info(
                "ContingencyExecutor [SIM]: create_order symbol=%s type=%s side=%s "
                "amount=%.6f price=%.4f params=%s → %s",
                symbol, order_type, side, amount, price, params, synth_id,
            )
            return synth_id
        try:
            fn = getattr(self._connector, "create_order", None)
            if fn is None:
                return None
            result = fn(symbol, order_type, side, amount, price, params or {})
            if asyncio.iscoroutine(result):
                result = await result
            return str(result.get("id", "")) if isinstance(result, dict) else str(result)
        except Exception:
            logger.exception(
                "ContingencyExecutor: create_order failed symbol=%s type=%s side=%s",
                symbol, order_type, side,
            )
            if self._metrics:
                self._metrics.inc_error(kind="execution")
            return None

    async def _raw_cancel(self, order_id: str, symbol: str) -> bool:
        """Call connector.cancel_order(); returns True on success."""
        if self._connector is None:
            logger.info(
                "ContingencyExecutor [SIM]: cancel_order order_id=%s symbol=%s",
                order_id, symbol,
            )
            return True
        try:
            fn = getattr(self._connector, "cancel_order", None)
            if fn is None:
                return False
            result = fn(order_id, symbol)
            if asyncio.iscoroutine(result):
                await result
            return True
        except Exception:
            logger.exception(
                "ContingencyExecutor: cancel_order failed order_id=%s symbol=%s",
                order_id, symbol,
            )
            if self._metrics:
                self._metrics.inc_error(kind="execution")
            return False

    @staticmethod
    def _leg_by_type(
        group: ConditionalGroup,
        leg_type: LegType,
    ) -> Optional[OrderLeg]:
        for leg in group.legs:
            if leg.leg_type == leg_type:
                return leg
        return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Full state snapshot: ConditionalOrderManager + OTO groups."""
        cond_snap = self._cond_mgr.snapshot()
        oto_snap = [
            {
                "group_id":    oto.group_id,
                "symbol":      oto.symbol,
                "exchange":    oto.exchange,
                "status":      oto.status.value,
                "trigger_id":  oto.trigger_order_id,
                "trigger_price": oto.trigger_price,
                "trigger_qty": oto.trigger_qty,
                "dependents":  len(oto.dependent_specs),
                "submitted":   len(oto.submitted_dependent_ids),
            }
            for oto in self._oto_groups.values()
        ]
        return {"conditional": cond_snap, "oto": oto_snap}
