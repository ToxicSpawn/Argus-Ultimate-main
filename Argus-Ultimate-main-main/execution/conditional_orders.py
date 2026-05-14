"""
Conditional Orders — OCO (One-Cancels-Other) and bracket order management.

Manages pairs/groups of orders where filling one automatically cancels the
remaining legs.  Works with any exchange connector that exposes a
``cancel_order(order_id, symbol)`` coroutine or method.  Passing
``connector=None`` runs in simulation mode (logs instead of calling exchange).

Supported order group types:
  OCO      — two-leg take-profit + stop-loss pair; first fill cancels the other
  BRACKET  — three-leg: entry order + OCO around it
  TRAILING — trailing stop that ratchets with favourable price movement; never
             moves against the position

Group lifecycle:
  ACTIVE → FILLED (first leg filled, others cancelled)
         → CANCELLED (all legs cancelled externally)
         → PARTIAL (entry filled, OCO legs pending; for BRACKET only)

Thread-safety note: on_fill() and on_price_update() modify internal state.
For concurrent access from multiple threads, wrap calls in an external lock.
"""
from __future__ import annotations

import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LegType(enum.Enum):
    """Role of an individual order leg within a conditional group."""

    ENTRY = "entry"
    TP = "tp"           # take-profit
    SL = "sl"           # stop-loss
    TRAILING = "trailing"


class GroupType(enum.Enum):
    """Type of conditional order group."""

    OCO = "oco"
    BRACKET = "bracket"
    TRAILING = "trailing"


class GroupStatus(enum.Enum):
    """Lifecycle state of a conditional order group."""

    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PARTIAL = "partial"    # entry filled; OCO legs still active (BRACKET only)


class LegStatus(enum.Enum):
    """Status of a single order leg."""

    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OrderLeg:
    """One order leg within a conditional group."""

    order_id: str
    side: str               # "buy" or "sell"
    price: float            # limit / stop price
    quantity: float
    leg_type: LegType
    exchange: str
    status: LegStatus = LegStatus.PENDING
    filled_qty: float = 0.0
    created_ts: float = field(default_factory=time.time)


@dataclass
class ConditionalGroup:
    """A linked set of order legs managed as a single conditional unit."""

    group_id: str
    symbol: str
    legs: List[OrderLeg]
    group_type: GroupType
    created_ts: float = field(default_factory=time.time)
    status: GroupStatus = GroupStatus.ACTIVE


# ---------------------------------------------------------------------------
# ConditionalOrderManager
# ---------------------------------------------------------------------------

class ConditionalOrderManager:
    """
    Manages OCO, bracket, and trailing stop order groups.

    ``connector`` may be any object with a ``cancel_order(order_id, symbol)``
    method (sync or async placeholder).  When ``connector`` is None the manager
    operates in simulation mode and logs all intended actions.

    Typical usage (simulation):
        mgr = ConditionalOrderManager()
        gid = mgr.create_oco("BTC/USD", tp_price=66000, sl_price=63000,
                              quantity=0.1, exchange="kraken")
        mgr.on_fill("tp-order-id", filled_qty=0.1, fill_price=66005)
        # → automatically "cancels" the SL leg (logged in simulation mode)
    """

    def __init__(self, connector: Optional[object] = None) -> None:
        self._connector = connector
        self._groups: Dict[str, ConditionalGroup] = {}  # group_id → group
        self._order_to_group: Dict[str, str] = {}       # order_id → group_id

        mode = "LIVE" if connector is not None else "SIMULATION"
        logger.info("ConditionalOrderManager initialised in %s mode", mode)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def create_oco(
        self,
        symbol: str,
        tp_price: float,
        sl_price: float,
        quantity: float,
        exchange: str,
        side: str = "sell",   # closing side for a long position
    ) -> str:
        """
        Create an OCO (One-Cancels-Other) group.

        Returns the group_id (UUID string).
        The caller is responsible for actually placing the TP and SL orders on
        the exchange and passing the resulting order IDs to this method via the
        ``tp_order_id`` / ``sl_order_id`` parameters if they are already known.
        When IDs are not yet known, the manager generates synthetic IDs for
        simulation/tracking purposes.
        """
        tp_id = f"oco-tp-{uuid.uuid4().hex[:12]}"
        sl_id = f"oco-sl-{uuid.uuid4().hex[:12]}"

        tp_leg = OrderLeg(
            order_id=tp_id,
            side=side,
            price=tp_price,
            quantity=quantity,
            leg_type=LegType.TP,
            exchange=exchange,
        )
        sl_leg = OrderLeg(
            order_id=sl_id,
            side=side,
            price=sl_price,
            quantity=quantity,
            leg_type=LegType.SL,
            exchange=exchange,
        )

        group = ConditionalGroup(
            group_id=str(uuid.uuid4()),
            symbol=symbol,
            legs=[tp_leg, sl_leg],
            group_type=GroupType.OCO,
        )
        self._register_group(group)
        logger.info(
            "ConditionalOrderManager: created OCO group=%s symbol=%s tp=%.4f sl=%.4f qty=%.6f",
            group.group_id, symbol, tp_price, sl_price, quantity,
        )
        return group.group_id

    def create_bracket(
        self,
        symbol: str,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        quantity: float,
        exchange: str,
        entry_side: str = "buy",
    ) -> str:
        """
        Create a bracket order: entry + OCO (TP + SL).

        The OCO legs start as PENDING and become active after the entry fills.
        Returns the group_id.
        """
        close_side = "sell" if entry_side.lower() == "buy" else "buy"

        entry_id = f"brk-entry-{uuid.uuid4().hex[:12]}"
        tp_id = f"brk-tp-{uuid.uuid4().hex[:12]}"
        sl_id = f"brk-sl-{uuid.uuid4().hex[:12]}"

        entry_leg = OrderLeg(
            order_id=entry_id,
            side=entry_side,
            price=entry_price,
            quantity=quantity,
            leg_type=LegType.ENTRY,
            exchange=exchange,
        )
        tp_leg = OrderLeg(
            order_id=tp_id,
            side=close_side,
            price=tp_price,
            quantity=quantity,
            leg_type=LegType.TP,
            exchange=exchange,
        )
        sl_leg = OrderLeg(
            order_id=sl_id,
            side=close_side,
            price=sl_price,
            quantity=quantity,
            leg_type=LegType.SL,
            exchange=exchange,
        )

        group = ConditionalGroup(
            group_id=str(uuid.uuid4()),
            symbol=symbol,
            legs=[entry_leg, tp_leg, sl_leg],
            group_type=GroupType.BRACKET,
        )
        self._register_group(group)
        logger.info(
            "ConditionalOrderManager: created BRACKET group=%s symbol=%s "
            "entry=%.4f tp=%.4f sl=%.4f qty=%.6f",
            group.group_id, symbol, entry_price, tp_price, sl_price, quantity,
        )
        return group.group_id

    def create_trailing_stop(
        self,
        symbol: str,
        trail_pct: float,
        quantity: float,
        exchange: str,
        initial_price: float = 0.0,
        side: str = "sell",
    ) -> str:
        """
        Create a trailing stop order group.

        ``trail_pct`` is the percentage distance from the current high-water
        mark (e.g. 0.02 = 2 %).  ``initial_price`` seeds the high-water mark;
        pass 0.0 to set it lazily on the first on_price_update() call.
        Returns the group_id.
        """
        ts_id = f"trail-{uuid.uuid4().hex[:12]}"
        trail_leg = OrderLeg(
            order_id=ts_id,
            side=side,
            price=0.0,    # computed dynamically
            quantity=quantity,
            leg_type=LegType.TRAILING,
            exchange=exchange,
        )

        group = ConditionalGroup(
            group_id=str(uuid.uuid4()),
            symbol=symbol,
            legs=[trail_leg],
            group_type=GroupType.TRAILING,
        )
        group.__dict__["_trail_pct"] = trail_pct
        group.__dict__["_hwm"] = initial_price   # high-water mark
        group.__dict__["_stop_price"] = (
            initial_price * (1.0 - trail_pct) if initial_price > 0 else 0.0
        )

        self._register_group(group)
        logger.info(
            "ConditionalOrderManager: created TRAILING group=%s symbol=%s "
            "trail_pct=%.2f%% qty=%.6f initial_price=%.4f",
            group.group_id, symbol, trail_pct * 100, quantity, initial_price,
        )
        return group.group_id

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_fill(self, order_id: str, filled_qty: float, fill_price: float) -> None:
        """
        Call when a fill is received from the exchange for ``order_id``.

        Marks the leg as filled; cancels all sibling legs in the group.
        For BRACKET groups: filling the entry leg moves the group to PARTIAL
        and activates the OCO legs rather than cancelling them.
        """
        group_id = self._order_to_group.get(order_id)
        if group_id is None:
            logger.debug(
                "ConditionalOrderManager.on_fill: order_id=%s not tracked", order_id
            )
            return

        group = self._groups.get(group_id)
        if group is None or group.status not in {GroupStatus.ACTIVE, GroupStatus.PARTIAL}:
            return

        filled_leg = self._find_leg(group, order_id)
        if filled_leg is None:
            return

        filled_leg.status = LegStatus.FILLED
        filled_leg.filled_qty = filled_qty
        logger.info(
            "ConditionalOrderManager: leg filled group=%s order=%s type=%s "
            "qty=%.6f price=%.4f",
            group_id, order_id, filled_leg.leg_type.value, filled_qty, fill_price,
        )

        if filled_leg.leg_type == LegType.ENTRY and group.group_type == GroupType.BRACKET:
            # Entry filled: activate OCO legs and mark group as PARTIAL
            group.status = GroupStatus.PARTIAL
            logger.info(
                "ConditionalOrderManager: BRACKET entry filled — OCO legs now active group=%s",
                group_id,
            )
            return

        # For OCO or TP/SL fill: cancel all other pending legs
        for leg in group.legs:
            if leg.order_id != order_id and leg.status == LegStatus.PENDING:
                self._cancel_leg(leg, group.symbol)
                leg.status = LegStatus.CANCELLED

        group.status = GroupStatus.FILLED
        logger.info(
            "ConditionalOrderManager: group FILLED group=%s", group_id
        )

    def on_price_update(self, symbol: str, price: float) -> None:
        """
        Call on every price tick for ``symbol`` to update trailing stops.

        Advances the stop price when the market moves in the favourable
        direction (for TRAILING groups).  Never moves the stop against the
        position.
        """
        for group in self._groups.values():
            if (
                group.symbol != symbol
                or group.group_type != GroupType.TRAILING
                or group.status != GroupStatus.ACTIVE
            ):
                continue

            trail_pct: float = group.__dict__.get("_trail_pct", 0.02)
            hwm: float = group.__dict__.get("_hwm", 0.0)
            stop_price: float = group.__dict__.get("_stop_price", 0.0)

            # Determine position direction from the leg side
            trail_leg = group.legs[0] if group.legs else None
            if trail_leg is None:
                continue
            is_long_position = trail_leg.side.lower() == "sell"

            if is_long_position:
                # Long position: advance HWM upward, advance stop upward
                if price > hwm or hwm == 0.0:
                    new_stop = price * (1.0 - trail_pct)
                    if new_stop > stop_price:
                        logger.debug(
                            "ConditionalOrderManager: trailing stop advanced "
                            "group=%s price=%.4f old_stop=%.4f new_stop=%.4f",
                            group.group_id, price, stop_price, new_stop,
                        )
                        group.__dict__["_hwm"] = price
                        group.__dict__["_stop_price"] = new_stop
                        trail_leg.price = new_stop

                # Check if stop triggered
                if stop_price > 0 and price <= stop_price:
                    logger.warning(
                        "ConditionalOrderManager: trailing stop TRIGGERED "
                        "group=%s symbol=%s price=%.4f stop=%.4f",
                        group.group_id, symbol, price, stop_price,
                    )
                    trail_leg.status = LegStatus.FILLED
                    trail_leg.filled_qty = trail_leg.quantity
                    group.status = GroupStatus.FILLED
            else:
                # Short position: advance HWM downward (track low-water mark)
                if price < hwm or hwm == 0.0:
                    new_stop = price * (1.0 + trail_pct)
                    if stop_price == 0.0 or new_stop < stop_price:
                        logger.debug(
                            "ConditionalOrderManager: short trailing stop advanced "
                            "group=%s price=%.4f old_stop=%.4f new_stop=%.4f",
                            group.group_id, price, stop_price, new_stop,
                        )
                        group.__dict__["_hwm"] = price
                        group.__dict__["_stop_price"] = new_stop
                        trail_leg.price = new_stop

                # Check if stop triggered
                if stop_price > 0 and price >= stop_price:
                    logger.warning(
                        "ConditionalOrderManager: short trailing stop TRIGGERED "
                        "group=%s symbol=%s price=%.4f stop=%.4f",
                        group.group_id, symbol, price, stop_price,
                    )
                    trail_leg.status = LegStatus.FILLED
                    trail_leg.filled_qty = trail_leg.quantity
                    group.status = GroupStatus.FILLED

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def cancel_group(self, group_id: str) -> None:
        """
        Cancel all active legs in a group and mark the group as CANCELLED.

        No-op if the group is already terminal.
        """
        group = self._groups.get(group_id)
        if group is None:
            logger.warning(
                "ConditionalOrderManager.cancel_group: group_id=%s not found", group_id
            )
            return
        if group.status in {GroupStatus.FILLED, GroupStatus.CANCELLED}:
            logger.debug(
                "ConditionalOrderManager.cancel_group: group=%s already terminal (%s)",
                group_id, group.status.value,
            )
            return

        for leg in group.legs:
            if leg.status == LegStatus.PENDING:
                self._cancel_leg(leg, group.symbol)
                leg.status = LegStatus.CANCELLED

        group.status = GroupStatus.CANCELLED
        logger.info(
            "ConditionalOrderManager: group CANCELLED group=%s", group_id
        )

    def get_active_groups(self) -> List[ConditionalGroup]:
        """Return all groups in ACTIVE or PARTIAL state."""
        return [
            g for g in self._groups.values()
            if g.status in {GroupStatus.ACTIVE, GroupStatus.PARTIAL}
        ]

    def snapshot(self) -> Dict:
        """Return a serialisable snapshot of all managed groups."""
        result: Dict = {"timestamp": time.time(), "groups": []}
        for group in self._groups.values():
            result["groups"].append(
                {
                    "group_id": group.group_id,
                    "symbol": group.symbol,
                    "group_type": group.group_type.value,
                    "status": group.status.value,
                    "created_ts": group.created_ts,
                    "legs": [
                        {
                            "order_id": leg.order_id,
                            "side": leg.side,
                            "price": leg.price,
                            "quantity": leg.quantity,
                            "leg_type": leg.leg_type.value,
                            "status": leg.status.value,
                            "filled_qty": leg.filled_qty,
                            "exchange": leg.exchange,
                        }
                        for leg in group.legs
                    ],
                }
            )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register_group(self, group: ConditionalGroup) -> None:
        """Add group and build order_id → group_id index."""
        self._groups[group.group_id] = group
        for leg in group.legs:
            self._order_to_group[leg.order_id] = group.group_id

    @staticmethod
    def _find_leg(group: ConditionalGroup, order_id: str) -> Optional[OrderLeg]:
        """Return the leg matching order_id, or None."""
        for leg in group.legs:
            if leg.order_id == order_id:
                return leg
        return None

    def _cancel_leg(self, leg: OrderLeg, symbol: str) -> None:
        """
        Attempt to cancel a leg via the connector.

        Falls back to simulation logging if the connector is absent or raises.
        """
        if self._connector is None:
            logger.info(
                "ConditionalOrderManager [SIM]: would cancel order_id=%s symbol=%s",
                leg.order_id, symbol,
            )
            return

        try:
            cancel_fn = getattr(self._connector, "cancel_order", None)
            if cancel_fn is None:
                logger.warning(
                    "ConditionalOrderManager: connector has no cancel_order(); "
                    "logging cancellation for order_id=%s",
                    leg.order_id,
                )
                return
            result = cancel_fn(leg.order_id, symbol)
            # Handle coroutines gracefully — do not await (caller is sync)
            if hasattr(result, "__await__") or hasattr(result, "send"):
                logger.warning(
                    "ConditionalOrderManager: cancel_order() returned a coroutine "
                    "for order_id=%s — caller must await in async context",
                    leg.order_id,
                )
            else:
                logger.info(
                    "ConditionalOrderManager: cancelled order_id=%s symbol=%s",
                    leg.order_id, symbol,
                )
        except Exception:
            logger.exception(
                "ConditionalOrderManager: failed to cancel order_id=%s symbol=%s",
                leg.order_id, symbol,
            )
