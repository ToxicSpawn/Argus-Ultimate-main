#!/usr/bin/env python3
"""
FIX-Style Binary Protocol Adapter — lightweight FIX-inspired serialisation.

Minimises serialisation overhead versus REST/JSON by encoding messages as
pipe-delimited ``|tag=value|`` byte strings with a mod-256 checksum, mirroring
the structure of real FIX (SOH-delimited, tag=value pairs).

Classes
-------
FIXTag          — standard FIX tag numbers (enum)
FIXSide         — order side (enum)
FIXOrdType      — order type (enum)
FIXTimeInForce  — time-in-force (enum)
FIXOrderMessage — outbound order (dataclass + encode/decode)
FIXExecutionReport — inbound fill/cancel/reject (dataclass)
BinaryOrderEncoder — zero-copy pre-allocated encoder
FIXAdapter      — ccxt wrapper with FIX-style validation

Usage::

    adapter = FIXAdapter(ccxt_exchange)
    msg = adapter.maker_only_order("BTC/USD", FIXSide.BUY, 30_000.0, 0.1)
    valid, reason = adapter.validate_order(msg)
    ccxt_params = adapter.order_to_ccxt(msg)
"""
from __future__ import annotations

import logging
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FIX Enumerations
# ---------------------------------------------------------------------------

class FIXTag(IntEnum):
    """Key FIX 4.x tag numbers."""
    AVGPX        = 6
    CLORDID      = 11
    CUMQTY       = 14
    ORDTYPE      = 40
    ORDERQTY     = 38
    PRICE        = 44
    SIDE         = 54
    SYMBOL       = 55
    TIMEINFORCE  = 59
    TRANSACTTIME = 60
    EXECTYPE     = 150
    LEAVESQTY    = 151


class FIXSide(IntEnum):
    BUY  = 1
    SELL = 2


class FIXOrdType(IntEnum):
    MARKET     = 1
    LIMIT      = 2
    STOP       = 3
    STOP_LIMIT = 4


class FIXTimeInForce(IntEnum):
    DAY = 0   # Good for day
    GTC = 1   # Good till cancel
    IOC = 3   # Immediate or cancel
    FOK = 4   # Fill or kill
    GTX = 5   # Good till crossing (maker-only / post-only)


class FIXExecType(IntEnum):
    NEW           = 0
    PARTIAL_FILL  = 1
    FILL          = 2
    CANCELED      = 4
    REPLACED      = 5
    REJECTED      = 8
    EXPIRED       = 9


# ---------------------------------------------------------------------------
# FIXOrderMessage — outbound new-order single
# ---------------------------------------------------------------------------

@dataclass
class FIXOrderMessage:
    """Outbound FIX-style order (New Order – Single equivalent).

    Parameters
    ----------
    symbol : str
        Instrument symbol (tag 55).
    side : FIXSide
        BUY or SELL (tag 54).
    order_qty : float
        Order quantity (tag 38).
    price : float
        Limit price; 0.0 for market orders (tag 44).
    ord_type : FIXOrdType
        Order type (tag 40).
    time_in_force : FIXTimeInForce
        Time-in-force (tag 59).
    cl_ord_id : str
        Client order ID (tag 11); auto-generated if empty.
    transact_time : float
        Unix timestamp of the order (tag 60); defaults to now.
    """

    symbol:        str
    side:          FIXSide
    order_qty:     float
    price:         float          = 0.0
    ord_type:      FIXOrdType     = FIXOrdType.LIMIT
    time_in_force: FIXTimeInForce = FIXTimeInForce.GTC
    cl_ord_id:     str            = field(default_factory=lambda: str(uuid.uuid4())[:16])
    transact_time: float          = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Encode as ``|tag=value|`` pipe-delimited bytes.

        In real FIX the delimiter is SOH (0x01); here we use ``|`` for
        readability while keeping the tag=value structure identical.
        """
        parts = [
            f"{FIXTag.CLORDID}={self.cl_ord_id}",
            f"{FIXTag.SYMBOL}={self.symbol}",
            f"{FIXTag.SIDE}={int(self.side)}",
            f"{FIXTag.ORDERQTY}={self.order_qty:.8f}",
            f"{FIXTag.PRICE}={self.price:.8f}",
            f"{FIXTag.ORDTYPE}={int(self.ord_type)}",
            f"{FIXTag.TIMEINFORCE}={int(self.time_in_force)}",
            f"{FIXTag.TRANSACTTIME}={self.transact_time:.6f}",
        ]
        body = "|".join(parts)
        chk  = _mod256_checksum(body.encode())
        return f"|{body}|{FIXTag.CLORDID}=CHKSUM:{chk:03d}|".encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "FIXOrderMessage":
        """Decode a ``to_bytes()``-encoded message."""
        raw = data.decode().strip("|")
        tag_map: Dict[int, str] = {}
        for part in raw.split("|"):
            if "=" not in part:
                continue
            tag_str, _, val = part.partition("=")
            # Skip the CHKSUM pseudo-entry (value starts with 'CHKSUM:')
            if val.startswith("CHKSUM:"):
                continue
            try:
                tag_int = int(tag_str)
                # Only record the first occurrence of each tag
                if tag_int not in tag_map:
                    tag_map[tag_int] = val
            except ValueError:
                pass  # non-numeric tag_str

        return cls(
            symbol        = tag_map[FIXTag.SYMBOL],
            side          = FIXSide(int(tag_map[FIXTag.SIDE])),
            order_qty     = float(tag_map[FIXTag.ORDERQTY]),
            price         = float(tag_map[FIXTag.PRICE]),
            ord_type      = FIXOrdType(int(tag_map[FIXTag.ORDTYPE])),
            time_in_force = FIXTimeInForce(int(tag_map[FIXTag.TIMEINFORCE])),
            cl_ord_id     = tag_map[FIXTag.CLORDID],
            transact_time = float(tag_map[FIXTag.TRANSACTTIME]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict (for logging or JSON serialisation)."""
        return {
            "cl_ord_id":     self.cl_ord_id,
            "symbol":        self.symbol,
            "side":          self.side.name,
            "order_qty":     self.order_qty,
            "price":         self.price,
            "ord_type":      self.ord_type.name,
            "time_in_force": self.time_in_force.name,
            "transact_time": self.transact_time,
        }

    def checksum(self) -> int:
        """Mod-256 checksum of the raw encoded bytes (excluding checksum field)."""
        parts = [
            f"{FIXTag.CLORDID}={self.cl_ord_id}",
            f"{FIXTag.SYMBOL}={self.symbol}",
            f"{FIXTag.SIDE}={int(self.side)}",
            f"{FIXTag.ORDERQTY}={self.order_qty:.8f}",
            f"{FIXTag.PRICE}={self.price:.8f}",
            f"{FIXTag.ORDTYPE}={int(self.ord_type)}",
            f"{FIXTag.TIMEINFORCE}={int(self.time_in_force)}",
            f"{FIXTag.TRANSACTTIME}={self.transact_time:.6f}",
        ]
        return _mod256_checksum("|".join(parts).encode())


# ---------------------------------------------------------------------------
# FIXExecutionReport — inbound fill / cancel / reject
# ---------------------------------------------------------------------------

@dataclass
class FIXExecutionReport:
    """Inbound FIX execution report (tag 150 – ExecType driven).

    Parameters
    ----------
    cl_ord_id : str
        Client order ID echoed back (tag 11).
    exec_type : FIXExecType
        Execution type (tag 150).
    symbol : str
        Instrument (tag 55).
    side : FIXSide
        Side (tag 54).
    leaves_qty : float
        Remaining open quantity (tag 151).
    cum_qty : float
        Cumulative filled quantity (tag 14).
    avg_px : float
        Average fill price (tag 6).
    order_qty : float
        Original order quantity (tag 38).
    transact_time : float
        Transaction time (tag 60).
    text : str
        Free-text reject reason if applicable.
    """

    cl_ord_id:     str
    exec_type:     FIXExecType
    symbol:        str
    side:          FIXSide
    leaves_qty:    float
    cum_qty:       float
    avg_px:        float
    order_qty:     float          = 0.0
    transact_time: float          = field(default_factory=time.time)
    text:          str            = ""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def fill_quantity(self) -> float:
        """Quantity filled in this report."""
        return self.cum_qty

    @property
    def fill_price(self) -> float:
        """Average fill price."""
        return self.avg_px

    @property
    def leaves_quantity(self) -> float:
        """Remaining unfilled quantity."""
        return self.leaves_qty

    def is_fill(self) -> bool:
        """True when the order is completely filled."""
        return self.exec_type == FIXExecType.FILL

    def is_partial_fill(self) -> bool:
        """True when only part of the order has been filled."""
        return self.exec_type == FIXExecType.PARTIAL_FILL

    def is_cancel(self) -> bool:
        """True when the order has been cancelled."""
        return self.exec_type == FIXExecType.CANCELED

    def is_reject(self) -> bool:
        """True when the order was rejected by the venue."""
        return self.exec_type == FIXExecType.REJECTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cl_ord_id":     self.cl_ord_id,
            "exec_type":     self.exec_type.name,
            "symbol":        self.symbol,
            "side":          self.side.name,
            "leaves_qty":    self.leaves_qty,
            "cum_qty":       self.cum_qty,
            "avg_px":        self.avg_px,
            "order_qty":     self.order_qty,
            "transact_time": self.transact_time,
            "text":          self.text,
        }


# ---------------------------------------------------------------------------
# BinaryOrderEncoder — zero-copy pre-allocated encoder
# ---------------------------------------------------------------------------

class BinaryOrderEncoder:
    """Zero-copy binary order encoder using a pre-allocated 512-byte buffer.

    For latency-sensitive paths this avoids repeated heap allocation.
    The caller receives a ``memoryview`` slice; copy it if persistence is
    needed beyond the next encode call.

    Usage::

        enc = BinaryOrderEncoder()
        mv  = enc.encode_new_order("BTC/USD", FIXSide.BUY, 0.1, 30000.0,
                                   FIXOrdType.LIMIT, FIXTimeInForce.GTX)
        raw_bytes = bytes(mv)
    """

    _BUFFER_SIZE = 512

    def __init__(self) -> None:
        self._buf    = bytearray(self._BUFFER_SIZE)
        self._view   = memoryview(self._buf)
        self._seq_no = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_clordid(self) -> str:
        self._seq_no += 1
        return f"ORD{self._seq_no:010d}"

    def _write(self, payload: str) -> memoryview:
        """Write *payload* into buffer and return a memoryview slice."""
        encoded = payload.encode()
        n = len(encoded)
        if n > self._BUFFER_SIZE:
            raise ValueError(f"Encoded order ({n} bytes) exceeds buffer ({self._BUFFER_SIZE})")
        self._buf[:n] = encoded
        return self._view[:n]

    def _build_message(self, fields: Dict[int, str]) -> memoryview:
        parts = [f"{tag}={val}" for tag, val in sorted(fields.items())]
        body  = "|".join(parts)
        chk   = _mod256_checksum(body.encode())
        msg   = f"|{body}|{FIXTag.CLORDID}=CHKSUM:{chk:03d}|"
        return self._write(msg)

    # ------------------------------------------------------------------
    # Public encode methods
    # ------------------------------------------------------------------

    def encode_new_order(
        self,
        symbol:    str,
        side:      FIXSide,
        qty:       float,
        price:     float,
        ord_type:  FIXOrdType,
        tif:       FIXTimeInForce,
    ) -> memoryview:
        """Encode a new order single into the pre-allocated buffer."""
        clordid = self._next_clordid()
        fields  = {
            FIXTag.CLORDID:      clordid,
            FIXTag.SYMBOL:       symbol,
            FIXTag.SIDE:         str(int(side)),
            FIXTag.ORDERQTY:     f"{qty:.8f}",
            FIXTag.PRICE:        f"{price:.8f}",
            FIXTag.ORDTYPE:      str(int(ord_type)),
            FIXTag.TIMEINFORCE:  str(int(tif)),
            FIXTag.TRANSACTTIME: f"{time.time():.6f}",
        }
        return self._build_message(fields)

    def encode_cancel(
        self,
        orig_clordid: str,
        symbol:       str,
    ) -> memoryview:
        """Encode an order cancel request."""
        fields = {
            FIXTag.CLORDID:      self._next_clordid(),
            11000:               orig_clordid,   # OrigClOrdID pseudo-tag 11000
            FIXTag.SYMBOL:       symbol,
            FIXTag.TRANSACTTIME: f"{time.time():.6f}",
            9999:                "CANCEL",        # MsgType pseudo-tag
        }
        return self._build_message(fields)

    def encode_cancel_replace(
        self,
        orig_clordid: str,
        symbol:       str,
        new_price:    float,
        new_qty:      float,
    ) -> memoryview:
        """Encode a cancel-replace (order amendment) request."""
        fields = {
            FIXTag.CLORDID:      self._next_clordid(),
            11000:               orig_clordid,
            FIXTag.SYMBOL:       symbol,
            FIXTag.PRICE:        f"{new_price:.8f}",
            FIXTag.ORDERQTY:     f"{new_qty:.8f}",
            FIXTag.TRANSACTTIME: f"{time.time():.6f}",
            9999:                "CANCEL_REPLACE",
        }
        return self._build_message(fields)

    def benchmark_encode(self, n: int = 100_000) -> float:
        """Benchmark encode_new_order.

        Returns
        -------
        float
            Nanoseconds per operation.
        """
        import timeit
        elapsed = timeit.timeit(
            lambda: self.encode_new_order(
                "BTC/USD", FIXSide.BUY, 0.1, 30_000.0,
                FIXOrdType.LIMIT, FIXTimeInForce.GTC,
            ),
            number=n,
        )
        ns_per_op = (elapsed / n) * 1_000_000_000
        logger.info("BinaryOrderEncoder: %.1f ns/op over %d iterations", ns_per_op, n)
        return ns_per_op


# ---------------------------------------------------------------------------
# FIXAdapter — ccxt wrapper with FIX-style validation
# ---------------------------------------------------------------------------

class FIXAdapter:
    """Wraps a ccxt exchange client with FIX-style pre-flight validation and
    conversion utilities.

    Parameters
    ----------
    exchange : Any
        A ccxt exchange instance (or any object with ``create_order``).
    dry_run : bool
        If True, ``create_order`` is never called.
    """

    _MIN_QTY   = 1e-8
    _MAX_PRICE = 1e9

    def __init__(self, exchange: Any = None, dry_run: bool = True) -> None:
        self._exchange = exchange
        self.dry_run   = dry_run
        self._encoder  = BinaryOrderEncoder()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_order(self, order: FIXOrderMessage) -> Tuple[bool, str]:
        """Pre-flight validation of a FIX order message.

        Returns
        -------
        (bool, str)
            ``(True, "")`` if valid; ``(False, reason)`` otherwise.
        """
        if not order.symbol:
            return False, "Symbol is empty"
        if order.order_qty <= 0:
            return False, f"Invalid order_qty={order.order_qty}"
        if order.order_qty < self._MIN_QTY:
            return False, f"order_qty below minimum ({self._MIN_QTY})"
        if order.ord_type in (FIXOrdType.LIMIT, FIXOrdType.STOP_LIMIT):
            if order.price <= 0:
                return False, f"Limit order requires positive price, got {order.price}"
            if order.price > self._MAX_PRICE:
                return False, f"Price {order.price} exceeds maximum {self._MAX_PRICE}"
        if order.time_in_force == FIXTimeInForce.GTX and order.ord_type != FIXOrdType.LIMIT:
            return False, "GTX (maker-only) requires LIMIT order type"
        if not order.cl_ord_id:
            return False, "ClOrdID is empty"
        return True, ""

    # ------------------------------------------------------------------
    # Conversion utilities
    # ------------------------------------------------------------------

    def order_to_ccxt(self, order: FIXOrderMessage) -> Dict[str, Any]:
        """Convert a FIXOrderMessage to a ccxt ``create_order`` parameter dict."""
        side_str = "buy" if order.side == FIXSide.BUY else "sell"
        type_map = {
            FIXOrdType.MARKET:     "market",
            FIXOrdType.LIMIT:      "limit",
            FIXOrdType.STOP:       "stop",
            FIXOrdType.STOP_LIMIT: "stop_limit",
        }
        order_type = type_map.get(order.ord_type, "limit")

        params: Dict[str, Any] = {"clientOrderId": order.cl_ord_id}

        # Map TIF
        tif_map = {
            FIXTimeInForce.DAY: "Day",
            FIXTimeInForce.GTC: "GTC",
            FIXTimeInForce.IOC: "IOC",
            FIXTimeInForce.FOK: "FOK",
            FIXTimeInForce.GTX: "GTX",  # Kraken post-only
        }
        if order.time_in_force in tif_map:
            params["timeInForce"] = tif_map[order.time_in_force]

        if order.time_in_force == FIXTimeInForce.GTX:
            params["postOnly"] = True

        result: Dict[str, Any] = {
            "symbol":  order.symbol,
            "type":    order_type,
            "side":    side_str,
            "amount":  order.order_qty,
            "params":  params,
        }
        if order.price > 0:
            result["price"] = order.price
        return result

    def ccxt_to_execution_report(self, ccxt_response: Dict[str, Any]) -> FIXExecutionReport:
        """Convert a ccxt order response dict to a FIXExecutionReport.

        Parameters
        ----------
        ccxt_response : dict
            Dict as returned by ``ccxt.create_order`` or ``ccxt.fetch_order``.
        """
        status_map = {
            "open":    FIXExecType.NEW,
            "closed":  FIXExecType.FILL,
            "canceled": FIXExecType.CANCELED,
            "cancelled": FIXExecType.CANCELED,
            "rejected": FIXExecType.REJECTED,
            "expired":  FIXExecType.EXPIRED,
        }
        status    = ccxt_response.get("status", "open")
        filled    = float(ccxt_response.get("filled") or 0.0)
        amount    = float(ccxt_response.get("amount") or 0.0)
        remaining = float(ccxt_response.get("remaining") or (amount - filled))
        avg_price = float(ccxt_response.get("average") or ccxt_response.get("price") or 0.0)

        # Determine ExecType
        if status in ("closed",) and remaining == 0 and filled > 0:
            exec_type = FIXExecType.FILL
        elif filled > 0 and remaining > 0:
            exec_type = FIXExecType.PARTIAL_FILL
        else:
            exec_type = status_map.get(status, FIXExecType.NEW)

        side_raw = ccxt_response.get("side", "buy").lower()
        side     = FIXSide.BUY if side_raw == "buy" else FIXSide.SELL

        return FIXExecutionReport(
            cl_ord_id     = str(ccxt_response.get("clientOrderId") or ccxt_response.get("id", "")),
            exec_type     = exec_type,
            symbol        = str(ccxt_response.get("symbol", "")),
            side          = side,
            leaves_qty    = remaining,
            cum_qty       = filled,
            avg_px        = avg_price,
            order_qty     = amount,
            transact_time = float(ccxt_response.get("timestamp", time.time() * 1000)) / 1000.0,
            text          = str(ccxt_response.get("info", {}).get("reason", "")),
        )

    # ------------------------------------------------------------------
    # Maker-only helper
    # ------------------------------------------------------------------

    def maker_only_order(
        self,
        symbol: str,
        side:   FIXSide,
        price:  float,
        qty:    float,
    ) -> FIXOrderMessage:
        """Build a GTX (post-only) limit order to earn maker rebate.

        Parameters
        ----------
        symbol : str
            Instrument, e.g. ``"BTC/USD"``.
        side : FIXSide
            BUY or SELL.
        price : float
            Limit price.
        qty : float
            Order quantity.

        Returns
        -------
        FIXOrderMessage
            Ready-to-validate/submit order with ``time_in_force=GTX``.
        """
        msg = FIXOrderMessage(
            symbol        = symbol,
            side          = side,
            order_qty     = qty,
            price         = price,
            ord_type      = FIXOrdType.LIMIT,
            time_in_force = FIXTimeInForce.GTX,
        )
        logger.debug("maker_only_order: %s", msg.to_dict())
        return msg

    # ------------------------------------------------------------------
    # Submit (wraps ccxt)
    # ------------------------------------------------------------------

    def submit(self, order: FIXOrderMessage) -> FIXExecutionReport:
        """Validate and (optionally) submit *order* via ccxt.

        In dry-run mode returns a synthetic ACK execution report.
        """
        valid, reason = self.validate_order(order)
        if not valid:
            logger.error("Order validation failed: %s", reason)
            return FIXExecutionReport(
                cl_ord_id  = order.cl_ord_id,
                exec_type  = FIXExecType.REJECTED,
                symbol     = order.symbol,
                side       = order.side,
                leaves_qty = order.order_qty,
                cum_qty    = 0.0,
                avg_px     = 0.0,
                order_qty  = order.order_qty,
                text       = reason,
            )

        if self.dry_run or self._exchange is None:
            logger.info("DRY-RUN: would submit %s", order.to_dict())
            return FIXExecutionReport(
                cl_ord_id  = order.cl_ord_id,
                exec_type  = FIXExecType.NEW,
                symbol     = order.symbol,
                side       = order.side,
                leaves_qty = order.order_qty,
                cum_qty    = 0.0,
                avg_px     = order.price,
                order_qty  = order.order_qty,
            )

        params = self.order_to_ccxt(order)
        try:
            resp = self._exchange.create_order(
                symbol = params["symbol"],
                type   = params["type"],
                side   = params["side"],
                amount = params["amount"],
                price  = params.get("price"),
                params = params.get("params", {}),
            )
            return self.ccxt_to_execution_report(resp)
        except Exception as exc:
            logger.exception("Order submission failed: %s", exc)
            return FIXExecutionReport(
                cl_ord_id  = order.cl_ord_id,
                exec_type  = FIXExecType.REJECTED,
                symbol     = order.symbol,
                side       = order.side,
                leaves_qty = order.order_qty,
                cum_qty    = 0.0,
                avg_px     = 0.0,
                order_qty  = order.order_qty,
                text       = str(exc),
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mod256_checksum(data: bytes) -> int:
    """Return mod-256 sum of all bytes in *data* (standard FIX checksum)."""
    return sum(data) % 256
