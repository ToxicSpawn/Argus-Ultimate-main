"""Paper Exchange — realistic fill simulation for dry-run / paper trading.

Improvements over the original flat-random slippage model:
1. ATR-scaled slippage — fill adversity is proportional to current ATR
   (ATR * slip_atr_factor, default 0.05). In low-vol markets slippage
   is tight; in high-vol (crash) it widens automatically. Replaces the
   hardcoded uniform(0.0002, 0.0015) that was flat regardless of regime.
2. Volume-impact model — orders >0.5% of 24h average volume get an
   additional market-impact penalty (sqrt(order_size/adv) * impact_factor).
   This prevents the backtest from assuming unlimited liquidity.
3. TWAP child-order splitting — orders flagged with params={"twap": True}
   are split into N child fills spread across sim_twap_slices intervals,
   each with independent slippage. Returns a synthetic "parent" fill.
4. Maker/taker fee routing — limit orders that would fill passively get
   maker fee (0.016%); market orders get taker fee (0.026%). The original
   used 0.16%/0.26% (10x too high — Binance VIP0 is 0.1%/0.1%).
5. All original create_order / fetch_order / cancel_order / fetch_orders
   API preserved for backward compatibility.
"""
from __future__ import annotations

import logging
import math
import random as _rng
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fee schedule (Binance VIP0 defaults)
# ---------------------------------------------------------------------------
TAKER_FEE  = 0.00100   # 0.10%
MAKER_FEE  = 0.00080   # 0.08%

# Slippage: base is ATR * this factor
SLIP_ATR_FACTOR   = 0.05   # 5% of current ATR
SLIP_MAX_PCT      = 0.003  # hard cap 0.30% adverse

# Volume impact: sqrt(order_notional / adv_notional) * this factor
VOLUME_IMPACT_FACTOR = 0.10
ADV_THRESHOLD_PCT    = 0.005   # orders > 0.5% of ADV get impact penalty

# TWAP defaults
DEFAULT_TWAP_SLICES = 4

# Partial fill: 90-100% of requested amount
FILL_RATE_MIN = 0.90


class _PaperCCXTWrapper:
    """
    Wraps a CCXT async exchange so create_order/fetch_order return realistic
    mock results for paper / dry-run trading.

    All fetch_ticker / fetch_ohlcv calls delegate to the real exchange.
    """

    def __init__(
        self,
        exchange:          Any,
        name:              str   = "ccxt",
        slip_atr_factor:   float = SLIP_ATR_FACTOR,
        slip_max_pct:      float = SLIP_MAX_PCT,
        taker_fee:         float = TAKER_FEE,
        maker_fee:         float = MAKER_FEE,
        twap_slices:       int   = DEFAULT_TWAP_SLICES,
    ) -> None:
        self._exchange        = exchange
        self._name            = name
        self._slip_atr_factor = slip_atr_factor
        self._slip_max_pct    = slip_max_pct
        self._taker_fee       = taker_fee
        self._maker_fee       = maker_fee
        self._twap_slices     = twap_slices
        self._paper_orders:   Dict[str, Dict[str, Any]] = {}

        # Per-symbol ATR cache — updated by callers via set_atr()
        self._atr_cache: Dict[str, float] = {}
        # Per-symbol 24h ADV notional cache
        self._adv_cache: Dict[str, float] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._exchange, name)

    # ------------------------------------------------------------------
    # ATR / ADV update hooks (call from your bar-close handler)
    # ------------------------------------------------------------------

    def set_atr(self, symbol: str, atr: float) -> None:
        """Update the ATR used for slippage scaling for this symbol."""
        self._atr_cache[symbol] = float(atr)

    def set_adv(self, symbol: str, adv_notional: float) -> None:
        """Update 24h average daily volume (notional) for this symbol."""
        self._adv_cache[symbol] = float(adv_notional)

    # ------------------------------------------------------------------
    # Core order simulation
    # ------------------------------------------------------------------

    async def create_order(
        self,
        symbol:  str,
        type:    str,
        side:    str,
        amount:  float,
        price:   Optional[float] = None,
        params:  Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params   = params or {}
        use_twap = params.get("twap", False)

        # Fetch reference price
        ref_price = await self._fetch_ref_price(symbol, price)

        if use_twap and ref_price and ref_price > 0:
            return await self._simulate_twap(
                symbol, type, side, amount, ref_price, params
            )
        return self._simulate_fill(
            symbol, type, side, amount, ref_price, params
        )

    async def fetch_order(
        self,
        order_id: str,
        symbol:   Optional[str] = None,
        params:   Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if order_id in self._paper_orders:
            return self._paper_orders[order_id]
        return {"id": order_id, "status": "closed", "filled": 0,
                "remaining": 0, "symbol": symbol or ""}

    async def cancel_order(
        self,
        order_id: str,
        symbol:   Optional[str] = None,
        params:   Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if order_id in self._paper_orders:
            self._paper_orders[order_id]["status"] = "canceled"
        return {"id": order_id, "status": "canceled"}

    async def fetch_orders(
        self,
        symbol: Optional[str] = None,
        since:  Optional[int]  = None,
        limit:  Optional[int]  = None,
        params: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        orders = list(self._paper_orders.values())
        if symbol:
            orders = [o for o in orders if o.get("symbol") == symbol]
        return orders[-int(limit or 50):]

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------

    def _simulate_fill(
        self,
        symbol:    str,
        order_type: str,
        side:      str,
        amount:    float,
        ref_price: Optional[float],
        params:    Dict[str, Any],
        child_idx: int = 0,
    ) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)

        fill_price = ref_price or 0.0

        slippage_pct = 0.0
        if fill_price > 0:
            # ATR-scaled base slippage
            atr   = self._atr_cache.get(symbol, fill_price * 0.001)
            base  = (atr / fill_price) * self._slip_atr_factor
            # Add gaussian noise
            noise = abs(_rng.gauss(0, base * 0.3))
            slippage_pct = min(self._slip_max_pct, base + noise)

            # Volume impact penalty
            adv = self._adv_cache.get(symbol, 0.0)
            if adv > 0:
                order_notional = amount * fill_price
                adv_frac = order_notional / adv
                if adv_frac > ADV_THRESHOLD_PCT:
                    impact = math.sqrt(adv_frac) * VOLUME_IMPACT_FACTOR
                    slippage_pct = min(self._slip_max_pct, slippage_pct + impact)

            if side == "buy":
                fill_price *= (1.0 + slippage_pct)
            else:
                fill_price *= (1.0 - slippage_pct)

        # Partial fill (90-100%)
        fill_rate     = _rng.uniform(FILL_RATE_MIN, 1.0)
        filled_amount = float(amount) * fill_rate
        remaining     = float(amount) - filled_amount

        # Fee routing
        is_passive = order_type in ("limit", "postOnly")
        fee_rate   = self._maker_fee if is_passive else self._taker_fee

        sim_latency_ms = _rng.randint(20, 200)

        logger.info(
            "PAPER[%s] #%d %s %s %s amt=%.6f fill=%.6f@%.4f slip=%.4f%% fee=%.4f%% lat=%dms",
            self._name, child_idx, order_type, side, symbol,
            amount, filled_amount, fill_price,
            slippage_pct * 100, fee_rate * 100, sim_latency_ms,
        )

        order = {
            "id":        f"paper_{self._name}_{now_ms}_{child_idx}",
            "symbol":    symbol,
            "type":      order_type,
            "side":      side,
            "amount":    float(amount),
            "price":     float(fill_price) if fill_price else None,
            "average":   float(fill_price) if fill_price else None,
            "cost":      filled_amount * float(fill_price) if fill_price else 0.0,
            "status":    "closed" if remaining < 1e-8 else "partially_filled",
            "filled":    filled_amount,
            "remaining": remaining,
            "timestamp": now_ms,
            "fee": {
                "cost":     filled_amount * float(fill_price or 0.0) * fee_rate,
                "rate":     fee_rate,
                "currency": "USD",
            },
            "info": {
                "simulated_latency_ms": sim_latency_ms,
                "slippage_pct":         round(slippage_pct * 100, 6),
                "fill_rate":            fill_rate,
                "atr_used":             self._atr_cache.get(symbol),
                "fee_type":             "maker" if is_passive else "taker",
                "child_idx":            child_idx,
            },
        }
        self._paper_orders[order["id"]] = order
        return order

    async def _simulate_twap(
        self,
        symbol:    str,
        order_type: str,
        side:      str,
        amount:    float,
        ref_price: float,
        params:    Dict[str, Any],
    ) -> Dict[str, Any]:
        """Split into N child fills and return a synthetic parent order."""
        n          = self._twap_slices
        slice_amt  = amount / n
        children   = []
        total_cost = 0.0
        total_fill = 0.0
        total_fee  = 0.0

        for i in range(n):
            child = self._simulate_fill(
                symbol, order_type, side, slice_amt, ref_price, params, child_idx=i
            )
            children.append(child)
            total_cost += child["cost"]
            total_fill += child["filled"]
            total_fee  += child["fee"]["cost"]

        avg_price = total_cost / total_fill if total_fill > 0 else ref_price
        now_ms    = int(time.time() * 1000)

        parent = {
            "id":        f"paper_{self._name}_{now_ms}_twap",
            "symbol":    symbol,
            "type":      "twap",
            "side":      side,
            "amount":    float(amount),
            "price":     avg_price,
            "average":   avg_price,
            "cost":      total_cost,
            "status":    "closed",
            "filled":    total_fill,
            "remaining": amount - total_fill,
            "timestamp": now_ms,
            "fee":       {"cost": total_fee, "currency": "USD"},
            "info":      {"twap_children": len(children), "slices": n},
        }
        self._paper_orders[parent["id"]] = parent
        logger.info(
            "PAPER[%s] TWAP parent: %s %s avg=%.4f total_fill=%.6f",
            self._name, side, symbol, avg_price, total_fill,
        )
        return parent

    async def _fetch_ref_price(
        self,
        symbol: str,
        price:  Optional[float],
    ) -> Optional[float]:
        if price and price > 0:
            return float(price)
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return float(ticker.get("last", 0) or 0) or None
        except Exception:
            return None
