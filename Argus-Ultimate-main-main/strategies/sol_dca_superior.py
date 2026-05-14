"""
SOL Safe-DCA Strategy — beats 3Commas DCA while avoiding its capital-lock failure mode.

How 3Commas DCA works (and fails):
  - Opens base order, then adds safety orders on fixed % drops
  - Never closes at a loss — keeps averaging until TP hit or capital exhausted
  - In a sustained downtrend, ALL capital is locked in a losing deal

How Safe-DCA improves on it:
  1. ATR-scaled entry spacing (not fixed %) — adapts to SOL's volatility regime
  2. Hard cap of 3 safety orders maximum (total deal = 4x base size)
  3. Deal-level stop-loss at 12% below avg entry (3Commas has no deal SL)
  4. TP ladder via sol_tp_ladder.py: 40% at 1.5%, 40% at 3%, 20% trails
  5. SOL regime filter: only activates in RANGING or mild TRENDING regime
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


MAX_SAFETY_ORDERS = 3
BASE_ORDER_PCT = 0.18        # 18% of available capital per base order
SAFETY_ORDER_SCALE = 1.5     # each safety order = prev * scale
DEAL_STOP_PCT = 0.12         # close entire deal if price 12% below avg entry
ATR_ENTRY_MULTIPLIER = 1.2   # space safety orders 1.2x ATR apart


@dataclass
class SafeDCADeal:
    symbol: str
    base_entry: float
    base_qty: float
    atr_at_open: float
    entries: List[float] = field(default_factory=list)
    quantities: List[float] = field(default_factory=list)
    filled_orders: int = 0
    open_time: float = field(default_factory=time.time)

    def avg_entry(self) -> float:
        if not self.entries:
            return self.base_entry
        total_cost = sum(e * q for e, q in zip(self.entries, self.quantities))
        total_qty = sum(self.quantities)
        return total_cost / total_qty if total_qty > 0 else self.base_entry

    def total_qty(self) -> float:
        return sum(self.quantities)

    def deal_stop_price(self) -> float:
        return self.avg_entry() * (1.0 - DEAL_STOP_PCT)

    def safety_order_price(self, order_num: int) -> float:
        """Price at which to place safety order N (1-indexed)."""
        # Space by ATR * multiplier, increasing per order
        spacing = self.atr_at_open * ATR_ENTRY_MULTIPLIER * order_num
        return self.base_entry - spacing

    def safety_order_qty(self, order_num: int) -> float:
        """Quantity for safety order N (geometric scale from base qty)."""
        return self.base_qty * (SAFETY_ORDER_SCALE ** order_num)

    def can_add_safety_order(self) -> bool:
        return self.filled_orders < MAX_SAFETY_ORDERS


class SolSafeDCAStrategy:
    """
    Safe-DCA strategy optimised for SOL/USD.

    Usage in the strategy router:
        strategy = SolSafeDCAStrategy(capital_aud=1000.0)
        # On each tick: strategy.on_tick(symbol, price, atr, regime, available_capital)
    """

    strategy_id = "sol_safe_dca"
    target_symbol = "SOL/USD"
    enabled_regimes = {"ranging", "mild_trending"}

    def __init__(self, capital_aud: float) -> None:
        self.capital_aud = capital_aud
        self.active_deal: Optional[SafeDCADeal] = None
        self.completed_deals: List[Dict] = []

    # ------------------------------------------------------------------
    def on_tick(
        self,
        price: float,
        atr: float,
        regime: str,
        available_capital: float,
    ) -> Optional[Dict]:
        """
        Call on every price update. Returns an order dict or None.

        Order dict schema:
          {'action': 'buy'|'sell', 'symbol': str, 'qty': float,
           'price': float, 'reason': str}
        """
        if regime not in self.enabled_regimes:
            return None

        # No active deal — open base order
        if self.active_deal is None:
            return self._open_base_order(price, atr, available_capital)

        deal = self.active_deal

        # Deal stop hit — exit entire position
        if price <= deal.deal_stop_price():
            return self._close_deal(deal, price, reason="deal_stop")

        # Safety order trigger
        if deal.can_add_safety_order():
            next_safety_num = deal.filled_orders + 1
            trigger_price = deal.safety_order_price(next_safety_num)
            if price <= trigger_price:
                qty = deal.safety_order_qty(next_safety_num)
                if qty * price <= available_capital:
                    deal.entries.append(price)
                    deal.quantities.append(qty)
                    deal.filled_orders += 1
                    return {
                        "action": "buy",
                        "symbol": self.target_symbol,
                        "qty": qty,
                        "price": price,
                        "reason": f"safety_order_{next_safety_num}",
                        "strategy_id": self.strategy_id,
                    }

        return None

    def on_tp_hit(self, price: float, tier: int, qty_fraction: float) -> Optional[Dict]:
        """Called by sol_tp_ladder when a TP tier is triggered."""
        if self.active_deal is None:
            return None
        sell_qty = self.active_deal.total_qty() * qty_fraction
        return {
            "action": "sell",
            "symbol": self.target_symbol,
            "qty": sell_qty,
            "price": price,
            "reason": f"tp_ladder_tier_{tier}",
            "strategy_id": self.strategy_id,
        }

    # ------------------------------------------------------------------
    def _open_base_order(self, price: float, atr: float, available_capital: float) -> Optional[Dict]:
        base_capital = available_capital * BASE_ORDER_PCT
        qty = base_capital / price
        if qty <= 0 or base_capital > available_capital:
            return None
        self.active_deal = SafeDCADeal(
            symbol=self.target_symbol,
            base_entry=price,
            base_qty=qty,
            atr_at_open=atr,
            entries=[price],
            quantities=[qty],
            filled_orders=0,
        )
        return {
            "action": "buy",
            "symbol": self.target_symbol,
            "qty": qty,
            "price": price,
            "reason": "base_order",
            "strategy_id": self.strategy_id,
        }

    def _close_deal(self, deal: SafeDCADeal, price: float, reason: str) -> Dict:
        pnl_pct = (price - deal.avg_entry()) / deal.avg_entry()
        self.completed_deals.append({
            "symbol": deal.symbol,
            "avg_entry": deal.avg_entry(),
            "exit_price": price,
            "pnl_pct": pnl_pct,
            "safety_orders_used": deal.filled_orders,
            "reason": reason,
        })
        self.active_deal = None
        return {
            "action": "sell",
            "symbol": self.target_symbol,
            "qty": deal.total_qty(),
            "price": price,
            "reason": reason,
            "strategy_id": self.strategy_id,
        }
