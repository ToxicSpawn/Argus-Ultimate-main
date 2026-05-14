"""
Avellaneda-Stoikov Optimal Market Making Strategy.

Computes reservation price and optimal bid-ask spread based on:
  r(s, q, t) = s - q * γ * σ² * (T - t)           [reservation price]
  δ*(t)      = γσ²(T-t) + (2/γ) * ln(1 + γ/k)     [optimal half-spread]

Parameters:
  s  = current mid price
  q  = current inventory (positive = long, negative = short)
  γ  = risk aversion coefficient (higher = tighter inventory management)
  σ  = short-term volatility (per unit time)
  T  = session length (normalised to 1.0)
  t  = elapsed fraction of session
  k  = order book depth parameter

Reference:
  Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
  https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Strategy defaults
GAMMA_DEFAULT   = 0.1    # risk aversion
K_DEFAULT       = 1.5    # order book depth
T_SESSION       = 3600   # 1-hour virtual session in seconds
SIGMA_WINDOW    = 20     # bars for vol estimation
MAX_INVENTORY   = 5.0    # units — quote-skew kicks in near this
MIN_SPREAD_BPS  = 8.0    # minimum quote spread in basis points
MAX_SPREAD_BPS  = 80.0   # maximum quote spread (protect against low-vol collapse)


class AvellanedaStoikovMM:
    """
    Optimal market making strategy for a single symbol.

    Quotes bid and ask around a reservation price that adjusts
    for current inventory, penalising overexposure.
    """

    def __init__(
        self,
        symbol: str = "BTC/USD",
        gamma: float = GAMMA_DEFAULT,
        k: float = K_DEFAULT,
        session_seconds: float = T_SESSION,
        max_inventory: float = MAX_INVENTORY,
        min_spread_bps: float = MIN_SPREAD_BPS,
    ):
        self.symbol = symbol
        self.gamma = gamma
        self.k = k
        self.session_seconds = session_seconds
        self.max_inventory = max_inventory
        self.min_spread_bps = min_spread_bps

        self.inventory: float = 0.0
        self.current_inventory: float = 0.0  # alias tracked via update_inventory
        self._mid_prices: Deque[float] = deque(maxlen=SIGMA_WINDOW + 5)
        self._session_start: float = time.time()
        self._fills: List[Dict[str, Any]] = []
        self.total_pnl: float = 0.0
        self._last_mid: float = 0.0

    def _estimate_sigma(self) -> float:
        """Estimate per-second volatility from log returns of recent mid prices."""
        if len(self._mid_prices) < 3:
            return 0.0001  # 1bp fallback
        prices = np.array(list(self._mid_prices))
        log_returns = np.diff(np.log(prices))
        if len(log_returns) == 0:
            return 0.0001
        # Return per-second vol (assume 1 bar ≈ 60 seconds for crypto)
        bar_vol = float(np.std(log_returns))
        return bar_vol / math.sqrt(60.0)

    def _time_remaining(self) -> float:
        """Fraction of session remaining [0.001, 1.0]."""
        elapsed = time.time() - self._session_start
        remaining = max(self.session_seconds - elapsed, 1.0)
        t_rem = remaining / self.session_seconds
        # Reset session after it expires
        if elapsed >= self.session_seconds:
            self._session_start = time.time()
            t_rem = 1.0
        return max(t_rem, 0.001)

    def reservation_price(self, mid: float) -> float:
        """
        Compute inventory-adjusted fair value.

        The reservation price shifts away from mid in the direction
        that would reduce inventory, encouraging mean-reversion of position.
        """
        sigma = self._estimate_sigma()
        t_rem = self._time_remaining()
        return mid - self.inventory * self.gamma * sigma ** 2 * t_rem

    def optimal_spread(self, mid: float) -> float:
        """
        Compute optimal bid-ask half-spread.

        Returns the half-spread (add/subtract from reservation price).
        """
        sigma = self._estimate_sigma()
        t_rem = self._time_remaining()

        try:
            term1 = self.gamma * sigma ** 2 * t_rem
            term2 = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
            half_spread = (term1 + term2) / 2.0
        except (ValueError, ZeroDivisionError):
            half_spread = 0.001 * mid  # 10bps fallback

        # Clamp to [min, max] bps
        min_half = (self.min_spread_bps / 20000.0) * mid  # min_bps / 2 sides / 10000
        max_half = (MAX_SPREAD_BPS / 20000.0) * mid
        return float(np.clip(half_spread, min_half, max_half))

    def _inventory_skew(self) -> Tuple[float, float]:
        """
        When inventory is building up, skew quotes to attract the offsetting side.

        Returns (bid_skew, ask_skew) — positive = shift price outward.

        At 80% of max_inventory, aggressive skewing kicks in to encourage
        unwinding the heavy side.
        """
        if self.max_inventory == 0:
            return 0.0, 0.0
        inv_ratio = self.inventory / self.max_inventory  # [-1, 1]
        abs_ratio = abs(inv_ratio)

        # Aggressive skew when above 80% capacity
        if abs_ratio > 0.8:
            skew_factor = abs_ratio * 0.5  # up to 50% quote adjustment
            if inv_ratio > 0:
                return -skew_factor, -skew_factor * 2.5
            else:
                return skew_factor * 2.5, skew_factor

        skew_factor = abs_ratio * 0.3  # up to 30% quote adjustment
        if inv_ratio > 0.6:
            # Long heavy: push ask down (sell cheaper), pull bid down (buy less aggressively)
            return -skew_factor, -skew_factor * 2
        elif inv_ratio < -0.6:
            # Short heavy: push bid up, push ask up
            return skew_factor * 2, skew_factor
        return 0.0, 0.0

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compute optimal quotes given current market state.

        Args:
            market_data: {
                "symbol": str,
                "price": float,      # last trade price or mid
                "bid": float,        # optional best bid
                "ask": float,        # optional best ask
                "volume": float,     # optional recent volume
            }

        Returns:
            Signal dict with bid/ask quotes and metadata, or None if price unavailable.
        """
        price: float = float(market_data.get("price", 0.0))
        if price <= 0:
            return None

        bid_input: float = float(market_data.get("bid", price * 0.9995))
        ask_input: float = float(market_data.get("ask", price * 1.0005))
        mid = (bid_input + ask_input) / 2.0 if bid_input > 0 and ask_input > 0 else price

        self._mid_prices.append(mid)
        self._last_mid = mid

        r_price = self.reservation_price(mid)
        half_spread = self.optimal_spread(mid)
        bid_skew, ask_skew = self._inventory_skew()

        bid_quote = r_price - half_spread * (1.0 + bid_skew)
        ask_quote = r_price + half_spread * (1.0 + ask_skew)

        # Safety: bid must be < ask, both must be > 0
        if bid_quote >= ask_quote or bid_quote <= 0:
            return None

        spread_bps = (ask_quote - bid_quote) / mid * 10000

        # Don't quote when inventory is at hard limit
        if abs(self.inventory) >= self.max_inventory:
            side_to_close = "SELL" if self.inventory > 0 else "BUY"
            logger.warning(
                "MM %s: inventory at limit (%.2f) — skipping quote, need %s",
                self.symbol, self.inventory, side_to_close,
            )
            return None

        return {
            "symbol": self.symbol,
            "action": "QUOTE",
            "bid": round(bid_quote, 8),
            "ask": round(ask_quote, 8),
            "confidence": 0.70,
            "price": mid,
            "source": "market_maker_avellaneda",
            "reservation_price": round(r_price, 8),
            "half_spread": round(half_spread, 8),
            "spread_bps": round(spread_bps, 2),
            "inventory": self.inventory,
            "inventory_ratio": round(self.inventory / self.max_inventory, 3),
            "sigma_per_sec": round(self._estimate_sigma(), 8),
        }

    def record_fill(self, side: str, price: float, amount: float) -> None:
        """
        Record a fill (maker trade) and update inventory + PnL.

        Args:
            side: "buy" (we bought — inventory increases) or "sell"
            price: fill price
            amount: quantity filled in base currency units
        """
        if side.lower() == "buy":
            self.inventory += amount
            cost = -price * amount
        else:
            self.inventory -= amount
            cost = price * amount

        self.total_pnl += cost
        self._fills.append({
            "side": side,
            "price": price,
            "amount": amount,
            "inventory_after": self.inventory,
            "ts": time.time(),
        })

        logger.info(
            "MM fill: %s %.6f @ %.2f | inv=%.4f pnl_est=%.4f",
            side, amount, price, self.inventory, self.total_pnl,
        )

    def update_inventory(self, fill_side: str, fill_qty: float) -> None:
        """
        Update current inventory based on a fill.

        Args:
            fill_side: "buy" (inventory increases) or "sell" (inventory decreases)
            fill_qty: quantity filled in base currency units (always positive)
        """
        if fill_side.lower() == "buy":
            self.inventory += abs(fill_qty)
        else:
            self.inventory -= abs(fill_qty)
        self.current_inventory = self.inventory
        logger.debug(
            "MM %s: inventory updated via update_inventory → %.6f",
            self.symbol, self.inventory,
        )

    def reset_session(self) -> None:
        """Start a new market-making session (resets timer, keeps inventory)."""
        self._session_start = time.time()
        logger.info("MM %s: new session started", self.symbol)

    def get_status(self) -> Dict[str, Any]:
        """Return current market making status."""
        return {
            "symbol": self.symbol,
            "inventory": self.inventory,
            "total_pnl_estimate": self.total_pnl,
            "n_fills": len(self._fills),
            "last_mid": self._last_mid,
            "session_t_remaining": round(self._time_remaining(), 3),
            "estimated_sigma": round(self._estimate_sigma(), 8),
        }
