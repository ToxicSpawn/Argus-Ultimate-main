"""
core/market_maker_spread.py --- ATR-Adaptive Market-Making Spread Manager.

Peak-Potential Batch additions
-------------------------------
* ATR-normalised base spread: spread_bps = k * ATR_pct so quoting auto-widens
  in volatile conditions and tightens in calm regimes without manual tuning.
* Kelly-weighted quote sizing: bid/ask quantities are scaled by a Kelly
  fraction derived from recent fill PnL, preventing over-exposure on one side.
* LOB imbalance skew: the mid-quote is shifted proportionally to order-book
  bid/ask volume imbalance, reducing adverse selection.
* Regime gating: four regime states (TRENDING_UP, TRENDING_DOWN, RANGING,
  HIGH_VOL_CRASH) gate whether MM is active and by how much spread widens.
* Inventory management: tracks net inventory and widens the spread on the
  heavy side to mean-revert position passively.
* signal_confidence hook: if an external directional signal is present, the
  spread is tightened on the favoured side and widened on the opposing side.
* autonomous_brain integration: exposes a mm_spread_state() dict that can be
  placed directly into market_state for AutonomousBrain consumption.

Standalone — pure Python + numpy, no hard imports on the rest of Argus.

Usage::

    mgr = MarketMakerSpreadManager(symbol="BTC/USDT", config=cfg)

    # Each tick:
    quote = mgr.compute_quote(
        mid_price=67_000.0,
        atr_pct=0.0035,          # ATR as fraction of price
        lob_bid_vol=12.5,        # total bid depth (e.g. top-5 levels)
        lob_ask_vol=8.0,
        regime="RANGING",
        signal_confidence=0.0,   # -1..+1 directional bias, 0 = neutral
        kelly_fraction=0.08,     # from KellySizer.compute().position_pct
    )
    print(quote.bid_price, quote.ask_price, quote.bid_qty, quote.ask_qty)

    # After a fill:
    mgr.record_fill(side="buy", qty=0.1, price=66_990.0, pnl=5.2)

    # Feed into AutonomousBrain:
    market_state["mm_spread"] = mgr.mm_spread_state()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regime multipliers for spread width (wider = more conservative quoting)
_REGIME_SPREAD_MULT: Dict[str, float] = {
    "TRENDING_UP":    1.4,   # directional flow → adverse selection risk ↑
    "TRENDING_DOWN":  1.4,
    "RANGING":        1.0,   # ideal MM conditions
    "COILED":         1.2,   # pre-breakout uncertainty
    "HIGH_VOL_CRASH": 3.0,   # protect inventory at all costs
    "UNKNOWN":        1.5,
}

# Regime gate: MM is active only in these regimes
_REGIME_ACTIVE: Dict[str, bool] = {
    "TRENDING_UP":    True,
    "TRENDING_DOWN":  True,
    "RANGING":        True,
    "COILED":         False,  # don't MM into a potential breakout
    "HIGH_VOL_CRASH": False,
    "UNKNOWN":        False,
}

_MIN_SPREAD_BPS: float = 2.0
_MAX_SPREAD_BPS: float = 500.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MMQuote:
    """A single market-making two-sided quote."""
    symbol: str
    mid_price: float
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float
    spread_bps: float
    skew_bps: float           # mid-price shift applied
    regime: str
    active: bool              # False = do not submit quote
    reason: str               # human-readable explanation
    timestamp: float = field(default_factory=time.time)

    @property
    def spread_pct(self) -> float:
        return self.spread_bps / 10_000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "mid_price": self.mid_price,
            "bid_price": round(self.bid_price, 8),
            "ask_price": round(self.ask_price, 8),
            "bid_qty": round(self.bid_qty, 8),
            "ask_qty": round(self.ask_qty, 8),
            "spread_bps": round(self.spread_bps, 3),
            "skew_bps": round(self.skew_bps, 3),
            "regime": self.regime,
            "active": self.active,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class FillRecord:
    """A single fill recorded for PnL tracking."""
    side: str          # "buy" or "sell"
    qty: float
    price: float
    pnl: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# MarketMakerSpreadManager
# ---------------------------------------------------------------------------

class MarketMakerSpreadManager:
    """ATR-adaptive, Kelly-weighted, LOB-skewed market-making spread manager.

    Parameters
    ----------
    symbol : str
        The trading pair this manager quotes (e.g. ``"BTC/USDT"``).
    base_spread_k : float
        Spread multiplier: ``spread_bps = base_spread_k * ATR_pct * 10_000``.
        Default 1.5 means spread = 1.5× the ATR expressed in bps.
    base_qty : float
        Default quote quantity on each side (in base asset units).
    inventory_skew_k : float
        How aggressively to widen the heavy side per unit of net inventory.
        ``spread_add_bps = inventory_skew_k * |net_inventory| / base_qty``.
    lob_skew_k : float
        LOB imbalance → mid-price skew coefficient.
        ``skew_bps = lob_skew_k * imbalance`` where imbalance ∈ [-1, 1].
    signal_skew_k : float
        Directional signal → asymmetric spread tightening/widening coefficient.
    kelly_qty_blend : float
        Weight in [0, 1] for Kelly-sized qty vs base_qty.
        0 = always use base_qty, 1 = fully Kelly-driven.
    max_inventory : float
        Hard inventory cap in base asset units.  Quoting on the heavy side
        is suspended when |net_inventory| >= max_inventory.
    fill_window : int
        Rolling window for fill PnL tracking (used in snapshot / brain feed).
    config : dict or object, optional
        Unified config — reads ``market_maker_spread`` section for overrides.
    """

    def __init__(
        self,
        symbol: str,
        base_spread_k: float = 1.5,
        base_qty: float = 0.01,
        inventory_skew_k: float = 10.0,
        lob_skew_k: float = 5.0,
        signal_skew_k: float = 8.0,
        kelly_qty_blend: float = 0.5,
        max_inventory: float = 1.0,
        fill_window: int = 200,
        config: Any = None,
    ) -> None:
        self.symbol = symbol

        # Apply config overrides if present
        cfg = self._extract_cfg(config)
        self._k = float(cfg.get("base_spread_k", base_spread_k))
        self._base_qty = float(cfg.get("base_qty", base_qty))
        self._inv_skew_k = float(cfg.get("inventory_skew_k", inventory_skew_k))
        self._lob_skew_k = float(cfg.get("lob_skew_k", lob_skew_k))
        self._sig_skew_k = float(cfg.get("signal_skew_k", signal_skew_k))
        self._kelly_blend = float(cfg.get("kelly_qty_blend", kelly_qty_blend))
        self._max_inv = float(cfg.get("max_inventory", max_inventory))
        self._fill_window = int(cfg.get("fill_window", fill_window))

        # State
        self._net_inventory: float = 0.0
        self._fills: deque = deque(maxlen=self._fill_window)
        self._quote_count: int = 0
        self._last_quote: Optional[MMQuote] = None
        self._last_atr_pct: float = 0.001
        self._cumulative_pnl: float = 0.0

        logger.info(
            "MarketMakerSpreadManager(%s): k=%.2f base_qty=%.4f "
            "inv_skew_k=%.1f lob_skew_k=%.1f max_inv=%.4f",
            symbol, self._k, self._base_qty,
            self._inv_skew_k, self._lob_skew_k, self._max_inv,
        )

    # ------------------------------------------------------------------
    # Core quoting
    # ------------------------------------------------------------------

    def compute_quote(
        self,
        mid_price: float,
        atr_pct: float,
        lob_bid_vol: float = 0.0,
        lob_ask_vol: float = 0.0,
        regime: str = "UNKNOWN",
        signal_confidence: float = 0.0,
        kelly_fraction: float = 0.0,
    ) -> MMQuote:
        """Compute a two-sided market-making quote.

        Parameters
        ----------
        mid_price : float
            Current fair mid-price.
        atr_pct : float
            ATR expressed as a fraction of mid_price (e.g. 0.0035 = 0.35%).
        lob_bid_vol : float
            Aggregate bid-side depth (any consistent unit).
        lob_ask_vol : float
            Aggregate ask-side depth.
        regime : str
            Current market regime string.
        signal_confidence : float
            Directional signal in [-1, 1]: +1 = strong buy, -1 = strong sell.
        kelly_fraction : float
            Kelly-sized position fraction from KellySizer (used to scale qty).
        """
        self._quote_count += 1
        self._last_atr_pct = max(atr_pct, 1e-6)

        # --- 1. Regime gate -----------------------------------------------
        active = _REGIME_ACTIVE.get(regime, False)
        regime_mult = _REGIME_SPREAD_MULT.get(regime, 1.5)

        if not active:
            return MMQuote(
                symbol=self.symbol,
                mid_price=mid_price,
                bid_price=mid_price,
                ask_price=mid_price,
                bid_qty=0.0,
                ask_qty=0.0,
                spread_bps=0.0,
                skew_bps=0.0,
                regime=regime,
                active=False,
                reason=f"MM inactive in regime '{regime}'",
            )

        # --- 2. ATR-normalised base spread --------------------------------
        base_spread_bps = self._k * atr_pct * 10_000.0 * regime_mult
        base_spread_bps = float(np.clip(base_spread_bps, _MIN_SPREAD_BPS, _MAX_SPREAD_BPS))

        # --- 3. Inventory spread widening --------------------------------
        inv_ratio = self._net_inventory / max(self._base_qty, 1e-9)
        inv_add_bps = self._inv_skew_k * abs(inv_ratio)
        spread_bps = base_spread_bps + inv_add_bps

        # --- 4. LOB imbalance skew ----------------------------------------
        total_vol = lob_bid_vol + lob_ask_vol
        if total_vol > 0:
            imbalance = (lob_bid_vol - lob_ask_vol) / total_vol   # [-1, +1]
        else:
            imbalance = 0.0
        # Positive imbalance (more bids) → price likely to rise → skew ask up
        skew_bps = self._lob_skew_k * float(imbalance)

        # --- 5. Signal confidence skew ------------------------------------
        # Positive signal → tighten ask (more willing to sell), widen bid.
        # Negative signal → tighten bid (more willing to buy), widen ask.
        sig_skew_bps = self._sig_skew_k * float(np.clip(signal_confidence, -1.0, 1.0))

        # Combined skew on mid (positive = shift ask up, bid up)
        total_skew_bps = skew_bps + sig_skew_bps

        # Asymmetric spread application
        half = spread_bps / 2.0
        bid_offset_bps = half + max(0.0, -total_skew_bps)   # widen on sell pressure
        ask_offset_bps = half + max(0.0,  total_skew_bps)   # widen on buy pressure

        skewed_mid = mid_price * (1.0 + total_skew_bps / 10_000.0)
        bid_price = skewed_mid * (1.0 - bid_offset_bps / 10_000.0)
        ask_price = skewed_mid * (1.0 + ask_offset_bps / 10_000.0)

        # --- 6. Kelly-weighted quantity -----------------------------------
        kelly_qty = self._base_qty * max(0.0, kelly_fraction) / max(0.01, 0.10)
        blended_qty = (
            self._kelly_blend * kelly_qty
            + (1.0 - self._kelly_blend) * self._base_qty
        )
        blended_qty = max(1e-8, blended_qty)

        # Inventory cap: suspend quoting on heavy side
        bid_qty = blended_qty
        ask_qty = blended_qty

        if self._net_inventory >= self._max_inv:
            bid_qty = 0.0   # already long max — don't add more
        elif self._net_inventory <= -self._max_inv:
            ask_qty = 0.0   # already short max — don't add more

        # Skew quantity toward the mean-reverting side
        if self._net_inventory > 0:
            # Long inventory → prefer to sell; reduce bid qty
            inv_scale = max(0.1, 1.0 - abs(inv_ratio) * 0.3)
            bid_qty *= inv_scale
            ask_qty = min(blended_qty * 1.5, ask_qty * (2.0 - inv_scale))
        elif self._net_inventory < 0:
            inv_scale = max(0.1, 1.0 - abs(inv_ratio) * 0.3)
            ask_qty *= inv_scale
            bid_qty = min(blended_qty * 1.5, bid_qty * (2.0 - inv_scale))

        reason = (
            f"ATR={atr_pct*100:.3f}% base_spread={base_spread_bps:.1f}bps "
            f"regime_mult={regime_mult:.1f} inv={self._net_inventory:+.4f} "
            f"lob_imb={imbalance:+.2f} sig={signal_confidence:+.2f} "
            f"final_spread={spread_bps:.1f}bps skew={total_skew_bps:+.1f}bps"
        )

        quote = MMQuote(
            symbol=self.symbol,
            mid_price=mid_price,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            spread_bps=spread_bps,
            skew_bps=total_skew_bps,
            regime=regime,
            active=True,
            reason=reason,
        )
        self._last_quote = quote
        return quote

    # ------------------------------------------------------------------
    # Fill recording & inventory tracking
    # ------------------------------------------------------------------

    def record_fill(
        self,
        side: str,
        qty: float,
        price: float,
        pnl: float,
    ) -> None:
        """Record a fill and update net inventory.

        Parameters
        ----------
        side : str
            ``'buy'`` or ``'sell'``.
        qty : float
            Filled quantity in base asset.
        price : float
            Fill price.
        pnl : float
            Realised PnL of this fill (AUD or quote currency).
        """
        qty = abs(float(qty))
        pnl = float(pnl)
        side = side.lower().strip()

        if side == "buy":
            self._net_inventory += qty
        elif side == "sell":
            self._net_inventory -= qty

        self._cumulative_pnl += pnl
        self._fills.append(FillRecord(side=side, qty=qty, price=price, pnl=pnl))
        logger.debug(
            "MM fill: %s %.6f @ %.4f pnl=%.4f net_inv=%+.6f",
            side, qty, price, pnl, self._net_inventory,
        )

    def reset_inventory(self, value: float = 0.0) -> None:
        """Force-reset net inventory (e.g. after a hedge trade)."""
        self._net_inventory = float(value)

    # ------------------------------------------------------------------
    # AutonomousBrain integration
    # ------------------------------------------------------------------

    def mm_spread_state(self) -> Dict[str, Any]:
        """Return a dict suitable for injection into market_state['mm_spread'].

        The AutonomousBrain can read this to make decisions such as:
        - pausing MM when inventory is near the cap
        - adjusting risk when spread is abnormally wide
        - flagging for venue switch when fill rate drops
        """
        recent_fills = list(self._fills)
        recent_pnl = [f.pnl for f in recent_fills[-50:]]
        fill_win_rate = (
            sum(1 for p in recent_pnl if p > 0) / max(len(recent_pnl), 1)
        )
        avg_pnl = float(np.mean(recent_pnl)) if recent_pnl else 0.0

        last = self._last_quote
        return {
            "symbol": self.symbol,
            "net_inventory": round(self._net_inventory, 8),
            "inventory_utilisation": round(
                abs(self._net_inventory) / max(self._max_inv, 1e-9), 4
            ),
            "cumulative_pnl": round(self._cumulative_pnl, 6),
            "fill_win_rate_50": round(fill_win_rate, 4),
            "avg_fill_pnl_50": round(avg_pnl, 6),
            "total_fills": len(self._fills),
            "quote_count": self._quote_count,
            "last_spread_bps": round(last.spread_bps, 3) if last else None,
            "last_skew_bps": round(last.skew_bps, 3) if last else None,
            "last_active": last.active if last else False,
            "last_regime": last.regime if last else "UNKNOWN",
            "last_atr_pct": round(self._last_atr_pct, 6),
        }

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Full diagnostic snapshot."""
        recent = [f.pnl for f in self._fills]
        return {
            "symbol": self.symbol,
            "net_inventory": self._net_inventory,
            "max_inventory": self._max_inv,
            "cumulative_pnl": self._cumulative_pnl,
            "total_fills": len(self._fills),
            "quote_count": self._quote_count,
            "recent_pnl_mean": float(np.mean(recent)) if recent else 0.0,
            "recent_pnl_std": float(np.std(recent)) if recent else 0.0,
            "config": {
                "base_spread_k": self._k,
                "base_qty": self._base_qty,
                "inventory_skew_k": self._inv_skew_k,
                "lob_skew_k": self._lob_skew_k,
                "signal_skew_k": self._sig_skew_k,
                "kelly_qty_blend": self._kelly_blend,
                "max_inventory": self._max_inv,
            },
            "last_quote": self._last_quote.to_dict() if self._last_quote else None,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cfg(config: Any) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return config.get("market_maker_spread", {}) or {}
        return getattr(config, "market_maker_spread", None) or {}


__all__ = ["MarketMakerSpreadManager", "MMQuote", "FillRecord"]
