"""
Deribit Options Signal — generates signals from options market structure.

Options market provides forward-looking signals:
  - Put/Call ratio > 1.5: bearish sentiment (many protective puts)
  - IV smile skew: if puts >> calls in IV, market fears downside
  - Max pain: price where most options expire worthless (gravitational pull)
  - Gamma exposure (GEX): when negative, dealers must sell into drops

Data source: Deribit public API (no auth required for market data).
Falls back to neutral signal if API unavailable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DERIBIT_API_BASE = "https://www.deribit.com/api/v2/public/"


@dataclass
class OptionsSnapshot:
    symbol: str
    expiry: str
    put_call_ratio: float
    iv_skew_pct: float
    max_pain_price: float
    gex_usd: float
    implied_move_pct: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptionsSignal:
    symbol: str
    direction: str  # "BEARISH" / "BULLISH" / "NEUTRAL"
    confidence: float
    rationale: str
    iv_percentile: float
    timestamp: float = field(default_factory=time.time)


class DeribitOptionsSignal:
    """Generates trading signals from Deribit options market structure."""

    def __init__(self, symbol: str = "BTC", cache_ttl: int = 300) -> None:
        self.symbol = symbol.upper()
        self.cache_ttl = cache_ttl
        self._cache: Optional[tuple[float, OptionsSnapshot]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_snapshot(self) -> Optional[OptionsSnapshot]:
        """Fetch options market snapshot (cached)."""
        now = time.time()
        if self._cache is not None and (now - self._cache[0]) < self.cache_ttl:
            return self._cache[1]
        try:
            loop = asyncio.get_running_loop()
            snapshot = await loop.run_in_executor(None, self._fetch_snapshot)
            if snapshot is not None:
                self._cache = (now, snapshot)
            return snapshot
        except Exception:
            logger.debug("DeribitOptionsSignal: fetch failed", exc_info=True)
            return None

    async def generate_signal(self) -> OptionsSignal:
        """Generate a directional signal from current options structure."""
        snapshot = await self.get_snapshot()
        if snapshot is None:
            return OptionsSignal(
                symbol=self.symbol,
                direction="NEUTRAL",
                confidence=0.0,
                rationale="Options data unavailable",
                iv_percentile=50.0,
            )
        return self._interpret(snapshot)

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def _fetch_snapshot(self) -> Optional[OptionsSnapshot]:
        """Synchronous HTTP fetch (runs in executor)."""
        try:
            instruments = self._get_instruments()
            if not instruments:
                return None

            pc_ratio = self._parse_put_call_ratio(instruments)
            iv_skew = self._parse_iv_skew(instruments)
            max_pain = self._compute_max_pain(instruments)
            gex = self._compute_gex(instruments)
            implied_move = self._compute_implied_move(instruments)

            # Use the nearest expiry as the label
            expiry = self._nearest_expiry(instruments)

            return OptionsSnapshot(
                symbol=self.symbol,
                expiry=expiry,
                put_call_ratio=pc_ratio,
                iv_skew_pct=iv_skew,
                max_pain_price=max_pain,
                gex_usd=gex,
                implied_move_pct=implied_move,
            )
        except Exception:
            logger.debug("DeribitOptionsSignal: _fetch_snapshot error", exc_info=True)
            return None

    def _get_instruments(self) -> list:
        """Fetch all option instruments for symbol."""
        url = (
            f"{DERIBIT_API_BASE}get_instruments"
            f"?currency={self.symbol}&kind=option&expired=false"
        )
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read())
            return data.get("result", [])
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_put_call_ratio(self, instruments: list) -> float:
        """Put/call ratio by open interest."""
        puts = sum(i.get("open_interest", 0) for i in instruments if i.get("option_type") == "put")
        calls = sum(i.get("open_interest", 0) for i in instruments if i.get("option_type") == "call")
        if calls <= 0:
            return 1.0
        return puts / calls

    def _parse_iv_skew(self, instruments: list) -> float:
        """IV skew: avg put IV minus avg call IV (pct points)."""
        put_ivs = [i.get("mark_iv", 0) for i in instruments if i.get("option_type") == "put" and i.get("mark_iv", 0) > 0]
        call_ivs = [i.get("mark_iv", 0) for i in instruments if i.get("option_type") == "call" and i.get("mark_iv", 0) > 0]
        if not put_ivs or not call_ivs:
            return 0.0
        return (sum(put_ivs) / len(put_ivs)) - (sum(call_ivs) / len(call_ivs))

    def _compute_max_pain(self, instruments: list) -> float:
        """Strike where total OI value is minimised (max pain level)."""
        strikes: dict[float, float] = {}
        for inst in instruments:
            strike = inst.get("strike", 0)
            oi = inst.get("open_interest", 0)
            if strike > 0:
                strikes[strike] = strikes.get(strike, 0) + oi
        if not strikes:
            return 0.0
        return min(strikes, key=lambda s: -strikes[s])

    def _compute_gex(self, instruments: list) -> float:
        """
        Simplified Gamma Exposure: positive GEX = dealers long gamma (stabilising),
        negative GEX = dealers short gamma (destabilising).
        """
        gex = 0.0
        for inst in instruments:
            gamma = inst.get("greeks", {}).get("gamma", 0) if isinstance(inst.get("greeks"), dict) else 0
            oi = inst.get("open_interest", 0)
            option_type = inst.get("option_type", "")
            sign = 1 if option_type == "call" else -1
            gex += sign * gamma * oi * 100  # 100 USD per contract approx
        return gex

    def _compute_implied_move(self, instruments: list) -> float:
        """Estimate implied move from ATM straddle IV."""
        atm_ivs = [
            i.get("mark_iv", 0)
            for i in instruments
            if i.get("mark_iv", 0) > 0 and i.get("in_the_money") is False
        ]
        if not atm_ivs:
            return 5.0  # default 5%
        avg_iv = sum(atm_ivs[:10]) / min(len(atm_ivs), 10)
        # Implied move ≈ IV * sqrt(days/365); use 30-day proxy
        return avg_iv * (30 / 365) ** 0.5

    def _nearest_expiry(self, instruments: list) -> str:
        expiries = [i.get("expiration_timestamp", 0) for i in instruments if i.get("expiration_timestamp")]
        if not expiries:
            return "unknown"
        nearest_ms = min(expiries)
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(nearest_ms / 1000, tz=timezone.utc)
        return dt.strftime("%d%b%y").upper()

    # ------------------------------------------------------------------
    # Signal interpretation
    # ------------------------------------------------------------------

    def _interpret(self, snap: OptionsSnapshot) -> OptionsSignal:
        """Convert options snapshot to directional signal."""
        bearish_score = 0.0
        bullish_score = 0.0
        reasons = []

        # Put/call ratio
        if snap.put_call_ratio > 1.5:
            bearish_score += 0.30
            reasons.append(f"PC ratio {snap.put_call_ratio:.2f} (bearish)")
        elif snap.put_call_ratio < 0.7:
            bullish_score += 0.25
            reasons.append(f"PC ratio {snap.put_call_ratio:.2f} (bullish)")

        # IV skew (puts > calls in IV = fear)
        if snap.iv_skew_pct > 5.0:
            bearish_score += 0.25
            reasons.append(f"IV skew {snap.iv_skew_pct:.1f}% (puts expensive)")
        elif snap.iv_skew_pct < -3.0:
            bullish_score += 0.20
            reasons.append(f"IV skew {snap.iv_skew_pct:.1f}% (calls expensive)")

        # Negative gamma exposure → destabilising → bearish
        if snap.gex_usd < 0:
            gex_abs = abs(snap.gex_usd)
            contribution = min(0.20, gex_abs / 1_000_000 * 0.05)
            bearish_score += contribution
            reasons.append(f"Negative GEX ${snap.gex_usd:,.0f}")
        elif snap.gex_usd > 0:
            bullish_score += min(0.10, snap.gex_usd / 2_000_000 * 0.05)

        # High implied move → elevated risk
        if snap.implied_move_pct > 10.0:
            bearish_score += 0.10
            reasons.append(f"High implied move {snap.implied_move_pct:.1f}%")

        # Determine direction
        net = bearish_score - bullish_score
        if net > 0.30:
            direction = "BEARISH"
            confidence = min(0.90, net)
        elif net < -0.25:
            direction = "BULLISH"
            confidence = min(0.85, abs(net))
        else:
            direction = "NEUTRAL"
            confidence = max(0.0, 0.50 - abs(net) * 2)

        # Approximate IV percentile from implied move
        iv_percentile = min(99.0, snap.implied_move_pct * 6.0)

        return OptionsSignal(
            symbol=self.symbol,
            direction=direction,
            confidence=confidence,
            rationale="; ".join(reasons) if reasons else "No strong signal",
            iv_percentile=iv_percentile,
        )

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    def generate_orders(
        self, signal: OptionsSignal, portfolio_value: float
    ) -> List[Dict]:
        """Convert an options sentiment signal to protective put orders.

        For BEARISH signals: buy protective puts (hedge existing long exposure).
        For BULLISH signals: sell puts (collect premium on expected stability).
        For NEUTRAL signals: no orders.

        Parameters
        ----------
        signal : OptionsSignal
            The sentiment signal from ``generate_signal()``.
        portfolio_value : float
            Current total portfolio value in USD.

        Returns
        -------
        List of order dicts with keys: symbol, side, quantity, order_type, reason.
        """
        if signal.direction == "NEUTRAL" or signal.confidence < 0.1:
            return []

        # Size the hedge: fraction of portfolio proportional to confidence
        hedge_fraction = min(0.20, signal.confidence * 0.25)
        notional = portfolio_value * hedge_fraction

        orders: List[Dict] = []

        if signal.direction == "BEARISH":
            # Buy protective puts to hedge downside
            orders.append({
                "symbol": f"{signal.symbol}-PUT",
                "side": "BUY",
                "quantity": notional,
                "order_type": "limit",
                "reason": f"protective_put_bearish_conf_{signal.confidence:.2f}",
            })
        elif signal.direction == "BULLISH":
            # Sell puts to collect premium (bullish outlook)
            orders.append({
                "symbol": f"{signal.symbol}-PUT",
                "side": "SELL",
                "quantity": notional * 0.5,  # smaller size for short puts
                "order_type": "limit",
                "reason": f"sell_put_bullish_conf_{signal.confidence:.2f}",
            })

        return orders
