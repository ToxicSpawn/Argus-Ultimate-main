"""
Volatility Arbitrage — trades when implied volatility diverges from realized volatility.

When IV (from Deribit options) significantly exceeds realised vol: sell options (vol expensive).
When IV significantly below realised vol: buy options (vol cheap).
Delta-hedge the options position with perpetuals.

Signal strength = (IV - RV) / RV * 100  → vol premium in percent
Entry: |vol_premium| > MIN_VOL_PREMIUM_PCT (default 15%)
Exit: |vol_premium| < EXIT_VOL_PREMIUM_PCT (default 5%) OR max hold exceeded
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_VOL_PREMIUM_PCT: float = 15.0    # minimum |IV - RV| / RV * 100 to enter
EXIT_VOL_PREMIUM_PCT: float = 5.0    # exit when |vol_premium| falls below this
REALISED_VOL_WINDOW: int = 24        # number of hourly price observations for RV
MIN_IV_OBSERVATIONS: int = 5         # minimum IV observations before evaluating

# Annualisation for hourly log-return std → annualised vol
# sqrt(252 trading days * 24 hours) for hourly data
_HOURLY_ANNUALISATION: float = math.sqrt(252 * 24)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class VolSnapshot:
    """Point-in-time volatility surface summary for a symbol."""

    symbol: str
    implied_vol_pct: float    # current IV (annualised %)
    realised_vol_pct: float   # RV over REALISED_VOL_WINDOW (annualised %)
    vol_premium_pct: float    # (IV - RV) / RV * 100  (positive → IV expensive)
    skew_pct: float           # put-call skew (informational)
    timestamp: float = field(default_factory=time.time)


@dataclass
class VolArbSignal:
    """Trading signal emitted by VolatilityArb."""

    symbol: str
    action: str               # SELL_VOL | BUY_VOL | EXIT | HOLD
    vol_premium_pct: float
    iv_pct: float
    rv_pct: float
    hedge_delta: float        # suggested delta for perp hedge (0.0–1.0)
    position_size_usd: float
    reason: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class VolatilityArb:
    """
    Volatility arbitrage strategy.

    Compares implied volatility (provided externally from options data) with
    realised volatility (computed internally from a rolling price history).

    When the vol premium exceeds the threshold, emit SELL_VOL or BUY_VOL
    signals.  No live option orders are placed here — signals are returned
    for the execution layer.

    Thread-safe: all mutable state protected by a lock.
    """

    def __init__(
        self,
        min_premium_pct: float = MIN_VOL_PREMIUM_PCT,
        capital_per_trade: float = 300.0,
    ) -> None:
        self._min_premium = min_premium_pct
        self._exit_premium = EXIT_VOL_PREMIUM_PCT
        self._capital = capital_per_trade
        self._lock = threading.Lock()

        # symbol -> deque of (timestamp, price) for RV computation
        self._price_history: Dict[str, Deque[Tuple[float, float]]] = {}
        # symbol -> (iv_pct, skew_pct, timestamp) latest IV observations
        self._iv_data: Dict[str, List[Tuple[float, float, float]]] = {}
        # symbol -> active position flag (SELL_VOL | BUY_VOL | None)
        self._positions: Dict[str, Optional[str]] = {}

        logger.info(
            "VolatilityArb initialised: min_premium=%.1f%% exit_premium=%.1f%% "
            "capital=%.0f USD",
            min_premium_pct, EXIT_VOL_PREMIUM_PCT, capital_per_trade,
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update_iv(
        self,
        symbol: str,
        implied_vol_pct: float,
        skew_pct: float = 0.0,
    ) -> None:
        """
        Record a new implied volatility observation.

        Parameters
        ----------
        symbol:
            E.g. ``"BTC/USD"``.
        implied_vol_pct:
            Annualised implied volatility in percent (e.g. ``85.0`` for 85%).
        skew_pct:
            Put-call skew in percent (informational; positive = puts expensive).
        """
        if implied_vol_pct < 0:
            logger.warning(
                "update_iv: negative IV %.2f for %s — ignored", implied_vol_pct, symbol
            )
            return

        ts = time.time()
        with self._lock:
            if symbol not in self._iv_data:
                self._iv_data[symbol] = []
            self._iv_data[symbol].append((implied_vol_pct, skew_pct, ts))
            # Keep only last MIN_IV_OBSERVATIONS * 2 to bound memory
            max_keep = max(MIN_IV_OBSERVATIONS * 2, 20)
            if len(self._iv_data[symbol]) > max_keep:
                self._iv_data[symbol] = self._iv_data[symbol][-max_keep:]

        logger.debug("update_iv: %s iv=%.2f%% skew=%.2f%%", symbol, implied_vol_pct, skew_pct)

    def update_prices(self, symbol: str, price: float, timestamp: float) -> None:
        """
        Ingest a price observation used to compute realised volatility.

        Intended to be called on each new price bar (e.g. hourly close).

        Parameters
        ----------
        symbol:
            E.g. ``"BTC/USD"``.
        price:
            Mid or close price in USD.
        timestamp:
            Unix timestamp of the observation.
        """
        if price <= 0:
            logger.warning(
                "update_prices: non-positive price %.6f for %s — ignored", price, symbol
            )
            return

        with self._lock:
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=REALISED_VOL_WINDOW + 1)
            self._price_history[symbol].append((timestamp, price))

        logger.debug("update_prices: %s price=%.4f", symbol, price)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def evaluate(self, symbol: str) -> Optional[VolArbSignal]:
        """
        Evaluate the vol arb opportunity for *symbol*.

        Returns ``None`` if there is insufficient data.
        """
        with self._lock:
            iv_list = list(self._iv_data.get(symbol, []))
            price_hist = list(self._price_history.get(symbol, []))
            current_position = self._positions.get(symbol)

        if len(iv_list) < MIN_IV_OBSERVATIONS:
            logger.debug(
                "evaluate: %s insufficient IV observations (%d < %d)",
                symbol, len(iv_list), MIN_IV_OBSERVATIONS,
            )
            return None

        if len(price_hist) < 2:
            logger.debug(
                "evaluate: %s insufficient price history (%d)", symbol, len(price_hist)
            )
            return None

        # Use the latest IV observation
        iv_pct, skew_pct, _ts = iv_list[-1]

        # Compute RV from price history
        prices = [p for (_t, p) in price_hist]
        rv_pct = self.get_realised_vol(symbol)

        if rv_pct == 0.0:
            logger.debug("evaluate: %s RV is zero, skipping", symbol)
            return None

        vol_premium_pct = (iv_pct - rv_pct) / rv_pct * 100.0
        snapshot = VolSnapshot(
            symbol=symbol,
            implied_vol_pct=iv_pct,
            realised_vol_pct=rv_pct,
            vol_premium_pct=vol_premium_pct,
            skew_pct=skew_pct,
        )

        # Delta for perp hedge: simplified — use 0.5 as ATM delta proxy
        hedge_delta = 0.5

        # If already in a position, check for exit
        if current_position is not None:
            if abs(vol_premium_pct) < self._exit_premium:
                logger.info(
                    "EXIT vol arb: %s vol_premium=%.2f%% below exit threshold %.2f%%",
                    symbol, vol_premium_pct, self._exit_premium,
                )
                with self._lock:
                    self._positions[symbol] = None
                return VolArbSignal(
                    symbol=symbol,
                    action="EXIT",
                    vol_premium_pct=vol_premium_pct,
                    iv_pct=iv_pct,
                    rv_pct=rv_pct,
                    hedge_delta=hedge_delta,
                    position_size_usd=self._capital,
                    reason=f"vol_premium_collapsed:{vol_premium_pct:.2f}%<{self._exit_premium}%",
                )
            return VolArbSignal(
                symbol=symbol,
                action="HOLD",
                vol_premium_pct=vol_premium_pct,
                iv_pct=iv_pct,
                rv_pct=rv_pct,
                hedge_delta=hedge_delta,
                position_size_usd=self._capital,
                reason=f"holding:{current_position} vol_premium={vol_premium_pct:.2f}%",
            )

        # No position — evaluate entry
        if abs(vol_premium_pct) < self._min_premium:
            return VolArbSignal(
                symbol=symbol,
                action="HOLD",
                vol_premium_pct=vol_premium_pct,
                iv_pct=iv_pct,
                rv_pct=rv_pct,
                hedge_delta=hedge_delta,
                position_size_usd=0.0,
                reason=f"premium_insufficient:{abs(vol_premium_pct):.2f}%<{self._min_premium}%",
            )

        if vol_premium_pct > 0:
            action = "SELL_VOL"  # IV expensive relative to RV
        else:
            action = "BUY_VOL"  # IV cheap relative to RV

        logger.info(
            "%s signal: symbol=%s vol_premium=%.2f%% iv=%.2f%% rv=%.2f%%",
            action, symbol, vol_premium_pct, iv_pct, rv_pct,
        )
        with self._lock:
            self._positions[symbol] = action

        return VolArbSignal(
            symbol=symbol,
            action=action,
            vol_premium_pct=vol_premium_pct,
            iv_pct=iv_pct,
            rv_pct=rv_pct,
            hedge_delta=hedge_delta,
            position_size_usd=self._capital,
            reason=f"vol_premium={vol_premium_pct:.2f}%>threshold={self._min_premium}%",
        )

    def get_realised_vol(self, symbol: str) -> float:
        """
        Compute annualised realised volatility for *symbol* from stored prices.

        Returns 0.0 if insufficient data.
        """
        with self._lock:
            hist = list(self._price_history.get(symbol, []))

        if len(hist) < 2:
            return 0.0

        prices = [p for (_t, p) in hist]
        return self._compute_rv(prices, REALISED_VOL_WINDOW)

    def get_snapshot(self, symbol: str) -> Optional[VolSnapshot]:
        """Return the latest VolSnapshot for *symbol*, or None if insufficient data."""
        with self._lock:
            iv_list = list(self._iv_data.get(symbol, []))

        if len(iv_list) < MIN_IV_OBSERVATIONS:
            return None

        iv_pct, skew_pct, ts = iv_list[-1]
        rv_pct = self.get_realised_vol(symbol)
        if rv_pct == 0.0:
            return None

        vol_premium_pct = (iv_pct - rv_pct) / rv_pct * 100.0
        return VolSnapshot(
            symbol=symbol,
            implied_vol_pct=iv_pct,
            realised_vol_pct=rv_pct,
            vol_premium_pct=vol_premium_pct,
            skew_pct=skew_pct,
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_rv(self, prices: List[float], window: int) -> float:
        """
        Compute annualised realised volatility from a price series.

        Uses log returns over the most recent *window* observations.
        Annualisation assumes hourly bars: factor = sqrt(252 * 24).

        Returns 0.0 if there are fewer than 2 prices.
        """
        if len(prices) < 2:
            return 0.0

        # Use up to the last (window + 1) prices to get *window* log returns
        subset = prices[-(window + 1):]
        log_returns: List[float] = []
        for i in range(1, len(subset)):
            prev, curr = subset[i - 1], subset[i]
            if prev > 0 and curr > 0:
                log_returns.append(math.log(curr / prev))

        n = len(log_returns)
        if n < 1:
            return 0.0

        mean_r = sum(log_returns) / n
        variance = sum((r - mean_r) ** 2 for r in log_returns) / max(n - 1, 1)
        std = math.sqrt(variance)

        # Annualise: hourly data → multiply by sqrt(252 * 24)
        annualised = std * _HOURLY_ANNUALISATION * 100.0  # convert to percent
        return round(annualised, 4)
