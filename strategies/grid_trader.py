"""
Grid Trading Strategy — places layered buy/sell orders across a price range.

Grid trading profits from range-bound markets by buying at lower grid levels
and selling at upper grid levels.  When price oscillates within the grid,
each filled buy is paired with a sell one level above, capturing the spread.

Auto-detection mode: if no price bounds are given, the grid is centred on
the last observed price and the range is derived from recent volatility
(ATR-style standard deviation over a lookback window).

Example (BTC grid $60,000 – $62,000, 10 levels):
  Level 0  BUY  $60,000  0.00083 BTC
  Level 1  BUY  $60,222  0.00083 BTC
  ...
  Level 9  SELL $62,000  0.00083 BTC

When price crosses from level 3 to level 5, levels 3–4 fill as buys;
when it falls back, they flip to sells, realising the grid spread.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default lookback for auto-range volatility estimation
_DEFAULT_VOL_LOOKBACK = 100
# Default multiplier for auto-range (2× σ each side)
_DEFAULT_VOL_MULTIPLIER = 2.0


@dataclass
class GridLevel:
    """A single price level in the grid."""

    price: float
    side: str  # "buy" or "sell"
    size: float  # quantity in base asset
    filled: bool = False
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None

    def __repr__(self) -> str:
        status = "FILLED" if self.filled else "OPEN"
        return f"GridLevel({self.side.upper()} {self.price:.2f} qty={self.size:.6f} {status})"


@dataclass
class GridSignal:
    """A signal emitted when a grid level is crossed."""

    symbol: str
    side: str  # "buy" or "sell"
    price: float
    size: float
    level_index: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GridTrader:
    """
    Grid Trading Strategy.

    Places a ladder of buy orders below the current price and sell orders
    above it.  As price oscillates, filled buys are flipped to sells and
    vice-versa, capturing the grid spread on each round-trip.

    Parameters
    ----------
    vol_lookback : int
        Number of price observations used for auto-range estimation.
    vol_multiplier : float
        Multiplier applied to standard deviation for auto-range width.
    """

    def __init__(
        self,
        vol_lookback: int = _DEFAULT_VOL_LOOKBACK,
        vol_multiplier: float = _DEFAULT_VOL_MULTIPLIER,
    ) -> None:
        self.vol_lookback = max(vol_lookback, 10)
        self.vol_multiplier = max(vol_multiplier, 0.5)

        # Per-symbol state
        self._grids: Dict[str, List[GridLevel]] = {}
        self._price_history: Dict[str, Deque[float]] = {}
        self._last_price: Dict[str, float] = {}
        self._realized_pnl: Dict[str, float] = {}
        self._grid_params: Dict[str, Dict] = {}  # lower, upper, num_levels

        logger.info(
            "GridTrader initialised (vol_lookback=%d, vol_multiplier=%.2f)",
            self.vol_lookback,
            self.vol_multiplier,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_grid(
        self,
        symbol: str,
        lower_price: Optional[float] = None,
        upper_price: Optional[float] = None,
        num_levels: int = 10,
        capital_usd: float = 100.0,
    ) -> List[GridLevel]:
        """
        Create (or recreate) a grid for *symbol*.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. ``"BTC/USD"``).
        lower_price, upper_price : float or None
            Explicit price bounds.  If either is ``None`` the bounds are
            estimated from recent price volatility.
        num_levels : int
            Number of grid levels (min 3).
        capital_usd : float
            Total capital allocated to this grid.

        Returns
        -------
        list[GridLevel]
            The constructed grid levels.
        """
        num_levels = max(num_levels, 3)
        if capital_usd <= 0:
            raise ValueError("capital_usd must be positive")

        # Auto-detect range from volatility if bounds not provided
        if lower_price is None or upper_price is None:
            lower_price, upper_price = self._auto_range(symbol)
            if lower_price is None or upper_price is None:
                raise ValueError(
                    f"Cannot auto-detect range for {symbol}: "
                    f"need at least {self.vol_lookback} price observations "
                    f"(have {len(self._price_history.get(symbol, []))}). "
                    f"Provide explicit lower_price/upper_price."
                )

        if lower_price >= upper_price:
            raise ValueError(
                f"lower_price ({lower_price}) must be less than upper_price ({upper_price})"
            )

        step = (upper_price - lower_price) / (num_levels - 1)
        mid_price = (lower_price + upper_price) / 2.0
        capital_per_level = capital_usd / num_levels

        levels: List[GridLevel] = []
        for i in range(num_levels):
            price = lower_price + i * step
            side = "buy" if price < mid_price else "sell"
            # For the midpoint level, default to buy
            if math.isclose(price, mid_price, rel_tol=1e-9):
                side = "buy"
            size = capital_per_level / price if price > 0 else 0.0
            levels.append(GridLevel(price=round(price, 8), side=side, size=round(size, 8)))

        self._grids[symbol] = levels
        self._realized_pnl.setdefault(symbol, 0.0)
        self._grid_params[symbol] = {
            "lower": lower_price,
            "upper": upper_price,
            "num_levels": num_levels,
            "capital_usd": capital_usd,
        }

        logger.info(
            "Grid set up for %s: %d levels [%.2f – %.2f], capital=$%.2f",
            symbol,
            num_levels,
            lower_price,
            upper_price,
            capital_usd,
        )
        return list(levels)

    def update_price(self, symbol: str, price: float) -> None:
        """
        Record a new price observation for volatility estimation.

        Call this on every tick / candle close so that auto-range detection
        has enough data.
        """
        if price <= 0:
            return
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.vol_lookback * 2)
        self._price_history[symbol].append(price)
        self._last_price[symbol] = price

    def check_fills(self, symbol: str, current_price: float) -> List[GridSignal]:
        """
        Check which grid levels have been crossed by *current_price*.

        Returns a list of :class:`GridSignal` instances that should be
        executed.  Filled levels are marked and flipped (buy→sell or
        sell→buy) for the next crossing.
        """
        self.update_price(symbol, current_price)
        grid = self._grids.get(symbol)
        if not grid:
            return []

        signals: List[GridSignal] = []
        now = datetime.now(timezone.utc)

        for idx, level in enumerate(grid):
            if level.filled:
                continue

            triggered = False
            if level.side == "buy" and current_price <= level.price:
                triggered = True
            elif level.side == "sell" and current_price >= level.price:
                triggered = True

            if triggered:
                level.filled = True
                level.fill_price = current_price
                level.fill_time = now

                signals.append(
                    GridSignal(
                        symbol=symbol,
                        side=level.side,
                        price=level.price,
                        size=level.size,
                        level_index=idx,
                        timestamp=now,
                    )
                )

                # Track realised P&L for sell fills (buy was the entry)
                if level.side == "sell":
                    # Find the nearest filled buy below this level
                    buy_price = self._find_paired_buy(grid, idx)
                    if buy_price is not None:
                        pnl = (current_price - buy_price) * level.size
                        self._realized_pnl[symbol] = self._realized_pnl.get(symbol, 0.0) + pnl
                        logger.debug(
                            "Grid %s level %d SELL filled: pnl=%.4f (sell=%.2f buy=%.2f)",
                            symbol, idx, pnl, current_price, buy_price,
                        )

                # Flip the level for the next crossing
                level.side = "sell" if level.side == "buy" else "buy"
                level.filled = False
                level.fill_price = None
                level.fill_time = None

                logger.info(
                    "Grid %s level %d triggered %s @ %.2f (qty=%.6f), flipped to %s",
                    symbol,
                    idx,
                    signals[-1].side,
                    current_price,
                    level.size,
                    level.side,
                )

        return signals

    def get_pnl(self, symbol: Optional[str] = None) -> float:
        """
        Return realised + estimated unrealised P&L.

        Parameters
        ----------
        symbol : str or None
            If ``None``, returns total across all symbols.
        """
        if symbol is not None:
            realized = self._realized_pnl.get(symbol, 0.0)
            unrealized = self._estimate_unrealized(symbol)
            return realized + unrealized

        total = 0.0
        for sym in self._grids:
            total += self._realized_pnl.get(sym, 0.0) + self._estimate_unrealized(sym)
        return total

    def reset_grid(self, symbol: str, new_center_price: Optional[float] = None) -> List[GridLevel]:
        """
        Recenter the grid around *new_center_price* (or the last observed price).

        Preserves the same number of levels and capital allocation.
        """
        params = self._grid_params.get(symbol)
        if params is None:
            raise ValueError(f"No grid exists for {symbol}; call setup_grid() first")

        if new_center_price is None:
            new_center_price = self._last_price.get(symbol)
        if new_center_price is None:
            raise ValueError(f"No price available for {symbol}; provide new_center_price")

        old_range = params["upper"] - params["lower"]
        new_lower = new_center_price - old_range / 2.0
        new_upper = new_center_price + old_range / 2.0

        logger.info(
            "Recentering grid for %s around %.2f [%.2f – %.2f]",
            symbol, new_center_price, new_lower, new_upper,
        )
        return self.setup_grid(
            symbol,
            lower_price=new_lower,
            upper_price=new_upper,
            num_levels=params["num_levels"],
            capital_usd=params["capital_usd"],
        )

    def get_grid(self, symbol: str) -> List[GridLevel]:
        """Return a copy of the current grid levels for *symbol*."""
        return list(self._grids.get(symbol, []))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _auto_range(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Estimate price bounds from recent volatility."""
        history = self._price_history.get(symbol)
        if not history or len(history) < self.vol_lookback:
            return None, None

        prices = list(history)[-self.vol_lookback:]
        mean_price = statistics.mean(prices)
        if mean_price <= 0:
            return None, None

        stdev = statistics.stdev(prices) if len(prices) > 1 else mean_price * 0.02
        half_range = stdev * self.vol_multiplier
        # Ensure minimum range of 1% of price
        half_range = max(half_range, mean_price * 0.005)

        lower = mean_price - half_range
        upper = mean_price + half_range
        # Clamp lower to positive
        lower = max(lower, mean_price * 0.5)

        logger.debug(
            "Auto-range for %s: mean=%.2f stdev=%.2f range=[%.2f, %.2f]",
            symbol, mean_price, stdev, lower, upper,
        )
        return lower, upper

    def _find_paired_buy(self, grid: List[GridLevel], sell_idx: int) -> Optional[float]:
        """Find the nearest buy fill price below *sell_idx*."""
        # Look downward for the closest level that was a buy
        for i in range(sell_idx - 1, -1, -1):
            level = grid[i]
            # After a fill+flip, a formerly-bought level is now "sell"
            # We approximate using the grid level price
            if level.price < grid[sell_idx].price:
                return level.price
        return None

    def _estimate_unrealized(self, symbol: str) -> float:
        """Estimate unrealised P&L from open positions at last price."""
        grid = self._grids.get(symbol)
        last = self._last_price.get(symbol)
        if not grid or last is None:
            return 0.0

        unrealized = 0.0
        for level in grid:
            # Levels currently set as "sell" were bought; mark-to-market
            if level.side == "sell" and not level.filled:
                unrealized += (last - level.price) * level.size
        return unrealized
