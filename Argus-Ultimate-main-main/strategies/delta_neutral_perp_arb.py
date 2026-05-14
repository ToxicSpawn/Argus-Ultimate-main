"""
Delta-Neutral Perpetual Funding Arbitrage.

Hold spot asset on one exchange (e.g. Kraken) + short equivalent perp on another
(e.g. Bybit). Net directional exposure = zero. Collect funding payments.

Entry conditions:
  - Predicted funding rate > MIN_FUNDING_RATE_BPS (default 3.0 bps = 0.03%)
  - Annualised rate > MIN_ANNUAL_PCT (default 15%)
  - Sufficient spot liquidity on both venues
  - No existing position in this pair

Exit conditions:
  - Funding rate drops below EXIT_FUNDING_BPS (default 1.0 bps)
  - Position age > MAX_HOLD_PERIODS (default 72 * 8h = 576h)
  - Basis risk > MAX_BASIS_BPS (spot vs perp diverge > 50 bps)

P&L = funding_collected - borrow_cost - trading_fees
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FUNDING_RATE_BPS: float = 3.0    # minimum predicted funding to open (bps per 8h)
MIN_ANNUAL_PCT: float = 15.0         # minimum annualised rate to open (%)
EXIT_FUNDING_BPS: float = 1.0        # close position when funding falls below this
MAX_HOLD_PERIODS: int = 72           # max 8-hour periods to hold (= 576 hours)
MAX_BASIS_BPS: float = 50.0          # max tolerable spot-perp basis divergence (bps)
FUNDING_PERIODS_PER_DAY: int = 3     # 3 × 8h = 24h


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ArbPosition:
    """Represents an active delta-neutral arb position."""

    symbol: str
    spot_exchange: str
    perp_exchange: str
    spot_qty: float
    perp_qty: float
    spot_entry: float            # spot entry price (USD)
    perp_entry: float            # perp entry price (USD)
    funding_collected_bps: float # cumulative funding received (bps)
    periods_held: int            # number of 8-hour periods held
    opened_ts: float             # unix timestamp when opened
    basis_bps: float             # current basis: (perp - spot) / spot * 10_000


@dataclass
class ArbSignal:
    """Signal emitted by DeltaNeutralPerpArb for a given symbol."""

    symbol: str
    action: str                  # ENTER | EXIT | HOLD
    predicted_funding_bps: float # predicted funding for next period (bps)
    annual_rate_pct: float       # annualised equivalent (%)
    basis_bps: float             # current spot-perp basis (bps)
    reason: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class DeltaNeutralPerpArb:
    """
    Delta-neutral perpetual funding arbitrage strategy.

    Evaluates funding opportunities and manages the lifecycle of arb positions.
    Does NOT place real orders — returns signals only.  Safe in both
    paper-trading and simulation modes.

    Thread-safe: all mutable state protected by a single lock.
    """

    def __init__(
        self,
        spot_exchange: str = "kraken",
        perp_exchange: str = "bybit",
        capital_per_trade_usd: float = 500.0,
    ) -> None:
        self._spot_exchange = spot_exchange
        self._perp_exchange = perp_exchange
        self._capital = capital_per_trade_usd
        self._lock = threading.Lock()

        # symbol -> ArbPosition (active only)
        self._active: Dict[str, ArbPosition] = {}
        # closed positions kept for summary statistics
        self._closed: List[ArbPosition] = []

        logger.info(
            "DeltaNeutralPerpArb initialised: spot=%s perp=%s capital=%.0f USD",
            spot_exchange,
            perp_exchange,
            capital_per_trade_usd,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        spot_price: float,
        perp_price: float,
        predicted_funding_bps: float,
        funding_time_to_hours: float,  # noqa: ARG002  (reserved for future decay)
    ) -> ArbSignal:
        """
        Evaluate whether to enter, hold, or exit an arb for *symbol*.

        Parameters
        ----------
        symbol:
            E.g. ``"BTC/USD"``.
        spot_price:
            Current spot mid-price on *spot_exchange*.
        perp_price:
            Current perp mid-price on *perp_exchange*.
        predicted_funding_bps:
            Model-predicted funding for the upcoming settlement period (bps).
        funding_time_to_hours:
            Hours until the next funding settlement (informational).

        Returns
        -------
        ArbSignal
        """
        basis_bps = self._compute_basis_bps(spot_price, perp_price)
        annual_rate_pct = self._annualise(predicted_funding_bps)

        with self._lock:
            existing = self._active.get(symbol)

        if existing is not None:
            # We already hold a position — evaluate exit
            return self._evaluate_exit(
                symbol, existing, spot_price, perp_price,
                predicted_funding_bps, basis_bps, annual_rate_pct,
            )

        # No position — evaluate entry
        return self._evaluate_entry(
            symbol, spot_price, perp_price,
            predicted_funding_bps, basis_bps, annual_rate_pct,
        )

    def open_position(
        self,
        signal: ArbSignal,
        spot_price: float,
        perp_price: float,
    ) -> ArbPosition:
        """
        Record that an arb position was opened.

        Call this after the caller has confirmed both legs are filled.
        """
        if signal.action != "ENTER":
            raise ValueError(
                f"open_position called with non-ENTER signal: {signal.action}"
            )

        qty = self._capital / spot_price
        basis_bps = self._compute_basis_bps(spot_price, perp_price)
        pos = ArbPosition(
            symbol=signal.symbol,
            spot_exchange=self._spot_exchange,
            perp_exchange=self._perp_exchange,
            spot_qty=qty,
            perp_qty=qty,
            spot_entry=spot_price,
            perp_entry=perp_price,
            funding_collected_bps=0.0,
            periods_held=0,
            opened_ts=time.time(),
            basis_bps=basis_bps,
        )
        with self._lock:
            self._active[signal.symbol] = pos

        logger.info(
            "Opened arb position: symbol=%s qty=%.6f spot=%.2f perp=%.2f basis=%.2f bps",
            signal.symbol, qty, spot_price, perp_price, basis_bps,
        )
        return pos

    def update_position(
        self,
        symbol: str,
        spot_price: float,
        perp_price: float,
        latest_funding_bps: float,
    ) -> ArbSignal:
        """
        Called every 8 hours at funding settlement.

        Accumulates collected funding, increments periods_held, refreshes
        basis, and returns a signal indicating whether to continue or exit.
        """
        with self._lock:
            pos = self._active.get(symbol)

        if pos is None:
            return ArbSignal(
                symbol=symbol,
                action="HOLD",
                predicted_funding_bps=latest_funding_bps,
                annual_rate_pct=self._annualise(latest_funding_bps),
                basis_bps=self._compute_basis_bps(spot_price, perp_price),
                reason="no_active_position",
            )

        basis_bps = self._compute_basis_bps(spot_price, perp_price)
        with self._lock:
            pos.funding_collected_bps += latest_funding_bps
            pos.periods_held += 1
            pos.basis_bps = basis_bps

        annual_rate_pct = self._annualise(latest_funding_bps)
        logger.debug(
            "Funding settlement: symbol=%s funding=%.4f bps periods_held=%d "
            "total_collected=%.4f bps basis=%.2f bps",
            symbol, latest_funding_bps, pos.periods_held,
            pos.funding_collected_bps, basis_bps,
        )

        return self._evaluate_exit(
            symbol, pos, spot_price, perp_price,
            latest_funding_bps, basis_bps, annual_rate_pct,
        )

    def close_position(self, symbol: str) -> Optional[ArbPosition]:
        """
        Remove and return the closed position for *symbol*.

        Call after both legs have been exited.
        """
        with self._lock:
            pos = self._active.pop(symbol, None)
            if pos is not None:
                self._closed.append(pos)

        if pos is None:
            logger.warning("close_position: no active position for %s", symbol)
        else:
            logger.info(
                "Closed arb position: symbol=%s funding_collected=%.4f bps periods=%d",
                symbol, pos.funding_collected_bps, pos.periods_held,
            )
        return pos

    def get_active_positions(self) -> Dict[str, ArbPosition]:
        """Return a snapshot of all active positions."""
        with self._lock:
            return dict(self._active)

    def get_summary(self) -> Dict:
        """
        Return aggregate statistics.

        Returns
        -------
        dict with keys:
            total_funding_bps, n_active, n_closed, avg_annual_rate_pct
        """
        with self._lock:
            active = list(self._active.values())
            closed = list(self._closed)

        all_positions = active + closed
        total_funding = sum(p.funding_collected_bps for p in all_positions)
        avg_annual = 0.0
        if all_positions:
            avg_funding_per_period = total_funding / sum(
                max(p.periods_held, 1) for p in all_positions
            )
            avg_annual = self._annualise(avg_funding_per_period)

        return {
            "total_funding_bps": total_funding,
            "n_active": len(active),
            "n_closed": len(closed),
            "avg_annual_rate_pct": round(avg_annual, 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_basis_bps(self, spot: float, perp: float) -> float:
        """Basis = (perp - spot) / spot * 10_000 (bps)."""
        if spot <= 0:
            return 0.0
        return (perp - spot) / spot * 10_000.0

    def _annualise(self, funding_bps: float) -> float:
        """Convert per-8h funding (bps) to annualised percent."""
        # 3 periods/day * 365 days = 1095 periods/year
        return funding_bps * FUNDING_PERIODS_PER_DAY * 365 / 100.0

    def _evaluate_entry(
        self,
        symbol: str,
        spot_price: float,   # noqa: ARG002
        perp_price: float,   # noqa: ARG002
        predicted_funding_bps: float,
        basis_bps: float,
        annual_rate_pct: float,
    ) -> ArbSignal:
        """Evaluate whether entry conditions are met."""
        if predicted_funding_bps < MIN_FUNDING_RATE_BPS:
            return ArbSignal(
                symbol=symbol,
                action="HOLD",
                predicted_funding_bps=predicted_funding_bps,
                annual_rate_pct=annual_rate_pct,
                basis_bps=basis_bps,
                reason=f"funding_too_low:{predicted_funding_bps:.3f}<{MIN_FUNDING_RATE_BPS}bps",
            )

        if annual_rate_pct < MIN_ANNUAL_PCT:
            return ArbSignal(
                symbol=symbol,
                action="HOLD",
                predicted_funding_bps=predicted_funding_bps,
                annual_rate_pct=annual_rate_pct,
                basis_bps=basis_bps,
                reason=f"annual_rate_too_low:{annual_rate_pct:.1f}%<{MIN_ANNUAL_PCT}%",
            )

        if abs(basis_bps) > MAX_BASIS_BPS:
            return ArbSignal(
                symbol=symbol,
                action="HOLD",
                predicted_funding_bps=predicted_funding_bps,
                annual_rate_pct=annual_rate_pct,
                basis_bps=basis_bps,
                reason=f"basis_too_wide:{abs(basis_bps):.1f}>{MAX_BASIS_BPS}bps",
            )

        logger.info(
            "ENTER signal: symbol=%s funding=%.3f bps annual=%.1f%% basis=%.2f bps",
            symbol, predicted_funding_bps, annual_rate_pct, basis_bps,
        )
        return ArbSignal(
            symbol=symbol,
            action="ENTER",
            predicted_funding_bps=predicted_funding_bps,
            annual_rate_pct=annual_rate_pct,
            basis_bps=basis_bps,
            reason="entry_conditions_met",
        )

    def _evaluate_exit(
        self,
        symbol: str,
        pos: ArbPosition,
        spot_price: float,   # noqa: ARG002
        perp_price: float,   # noqa: ARG002
        latest_funding_bps: float,
        basis_bps: float,
        annual_rate_pct: float,
    ) -> ArbSignal:
        """Evaluate whether exit conditions are met for an existing position."""
        # Exit: funding rate fallen too low
        if latest_funding_bps < EXIT_FUNDING_BPS:
            logger.info(
                "EXIT signal (funding_low): symbol=%s funding=%.4f bps",
                symbol, latest_funding_bps,
            )
            return ArbSignal(
                symbol=symbol,
                action="EXIT",
                predicted_funding_bps=latest_funding_bps,
                annual_rate_pct=annual_rate_pct,
                basis_bps=basis_bps,
                reason=f"funding_below_exit:{latest_funding_bps:.4f}<{EXIT_FUNDING_BPS}bps",
            )

        # Exit: max hold exceeded
        if pos.periods_held >= MAX_HOLD_PERIODS:
            logger.info(
                "EXIT signal (max_hold): symbol=%s periods=%d",
                symbol, pos.periods_held,
            )
            return ArbSignal(
                symbol=symbol,
                action="EXIT",
                predicted_funding_bps=latest_funding_bps,
                annual_rate_pct=annual_rate_pct,
                basis_bps=basis_bps,
                reason=f"max_hold_exceeded:{pos.periods_held}>={MAX_HOLD_PERIODS}",
            )

        # Exit: basis risk too wide
        if abs(basis_bps) > MAX_BASIS_BPS:
            logger.warning(
                "EXIT signal (basis_risk): symbol=%s basis=%.2f bps",
                symbol, basis_bps,
            )
            return ArbSignal(
                symbol=symbol,
                action="EXIT",
                predicted_funding_bps=latest_funding_bps,
                annual_rate_pct=annual_rate_pct,
                basis_bps=basis_bps,
                reason=f"basis_risk:{abs(basis_bps):.1f}>{MAX_BASIS_BPS}bps",
            )

        return ArbSignal(
            symbol=symbol,
            action="HOLD",
            predicted_funding_bps=latest_funding_bps,
            annual_rate_pct=annual_rate_pct,
            basis_bps=basis_bps,
            reason=f"holding:periods={pos.periods_held} collected={pos.funding_collected_bps:.2f}bps",
        )
