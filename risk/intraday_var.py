"""
Intraday VaR — real-time position-level and portfolio Value-at-Risk.

Updates every tick/fill using an EWMA (exponentially weighted moving average)
of returns. Much faster to react to volatility spikes than historical VaR.

Key metrics:
  - Position-level 95% and 99% VaR in USD
  - Portfolio-level VaR with correlation adjustment
  - VaR utilisation ratio (current VaR / VaR limit)
  - Breach alert when portfolio VaR > limit

Usage:
    var_tracker = IntradayVaR(capital_usd=1000.0, var_limit_pct=0.02)
    var_tracker.update_price("BTC/USD", 65000.0)
    var_tracker.update_position("BTC/USD", qty_usd=200.0)
    snapshot = var_tracker.snapshot()
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_PRICE_HISTORY = 100


@dataclass
class PositionVaR:
    """Value-at-Risk metrics for a single position."""

    symbol: str
    qty_usd: float
    var_95_usd: float
    var_99_usd: float
    ewma_vol: float  # annualised volatility estimate
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "qty_usd": self.qty_usd,
            "var_95_usd": round(self.var_95_usd, 4),
            "var_99_usd": round(self.var_99_usd, 4),
            "ewma_vol": round(self.ewma_vol, 6),
            "computed_at": self.computed_at.isoformat(),
        }


class IntradayVaR:
    """
    Real-time intraday Value-at-Risk tracker using EWMA volatility.

    Thread-safe — all public methods acquire a lock before mutating state.

    Parameters
    ----------
    capital_usd:
        Total portfolio capital in USD. Used to compute the VaR limit in dollar
        terms and the utilisation ratio.
    var_limit_pct:
        Maximum acceptable portfolio VaR as a fraction of capital (default 2 %).
    ewma_lambda:
        EWMA decay factor (RiskMetrics classic = 0.94). Higher value = slower
        reaction to new data.
    confidence_95:
        Z-score for 95 % one-tailed VaR (default 1.645).
    confidence_99:
        Z-score for 99 % one-tailed VaR (default 2.326).
    min_observations:
        Minimum number of price observations required before VaR is considered
        reliable. Returns zero VaR until this threshold is met.
    """

    def __init__(
        self,
        capital_usd: float,
        var_limit_pct: float = 0.02,
        ewma_lambda: float = 0.94,
        confidence_95: float = 1.645,
        confidence_99: float = 2.326,
        min_observations: int = 10,
    ) -> None:
        if capital_usd <= 0:
            raise ValueError("capital_usd must be positive")
        if not (0 < var_limit_pct < 1):
            raise ValueError("var_limit_pct must be between 0 and 1")
        if not (0 < ewma_lambda < 1):
            raise ValueError("ewma_lambda must be between 0 and 1")

        self.capital_usd = capital_usd
        self.var_limit_pct = var_limit_pct
        self.var_limit_usd = capital_usd * var_limit_pct
        self.ewma_lambda = ewma_lambda
        self.confidence_95 = confidence_95
        self.confidence_99 = confidence_99
        self.min_observations = min_observations

        # Per-symbol state
        self._prices: Dict[str, deque] = {}          # deque of last N prices
        self._ewma_var: Dict[str, float] = {}         # EWMA variance (squared returns)
        self._positions: Dict[str, float] = {}        # symbol -> qty_usd
        self._obs_count: Dict[str, int] = {}          # number of updates seen

        self._correlation_source: Optional[object] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Correlation source injection
    # ------------------------------------------------------------------

    def set_correlation_source(self, correlation_monitor: object) -> None:
        """
        Set a live correlation source for portfolio VaR calculation.

        The *correlation_monitor* must expose a method that returns the
        average pairwise correlation (float or None).  Accepted method
        names (tried in order):

          1. ``get_average_correlation()``
          2. ``_last_avg_corr`` attribute

        When set, ``compute_portfolio_var()`` uses the live correlation
        instead of the hardcoded 0.7 fallback.
        """
        with self._lock:
            self._correlation_source = correlation_monitor
        logger.info("IntradayVaR: live correlation source attached")

    def _resolve_correlation_factor(self) -> float:
        """Return the correlation factor to use for portfolio VaR.

        Tries to read a live value from the injected correlation source.
        Falls back to 0.7 on any error or if no source is set.
        """
        if self._correlation_source is None:
            return 0.7

        try:
            # Try get_average_correlation() method first
            getter = getattr(self._correlation_source, "get_average_correlation", None)
            if callable(getter):
                val = getter()
                if val is not None and isinstance(val, (int, float)):
                    return max(0.0, min(1.0, float(val)))

            # Fallback: _last_avg_corr attribute
            attr_val = getattr(self._correlation_source, "_last_avg_corr", None)
            if attr_val is not None and isinstance(attr_val, (int, float)):
                return max(0.0, min(1.0, float(attr_val)))
        except Exception:
            logger.debug("IntradayVaR: correlation source read failed; using 0.7 fallback")

        return 0.7

    # ------------------------------------------------------------------
    # Public mutators
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """
        Ingest a new price tick for *symbol* and update EWMA volatility.

        Parameters
        ----------
        symbol:
            Asset identifier, e.g. "BTC/USD".
        price:
            Latest mid/last price.
        """
        if price <= 0:
            logger.warning("IntradayVaR.update_price: non-positive price %.6f for %s — skipped", price, symbol)
            return

        with self._lock:
            if symbol not in self._prices:
                self._prices[symbol] = deque(maxlen=_MAX_PRICE_HISTORY)
                self._ewma_var[symbol] = 0.0
                self._obs_count[symbol] = 0

            history = self._prices[symbol]

            if len(history) >= 1:
                prev_price = history[-1]
                if prev_price > 0:
                    log_return = math.log(price / prev_price)
                    ret_sq = log_return ** 2

                    lam = self.ewma_lambda
                    prev_var = self._ewma_var[symbol]

                    if prev_var == 0.0:
                        # Seed with the first squared return directly
                        self._ewma_var[symbol] = ret_sq
                    else:
                        self._ewma_var[symbol] = lam * prev_var + (1.0 - lam) * ret_sq

                    self._obs_count[symbol] += 1

            history.append(price)

    def update_position(self, symbol: str, qty_usd: float) -> None:
        """
        Set the current USD exposure for *symbol*.

        Parameters
        ----------
        symbol:
            Asset identifier.
        qty_usd:
            Signed USD notional (positive = long, negative = short).
        """
        with self._lock:
            if qty_usd == 0.0:
                self._positions.pop(symbol, None)
                logger.debug("IntradayVaR: position for %s removed (qty_usd=0)", symbol)
            else:
                self._positions[symbol] = qty_usd
                logger.debug("IntradayVaR: position %s updated to %.2f USD", symbol, qty_usd)

    # ------------------------------------------------------------------
    # VaR computations
    # ------------------------------------------------------------------

    def compute_position_var(self, symbol: str) -> PositionVaR:
        """
        Compute position-level VaR for *symbol* at 95 % and 99 % confidence.

        Returns zero VaR if fewer than *min_observations* have been received.
        """
        with self._lock:
            return self._compute_position_var_locked(symbol)

    def _compute_position_var_locked(self, symbol: str) -> PositionVaR:
        """Internal — must be called while holding self._lock."""
        qty_usd = self._positions.get(symbol, 0.0)
        obs = self._obs_count.get(symbol, 0)
        ewma_var = self._ewma_var.get(symbol, 0.0)

        if obs < self.min_observations or ewma_var <= 0.0:
            return PositionVaR(
                symbol=symbol,
                qty_usd=qty_usd,
                var_95_usd=0.0,
                var_99_usd=0.0,
                ewma_vol=0.0,
            )

        # Daily vol from EWMA variance (ewma_var is a per-tick variance; treat
        # each tick as a 1-minute bar and annualise assuming 525,600 min/yr)
        # For intraday purposes we use the raw EWMA std as a relative measure.
        ewma_std = math.sqrt(ewma_var)

        # Annualised vol (approximate, assuming ~1-min ticks, 525 600 per year)
        annualised_vol = ewma_std * math.sqrt(525_600)

        abs_exposure = abs(qty_usd)
        var_95 = abs_exposure * ewma_std * self.confidence_95
        var_99 = abs_exposure * ewma_std * self.confidence_99

        return PositionVaR(
            symbol=symbol,
            qty_usd=qty_usd,
            var_95_usd=var_95,
            var_99_usd=var_99,
            ewma_vol=annualised_vol,
        )

    def compute_portfolio_var(self) -> float:
        """
        Compute portfolio-level VaR in USD.

        Uses a simplified correlation adjustment: portfolio VaR is the sum of
        individual 99 % VaRs multiplied by a constant correlation factor of 0.7.
        This is conservative but avoids the need for a full covariance matrix
        when the position count is small.

        Returns
        -------
        float
            Portfolio VaR in USD at 99 % confidence.
        """
        with self._lock:
            if not self._positions:
                return 0.0

            sum_var = 0.0
            for symbol in list(self._positions.keys()):
                pv = self._compute_position_var_locked(symbol)
                sum_var += pv.var_99_usd

            # Correlation adjustment: use live correlation if available, else 0.7
            # Diversified VaR = sqrt(N) * rho_factor * sum_var / N  (simplified)
            # Simpler: portfolio_var = correlation_factor * sum_var
            correlation_factor = self._resolve_correlation_factor()
            portfolio_var = correlation_factor * sum_var
            return portfolio_var

    # ------------------------------------------------------------------
    # Snapshot / reporting
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return a complete point-in-time snapshot of all VaR metrics.

        Returns
        -------
        dict with keys:
            positions        — per-symbol PositionVaR dicts
            portfolio_var_usd
            var_limit_usd
            utilisation_pct  — portfolio_var / limit * 100
            breach           — True if portfolio_var > var_limit_usd
            capital_usd
            timestamp
        """
        with self._lock:
            positions_out: List[dict] = []
            for symbol in list(self._positions.keys()):
                pv = self._compute_position_var_locked(symbol)
                positions_out.append(pv.to_dict())

        portfolio_var = self.compute_portfolio_var()
        utilisation = (portfolio_var / self.var_limit_usd * 100.0) if self.var_limit_usd > 0 else 0.0
        breach = portfolio_var > self.var_limit_usd

        if breach:
            logger.warning(
                "IntradayVaR BREACH: portfolio VaR $%.2f exceeds limit $%.2f (%.1f %% utilisation)",
                portfolio_var,
                self.var_limit_usd,
                utilisation,
            )

        return {
            "positions": positions_out,
            "portfolio_var_usd": round(portfolio_var, 4),
            "var_limit_usd": round(self.var_limit_usd, 4),
            "utilisation_pct": round(utilisation, 2),
            "breach": breach,
            "capital_usd": self.capital_usd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all state — intended for end-of-day resets.

        After calling this method the tracker starts fresh with no price
        history, EWMA variances, or position data.
        """
        with self._lock:
            self._prices.clear()
            self._ewma_var.clear()
            self._positions.clear()
            self._obs_count.clear()
        logger.info("IntradayVaR: state reset (EOD)")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def active_symbols(self) -> List[str]:
        """Return a sorted list of symbols with non-zero positions."""
        with self._lock:
            return sorted(self._positions.keys())

    def utilisation(self) -> float:
        """Return VaR utilisation as a fraction (0–1+)."""
        portfolio_var = self.compute_portfolio_var()
        if self.var_limit_usd <= 0:
            return 0.0
        return portfolio_var / self.var_limit_usd

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._positions)
        return (
            f"IntradayVaR(capital=${self.capital_usd:.0f}, "
            f"limit_pct={self.var_limit_pct:.1%}, "
            f"positions={n}, "
            f"lambda={self.ewma_lambda})"
        )
