"""
Funding Cost Limiter — prevents funding rate costs from eroding returns.

Perpetual futures charge/pay funding every 8 hours. At -0.1% per 8h,
a long position loses 0.1% every 8 hours = ~109% annual cost.

This module:
  - Tracks accumulated funding paid/received per position
  - Alerts when annualised funding cost exceeds threshold
  - Recommends position reduction or flip to spot
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Funding periods per day (every 8 hours = 3x/day)
_FUNDING_PERIODS_PER_DAY: int = 3
_DAYS_PER_YEAR: int = 365

# Lookback window for annualised cost calculation
_LOOKBACK_DAYS: int = 7

# Maximum number of payment records retained per (symbol, exchange) key
_MAX_HISTORY_PER_POSITION: int = 500


def _annualise_rate(rate_8h: float) -> float:
    """Convert an 8-hour funding rate to an annualised rate."""
    return rate_8h * _FUNDING_PERIODS_PER_DAY * _DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FundingPayment:
    """
    Record of a single funding payment or receipt.

    A positive *payment_usd* means the position received funding (income).
    A negative *payment_usd* means the position paid funding (cost).
    """

    symbol: str
    exchange: str
    rate_pct: float        # 8-hour rate as a percentage, e.g. -0.05 = -0.05%
    payment_usd: float     # signed USD amount (negative = cost)
    position_usd: float    # notional position size in USD at time of payment
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "rate_pct": self.rate_pct,
            "payment_usd": round(self.payment_usd, 6),
            "position_usd": round(self.position_usd, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------


class FundingCostLimiter:
    """
    Track and limit the cost of holding perpetual futures positions via funding.

    Parameters
    ----------
    max_annual_cost_pct:
        Annualised funding cost above which a position should be exited
        (default 50 %).  Example: 0.50 = 50 % per year.
    max_8h_rate_pct:
        A single 8-hour funding rate above this threshold (absolute value,
        as a positive percentage) triggers an immediate alert regardless of
        the rolling average (default 0.075 % = 7.5 bps per 8h).
    alert_threshold_pct:
        Annualised cost above this level generates a WARNING log but does not
        yet recommend an exit (default 30 %).
    """

    def __init__(
        self,
        max_annual_cost_pct: float = 0.50,
        max_8h_rate_pct: float = 0.075,
        alert_threshold_pct: float = 0.30,
    ) -> None:
        if max_annual_cost_pct <= 0:
            raise ValueError("max_annual_cost_pct must be positive")
        if max_8h_rate_pct <= 0:
            raise ValueError("max_8h_rate_pct must be positive")
        if not (0 < alert_threshold_pct <= max_annual_cost_pct):
            raise ValueError(
                "alert_threshold_pct must be positive and <= max_annual_cost_pct"
            )

        self.max_annual_cost_pct = max_annual_cost_pct
        self.max_8h_rate_pct = max_8h_rate_pct
        self.alert_threshold_pct = alert_threshold_pct

        # payments[symbol][exchange] = list of FundingPayment
        self._payments: Dict[str, Dict[str, List[FundingPayment]]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key_payments(self, symbol: str, exchange: str) -> List[FundingPayment]:
        """Return the payment list for (symbol, exchange), creating it lazily."""
        sym_map = self._payments.setdefault(symbol, {})
        return sym_map.setdefault(exchange, [])

    def _all_payments_for_symbol(self, symbol: str) -> List[FundingPayment]:
        """Return all payment records for *symbol* across all exchanges."""
        result: List[FundingPayment] = []
        for pmts in self._payments.get(symbol, {}).values():
            result.extend(pmts)
        return result

    def _payments_in_window(
        self, symbol: str, lookback_days: int = _LOOKBACK_DAYS
    ) -> List[FundingPayment]:
        """Return payments for *symbol* within the rolling lookback window."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        all_pmts = self._all_payments_for_symbol(symbol)
        return [p for p in all_pmts if p.timestamp >= cutoff]

    # ------------------------------------------------------------------
    # Public mutators
    # ------------------------------------------------------------------

    def record_payment(
        self,
        symbol: str,
        exchange: str,
        rate_pct: float,
        payment_usd: float,
        position_usd: float,
    ) -> None:
        """
        Record a funding payment/receipt for *symbol* on *exchange*.

        Parameters
        ----------
        symbol:
            Trading symbol, e.g. "BTC/USD".
        exchange:
            Exchange identifier, e.g. "kraken".
        rate_pct:
            8-hour funding rate as a percentage.  Negative = paying funding.
        payment_usd:
            Signed USD amount of this funding event (negative = cost).
        position_usd:
            Notional position size in USD at the time of the payment.
        """
        pmt = FundingPayment(
            symbol=symbol,
            exchange=exchange,
            rate_pct=rate_pct,
            payment_usd=payment_usd,
            position_usd=position_usd,
        )

        pmts = self._key_payments(symbol, exchange)
        pmts.append(pmt)

        # Trim old records to prevent unbounded growth
        if len(pmts) > _MAX_HISTORY_PER_POSITION:
            excess = len(pmts) - _MAX_HISTORY_PER_POSITION
            del pmts[:excess]

        # Alert on extreme single-period rate
        if abs(rate_pct) > self.max_8h_rate_pct:
            logger.warning(
                "FundingCostLimiter: extreme 8h funding rate %.4f%% for %s on %s "
                "(threshold ±%.4f%%)",
                rate_pct,
                symbol,
                exchange,
                self.max_8h_rate_pct,
            )

        # Check rolling annualised cost
        annual = self.get_annualised_cost(symbol)
        if annual > self.max_annual_cost_pct:
            logger.warning(
                "FundingCostLimiter: annualised funding cost %.1f%% for %s exceeds "
                "max %.1f%% — EXIT recommended",
                annual * 100.0,
                symbol,
                self.max_annual_cost_pct * 100.0,
            )
        elif annual > self.alert_threshold_pct:
            logger.warning(
                "FundingCostLimiter: annualised funding cost %.1f%% for %s exceeds "
                "alert threshold %.1f%%",
                annual * 100.0,
                symbol,
                self.alert_threshold_pct * 100.0,
            )
        else:
            logger.debug(
                "FundingCostLimiter: recorded %.6f USD funding for %s (annualised %.2f%%)",
                payment_usd,
                symbol,
                annual * 100.0,
            )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_annualised_cost(self, symbol: str) -> float:
        """
        Compute the annualised funding cost for *symbol* using the last 7 days
        of payment records.

        The cost is expressed as a fraction (e.g. 0.50 = 50 % per year).
        A positive value means net cost (paying funding); a negative value means
        net income (receiving funding).

        Uses the average 8-hour rate over the lookback window and annualises as:
            annual_cost = mean(|rate_8h|) * 3 * 365

        If the position is predominantly receiving funding (net income), the
        returned cost is negative (beneficial).
        """
        payments = self._payments_in_window(symbol)
        if not payments:
            return 0.0

        # Weighted average rate by position size to account for size changes
        total_weight = sum(abs(p.position_usd) for p in payments)
        if total_weight == 0:
            rates = [p.rate_pct / 100.0 for p in payments]
            avg_rate_8h = sum(rates) / len(rates)
        else:
            avg_rate_8h = sum(
                (p.rate_pct / 100.0) * abs(p.position_usd) for p in payments
            ) / total_weight

        # Annualise: multiply by periods per day * days per year
        annual = avg_rate_8h * _FUNDING_PERIODS_PER_DAY * _DAYS_PER_YEAR

        # Convention: positive = cost (negative rate on long), negative = income
        return annual

    def should_exit(self, symbol: str) -> bool:
        """
        Return True if the annualised funding cost for *symbol* exceeds the
        configured maximum.
        """
        return self.get_annualised_cost(symbol) > self.max_annual_cost_pct

    def get_recommendations(self) -> List[dict]:
        """
        Return a recommendation for each symbol with tracked funding payments.

        Each recommendation is a dict with:
            symbol               — asset symbol
            annualised_cost_pct  — cost as a percentage (e.g. 55.0 = 55 %)
            action               — "HOLD", "REDUCE", or "EXIT"
            note                 — human-readable explanation
        """
        recommendations: List[dict] = []
        all_symbols = list(self._payments.keys())

        for symbol in sorted(all_symbols):
            annual = self.get_annualised_cost(symbol)
            annual_pct = annual * 100.0

            if annual > self.max_annual_cost_pct:
                action = "EXIT"
                note = (
                    f"Annualised funding cost {annual_pct:.1f}% exceeds maximum "
                    f"{self.max_annual_cost_pct * 100:.1f}%. Consider closing or "
                    f"switching to spot."
                )
            elif annual > self.alert_threshold_pct:
                action = "REDUCE"
                note = (
                    f"Annualised funding cost {annual_pct:.1f}% exceeds alert "
                    f"threshold {self.alert_threshold_pct * 100:.1f}%. Consider "
                    f"reducing position size."
                )
            else:
                action = "HOLD"
                note = (
                    f"Annualised funding cost {annual_pct:.1f}% is within "
                    f"acceptable limits."
                )

            recommendations.append(
                {
                    "symbol": symbol,
                    "annualised_cost_pct": round(annual_pct, 4),
                    "action": action,
                    "note": note,
                }
            )

        return recommendations

    # ------------------------------------------------------------------
    # Exit recommendations
    # ------------------------------------------------------------------

    def get_exit_recommendations(self, exit_threshold_pct: Optional[float] = None) -> List[dict]:
        """
        Return positions where annualised funding cost exceeds the exit threshold.

        Each dict contains:
            symbol              — asset symbol
            exchange            — exchange with highest cost for this symbol
            annualized_cost_pct — cost as a fraction (e.g. 0.55 = 55%)
            recommendation      — "EXIT" (>= max_annual_cost_pct) or "REDUCE"

        Parameters
        ----------
        exit_threshold_pct:
            Override for the exit threshold (fraction).  Defaults to
            ``self.alert_threshold_pct`` so positions above the alert
            level are surfaced.
        """
        threshold = exit_threshold_pct if exit_threshold_pct is not None else self.alert_threshold_pct
        results: List[dict] = []

        for symbol in sorted(self._payments.keys()):
            annual = self.get_annualised_cost(symbol)
            if annual <= threshold:
                continue

            # Find the exchange with the most recent payment for context
            best_exchange = "unknown"
            latest_ts = None
            for exchange, pmts in self._payments.get(symbol, {}).items():
                if pmts:
                    last = pmts[-1].timestamp
                    if latest_ts is None or last > latest_ts:
                        latest_ts = last
                        best_exchange = exchange

            recommendation = "EXIT" if annual >= self.max_annual_cost_pct else "REDUCE"
            results.append({
                "symbol": symbol,
                "exchange": best_exchange,
                "annualized_cost_pct": annual,
                "recommendation": recommendation,
            })

        return results

    # ------------------------------------------------------------------
    # Snapshot / reporting
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return a complete funding cost snapshot.

        Returns
        -------
        dict with keys:
            positions              — per-symbol cost and recommendation
            total_funding_paid_30d — total USD paid in funding over last 30 days
                                     (negative = cost, positive = received)
            worst_position         — symbol with highest annualised cost
            recommendations        — output of get_recommendations()
            timestamp
        """
        cutoff_30d = datetime.now(timezone.utc) - timedelta(days=30)

        total_30d = 0.0
        positions_out: Dict[str, dict] = {}
        worst_symbol: Optional[str] = None
        worst_cost: float = -float("inf")

        for symbol in sorted(self._payments.keys()):
            annual_cost = self.get_annualised_cost(symbol)

            # 30-day total payments
            all_pmts = self._all_payments_for_symbol(symbol)
            paid_30d = sum(
                p.payment_usd for p in all_pmts if p.timestamp >= cutoff_30d
            )
            total_30d += paid_30d

            positions_out[symbol] = {
                "annualised_cost_pct": round(annual_cost * 100.0, 4),
                "paid_30d_usd": round(paid_30d, 6),
                "payment_count": len(all_pmts),
                "should_exit": self.should_exit(symbol),
            }

            if annual_cost > worst_cost:
                worst_cost = annual_cost
                worst_symbol = symbol

        return {
            "positions": positions_out,
            "total_funding_paid_30d": round(total_30d, 6),
            "worst_position": worst_symbol,
            "worst_position_annual_cost_pct": round(worst_cost * 100.0, 4) if worst_symbol else None,
            "recommendations": self.get_recommendations(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def __repr__(self) -> str:
        n_symbols = len(self._payments)
        return (
            f"FundingCostLimiter("
            f"max_annual={self.max_annual_cost_pct:.0%}, "
            f"max_8h_rate={self.max_8h_rate_pct:.4f}%, "
            f"alert={self.alert_threshold_pct:.0%}, "
            f"symbols={n_symbols})"
        )
