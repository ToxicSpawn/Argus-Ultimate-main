#!/usr/bin/env python3
"""
End of Financial Year Tax-Loss Harvester — Australian EOFY-aware loss realisation.

Scans unrealised losses, calculates ATO tax savings at the configured marginal
rate, flags wash-sale risk (30-day ATO "bed and breakfast" window), and
prioritises candidates by deadline urgency and saving magnitude.

Key assumptions:
- Australian FY ends 30 June.  Default fiscal_year_end = ``"2026-06-30"``.
- Marginal tax rate defaults to **32.5%** (AU $45k–$120k bracket).
- Wash-sale lookback is **30 calendar days** (conservative; ATO may flag
  aggressive patterns even though AU law lacks a US-style IRC §1091).
- All monetary values are in AUD.
- CGT discount (50%) applies to assets held >12 months; losses from those
  assets are still fully deductible.

Usage::

    harvester = EOFYHarvester()
    candidates = harvester.scan_unrealized_losses(positions, prices)
    plan = harvester.get_eofy_strategy()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MARGINAL_RATE = 0.325   # 32.5% ATO bracket ($45k–$120k)
_WASH_SALE_DAYS = 30             # ATO "bed and breakfast" lookback
_DEFAULT_EOFY = "2026-06-30"

# Priority weights
_WEIGHT_SAVING = 0.4
_WEIGHT_DAYS = 0.3
_WEIGHT_WASH = 0.3


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HarvestCandidate:
    """A position eligible for tax-loss harvesting before EOFY."""

    symbol: str
    unrealized_loss_aud: float     # negative value = loss
    tax_saving_aud: float          # positive = expected tax reduction
    wash_sale_risk: bool           # True if recently sold and re-bought
    days_to_eofy: int              # calendar days until fiscal year end
    priority: float                # 0–1, higher = harvest sooner
    entry_price_aud: float = 0.0
    current_price_aud: float = 0.0
    quantity: float = 0.0
    days_held: int = 0


@dataclass
class EOFYPlan:
    """Complete EOFY tax-loss harvesting strategy."""

    candidates: List[HarvestCandidate]
    total_potential_saving: float   # sum of all candidate tax savings (AUD)
    recommended_actions: List[str]  # human-readable action items
    deadline_date: date             # fiscal year end date
    marginal_rate: float = _DEFAULT_MARGINAL_RATE
    wash_sale_window_days: int = _WASH_SALE_DAYS


# ---------------------------------------------------------------------------
# EOFY Harvester
# ---------------------------------------------------------------------------


class EOFYHarvester:
    """End-of-financial-year tax-loss harvesting engine.

    Parameters
    ----------
    marginal_rate : float
        ATO marginal tax rate for savings calculation.  Default 32.5%.
    wash_sale_days : int
        Number of days for wash-sale lookback window.  Default 30.
    recent_sales : set of str, optional
        Symbols sold within the wash-sale window (caller provides from
        trade ledger).  Used for wash-sale risk flagging.
    """

    def __init__(
        self,
        marginal_rate: float = _DEFAULT_MARGINAL_RATE,
        wash_sale_days: int = _WASH_SALE_DAYS,
        recent_sales: Optional[Set[str]] = None,
    ) -> None:
        self._marginal_rate = marginal_rate
        self._wash_sale_days = wash_sale_days
        self._recent_sales: Set[str] = recent_sales or set()
        self._candidates: List[HarvestCandidate] = []
        logger.info(
            "EOFYHarvester initialised — marginal_rate=%.1f%%, wash_sale_days=%d",
            marginal_rate * 100, wash_sale_days,
        )

    # ------------------------------------------------------------------
    # Recent sales tracking
    # ------------------------------------------------------------------

    def add_recent_sale(self, symbol: str) -> None:
        """Register a symbol as recently sold (within wash-sale window)."""
        self._recent_sales.add(symbol.upper())

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_unrealized_losses(
        self,
        positions: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
        fiscal_year_end: str = _DEFAULT_EOFY,
    ) -> List[HarvestCandidate]:
        """Scan open positions for unrealised losses.

        Parameters
        ----------
        positions : dict
            Mapping of symbol → position dict with keys:
            ``quantity``, ``entry_price_aud``, optionally ``entry_date`` (ISO str).
        current_prices : dict
            Mapping of symbol → current AUD price.
        fiscal_year_end : str
            ISO date string for the fiscal year end (default ``"2026-06-30"``).

        Returns
        -------
        list of HarvestCandidate
            Only positions with unrealised losses, sorted by priority descending.
        """
        eofy = date.fromisoformat(fiscal_year_end)
        today = date.today()
        days_to_eofy = max(0, (eofy - today).days)

        candidates: List[HarvestCandidate] = []

        for symbol, pos in positions.items():
            quantity = float(pos.get("quantity", 0))
            entry_price = float(pos.get("entry_price_aud", 0))
            current_price = current_prices.get(symbol, current_prices.get(symbol.upper(), 0))

            if quantity <= 0 or entry_price <= 0 or current_price <= 0:
                continue

            unrealized_pnl = (current_price - entry_price) * quantity

            # Only interested in losses
            if unrealized_pnl >= 0:
                continue

            loss_aud = abs(unrealized_pnl)
            tax_saving = loss_aud * self._marginal_rate

            # Wash-sale risk
            wash_risk = symbol.upper() in self._recent_sales

            # Days held
            entry_date_str = pos.get("entry_date")
            if entry_date_str:
                try:
                    entry_date = date.fromisoformat(str(entry_date_str)[:10])
                    days_held = (today - entry_date).days
                except (ValueError, TypeError):
                    days_held = 0
            else:
                days_held = 0

            # Priority: higher saving + fewer days to EOFY + no wash risk
            saving_score = min(1.0, tax_saving / 500.0)  # normalise to ~$500
            urgency_score = max(0.0, 1.0 - days_to_eofy / 180.0) if days_to_eofy < 180 else 0.0
            wash_penalty = 0.0 if not wash_risk else 1.0
            priority = (
                _WEIGHT_SAVING * saving_score
                + _WEIGHT_DAYS * urgency_score
                + _WEIGHT_WASH * (1.0 - wash_penalty)
            )

            candidates.append(
                HarvestCandidate(
                    symbol=symbol,
                    unrealized_loss_aud=round(-loss_aud, 2),
                    tax_saving_aud=round(tax_saving, 2),
                    wash_sale_risk=wash_risk,
                    days_to_eofy=days_to_eofy,
                    priority=round(priority, 4),
                    entry_price_aud=entry_price,
                    current_price_aud=current_price,
                    quantity=quantity,
                    days_held=days_held,
                )
            )

        candidates.sort(key=lambda c: -c.priority)
        self._candidates = candidates

        logger.info(
            "EOFYHarvester: scanned %d positions, found %d loss candidates "
            "(total potential saving $%.2f AUD)",
            len(positions), len(candidates),
            sum(c.tax_saving_aud for c in candidates),
        )
        return candidates

    # ------------------------------------------------------------------
    # EOFY strategy
    # ------------------------------------------------------------------

    def get_eofy_strategy(
        self,
        fiscal_year_end: str = _DEFAULT_EOFY,
    ) -> EOFYPlan:
        """Generate a complete EOFY tax-loss harvesting plan.

        Parameters
        ----------
        fiscal_year_end : str
            ISO date for fiscal year end.

        Returns
        -------
        EOFYPlan
            Plan with candidates, total saving, and recommended actions.
        """
        eofy = date.fromisoformat(fiscal_year_end)
        today = date.today()
        days_to_eofy = max(0, (eofy - today).days)

        candidates = list(self._candidates)
        total_saving = sum(c.tax_saving_aud for c in candidates)

        actions: List[str] = []

        if not candidates:
            actions.append("No unrealised losses available for harvesting.")
        else:
            # Separate by wash-sale risk
            clean = [c for c in candidates if not c.wash_sale_risk]
            risky = [c for c in candidates if c.wash_sale_risk]

            if clean:
                actions.append(
                    f"Harvest {len(clean)} position(s) with no wash-sale risk "
                    f"(saving ${sum(c.tax_saving_aud for c in clean):.2f} AUD)."
                )
                for c in clean[:5]:
                    actions.append(
                        f"  - Sell {c.symbol}: loss ${abs(c.unrealized_loss_aud):.2f}, "
                        f"saving ${c.tax_saving_aud:.2f}"
                    )

            if risky:
                actions.append(
                    f"WARNING: {len(risky)} position(s) have wash-sale risk "
                    f"(sold within {self._wash_sale_days} days). "
                    f"Wait {self._wash_sale_days} days before re-entering."
                )

            if days_to_eofy <= 30:
                actions.append(
                    f"URGENT: Only {days_to_eofy} days until EOFY ({fiscal_year_end}). "
                    f"Harvest losses NOW to claim in this FY."
                )
            elif days_to_eofy <= 90:
                actions.append(
                    f"NOTICE: {days_to_eofy} days until EOFY. "
                    f"Begin harvesting positions with largest losses."
                )

            # Settlement time warning (T+0 for crypto but withdrawal delays)
            if days_to_eofy <= 7:
                actions.append(
                    "CRITICAL: Less than 7 days to EOFY. Ensure trades settle "
                    "before 30 June midnight AEST."
                )

        plan = EOFYPlan(
            candidates=candidates,
            total_potential_saving=round(total_saving, 2),
            recommended_actions=actions,
            deadline_date=eofy,
            marginal_rate=self._marginal_rate,
            wash_sale_window_days=self._wash_sale_days,
        )

        logger.info(
            "EOFYHarvester: EOFY plan — %d candidates, total saving $%.2f AUD, "
            "%d days to deadline",
            len(candidates), total_saving, days_to_eofy,
        )
        return plan
