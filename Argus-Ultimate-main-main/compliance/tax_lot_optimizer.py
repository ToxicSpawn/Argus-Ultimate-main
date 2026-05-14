"""
Tax Lot Optimizer — selects which cost lots to sell to minimise CGT.

Strategies:
  - HIFO (Highest In, First Out): minimises gain by using highest cost base
  - LOFO (Lowest In, First Out): maximises gain (useful if you have losses to use)
  - MIN_TAX: picks lot to minimise current-year tax considering discount eligibility
  - DISCOUNT_FIRST: prefer lots held > 12 months (maximise discount eligible gains)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TWELVE_MONTHS_SECONDS = 365.25 * 86400


@dataclass
class TaxLot:
    lot_id: str
    asset: str
    quantity: float
    cost_per_unit_aud: float
    acquisition_ts: float
    exchange: str = ""


@dataclass
class LotSelection:
    lot_id: str
    quantity_used: float
    cost_base_aud: float
    is_discount_eligible: bool
    estimated_tax_aud: float


class TaxLotOptimizer:
    """
    Selects optimal cost lots when disposing of crypto to minimise CGT.

    Parameters
    ----------
    strategy : str
        One of: "HIFO", "LOFO", "MIN_TAX", "DISCOUNT_FIRST".
    tax_rate : float
        Marginal income tax rate (default 32.5% — middle ATO bracket).
    cgt_discount : float
        CGT discount fraction (default 50%).
    """

    STRATEGIES = ("HIFO", "LOFO", "MIN_TAX", "DISCOUNT_FIRST")

    def __init__(
        self,
        strategy: str = "MIN_TAX",
        tax_rate: float = 0.325,
        cgt_discount: float = 0.50,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.tax_rate = tax_rate
        self.cgt_discount = cgt_discount
        self._lots: Dict[str, List[TaxLot]] = {}  # asset → list of lots

    # ------------------------------------------------------------------
    # Lot management
    # ------------------------------------------------------------------

    def add_lot(self, lot: TaxLot) -> None:
        """Register an acquisition lot."""
        asset = lot.asset.upper()
        if asset not in self._lots:
            self._lots[asset] = []
        self._lots[asset].append(lot)
        logger.debug("TaxLot: added lot %s for %s qty=%.6f cost=AUD%.2f",
                     lot.lot_id, asset, lot.quantity, lot.cost_per_unit_aud)

    def available_lots(self, asset: str) -> List[TaxLot]:
        """Return available lots sorted according to current strategy."""
        asset = asset.upper()
        lots = [l for l in self._lots.get(asset, []) if l.quantity > 0]
        if self.strategy == "HIFO":
            return sorted(lots, key=lambda l: l.cost_per_unit_aud, reverse=True)
        elif self.strategy == "LOFO":
            return sorted(lots, key=lambda l: l.cost_per_unit_aud)
        elif self.strategy == "MIN_TAX":
            # Sort by estimated tax per unit ascending (cheapest tax first)
            now = time.time()
            def tax_key(lot: TaxLot) -> float:
                gain = 1.0  # proxy per unit (actual proceeds unknown here)
                discount = lot.cost_per_unit_aud  # higher cost base → lower gain → lower tax
                eligible = (now - lot.acquisition_ts) >= TWELVE_MONTHS_SECONDS
                effective = discount if eligible else discount / (1 - self.cgt_discount)
                return -effective  # higher cost base → sort first (ascending tax)
            return sorted(lots, key=tax_key)
        elif self.strategy == "DISCOUNT_FIRST":
            # Discount-eligible lots first (held > 12 months), then by FIFO
            now = time.time()
            return sorted(lots, key=lambda l: (
                0 if (now - l.acquisition_ts) >= TWELVE_MONTHS_SECONDS else 1,
                l.acquisition_ts,
            ))
        return lots

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_lots(
        self,
        asset: str,
        quantity_to_sell: float,
        current_ts: float,
        proceeds_aud: float,
    ) -> List[LotSelection]:
        """
        Select which lots to use when selling `quantity_to_sell` of `asset`.

        Returns a list of LotSelection detailing which lots are used and
        the estimated tax for each.
        """
        asset = asset.upper()
        if self.strategy == "HIFO":
            return self._select_hifo(asset, quantity_to_sell, current_ts, proceeds_aud)
        elif self.strategy == "LOFO":
            return self._select_lofo(asset, quantity_to_sell, current_ts, proceeds_aud)
        elif self.strategy in ("MIN_TAX", "DISCOUNT_FIRST"):
            return self._select_min_tax(asset, quantity_to_sell, current_ts, proceeds_aud)
        return self._select_fifo(asset, quantity_to_sell, current_ts, proceeds_aud)

    def estimate_tax(self, selections: List[LotSelection], proceeds_aud: float) -> float:
        """Estimate total AUD tax for the given lot selections."""
        total_cost = sum(s.cost_base_aud for s in selections)
        total_qty = sum(s.quantity_used for s in selections)
        if total_qty <= 0:
            return 0.0

        gain = proceeds_aud - total_cost
        if gain <= 0:
            return 0.0

        # Weighted discount eligibility
        eligible_qty = sum(s.quantity_used for s in selections if s.is_discount_eligible)
        eligible_frac = eligible_qty / total_qty
        eligible_gain = gain * eligible_frac
        ineligible_gain = gain * (1 - eligible_frac)

        taxable = ineligible_gain + eligible_gain * (1 - self.cgt_discount)
        return taxable * self.tax_rate

    # ------------------------------------------------------------------
    # Private selectors
    # ------------------------------------------------------------------

    def _select_hifo(self, asset: str, qty: float, ts: float, proceeds: float) -> List[LotSelection]:
        lots = sorted(self._lots.get(asset, []), key=lambda l: l.cost_per_unit_aud, reverse=True)
        return self._consume_lots(lots, qty, ts, proceeds)

    def _select_lofo(self, asset: str, qty: float, ts: float, proceeds: float) -> List[LotSelection]:
        lots = sorted(self._lots.get(asset, []), key=lambda l: l.cost_per_unit_aud)
        return self._consume_lots(lots, qty, ts, proceeds)

    def _select_min_tax(self, asset: str, qty: float, ts: float, proceeds: float) -> List[LotSelection]:
        lots = self.available_lots(asset)
        return self._consume_lots(lots, qty, ts, proceeds)

    def _select_fifo(self, asset: str, qty: float, ts: float, proceeds: float) -> List[LotSelection]:
        lots = sorted(self._lots.get(asset, []), key=lambda l: l.acquisition_ts)
        return self._consume_lots(lots, qty, ts, proceeds)

    def _consume_lots(
        self,
        lots: List[TaxLot],
        total_qty: float,
        disposal_ts: float,
        proceeds_aud: float,
    ) -> List[LotSelection]:
        selections: List[LotSelection] = []
        remaining = total_qty

        for lot in lots:
            if remaining <= 0:
                break
            if lot.quantity <= 0:
                continue

            use_qty = min(lot.quantity, remaining)
            cost_base = use_qty * lot.cost_per_unit_aud
            eligible = (disposal_ts - lot.acquisition_ts) >= TWELVE_MONTHS_SECONDS

            # Per-lot tax estimate
            unit_proceeds = proceeds_aud * (use_qty / total_qty) if total_qty > 0 else 0
            gain = unit_proceeds - cost_base
            if gain > 0:
                taxable = gain * (1 - self.cgt_discount) if eligible else gain
                est_tax = taxable * self.tax_rate
            else:
                est_tax = 0.0

            selections.append(LotSelection(
                lot_id=lot.lot_id,
                quantity_used=use_qty,
                cost_base_aud=cost_base,
                is_discount_eligible=eligible,
                estimated_tax_aud=est_tax,
            ))
            remaining -= use_qty

        if remaining > 0:
            logger.warning(
                "TaxLotOptimizer: %.6f qty of %s has no matching lots",
                remaining, lots[0].asset if lots else "?",
            )

        return selections
