"""
ATO Capital Gains Tax Calculator — Australian Tax Office CGT compliance.

Australian CGT rules for crypto:
  - Each crypto disposal is a CGT event
  - Cost base = AUD value at acquisition
  - Capital gain = proceeds - cost_base
  - 50% CGT discount if held > 12 months
  - Losses can offset gains (not income)

FX gain/loss notes:
  - When trading USD-quoted pairs with AUD capital, the USD acquired/disposed
    at each trade creates a separate AUD/USD FX CGT event under TR 2014/25.
  - record_fx_event() records each such event; get_fx_gains_report() aggregates
    them for the financial year.

DISCLAIMER: Not financial/tax advice. Verify with a registered tax agent.
"""
from __future__ import annotations

import csv
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Acquisition:
    asset: str
    quantity: float
    cost_base_aud: float
    timestamp: float
    exchange: str = ""
    notes: str = ""


@dataclass
class Disposal:
    asset: str
    quantity: float
    proceeds_aud: float
    cost_base_aud: float
    capital_gain_aud: float
    discount_eligible: bool  # True if held > 12 months
    discounted_gain_aud: float
    timestamp: float
    exchange: str = ""


@dataclass
class FXEvent:
    """
    AUD/USD foreign-exchange CGT event arising from a USD-denominated trade.

    Under TR 2014/25, each time an Australian resident acquires or disposes of
    a USD asset (including the USD leg of a crypto trade), the AUD/USD movement
    between acquisition and settlement creates a separate capital gain or loss.

    Fields
    ------
    amount_usd          : USD amount involved in the trade.
    rate_at_trade       : AUD/USD rate when the position was opened (USD per 1 AUD).
                          The AUD cost base = amount_usd / rate_at_trade.
    rate_at_settlement  : AUD/USD rate at settlement / disposal of the USD leg.
                          The AUD proceeds = amount_usd / rate_at_settlement.
    gain_aud            : Computed gain (positive) or loss (negative) in AUD.
    timestamp           : Unix timestamp of the settlement event.
    """
    amount_usd: float
    rate_at_trade: float
    rate_at_settlement: float
    gain_aud: float
    timestamp: float


class ATOCapitalGainsTracker:
    """
    Tracks crypto acquisitions and disposals for ATO CGT compliance.
    Uses FIFO (first-in, first-out) for cost base matching.

    Australian financial year starts 1 July (month 7).
    """

    CGT_DISCOUNT = 0.50  # 50% discount for assets held > 12 months
    TWELVE_MONTHS_SECONDS = 365.25 * 86400

    def __init__(self, financial_year_start_month: int = 7) -> None:
        self.fy_start_month = financial_year_start_month
        # Per-asset acquisition queues (FIFO)
        self._acquisitions: Dict[str, Deque[Acquisition]] = {}
        # Complete acquisition history (never pruned — needed for wash-sale detection)
        self._all_acquisitions: List[Acquisition] = []
        # All disposal records
        self._disposals: List[Disposal] = []
        # AUD/USD FX CGT events (TR 2014/25)
        self._fx_events: List[FXEvent] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_acquisition(
        self,
        asset: str,
        quantity: float,
        cost_base_aud: float,
        timestamp: float,
        exchange: str = "",
        notes: str = "",
    ) -> None:
        """Record a new crypto acquisition (buy or receive)."""
        asset = asset.upper()
        if asset not in self._acquisitions:
            self._acquisitions[asset] = deque()
        acq = Acquisition(
            asset=asset,
            quantity=quantity,
            cost_base_aud=cost_base_aud,
            timestamp=timestamp,
            exchange=exchange,
            notes=notes,
        )
        self._acquisitions[asset].append(acq)
        self._all_acquisitions.append(acq)
        logger.debug("ATO: acquired %.6f %s @ AUD %.2f", quantity, asset, cost_base_aud)

    def record_disposal(
        self,
        asset: str,
        quantity: float,
        proceeds_aud: float,
        timestamp: float,
        exchange: str = "",
    ) -> Disposal:
        """
        Record a disposal event and compute capital gain/loss.

        Uses FIFO to match against acquisition lots.
        Returns the computed Disposal dataclass.
        """
        asset = asset.upper()
        # Check CGT discount eligibility BEFORE consuming lots (FIFO oldest lot)
        discount_eligible = self._check_discount_eligible(asset, quantity, timestamp)
        cost_base = self._match_cost_base(asset, quantity, timestamp)
        capital_gain = proceeds_aud - cost_base
        discounted = capital_gain * (1 - self.CGT_DISCOUNT) if (discount_eligible and capital_gain > 0) else capital_gain

        disposal = Disposal(
            asset=asset,
            quantity=quantity,
            proceeds_aud=proceeds_aud,
            cost_base_aud=cost_base,
            capital_gain_aud=capital_gain,
            discount_eligible=discount_eligible,
            discounted_gain_aud=discounted,
            timestamp=timestamp,
            exchange=exchange,
        )
        self._disposals.append(disposal)
        logger.info(
            "ATO: disposed %.6f %s | gain AUD %.2f | discount=%s",
            quantity, asset, capital_gain, discount_eligible,
        )
        return disposal

    def record_fx_event(
        self,
        amount_usd: float,
        rate_at_trade: float,
        rate_at_settlement: float,
        date: float,
    ) -> FXEvent:
        """
        Record an AUD/USD FX gain/loss arising from a USD-denominated trade.

        When an Australian-resident trader uses AUD capital to trade a
        USD-quoted crypto pair, the USD acquired (or released) at each fill
        creates a separate FX CGT event under TR 2014/25.  The gain/loss is
        the difference between the AUD cost base of the USD and the AUD
        proceeds when the USD leg is settled or converted.

        Parameters
        ----------
        amount_usd        : Absolute USD amount involved (always positive).
        rate_at_trade     : AUD/USD rate at the time the USD was acquired
                            (i.e. when the crypto buy was executed).
                            Express as USD per 1 AUD (e.g. 0.65 means 1 AUD = 0.65 USD).
        rate_at_settlement: AUD/USD rate at settlement of the USD leg
                            (e.g. when the crypto was sold / USD converted back).
        date              : Unix timestamp of the settlement event.

        Returns
        -------
        FXEvent with the computed gain_aud stored and appended to the internal
        list.

        Example
        -------
        If you bought BTC with 650 USD when 1 AUD = 0.65 USD (cost base AUD 1000),
        and later the USD leg settled when 1 AUD = 0.70 USD (proceeds AUD 928.57),
        the FX loss is AUD -71.43.
        """
        if rate_at_trade <= 0 or rate_at_settlement <= 0:
            raise ValueError("FX rates must be positive")
        if amount_usd < 0:
            raise ValueError("amount_usd must be non-negative")

        cost_base_aud = amount_usd / rate_at_trade
        proceeds_aud = amount_usd / rate_at_settlement
        gain_aud = proceeds_aud - cost_base_aud

        event = FXEvent(
            amount_usd=amount_usd,
            rate_at_trade=rate_at_trade,
            rate_at_settlement=rate_at_settlement,
            gain_aud=gain_aud,
            timestamp=float(date),
        )
        self._fx_events.append(event)
        logger.debug(
            "ATO FX: USD %.2f | entry rate %.4f → settlement rate %.4f | gain AUD %.2f",
            amount_usd, rate_at_trade, rate_at_settlement, gain_aud,
        )
        return event

    def get_fx_gains_report(self, fy_start: float, fy_end: float) -> Dict:
        """
        Return a summary of AUD/USD FX gains and losses for a given period.

        Parameters
        ----------
        fy_start : Unix timestamp of the first second of the financial year
                   (inclusive).
        fy_end   : Unix timestamp of the first second of the *next* financial
                   year (exclusive).

        Returns
        -------
        Dictionary with keys:
          - ``total_fx_gains_aud``  : Sum of all positive FX gains.
          - ``total_fx_losses_aud`` : Sum of all negative FX gains (negative).
          - ``net_fx_gain_aud``     : Net FX gain/loss (gains + losses).
          - ``event_count``         : Number of FX events in the period.
          - ``events``              : List of raw FXEvent objects in the period.
        """
        in_period = [
            e for e in self._fx_events if fy_start <= e.timestamp < fy_end
        ]
        gains = sum(e.gain_aud for e in in_period if e.gain_aud > 0)
        losses = sum(e.gain_aud for e in in_period if e.gain_aud < 0)
        return {
            "total_fx_gains_aud": gains,
            "total_fx_losses_aud": losses,
            "net_fx_gain_aud": gains + losses,
            "event_count": len(in_period),
            "events": in_period,
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_fy_summary(self, financial_year: int) -> Dict:
        """
        Summary for the Australian financial year ending 30 June of `financial_year`.

        E.g. financial_year=2026 → FY2025-26 → 1 Jul 2025 to 30 Jun 2026.
        """
        fy_start, fy_end = self._fy_bounds(financial_year)
        in_fy = [d for d in self._disposals if fy_start <= d.timestamp < fy_end]

        total_gains = sum(d.capital_gain_aud for d in in_fy if d.capital_gain_aud > 0)
        total_losses = sum(d.capital_gain_aud for d in in_fy if d.capital_gain_aud < 0)
        net = total_gains + total_losses
        discount_eligible_gains = sum(
            d.capital_gain_aud * self.CGT_DISCOUNT
            for d in in_fy
            if d.discount_eligible and d.capital_gain_aud > 0
        )
        discounted_net = net - discount_eligible_gains

        by_asset: Dict[str, Dict] = {}
        for d in in_fy:
            if d.asset not in by_asset:
                by_asset[d.asset] = {"gain": 0.0, "loss": 0.0, "disposals": 0}
            if d.capital_gain_aud >= 0:
                by_asset[d.asset]["gain"] += d.capital_gain_aud
            else:
                by_asset[d.asset]["loss"] += d.capital_gain_aud
            by_asset[d.asset]["disposals"] += 1

        return {
            "financial_year": financial_year,
            "total_gains_aud": total_gains,
            "total_losses_aud": total_losses,
            "net_capital_gain_aud": net,
            "discount_deducted_aud": discount_eligible_gains,
            "discounted_net_aud": discounted_net,
            "disposal_count": len(in_fy),
            "by_asset": by_asset,
        }

    def generate_schedule(self, financial_year: int) -> List[Disposal]:
        """Return all disposals in the given financial year."""
        fy_start, fy_end = self._fy_bounds(financial_year)
        return [d for d in self._disposals if fy_start <= d.timestamp < fy_end]

    def export_csv(self, financial_year: int, output_path: str) -> None:
        """Export CGT schedule to CSV for ATO lodgement."""
        from datetime import datetime, timezone
        schedule = self.generate_schedule(financial_year)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Asset", "Quantity", "Proceeds AUD", "Cost Base AUD",
                "Capital Gain AUD", "Discount Eligible", "Discounted Gain AUD",
                "Disposal Date", "Exchange",
            ])
            for d in schedule:
                dt = datetime.fromtimestamp(d.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
                writer.writerow([
                    d.asset, f"{d.quantity:.8f}", f"{d.proceeds_aud:.2f}",
                    f"{d.cost_base_aud:.2f}", f"{d.capital_gain_aud:.2f}",
                    "Yes" if d.discount_eligible else "No",
                    f"{d.discounted_gain_aud:.2f}", dt, d.exchange,
                ])
        logger.info("ATO: exported %d disposals to %s", len(schedule), output_path)

    # ------------------------------------------------------------------
    # Wash sale / bed-and-breakfast detection
    # ------------------------------------------------------------------

    def check_wash_sale_risk(
        self, asset: str, lookback_days: int = 30
    ) -> Optional[dict]:
        """
        Pre-trade check: returns wash sale info if buying ``asset`` now would
        constitute a wash sale (i.e. same asset was sold at a loss within the
        last ``lookback_days``).

        Returns None if no wash sale risk, otherwise a dict with details of
        the most recent loss disposal.
        """
        import time as _time

        asset = asset.upper()
        now = _time.time()
        lookback_seconds = lookback_days * 86400
        window_start = now - lookback_seconds

        for disposal in reversed(self._disposals):
            if disposal.asset != asset:
                continue
            if disposal.timestamp < window_start:
                break  # disposals are appended chronologically
            if disposal.capital_gain_aud < 0:
                return {
                    "symbol": asset,
                    "disposal_date": disposal.timestamp,
                    "disposal_proceeds": disposal.proceeds_aud,
                    "disposal_cost_base": disposal.cost_base_aud,
                    "potential_disallowed_loss": abs(disposal.capital_gain_aud),
                    "days_since_disposal": (now - disposal.timestamp) / 86400,
                }
        return None

    def detect_wash_sales(self, lookback_days: int = 30) -> List[dict]:
        """
        Identify potential wash sales (bed-and-breakfast arrangements).

        Under ATO guidance, if a taxpayer disposes of an asset at a loss and
        reacquires the same (or substantially similar) asset within a short
        window, the CGT discount may not apply and the arrangement may be
        treated as a wash sale.  The traditional "bed and breakfast" period
        used by the ATO is 30 calendar days.

        This method scans all recorded disposals at a loss and checks whether
        a reacquisition of the same asset occurred within ``lookback_days``
        after the disposal.

        Parameters
        ----------
        lookback_days : int
            Number of calendar days after a loss disposal to look for a
            reacquisition of the same asset.  Default 30 (ATO convention).

        Returns
        -------
        List of dicts, each containing:
          - symbol                   : asset ticker (e.g. "BTC")
          - disposal_date            : Unix timestamp of the loss disposal
          - disposal_proceeds        : AUD proceeds of the disposal
          - disposal_cost_base       : AUD cost base of the disposal
          - reacquisition_date       : Unix timestamp of the matching buy
          - reacquisition_cost       : AUD cost base of the reacquisition
          - potential_disallowed_loss: absolute value of the loss that may
                                       be disallowed (positive number)
        """
        lookback_seconds = lookback_days * 86400
        flagged: List[dict] = []

        # Build per-asset sorted acquisition timeline from the complete
        # history (_all_acquisitions includes consumed lots).
        acq_timeline: Dict[str, List[Acquisition]] = {}
        for acq in self._all_acquisitions:
            acq_timeline.setdefault(acq.asset, []).append(acq)
        for asset in acq_timeline:
            acq_timeline[asset].sort(key=lambda a: a.timestamp)

        for disposal in self._disposals:
            if disposal.capital_gain_aud >= 0:
                continue  # not a loss — skip

            asset = disposal.asset
            window_start = disposal.timestamp
            window_end = disposal.timestamp + lookback_seconds

            # Find the earliest reacquisition within the lookback window
            for acq in acq_timeline.get(asset, []):
                if acq.timestamp <= window_start:
                    continue
                if acq.timestamp > window_end:
                    break  # sorted — no need to check further
                flagged.append({
                    "symbol": asset,
                    "disposal_date": disposal.timestamp,
                    "disposal_proceeds": disposal.proceeds_aud,
                    "disposal_cost_base": disposal.cost_base_aud,
                    "reacquisition_date": acq.timestamp,
                    "reacquisition_cost": acq.cost_base_aud,
                    "potential_disallowed_loss": abs(disposal.capital_gain_aud),
                })
                break  # only flag the first reacquisition per disposal

        logger.info("ATO wash-sale scan: %d potential wash sales flagged", len(flagged))
        return flagged

    def get_wash_sale_report(self, fy_start: float, fy_end: float) -> dict:
        """
        Summarise flagged wash sales for a given period.

        Parameters
        ----------
        fy_start : float
            Unix timestamp of the period start (inclusive).
        fy_end : float
            Unix timestamp of the period end (exclusive).

        Returns
        -------
        dict with keys:
          - flagged_count              : number of wash sales in period
          - total_disallowed_loss_aud  : sum of potentially disallowed losses
          - by_asset                   : {asset: count} breakdown
          - wash_sales                 : list of individual wash sale dicts
        """
        all_ws = self.detect_wash_sales()
        in_period = [
            ws for ws in all_ws
            if fy_start <= ws["disposal_date"] < fy_end
        ]
        by_asset: Dict[str, int] = {}
        total_disallowed = 0.0
        for ws in in_period:
            by_asset[ws["symbol"]] = by_asset.get(ws["symbol"], 0) + 1
            total_disallowed += ws["potential_disallowed_loss"]

        return {
            "flagged_count": len(in_period),
            "total_disallowed_loss_aud": total_disallowed,
            "by_asset": by_asset,
            "wash_sales": in_period,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _match_cost_base(self, asset: str, quantity: float, timestamp: float) -> float:
        """FIFO cost base matching. Removes consumed lots from queue."""
        queue = self._acquisitions.get(asset, deque())
        remaining = quantity
        total_cost = 0.0

        while remaining > 0 and queue:
            lot = queue[0]
            if lot.quantity <= remaining:
                # Consume entire lot
                total_cost += lot.cost_base_aud
                remaining -= lot.quantity
                queue.popleft()
            else:
                # Partial consume
                fraction = remaining / lot.quantity
                total_cost += lot.cost_base_aud * fraction
                lot.quantity -= remaining
                # Update cost_base proportionally
                lot.cost_base_aud *= (1 - fraction)
                remaining = 0

        if remaining > 0:
            logger.warning(
                "ATO: disposal of %.6f %s has %.6f qty without matching acquisition",
                quantity, asset, remaining,
            )

        return total_cost

    def _check_discount_eligible(self, asset: str, quantity: float, disposal_ts: float) -> bool:
        """Check if the FIFO lot would be discount-eligible (held > 12 months)."""
        queue = self._acquisitions.get(asset, deque())
        if not queue:
            return False
        # The oldest lot (FIFO) determines eligibility
        oldest = queue[0]
        return (disposal_ts - oldest.timestamp) >= self.TWELVE_MONTHS_SECONDS

    def _fy_bounds(self, financial_year: int):
        """Return (start_ts, end_ts) for an Australian financial year."""
        from datetime import datetime, timezone
        start = datetime(financial_year - 1, self.fy_start_month, 1, tzinfo=timezone.utc)
        end = datetime(financial_year, self.fy_start_month, 1, tzinfo=timezone.utc)
        return start.timestamp(), end.timestamp()
