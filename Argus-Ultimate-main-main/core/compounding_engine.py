"""
compounding_engine.py
=====================
Weekly profit reinvestment and capital allocation resizing for Argus trading bot.

At $4/day compounding with 100% reinvestment, capital doubles within ~18 months.
This module automates reinvestment and projects growth trajectories.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompoundingConfig:
    """Configuration for the compounding engine."""

    initial_capital_aud: float = 1000.0
    aud_usd_rate: float = 0.62
    # 1.0 = 100% reinvest; 0.5 = withdraw half, reinvest half
    reinvest_pct: float = 1.0
    reinvest_interval_days: int = 7
    # Only trigger a reinvestment cycle if accumulated net PnL > this amount
    min_reinvest_amount_usd: float = 10.0
    # MEXC VIP volume thresholds (USD/month) → maker fee
    fee_tier_thresholds: Dict[float, float] = field(
        default_factory=lambda: {
            1_000_000: 0.0,        # Lv1 – 0% maker
            5_000_000: -0.0001,    # Lv2 – -0.01% maker rebate
            10_000_000: -0.0002,   # Lv3 – -0.02% maker rebate
        }
    )
    # Target fractional allocations across strategies (must sum to 1.0)
    target_allocations: Dict[str, float] = field(
        default_factory=lambda: {
            "mm": 0.55,
            "funding_arb": 0.40,
            "reserve": 0.05,
        }
    )
    # Minimum capital deployed per active strategy in USD
    min_strategy_allocation_usd: float = 50.0


@dataclass
class ReinvestmentResult:
    """Result of a weekly reinvestment cycle."""

    date: date
    pnl_period_usd: float
    reinvested_usd: float
    withdrawn_usd: float
    new_capital_usd: float
    new_capital_aud: float
    new_allocations: Dict[str, float]
    fee_tier_upgraded: bool
    new_fee_tier: str


@dataclass
class GrowthSnapshot:
    """Weekly forward projection snapshot."""

    week: int
    date: date
    capital_usd: float
    capital_aud: float
    daily_return_usd: float
    mm_allocation: float
    funding_allocation: float
    reserve: float
    fee_tier: str
    annualised_return_pct: float


# ---------------------------------------------------------------------------
# MEXC fee tier helper
# ---------------------------------------------------------------------------

# Ordered list: (min_monthly_volume_usd, tier_name, maker_fee, taker_fee)
_MEXC_FEE_TIERS: List[Tuple[float, str, float, float]] = [
    (0.0,          "Lv0", 0.0000,  0.0005),
    (1_000_000.0,  "Lv1", 0.0000,  0.0004),
    (5_000_000.0,  "Lv2", -0.0001, 0.0002),
    (10_000_000.0, "Lv3", -0.0002, 0.0001),
]


def _mexc_tier_for_volume(monthly_volume_usd: float) -> str:
    """Return MEXC tier name for the given 30-day trading volume."""
    tier_name = "Lv0"
    for min_vol, name, _maker, _taker in _MEXC_FEE_TIERS:
        if monthly_volume_usd >= min_vol:
            tier_name = name
        else:
            break
    return tier_name


def _mexc_tier_fees(tier_name: str) -> Tuple[float, float]:
    """Return (maker_fee, taker_fee) for a MEXC tier name."""
    for _min_vol, name, maker, taker in _MEXC_FEE_TIERS:
        if name == tier_name:
            return maker, taker
    return 0.0, 0.0005  # Lv0 default


def _next_mexc_tier(current_tier: str) -> Optional[Tuple[str, float]]:
    """Return (next_tier_name, min_volume_required) or None if already at max."""
    for i, (_min_vol, name, _maker, _taker) in enumerate(_MEXC_FEE_TIERS):
        if name == current_tier and i + 1 < len(_MEXC_FEE_TIERS):
            next_min, next_name, _m, _t = _MEXC_FEE_TIERS[i + 1]
            return next_name, next_min
    return None


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class CompoundingEngine:
    """
    Manages weekly profit reinvestment, allocation resizing, and growth projection
    for the Argus trading bot.

    Usage
    -----
    >>> cfg = CompoundingConfig(initial_capital_aud=1000.0)
    >>> engine = CompoundingEngine(cfg)
    >>> engine.record_daily_pnl(date.today(), pnl_usd=4.0, strategy="mm")
    >>> result = engine.run_weekly_reinvestment()
    >>> print(engine.get_growth_summary())
    """

    def __init__(self, config: CompoundingConfig) -> None:
        self.config = config
        self._initial_capital_usd: float = config.initial_capital_aud * config.aud_usd_rate
        self._total_capital_usd: float = self._initial_capital_usd

        # Daily PnL log: list of (date, pnl_usd, strategy)
        self._pnl_log: List[Dict] = []

        # Reinvestment history
        self._reinvestment_history: List[ReinvestmentResult] = []

        # Totals
        self._total_reinvested_usd: float = 0.0
        self._total_withdrawn_usd: float = 0.0
        self._total_profit_usd: float = 0.0

        # Tracking
        self._last_reinvestment_date: Optional[date] = None
        self._start_date: date = date.today()

        # Current MEXC monthly volume estimate (updated externally or via reinvestment)
        self._estimated_monthly_volume_usd: float = 0.0
        self._current_fee_tier: str = "Lv0"

        logger.info(
            "CompoundingEngine initialised: AUD %.2f → USD %.2f",
            config.initial_capital_aud,
            self._initial_capital_usd,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_daily_pnl(
        self,
        record_date: date,
        pnl_usd: float,
        strategy: str,
    ) -> None:
        """Record the realised PnL for one trading day."""
        self._pnl_log.append(
            {
                "date": record_date,
                "pnl_usd": pnl_usd,
                "strategy": strategy,
            }
        )
        logger.debug(
            "PnL recorded: %s | strategy=%s | pnl_usd=%.4f",
            record_date,
            strategy,
            pnl_usd,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pnl_since_last_reinvestment(self) -> float:
        """Sum PnL entries since the last reinvestment date."""
        cutoff = self._last_reinvestment_date
        total = 0.0
        for entry in self._pnl_log:
            if cutoff is None or entry["date"] > cutoff:
                total += entry["pnl_usd"]
        return total

    def _rolling_7d_avg_pnl(self) -> float:
        """Compute rolling 7-day average daily PnL from recorded history."""
        if not self._pnl_log:
            return 4.0  # Fallback assumption: $4/day
        today = date.today()
        cutoff = today - timedelta(days=7)
        recent = [e["pnl_usd"] for e in self._pnl_log if e["date"] >= cutoff]
        if not recent:
            # Use last 7 entries regardless of date
            recent = [e["pnl_usd"] for e in self._pnl_log[-7:]]
        return sum(recent) / len(recent) if recent else 4.0

    def _fee_tier_for_capital(self, capital_usd: float) -> str:
        """
        Estimate MEXC fee tier based on approximate monthly trading volume.
        Market-making strategies typically turn over ~3× capital per day.
        30-day volume ≈ capital × 3 × 30.
        """
        monthly_vol = capital_usd * 3.0 * 30.0
        return _mexc_tier_for_volume(monthly_vol)

    def _days_running(self) -> int:
        return (date.today() - self._start_date).days + 1

    # ------------------------------------------------------------------
    # Public: capital queries
    # ------------------------------------------------------------------

    def get_current_capital(self) -> float:
        """Total capital in USD (initial + all reinvested profits)."""
        return self._total_capital_usd

    def get_current_capital_aud(self) -> float:
        """Total capital in AUD."""
        return self._total_capital_usd / self.config.aud_usd_rate

    # ------------------------------------------------------------------
    # Allocation resizing
    # ------------------------------------------------------------------

    def resize_allocations(self, new_capital_usd: float) -> Dict[str, float]:
        """
        Compute per-strategy USD allocations from target_allocations fractions.

        Enforces minimum $50 per active strategy. If capital is too small to
        honour all minimums, strategies are ranked by target allocation and the
        lowest-priority ones receive $0.
        """
        config = self.config
        allocations: Dict[str, float] = {}

        # Sort strategies by target allocation descending (highest priority first)
        sorted_strategies = sorted(
            config.target_allocations.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )

        remaining = new_capital_usd
        assigned: Dict[str, float] = {}

        for strategy, fraction in sorted_strategies:
            raw = new_capital_usd * fraction
            # Enforce minimum
            if raw < config.min_strategy_allocation_usd and strategy != "reserve":
                if remaining >= config.min_strategy_allocation_usd:
                    assigned[strategy] = config.min_strategy_allocation_usd
                    remaining -= config.min_strategy_allocation_usd
                else:
                    assigned[strategy] = 0.0
            else:
                assigned[strategy] = raw
                remaining -= raw

        # Reserve gets whatever the fraction says (can be < min)
        if "reserve" in config.target_allocations:
            assigned["reserve"] = new_capital_usd * config.target_allocations["reserve"]

        allocations = assigned
        logger.debug("Allocations at USD %.2f: %s", new_capital_usd, allocations)
        return allocations

    # ------------------------------------------------------------------
    # Weekly reinvestment
    # ------------------------------------------------------------------

    def run_weekly_reinvestment(self) -> ReinvestmentResult:
        """
        Execute the weekly reinvestment cycle.

        1. Computes net PnL since last reinvestment.
        2. Splits PnL into reinvested portion and withdrawn portion.
        3. Adds reinvested amount to total capital.
        4. Resizes per-strategy allocations.
        5. Checks for MEXC fee tier upgrade.
        6. Records and returns a ReinvestmentResult.
        """
        today = date.today()
        period_pnl = self._pnl_since_last_reinvestment()
        self._total_profit_usd += period_pnl

        reinvested_usd: float
        withdrawn_usd: float

        if period_pnl <= 0:
            # No reinvestment when PnL is negative or zero
            reinvested_usd = 0.0
            withdrawn_usd = 0.0
        elif period_pnl < self.config.min_reinvest_amount_usd:
            # Below minimum threshold – carry over to next cycle
            logger.info(
                "Period PnL USD %.4f below min_reinvest_amount_usd %.2f – deferring.",
                period_pnl,
                self.config.min_reinvest_amount_usd,
            )
            reinvested_usd = 0.0
            withdrawn_usd = 0.0
        else:
            reinvested_usd = period_pnl * self.config.reinvest_pct
            withdrawn_usd = period_pnl * (1.0 - self.config.reinvest_pct)

        old_capital = self._total_capital_usd
        self._total_capital_usd += reinvested_usd
        self._total_reinvested_usd += reinvested_usd
        self._total_withdrawn_usd += withdrawn_usd

        new_capital_usd = self._total_capital_usd
        new_capital_aud = self.get_current_capital_aud()
        new_allocations = self.resize_allocations(new_capital_usd)

        # Check fee tier upgrade
        old_tier = self._current_fee_tier
        new_tier = self._fee_tier_for_capital(new_capital_usd)
        tier_upgraded = new_tier != old_tier and _MEXC_FEE_TIERS.index(
            next(t for t in _MEXC_FEE_TIERS if t[1] == new_tier)
        ) > _MEXC_FEE_TIERS.index(
            next(t for t in _MEXC_FEE_TIERS if t[1] == old_tier)
        )
        if tier_upgraded:
            logger.info(
                "MEXC fee tier upgraded: %s → %s at capital USD %.2f",
                old_tier,
                new_tier,
                new_capital_usd,
            )
        self._current_fee_tier = new_tier

        result = ReinvestmentResult(
            date=today,
            pnl_period_usd=period_pnl,
            reinvested_usd=reinvested_usd,
            withdrawn_usd=withdrawn_usd,
            new_capital_usd=new_capital_usd,
            new_capital_aud=new_capital_aud,
            new_allocations=new_allocations,
            fee_tier_upgraded=tier_upgraded,
            new_fee_tier=new_tier,
        )

        self._reinvestment_history.append(result)
        self._last_reinvestment_date = today

        logger.info(
            "Reinvestment: PnL=%.4f reinvested=%.4f withdrawn=%.4f "
            "new_capital_usd=%.2f tier=%s",
            period_pnl,
            reinvested_usd,
            withdrawn_usd,
            new_capital_usd,
            new_tier,
        )
        return result

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def project_growth(
        self,
        days: int = 365,
        daily_return_usd: Optional[float] = None,
    ) -> List[GrowthSnapshot]:
        """
        Project weekly capital growth for `days` days.

        Parameters
        ----------
        days:
            Number of days to project.
        daily_return_usd:
            Fixed daily PnL in USD. If None, the rolling 7-day average is used.

        Returns
        -------
        List of GrowthSnapshot, one per week.
        """
        if daily_return_usd is None:
            daily_return_usd = self._rolling_7d_avg_pnl()
            if daily_return_usd <= 0:
                daily_return_usd = 4.0  # Default assumption

        snapshots: List[GrowthSnapshot] = []
        capital = self._total_capital_usd
        today = date.today()
        num_weeks = max(1, days // 7)

        for week in range(1, num_weeks + 1):
            weekly_pnl = daily_return_usd * 7.0 * self.config.reinvest_pct
            capital += weekly_pnl

            snap_date = today + timedelta(weeks=week)
            capital_aud = capital / self.config.aud_usd_rate
            allocations = self.resize_allocations(capital)
            fee_tier = self._fee_tier_for_capital(capital)

            # Annualised return relative to initial capital
            elapsed_days = week * 7
            if elapsed_days > 0 and self._initial_capital_usd > 0:
                total_return = (capital - self._initial_capital_usd) / self._initial_capital_usd
                ann_return_pct = (
                    ((1.0 + total_return) ** (365.0 / elapsed_days) - 1.0) * 100.0
                )
            else:
                ann_return_pct = 0.0

            snapshots.append(
                GrowthSnapshot(
                    week=week,
                    date=snap_date,
                    capital_usd=round(capital, 4),
                    capital_aud=round(capital_aud, 4),
                    daily_return_usd=round(daily_return_usd, 4),
                    mm_allocation=round(allocations.get("mm", 0.0), 4),
                    funding_allocation=round(allocations.get("funding_arb", 0.0), 4),
                    reserve=round(allocations.get("reserve", 0.0), 4),
                    fee_tier=fee_tier,
                    annualised_return_pct=round(ann_return_pct, 2),
                )
            )

        return snapshots

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_growth_summary(self) -> Dict:
        """
        Return a comprehensive summary dict suitable for dashboards and reports.
        """
        capital_usd = self._total_capital_usd
        capital_aud = self.get_current_capital_aud()
        days = self._days_running()

        # Annualised return
        if days > 1 and self._initial_capital_usd > 0:
            total_return = (capital_usd - self._initial_capital_usd) / self._initial_capital_usd
            ann_return_pct = (
                ((1.0 + total_return) ** (365.0 / days) - 1.0) * 100.0
                if total_return > -1.0
                else -100.0
            )
        else:
            ann_return_pct = 0.0

        # Fee tier info
        current_tier = self._current_fee_tier
        next_tier_info = _next_mexc_tier(current_tier)
        monthly_vol_needed: Optional[float] = None
        capital_needed_usd: Optional[float] = None
        if next_tier_info:
            next_tier_name, next_min_vol = next_tier_info
            current_monthly_vol = capital_usd * 3.0 * 30.0
            monthly_vol_needed = max(0.0, next_min_vol - current_monthly_vol)
            # Capital needed so that capital × 3 × 30 = next_min_vol
            required_capital = next_min_vol / (3.0 * 30.0)
            capital_needed_usd = max(0.0, required_capital - capital_usd)
        else:
            next_tier_name = "MAX"

        # 1-year projection
        proj = self.project_growth(days=365)
        proj_1yr_usd = proj[-1].capital_usd if proj else capital_usd
        proj_1yr_aud = proj[-1].capital_aud if proj else capital_aud

        return {
            "current_capital_usd": round(capital_usd, 4),
            "current_capital_aud": round(capital_aud, 4),
            "total_profit_usd": round(self._total_profit_usd, 4),
            "total_reinvested_usd": round(self._total_reinvested_usd, 4),
            "total_withdrawn_usd": round(self._total_withdrawn_usd, 4),
            "days_running": days,
            "annualised_return_pct": round(ann_return_pct, 2),
            "current_fee_tier": current_tier,
            "next_fee_tier_threshold": next_tier_name,
            "volume_to_next_tier_usd": (
                round(monthly_vol_needed, 2) if monthly_vol_needed is not None else None
            ),
            "capital_needed_for_next_tier_usd": (
                round(capital_needed_usd, 2) if capital_needed_usd is not None else None
            ),
            "projected_1yr_capital_usd": round(proj_1yr_usd, 4),
            "projected_1yr_capital_aud": round(proj_1yr_aud, 4),
        }

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    def generate_growth_report(self) -> str:
        """
        Generate a formatted weekly growth report including projections table.

        Returns a multi-line string suitable for logging or Slack/Telegram alerts.
        """
        summary = self.get_growth_summary()
        lines: List[str] = []

        lines.append("=" * 60)
        lines.append("  ARGUS COMPOUNDING ENGINE — WEEKLY GROWTH REPORT")
        lines.append("=" * 60)
        lines.append(f"  Date            : {date.today()}")
        lines.append(f"  Days Running    : {summary['days_running']}")
        lines.append("")
        lines.append("  CAPITAL")
        lines.append(f"    Current USD   : ${summary['current_capital_usd']:>10,.2f}")
        lines.append(f"    Current AUD   : ${summary['current_capital_aud']:>10,.2f}")
        lines.append(f"    Total Profit  : ${summary['total_profit_usd']:>10,.2f} USD")
        lines.append(
            f"    Reinvested    : ${summary['total_reinvested_usd']:>10,.2f} USD"
        )
        lines.append(
            f"    Withdrawn     : ${summary['total_withdrawn_usd']:>10,.2f} USD"
        )
        lines.append(f"    Ann. Return   : {summary['annualised_return_pct']:>8.1f}%")
        lines.append("")
        lines.append("  FEE TIER (MEXC)")
        lines.append(f"    Current Tier  : {summary['current_fee_tier']}")
        lines.append(f"    Next Tier     : {summary['next_fee_tier_threshold']}")
        if summary["volume_to_next_tier_usd"] is not None:
            lines.append(
                f"    Vol Needed    : ${summary['volume_to_next_tier_usd']:>10,.0f} / month"
            )
            lines.append(
                f"    Capital Needed: ${summary['capital_needed_for_next_tier_usd']:>10,.2f} USD"
            )
        lines.append("")
        lines.append("  1-YEAR PROJECTION")
        lines.append(
            f"    Projected USD : ${summary['projected_1yr_capital_usd']:>10,.2f}"
        )
        lines.append(
            f"    Projected AUD : ${summary['projected_1yr_capital_aud']:>10,.2f}"
        )
        lines.append("")

        # Projection table
        projections = self.project_growth(days=365)
        # Show every 4 weeks (monthly) + final week
        milestones = [s for s in projections if s.week % 4 == 0 or s.week == 1]
        if projections and projections[-1] not in milestones:
            milestones.append(projections[-1])

        header = (
            f"  {'Wk':>3}  {'Date':<12}  {'Capital USD':>12}  "
            f"{'Capital AUD':>12}  {'Daily $':>8}  {'Tier':<5}  {'Ann%':>7}"
        )
        lines.append("  GROWTH TABLE (weekly, monthly highlights)")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for snap in milestones:
            lines.append(
                f"  {snap.week:>3}  {str(snap.date):<12}  "
                f"${snap.capital_usd:>11,.2f}  "
                f"${snap.capital_aud:>11,.2f}  "
                f"${snap.daily_return_usd:>7,.2f}  "
                f"{snap.fee_tier:<5}  "
                f"{snap.annualised_return_pct:>6.1f}%"
            )
        lines.append("=" * 60)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reinvestment history
    # ------------------------------------------------------------------

    def get_reinvestment_history(self) -> List[ReinvestmentResult]:
        """Return list of all past ReinvestmentResult records."""
        return list(self._reinvestment_history)

    def get_pnl_log(self) -> List[Dict]:
        """Return raw daily PnL log."""
        return list(self._pnl_log)

    # ------------------------------------------------------------------
    # State serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialise engine state for persistence."""
        return {
            "initial_capital_usd": self._initial_capital_usd,
            "total_capital_usd": self._total_capital_usd,
            "total_reinvested_usd": self._total_reinvested_usd,
            "total_withdrawn_usd": self._total_withdrawn_usd,
            "total_profit_usd": self._total_profit_usd,
            "last_reinvestment_date": (
                self._last_reinvestment_date.isoformat()
                if self._last_reinvestment_date
                else None
            ),
            "start_date": self._start_date.isoformat(),
            "current_fee_tier": self._current_fee_tier,
            "pnl_log": [
                {
                    "date": e["date"].isoformat(),
                    "pnl_usd": e["pnl_usd"],
                    "strategy": e["strategy"],
                }
                for e in self._pnl_log
            ],
        }

    def from_dict(self, data: Dict) -> None:
        """Restore engine state from a serialised dict."""
        self._initial_capital_usd = data.get("initial_capital_usd", self._initial_capital_usd)
        self._total_capital_usd = data.get("total_capital_usd", self._total_capital_usd)
        self._total_reinvested_usd = data.get("total_reinvested_usd", 0.0)
        self._total_withdrawn_usd = data.get("total_withdrawn_usd", 0.0)
        self._total_profit_usd = data.get("total_profit_usd", 0.0)
        self._current_fee_tier = data.get("current_fee_tier", "Lv0")

        lr = data.get("last_reinvestment_date")
        self._last_reinvestment_date = date.fromisoformat(lr) if lr else None

        sd = data.get("start_date")
        self._start_date = date.fromisoformat(sd) if sd else date.today()

        self._pnl_log = [
            {
                "date": date.fromisoformat(e["date"]),
                "pnl_usd": e["pnl_usd"],
                "strategy": e["strategy"],
            }
            for e in data.get("pnl_log", [])
        ]
