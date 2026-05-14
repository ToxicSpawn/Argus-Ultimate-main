"""
capital_growth_tracker.py
=========================
Tracks and analyses capital growth, P&L history, risk-adjusted performance,
and compounding milestones for the Argus trading bot.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Milestone:
    """Capital growth milestone."""

    name: str
    target_aud: float
    achieved: bool = False
    achieved_date: Optional[str] = None      # ISO date string
    estimated_days: Optional[float] = None   # Days from start to achieve at current rate


@dataclass
class SessionRecord:
    """Single trading session record."""

    date: date
    pnl_usd: float
    strategy: str
    exchange: str
    num_fills: int
    adverse_fills: int


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CapitalGrowthTracker:
    """
    Records trading sessions and exposes capital growth analytics:
    - Equity curve
    - Sharpe / Sortino ratios
    - Maximum drawdown
    - Milestone progress
    - Best strategy and exchange rankings
    - Serialisation to/from JSON
    """

    _DEFAULT_MILESTONES: List[Tuple[str, float]] = [
        ("50% gain", 1_500.0),
        ("Doubled", 2_000.0),
        ("5×", 5_000.0),
        ("10×", 10_000.0),
    ]

    def __init__(
        self,
        initial_capital_aud: float = 1_000.0,
        aud_usd_rate: float = 0.62,
    ) -> None:
        self._initial_capital_aud = initial_capital_aud
        self._aud_usd_rate = aud_usd_rate
        self._initial_capital_usd = initial_capital_aud * aud_usd_rate
        self._start_date: date = date.today()

        self._sessions: List[SessionRecord] = []

        self._milestones: List[Milestone] = [
            Milestone(name=name, target_aud=target)
            for name, target in self._DEFAULT_MILESTONES
        ]

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_session(
        self,
        session_date: date,
        pnl_usd: float,
        strategy: str,
        exchange: str,
        num_fills: int,
        adverse_fills: int,
    ) -> None:
        """Record the result of one trading session."""
        session = SessionRecord(
            date=session_date,
            pnl_usd=pnl_usd,
            strategy=strategy,
            exchange=exchange,
            num_fills=num_fills,
            adverse_fills=adverse_fills,
        )
        self._sessions.append(session)
        self._check_milestones()
        logger.debug(
            "Session recorded: %s | %s | %s | pnl=%.4f",
            session_date,
            strategy,
            exchange,
            pnl_usd,
        )

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """
        Return daily equity snapshots sorted by date.

        Each snapshot: {date, capital_usd, capital_aud, daily_pnl, cumulative_pnl}
        """
        if not self._sessions:
            return []

        # Aggregate PnL per date
        pnl_by_date: Dict[date, float] = {}
        for session in self._sessions:
            pnl_by_date[session.date] = pnl_by_date.get(session.date, 0.0) + session.pnl_usd

        sorted_dates = sorted(pnl_by_date.keys())
        curve: List[Dict[str, Any]] = []
        cumulative = 0.0

        for d in sorted_dates:
            daily = pnl_by_date[d]
            cumulative += daily
            capital_usd = self._initial_capital_usd + cumulative
            capital_aud = capital_usd / self._aud_usd_rate
            curve.append(
                {
                    "date": d.isoformat(),
                    "capital_usd": round(capital_usd, 4),
                    "capital_aud": round(capital_aud, 4),
                    "daily_pnl": round(daily, 4),
                    "cumulative_pnl": round(cumulative, 4),
                }
            )

        return curve

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    def _daily_returns(self) -> List[float]:
        """Aggregate daily PnL as a return (pnl / capital at start of day)."""
        curve = self.get_equity_curve()
        if len(curve) < 2:
            return []
        returns = []
        for i in range(1, len(curve)):
            prev_cap = curve[i - 1]["capital_usd"]
            if prev_cap > 0:
                returns.append(curve[i]["daily_pnl"] / prev_cap)
            else:
                returns.append(0.0)
        return returns

    def _mean_and_std(self, values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 0.0
        return mean, std

    def get_rolling_sharpe(self, window_days: int = 30) -> float:
        """
        Compute rolling Sharpe ratio over the last `window_days` days.
        Assumes risk-free rate ≈ 0 (crypto/intraday context).
        Annualised using sqrt(252).
        """
        returns = self._get_recent_daily_returns(window_days)
        if len(returns) < 2:
            return 0.0
        mean, std = self._mean_and_std(returns)
        if std == 0.0:
            return 0.0
        return (mean / std) * math.sqrt(252)

    def get_rolling_sortino(self, window_days: int = 30) -> float:
        """
        Compute rolling Sortino ratio using downside deviation only.
        Annualised using sqrt(252).
        """
        returns = self._get_recent_daily_returns(window_days)
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0.0]
        if not downside:
            # No negative days: return a very high Sortino (bounded)
            return 999.0
        downside_variance = sum(r ** 2 for r in downside) / len(downside)
        downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0.0
        if downside_dev == 0.0:
            return 0.0
        return (mean / downside_dev) * math.sqrt(252)

    def _get_recent_daily_returns(self, window_days: int) -> List[float]:
        """Returns the last `window_days` daily return values."""
        all_returns = self._daily_returns()
        return all_returns[-window_days:] if len(all_returns) >= window_days else all_returns

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def get_max_drawdown(self) -> Tuple[float, str, str]:
        """
        Compute maximum drawdown.

        Returns
        -------
        (max_drawdown_pct, peak_date_str, trough_date_str)
        max_drawdown_pct is expressed as a positive percentage (e.g. 5.0 = 5%).
        """
        curve = self.get_equity_curve()
        if len(curve) < 2:
            return 0.0, "", ""

        max_dd = 0.0
        peak_cap = curve[0]["capital_usd"]
        peak_date = curve[0]["date"]
        trough_date = curve[0]["date"]
        best_peak_date = peak_date
        best_trough_date = trough_date

        for snap in curve[1:]:
            cap = snap["capital_usd"]
            if cap > peak_cap:
                peak_cap = cap
                peak_date = snap["date"]
            else:
                dd = (peak_cap - cap) / peak_cap * 100.0 if peak_cap > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
                    best_peak_date = peak_date
                    best_trough_date = snap["date"]

        return round(max_dd, 4), best_peak_date, best_trough_date

    # ------------------------------------------------------------------
    # Milestones
    # ------------------------------------------------------------------

    def _current_capital_aud(self) -> float:
        curve = self.get_equity_curve()
        if not curve:
            return self._initial_capital_aud
        return curve[-1]["capital_aud"]

    def _check_milestones(self) -> None:
        """Mark milestones as achieved if current capital has passed them."""
        current_aud = self._current_capital_aud()
        for milestone in self._milestones:
            if not milestone.achieved and current_aud >= milestone.target_aud:
                milestone.achieved = True
                milestone.achieved_date = date.today().isoformat()
                logger.info(
                    "Milestone achieved: %s (%.0f AUD)", milestone.name, milestone.target_aud
                )

    def _daily_avg_pnl_usd(self) -> float:
        """Rolling average daily PnL in USD from all sessions."""
        curve = self.get_equity_curve()
        if not curve:
            return 4.0  # default assumption
        total_pnl = sum(s["daily_pnl"] for s in curve)
        return total_pnl / len(curve) if curve else 4.0

    def get_milestones(self) -> List[Milestone]:
        """
        Return milestones with estimated days-to-achieve at current rate.
        """
        self._check_milestones()
        current_aud = self._current_capital_aud()
        daily_pnl_usd = self._daily_avg_pnl_usd()
        daily_pnl_aud = daily_pnl_usd / self._aud_usd_rate

        days_elapsed = (date.today() - self._start_date).days + 1

        for milestone in self._milestones:
            if not milestone.achieved:
                gap_aud = milestone.target_aud - current_aud
                if daily_pnl_aud > 0:
                    milestone.estimated_days = round(gap_aud / daily_pnl_aud, 1)
                else:
                    milestone.estimated_days = None
            else:
                milestone.estimated_days = None

        return list(self._milestones)

    # ------------------------------------------------------------------
    # Strategy & exchange analytics
    # ------------------------------------------------------------------

    def get_best_strategy(self) -> Dict[str, Any]:
        """
        Return best strategy rankings by total PnL, by Sharpe proxy, and by fill count.
        """
        if not self._sessions:
            return {"by_pnl": None, "by_sharpe_proxy": None, "by_fill_count": None}

        # Aggregate per strategy
        strat_pnl: Dict[str, float] = {}
        strat_fills: Dict[str, int] = {}
        strat_returns: Dict[str, List[float]] = {}

        for s in self._sessions:
            strat_pnl[s.strategy] = strat_pnl.get(s.strategy, 0.0) + s.pnl_usd
            strat_fills[s.strategy] = strat_fills.get(s.strategy, 0) + s.num_fills
            if s.strategy not in strat_returns:
                strat_returns[s.strategy] = []
            strat_returns[s.strategy].append(s.pnl_usd)

        best_pnl = max(strat_pnl, key=strat_pnl.__getitem__)
        best_fills = max(strat_fills, key=strat_fills.__getitem__)

        # Sharpe proxy: mean/std of session PnL
        sharpe_scores: Dict[str, float] = {}
        for strat, returns in strat_returns.items():
            mean, std = self._mean_and_std(returns)
            sharpe_scores[strat] = mean / std if std > 0 else (mean if mean > 0 else 0.0)
        best_sharpe = max(sharpe_scores, key=sharpe_scores.__getitem__)

        return {
            "by_pnl": {
                "strategy": best_pnl,
                "total_pnl_usd": round(strat_pnl[best_pnl], 4),
                "all_strategies": {k: round(v, 4) for k, v in strat_pnl.items()},
            },
            "by_sharpe_proxy": {
                "strategy": best_sharpe,
                "sharpe_score": round(sharpe_scores[best_sharpe], 4),
            },
            "by_fill_count": {
                "strategy": best_fills,
                "total_fills": strat_fills[best_fills],
                "all_strategies": strat_fills,
            },
        }

    def get_best_exchange(self) -> Dict[str, Any]:
        """
        Return best exchange by total PnL and by adverse fill rate.
        """
        if not self._sessions:
            return {"by_pnl": None, "by_adverse_fill_rate": None}

        exch_pnl: Dict[str, float] = {}
        exch_fills: Dict[str, int] = {}
        exch_adverse: Dict[str, int] = {}

        for s in self._sessions:
            exch_pnl[s.exchange] = exch_pnl.get(s.exchange, 0.0) + s.pnl_usd
            exch_fills[s.exchange] = exch_fills.get(s.exchange, 0) + s.num_fills
            exch_adverse[s.exchange] = exch_adverse.get(s.exchange, 0) + s.adverse_fills

        best_pnl_exch = max(exch_pnl, key=exch_pnl.__getitem__)

        # Adverse rate = adverse_fills / total_fills (lower is better)
        adverse_rates: Dict[str, float] = {
            ex: (exch_adverse[ex] / exch_fills[ex] if exch_fills[ex] > 0 else 0.0)
            for ex in exch_pnl
        }
        best_adverse_exch = min(adverse_rates, key=adverse_rates.__getitem__)

        return {
            "by_pnl": {
                "exchange": best_pnl_exch,
                "total_pnl_usd": round(exch_pnl[best_pnl_exch], 4),
                "all_exchanges": {k: round(v, 4) for k, v in exch_pnl.items()},
            },
            "by_adverse_fill_rate": {
                "exchange": best_adverse_exch,
                "adverse_rate": round(adverse_rates[best_adverse_exch], 4),
                "all_exchanges": {k: round(v, 4) for k, v in adverse_rates.items()},
            },
        }

    # ------------------------------------------------------------------
    # Weekly report
    # ------------------------------------------------------------------

    def generate_weekly_report(self) -> str:
        """
        Generate a human-readable weekly summary report.
        """
        curve = self.get_equity_curve()
        sessions = self._sessions

        # --- Last 7 days ---
        cutoff = date.today() - timedelta(days=6)
        recent_sessions = [s for s in sessions if s.date >= cutoff]
        week_pnl = sum(s.pnl_usd for s in recent_sessions)
        week_fills = sum(s.num_fills for s in recent_sessions)
        week_adverse = sum(s.adverse_fills for s in recent_sessions)
        week_adverse_rate = (
            week_adverse / week_fills * 100.0 if week_fills > 0 else 0.0
        )

        # Best pair / exchange this week (by PnL per strategy / exchange)
        week_strat_pnl: Dict[str, float] = {}
        week_exch_pnl: Dict[str, float] = {}
        for s in recent_sessions:
            week_strat_pnl[s.strategy] = week_strat_pnl.get(s.strategy, 0.0) + s.pnl_usd
            week_exch_pnl[s.exchange] = week_exch_pnl.get(s.exchange, 0.0) + s.pnl_usd
        best_week_strat = max(week_strat_pnl, key=week_strat_pnl.__getitem__) if week_strat_pnl else "N/A"
        best_week_exch = max(week_exch_pnl, key=week_exch_pnl.__getitem__) if week_exch_pnl else "N/A"

        # --- Running totals ---
        total_profit = sum(e["daily_pnl"] for e in curve) if curve else 0.0
        days_running = (date.today() - self._start_date).days + 1
        current_aud = self._current_capital_aud()
        current_usd = current_aud * self._aud_usd_rate if curve else self._initial_capital_usd

        if days_running > 1 and self._initial_capital_usd > 0:
            total_ret = (current_usd - self._initial_capital_usd) / self._initial_capital_usd
            ann_pct = ((1.0 + total_ret) ** (365.0 / days_running) - 1.0) * 100.0 if total_ret > -1.0 else -100.0
        else:
            ann_pct = 0.0

        sharpe = self.get_rolling_sharpe(30)
        sortino = self.get_rolling_sortino(30)
        max_dd, dd_start, dd_end = self.get_max_drawdown()

        # --- Milestone progress ---
        milestones = self.get_milestones()
        next_m = next((m for m in milestones if not m.achieved), None)

        lines: List[str] = []
        lines.append("=" * 56)
        lines.append("  ARGUS CAPITAL GROWTH TRACKER — WEEKLY REPORT")
        lines.append("=" * 56)
        lines.append(f"  Report Date : {date.today()}")
        lines.append(f"  Days Running: {days_running}")
        lines.append("")
        lines.append("  WEEK SUMMARY (last 7 days)")
        lines.append(f"    PnL         : ${week_pnl:>8,.4f} USD")
        lines.append(f"    Fills       : {week_fills}")
        lines.append(f"    Adverse Rate: {week_adverse_rate:.1f}%")
        lines.append(f"    Best Strat  : {best_week_strat}")
        lines.append(f"    Best Exch   : {best_week_exch}")
        lines.append("")
        lines.append("  RUNNING TOTALS")
        lines.append(f"    Capital AUD : ${current_aud:>10,.2f}")
        lines.append(f"    Capital USD : ${current_usd:>10,.2f}")
        lines.append(f"    Total Profit: ${total_profit:>10,.4f} USD")
        lines.append(f"    Ann. Return : {ann_pct:>7.1f}%")
        lines.append(f"    Sharpe(30d) : {sharpe:>7.2f}")
        lines.append(f"    Sortino(30d): {sortino:>7.2f}")
        lines.append(f"    Max Drawdown: {max_dd:.2f}%  ({dd_start} → {dd_end})")
        lines.append("")
        lines.append("  MILESTONES")
        for m in milestones:
            if m.achieved:
                lines.append(f"    ✓ {m.name:<12} AUD {m.target_aud:>8,.0f}  (achieved {m.achieved_date})")
            else:
                gap = m.target_aud - current_aud
                eta = f"~{m.estimated_days:.0f}d" if m.estimated_days else "TBD"
                lines.append(
                    f"    ○ {m.name:<12} AUD {m.target_aud:>8,.0f}  gap=${gap:,.0f}  ETA={eta}"
                )

        if next_m:
            lines.append("")
            lines.append(f"  NEXT MILESTONE: {next_m.name} (AUD {next_m.target_aud:,.0f})")
            if next_m.estimated_days:
                eta_date = date.today() + timedelta(days=int(next_m.estimated_days))
                lines.append(f"    Estimated date: {eta_date}")
        lines.append("=" * 56)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "data/growth_tracker.json") -> None:
        """Persist tracker state to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data: Dict[str, Any] = {
            "initial_capital_aud": self._initial_capital_aud,
            "aud_usd_rate": self._aud_usd_rate,
            "start_date": self._start_date.isoformat(),
            "sessions": [
                {
                    "date": s.date.isoformat(),
                    "pnl_usd": s.pnl_usd,
                    "strategy": s.strategy,
                    "exchange": s.exchange,
                    "num_fills": s.num_fills,
                    "adverse_fills": s.adverse_fills,
                }
                for s in self._sessions
            ],
            "milestones": [
                {
                    "name": m.name,
                    "target_aud": m.target_aud,
                    "achieved": m.achieved,
                    "achieved_date": m.achieved_date,
                    "estimated_days": m.estimated_days,
                }
                for m in self._milestones
            ],
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.info("CapitalGrowthTracker saved to %s", path)

    def load(self, path: str = "data/growth_tracker.json") -> None:
        """Restore tracker state from a JSON file."""
        with open(path) as fh:
            data = json.load(fh)

        self._initial_capital_aud = data.get("initial_capital_aud", self._initial_capital_aud)
        self._aud_usd_rate = data.get("aud_usd_rate", self._aud_usd_rate)
        self._initial_capital_usd = self._initial_capital_aud * self._aud_usd_rate
        sd = data.get("start_date")
        self._start_date = date.fromisoformat(sd) if sd else date.today()

        self._sessions = [
            SessionRecord(
                date=date.fromisoformat(s["date"]),
                pnl_usd=s["pnl_usd"],
                strategy=s["strategy"],
                exchange=s["exchange"],
                num_fills=s["num_fills"],
                adverse_fills=s["adverse_fills"],
            )
            for s in data.get("sessions", [])
        ]

        self._milestones = [
            Milestone(
                name=m["name"],
                target_aud=m["target_aud"],
                achieved=m["achieved"],
                achieved_date=m.get("achieved_date"),
                estimated_days=m.get("estimated_days"),
            )
            for m in data.get("milestones", [])
        ]

        logger.info("CapitalGrowthTracker loaded from %s", path)
