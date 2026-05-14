"""
Portfolio Stress Tester — applies historical crash scenarios to current positions.

Scenarios included:
  - COVID crash (Feb-Mar 2020): BTC -63%, ETH -75%
  - FTX collapse (Nov 2022): BTC -25%, ETH -30%
  - LUNA collapse (May 2022): BTC -40%, ETH -45%
  - 2018 bear market: BTC -84%, ETH -94%
  - Flash crash (Mar 2020 single day): BTC -50%

Each scenario returns estimated P&L impact on current portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StressScenario:
    """Definition of a historical stress scenario."""

    name: str
    description: str
    asset_shocks: Dict[str, float]   # {"BTC": -0.63, "ETH": -0.75} etc.
    duration_days: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "asset_shocks": self.asset_shocks,
            "duration_days": self.duration_days,
        }


@dataclass
class StressResult:
    """Outcome of applying a single scenario to current positions."""

    scenario_name: str
    pnl_usd: float
    pnl_pct: float
    worst_asset: str
    worst_asset_loss_usd: float
    survived: bool  # True if pnl > -capital * 0.5
    applied_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "pnl_usd": round(self.pnl_usd, 4),
            "pnl_pct": round(self.pnl_pct * 100.0, 2),
            "worst_asset": self.worst_asset,
            "worst_asset_loss_usd": round(self.worst_asset_loss_usd, 4),
            "survived": self.survived,
            "applied_at": self.applied_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Symbol normalisation helpers
# ---------------------------------------------------------------------------

_SYMBOL_MAP: Dict[str, str] = {
    # BTC variants
    "BTC/USD": "BTC",
    "BTC/USDT": "BTC",
    "BTC/AUD": "BTC",
    "XBTUSD": "BTC",
    "XBT/USD": "BTC",
    "BTC-USD": "BTC",
    "BTC-PERP": "BTC",
    "BTCUSDT": "BTC",
    # ETH variants
    "ETH/USD": "ETH",
    "ETH/USDT": "ETH",
    "ETH/AUD": "ETH",
    "ETH-USD": "ETH",
    "ETH-PERP": "ETH",
    "ETHUSDT": "ETH",
    # SOL variants
    "SOL/USD": "SOL",
    "SOL/USDT": "SOL",
    "SOL-USD": "SOL",
    "SOL-PERP": "SOL",
    "SOLUSDT": "SOL",
    # Others
    "AVAX/USD": "AVAX",
    "AVAX/USDT": "AVAX",
    "MATIC/USD": "MATIC",
    "MATIC/USDT": "MATIC",
    "DOT/USD": "DOT",
    "DOT/USDT": "DOT",
    "LINK/USD": "LINK",
    "LINK/USDT": "LINK",
    "ADA/USD": "ADA",
    "ADA/USDT": "ADA",
    "XRP/USD": "XRP",
    "XRP/USDT": "XRP",
}


def _normalise_symbol(symbol: str) -> str:
    """
    Map a trading symbol to its canonical base-asset ticker.

    Falls back to uppercased input split on '/' if not found in the map.
    """
    if symbol in _SYMBOL_MAP:
        return _SYMBOL_MAP[symbol]
    upper = symbol.upper()
    if upper in _SYMBOL_MAP:
        return _SYMBOL_MAP[upper]
    # Generic: take everything before the first '/', '-', or '_'
    for sep in ("/", "-", "_"):
        if sep in upper:
            return upper.split(sep)[0]
    return upper


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PortfolioStressTester:
    """
    Apply historical crash scenarios to a snapshot of current positions.

    Parameters
    ----------
    capital_usd:
        Total portfolio capital in USD. Used to compute P&L percentages and
        to determine whether a scenario result counts as "survived".
    stop_loss_pct:
        The fraction of capital at which the system would have been forced out
        of positions. Scenarios that exceed this loss are flagged as not
        survived regardless of the *survived* threshold.
    """

    SCENARIOS: List[StressScenario] = [
        StressScenario(
            name="COVID Crash 2020",
            description=(
                "Feb–Mar 2020 pandemic panic. BTC fell from ~$10 500 to ~$3 850 "
                "(-63 %). ETH fell from ~$280 to ~$70 (-75 %)."
            ),
            asset_shocks={
                "BTC": -0.63,
                "ETH": -0.75,
                "SOL": -0.70,  # SOL existed but illiquid; proxy used
                "AVAX": -0.72,
                "LINK": -0.78,
                "DOT": -0.75,
                "ADA": -0.74,
                "XRP": -0.65,
                "MATIC": -0.72,
            },
            duration_days=32,
        ),
        StressScenario(
            name="FTX Collapse Nov 2022",
            description=(
                "Nov 2022 FTX exchange bankruptcy. BTC fell ~25 %, ETH ~30 % "
                "in the week following the insolvency revelation."
            ),
            asset_shocks={
                "BTC": -0.25,
                "ETH": -0.30,
                "SOL": -0.60,  # SOL uniquely impacted (FTX held large SOL)
                "AVAX": -0.35,
                "LINK": -0.32,
                "DOT": -0.33,
                "ADA": -0.30,
                "XRP": -0.20,
                "MATIC": -0.35,
            },
            duration_days=7,
        ),
        StressScenario(
            name="LUNA/UST Collapse May 2022",
            description=(
                "May 2022 LUNA/UST de-pegging spiral. BTC fell ~40 %, "
                "ETH ~45 %. Altcoins hit harder."
            ),
            asset_shocks={
                "BTC": -0.40,
                "ETH": -0.45,
                "SOL": -0.55,
                "AVAX": -0.58,
                "LINK": -0.50,
                "DOT": -0.52,
                "ADA": -0.60,
                "XRP": -0.40,
                "MATIC": -0.55,
            },
            duration_days=10,
        ),
        StressScenario(
            name="2018 Crypto Bear Market",
            description=(
                "Jan 2018 – Dec 2018 extended bear market. BTC peak-to-trough "
                "-84 %, ETH -94 %. Most altcoins lost 90–98 %."
            ),
            asset_shocks={
                "BTC": -0.84,
                "ETH": -0.94,
                "SOL": -0.90,  # SOL post-2020 proxy
                "AVAX": -0.92,
                "LINK": -0.88,
                "DOT": -0.90,
                "ADA": -0.96,
                "XRP": -0.95,
                "MATIC": -0.96,
            },
            duration_days=350,
        ),
        StressScenario(
            name="Flash Crash Mar 2020",
            description=(
                "12 Mar 2020 single-day flash crash. BTC lost ~50 % intraday "
                "as leveraged longs were liquidated in cascade."
            ),
            asset_shocks={
                "BTC": -0.50,
                "ETH": -0.55,
                "SOL": -0.50,
                "AVAX": -0.50,
                "LINK": -0.52,
                "DOT": -0.50,
                "ADA": -0.50,
                "XRP": -0.45,
                "MATIC": -0.50,
            },
            duration_days=1,
        ),
        StressScenario(
            name="Binance Hack May 2019",
            description=(
                "7 May 2019 Binance security breach. BTC fell ~15 % in days "
                "following the $40 M hack announcement."
            ),
            asset_shocks={
                "BTC": -0.15,
                "ETH": -0.18,
                "SOL": -0.16,
                "AVAX": -0.17,
                "LINK": -0.20,
                "DOT": -0.18,
                "ADA": -0.20,
                "XRP": -0.15,
                "MATIC": -0.18,
            },
            duration_days=5,
        ),
    ]

    def __init__(self, capital_usd: float, stop_loss_pct: float = 0.15) -> None:
        if capital_usd <= 0:
            raise ValueError("capital_usd must be positive")
        if not (0 < stop_loss_pct < 1):
            raise ValueError("stop_loss_pct must be between 0 and 1")

        self.capital_usd = capital_usd
        self.stop_loss_pct = stop_loss_pct
        self._stop_loss_usd = capital_usd * stop_loss_pct

    # ------------------------------------------------------------------
    # Core scenario runner
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        scenario: StressScenario,
        positions: Dict[str, float],
    ) -> StressResult:
        """
        Apply *scenario* shocks to *positions* and return the P&L result.

        Parameters
        ----------
        scenario:
            The stress scenario to apply.
        positions:
            Mapping of {trading_symbol: qty_usd}.  qty_usd is the signed USD
            notional (positive = long, negative = short).
        """
        total_pnl = 0.0
        worst_asset: str = ""
        worst_loss_usd: float = 0.0

        for symbol, qty_usd in positions.items():
            base = _normalise_symbol(symbol)
            shock = scenario.asset_shocks.get(base, 0.0)
            if shock == 0.0 and base not in scenario.asset_shocks:
                # Unknown asset — apply median shock as proxy
                if scenario.asset_shocks:
                    shocks = list(scenario.asset_shocks.values())
                    shocks.sort()
                    mid = len(shocks) // 2
                    shock = shocks[mid]
                    logger.debug(
                        "StressTester: no shock for %s (%s) in scenario '%s' — using proxy %.2f",
                        symbol,
                        base,
                        scenario.name,
                        shock,
                    )

            # P&L = position size * shock (negative shock = loss for long)
            pnl = qty_usd * shock
            total_pnl += pnl

            if pnl < worst_loss_usd:
                worst_loss_usd = pnl
                worst_asset = symbol

        pnl_pct = total_pnl / self.capital_usd if self.capital_usd else 0.0

        # Survived if loss does not exceed 50 % of capital
        survived = total_pnl > -(self.capital_usd * 0.50)

        if not survived:
            logger.warning(
                "StressTester scenario '%s': portfolio would NOT survive — "
                "loss $%.2f (%.1f %% of capital)",
                scenario.name,
                total_pnl,
                pnl_pct * 100.0,
            )

        return StressResult(
            scenario_name=scenario.name,
            pnl_usd=total_pnl,
            pnl_pct=pnl_pct,
            worst_asset=worst_asset or "(none)",
            worst_asset_loss_usd=worst_loss_usd,
            survived=survived,
        )

    # ------------------------------------------------------------------
    # Batch runners
    # ------------------------------------------------------------------

    def run_all(self, positions: Dict[str, float]) -> List[StressResult]:
        """
        Run every scenario against *positions* and return all results.

        Results are sorted from largest loss to smallest.
        """
        results = [self.run_scenario(s, positions) for s in self.SCENARIOS]
        results.sort(key=lambda r: r.pnl_usd)
        return results

    def worst_case(self, positions: Dict[str, float]) -> StressResult:
        """Return the single worst-case scenario result."""
        results = self.run_all(positions)
        if not results:
            raise RuntimeError("No scenarios defined — cannot compute worst case")
        return results[0]  # sorted ascending (most negative first)

    def summary(self, positions: Dict[str, float]) -> dict:
        """
        Run all scenarios and return an aggregate summary dict.

        Returns
        -------
        dict with keys:
            results          — list of per-scenario result dicts
            avg_loss_usd     — average P&L across all scenarios
            max_loss_usd     — worst single-scenario P&L
            scenarios_survived  — count of scenarios where survived=True
            total_scenarios
            capital_usd
            timestamp
        """
        results = self.run_all(positions)
        if not results:
            return {
                "results": [],
                "avg_loss_usd": 0.0,
                "max_loss_usd": 0.0,
                "scenarios_survived": 0,
                "total_scenarios": 0,
                "capital_usd": self.capital_usd,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        pnl_values = [r.pnl_usd for r in results]
        avg_loss = sum(pnl_values) / len(pnl_values)
        max_loss = min(pnl_values)  # most negative
        survived_count = sum(1 for r in results if r.survived)

        return {
            "results": [r.to_dict() for r in results],
            "avg_loss_usd": round(avg_loss, 4),
            "max_loss_usd": round(max_loss, 4),
            "scenarios_survived": survived_count,
            "total_scenarios": len(results),
            "capital_usd": self.capital_usd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def add_scenario(self, scenario: StressScenario) -> None:
        """Append a custom scenario to the list at runtime."""
        self.SCENARIOS = list(self.SCENARIOS)  # ensure it is not a shared class list
        self.SCENARIOS.append(scenario)
        logger.info("StressTester: added custom scenario '%s'", scenario.name)

    def __repr__(self) -> str:
        return (
            f"PortfolioStressTester(capital=${self.capital_usd:.0f}, "
            f"stop_loss={self.stop_loss_pct:.1%}, "
            f"scenarios={len(self.SCENARIOS)})"
        )
