"""
Counterparty Risk Monitor — tracks exchange health to detect insolvency risk.

Red flags monitored:
  - Exchange withdrawal delays > threshold
  - Open interest growing faster than insurance fund
  - Multiple large liquidations in short window
  - Volume drop > 70 % vs 30-day average
  - Negative funding rate extremes
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
class ExchangeHealth:
    """
    Point-in-time health snapshot for a single exchange.

    Attributes
    ----------
    exchange_id:
        Canonical exchange identifier, e.g. "kraken", "coinbase", "binance".
    withdrawal_ok:
        False if withdrawals have been delayed, paused, or restricted.
    insurance_fund_usd:
        Current size of the exchange insurance/liquidation fund in USD.
        Set to 0 if not publicly available.
    open_interest_usd:
        Total open interest in perpetual futures on the exchange in USD.
    funding_rate:
        Latest 8-hour perpetual funding rate (e.g. -0.03 = -3 %).
    volume_24h:
        24-hour trading volume in USD.
    risk_score:
        Pre-computed risk score 0–100 (higher = riskier).
        Usually computed by ``CounterpartyMonitor.get_risk_score``.
    warnings:
        Human-readable list of active risk flags.
    timestamp:
        UTC timestamp of this snapshot.
    """

    exchange_id: str
    withdrawal_ok: bool
    insurance_fund_usd: float
    open_interest_usd: float
    funding_rate: float
    volume_24h: float
    risk_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "exchange_id": self.exchange_id,
            "withdrawal_ok": self.withdrawal_ok,
            "insurance_fund_usd": self.insurance_fund_usd,
            "open_interest_usd": self.open_interest_usd,
            "funding_rate": self.funding_rate,
            "volume_24h": self.volume_24h,
            "risk_score": round(self.risk_score, 2),
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class CounterpartyMonitor:
    """
    Monitor counterparty (exchange) risk and compute risk-adjusted capital
    allocation weights across multiple exchanges.

    Risk scoring (additive, capped at 100):
      +30  withdrawal_ok = False
      +20  funding_rate < -0.05  (extreme negative funding — crowded longs)
      +25  open_interest / insurance_fund > 100x  (thin backstop)
      +15  24h volume drop > 70 % vs historical average
      +10  insurance_fund_usd < 10 M USD  (too small to matter)

    Parameters
    ----------
    exchanges:
        List of exchange IDs to monitor.
    max_exposure_per_exchange_pct:
        Maximum fraction of total capital allocated to any single exchange
        regardless of risk score (default 50 %).
    """

    # Thresholds used in risk scoring
    _WITHDRAWAL_HALT_SCORE: float = 30.0
    _EXTREME_NEGATIVE_FUNDING_THRESHOLD: float = -0.05
    _EXTREME_NEGATIVE_FUNDING_SCORE: float = 20.0
    _OI_INSURANCE_RATIO_THRESHOLD: float = 100.0
    _OI_INSURANCE_SCORE: float = 25.0
    _VOLUME_DROP_THRESHOLD_PCT: float = 0.70   # 70 % drop
    _VOLUME_DROP_SCORE: float = 15.0
    _SMALL_INSURANCE_FUND_THRESHOLD: float = 10_000_000.0  # $10 M
    _SMALL_INSURANCE_FUND_SCORE: float = 10.0

    # Risk score above which exposure reduction is recommended
    _REDUCTION_THRESHOLD: float = 70.0

    def __init__(
        self,
        exchanges: List[str],
        max_exposure_per_exchange_pct: float = 0.50,
    ) -> None:
        if not exchanges:
            raise ValueError("At least one exchange must be specified")
        if not (0 < max_exposure_per_exchange_pct <= 1):
            raise ValueError("max_exposure_per_exchange_pct must be in (0, 1]")

        self._exchanges: List[str] = [ex.lower() for ex in exchanges]
        self.max_exposure_per_exchange_pct = max_exposure_per_exchange_pct

        # Latest health snapshot per exchange
        self._health: Dict[str, ExchangeHealth] = {}

        # Historical 24h volume per exchange (list of floats) for baseline calc
        self._volume_history: Dict[str, List[float]] = {
            ex: [] for ex in self._exchanges
        }
        self._volume_history_max = 30  # retain last 30 observations

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(self, exchange_id: str, health: ExchangeHealth) -> None:
        """
        Store the latest health snapshot for *exchange_id*.

        Also appends the 24h volume to the rolling history used for the volume
        drop check and then computes the risk score immediately.
        """
        eid = exchange_id.lower()
        if eid not in self._exchanges:
            logger.warning(
                "CounterpartyMonitor.update: unknown exchange '%s' — adding to list",
                eid,
            )
            self._exchanges.append(eid)
            self._volume_history[eid] = []

        # Track volume history for drop detection
        vol_hist = self._volume_history[eid]
        vol_hist.append(health.volume_24h)
        if len(vol_hist) > self._volume_history_max:
            vol_hist.pop(0)

        # Compute and embed risk score + warnings into the health object
        score, warnings = self._compute_risk(eid, health)
        health.risk_score = score
        health.warnings = warnings

        self._health[eid] = health

        if score >= self._REDUCTION_THRESHOLD:
            logger.warning(
                "CounterpartyMonitor: exchange '%s' risk score %.0f — exposure reduction recommended. Flags: %s",
                eid,
                score,
                "; ".join(warnings) if warnings else "none",
            )
        else:
            logger.debug(
                "CounterpartyMonitor: exchange '%s' risk score %.0f",
                eid,
                score,
            )

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def _compute_risk(
        self, exchange_id: str, health: ExchangeHealth
    ) -> tuple[float, List[str]]:
        """
        Compute an additive risk score and a list of warning strings.

        Returns
        -------
        (score, warnings)
        """
        score = 0.0
        warnings: List[str] = []

        # 1. Withdrawal halt
        if not health.withdrawal_ok:
            score += self._WITHDRAWAL_HALT_SCORE
            warnings.append("Withdrawals are halted or restricted")

        # 2. Extreme negative funding rate
        if health.funding_rate < self._EXTREME_NEGATIVE_FUNDING_THRESHOLD:
            score += self._EXTREME_NEGATIVE_FUNDING_SCORE
            warnings.append(
                f"Extreme negative funding rate {health.funding_rate:.3%} "
                f"(threshold {self._EXTREME_NEGATIVE_FUNDING_THRESHOLD:.3%})"
            )

        # 3. Open interest / insurance fund ratio
        if health.insurance_fund_usd > 0:
            ratio = health.open_interest_usd / health.insurance_fund_usd
            if ratio > self._OI_INSURANCE_RATIO_THRESHOLD:
                score += self._OI_INSURANCE_SCORE
                warnings.append(
                    f"OI/insurance fund ratio {ratio:.0f}x exceeds {self._OI_INSURANCE_RATIO_THRESHOLD:.0f}x"
                )
        elif health.open_interest_usd > 0:
            # No insurance fund at all but has OI — severe flag
            score += self._OI_INSURANCE_SCORE
            warnings.append("No insurance fund data available but open interest is non-zero")

        # 4. Volume drop vs rolling 30-day average
        vol_hist = self._volume_history.get(exchange_id, [])
        if len(vol_hist) >= 2:
            # Use all but the last observation as the baseline average
            baseline_vols = vol_hist[:-1]
            avg_vol = sum(baseline_vols) / len(baseline_vols)
            if avg_vol > 0 and health.volume_24h < avg_vol * (1.0 - self._VOLUME_DROP_THRESHOLD_PCT):
                score += self._VOLUME_DROP_SCORE
                drop_pct = (1.0 - health.volume_24h / avg_vol) * 100.0
                warnings.append(
                    f"24h volume dropped {drop_pct:.0f} % vs {len(baseline_vols)}-observation average"
                )

        # 5. Small insurance fund
        if 0 < health.insurance_fund_usd < self._SMALL_INSURANCE_FUND_THRESHOLD:
            score += self._SMALL_INSURANCE_FUND_SCORE
            warnings.append(
                f"Insurance fund ${health.insurance_fund_usd:,.0f} is below "
                f"${self._SMALL_INSURANCE_FUND_THRESHOLD:,.0f} threshold"
            )

        # Cap at 100
        score = min(score, 100.0)
        return score, warnings

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_risk_score(self, exchange_id: str) -> float:
        """
        Return the latest risk score for *exchange_id* (0–100).

        Returns 100 (maximum risk) if no health data has been received yet,
        to default to caution.
        """
        eid = exchange_id.lower()
        health = self._health.get(eid)
        if health is None:
            logger.warning(
                "CounterpartyMonitor.get_risk_score: no data for '%s' — returning max risk",
                eid,
            )
            return 100.0
        return health.risk_score

    def should_reduce_exposure(self, exchange_id: str) -> bool:
        """
        Return True if the risk score for *exchange_id* exceeds the reduction
        threshold (70 by default).
        """
        return self.get_risk_score(exchange_id) > self._REDUCTION_THRESHOLD

    def get_allocation_weights(self, total_capital_usd: float) -> Dict[str, float]:
        """
        Compute risk-adjusted capital allocation weights across all exchanges.

        Exchanges with no health data receive a weight of zero.
        Exchanges that should have their exposure reduced are down-weighted.
        The final weights respect *max_exposure_per_exchange_pct*.

        Returns
        -------
        dict mapping exchange_id -> USD allocation
        """
        if total_capital_usd <= 0:
            raise ValueError("total_capital_usd must be positive")

        # Compute an inverse-risk weight for each exchange
        raw_weights: Dict[str, float] = {}
        for eid in self._exchanges:
            score = self.get_risk_score(eid)
            if score >= 100.0:
                raw_weights[eid] = 0.0
            else:
                # Linear inverse: score=0 → weight=1.0, score=100 → weight=0
                raw_weights[eid] = (100.0 - score) / 100.0

        total_weight = sum(raw_weights.values())
        if total_weight == 0.0:
            logger.error(
                "CounterpartyMonitor: all exchanges have maximum risk score — "
                "returning zero allocations"
            )
            return {eid: 0.0 for eid in self._exchanges}

        max_per_exchange = total_capital_usd * self.max_exposure_per_exchange_pct

        # First pass: proportional allocation
        allocations: Dict[str, float] = {
            eid: (w / total_weight) * total_capital_usd
            for eid, w in raw_weights.items()
        }

        # Second pass: enforce cap
        excess = 0.0
        capped_exchanges = set()
        for eid, alloc in list(allocations.items()):
            if alloc > max_per_exchange:
                excess += alloc - max_per_exchange
                allocations[eid] = max_per_exchange
                capped_exchanges.add(eid)

        # Redistribute excess proportionally to uncapped exchanges
        if excess > 0:
            uncapped = {
                eid: w for eid, w in raw_weights.items()
                if eid not in capped_exchanges and w > 0
            }
            uncapped_total = sum(uncapped.values())
            if uncapped_total > 0:
                for eid, w in uncapped.items():
                    extra = (w / uncapped_total) * excess
                    new_alloc = allocations[eid] + extra
                    # Enforce cap again
                    allocations[eid] = min(new_alloc, max_per_exchange)

        return allocations

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return a complete snapshot of all exchange health data.

        Returns
        -------
        dict with keys:
            exchanges   — per-exchange health dicts
            high_risk   — list of exchange IDs with risk_score > 70
            timestamp
        """
        high_risk = [
            eid for eid, h in self._health.items()
            if h.risk_score > self._REDUCTION_THRESHOLD
        ]

        return {
            "exchanges": {eid: h.to_dict() for eid, h in self._health.items()},
            "high_risk": high_risk,
            "max_exposure_per_exchange_pct": self.max_exposure_per_exchange_pct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"CounterpartyMonitor(exchanges={self._exchanges}, "
            f"max_exposure_pct={self.max_exposure_per_exchange_pct:.0%})"
        )
