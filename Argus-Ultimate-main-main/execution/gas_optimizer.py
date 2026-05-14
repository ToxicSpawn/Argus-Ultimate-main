"""
Gas Fee Optimizer — optimise gas fees for DEX transactions.

Tracks recent gas prices, predicts near-term gas costs, and recommends
optimal priority fees based on urgency level.

Gas on L2s (Arbitrum, Base) is typically 10-100x cheaper than Ethereum
mainnet, making DEX HFT viable with sub-$1 transaction costs.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum gas price history entries
_MAX_HISTORY = 1000

# Default gas price if no history
_DEFAULT_BASE_FEE_GWEI = 0.1  # typical Arbitrum base fee
_DEFAULT_PRIORITY_FEE_GWEI = 0.01


@dataclass
class GasPriceEntry:
    """A single gas price observation."""

    base_fee: float       # gwei
    priority_fee: float   # gwei
    timestamp: float       # time.time()

    @property
    def total_fee(self) -> float:
        return self.base_fee + self.priority_fee


class GasOptimizer:
    """
    Gas fee optimizer for DEX transactions.

    Tracks gas prices, predicts near-term costs, and recommends
    priority fees based on trade urgency.

    Parameters
    ----------
    eth_price_usd : float
        Estimated ETH price for USD cost calculations.
    gas_units_swap : int
        Typical gas units for a swap transaction.
    """

    # Urgency percentile mapping
    URGENCY_PERCENTILES = {
        "low": 25,
        "normal": 50,
        "high": 90,
        "critical": 99,
    }

    def __init__(
        self,
        eth_price_usd: float = 3000.0,
        gas_units_swap: int = 150_000,
    ) -> None:
        self._gas_history: Deque[GasPriceEntry] = deque(maxlen=_MAX_HISTORY)
        self.eth_price_usd = eth_price_usd
        self.gas_units_swap = gas_units_swap

        logger.info(
            "GasOptimizer: eth_price=$%.0f gas_units=%d",
            eth_price_usd,
            gas_units_swap,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_gas_price(
        self,
        base_fee: float,
        priority_fee: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Track a gas price observation.

        Parameters
        ----------
        base_fee : float
            Base fee in gwei.
        priority_fee : float
            Priority fee (tip) in gwei.
        timestamp : float, optional
            Observation time (defaults to now).
        """
        if base_fee < 0 or priority_fee < 0:
            logger.warning("Negative gas price ignored: base=%.4f priority=%.4f", base_fee, priority_fee)
            return

        ts = timestamp if timestamp is not None else time.time()
        self._gas_history.append(GasPriceEntry(
            base_fee=base_fee,
            priority_fee=priority_fee,
            timestamp=ts,
        ))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_gas_price(self, blocks_ahead: int = 1) -> Dict[str, float]:
        """
        Predict gas price for the next N blocks based on recent history.

        Uses exponential weighted moving average of recent observations.

        Parameters
        ----------
        blocks_ahead : int
            Number of blocks to look ahead (1 block ~ 0.25s on Arbitrum).

        Returns
        -------
        dict
            {base_fee, priority_fee, total_fee, confidence}
        """
        if not self._gas_history:
            return {
                "base_fee": _DEFAULT_BASE_FEE_GWEI,
                "priority_fee": _DEFAULT_PRIORITY_FEE_GWEI,
                "total_fee": _DEFAULT_BASE_FEE_GWEI + _DEFAULT_PRIORITY_FEE_GWEI,
                "confidence": 0.0,
            }

        # Use recent entries (last 20 or all if fewer)
        recent = list(self._gas_history)[-20:]

        # EMA with decay factor
        alpha = 0.3
        ema_base = recent[0].base_fee
        ema_priority = recent[0].priority_fee

        for entry in recent[1:]:
            ema_base = alpha * entry.base_fee + (1 - alpha) * ema_base
            ema_priority = alpha * entry.priority_fee + (1 - alpha) * ema_priority

        # Detect trend: compare last 5 to previous 5
        if len(recent) >= 10:
            recent_avg = statistics.mean(e.base_fee for e in recent[-5:])
            older_avg = statistics.mean(e.base_fee for e in recent[-10:-5])
            trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            # Extrapolate trend
            ema_base *= (1.0 + trend * blocks_ahead * 0.1)
        else:
            trend = 0.0

        # Confidence based on history size
        confidence = min(1.0, len(self._gas_history) / 50.0)

        return {
            "base_fee": max(0.0, ema_base),
            "priority_fee": max(0.0, ema_priority),
            "total_fee": max(0.0, ema_base + ema_priority),
            "confidence": confidence,
            "trend": trend,
            "blocks_ahead": blocks_ahead,
        }

    # ------------------------------------------------------------------
    # Optimal fee calculation
    # ------------------------------------------------------------------

    def optimal_priority_fee(self, urgency: str = "normal") -> float:
        """
        Calculate optimal priority fee based on urgency level.

        Parameters
        ----------
        urgency : str
            One of 'low', 'normal', 'high', 'critical'.

        Returns
        -------
        float
            Recommended priority fee in gwei.
        """
        percentile = self.URGENCY_PERCENTILES.get(urgency.lower(), 50)

        if not self._gas_history:
            # Default recommendations per urgency
            defaults = {
                "low": 0.005,
                "normal": 0.01,
                "high": 0.05,
                "critical": 0.1,
            }
            return defaults.get(urgency.lower(), 0.01)

        priority_fees = sorted(e.priority_fee for e in self._gas_history)
        idx = max(0, min(len(priority_fees) - 1, int(len(priority_fees) * percentile / 100.0)))

        return priority_fees[idx]

    # ------------------------------------------------------------------
    # Cost checking
    # ------------------------------------------------------------------

    def is_gas_favorable(self, max_cost_usd: float = 5.0) -> bool:
        """
        Return True if current estimated gas cost is below threshold.

        Parameters
        ----------
        max_cost_usd : float
            Maximum acceptable gas cost in USD.

        Returns
        -------
        bool
            True if gas is cheap enough.
        """
        cost = self.current_cost_usd()
        return cost <= max_cost_usd

    def current_cost_usd(self, urgency: str = "normal") -> float:
        """
        Estimate current gas cost in USD for a swap transaction.

        Parameters
        ----------
        urgency : str
            Urgency level for priority fee.

        Returns
        -------
        float
            Estimated cost in USD.
        """
        prediction = self.predict_gas_price()
        priority = self.optimal_priority_fee(urgency)
        total_gwei = prediction["base_fee"] + priority
        total_eth = total_gwei * self.gas_units_swap / 1e9
        return total_eth * self.eth_price_usd

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return gas price statistics.

        Returns
        -------
        dict
            {avg, median, p95, min, max, trend, history_size, cost_usd_estimate}
        """
        if not self._gas_history:
            return {
                "avg": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "min": 0.0,
                "max": 0.0,
                "trend": "unknown",
                "history_size": 0,
                "cost_usd_estimate": 0.0,
            }

        total_fees = [e.total_fee for e in self._gas_history]
        sorted_fees = sorted(total_fees)

        avg = statistics.mean(total_fees)
        median = statistics.median(total_fees)
        p95_idx = max(0, min(len(sorted_fees) - 1, int(len(sorted_fees) * 0.95)))
        p95 = sorted_fees[p95_idx]

        # Trend detection
        if len(total_fees) >= 10:
            recent = statistics.mean(total_fees[-5:])
            older = statistics.mean(total_fees[-10:-5])
            if recent > older * 1.1:
                trend = "rising"
            elif recent < older * 0.9:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        cost_estimate = self.current_cost_usd()

        return {
            "avg": avg,
            "median": median,
            "p95": p95,
            "min": min(total_fees),
            "max": max(total_fees),
            "trend": trend,
            "history_size": len(self._gas_history),
            "cost_usd_estimate": cost_estimate,
        }

    def update_eth_price(self, price_usd: float) -> None:
        """Update the ETH price used for USD cost calculations."""
        if price_usd > 0:
            self.eth_price_usd = price_usd
