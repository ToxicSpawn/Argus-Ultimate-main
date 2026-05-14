#!/usr/bin/env python3
"""
Portfolio rebalancer with drift detection, regime-aware targets, and cost awareness.

Monitors portfolio weight drift against targets and generates rebalance orders
only when drift exceeds threshold and expected alpha exceeds transaction costs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RebalanceOrder:
    """A single rebalance instruction."""

    symbol: str
    side: str  # "buy" or "sell"
    target_weight: float
    current_weight: float
    delta_weight: float
    estimated_cost_bps: float


# ---------------------------------------------------------------------------
# Regime target allocations
# ---------------------------------------------------------------------------

_REGIME_TARGETS: Dict[str, Dict[str, float]] = {
    "crisis": {
        "BTC/AUD": 0.35,
        "ETH/AUD": 0.25,
        "SOL/AUD": 0.05,
        "XRP/AUD": 0.05,
        "ADA/AUD": 0.05,
        "DOT/AUD": 0.05,
        "AVAX/AUD": 0.05,
        "LINK/AUD": 0.05,
        "MATIC/AUD": 0.05,
        "ATOM/AUD": 0.05,
    },
    "volatile": {
        "BTC/AUD": 0.35,
        "ETH/AUD": 0.25,
        "SOL/AUD": 0.05,
        "XRP/AUD": 0.05,
        "ADA/AUD": 0.05,
        "DOT/AUD": 0.05,
        "AVAX/AUD": 0.05,
        "LINK/AUD": 0.05,
        "MATIC/AUD": 0.05,
        "ATOM/AUD": 0.05,
    },
    "bear": {
        "BTC/AUD": 0.40,
        "ETH/AUD": 0.20,
        "SOL/AUD": 0.05,
        "XRP/AUD": 0.05,
        "ADA/AUD": 0.05,
        "DOT/AUD": 0.05,
        "AVAX/AUD": 0.05,
        "LINK/AUD": 0.05,
        "MATIC/AUD": 0.05,
        "ATOM/AUD": 0.05,
    },
}

# Bull / trending: equal weight across 11 symbols (~9.09% each)
_EQUAL_SYMBOLS = [
    "BTC/AUD", "ETH/AUD", "SOL/AUD", "XRP/AUD", "ADA/AUD",
    "DOT/AUD", "AVAX/AUD", "LINK/AUD", "MATIC/AUD", "ATOM/AUD", "DOGE/AUD",
]
_equal_weight = round(1.0 / len(_EQUAL_SYMBOLS), 4)
_REGIME_TARGETS["trending"] = {s: _equal_weight for s in _EQUAL_SYMBOLS}
_REGIME_TARGETS["bull"] = dict(_REGIME_TARGETS["trending"])


class PortfolioRebalancer:
    """Drift-based, cost-aware portfolio rebalancer with regime targets."""

    def __init__(
        self,
        config: Any = None,
        transaction_cost_model: Any = None,
    ) -> None:
        self._config = config
        self._cost_model = transaction_cost_model

        # Configurable parameters (safe access)
        self._drift_threshold_pct: float = float(
            _cfg(config, "portfolio_rebalancer.drift_threshold_pct", 5.0)
        )
        self._min_interval_hours: float = float(
            _cfg(config, "portfolio_rebalancer.min_rebalance_interval_hours", 4)
        )
        self._cost_aware_skip: bool = bool(
            _cfg(config, "portfolio_rebalancer.cost_aware_skip", True)
        )

        self._last_rebalance_ts: float = 0.0
        logger.info(
            "PortfolioRebalancer initialised  drift_threshold=%.1f%%  min_interval=%dh",
            self._drift_threshold_pct,
            self._min_interval_hours,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> List[RebalanceOrder]:
        """Return rebalance orders for any symbol drifted beyond threshold."""

        # Enforce minimum rebalance interval
        now = time.time()
        elapsed_hours = (now - self._last_rebalance_ts) / 3600.0
        if self._last_rebalance_ts > 0 and elapsed_hours < self._min_interval_hours:
            logger.debug(
                "Rebalance skipped: %.1fh since last (min %dh)",
                elapsed_hours,
                self._min_interval_hours,
            )
            return []

        orders: List[RebalanceOrder] = []
        all_symbols = set(current_weights) | set(target_weights)

        for symbol in sorted(all_symbols):
            cur = current_weights.get(symbol, 0.0)
            tgt = target_weights.get(symbol, 0.0)
            drift_pct = abs(cur - tgt) * 100.0  # already in 0-1 range

            if drift_pct < self._drift_threshold_pct:
                continue

            delta = tgt - cur
            side = "buy" if delta > 0 else "sell"

            # Estimate cost via model if available
            est_cost_bps = 0.0
            if self._cost_model is not None:
                try:
                    cost_est = self._cost_model.estimate_cost(
                        symbol=symbol,
                        quantity=abs(delta),
                        side=side,
                        current_price=0.0,  # price unknown at rebalance planning
                    )
                    est_cost_bps = getattr(cost_est, "total_bps", 0.0)
                except Exception:
                    logger.debug("Cost estimate failed for %s, proceeding anyway", symbol)

            # Cost-aware skip: if cost > drift benefit (simplified heuristic)
            if self._cost_aware_skip and est_cost_bps > 0:
                drift_benefit_bps = drift_pct * 100.0  # rough proxy
                if est_cost_bps > drift_benefit_bps:
                    logger.info(
                        "Skip rebalance %s: cost %.1f bps > drift benefit %.1f bps",
                        symbol,
                        est_cost_bps,
                        drift_benefit_bps,
                    )
                    continue

            orders.append(
                RebalanceOrder(
                    symbol=symbol,
                    side=side,
                    target_weight=tgt,
                    current_weight=cur,
                    delta_weight=delta,
                    estimated_cost_bps=est_cost_bps,
                )
            )

        if orders:
            self._last_rebalance_ts = now
            logger.info("Rebalance triggered: %d orders generated", len(orders))

        return orders

    def get_regime_targets(self, regime: str) -> Dict[str, float]:
        """Return target weight allocation for a given market regime."""
        regime_lower = regime.lower().strip()
        targets = _REGIME_TARGETS.get(regime_lower)
        if targets is None:
            logger.warning(
                "Unknown regime '%s', falling back to equal-weight", regime
            )
            return dict(_REGIME_TARGETS["trending"])
        return dict(targets)

    @property
    def last_rebalance_ts(self) -> float:
        return self._last_rebalance_ts

    @last_rebalance_ts.setter
    def last_rebalance_ts(self, value: float) -> None:
        self._last_rebalance_ts = value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(config: Any, dotted_key: str, default: Any) -> Any:
    """Safely navigate nested config via dotted key."""
    if config is None:
        return default
    parts = dotted_key.split(".")
    obj = config
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj
