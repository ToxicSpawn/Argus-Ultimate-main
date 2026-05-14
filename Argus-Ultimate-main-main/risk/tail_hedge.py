"""
Tail Hedge Advisor — recommends protective positions during elevated crash risk.

Monitors regime + VaR signals to recommend hedges:
  - BTC put options (Deribit)
  - Short BTC futures (partial)
  - USDT/stablecoin flight
  - Inverse ETF equivalents

Does NOT execute hedges — returns recommendations for human/system approval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Urgency helpers
# ---------------------------------------------------------------------------

_MIN_URGENCY_CLAMP = 0.0
_MAX_URGENCY_CLAMP = 1.0


def _clamp(value: float, lo: float = _MIN_URGENCY_CLAMP, hi: float = _MAX_URGENCY_CLAMP) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HedgeRecommendation:
    """
    A single hedging recommendation produced by TailHedgeAdvisor.

    Attributes
    ----------
    instrument:
        Textual description of the hedging instrument, e.g.
        "BTC-31MAR26-60000-P (Deribit put option)".
    action:
        One of "BUY_PUT", "SHORT_FUTURES", or "INCREASE_CASH".
    size_usd:
        Recommended notional USD size for the hedge.
    rationale:
        Human-readable explanation of why this hedge is recommended.
    urgency:
        Severity score in [0, 1].  Values >= min_urgency are treated as
        actionable by TailHedgeAdvisor.should_hedge().
    estimated_cost_usd:
        Estimated up-front cost (option premium, slippage, etc.) in USD.
        For futures/cash hedges this is typically near zero.
    generated_at:
        UTC timestamp when this recommendation was created.
    """

    instrument: str
    action: str  # "BUY_PUT" | "SHORT_FUTURES" | "INCREASE_CASH"
    size_usd: float
    rationale: str
    urgency: float          # 0-1
    estimated_cost_usd: float
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "action": self.action,
            "size_usd": round(self.size_usd, 2),
            "rationale": self.rationale,
            "urgency": round(self.urgency, 4),
            "estimated_cost_usd": round(self.estimated_cost_usd, 2),
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


class TailHedgeAdvisor:
    """
    Recommend protective hedging positions when crash indicators are elevated.

    The advisor does NOT place orders.  It returns a list of
    ``HedgeRecommendation`` objects that must be reviewed and approved
    before execution.

    Parameters
    ----------
    capital_usd:
        Total portfolio capital in USD.
    hedge_budget_pct:
        Maximum fraction of capital available to spend on hedges (default 2 %).
        This limits the total ``estimated_cost_usd`` across all recommendations.
    min_urgency:
        Minimum urgency threshold for ``should_hedge()`` to return True
        (default 0.5).
    """

    # Thresholds for crash indicator evaluation
    CRASH_INDICATORS: Dict[str, float] = {
        # Regime strings that trigger hedging
        "high_vol_regime": 1.0,          # regime == "HIGH_VOL"
        "crisis_regime": 1.0,            # regime == "CRISIS"
        # Portfolio VaR thresholds (as fraction of capital)
        "var_moderate_threshold": 0.02,  # 2% of capital
        "var_elevated_threshold": 0.03,  # 3% of capital — triggers hedging
        "var_severe_threshold": 0.05,    # 5% — maximum urgency
        # Funding rate thresholds (8-hour rate)
        "funding_extreme_negative": -0.05,  # -5% per 8h
        "funding_very_negative": -0.02,     # -2% per 8h
        # Fear & Greed index (0 = extreme fear, 100 = extreme greed)
        "fear_extreme": 15.0,
        "fear_elevated": 25.0,
        # Option premium estimate as % of notional (rough approximation)
        "put_premium_pct": 0.03,        # ~3% of notional for 10% OTM put
        "short_futures_cost_pct": 0.001, # ~0.1% of notional (fees/slippage)
    }

    def __init__(
        self,
        capital_usd: float,
        hedge_budget_pct: float = 0.02,
        min_urgency: float = 0.5,
    ) -> None:
        if capital_usd <= 0:
            raise ValueError("capital_usd must be positive")
        if not (0 < hedge_budget_pct < 1):
            raise ValueError("hedge_budget_pct must be between 0 and 1")
        if not (0 <= min_urgency <= 1):
            raise ValueError("min_urgency must be in [0, 1]")

        self.capital_usd = capital_usd
        self.hedge_budget_pct = hedge_budget_pct
        self.hedge_budget_usd = capital_usd * hedge_budget_pct
        self.min_urgency = min_urgency

        # History of the last evaluate() call inputs for diagnostics
        self._last_inputs: Optional[dict] = None

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        regime: str,
        portfolio_var_pct: float,
        funding_rate: float,
        fear_greed_index: float,
    ) -> List[HedgeRecommendation]:
        """
        Evaluate current market conditions and return hedging recommendations.

        Parameters
        ----------
        regime:
            Current market regime string.  Recognised high-risk values are
            "HIGH_VOL", "CRISIS", "BEAR", "RISK_OFF".  Other values are treated
            as normal.
        portfolio_var_pct:
            Current portfolio VaR expressed as a fraction of capital
            (e.g. 0.03 = 3 %).
        funding_rate:
            Latest 8-hour perpetual funding rate (e.g. -0.02 = -2 %).
        fear_greed_index:
            Fear & Greed index in [0, 100].  Values below ~25 indicate fear.

        Returns
        -------
        List of HedgeRecommendation objects, possibly empty.
        """
        self._last_inputs = {
            "regime": regime,
            "portfolio_var_pct": portfolio_var_pct,
            "funding_rate": funding_rate,
            "fear_greed_index": fear_greed_index,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        ci = self.CRASH_INDICATORS
        recommendations: List[HedgeRecommendation] = []
        regime_upper = regime.upper()

        # ----------------------------------------------------------
        # 1. Regime-based urgency
        # ----------------------------------------------------------
        regime_urgency = 0.0
        if regime_upper in ("CRISIS",):
            regime_urgency = 1.0
        elif regime_upper in ("HIGH_VOL", "BEAR", "RISK_OFF"):
            regime_urgency = 0.7
        elif regime_upper in ("VOLATILE", "CAUTION"):
            regime_urgency = 0.4

        # ----------------------------------------------------------
        # 2. VaR-based urgency
        # ----------------------------------------------------------
        var_urgency = 0.0
        if portfolio_var_pct >= ci["var_severe_threshold"]:
            var_urgency = 1.0
        elif portfolio_var_pct >= ci["var_elevated_threshold"]:
            # Linear scale between elevated and severe
            span = ci["var_severe_threshold"] - ci["var_elevated_threshold"]
            var_urgency = 0.5 + 0.5 * (
                (portfolio_var_pct - ci["var_elevated_threshold"]) / span
            )
            var_urgency = _clamp(var_urgency, 0.5, 1.0)
        elif portfolio_var_pct >= ci["var_moderate_threshold"]:
            span = ci["var_elevated_threshold"] - ci["var_moderate_threshold"]
            var_urgency = 0.2 + 0.3 * (
                (portfolio_var_pct - ci["var_moderate_threshold"]) / span
            )
            var_urgency = _clamp(var_urgency, 0.2, 0.5)

        # ----------------------------------------------------------
        # 3. Funding-rate urgency
        # ----------------------------------------------------------
        funding_urgency = 0.0
        if funding_rate < ci["funding_extreme_negative"]:
            funding_urgency = 0.8
        elif funding_rate < ci["funding_very_negative"]:
            # Linear interpolation between very_negative and extreme_negative
            span = ci["funding_very_negative"] - ci["funding_extreme_negative"]
            funding_urgency = 0.4 + 0.4 * (
                (ci["funding_very_negative"] - funding_rate) / span
            )
            funding_urgency = _clamp(funding_urgency, 0.4, 0.8)

        # ----------------------------------------------------------
        # 4. Fear & Greed urgency
        # ----------------------------------------------------------
        fear_urgency = 0.0
        if fear_greed_index < ci["fear_extreme"]:
            fear_urgency = 0.7
        elif fear_greed_index < ci["fear_elevated"]:
            span = ci["fear_elevated"] - ci["fear_extreme"]
            fear_urgency = 0.3 + 0.4 * (
                (ci["fear_elevated"] - fear_greed_index) / span
            )
            fear_urgency = _clamp(fear_urgency, 0.3, 0.7)

        # ----------------------------------------------------------
        # Composite urgency: weighted average
        # ----------------------------------------------------------
        composite_urgency = _clamp(
            0.35 * regime_urgency
            + 0.35 * var_urgency
            + 0.15 * funding_urgency
            + 0.15 * fear_urgency
        )

        logger.debug(
            "TailHedgeAdvisor.evaluate: regime_u=%.2f var_u=%.2f "
            "funding_u=%.2f fear_u=%.2f composite=%.2f",
            regime_urgency,
            var_urgency,
            funding_urgency,
            fear_urgency,
            composite_urgency,
        )

        if composite_urgency < self.min_urgency:
            logger.debug(
                "TailHedgeAdvisor: composite urgency %.2f < min %.2f — no hedge recommended",
                composite_urgency,
                self.min_urgency,
            )
            return []

        # ----------------------------------------------------------
        # Build recommendations proportional to urgency
        # ----------------------------------------------------------
        budget = self.hedge_budget_usd

        # --- Recommendation 1: BTC put options ---
        # Allocate up to 60 % of hedge budget to puts when urgency is high
        put_budget_pct = _clamp(0.60 * composite_urgency)
        put_size_usd = budget * put_budget_pct
        if put_size_usd >= 10.0:  # Minimum viable size
            put_premium = put_size_usd * ci["put_premium_pct"]
            strike_description = "~10% OTM"
            expiry_description = "nearest monthly expiry"
            recommendations.append(
                HedgeRecommendation(
                    instrument=f"BTC put option {strike_description} {expiry_description} (Deribit)",
                    action="BUY_PUT",
                    size_usd=round(put_size_usd, 2),
                    rationale=(
                        f"Elevated crash risk (composite urgency={composite_urgency:.2f}). "
                        f"Regime='{regime}', portfolio VaR={portfolio_var_pct:.2%}, "
                        f"Fear&Greed={fear_greed_index:.0f}. "
                        f"Put provides downside protection if BTC drops sharply."
                    ),
                    urgency=_clamp(composite_urgency),
                    estimated_cost_usd=round(put_premium, 2),
                )
            )

        # --- Recommendation 2: Short BTC futures (partial hedge) ---
        # More appropriate in CRISIS/HIGH_VOL regimes than in mild volatility
        if composite_urgency >= 0.65 or regime_upper in ("CRISIS", "HIGH_VOL"):
            futures_budget_pct = _clamp(0.40 * composite_urgency)
            futures_size_usd = budget / ci["short_futures_cost_pct"] * futures_budget_pct * 0.10
            # Cap futures hedge at 25 % of capital to avoid over-hedging
            futures_size_usd = min(futures_size_usd, self.capital_usd * 0.25)
            futures_cost = futures_size_usd * ci["short_futures_cost_pct"]
            if futures_size_usd >= 50.0:
                recommendations.append(
                    HedgeRecommendation(
                        instrument="BTC-PERP short futures (Kraken/Binance)",
                        action="SHORT_FUTURES",
                        size_usd=round(futures_size_usd, 2),
                        rationale=(
                            f"Regime='{regime}' with urgency={composite_urgency:.2f}. "
                            f"Partial short futures hedge reduces delta exposure without "
                            f"full exit from spot positions."
                        ),
                        urgency=_clamp(composite_urgency * 0.9),
                        estimated_cost_usd=round(futures_cost, 2),
                    )
                )

        # --- Recommendation 3: Increase stablecoin/cash allocation ---
        # Always recommended when urgency is meaningful
        cash_target_pct = _clamp(0.30 + 0.40 * composite_urgency)
        cash_size_usd = self.capital_usd * cash_target_pct
        recommendations.append(
            HedgeRecommendation(
                instrument="USDT/USDC stablecoin (on-exchange or wallet)",
                action="INCREASE_CASH",
                size_usd=round(cash_size_usd, 2),
                rationale=(
                    f"Recommend holding {cash_target_pct:.0%} of portfolio in stablecoins "
                    f"given composite crash urgency={composite_urgency:.2f}. "
                    f"Funding rate={funding_rate:.4f}, Fear&Greed={fear_greed_index:.0f}."
                ),
                urgency=_clamp(composite_urgency * 0.85),
                estimated_cost_usd=0.0,  # No direct cost
            )
        )

        # Deduplicate / sort by urgency descending
        recommendations.sort(key=lambda r: r.urgency, reverse=True)

        logger.info(
            "TailHedgeAdvisor: %d hedge recommendation(s) generated "
            "(composite urgency=%.2f, regime=%s)",
            len(recommendations),
            composite_urgency,
            regime,
        )

        return recommendations

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_hedge_cost(self, recommendations: List[HedgeRecommendation]) -> float:
        """
        Return the total estimated USD cost of a list of recommendations.

        INCREASE_CASH recommendations do not contribute to cost.
        """
        return sum(r.estimated_cost_usd for r in recommendations)

    def should_hedge(
        self,
        regime: str,
        portfolio_var_pct: float,
        funding_rate: float,
        fear_greed_index: float,
    ) -> bool:
        """
        Convenience method — returns True if any recommendation exceeds
        *min_urgency*.

        Does not retain the recommendation list; call ``evaluate()`` separately
        to retrieve actionable items.
        """
        recs = self.evaluate(regime, portfolio_var_pct, funding_rate, fear_greed_index)
        return any(r.urgency >= self.min_urgency for r in recs)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(
        self,
        regime: str,
        portfolio_var_pct: float,
        funding_rate: float,
        fear_greed_index: float,
    ) -> dict:
        """
        Evaluate current conditions and return a complete snapshot dict.

        Returns
        -------
        dict with keys:
            recommendations       — list of recommendation dicts
            total_hedge_cost_usd  — combined estimated cost
            hedge_budget_usd      — maximum allowed spend
            should_hedge          — bool
            inputs                — the raw inputs used for evaluation
            timestamp
        """
        recs = self.evaluate(regime, portfolio_var_pct, funding_rate, fear_greed_index)
        total_cost = self.get_hedge_cost(recs)
        actionable = any(r.urgency >= self.min_urgency for r in recs)

        return {
            "recommendations": [r.to_dict() for r in recs],
            "total_hedge_cost_usd": round(total_cost, 2),
            "hedge_budget_usd": round(self.hedge_budget_usd, 2),
            "should_hedge": actionable,
            "inputs": {
                "regime": regime,
                "portfolio_var_pct": portfolio_var_pct,
                "funding_rate": funding_rate,
                "fear_greed_index": fear_greed_index,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Hedge order generation
    # ------------------------------------------------------------------

    def generate_hedge_orders(
        self,
        recommendations: List[HedgeRecommendation],
        portfolio_value: float,
        btc_price: float = 65000.0,
    ) -> List[dict]:
        """
        Convert advisory HedgeRecommendations into concrete order dicts.

        Each returned dict contains:
            symbol      — instrument symbol
            side        — "sell" or "buy"
            quantity    — amount in base units or USD
            order_type  — "market" or "limit"
            reason      — human-readable reason for the order

        Parameters
        ----------
        recommendations:
            List from ``evaluate()``.
        portfolio_value:
            Total portfolio value in USD.
        btc_price:
            Current BTC price in USD (used to convert USD size to BTC quantity).
        """
        if portfolio_value <= 0 or btc_price <= 0:
            return []

        orders: List[dict] = []

        for rec in recommendations:
            if rec.action == "SHORT_FUTURES":
                # quantity = portfolio_value * hedge_pct / btc_price
                hedge_pct = rec.size_usd / portfolio_value if portfolio_value > 0 else 0.0
                quantity = portfolio_value * hedge_pct / btc_price
                if quantity > 0:
                    orders.append({
                        "symbol": "BTC-PERP",
                        "side": "sell",
                        "quantity": round(quantity, 6),
                        "order_type": "market",
                        "reason": f"Tail hedge: short BTC futures (urgency={rec.urgency:.2f})",
                    })

            elif rec.action == "INCREASE_CASH":
                # allocation = size_usd directly (already USD amount)
                allocation_pct = rec.size_usd / portfolio_value if portfolio_value > 0 else 0.0
                quantity = portfolio_value * allocation_pct
                if quantity > 0:
                    orders.append({
                        "symbol": "USDT",
                        "side": "buy",
                        "quantity": round(quantity, 2),
                        "order_type": "market",
                        "reason": f"Tail hedge: stablecoin allocation {allocation_pct:.0%} (urgency={rec.urgency:.2f})",
                    })

            elif rec.action == "BUY_PUT":
                # For puts: quantity is in USD notional (contracts abstracted away)
                quantity = rec.size_usd
                if quantity > 0:
                    orders.append({
                        "symbol": "BTC-PUT-OTM",
                        "side": "buy",
                        "quantity": round(quantity, 2),
                        "order_type": "limit",
                        "reason": f"Tail hedge: buy BTC put option (urgency={rec.urgency:.2f})",
                    })

        return orders

    def __repr__(self) -> str:
        return (
            f"TailHedgeAdvisor("
            f"capital=${self.capital_usd:.0f}, "
            f"budget={self.hedge_budget_pct:.1%}, "
            f"min_urgency={self.min_urgency})"
        )
