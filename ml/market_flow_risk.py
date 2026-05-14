"""
Market Flow Risk Adapter - Risk adapts to real-time market conditions.

Risk parameters that adapt to market flow:
- Stop loss distance (wider in high vol, tighter in low vol)
- Maximum position size (reduce in low liquidity, increase in high vol)
- Take profit targets (adjust based on regime)
- Confidence thresholds (raise in uncertain conditions)
- Overall risk on/off (pause in extreme conditions)

This is the highest practical risk management - risk that breathes with the market.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RiskCondition(Enum):
    """Risk conditions."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"
    CRISIS = "crisis"


@dataclass
class MarketFlowRisk:
    """Real-time market flow risk assessment."""

    condition: str
    overall_score: float  # 0-1, higher = riskier
    volatility_score: float = 0.5
    liquidity_score: float = 0.5
    sentiment_score: float = 0.5
    regime_score: float = 0.5
    spread_score: float = 0.5
    volume_score: float = 0.5


@dataclass
class RiskParameters:
    """Adaptable risk parameters."""

    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.03
    max_position_pct: float = 0.10
    confidence_threshold: float = 0.50
    max_daily_loss_pct: float = 0.10
    max_concurrent_positions: int = 5
    use_trailing_stop: bool = True
    trailing_distance_pct: float = 0.01


@dataclass
class RiskDecision:
    """Complete risk adaptation decision."""

    # Parameter adjustments
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    position_size_multiplier: float = 1.0
    confidence_adjustment: float = 0.0
    daily_loss_multiplier: float = 1.0
    max_positions_adjustment: int = 0
    trailing_stop_enabled: bool = True
    trailing_distance_multiplier: float = 1.0

    # State
    risk_condition: str = "normal"
    should_pause_trading: bool = False
    should_halt_new_positions: bool = False

    # Reasoning
    reasoning: List[str] = field(default_factory=list)


class MarketFlowRiskAdapter:
    """
    Risk adapter that responds to real-time market flow.

    Monitors market conditions and adapts all risk parameters accordingly:
    - Stop loss: wider in high vol, tighter in low vol
    - Position size: smaller in bad conditions, larger in good
    - Take profit: regime-aware targets
    - Confidence: raise when uncertain, lower when confident
    - Pause/halt: in extreme conditions

    This is risk "breathing" with the market.
    """

    def __init__(
        self,
        # Base parameters
        base_stop_loss_pct: float = 0.015,
        base_take_profit_pct: float = 0.03,
        base_max_position_pct: float = 0.10,
        base_confidence_threshold: float = 0.50,

        # Volatility scaling
        use_volatility_scaling: bool = True,
        vol_stop_multiplier_range: Tuple[float, float] = (1.5, 0.8),  # (high vol, low vol)
        vol_position_multiplier_range: Tuple[float, float] = (0.5, 1.3),

        # Liquidity scaling
        use_liquidity_scaling: bool = True,
        min_volume_ratio: float = 0.8,
        min_spread_bps: float = 20.0,
        liquidity_stop_multiplier: float = 1.5,

        # Regime scaling
        use_regime_scaling: bool = True,
        trending_stop_mult: float = 1.0,
        ranging_stop_mult: float = 1.2,
        high_vol_stop_mult: float = 1.3,

        # Sentiment scaling
        use_sentiment_scaling: bool = True,
        fear_threshold: float = 30.0,
        greed_threshold: float = 70.0,
        fear_position_mult: float = 0.6,
        greed_position_mult: float = 0.8,

        # Emergency thresholds
        emergency_vol_threshold: float = 0.05,  # 5% daily vol = extreme
        emergency_spread_threshold: float = 100.0,  # 100 bps = halt
        emergency_volume_ratio_threshold: float = 0.3,  # 30% of normal = halt

        # Adaptive settings
        adaptation_rate: float = 0.10,
        adaptation_interval_trades: int = 20,

        **kwargs,
    ) -> None:
        # Base parameters
        self.base_stop_loss_pct = base_stop_loss_pct
        self.base_take_profit_pct = base_take_profit_pct
        self.base_max_position_pct = base_max_position_pct
        self.base_confidence_threshold = base_confidence_threshold

        # Scaling ranges
        self.use_volatility_scaling = use_volatility_scaling
        self.vol_stop_multiplier_range = vol_stop_multiplier_range
        self.vol_position_multiplier_range = vol_position_multiplier_range

        self.use_liquidity_scaling = use_liquidity_scaling
        self.min_volume_ratio = min_volume_ratio
        self.min_spread_bps = min_spread_bps
        self.liquidity_stop_multiplier = liquidity_stop_multiplier

        self.use_regime_scaling = use_regime_scaling
        self.trending_stop_mult = trending_stop_mult
        self.ranging_stop_mult = ranging_stop_mult
        self.high_vol_stop_mult = high_vol_stop_mult

        self.use_sentiment_scaling = use_sentiment_scaling
        self.fear_threshold = fear_threshold
        self.greed_threshold = greed_threshold
        self.fear_position_mult = fear_position_mult
        self.greed_position_mult = greed_position_mult

        # Emergency
        self.emergency_vol_threshold = emergency_vol_threshold
        self.emergency_spread_threshold = emergency_spread_threshold
        self.emergency_volume_ratio_threshold = emergency_volume_ratio_threshold

        # State
        self._current_risk = MarketFlowRisk(condition="normal", overall_score=0.5)
        self._parameter_history: List[RiskParameters] = []
        self._trades_since_adaptation = 0

    def assess_market_flow_risk(
        self,
        # Volatility
        current_volatility: float,
        historical_volatility: float,
        # Liquidity
        current_volume: float,
        average_volume: float,
        bid_ask_spread_bps: float,
        order_book_depth: float,
        # Regime
        current_regime: str,
        # Sentiment
        fear_greed_index: float,
        # Price
        price_change_pct: float,
    ) -> MarketFlowRisk:
        """Assess current market flow risk."""
        scores = {}

        # Volatility score (0 = safe, 1 = risky)
        if historical_volatility > 0:
            vol_ratio = current_volatility / historical_volatility
        else:
            vol_ratio = 1.0
        scores["volatility"] = min(1.0, vol_ratio * 2)  # Scale

        # Liquidity score
        volume_ratio = current_volume / average_volume if average_volume > 0 else 1.0
        spread_score = min(1.0, bid_ask_spread_bps / self.min_spread_bps)
        depth_score = 1.0 if order_book_depth > 1000 else 0.5
        scores["liquidity"] = (1 - spread_score) * 0.5 + (1 - volume_ratio) * 0.3 + (1 - depth_score) * 0.2

        # Sentiment score
        if fear_greed_index < self.fear_threshold:
            scores["sentiment"] = 0.8  # Fear = risky
        elif fear_greed_index > self.greed_threshold:
            scores["sentiment"] = 0.6  # Greed = somewhat risky
        else:
            scores["sentiment"] = 0.3  # Neutral

        # Regime score
        if current_regime == "high_volatility":
            scores["regime"] = 0.8
        elif current_regime in ["trending_up", "trending_down"]:
            scores["regime"] = 0.4
        elif current_regime == "ranging":
            scores["regime"] = 0.3
        else:
            scores["regime"] = 0.5

        # Spread score (direct)
        scores["spread"] = min(1.0, spread_score)

        # Volume score (inverse)
        scores["volume"] = 1.0 - min(1.0, volume_ratio)

        # Overall score
        overall = (
            scores["volatility"] * 0.25
            + scores["liquidity"] * 0.20
            + scores["sentiment"] * 0.15
            + scores["regime"] * 0.15
            + scores["spread"] * 0.15
            + scores["volume"] * 0.10
        )

        # Determine condition
        if overall > 0.8:
            condition = RiskCondition.CRISIS.value
        elif overall > 0.6:
            condition = RiskCondition.HIGH.value
        elif overall > 0.4:
            condition = RiskCondition.ELEVATED.value
        else:
            condition = RiskCondition.NORMAL.value

        risk = MarketFlowRisk(
            condition=condition,
            overall_score=overall,
            volatility_score=scores["volatility"],
            liquidity_score=scores["liquidity"],
            sentiment_score=scores["sentiment"],
            regime_score=scores["regime"],
            spread_score=scores["spread"],
            volume_score=scores["volume"],
        )

        self._current_risk = risk
        return risk

    def adapt_risk(
        self,
        market_risk: MarketFlowRisk,
        recent_performance: Dict[str, Any],
    ) -> RiskDecision:
        """
        Adapt risk parameters based on market flow and recent performance.
        """
        decision = RiskDecision(risk_condition=market_risk.condition)
        reasoning = []

        # Check for emergency conditions first
        if market_risk.condition in [RiskCondition.CRISIS.value, RiskCondition.HIGH.value]:
            decision.should_pause_trading = True
            reasoning.append(f"Risk condition: {market_risk.condition}")

        if market_risk.condition == RiskCondition.CRISIS.value:
            decision.should_halt_new_positions = True
            reasoning.append("CRISIS: Halting all new positions")

        # 1. Volatility-based stop loss
        if self.use_volatility_scaling:
            vol_score = market_risk.volatility_score
            if vol_score > 0.7:  # High vol
                decision.stop_loss_multiplier = self.vol_stop_multiplier_range[0]
                reasoning.append(f"High vol: widen stops x{self.vol_stop_multiplier_range[0]}")
            elif vol_score < 0.4:  # Low vol
                decision.stop_loss_multiplier = self.vol_stop_multiplier_range[1]
                reasoning.append(f"Low vol: tighten stops x{self.vol_stop_multiplier_range[1]}")

            # Position size
            position_mult = self.vol_position_multiplier_range[0] if vol_score > 0.7 else self.vol_position_multiplier_range[1]
            decision.position_size_multiplier *= position_mult
            if position_mult != 1.0:
                reasoning.append(f"Volatility position: x{position_mult}")

        # 2. Liquidity-based scaling
        if self.use_liquidity_scaling:
            liq_score = market_risk.liquidity_score
            if liq_score > 0.6:  # Poor liquidity
                decision.position_size_multiplier *= 0.5
                decision.stop_loss_multiplier *= self.liquidity_stop_multiplier
                reasoning.append("Poor liquidity: reduce size, widen stops")
            elif liq_score < 0.3:  # Good liquidity
                decision.position_size_multiplier *= 1.2

        # 3. Sentiment-based scaling
        if self.use_sentiment_scaling:
            sent_score = market_risk.sentiment_score
            if sent_score > 0.6:  # Fear
                decision.position_size_multiplier *= self.fear_position_mult
                decision.confidence_adjustment += 0.15
                reasoning.append(f"Fear: reduce size x{self.fear_position_mult}, raise confidence")
            elif sent_score > 0.4:  # Greed
                decision.position_size_multiplier *= self.greed_position_mult
                reasoning.append(f"Greed: cautious size x{self.greed_position_mult}")

        # 4. Regime-based take profit adjustment
        if self.use_regime_scaling:
            regime_score = market_risk.regime_score
            if regime_score > 0.6:  # High vol regime
                decision.take_profit_multiplier = 0.8  # Lower targets
                reasoning.append("High vol regime: lower targets")
            elif regime_score < 0.4:  # Trend regime can run further
                decision.take_profit_multiplier = 1.2  # Higher targets

        # 5. Performance-based fine-tuning
        win_rate = recent_performance.get("win_rate", 0.5)
        if win_rate < 0.35:  # Poor performance
            decision.stop_loss_multiplier *= 0.8  # Tighter stops
            decision.confidence_adjustment += 0.10
            reasoning.append("Poor win rate: tighter stops, higher confidence")
        elif win_rate > 0.60:  # Good performance
            decision.confidence_adjustment -= 0.05  # Can be less strict

        # 6. Emergency overrides
        if market_risk.spread_score > 0.8:  # Wide spreads
            decision.should_pause_trading = True
            reasoning.append(f"Wide spread: pause trading")

        if market_risk.volume_score > 0.8:  # Low volume
            decision.should_pause_trading = True
            reasoning.append(f"Very low volume: pause trading")

        decision.reasoning = reasoning

        return decision

    def get_current_parameters(
        self,
        decision: RiskDecision,
    ) -> RiskParameters:
        """Get adapted risk parameters."""
        # Note: Multipliers apply in different ways:
        # - stop_loss_multiplier > 1 means WIDER (safer)
        # - position_size_multiplier < 1 means SMALLER (safer)

        # Calculate effective values
        stop_loss = self.base_stop_loss_pct * decision.stop_loss_multiplier
        take_profit = self.base_take_profit_pct * decision.take_profit_multiplier

        # Position size: clamp between 0.01 and base_max
        position_adj = decision.position_size_multiplier
        max_position = max(0.01, min(0.20, self.base_max_position_pct * position_adj))

        # Confidence: adjust threshold
        confidence = max(0.3, min(0.8, self.base_confidence_threshold + decision.confidence_adjustment))

        # Max positions
        max_positions = max(1, 5 + decision.max_positions_adjustment)

        return RiskParameters(
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            max_position_pct=max_position,
            confidence_threshold=confidence,
            max_concurrent_positions=max_positions,
            use_trailing_stop=decision.trailing_stop_enabled,
        )

    def check_should_trade(
        self,
        risk_decision: RiskDecision,
    ) -> Tuple[bool, str]:
        """Check if should take new positions."""
        if risk_decision.should_halt_new_positions:
            return False, "CRISIS: All trading halted"

        if risk_decision.should_pause_trading:
            return False, f"Risk condition: {risk_decision.risk_condition}"

        return True, "OK"


def create_risk_adapter(**kwargs) -> MarketFlowRiskAdapter:
    """Factory function."""
    return MarketFlowRiskAdapter(**kwargs)


__all__ = [
    "MarketFlowRiskAdapter",
    "MarketFlowRisk",
    "RiskParameters",
    "RiskDecision",
    "RiskCondition",
    "create_risk_adapter",
]