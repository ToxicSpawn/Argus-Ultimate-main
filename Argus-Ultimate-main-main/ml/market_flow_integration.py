"""
Market Flow Trading Integration - Pipes all signals through adaptation.

This module integrates market flow adaptation into the main trading loop:
- Signal generation → Flow analysis → Risk assessment → Execution

All signals go through market flow adaptation before execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ml.market_flow_strategy import (
    MarketFlowAdaptiveStrategy,
    StrategySignal,
)
from ml.market_flow_risk import (
    MarketFlowRiskAdapter,
    MarketFlowRisk,
    RiskDecision,
)
from ml.ultimate_adaptation import (
    UltimateAdaptationEngine,
    SentimentData,
    LiquidityMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSignal:
    """Signal after full market flow integration."""

    raw_signal: StrategySignal
    adapted_direction: str
    adapted_confidence: float
    adapted_position_size: float
    adapted_stop_loss: float
    adapted_take_profit: float
    risk: MarketFlowRisk
    risk_decision: RiskDecision
    adaptation: Any
    should_execute: bool
    execution_reason: str


@dataclass
class TradingState:
    """Current trading state."""

    symbols_traded: Dict[str, float] = field(default_factory=dict)
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketFlowTradingIntegration:
    """
    Complete market flow adaptation integration.

    Pipeline:
    1. Fetch market data
    2. Generate raw signals (strategy)
    3. Assess market flow risk
    4. Adapt strategy parameters
    5. Apply risk decisions
    6. Execute or reject

    All signals go through this integration before execution.
    """

    def __init__(
        self,
        strategy_min_confidence: float = 0.50,
        strategy_base_position: float = 0.02,
        risk_base_stop_loss: float = 0.015,
        risk_base_take_profit: float = 0.03,
        risk_base_position: float = 0.10,
        use_ultimate_adaptation: bool = True,
        use_risk_adapter: bool = True,
        **kwargs,
    ) -> None:
        self.strategy = MarketFlowAdaptiveStrategy(
            min_confidence=strategy_min_confidence,
            base_position_pct=strategy_base_position,
        )

        self.risk_adapter = MarketFlowRiskAdapter(
            base_stop_loss_pct=risk_base_stop_loss,
            base_take_profit_pct=risk_base_take_profit,
            base_max_position_pct=risk_base_position,
        )

        self.ultimate_adapter = None
        if use_ultimate_adaptation:
            self.ultimate_adapter = UltimateAdaptationEngine()

        self.use_risk_adapter = use_risk_adapter
        self.use_ultimate_adaptation = use_ultimate_adaptation

        self._state = TradingState()
        self._total_trades = 0

    async def process_signal(
        self,
        symbol: str,
        ohlcv_data: list,
        market_metrics: dict,
    ) -> Optional[IntegratedSignal]:
        """Process signal through full market flow adaptation pipeline."""
        raw_signals = await self.strategy.generate_signals(
            symbol,
            ohlcv_data,
            equity=10000,
        )

        if not raw_signals:
            return None

        raw = raw_signals[0]

        risk = self.risk_adapter.assess_market_flow_risk(
            current_volatility=market_metrics.get("volatility", 0.02),
            historical_volatility=market_metrics.get("historical_vol", 0.02),
            current_volume=market_metrics.get("volume", 1000),
            average_volume=market_metrics.get("avg_volume", 1000),
            bid_ask_spread_bps=market_metrics.get("spread_bps", 10),
            order_book_depth=market_metrics.get("depth", 1000),
            current_regime=market_metrics.get("regime", "ranging"),
            fear_greed_index=market_metrics.get("fear_greed", 50),
            price_change_pct=market_metrics.get("price_change", 0),
        )

        perf = self.strategy.get_performance()
        recent_perf = {"win_rate": perf.win_rate, "expectancy": perf.expectancy}

        risk_decision = self.risk_adapter.adapt_risk(risk, recent_perf)

        should_execute, reason = self.risk_adapter.check_should_trade(risk_decision)

        if not should_execute:
            logger.warning(f"Signal rejected for {symbol}: {reason}")
            return None

        adaptation = None
        if self.use_ultimate_adaptation and self.ultimate_adapter:
            adaptation = self.ultimate_adapter.adapt(
                symbol=symbol,
                regime_history=market_metrics.get("regime_history", ["unknown"] * 5),
                open_positions=self._state.symbols_traded,
                current_momentum=market_metrics.get("momentum", 0),
                current_volatility=market_metrics.get("volatility", 0.02),
                recent_volatility=market_metrics.get("volatility", 0.02),
                historical_volatility=market_metrics.get("historical_vol", 0.02),
                sentiment=SentimentData(
                    fear_greed_index=market_metrics.get("fear_greed", 50),
                ),
                liquidity=LiquidityMetrics(
                    bid_ask_spread=market_metrics.get("spread_bps", 10) / 10000,
                    order_book_depth=market_metrics.get("depth", 1000),
                    volume_ratio=market_metrics.get("volume", 1000) / max(market_metrics.get("avg_volume", 1000), 1),
                ),
                consecutive_wins=self._state.consecutive_wins,
                consecutive_losses=self._state.consecutive_losses,
                avg_profit=perf.avg_win if perf.winning_trades > 0 else 0.02,
                current_equity=10000 + self._state.total_pnl,
            )

        position_mult = risk_decision.position_size_multiplier
        if adaptation and hasattr(adaptation, 'position_multiplier'):
            position_mult *= adaptation.position_multiplier

        adapted_size = raw.position_size_pct * position_mult
        adapted_confidence = raw.confidence + risk_decision.confidence_adjustment
        adapted_confidence = max(0.1, min(1.0, adapted_confidence))

        base_stop_pct = 0.015
        if risk_decision.stop_loss_multiplier != 1.0:
            adjusted_stop_pct = base_stop_pct * risk_decision.stop_loss_multiplier
            if raw.direction == "buy":
                adapted_stop = raw.entry_price * (1 - adjusted_stop_pct)
            else:
                adapted_stop = raw.entry_price * (1 + adjusted_stop_pct)
        else:
            adapted_stop = raw.stop_loss

        base_tp_pct = 0.03
        adjusted_tp_pct = base_tp_pct * risk_decision.take_profit_multiplier
        if raw.direction == "buy":
            adjusted_tp = raw.entry_price * (1 + adjusted_tp_pct)
        else:
            adjusted_tp = raw.entry_price * (1 - adjusted_tp_pct)

        return IntegratedSignal(
            raw_signal=raw,
            adapted_direction=raw.direction,
            adapted_confidence=adapted_confidence,
            adapted_position_size=min(adapted_size, 0.20),
            adapted_stop_loss=adapted_stop,
            adapted_take_profit=adjusted_tp,
            risk=risk,
            risk_decision=risk_decision,
            adaptation=adaptation,
            should_execute=should_execute,
            execution_reason=reason if not should_execute else "OK",
        )

    def record_trade(self, symbol: str, direction: str, pnl: float) -> None:
        """Record trade execution for state tracking."""
        is_win = pnl > 0

        if symbol in self._state.symbols_traded:
            del self._state.symbols_traded[symbol]

        self._state.total_pnl += pnl
        self._state.daily_pnl += pnl
        self._total_trades += 1

        if is_win:
            self._state.winning_trades += 1
            self._state.consecutive_wins += 1
            self._state.consecutive_losses = 0
        else:
            self._state.losing_trades += 1
            self._state.consecutive_losses += 1
            self._state.consecutive_wins = 0

        self.strategy.record_trade(
            symbol,
            direction,
            100,
            100 * (1 + pnl),
            0.02,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        perf = self.strategy.get_performance()

        return {
            "strategy": {
                "total_trades": self._total_trades,
                "wins": self._state.winning_trades,
                "losses": self._state.losing_trades,
                "win_rate": perf.win_rate,
                "consecutive_wins": self._state.consecutive_wins,
                "consecutive_losses": self._state.consecutive_losses,
            },
            "risk": {
                "current_condition": self.risk_adapter._current_risk.condition,
            },
            "positions": list(self._state.symbols_traded.keys()),
            "pnl": self._state.total_pnl,
        }


def create_integration(
    strategy_min_confidence: float = 0.50,
    risk_base_stop_loss: float = 0.015,
) -> MarketFlowTradingIntegration:
    """Factory function to create integration."""
    return MarketFlowTradingIntegration(
        strategy_min_confidence=strategy_min_confidence,
        risk_base_stop_loss=risk_base_stop_loss,
    )


__all__ = [
    "MarketFlowTradingIntegration",
    "IntegratedSignal",
    "TradingState",
    "create_integration",
]