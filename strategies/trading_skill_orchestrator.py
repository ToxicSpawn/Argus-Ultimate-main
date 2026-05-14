"""
Trading Skill Orchestrator
==========================
Combines ALL trading knowledge into a unified trading brain.

This orchestrator:
1. Runs ALL strategies in parallel
2. Uses ensemble voting to select best signals
3. Applies ALL risk management techniques
4. Learns which strategies work best in which conditions
5. Adapts every 0.5 seconds

Every trade benefits from ALL trading knowledge.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import all trading systems
from strategies.universal_strategies import (
    StrategyRegistry,
    Signal,
    get_strategy_registry,
)
from risk.universal_risk_management import (
    UnifiedRiskManager,
    get_risk_manager,
)
from learning.market_adaptive_parameters import (
    MarketAdaptiveParameters,
    get_adaptive_params,
)
from learning.meta_learning_engine import (
    MetaLearningEngine,
    get_meta_engine,
)
from learning.ensemble_learning import (
    EnsembleLearningSystem,
    get_ensemble,
)
from learning.causal_learning import (
    CausalLearningSystem,
    get_causal_system,
)


@dataclass
class TradingDecision:
    """Final trading decision with full context."""
    action: str  # buy, sell, hold
    confidence: float
    position_size: float
    stop_loss: float
    take_profit: float
    contributing_strategies: List[str]
    ensemble_agreement: float
    risk_adjustments: Dict[str, float]
    market_regime: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingSkillOrchestrator:
    """
    The brain of Argus - combines ALL trading knowledge.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                Trading Skill Orchestrator                │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │              Signal Generation Layer             │   │
    │  │  • Universal Strategies (20+ strategies)        │   │
    │  │  • Trend Following (SMA, EMA, ADX, SAR)        │   │
    │  │  • Momentum (RSI, MACD, ROC, Stochastic)       │   │
    │  │  • Mean Reversion (Bollinger, Z-Score, RSI)    │   │
    │  │  • Breakout (Range, Volume)                    │   │
    │  │  • Scalping (Spread, Order Flow)               │   │
    │  │  • Statistical Arbitrage (Pairs, Cointegration)│   │
    │  │  • Market Making                               │   │
    │  │  • Volume Profile                              │   │
    │  └─────────────────────────────────────────────────┘   │
    │                         │                               │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │              Signal Filtering Layer              │   │
    │  │  • Ensemble Voting (weighted by performance)    │   │
    │  │  • Market Adaptive Thresholds (0.5s learning)  │   │
    │  │  • Meta-Learning (optimal learning rates)      │   │
    │  │  • Predictive Regime Detection                 │   │
    │  │  • Causal Learning (understands WHY)           │   │
    │  └─────────────────────────────────────────────────┘   │
    │                         │                               │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │              Risk Management Layer               │   │
    │  │  • Kelly Criterion (optimal sizing)            │   │
    │  │  • Value at Risk (downside protection)         │   │
    │  │  • CVaR (expected tail loss)                   │   │
    │  │  • Max Drawdown Control                        │   │
    │  │  • Volatility Targeting                        │   │
    │  │  • Correlation Risk                            │   │
    │  │  • Tail Risk Hedging                           │   │
    │  │  • Dynamic Stop Losses                         │   │
    │  │  • Risk Parity                                 │   │
    │  │  • Monte Carlo Simulation                      │   │
    │  └─────────────────────────────────────────────────┘   │
    │                         │                               │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │              Execution Layer                     │   │
    │  │  • Position Sizing                             │   │
    │  │  • Stop Loss Placement                         │   │
    │  │  • Take Profit Placement                       │   │
    │  │  • Trailing Stops                              │   │
    │  └─────────────────────────────────────────────────┘   │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, capital: float = 10000.0):
        self.capital = capital
        self.initial_capital = capital
        
        # Initialize all subsystems
        self.strategy_registry = get_strategy_registry()
        self.risk_manager = get_risk_manager(capital)
        self.adaptive_params = get_adaptive_params()
        self.meta_learning = get_meta_engine()
        self.ensemble = get_ensemble()
        self.causal_learning = get_causal_system()
        
        # State
        self.current_regime: str = "ranging"
        self.positions: Dict[str, Dict] = {}
        self.trade_history: Deque[Dict] = deque(maxlen=500)
        
        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.total_pnl: float = 0.0
        self.peak_equity: float = capital
        
        # Learning cycles
        self.learning_cycles: int = 0
        self.last_learning_time: float = time.time()
        
        logger.info(
            f"TradingSkillOrchestrator initialized: "
            f"capital=${capital:,.2f}, "
            f"strategies={len(self.strategy_registry.strategies)}"
        )
    
    def analyze_market(self,
                       prices: List[float],
                       regime: str,
                       market_data: Optional[Dict[str, Any]] = None) -> TradingDecision:
        """
        Analyze market and make trading decision.
        
        This is the main entry point - called every cycle.
        """
        start_time = time.time()
        
        self.current_regime = regime
        self.adaptive_params.update_regime(regime)
        
        # Update learning every 0.5 seconds
        if time.time() - self.last_learning_time >= 0.5:
            self._run_learning_cycle(prices, regime)
            self.last_learning_time = time.time()
        
        # Step 1: Get signals from ALL strategies
        all_signals = self.strategy_registry.get_all_signals(
            prices, 
            **(market_data or {})
        )
        
        if not all_signals:
            return self._create_hold_decision("No signals generated")
        
        # Step 2: Apply adaptive thresholds
        filtered_signals = self._apply_adaptive_filter(all_signals, regime)
        
        if not filtered_signals:
            return self._create_hold_decision("No signals passed filter")
        
        # Step 3: Ensemble voting
        best_signal = self._ensemble_vote(filtered_signals)
        
        if not best_signal:
            return self._create_hold_decision("No consensus from ensemble")
        
        # Step 4: Check risk rules
        can_trade, risk_reason = self.risk_manager.should_take_trade()
        if not can_trade:
            return self._create_hold_decision(f"Risk blocked: {risk_reason}")
        
        # Step 5: Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            signal_confidence=best_signal.confidence,
            signal_strength=best_signal.strength,
            assets=["BTCUSDT"]
        )
        
        # Step 6: Calculate stops
        atr = np.std(prices[-14:]) if len(prices) >= 14 else prices[-1] * 0.02
        stop_loss, take_profit = self._calculate_stops(
            prices[-1], best_signal.action, atr, best_signal.confidence
        )
        
        # Step 7: Create decision
        decision = TradingDecision(
            action=best_signal.action,
            confidence=best_signal.confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            contributing_strategies=[best_signal.strategy],
            ensemble_agreement=self._calculate_agreement(filtered_signals, best_signal.action),
            risk_adjustments={
                "kelly": self.risk_manager.kelly.calculate(),
                "vol_multiplier": self.risk_manager.vol_targeting.get_volatility_multiplier(),
                "dd_multiplier": self.risk_manager.drawdown_control.get_position_multiplier(),
            },
            market_regime=regime,
            timestamp=time.time(),
            metadata={
                "signal_reason": best_signal.reason,
                "num_signals": len(all_signals),
                "num_filtered": len(filtered_signals),
                "learning_time_ms": (time.time() - start_time) * 1000,
            }
        )
        
        return decision
    
    def _run_learning_cycle(self, prices: List[float], regime: str) -> None:
        """Run learning cycle for all learning systems."""
        self.learning_cycles += 1
        
        # Update adaptive parameters
        self.adaptive_params.learn()
        
        # Update meta-learning
        self.meta_learning.adapt_all_learning_rates()
    
    def _apply_adaptive_filter(self, 
                                signals: List[Signal], 
                                regime: str) -> List[Signal]:
        """Apply adaptive filter thresholds."""
        filter_threshold = self.adaptive_params.get_filter_threshold(regime)
        
        filtered = []
        for signal in signals:
            # Get confidence floor for this signal type
            signal_type = signal.strategy.split("_")[0].lower()
            confidence_floor = self.adaptive_params.get_confidence_floor(signal_type)
            
            # Adjust confidence
            adjusted_confidence = max(signal.confidence, confidence_floor)
            
            if adjusted_confidence >= filter_threshold:
                signal.confidence = adjusted_confidence
                filtered.append(signal)
                self.adaptive_params.record_signal(
                    {"confidence": adjusted_confidence, "strategy": signal.strategy},
                    passed=True
                )
            else:
                self.adaptive_params.record_signal(
                    {"confidence": adjusted_confidence, "strategy": signal.strategy},
                    passed=False
                )
        
        return filtered
    
    def _ensemble_vote(self, signals: List[Signal]) -> Optional[Signal]:
        """Use ensemble voting to select best signal."""
        if not signals:
            return None
        
        # Group by action
        buy_signals = [s for s in signals if s.action == "buy"]
        sell_signals = [s for s in signals if s.action == "sell"]
        
        # Calculate weighted scores
        buy_score = sum(s.confidence * s.strength for s in buy_signals) if buy_signals else 0
        sell_score = sum(s.confidence * s.strength for s in sell_signals) if sell_signals else 0
        
        # Select winner
        if buy_score > sell_score and buy_signals:
            # Return highest confidence buy signal
            return max(buy_signals, key=lambda s: s.confidence * s.strength)
        elif sell_score > buy_score and sell_signals:
            return max(sell_signals, key=lambda s: s.confidence * s.strength)
        
        return None
    
    def _calculate_agreement(self, signals: List[Signal], action: str) -> float:
        """Calculate agreement level among signals."""
        if not signals:
            return 0.0
        
        agreeing = sum(1 for s in signals if s.action == action)
        return agreeing / len(signals)
    
    def _calculate_stops(self,
                         current_price: float,
                         action: str,
                         atr: float,
                         confidence: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit."""
        # ATR-based stops
        atr_multiplier = 2.0 - confidence  # Higher confidence = tighter stops
        atr_multiplier = np.clip(atr_multiplier, 1.0, 3.0)
        
        stop_distance = atr * atr_multiplier
        risk_reward = 2.0 + confidence  # Higher confidence = better R:R
        
        if action == "buy":
            stop_loss = current_price - stop_distance
            take_profit = current_price + stop_distance * risk_reward
        else:
            stop_loss = current_price + stop_distance
            take_profit = current_price - stop_distance * risk_reward
        
        return stop_loss, take_profit
    
    def _create_hold_decision(self, reason: str) -> TradingDecision:
        """Create a hold decision."""
        return TradingDecision(
            action="hold",
            confidence=0.0,
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            contributing_strategies=[],
            ensemble_agreement=0.0,
            risk_adjustments={},
            market_regime=self.current_regime,
            timestamp=time.time(),
            metadata={"reason": reason}
        )
    
    def record_trade_outcome(self,
                              decision: TradingDecision,
                              exit_price: float,
                              entry_price: float) -> None:
        """Record trade outcome for learning."""
        # Calculate PnL
        if decision.action == "buy":
            pnl = (exit_price - entry_price) * (decision.position_size / entry_price)
        else:
            pnl = (entry_price - exit_price) * (decision.position_size / entry_price)
        
        # Update tracking
        self.total_trades += 1
        self.total_pnl += pnl
        self.capital += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        
        if self.capital > self.peak_equity:
            self.peak_equity = self.capital
        
        # Record in all learning systems
        trade_record = {
            "pnl": pnl,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "side": decision.action,
            "position_value": decision.position_size,
            "timestamp": time.time(),
        }
        
        self.trade_history.append(trade_record)
        self.risk_manager.record_trade(
            pnl=pnl,
            entry_price=entry_price,
            exit_price=exit_price,
            side=decision.action,
            position_value=decision.position_size
        )
        
        # Record in adaptive params
        self.adaptive_params.record_trade(trade_record)
        
        # Record in causal learning
        self.causal_learning.record_trade(
            parameters={"position_size": decision.position_size},
            market_features={"regime": hash(self.current_regime) % 100 / 100},
            outcome=pnl
        )
        
        # Update strategy performance
        for strategy_name in decision.contributing_strategies:
            if strategy_name in self.strategy_registry.strategies:
                self.strategy_registry.strategies[strategy_name].record_outcome(pnl)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        # Calculate Sharpe (simplified)
        if len(self.trade_history) >= 10:
            pnls = [t["pnl"] for t in self.trade_history]
            sharpe = np.mean(pnls) / max(np.std(pnls), 0.001) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Calculate profit factor
        profits = [t["pnl"] for t in self.trade_history if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in self.trade_history if t["pnl"] < 0]
        profit_factor = sum(profits) / max(sum(losses), 0.001)
        
        # Calculate max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        running_capital = self.initial_capital
        
        for trade in self.trade_history:
            running_capital += trade["pnl"]
            if running_capital > peak:
                peak = running_capital
            dd = (peak - running_capital) / peak
            max_dd = max(max_dd, dd)
        
        return {
            "capital": self.capital,
            "total_return": (self.capital - self.initial_capital) / self.initial_capital,
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "avg_win": np.mean(profits) if profits else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "learning_cycles": self.learning_cycles,
            "current_regime": self.current_regime,
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get stats from all subsystems."""
        return {
            "performance": self.get_performance_stats(),
            "strategies": self.strategy_registry.get_strategy_stats(),
            "risk": self.risk_manager.get_risk_report(),
            "adaptive_params": self.adaptive_params.get_stats(),
            "meta_learning": self.meta_learning.get_learning_stats(),
            "ensemble": self.ensemble.get_stats(),
            "causal": self.causal_learning.get_causal_insights(),
        }


# Singleton
_orchestrator: Optional[TradingSkillOrchestrator] = None


def get_trading_orchestrator(capital: float = 10000.0) -> TradingSkillOrchestrator:
    """Get or create singleton trading orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TradingSkillOrchestrator(capital)
    return _orchestrator


def reset_trading_orchestrator() -> None:
    """Reset singleton."""
    global _orchestrator
    _orchestrator = None
