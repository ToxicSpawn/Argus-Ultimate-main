# pyright: reportMissingImports=false
"""
Maximum Edge Orchestrator
=========================
Coordinates ALL alpha sources for maximum risk-adjusted returns.

This is the "brain" that decides which strategies to run, how to allocate
capital, and when to increase/decrease exposure based on market conditions.

Strategy Hierarchy (risk-adjusted priority):
1. Funding Rate Arbitrage (10-30% APR, near risk-free) - PRIMARY
2. Cross-Exchange Arbitrage (5-15% APR, very low risk)
3. Market Making (15-25% APR, medium risk)
4. ML Price Prediction (Sharpe 1.5-2.5, medium risk)
5. Order Flow Alpha (20-50 bps, low risk)
6. Whale Tracking (20-40 bps, low risk)
7. Volatility Trading (Sharpe 1.0-3.0, medium risk)
8. Liquidation Hunting (50-200 bps, medium risk)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class StrategyTier(Enum):
    """Strategy tiers based on risk-adjusted returns."""
    GUARANTEED = auto()      # Risk-free income (funding arb, rebates)
    HIGH_EDGE = auto()       # High alpha, medium risk (ML, order flow)
    MARKET_MAKING = auto()   # Inventory profits (spread capture)
    VOLATILITY = auto()      # Vol trading, options
    DIRECTIONAL = auto()     # Trend following, momentum


@dataclass
class StrategyAllocation:
    """Capital allocation for a strategy."""
    strategy_name: str
    tier: StrategyTier
    allocated_pct: float        # % of capital allocated
    max_pct: float              # Maximum allocation
    current_edge: float         # Realized edge (bps)
    expected_edge: float        # Expected edge (bps)
    sharpe_ratio: float         # Realized Sharpe
    win_rate: float             # Realized win rate
    is_active: bool = True
    priority: int = 1           # 1 = highest priority


@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime: str                 # "trending", "ranging", "volatile", "crisis"
    volatility: float           # Realized volatility
    trend_strength: float       # Trend strength (-1 to 1)
    funding_environment: str    # "positive", "negative", "mixed"
    liquidity_score: float      # 0-1 liquidity health
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EdgeReport:
    """Comprehensive edge report across all strategies."""
    timestamp: datetime
    total_expected_edge_bps: float
    total_realized_edge_bps: float
    active_strategies: int
    capital_utilization_pct: float
    strategy_allocations: Dict[str, StrategyAllocation]
    market_regime: MarketRegime
    recommendations: List[str]


class MaximumEdgeOrchestrator:
    """
    The master orchestrator that coordinates ALL strategies for maximum
    risk-adjusted returns.
    
    Key responsibilities:
    1. Allocate capital across strategies based on risk-adjusted returns
    2. Dynamically rebalance as market conditions change
    3. Enable/disable strategies based on regime detection
    4. Optimize position sizing using continuous learning feedback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Maximum Edge Orchestrator."""
        self.config = config or {}
        
        # Strategy allocations
        self.allocations: Dict[str, StrategyAllocation] = {}
        
        # Market regime tracking
        self.current_regime: Optional[MarketRegime] = None
        
        # Performance tracking
        self.total_realized_pnl: float = 0.0
        self.total_trades: int = 0
        self.trade_history: List[Dict[str, Any]] = []
        
        # Learning integration
        self.learning_system: Optional[Any] = None
        
        # Initialize default strategy allocations
        self._initialize_strategies()
        
        logger.info("Maximum Edge Orchestrator initialized with %d strategies", 
                    len(self.allocations))
    
    def _initialize_strategies(self) -> None:
        """Initialize strategy allocations with default weights."""
        
        # TIER 1: GUARANTEED INCOME (Lowest risk)
        self.allocations["funding_rate_arb"] = StrategyAllocation(
            strategy_name="Funding Rate Arbitrage",
            tier=StrategyTier.GUARANTEED,
            allocated_pct=30.0,     # 30% of capital
            max_pct=50.0,
            current_edge=0.0,
            expected_edge=150.0,    # ~15 bps per period = 150 bps/month
            sharpe_ratio=3.0,       # Very stable
            win_rate=0.90,
            priority=1
        )
        
        self.allocations["cross_exchange_arb"] = StrategyAllocation(
            strategy_name="Cross-Exchange Arbitrage",
            tier=StrategyTier.GUARANTEED,
            allocated_pct=15.0,
            max_pct=25.0,
            current_edge=0.0,
            expected_edge=80.0,
            sharpe_ratio=2.5,
            win_rate=0.85,
            priority=2
        )
        
        self.allocations["maker_rebates"] = StrategyAllocation(
            strategy_name="Maker Rebates",
            tier=StrategyTier.GUARANTEED,
            allocated_pct=10.0,
            max_pct=20.0,
            current_edge=0.0,
            expected_edge=40.0,
            sharpe_ratio=3.5,
            win_rate=0.95,
            priority=3
        )
        
        # TIER 2: HIGH EDGE (Medium risk)
        self.allocations["ml_prediction"] = StrategyAllocation(
            strategy_name="ML Price Prediction",
            tier=StrategyTier.HIGH_EDGE,
            allocated_pct=15.0,
            max_pct=25.0,
            current_edge=0.0,
            expected_edge=60.0,
            sharpe_ratio=1.8,
            win_rate=0.58,
            priority=4
        )
        
        self.allocations["order_flow"] = StrategyAllocation(
            strategy_name="Order Flow Alpha",
            tier=StrategyTier.HIGH_EDGE,
            allocated_pct=10.0,
            max_pct=20.0,
            current_edge=0.0,
            expected_edge=35.0,
            sharpe_ratio=2.0,
            win_rate=0.55,
            priority=5
        )
        
        self.allocations["whale_tracking"] = StrategyAllocation(
            strategy_name="Whale Tracking",
            tier=StrategyTier.HIGH_EDGE,
            allocated_pct=5.0,
            max_pct=15.0,
            current_edge=0.0,
            expected_edge=30.0,
            sharpe_ratio=1.5,
            win_rate=0.60,
            priority=6
        )
        
        # TIER 3: MARKET MAKING
        self.allocations["market_making"] = StrategyAllocation(
            strategy_name="Market Making",
            tier=StrategyTier.MARKET_MAKING,
            allocated_pct=10.0,
            max_pct=20.0,
            current_edge=0.0,
            expected_edge=100.0,
            sharpe_ratio=1.5,
            win_rate=0.52,
            priority=7
        )
        
        # TIER 4: VOLATILITY
        self.allocations["volatility_arb"] = StrategyAllocation(
            strategy_name="Volatility Arbitrage",
            tier=StrategyTier.VOLATILITY,
            allocated_pct=5.0,
            max_pct=15.0,
            current_edge=0.0,
            expected_edge=50.0,
            sharpe_ratio=1.2,
            win_rate=0.55,
            priority=8
        )
    
    def update_market_regime(self, regime: MarketRegime) -> List[str]:
        """
        Update market regime and adjust strategy allocations.
        Returns list of actions taken.
        """
        self.current_regime = regime
        actions = []
        
        # Adjust allocations based on regime
        if regime.regime == "crisis":
            # Crisis mode: maximize guaranteed income, minimize directional
            actions.extend(self._adjust_for_crisis())
        
        elif regime.regime == "volatile":
            # High volatility: increase vol strategies, decrease market making
            actions.extend(self._adjust_for_volatility())
        
        elif regime.regime == "trending":
            # Trending: enable directional strategies, increase ML
            actions.extend(self._adjust_for_trending())
        
        elif regime.regime == "ranging":
            # Ranging: increase market making, decrease directional
            actions.extend(self._adjust_for_ranging())
        
        # Apply funding environment adjustments
        if regime.funding_environment == "positive":
            actions.extend(self._boost_funding_arb())
        elif regime.funding_environment == "negative":
            actions.extend(self._reduce_funding_arb())
        
        logger.info("Market regime updated to %s. Actions: %s", 
                    regime.regime, actions)
        return actions
    
    def _adjust_for_crisis(self) -> List[str]:
        """Adjust allocations for crisis mode."""
        actions = []
        
        # Boost guaranteed strategies
        for name, alloc in self.allocations.items():
            if alloc.tier == StrategyTier.GUARANTEED:
                new_pct = min(alloc.max_pct, alloc.allocated_pct * 1.5)
                if new_pct != alloc.allocated_pct:
                    alloc.allocated_pct = new_pct
                    actions.append(f"BOOST {name}: {alloc.allocated_pct:.1f}%")
        
        # Reduce directional strategies
        for name, alloc in self.allocations.items():
            if alloc.tier in [StrategyTier.DIRECTIONAL, StrategyTier.VOLATILITY]:
                new_pct = alloc.allocated_pct * 0.5
                alloc.allocated_pct = max(5.0, new_pct)
                actions.append(f"REDUCE {name}: {alloc.allocated_pct:.1f}%")
        
        return actions
    
    def _adjust_for_volatility(self) -> List[str]:
        """Adjust for high volatility."""
        actions = []
        
        # Boost volatility strategies
        vol_alloc = self.allocations.get("volatility_arb")
        if vol_alloc:
            vol_alloc.allocated_pct = min(vol_alloc.max_pct, vol_alloc.allocated_pct * 1.3)
            actions.append(f"BOOST volatility_arb: {vol_alloc.allocated_pct:.1f}%")
        
        # Reduce market making (adverse selection risk)
        mm_alloc = self.allocations.get("market_making")
        if mm_alloc:
            mm_alloc.allocated_pct = max(5.0, mm_alloc.allocated_pct * 0.7)
            actions.append(f"REDUCE market_making: {mm_alloc.allocated_pct:.1f}%")
        
        return actions
    
    def _adjust_for_trending(self) -> List[str]:
        """Adjust for trending market."""
        actions = []
        
        # Boost ML prediction (trends are predictable)
        ml_alloc = self.allocations.get("ml_prediction")
        if ml_alloc:
            ml_alloc.allocated_pct = min(ml_alloc.max_pct, ml_alloc.allocated_pct * 1.2)
            actions.append(f"BOOST ml_prediction: {ml_alloc.allocated_pct:.1f}%")
        
        # Enable whale tracking (whales follow trends)
        whale_alloc = self.allocations.get("whale_tracking")
        if whale_alloc:
            whale_alloc.allocated_pct = min(whale_alloc.max_pct, whale_alloc.allocated_pct * 1.2)
            actions.append(f"BOOST whale_tracking: {whale_alloc.allocated_pct:.1f}%")
        
        return actions
    
    def _adjust_for_ranging(self) -> List[str]:
        """Adjust for ranging/mean-reverting market."""
        actions = []
        
        # Boost market making (good for ranges)
        mm_alloc = self.allocations.get("market_making")
        if mm_alloc:
            mm_alloc.allocated_pct = min(mm_alloc.max_pct, mm_alloc.allocated_pct * 1.3)
            actions.append(f"BOOST market_making: {mm_alloc.allocated_pct:.1f}%")
        
        # Reduce directional
        ml_alloc = self.allocations.get("ml_prediction")
        if ml_alloc:
            ml_alloc.allocated_pct = max(5.0, ml_alloc.allocated_pct * 0.8)
            actions.append(f"REDUCE ml_prediction: {ml_alloc.allocated_pct:.1f}%")
        
        return actions
    
    def _boost_funding_arb(self) -> List[str]:
        """Boost funding arb when rates are positive."""
        actions = []
        funding_alloc = self.allocations.get("funding_rate_arb")
        if funding_alloc and funding_alloc.allocated_pct < funding_alloc.max_pct:
            funding_alloc.allocated_pct = min(
                funding_alloc.max_pct,
                funding_alloc.allocated_pct * 1.1
            )
            actions.append(f"BOOST funding_rate_arb: {funding_alloc.allocated_pct:.1f}%")
        return actions
    
    def _reduce_funding_arb(self) -> List[str]:
        """Reduce funding arb when rates turn negative."""
        actions = []
        funding_alloc = self.allocations.get("funding_rate_arb")
        if funding_alloc:
            funding_alloc.allocated_pct = max(10.0, funding_alloc.allocated_pct * 0.8)
            actions.append(f"REDUCE funding_rate_arb: {funding_alloc.allocated_pct:.1f}%")
        return actions
    
    def record_trade_result(
        self,
        strategy_name: str,
        pnl: float,
        edge_bps: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a trade result for learning and rebalancing."""
        
        if strategy_name in self.allocations:
            alloc = self.allocations[strategy_name]
            
            # Update rolling metrics (simplified EMA)
            alpha = 0.1  # Learning rate
            alloc.current_edge = alloc.current_edge * (1 - alpha) + edge_bps * alpha
            
            # Update win rate
            is_win = 1.0 if pnl > 0 else 0.0
            alloc.win_rate = alloc.win_rate * (1 - alpha) + is_win * alpha
        
        # Record in history
        self.total_realized_pnl += pnl
        self.total_trades += 1
        self.trade_history.append({
            "timestamp": datetime.now(),
            "strategy": strategy_name,
            "pnl": pnl,
            "edge_bps": edge_bps,
            "metadata": metadata or {}
        })
        
        # Trigger rebalancing every 50 trades
        if self.total_trades % 50 == 0:
            self._rebalance_allocations()
    
    def _rebalance_allocations(self) -> None:
        """Rebalance allocations based on realized performance."""
        logger.info("Rebalancing strategy allocations based on %d trades", 
                    self.total_trades)
        
        # Calculate performance scores
        scores: Dict[str, float] = {}
        for name, alloc in self.allocations.items():
            # Score = Sharpe * Win Rate * Current Edge
            scores[name] = alloc.sharpe_ratio * alloc.win_rate * max(0.1, alloc.current_edge / alloc.expected_edge)
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            for name in scores:
                scores[name] /= total_score
        
        # Reallocate based on scores (with constraints)
        for name, alloc in self.allocations.items():
            target_pct = scores.get(name, 0.0) * 100.0
            
            # Apply min/max constraints
            target_pct = max(5.0, min(alloc.max_pct, target_pct))
            
            # Smooth adjustment (don't change too fast)
            alloc.allocated_pct = alloc.allocated_pct * 0.7 + target_pct * 0.3
        
        # Normalize to 100%
        total_pct = sum(a.allocated_pct for a in self.allocations.values())
        if total_pct > 0:
            scale = 100.0 / total_pct
            for alloc in self.allocations.values():
                alloc.allocated_pct *= scale
    
    def get_edge_report(self) -> EdgeReport:
        """Generate comprehensive edge report."""
        
        # Calculate totals
        total_expected = sum(
            a.expected_edge * (a.allocated_pct / 100.0) 
            for a in self.allocations.values() if a.is_active
        )
        total_realized = sum(
            a.current_edge * (a.allocated_pct / 100.0) 
            for a in self.allocations.values() if a.is_active
        )
        active_count = sum(1 for a in self.allocations.values() if a.is_active)
        capital_util = sum(a.allocated_pct for a in self.allocations.values())
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return EdgeReport(
            timestamp=datetime.now(),
            total_expected_edge_bps=total_expected,
            total_realized_edge_bps=total_realized,
            active_strategies=active_count,
            capital_utilization_pct=capital_util,
            strategy_allocations=dict(self.allocations),
            market_regime=self.current_regime or MarketRegime(
                regime="unknown", volatility=0.0, trend_strength=0.0,
                funding_environment="mixed", liquidity_score=0.5
            ),
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check for underperforming strategies
        for name, alloc in self.allocations.items():
            if alloc.is_active and alloc.win_rate < 0.45:
                recommendations.append(
                    f"⚠️ {name} win rate below 45% ({alloc.win_rate:.1%}). Consider reducing allocation."
                )
            
            if alloc.current_edge < alloc.expected_edge * 0.5:
                recommendations.append(
                    f"📉 {name} edge below 50% of expected. Review entry criteria."
                )
        
        # Check capital utilization
        total_pct = sum(a.allocated_pct for a in self.allocations.values())
        if total_pct < 80:
            recommendations.append(
                f"💰 Capital utilization at {total_pct:.1f}%. Consider deploying more capital."
            )
        
        # Check funding environment
        if self.current_regime and self.current_regime.funding_environment == "positive":
            recommendations.append(
                "📈 Positive funding environment. Prioritize funding rate arbitrage."
            )
        
        return recommendations
    
    def get_strategy_for_signal(self, signal_type: str) -> Optional[str]:
        """
        Given a signal type, return the best strategy to execute it.
        Used by the unified trading system.
        """
        signal_to_strategy = {
            "funding_rate": "funding_rate_arb",
            "arbitrage": "cross_exchange_arb",
            "ml_directional": "ml_prediction",
            "order_flow": "order_flow",
            "whale": "whale_tracking",
            "market_making": "market_making",
            "volatility": "volatility_arb",
        }
        return signal_to_strategy.get(signal_type)


# Singleton instance
_orchestrator: Optional[MaximumEdgeOrchestrator] = None


def get_maximum_edge_orchestrator(
    config: Optional[Dict[str, Any]] = None
) -> MaximumEdgeOrchestrator:
    """Get or create the Maximum Edge Orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MaximumEdgeOrchestrator(config)
    return _orchestrator


__all__ = [
    "MaximumEdgeOrchestrator",
    "StrategyAllocation",
    "StrategyTier",
    "MarketRegime",
    "EdgeReport",
    "get_maximum_edge_orchestrator",
]
