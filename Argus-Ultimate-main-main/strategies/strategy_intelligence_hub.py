# pyright: reportMissingImports=false
"""
Ultimate Strategy Intelligence Hub
====================================
Master coordinator for ALL trading strategies.

This system:
1. CATEGORIZES all 99+ strategies by type and regime suitability
2. TRACKS performance metrics for each strategy in real-time
3. ENABLES Champion-Challenger testing for continuous improvement
4. DETECTS strategy decay and auto-rotates
5. OPTIMIZES parameters using continuous learning feedback
6. MAPS strategies to market regimes for optimal allocation

Strategy Categories:
- ARB: Arbitrage (funding, cross-exchange, DEX-CEX, cross-chain)
- MM: Market Making (Avellaneda-Stoikov, grid, micro-capital)
- MOMENTUM: Trend following, breakout, momentum
- MEAN_REV: Mean reversion, pairs trading, statistical arb
- VOLATILITY: Vol arb, options, gamma scalping
- ML: Machine learning based strategies
- SPECIAL: Liquidation hunting, MEV, oracle deviation
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StrategyCategory(Enum):
    """High-level strategy categories."""
    ARBITRAGE = auto()         # Risk-free or low-risk arbitrage
    MARKET_MAKING = auto()     # Spread capture, liquidity provision
    MOMENTUM = auto()          # Trend following, breakout
    MEAN_REVERSION = auto()    # Pairs trading, statistical arb
    VOLATILITY = auto()        # Vol trading, options
    ML_PREDICTIVE = auto()     # ML-based prediction
    SPECIAL = auto()           # Unique alpha sources


class MarketRegime(Enum):
    """Market regimes for strategy mapping."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    CRISIS = auto()
    RECOVERY = auto()


class StrategyStatus(Enum):
    """Strategy operational status."""
    ACTIVE = auto()            # Running normally
    CHAMPION = auto()          # Best performing in category
    CHALLENGER = auto()        # Being tested against champion
    COOLDOWN = auto()          # Temporarily disabled (poor performance)
    RETIRED = auto()           # Permanently disabled
    TESTING = auto()           # Initial testing phase


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""
    strategy_name: str
    category: StrategyCategory
    
    # Core metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # PnL
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Timing
    avg_hold_time_minutes: float = 0.0
    avg_time_between_trades: float = 0.0
    
    # Edge
    avg_edge_bps: float = 0.0
    realized_vs_expected: float = 1.0  # Ratio of realized to expected edge
    
    # Decay detection
    recent_win_rate: float = 0.0       # Last 20 trades
    historical_win_rate: float = 0.0   # All trades
    decay_score: float = 0.0           # 0-1, higher = more decay
    
    # Metadata
    last_trade_time: Optional[datetime] = None
    status: StrategyStatus = StrategyStatus.ACTIVE
    regime_scores: Dict[str, float] = field(default_factory=dict)  # Regime -> score
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def is_profitable(self) -> bool:
        return self.total_pnl > 0
    
    @property
    def kelly_fraction(self) -> float:
        """Kelly criterion optimal fraction."""
        if self.win_rate == 0 or self.profit_factor == 0:
            return 0.0
        win_loss_ratio = self.profit_factor
        kelly = self.win_rate - ((1 - self.win_rate) / win_loss_ratio)
        return max(0.0, min(kelly, 0.25))  # Cap at 25% for safety


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    name: str
    category: StrategyCategory
    enabled: bool = True
    base_position_pct: float = 5.0      # % of capital per trade
    max_position_pct: float = 10.0
    min_confidence: float = 0.5
    applicable_regimes: List[MarketRegime] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Adaptive parameters (updated by learning)
    learning_rate: float = 0.01
    exploration_rate: float = 0.1       # Chance of trying new parameters


@dataclass
class ChampionChallengerResult:
    """Result of A/B test between champion and challenger."""
    champion_name: str
    challenger_name: str
    category: StrategyCategory
    
    # Test results
    champion_trades: int
    challenger_trades: int
    champion_pnl: float
    challenger_pnl: float
    champion_sharpe: float
    challenger_sharpe: float
    
    # Statistical significance
    p_value: float
    is_significant: bool
    
    # Decision
    should_swap: bool
    confidence: float
    reason: str


class StrategyIntelligenceHub:
    """
    Master coordinator for ALL trading strategies.
    
    This is the "brain" that decides:
    1. Which strategies to run
    2. How much capital to allocate to each
    3. When to swap champion/challenger
    4. When to retire underperforming strategies
    5. How to optimize parameters continuously
    """
    
    # Decay detection thresholds
    DECAY_WARNING_THRESHOLD = 0.3      # Start monitoring
    DECAY_CRITICAL_THRESHOLD = 0.6     # Move to cooldown
    DECAY_RETIRE_THRESHOLD = 0.8       # Retire strategy
    
    # Champion-challenger settings
    MIN_TRADES_FOR_COMPARISON = 20
    SIGNIFICANCE_LEVEL = 0.05
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Strategy Intelligence Hub."""
        self.config = config or {}
        
        # Strategy registry
        self.strategies: Dict[str, StrategyConfig] = {}
        self.metrics: Dict[str, StrategyMetrics] = {}
        
        # Performance history
        self.trade_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # Champion-challenger tracking
        self.champion_challenger: Dict[StrategyCategory, Tuple[str, str]] = {}
        self.test_results: List[ChampionChallengerResult] = []
        
        # Current market regime
        self.current_regime: Optional[MarketRegime] = None
        
        # Capital allocation
        self.total_capital: float = 1000.0
        self.allocations: Dict[str, float] = {}  # strategy -> % of capital
        
        # Initialize with all known strategies
        self._register_default_strategies()
        
        logger.info("Strategy Intelligence Hub initialized with %d strategies",
                    len(self.strategies))
    
    def _register_default_strategies(self) -> None:
        """Register all known strategies from the codebase."""
        
        # ARBITRAGE STRATEGIES
        arb_strategies = [
            ("funding_rate_arb", "Funding Rate Arbitrage", 30.0),
            ("cross_exchange_arb", "Cross-Exchange Arbitrage", 15.0),
            ("dex_cex_arb", "DEX-CEX Arbitrage", 10.0),
            ("cross_chain_bridge_arb", "Cross-Chain Bridge Arb", 8.0),
            ("triangular_arbitrage", "Triangular Arbitrage", 5.0),
            ("delta_neutral_perp_arb", "Delta Neutral Perp Arb", 12.0),
            ("futures_basis_arb", "Futures Basis Arbitrage", 10.0),
            ("flash_loan_arb", "Flash Loan Arbitrage", 5.0),
        ]
        
        for name, display, base_pos in arb_strategies:
            self.register_strategy(StrategyConfig(
                name=name,
                category=StrategyCategory.ARBITRAGE,
                base_position_pct=base_pos,
                applicable_regimes=[
                    MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN,
                    MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY
                ]
            ))
        
        # MARKET MAKING STRATEGIES
        mm_strategies = [
            ("market_maker", "Basic Market Making", 10.0),
            ("market_maker_avellaneda", "Avellaneda-Stoikov MM", 15.0),
            ("grid_trader", "Grid Trading", 12.0),
            ("grid_mean_reversion", "Grid Mean Reversion", 10.0),
            ("micro_capital_mm", "Micro Capital MM", 8.0),
        ]
        
        for name, display, base_pos in mm_strategies:
            self.register_strategy(StrategyConfig(
                name=name,
                category=StrategyCategory.MARKET_MAKING,
                base_position_pct=base_pos,
                applicable_regimes=[
                    MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY
                ]
            ))
        
        # MOMENTUM STRATEGIES
        momentum_strategies = [
            ("momentum", "Basic Momentum", 8.0),
            ("trend_following", "Trend Following", 10.0),
            ("breakout", "Breakout Trading", 8.0),
            ("volatility_breakout", "Volatility Breakout", 7.0),
            ("aggressive_scalper", "Aggressive Scalper", 5.0),
            ("sol_momentum_scalper", "SOL Momentum Scalper", 5.0),
        ]
        
        for name, display, base_pos in momentum_strategies:
            self.register_strategy(StrategyConfig(
                name=name,
                category=StrategyCategory.MOMENTUM,
                base_position_pct=base_pos,
                applicable_regimes=[
                    MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN,
                    MarketRegime.RECOVERY
                ]
            ))
        
        # MEAN REVERSION STRATEGIES
        mr_strategies = [
            ("mean_reversion", "Basic Mean Reversion", 8.0),
            ("pairs_trading", "Pairs Trading", 10.0),
            ("stat_arb", "Statistical Arbitrage", 8.0),
            ("stat_arb_cointegration", "Cointegration Arb", 8.0),
            ("kalman_pairs", "Kalman Filter Pairs", 7.0),
            ("bb_squeeze", "Bollinger Squeeze", 6.0),
            ("vwap_reversion", "VWAP Reversion", 7.0),
        ]
        
        for name, display, base_pos in mr_strategies:
            self.register_strategy(StrategyConfig(
                name=name,
                category=StrategyCategory.MEAN_REVERSION,
                base_position_pct=base_pos,
                applicable_regimes=[MarketRegime.RANGING]
            ))
        
        # VOLATILITY STRATEGIES
        vol_strategies = [
            ("volatility_arb", "Volatility Arbitrage", 10.0),
            ("options_vol_arb", "Options Vol Arb", 8.0),
            ("deribit_options", "Deribit Options", 7.0),
        ]
        
        for name, display, base_pos in vol_strategies:
            self.register_strategy(StrategyConfig(
                name=name,
                category=StrategyCategory.VOLATILITY,
                base_position_pct=base_pos,
                applicable_regimes=[
                    MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS
                ]
            ))
        
        # ML STRATEGIES
        ml_strategies = [
            ("ml_ensemble", "ML Ensemble", 12.0),
            ("online_learner", "Online Learner", 8.0),
            ("meta_learner", "Meta Learner", 7.0),
            ("bandit_router", "Bandit Router", 10.0),
        ]
        
        for name, display, base_pos in ml_strategies:
            self.register_strategy(StrategyConfig(
                name=name,
                category=StrategyCategory.ML_PREDICTIVE,
                base_position_pct=base_pos,
                applicable_regimes=[
                    MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN,
                    MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY
                ]
            ))
        
        # SPECIAL STRATEGIES
        special_strategies = [
            ("liquidation_cascade", "Liquidation Cascade Hunter", 10.0),
            ("mev_sandwich", "MEV Sandwich", 5.0),
            ("oracle_deviation", "Oracle Deviation", 7.0),
            ("seasonal_patterns", "Seasonal Patterns", 6.0),
            ("session_effect", "Session Effect", 5.0),
        ]
        
        for name, display, base_pos in special_strategies:
            self.register_strategy(StrategyConfig(
                name=name,
                category=StrategyCategory.SPECIAL,
                base_position_pct=base_pos,
                applicable_regimes=[
                    MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS,
                    MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN
                ]
            ))
    
    def register_strategy(self, config: StrategyConfig) -> None:
        """Register a new strategy."""
        self.strategies[config.name] = config
        self.metrics[config.name] = StrategyMetrics(
            strategy_name=config.name,
            category=config.category
        )
        self.allocations[config.name] = 0.0
    
    def record_trade(
        self,
        strategy_name: str,
        pnl: float,
        edge_bps: float,
        hold_time_minutes: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a trade for a strategy."""
        if strategy_name not in self.metrics:
            logger.warning("Strategy %s not registered", strategy_name)
            return
        
        metrics = self.metrics[strategy_name]
        metrics.total_trades += 1
        metrics.total_pnl += pnl
        metrics.last_trade_time = datetime.now()
        
        if pnl > 0:
            metrics.winning_trades += 1
            metrics.gross_profit += pnl
        else:
            metrics.losing_trades += 1
            metrics.gross_loss += abs(pnl)
        
        # Update drawdown
        if metrics.total_pnl < 0:
            metrics.current_drawdown = abs(metrics.total_pnl)
            metrics.max_drawdown = max(metrics.max_drawdown, metrics.current_drawdown)
        else:
            metrics.current_drawdown = 0.0
        
        # Update edge
        metrics.avg_edge_bps = (
            metrics.avg_edge_bps * (metrics.total_trades - 1) + edge_bps
        ) / metrics.total_trades
        
        # Update hold time
        metrics.avg_hold_time_minutes = (
            metrics.avg_hold_time_minutes * (metrics.total_trades - 1) + hold_time_minutes
        ) / metrics.total_trades
        
        # Update ratios
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        
        if metrics.total_trades > 0:
            metrics.expectancy = metrics.total_pnl / metrics.total_trades
        
        # Record in history
        self.trade_history[strategy_name].append({
            "timestamp": datetime.now(),
            "pnl": pnl,
            "edge_bps": edge_bps,
            "hold_time": hold_time_minutes,
            "metadata": metadata or {}
        })
        
        # Update decay detection
        self._update_decay_detection(strategy_name)
        
        # Update Sharpe (simplified)
        self._update_risk_metrics(strategy_name)
    
    def _update_decay_detection(self, strategy_name: str) -> None:
        """Update strategy decay detection metrics."""
        metrics = self.metrics[strategy_name]
        history = list(self.trade_history[strategy_name])
        
        if len(history) < 20:
            return
        
        # Recent win rate (last 20 trades)
        recent = history[-20:]
        recent_wins = sum(1 for h in recent if h["pnl"] > 0)
        metrics.recent_win_rate = recent_wins / len(recent)
        
        # Historical win rate
        metrics.historical_win_rate = metrics.win_rate
        
        # Decay score: difference between recent and historical
        if metrics.historical_win_rate > 0:
            decay = max(0, metrics.historical_win_rate - metrics.recent_win_rate)
            metrics.decay_score = decay / metrics.historical_win_rate
        else:
            metrics.decay_score = 0.0
        
        # Update status based on decay
        if metrics.decay_score >= self.DECAY_RETIRE_THRESHOLD:
            metrics.status = StrategyStatus.RETIRED
            logger.warning("Strategy %s RETIRED due to decay (score: %.2f)",
                          strategy_name, metrics.decay_score)
        elif metrics.decay_score >= self.DECAY_CRITICAL_THRESHOLD:
            metrics.status = StrategyStatus.COOLDOWN
            logger.warning("Strategy %s COOLDOWN due to decay (score: %.2f)",
                          strategy_name, metrics.decay_score)
        elif metrics.decay_score >= self.DECAY_WARNING_THRESHOLD:
            logger.info("Strategy %s decay warning (score: %.2f)",
                       strategy_name, metrics.decay_score)
    
    def _update_risk_metrics(self, strategy_name: str) -> None:
        """Update risk-adjusted metrics."""
        metrics = self.metrics[strategy_name]
        history = list(self.trade_history[strategy_name])
        
        if len(history) < 10:
            return
        
        # Calculate Sharpe ratio (simplified)
        pnls = [h["pnl"] for h in history]
        if np.std(pnls) > 0:
            metrics.sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        
        # Calculate Sortino ratio (only downside deviation)
        negative_pnls = [p for p in pnls if p < 0]
        if negative_pnls and np.std(negative_pnls) > 0:
            metrics.sortino_ratio = np.mean(pnls) / np.std(negative_pnls) * np.sqrt(252)
    
    def update_regime(self, regime: MarketRegime) -> List[str]:
        """
        Update market regime and return list of strategies to activate.
        
        Returns:
            List of strategy names that should be active for this regime
        """
        self.current_regime = regime
        
        # Get all strategies that work in this regime
        active_strategies = []
        
        for name, config in self.strategies.items():
            metrics = self.metrics[name]
            
            # Check if strategy is applicable to regime
            if regime not in config.applicable_regimes:
                continue
            
            # Check if strategy is healthy
            if metrics.status in [StrategyStatus.RETIRED, StrategyStatus.COOLDOWN]:
                continue
            
            # Check minimum requirements
            if metrics.total_trades > 10 and metrics.decay_score > self.DECAY_CRITICAL_THRESHOLD:
                continue
            
            active_strategies.append(name)
        
        # Update regime scores for all strategies
        for name in active_strategies:
            self.metrics[name].regime_scores[regime.name] = (
                self.metrics[name].regime_scores.get(regime.name, 0.5) * 0.9 + 0.1
            )
        
        logger.info("Regime updated to %s. Active strategies: %d", 
                    regime.name, len(active_strategies))
        
        return active_strategies
    
    def calculate_allocations(self) -> Dict[str, float]:
        """
        Calculate optimal capital allocation across strategies.
        
        Uses:
        1. Kelly criterion for position sizing
        2. Sharpe ratio for risk-adjusted returns
        3. Decay score to reduce allocation to decaying strategies
        4. Regime suitability
        """
        allocations: Dict[str, float] = {}
        
        # Calculate allocation scores
        scores: Dict[str, float] = {}
        total_score = 0.0
        
        for name, config in self.strategies.items():
            metrics = self.metrics[name]
            
            # Skip unhealthy strategies
            if metrics.status in [StrategyStatus.RETIRED]:
                scores[name] = 0.0
                continue
            
            if metrics.status == StrategyStatus.COOLDOWN:
                scores[name] = 0.1  # Minimal allocation during cooldown
                total_score += 0.1
                continue
            
            # Base score from configuration
            score = config.base_position_pct
            
            # Adjust by performance (if we have data)
            if metrics.total_trades >= 10:
                # Sharpe contribution
                if metrics.sharpe_ratio > 0:
                    score *= min(2.0, 1.0 + metrics.sharpe_ratio * 0.5)
                
                # Decay penalty
                score *= (1.0 - metrics.decay_score)
                
                # Profit factor contribution
                if metrics.profit_factor > 1.0:
                    score *= min(1.5, metrics.profit_factor)
            
            # Regime bonus
            if self.current_regime and self.current_regime.name in [
                r.name for r in config.applicable_regimes
            ]:
                score *= 1.2
            
            scores[name] = max(0.0, score)
            total_score += score
        
        # Normalize to percentages
        if total_score > 0:
            for name in scores:
                allocations[name] = (scores[name] / total_score) * 100.0
        else:
            # Equal allocation if no scores
            num_active = sum(1 for s in scores.values() if s > 0)
            if num_active > 0:
                for name in scores:
                    if scores[name] > 0:
                        allocations[name] = 100.0 / num_active
        
        self.allocations = allocations
        return allocations
    
    def run_champion_challenger_test(
        self,
        category: StrategyCategory
    ) -> Optional[ChampionChallengerResult]:
        """
        Run A/B test between champion and challenger in a category.
        
        Returns:
            ChampionChallengerResult if test is conclusive
        """
        # Get strategies in this category
        category_strategies = [
            name for name, config in self.strategies.items()
            if config.category == category
            and self.metrics[name].total_trades >= self.MIN_TRADES_FOR_COMPARISON
            and self.metrics[name].status != StrategyStatus.RETIRED
        ]
        
        if len(category_strategies) < 2:
            return None
        
        # Sort by Sharpe ratio
        sorted_strats = sorted(
            category_strategies,
            key=lambda x: self.metrics[x].sharpe_ratio,
            reverse=True
        )
        
        champion_name = sorted_strats[0]
        challenger_name = sorted_strats[1]
        
        champion_metrics = self.metrics[champion_name]
        challenger_metrics = self.metrics[challenger_name]
        
        # Calculate statistical significance (simplified t-test)
        champion_pnls = [h["pnl"] for h in self.trade_history[champion_name]]
        challenger_pnls = [h["pnl"] for h in self.trade_history[challenger_name]]
        
        # Simplified significance check
        champion_mean = np.mean(champion_pnls) if champion_pnls else 0
        challenger_mean = np.mean(challenger_pnls) if challenger_pnls else 0
        
        # Use standard error for significance
        if len(champion_pnls) > 5 and len(challenger_pnls) > 5:
            champion_se = np.std(champion_pnls) / np.sqrt(len(champion_pnls))
            challenger_se = np.std(challenger_pnls) / np.sqrt(len(challenger_pnls))
            
            diff = abs(champion_mean - challenger_mean)
            combined_se = np.sqrt(champion_se**2 + challenger_se**2)
            
            if combined_se > 0:
                z_score = diff / combined_se
                p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
            else:
                p_value = 1.0
        else:
            p_value = 1.0
        
        is_significant = p_value < self.SIGNIFICANCE_LEVEL
        
        # Determine if swap is needed
        should_swap = (
            is_significant and
            challenger_metrics.sharpe_ratio > champion_metrics.sharpe_ratio * 1.1
        )
        
        reason = ""
        if should_swap:
            reason = (f"Challenger {challenger_name} outperforms champion {champion_name} "
                     f"(Sharpe: {challenger_metrics.sharpe_ratio:.2f} vs "
                     f"{champion_metrics.sharpe_ratio:.2f}, p={p_value:.4f})")
        else:
            reason = f"Champion {champion_name} retained (p={p_value:.4f})"
        
        result = ChampionChallengerResult(
            champion_name=champion_name,
            challenger_name=challenger_name,
            category=category,
            champion_trades=champion_metrics.total_trades,
            challenger_trades=challenger_metrics.total_trades,
            champion_pnl=champion_metrics.total_pnl,
            challenger_pnl=challenger_metrics.total_pnl,
            champion_sharpe=champion_metrics.sharpe_ratio,
            challenger_sharpe=challenger_metrics.sharpe_ratio,
            p_value=p_value,
            is_significant=is_significant,
            should_swap=should_swap,
            confidence=1.0 - p_value,
            reason=reason
        )
        
        self.test_results.append(result)
        
        # Execute swap if needed
        if should_swap:
            champion_metrics.status = StrategyStatus.ACTIVE
            challenger_metrics.status = StrategyStatus.CHAMPION
            self.champion_challenger[category] = (challenger_name, champion_name)
            logger.info("CHAMPION SWAP: %s → %s", champion_name, challenger_name)
        else:
            champion_metrics.status = StrategyStatus.CHAMPION
            challenger_metrics.status = StrategyStatus.CHALLENGER
            self.champion_challenger[category] = (champion_name, challenger_name)
        
        return result
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def get_strategy_report(self) -> Dict[str, Any]:
        """Generate comprehensive strategy performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_strategies": len(self.strategies),
            "current_regime": self.current_regime.name if self.current_regime else None,
            "categories": {},
            "top_performers": [],
            "decaying_strategies": [],
            "champion_challenger": {},
        }
        
        # Group by category
        for category in StrategyCategory:
            cat_strategies = [
                name for name, config in self.strategies.items()
                if config.category == category
            ]
            
            if cat_strategies:
                cat_metrics = [self.metrics[name] for name in cat_strategies]
                report["categories"][category.name] = {
                    "count": len(cat_strategies),
                    "total_pnl": sum(m.total_pnl for m in cat_metrics),
                    "avg_sharpe": np.mean([m.sharpe_ratio for m in cat_metrics]),
                    "active": sum(1 for m in cat_metrics if m.status == StrategyStatus.ACTIVE),
                    "champion": next(
                        (m.strategy_name for m in cat_metrics 
                         if m.status == StrategyStatus.CHAMPION),
                        None
                    ),
                }
        
        # Top performers by Sharpe
        sorted_by_sharpe = sorted(
            self.metrics.values(),
            key=lambda m: m.sharpe_ratio,
            reverse=True
        )
        report["top_performers"] = [
            {
                "name": m.strategy_name,
                "sharpe": m.sharpe_ratio,
                "pnl": m.total_pnl,
                "win_rate": m.win_rate,
                "trades": m.total_trades,
            }
            for m in sorted_by_sharpe[:10]
            if m.total_trades > 0
        ]
        
        # Decaying strategies
        report["decaying_strategies"] = [
            {
                "name": m.strategy_name,
                "decay_score": m.decay_score,
                "recent_win_rate": m.recent_win_rate,
                "historical_win_rate": m.historical_win_rate,
                "status": m.status.name,
            }
            for m in self.metrics.values()
            if m.decay_score >= self.DECAY_WARNING_THRESHOLD
        ]
        
        # Champion-challenger status
        for category, (champion, challenger) in self.champion_challenger.items():
            report["champion_challenger"][category.name] = {
                "champion": champion,
                "challenger": challenger,
            }
        
        return report
    
    def get_strategy_allocation(self, strategy_name: str) -> float:
        """Get current allocation for a strategy."""
        return self.allocations.get(strategy_name, 0.0)
    
    def should_execute_strategy(self, strategy_name: str) -> Tuple[bool, str]:
        """Determine if a strategy should execute based on current conditions."""
        if strategy_name not in self.strategies:
            return False, "Strategy not registered"
        
        config = self.strategies[strategy_name]
        metrics = self.metrics[strategy_name]
        
        # Check if enabled
        if not config.enabled:
            return False, "Strategy disabled"
        
        # Check status
        if metrics.status == StrategyStatus.RETIRED:
            return False, "Strategy retired"
        
        if metrics.status == StrategyStatus.COOLDOWN:
            return False, "Strategy in cooldown"
        
        # Check allocation
        allocation = self.get_strategy_allocation(strategy_name)
        if allocation < 1.0:
            return False, f"Low allocation ({allocation:.1f}%)"
        
        # Check regime
        if self.current_regime and self.current_regime not in config.applicable_regimes:
            return False, f"Not suitable for {self.current_regime.name}"
        
        return True, "OK"


# Singleton instance
_hub: Optional[StrategyIntelligenceHub] = None


def get_strategy_intelligence_hub(
    config: Optional[Dict[str, Any]] = None
) -> StrategyIntelligenceHub:
    """Get or create the Strategy Intelligence Hub singleton."""
    global _hub
    if _hub is None:
        _hub = StrategyIntelligenceHub(config)
    return _hub


__all__ = [
    "StrategyIntelligenceHub",
    "StrategyCategory",
    "StrategyConfig",
    "StrategyMetrics",
    "StrategyStatus",
    "MarketRegime",
    "ChampionChallengerResult",
    "get_strategy_intelligence_hub",
]
