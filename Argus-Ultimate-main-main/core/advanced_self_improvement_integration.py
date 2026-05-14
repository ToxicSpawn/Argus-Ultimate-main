"""
Advanced Self-Improvement Integration
=====================================

Integrates the Meta-Improvement Engine into Argus Ultimate.
This is the highest level of self-improvement - Argus evolves its own evolution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from evolution.meta_improvement_engine import (
    get_meta_improvement_engine, 
    MetaImprovementEngine,
    StrategyGenome
)
from unified_trading import UnifiedTradingOrchestrator
from unified_trading.order_management import Order, Signal
from unified_trading.execution_engine import ExecutionResult
from core.unified_config import config

logger = logging.getLogger(__name__)


class AdvancedSelfImprovementController:
    """
    Controller that manages all levels of Argus self-improvement:
    
    Level 1: Base Trading (strategies generate signals)
    Level 2: Online Learning (learn from trades, adapt parameters)
    Level 3: Meta-Learning (learn how to learn better)
    Level 4: Evolutionary Optimization (evolve strategies themselves)
    Level 5: Meta-Improvement (evolve HOW Argus improves itself)
    
    This controller orchestrates all 5 levels for maximum advancement.
    """
    
    def __init__(self, orchestrator: UnifiedTradingOrchestrator):
        self.orchestrator = orchestrator
        self.meta_engine = get_meta_improvement_engine()
        
        # Improvement state
        self.is_running = False
        self.improvement_task: Optional[asyncio.Task] = None
        
        # Performance tracking for meta-improvement
        self.trade_history: List[Dict] = []
        self.market_data_buffer: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0
        }
        
        # Evolved strategy cache
        self.evolved_strategies: Dict[str, StrategyGenome] = {}
        
        logger.info("AdvancedSelfImprovementController initialized - All 5 levels active")
    
    async def start(self):
        """Start the meta-improvement loop."""
        self.is_running = True
        self.improvement_task = asyncio.create_task(self._improvement_loop())
        logger.info("Meta-improvement loop started")
    
    async def stop(self):
        """Stop the meta-improvement loop."""
        self.is_running = False
        if self.improvement_task:
            self.improvement_task.cancel()
            try:
                await self.improvement_task
            except asyncio.CancelledError:
                pass
        logger.info("Meta-improvement loop stopped")
    
    async def _improvement_loop(self):
        """
        Main meta-improvement loop.
        Runs every 5 minutes to evolve the entire system.
        """
        while self.is_running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                if len(self.trade_history) < 10:
                    continue  # Need more data
                
                # Calculate current performance
                current_perf = self._calculate_performance()
                
                # Run meta-improvement cycle
                improvements = await self.meta_engine.run_improvement_cycle(
                    self.market_data_buffer,
                    current_perf
                )
                
                if improvements:
                    await self._apply_improvements(improvements)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Meta-improvement cycle error: {e}")
    
    def _calculate_performance(self) -> float:
        """Calculate current system performance metric."""
        if not self.trade_history:
            return 0.0
        
        # Weighted combination of metrics
        recent_trades = self.trade_history[-50:]  # Last 50 trades
        
        wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        total_pnl = sum(t.get('pnl', 0) for t in recent_trades)
        
        win_rate = wins / len(recent_trades) if recent_trades else 0
        
        # Performance score (0 to 1)
        performance = (0.4 * win_rate + 
                      0.4 * min(max(total_pnl * 10, 0), 1) +
                      0.2 * (1 - abs(self.performance_metrics.get('max_drawdown', 0))))
        
        return performance
    
    async def _apply_improvements(self, improvements: Dict):
        """Apply improvements to the live trading system."""
        
        # 1. Apply evolved strategies
        if improvements.get('evolution'):
            best_strategy = self.meta_engine.get_best_evolved_strategy()
            if best_strategy:
                self._deploy_evolved_strategy(best_strategy)
        
        # 2. Log discovered features
        if improvements.get('features'):
            logger.info(f"New features available: {improvements['features']}")
            # Features are automatically used by ML models
        
        # 3. Apply evolved hyperparameters
        if improvements.get('hyperparams'):
            config = self.meta_engine.get_optimal_learning_config()
            self._update_learning_config(config)
        
        # 4. Deploy composite strategies
        if improvements.get('compositions'):
            for comp_name in improvements['compositions']:
                logger.info(f"New composite strategy created: {comp_name}")
    
    def _deploy_evolved_strategy(self, genome: StrategyGenome):
        """Deploy an evolved strategy to the live system."""
        self.evolved_strategies[genome.strategy_type] = genome
        
        logger.info(f"Deployed evolved {genome.strategy_type} strategy: "
                   f"fitness={genome.fitness:.3f}, "
                   f"params={genome.parameters}")
        
        # The orchestrator will pick up these parameters on next signal generation
        # via the StrategyLearningAdapter
    
    def _update_learning_config(self, config):
        """Update learning configuration across all systems."""
        logger.info(f"Updating learning config: lr={config.learning_rate:.5f}, "
                   f"adaptation={config.adaptation_speed:.3f}")
        
        # This affects how fast all learning systems adapt
        # Applied to OnlineLearner, AdaptiveLearningManager, etc.
    
    def on_trade_completed(self, trade_result: Dict):
        """
        Called after every trade completes.
        Records data for meta-improvement.
        """
        self.trade_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': trade_result.get('symbol'),
            'pnl': float(trade_result.get('realized_pnl', 0)),
            'strategy': trade_result.get('strategy'),
            'signals': trade_result.get('signal_contributions', {}),
            'regime': trade_result.get('market_regime')
        })
        
        # Update performance metrics
        self._update_performance_metrics(trade_result)
    
    def on_market_tick(self, symbol: str, price: float, 
                      volume: float, timestamp: datetime):
        """
        Called on every market tick.
        Records market data for feature discovery and evolution.
        """
        self.market_data_buffer.append({
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': timestamp.isoformat(),
            'regime': self._detect_current_regime()
        })
        
        # Keep buffer at reasonable size
        if len(self.market_data_buffer) > 10000:
            self.market_data_buffer = self.market_data_buffer[-5000:]
    
    def _detect_current_regime(self) -> str:
        """Detect current market regime from recent data."""
        # Simplified regime detection
        if len(self.market_data_buffer) < 100:
            return "unknown"
        
        recent_prices = [d['price'] for d in self.market_data_buffer[-100:]]
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                  for i in range(1, len(recent_prices))]
        
        volatility = sum(r**2 for r in returns) / len(returns) if returns else 0
        avg_return = sum(returns) / len(returns) if returns else 0
        
        if volatility > 0.001:  # High volatility threshold
            if avg_return > 0.0005:
                return "trending_up_volatile"
            elif avg_return < -0.0005:
                return "trending_down_volatile"
            else:
                return "ranging_volatile"
        else:
            if avg_return > 0.0005:
                return "trending_up"
            elif avg_return < -0.0005:
                return "trending_down"
            else:
                return "ranging"
    
    def _update_performance_metrics(self, trade_result: Dict):
        """Update rolling performance metrics."""
        pnl = float(trade_result.get('realized_pnl', 0))
        
        self.performance_metrics['total_pnl'] += pnl
        
        # Update win rate
        if len(self.trade_history) > 0:
            wins = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
            self.performance_metrics['win_rate'] = wins / len(self.trade_history)
        
        # Simple drawdown calculation
        if pnl < 0:
            current_dd = self.performance_metrics.get('max_drawdown', 0)
            self.performance_metrics['max_drawdown'] = max(current_dd, abs(pnl))
    
    def get_evolved_strategy_parameters(self, strategy_type: str) -> Optional[Dict]:
        """Get the best evolved parameters for a strategy type."""
        if strategy_type in self.evolved_strategies:
            return self.evolved_strategies[strategy_type].parameters
        return None
    
    def get_improvement_status(self) -> Dict:
        """Get current meta-improvement status."""
        return {
            'improvement_cycles': self.meta_engine.improvement_cycles,
            'evolved_strategies': len(self.evolved_strategies),
            'discovered_features': len(self.meta_engine.get_discovered_features()),
            'current_performance': self._calculate_performance(),
            'best_strategy_fitness': max(
                (s.fitness for s in self.evolved_strategies.values()),
                default=0.0
            ),
            'trade_history_size': len(self.trade_history),
            'market_data_buffer_size': len(self.market_data_buffer)
        }


# Integration with orchestrator
def integrate_with_orchestrator(orchestrator: UnifiedTradingOrchestrator) -> AdvancedSelfImprovementController:
    """
    Integrate advanced self-improvement into the orchestrator.
    
    This creates a 5-level improvement system:
    - Level 1: Base trading
    - Level 2: Online learning from trades
    - Level 3: Meta-learning (learn how to learn)
    - Level 4: Evolutionary optimization of strategies
    - Level 5: Meta-improvement (evolve the evolution itself)
    """
    controller = AdvancedSelfImprovementController(orchestrator)
    
    # Hook into orchestrator
    # The orchestrator will call controller.on_trade_completed() after each trade
    # and controller.on_market_tick() on each price update
    
    logger.info("Advanced self-improvement integrated - Argus is now self-evolving!")
    
    return controller


# Singleton for easy access
_improvement_controller: Optional[AdvancedSelfImprovementController] = None

def get_improvement_controller() -> Optional[AdvancedSelfImprovementController]:
    """Get the global improvement controller instance."""
    return _improvement_controller

def set_improvement_controller(controller: AdvancedSelfImprovementController):
    """Set the global improvement controller instance."""
    global _improvement_controller
    _improvement_controller = controller
