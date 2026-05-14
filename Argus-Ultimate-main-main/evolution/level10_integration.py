"""
evolution/level10_integration.py — Level 10 Integration with Argus Trading System

Wires the Level 10 Self-Evolving System into Argus's main trading loop.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np

from evolution.level10_self_evolving_system import (
    Level10System,
    GenomeFactory,
    create_level10_system,
)

logger = logging.getLogger(__name__)


class Level10Integration:
    """
    Integrates Level 10 Self-Evolving System with Argus trading system.
    
    Features:
    - Evolves strategies during trading
    - Learns from market conditions
    - Generates and tests hypotheses
    - Self-improves code based on performance
    - Remembers what worked in similar conditions
    
    Configuration (in unified_config.yaml):
        level10:
          enabled: false          # Master switch
          population_size: 50     # Evolution population
          evolution_interval: 3600  # Evolve every N seconds
          complexity: 0.5         # Strategy complexity (0-1)
          auto_improve: true      # Auto-improve best strategies
          research_enabled: true  # Enable hypothesis research
    """
    
    def __init__(
        self,
        system: Any,  # UnifiedTradingSystem
        config: Optional[Dict] = None,
    ):
        self.system = system
        self.config = config or {}
        
        # Configuration
        self.enabled = bool(self.config.get("enabled", False))
        self.population_size = int(self.config.get("population_size", 50))
        self.evolution_interval = int(self.config.get("evolution_interval", 3600))
        self.complexity = float(self.config.get("complexity", 0.5))
        self.auto_improve = bool(self.config.get("auto_improve", True))
        self.research_enabled = bool(self.config.get("research_enabled", True))
        
        # Level 10 system
        self.level10: Optional[Level10System] = None
        self._evolution_task: Optional[asyncio.Task] = None
        self._last_evolution: Optional[datetime] = None
        
        # Performance tracking
        self._performance_history: List[float] = []
        self._best_strategies: List[Dict] = []
    
    async def initialize(self) -> None:
        """Initialize Level 10 system."""
        if not self.enabled:
            logger.info("Level 10 integration disabled")
            return
        
        logger.info("Initializing Level 10 Self-Evolving System...")
        
        # Create Level 10 system
        self.level10 = create_level10_system(population_size=self.population_size)
        self.level10.initialize(complexity=self.complexity)
        
        logger.info(
            "Level 10 initialized: population=%d, complexity=%.2f",
            self.population_size,
            self.complexity,
        )
    
    async def start_evolution_loop(self) -> None:
        """Start background evolution loop."""
        if not self.enabled or self.level10 is None:
            return
        
        logger.info("Starting Level 10 evolution loop (interval: %ds)", self.evolution_interval)
        
        self._evolution_task = asyncio.create_task(self._evolution_loop())
    
    async def stop(self) -> None:
        """Stop Level 10 system."""
        if self._evolution_task:
            self._evolution_task.cancel()
            try:
                await self._evolution_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Level 10 stopped")
    
    async def _evolution_loop(self) -> None:
        """Background evolution loop."""
        while True:
            try:
                await asyncio.sleep(self.evolution_interval)
                await self._evolve_generation()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Evolution loop error: %s", e)
                await asyncio.sleep(60)  # Wait before retry
    
    async def _evolve_generation(self) -> Dict:
        """Evolve one generation of strategies."""
        if self.level10 is None:
            return {"status": "not_initialized"}
        
        logger.info("Starting evolution generation %d", self.level10.current_generation + 1)
        
        # Define fitness function based on current market conditions
        def fitness_fn(genome):
            return self._evaluate_genome(genome)
        
        # Evolve
        results = self.level10.evolve(fitness_fn, generations=1)
        
        # Learn from this generation
        self.level10.learn_from_generation({
            "market_conditions": self._get_market_conditions(),
            "best_fitness": results["best_fitness"],
        })
        
        # Get best strategies
        best = self.level10.get_best_strategies(n=3)
        self._best_strategies = best
        
        self._last_evolution = datetime.now()
        
        logger.info(
            "Evolution complete: gen=%d, best_fitness=%.4f, strategies=%d",
            results["generations_completed"],
            results["best_fitness"],
            len(best),
        )
        
        return results
    
    def _evaluate_genome(self, genome) -> float:
        """
        Evaluate a genome's fitness.
        
        In production, this would run a backtest.
        For now, uses a simplified evaluation.
        """
        base_fitness = 0.5
        
        # Bonus for certain gene combinations
        if genome.get_gene("use_trend_filter"):
            base_fitness += 0.05
        if genome.get_gene("primary_indicator"):
            base_fitness += 0.03
        if genome.get_gene("rsi_period"):
            rsi_period = genome.get_gene("rsi_period").value
            if 10 <= rsi_period <= 20:
                base_fitness += 0.02
        
        # Add market condition bonus
        conditions = self._get_market_conditions()
        if conditions.get("volatility", 0) > 0.02:
            if genome.get_gene("use_volatility_filter"):
                base_fitness += 0.05
        
        # Add noise to prevent overfitting
        base_fitness += np.random.randn() * 0.05
        
        return max(0.0, min(1.0, base_fitness))
    
    def _get_market_conditions(self) -> Dict:
        """Get current market conditions from Argus."""
        conditions = {
            "timestamp": datetime.now().isoformat(),
            "volatility": 0.02,
            "trend": "neutral",
        }
        
        # Try to get real conditions from system
        try:
            if hasattr(self.system, "regime_detector"):
                regime = getattr(self.system.regime_detector, "current_regime", None)
                if regime:
                    conditions["regime"] = str(regime)
            
            if hasattr(self.system, "volatility_model"):
                vol = getattr(self.system.volatility_model, "current_volatility", None)
                if vol:
                    conditions["volatility"] = float(vol)
        except Exception:
            pass
        
        return conditions
    
    async def on_trade_result(self, trade: Dict) -> None:
        """
        Called when a trade completes.
        
        Used to:
        - Update strategy memory
        - Generate hypotheses
        - Trigger immediate evolution if performance drops
        """
        if not self.enabled or self.level10 is None:
            return
        
        pnl = trade.get("pnl", 0)
        self._performance_history.append(pnl)
        
        # Keep only last 100 trades
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]
        
        # Remember this trade
        conditions = self._get_market_conditions()
        best_genome = self.level10.evolution.get_best_genome()
        self.level10.memory.remember_strategy(
            best_genome if best_genome else GenomeFactory.create_random_genome(),
            market_conditions=conditions,
            performance={"pnl": pnl, "trade": trade},
        )
        
        # Check if we need to evolve immediately (performance drop)
        if len(self._performance_history) >= 20:
            recent_avg = np.mean(self._performance_history[-10:])
            older_avg = np.mean(self._performance_history[-20:-10])
            
            if recent_avg < older_avg * 0.5:  # 50% drop
                logger.warning("Performance drop detected, triggering immediate evolution")
                await self._evolve_generation()
    
    def get_best_code(self) -> Optional[str]:
        """Get the code of the best evolved strategy."""
        if not self.enabled or self.level10 is None:
            return None
        
        return self.level10.get_best_code()
    
    def get_report(self) -> Dict:
        """Get Level 10 system report."""
        if not self.enabled or self.level10 is None:
            return {"enabled": False}
        
        report = self.level10.get_system_report()
        report["last_evolution"] = (
            self._last_evolution.isoformat() if self._last_evolution else None
        )
        report["performance_history"] = {
            "trades": len(self._performance_history),
            "avg_pnl": float(np.mean(self._performance_history)) if self._performance_history else 0,
        }
        
        return report


def create_level10_integration(
    system: Any,
    config: Optional[Dict] = None,
) -> Level10Integration:
    """Factory function to create Level 10 integration."""
    return Level10Integration(system=system, config=config)
