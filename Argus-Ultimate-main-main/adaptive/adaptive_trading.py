#!/usr/bin/env python3
"""
ADAPTIVE TRADING SYSTEM
=======================
Argus ADAPTS to market conditions - never controls them.

This is the unified adaptation system that:
1. Detects market conditions in real-time
2. Adapts strategy selection
3. Adapts position sizes
4. Adapts risk limits
5. Protects capital in bad conditions
6. Maximizes returns in good conditions

KEY PRINCIPLE: Follow the market, don't fight it.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class AdaptiveTradingSystem:
    """
    Unified adaptive trading system.
    
    Components:
    - Market Adaptation: Detects conditions
    - Strategy Adaptation: Selects best strategies
    - Risk Adaptation: Adjusts risk limits
    """
    
    def __init__(self, capital: float = 10000):
        self.capital = capital
        
        # Import adaptation modules
        from market_adaptation import MarketAdaptationSystem
        from strategy_adaptation import StrategyAdaptationSystem
        from risk_adaptation import RiskAdaptationSystem
        
        self.market = MarketAdaptationSystem()
        self.strategy = StrategyAdaptationSystem()
        self.risk = RiskAdaptationSystem(base_capital=capital)
        
        self.cycle_count = 0
        self.positions = {}
        
        logger.info("=" * 60)
        logger.info("ADAPTIVE TRADING SYSTEM")
        logger.info("Follows market conditions - never controls them")
        logger.info(f"Capital: ${capital}")
        logger.info("=" * 60)
    
    async def run_cycle(self, market_data: dict):
        """
        Run one adaptation cycle.
        
        market_data should contain:
        - prices: List[float]
        - volumes: List[float]
        - orderbook: Dict with bids/asks
        """
        self.cycle_count += 1
        
        prices = market_data.get("prices", [])
        volumes = market_data.get("volumes", [])
        orderbook = market_data.get("orderbook", {"bids": [], "asks": []})
        
        # 1. Analyze market conditions
        market_state = await self.market.analyze_market(
            price_data=prices,
            volume_data=volumes,
            orderbook=orderbook,
        )
        
        # 2. Adapt strategies to conditions
        strategy_weights = await self.strategy.adapt_to_condition(
            condition=market_state.condition.value,
            volatility=market_state.volatility,
            volume_ratio=market_state.volume_ratio,
        )
        
        # 3. Adapt risk to conditions
        risk_limits = await self.risk.adapt_to_market(
            market_condition=market_state.condition.value,
            volatility=market_state.volatility,
            trend_strength=market_state.trend_strength,
        )
        
        # 4. Check if we should trade
        should_trade, reason = self.market.should_trade()
        
        # 5. Check risk limits
        should_stop, stop_reason = self.risk.should_stop_trading()
        
        # 6. Get signals from active strategies
        signals = []
        if should_trade and not should_stop:
            for strategy_name in self.strategy.active_strategies:
                signal = self.strategy.get_strategy_signal(strategy_name, market_data)
                if signal and signal.get("action") != "hold":
                    signals.append(signal)
        
        # Log cycle summary
        if self.cycle_count % 10 == 0:
            logger.info(
                f"Cycle {self.cycle_count}: "
                f"Condition={market_state.condition.value} "
                f"({market_state.confidence:.0%}), "
                f"Strategies={len(self.strategy.active_strategies)}, "
                f"Signals={len(signals)}"
            )
        
        return {
            "cycle": self.cycle_count,
            "market_condition": market_state.condition.value,
            "market_confidence": market_state.confidence,
            "should_trade": should_trade,
            "trade_reason": reason,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "position_multiplier": self.market.position_multiplier,
            "risk_multiplier": self.market.risk_multiplier,
            "active_strategies": self.strategy.active_strategies,
            "strategy_weights": strategy_weights,
            "signals": signals,
            "risk_limits": {
                "max_position_pct": risk_limits.max_position_pct,
                "stop_loss_pct": risk_limits.stop_loss_pct,
                "take_profit_pct": risk_limits.take_profit_pct,
            },
        }
    
    def get_status(self) -> dict:
        """Get current system status."""
        return {
            "cycle": self.cycle_count,
            "capital": self.capital,
            "market": self.market.get_adaptation_params(),
            "risk": self.risk.get_risk_summary(),
            "strategy": self.strategy.get_adaptation_summary(),
        }


async def demo_adaptation():
    """Demo the adaptation system with simulated market data."""
    import numpy as np
    
    system = AdaptiveTradingSystem(capital=10000)
    
    # Simulate different market conditions
    conditions = [
        ("Bull Market", 50000, 0.02, 1.5),   # price, volatility, volume
        ("Sideways", 50000, 0.01, 0.8),
        ("High Volatility", 50000, 0.08, 3.0),
        ("Crash", 45000, 0.15, 5.0),
        ("Recovery", 47000, 0.05, 2.0),
    ]
    
    print("\n" + "=" * 60)
    print("ADAPTATION DEMO")
    print("=" * 60)
    
    for name, base_price, vol, vol_ratio in conditions:
        # Generate simulated data
        prices = [base_price + np.random.randn() * base_price * vol for _ in range(100)]
        volumes = [1000 * vol_ratio + np.random.randn() * 100 for _ in range(100)]
        orderbook = {
            "bids": [[base_price * 0.999, 10], [base_price * 0.998, 20]],
            "asks": [[base_price * 1.001, 10], [base_price * 1.002, 20]],
        }
        
        result = await system.run_cycle({
            "prices": prices,
            "volumes": volumes,
            "orderbook": orderbook,
        })
        
        print(f"\n{name}:")
        print(f"  Condition: {result['market_condition']} ({result['market_confidence']:.0%})")
        print(f"  Should Trade: {result['should_trade']} - {result['trade_reason']}")
        print(f"  Position Size: {result['position_multiplier']:.0%} of normal")
        print(f"  Risk Limits: {result['risk_limits']}")
        print(f"  Active Strategies: {result['active_strategies']}")
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY:")
    print("- In good conditions: larger positions, more strategies")
    print("- In bad conditions: smaller positions, fewer strategies")
    print("- In crash: minimal trading, capital preservation")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_adaptation())
