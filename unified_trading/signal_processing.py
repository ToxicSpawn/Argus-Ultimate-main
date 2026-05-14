"""
Signal Processing Module
==========================

Aggregates and processes trading signals from multiple strategies.
Refactored from unified_trading_system.py.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
from enum import Enum
from collections import defaultdict

from unified_trading.order_management import Signal, OrderSide
from core.exception_manager import (
    StrategyError,
    ModelPredictionError,
    handle_errors
)

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength classification."""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NEUTRAL = 0


@dataclass
class ProcessedSignal:
    """Processed and validated trading signal."""
    symbol: str
    side: OrderSide
    confidence: float
    strength: SignalStrength
    suggested_qty: Decimal
    suggested_price: Optional[Decimal]
    strategies: List[str]  # Contributing strategies
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if signal is still valid (not expired)."""
        if self.expires_at is None:
            return True
        return datetime.utcnow() < self.expires_at


@dataclass
class SignalFilter:
    """Filter configuration for signals."""
    min_confidence: float = 0.5
    min_strength: SignalStrength = SignalStrength.WEAK
    max_age_seconds: int = 60
    require_confirmation: bool = False
    blacklist_symbols: List[str] = field(default_factory=list)
    whitelist_symbols: List[str] = field(default_factory=list)


class SignalAggregator:
    """
    Aggregates signals from multiple strategies into consensus signal.
    """
    
    def __init__(self):
        self._signals: Dict[str, List[Signal]] = defaultdict(list)
        self._lock = asyncio.Lock()
        
        logger.info("SignalAggregator initialized")
    
    async def add_signal(self, signal: Signal):
        """Add signal from a strategy."""
        async with self._lock:
            self._signals[signal.symbol].append(signal)
            
            # Clean old signals (keep last 100 per symbol)
            if len(self._signals[signal.symbol]) > 100:
                self._signals[signal.symbol] = self._signals[signal.symbol][-100:]
    
    async def aggregate(self, symbol: str) -> Optional[ProcessedSignal]:
        """
        Aggregate signals for symbol into consensus.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            ProcessedSignal or None if no consensus
        """
        async with self._lock:
            signals = self._signals.get(symbol, [])
            
            if not signals:
                return None
            
            # Get recent signals (last 5 minutes)
            recent = [
                s for s in signals
                if (datetime.utcnow() - s.timestamp).seconds < 300
            ]
            
            if not recent:
                return None
            
            # Count buy/sell signals
            buy_signals = [s for s in recent if s.side == OrderSide.BUY]
            sell_signals = [s for s in recent if s.side == OrderSide.SELL]
            
            # Determine consensus side
            if len(buy_signals) > len(sell_signals) * 2:  # 2:1 ratio
                consensus_side = OrderSide.BUY
                avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
                contributing = [s.strategy for s in buy_signals]
            elif len(sell_signals) > len(buy_signals) * 2:
                consensus_side = OrderSide.SELL
                avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
                contributing = [s.strategy for s in sell_signals]
            else:
                return None  # No clear consensus
            
            # Determine strength
            if avg_confidence >= 0.9:
                strength = SignalStrength.VERY_STRONG
            elif avg_confidence >= 0.75:
                strength = SignalStrength.STRONG
            elif avg_confidence >= 0.6:
                strength = SignalStrength.MODERATE
            elif avg_confidence >= 0.5:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            # Calculate suggested quantity
            total_qty = sum(s.suggested_qty for s in recent if s.side == consensus_side)
            avg_qty = total_qty / len([s for s in recent if s.side == consensus_side])
            
            # Get suggested price
            prices = [s.suggested_price for s in recent if s.side == consensus_side and s.suggested_price]
            avg_price = sum(prices) / len(prices) if prices else None
            
            return ProcessedSignal(
                symbol=symbol,
                side=consensus_side,
                confidence=avg_confidence,
                strength=strength,
                suggested_qty=avg_qty,
                suggested_price=avg_price,
                strategies=list(set(contributing)),
                metadata={
                    "signal_count": len(recent),
                    "buy_count": len(buy_signals),
                    "sell_count": len(sell_signals)
                },
                expires_at=datetime.utcnow() + timedelta(seconds=60)
            )


class SignalProcessor:
    """
    Main signal processor that coordinates strategy signals.
    """
    
    def __init__(self):
        self._strategies: Dict[str, Callable] = {}
        self._aggregator = SignalAggregator()
        self._filter = SignalFilter()
        self._processed_signals: List[ProcessedSignal] = []
        self._lock = asyncio.Lock()
        
        # Statistics
        self._signals_generated = 0
        self._signals_filtered = 0
        self._signals_expired = 0
        
        logger.info("SignalProcessor initialized")
    
    async def initialize(self):
        """Initialize signal processor."""
        logger.info("Signal processor initialized")
    
    async def register_strategy(self, name: str, strategy_func: Callable):
        """
        Register a trading strategy.
        
        Args:
            name: Strategy name
            strategy_func: Async function that generates signals
        """
        self._strategies[name] = strategy_func
        logger.info(f"Strategy registered: {name}")
    
    @handle_errors(logger_name=__name__, reraise=False)
    async def generate_signals(
        self,
        symbol: str,
        price: float,
        **market_data
    ) -> List[Signal]:
        """
        Generate signals from all registered strategies.
        
        Args:
            symbol: Trading symbol
            price: Current price
            **market_data: Additional market data
            
        Returns:
            List of generated signals
        """
        signals = []
        
        # Generate signals from each strategy
        for name, strategy_func in self._strategies.items():
            try:
                if asyncio.iscoroutinefunction(strategy_func):
                    signal = await strategy_func(symbol, price, **market_data)
                else:
                    signal = strategy_func(symbol, price, **market_data)
                
                if signal:
                    signals.append(signal)
                    await self._aggregator.add_signal(signal)
                    self._signals_generated += 1
                    
            except Exception as e:
                logger.error(f"Strategy {name} failed: {e}")
                raise StrategyError(
                    f"Strategy {name} failed: {e}",
                    strategy_name=name,
                    symbol=symbol
                ) from e
        
        return signals
    
    @handle_errors(logger_name=__name__, reraise=False)
    async def process_signals(self, symbol: str) -> List[ProcessedSignal]:
        """
        Process and filter aggregated signals.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of valid processed signals
        """
        # Aggregate signals
        aggregated = await self._aggregator.aggregate(symbol)
        
        if not aggregated:
            return []
        
        # Apply filters
        valid_signals = []
        
        if self._apply_filter(aggregated):
            valid_signals.append(aggregated)
            async with self._lock:
                self._processed_signals.append(aggregated)
        else:
            self._signals_filtered += 1
        
        # Clean old processed signals
        await self._cleanup_old_signals()
        
        return valid_signals
    
    def _apply_filter(self, signal: ProcessedSignal) -> bool:
        """Apply filters to signal."""
        # Check confidence
        if signal.confidence < self._filter.min_confidence:
            return False
        
        # Check strength
        if signal.strength.value < self._filter.min_strength.value:
            return False
        
        # Check symbol lists
        if self._filter.blacklist_symbols and signal.symbol in self._filter.blacklist_symbols:
            return False
        
        if self._filter.whitelist_symbols and signal.symbol not in self._filter.whitelist_symbols:
            return False
        
        # Check validity (not expired)
        if not signal.is_valid():
            self._signals_expired += 1
            return False
        
        return True
    
    async def _cleanup_old_signals(self):
        """Remove expired processed signals."""
        async with self._lock:
            self._processed_signals = [
                s for s in self._processed_signals
                if s.is_valid()
            ]
    
    async def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[ProcessedSignal]:
        """Get recent processed signals."""
        async with self._lock:
            signals = self._processed_signals[-limit:]
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        return signals
    
    async def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal processing statistics."""
        return {
            "signals_generated": self._signals_generated,
            "signals_filtered": self._signals_filtered,
            "signals_expired": self._signals_expired,
            "active_strategies": len(self._strategies),
            "processed_signals": len(self._processed_signals),
            "filter_config": {
                "min_confidence": self._filter.min_confidence,
                "min_strength": self._filter.min_strength.name
            }
        }
    
    async def update_filter(self, **kwargs):
        """Update signal filter configuration."""
        for key, value in kwargs.items():
            if hasattr(self._filter, key):
                setattr(self._filter, key, value)
        
        logger.info(f"Signal filter updated: {kwargs}")
    
    async def clear_signals(self):
        """Clear all stored signals."""
        async with self._lock:
            self._processed_signals.clear()
            self._aggregator._signals.clear()
        
        logger.info("All signals cleared")
    
    async def start(self):
        """Start signal processing."""
        logger.info("Signal processor started")
    
    async def stop(self):
        """Stop signal processing."""
        await self.clear_signals()
        logger.info("Signal processor stopped")


# Built-in strategy functions
async def momentum_strategy(
    symbol: str,
    price: float,
    **kwargs
) -> Optional[Signal]:
    """Simple momentum strategy."""
    # Get price history
    prices = kwargs.get('prices', [])
    
    if len(prices) < 20:
        return None
    
    # Calculate momentum
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    momentum = sum(returns[-10:])  # Last 10 periods
    
    if momentum > 0.02:  # 2% upward momentum
        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            confidence=min(abs(momentum) * 10, 0.95),
            strategy="momentum",
            suggested_qty=Decimal("0.1"),
            suggested_price=Decimal(str(price))
        )
    elif momentum < -0.02:
        return Signal(
            symbol=symbol,
            side=OrderSide.SELL,
            confidence=min(abs(momentum) * 10, 0.95),
            strategy="momentum",
            suggested_qty=Decimal("0.1"),
            suggested_price=Decimal(str(price))
        )
    
    return None


async def mean_reversion_strategy(
    symbol: str,
    price: float,
    **kwargs
) -> Optional[Signal]:
    """Simple mean reversion strategy."""
    prices = kwargs.get('prices', [])
    
    if len(prices) < 50:
        return None
    
    # Calculate mean and std
    mean = sum(prices[-50:]) / 50
    std = (sum((p - mean) ** 2 for p in prices[-50:]) / 50) ** 0.5
    
    # Z-score
    z_score = (price - mean) / std if std > 0 else 0
    
    if z_score < -2.0:  # 2 std dev below mean
        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            confidence=min(abs(z_score) / 3, 0.95),
            strategy="mean_reversion",
            suggested_qty=Decimal("0.1"),
            suggested_price=Decimal(str(price))
        )
    elif z_score > 2.0:
        return Signal(
            symbol=symbol,
            side=OrderSide.SELL,
            confidence=min(abs(z_score) / 3, 0.95),
            strategy="mean_reversion",
            suggested_qty=Decimal("0.1"),
            suggested_price=Decimal(str(price))
        )
    
    return None


from datetime import timedelta
