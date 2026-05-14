"""
Social Trading Signals Module
==============================
Tracks and copies successful traders:
- Top trader identification
- Signal aggregation
- Copy trading logic
- Performance tracking
- Risk-adjusted position sizing

Sources:
- Twitter/X alpha accounts
- Telegram signal groups
- Discord trading communities
- On-chain whale tracking
- Exchange leaderboards
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Signal sources."""
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    ON_CHAIN = "on_chain"
    EXCHANGE_LEADERBOARD = "exchange_leaderboard"
    COPY_TRADING = "copy_trading"


class SignalType(Enum):
    """Signal types."""
    ENTRY = "entry"
    EXIT = "exit"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


@dataclass
class TraderProfile:
    """Profile of a tracked trader."""
    trader_id: str
    name: str
    source: SignalSource
    total_pnl_usd: float = 0.0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    followers: int = 0
    trust_score: float = 0.5  # 0-1
    last_active: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class TradingSignal:
    """A trading signal from a trader."""
    signal_id: str
    trader_id: str
    source: SignalSource
    symbol: str
    signal_type: SignalType
    direction: str  # "long" or "short"
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.5
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CopyTrade:
    """A copied trade."""
    original_signal: TradingSignal
    copied_at: float
    position_size_usd: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    status: str = "open"  # open, closed, cancelled
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0


class TraderTracker:
    """
    Trader Tracker
    ==============
    Tracks and ranks successful traders.
    """
    
    def __init__(self):
        self.traders: Dict[str, TraderProfile] = {}
        self.trader_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Trust score weights
        self.score_weights = {
            "win_rate": 0.25,
            "sharpe_ratio": 0.25,
            "consistency": 0.20,
            "track_record": 0.15,
            "followers": 0.15
        }
    
    def add_trader(self, trader: TraderProfile) -> None:
        """Add trader to tracking."""
        self.traders[trader.trader_id] = trader
        logger.info(f"Tracking trader: {trader.name} ({trader.source.value})")
    
    def update_trader_stats(
        self,
        trader_id: str,
        trade_result: Dict[str, Any]
    ) -> None:
        """Update trader statistics with new trade."""
        if trader_id not in self.traders:
            return
        
        trader = self.traders[trader_id]
        
        # Update stats
        pnl = trade_result.get("pnl_pct", 0)
        trader.total_trades += 1
        
        if pnl > 0:
            old_wins = trader.win_rate * (trader.total_trades - 1)
            trader.win_rate = (old_wins + 1) / trader.total_trades
        else:
            old_wins = trader.win_rate * (trader.total_trades - 1)
            trader.win_rate = old_wins / trader.total_trades
        
        # Update average return
        old_avg = trader.avg_return_pct * (trader.total_trades - 1)
        trader.avg_return_pct = (old_avg + pnl) / trader.total_trades
        
        # Update total PnL
        trader.total_pnl_usd += trade_result.get("pnl_usd", 0)
        
        # Store history
        if trader_id not in self.trader_history:
            self.trader_history[trader_id] = []
        self.trader_history[trader_id].append({
            **trade_result,
            "timestamp": time.time()
        })
        
        # Recalculate trust score
        trader.trust_score = self.calculate_trust_score(trader)
    
    def calculate_trust_score(self, trader: TraderProfile) -> float:
        """Calculate trust score for a trader."""
        scores = {}
        
        # Win rate score (0-1)
        scores["win_rate"] = min(trader.win_rate / 0.7, 1.0)  # 70% = max score
        
        # Sharpe ratio score (0-1)
        scores["sharpe_ratio"] = min(max(trader.sharpe_ratio, 0) / 3, 1.0)  # 3.0 = max
        
        # Consistency score based on history
        history = self.trader_history.get(trader.trader_id, [])
        if len(history) >= 10:
            returns = [h.get("pnl_pct", 0) for h in history[-50:]]
            consistency = 1 - min(np.std(returns) / 10, 1)
            scores["consistency"] = consistency
        else:
            scores["consistency"] = 0.5
        
        # Track record length
        scores["track_record"] = min(len(history) / 100, 1.0)
        
        # Followers (log scale)
        scores["followers"] = min(np.log10(max(trader.followers, 1)) / 6, 1.0)
        
        # Weighted average
        trust_score = sum(
            scores.get(key, 0.5) * weight
            for key, weight in self.score_weights.items()
        )
        
        return trust_score
    
    def get_top_traders(
        self,
        n: int = 10,
        min_trades: int = 20,
        source: Optional[SignalSource] = None
    ) -> List[TraderProfile]:
        """Get top traders by trust score."""
        filtered = [
            t for t in self.traders.values()
            if t.total_trades >= min_trades
            and (source is None or t.source == source)
        ]
        
        return sorted(filtered, key=lambda t: t.trust_score, reverse=True)[:n]
    
    def get_trader_performance(self, trader_id: str) -> Dict[str, Any]:
        """Get detailed performance metrics for a trader."""
        if trader_id not in self.traders:
            return {}
        
        trader = self.traders[trader_id]
        history = self.trader_history.get(trader_id, [])
        
        if not history:
            return {"trader": trader.name, "trades": 0}
        
        returns = [h.get("pnl_pct", 0) for h in history]
        
        # Calculate metrics
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        return {
            "trader": trader.name,
            "source": trader.source.value,
            "total_trades": len(history),
            "win_rate": trader.win_rate * 100,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            "sharpe_ratio": trader.sharpe_ratio,
            "max_drawdown": trader.max_drawdown,
            "total_pnl": trader.total_pnl_usd,
            "trust_score": trader.trust_score
        }


class SignalAggregator:
    """
    Signal Aggregator
    =================
    Aggregates signals from multiple traders.
    """
    
    def __init__(self):
        self.signals: List[TradingSignal] = []
        self.signal_scores: Dict[str, float] = {}
    
    def add_signal(self, signal: TradingSignal) -> None:
        """Add a new signal."""
        self.signals.append(signal)
        
        # Keep only recent signals
        cutoff = time.time() - 24 * 3600  # 24 hours
        self.signals = [s for s in self.signals if s.timestamp > cutoff]
    
    def aggregate_signals(
        self,
        symbol: str,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Aggregate signals for a symbol."""
        relevant = [
            s for s in self.signals
            if s.symbol == symbol
            and s.confidence >= min_confidence
            and (s.expires_at is None or s.expires_at > time.time())
        ]
        
        if not relevant:
            return {
                "symbol": symbol,
                "signal": "neutral",
                "confidence": 0,
                "signal_count": 0
            }
        
        # Count directions
        long_signals = [s for s in relevant if s.direction == "long"]
        short_signals = [s for s in relevant if s.direction == "short"]
        
        long_confidence = sum(s.confidence for s in long_signals)
        short_confidence = sum(s.confidence for s in short_signals)
        
        total_confidence = long_confidence + short_confidence
        
        if total_confidence == 0:
            return {
                "symbol": symbol,
                "signal": "neutral",
                "confidence": 0,
                "signal_count": len(relevant)
            }
        
        # Determine consensus
        if long_confidence > short_confidence * 1.5:
            direction = "long"
            confidence = long_confidence / total_confidence
        elif short_confidence > long_confidence * 1.5:
            direction = "short"
            confidence = short_confidence / total_confidence
        else:
            direction = "neutral"
            confidence = 0.5
        
        # Get entry prices
        entry_prices = [s.entry_price for s in relevant if s.direction == direction]
        avg_entry = np.mean(entry_prices) if entry_prices else 0
        
        # Get stop losses and take profits
        stop_losses = [s.stop_loss for s in relevant if s.stop_loss and s.direction == direction]
        take_profits = [s.take_profit for s in relevant if s.take_profit and s.direction == direction]
        
        return {
            "symbol": symbol,
            "signal": direction,
            "confidence": confidence,
            "signal_count": len(relevant),
            "long_signals": len(long_signals),
            "short_signals": len(short_signals),
            "avg_entry_price": avg_entry,
            "suggested_stop_loss": np.mean(stop_losses) if stop_losses else None,
            "suggested_take_profit": np.mean(take_profits) if take_profits else None,
            "sources": list(set(s.source.value for s in relevant))
        }
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[TradingSignal]:
        """Get active signals, optionally filtered by symbol."""
        cutoff = time.time() - 24 * 3600
        
        active = [
            s for s in self.signals
            if s.timestamp > cutoff
            and (s.expires_at is None or s.expires_at > time.time())
        ]
        
        if symbol:
            active = [s for s in active if s.symbol == symbol]
        
        return active


class CopyTradingEngine:
    """
    Copy Trading Engine
    ===================
    Automatically copies trades from tracked traders.
    """
    
    def __init__(self, capital: float = 10000):
        self.capital = capital
        self.trader_tracker = TraderTracker()
        self.signal_aggregator = SignalAggregator()
        
        self.copy_trades: List[CopyTrade] = []
        self.max_position_pct: float = 0.1  # Max 10% per trade
        self.max_correlated_trades: int = 5
        
        # Copy settings
        self.min_trust_score: float = 0.6
        self.min_signal_confidence: float = 0.5
    
    def calculate_position_size(
        self,
        trader: TraderProfile,
        signal: TradingSignal,
        current_price: float
    ) -> float:
        """Calculate position size based on trader trust and signal confidence."""
        # Base size from capital
        base_size = self.capital * self.max_position_pct
        
        # Adjust by trust score
        trust_multiplier = trader.trust_score
        
        # Adjust by signal confidence
        confidence_multiplier = signal.confidence
        
        # Adjust by trader's historical performance
        performance_multiplier = min(1 + trader.avg_return_pct / 100, 2)
        
        # Final size
        position_size = base_size * trust_multiplier * confidence_multiplier * performance_multiplier
        
        # Cap at max
        max_size = self.capital * 0.25  # Never more than 25%
        position_size = min(position_size, max_size)
        
        return position_size
    
    def should_copy_signal(
        self,
        signal: TradingSignal,
        trader: TraderProfile
    ) -> Tuple[bool, str]:
        """Determine if a signal should be copied."""
        # Check trust score
        if trader.trust_score < self.min_trust_score:
            return False, f"Low trust score: {trader.trust_score:.2f}"
        
        # Check signal confidence
        if signal.confidence < self.min_signal_confidence:
            return False, f"Low signal confidence: {signal.confidence:.2f}"
        
        # Check if signal is expired
        if signal.expires_at and signal.expires_at < time.time():
            return False, "Signal expired"
        
        # Check for correlated positions
        open_symbols = set(t.original_signal.symbol for t in self.copy_trades if t.status == "open")
        if signal.symbol in open_symbols:
            return False, f"Already have position in {signal.symbol}"
        
        # Check max correlated trades
        if len(open_symbols) >= self.max_correlated_trades:
            return False, f"Max correlated trades reached ({self.max_correlated_trades})"
        
        return True, "OK"
    
    async def execute_copy_trade(
        self,
        signal: TradingSignal,
        trader: TraderProfile,
        current_price: float
    ) -> Optional[CopyTrade]:
        """Execute a copy trade."""
        should_copy, reason = self.should_copy_signal(signal, trader)
        
        if not should_copy:
            logger.info(f"Not copying signal {signal.signal_id}: {reason}")
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(trader, signal, current_price)
        
        # Create copy trade
        copy_trade = CopyTrade(
            original_signal=signal,
            copied_at=time.time(),
            position_size_usd=position_size,
            entry_price=current_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        self.copy_trades.append(copy_trade)
        
        logger.info(
            f"Copy trade executed: {signal.symbol} {signal.direction} "
            f"${position_size:.2f} @ ${current_price:.2f} "
            f"(from {trader.name})"
        )
        
        return copy_trade
    
    def close_copy_trade(
        self,
        copy_trade: CopyTrade,
        exit_price: float
    ) -> CopyTrade:
        """Close a copy trade."""
        entry = copy_trade.entry_price
        exit_p = exit_price
        
        if copy_trade.original_signal.direction == "long":
            pnl_pct = (exit_p - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_p) / entry * 100
        
        pnl_usd = copy_trade.position_size_usd * pnl_pct / 100
        
        copy_trade.status = "closed"
        copy_trade.pnl_pct = pnl_pct
        copy_trade.pnl_usd = pnl_usd
        
        # Update capital
        self.capital += pnl_usd
        
        return copy_trade
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get copy trading performance summary."""
        closed_trades = [t for t in self.copy_trades if t.status == "closed"]
        
        if not closed_trades:
            return {
                "total_trades": 0,
                "current_capital": self.capital,
                "open_positions": len([t for t in self.copy_trades if t.status == "open"])
            }
        
        pnls = [t.pnl_pct for t in closed_trades]
        wins = [t for t in closed_trades if t.pnl_usd > 0]
        losses = [t for t in closed_trades if t.pnl_usd <= 0]
        
        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(closed_trades) * 100,
            "avg_win_pct": np.mean([t.pnl_pct for t in wins]) if wins else 0,
            "avg_loss_pct": np.mean([t.pnl_pct for t in losses]) if losses else 0,
            "total_pnl_usd": sum(t.pnl_usd for t in closed_trades),
            "total_pnl_pct": sum(pnls),
            "current_capital": self.capital,
            "open_positions": len([t for t in self.copy_trades if t.status == "open"]),
            "best_trade": max(closed_trades, key=lambda t: t.pnl_usd).pnl_usd if closed_trades else 0,
            "worst_trade": min(closed_trades, key=lambda t: t.pnl_usd).pnl_usd if closed_trades else 0
        }


# Export
__all__ = [
    "SignalSource",
    "SignalType",
    "TraderProfile",
    "TradingSignal",
    "CopyTrade",
    "TraderTracker",
    "SignalAggregator",
    "CopyTradingEngine"
]
