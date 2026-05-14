"""
Self-Learning Universe Expansion

Automatically discovers and validates new trading pairs:
- Monitors exchange for available pairs
- Filters by liquidity, volatility, spread
- Paper-tests new pairs before adding to active universe
- Removes pairs that underperform

Uses a multi-stage pipeline:
1. Discovery → Find candidate pairs
2. Validation → Check liquidity, spreads, data quality
3. Paper Testing → Simulate trades for N cycles
4. Promotion → Add to active universe if profitable
5. Monitoring → Continuous evaluation, demotion if underperforming
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PairStatus(Enum):
    """Status of a trading pair in the universe."""
    CANDIDATE = "candidate"      # Discovered, not yet validated
    VALIDATING = "validating"    # Checking liquidity/spreads
    PAPER_TESTING = "paper_testing"  # Simulating trades
    ACTIVE = "active"           # In active trading universe
    MONITORING = "monitoring"   # Active but being evaluated for removal
    DEMOTED = "demoted"         # Removed from active, cooling off
    BLACKLISTED = "blacklisted" # Permanently excluded


@dataclass
class PairMetrics:
    """Metrics for a trading pair."""
    symbol: str
    status: PairStatus = PairStatus.CANDIDATE
    
    # Liquidity metrics
    avg_volume_usd: float = 0.0
    bid_ask_spread_pct: float = 0.0
    orderbook_depth_usd: float = 0.0
    
    # Volatility metrics
    avg_daily_range_pct: float = 0.0
    atr_pct: float = 0.0
    
    # Performance metrics (from paper testing)
    paper_trades: int = 0
    paper_wins: int = 0
    paper_pnl: float = 0.0
    paper_sharpe: float = 0.0
    
    # Tracking
    discovered_at: float = field(default_factory=time.time)
    last_evaluated: float = 0.0
    evaluation_count: int = 0
    
    # Rejection reasons
    rejection_reasons: List[str] = field(default_factory=list)
    
    @property
    def paper_win_rate(self) -> float:
        return self.paper_wins / max(self.paper_trades, 1)
    
    @property
    def paper_avg_pnl(self) -> float:
        return self.paper_pnl / max(self.paper_trades, 1)


class SelfLearningUniverse:
    """
    Self-learning universe that discovers and manages trading pairs.
    
    Automatically expands the trading universe based on:
    - Exchange availability
    - Liquidity requirements
    - Paper trading performance
    - Ongoing monitoring
    """
    
    # Default validation thresholds
    DEFAULT_THRESHOLDS = {
        "min_volume_usd": 1_000_000,      # $1M daily volume
        "max_spread_pct": 0.002,          # 0.2% max spread
        "min_atr_pct": 0.01,              # 1% min volatility
        "max_atr_pct": 0.15,              # 15% max volatility
        "min_orderbook_depth_usd": 100_000,  # $100k depth
        "paper_trades_required": 10,      # Min paper trades
        "paper_min_win_rate": 0.40,       # 40% min win rate
        "paper_min_sharpe": 0.0,          # Positive Sharpe
        "max_pairs": 30,                  # Max universe size
        "evaluation_interval_hours": 24,  # How often to re-evaluate
        "demotion_threshold": -0.02,      # -2% avg PnL triggers demotion
        "cooldown_hours": 48,             # Hours before re-promotion
    }
    
    def __init__(
        self,
        active_pairs: List[str],
        thresholds: Optional[Dict[str, float]] = None,
        data_dir: str = "data",
    ):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.data_dir = data_dir
        
        # Initialize active pairs
        self.pairs: Dict[str, PairMetrics] = {}
        for symbol in active_pairs:
            self.pairs[symbol] = PairMetrics(
                symbol=symbol,
                status=PairStatus.ACTIVE,
            )
        
        # Discovery state
        self.candidate_queue: deque = deque(maxlen=100)
        self.last_discovery_time: float = 0
        self.last_evaluation_time: float = 0
        
        # Statistics
        self.total_discovered: int = len(active_pairs)
        self.total_promoted: int = 0
        self.total_demoted: int = 0
        
        logger.info(
            "SelfLearningUniverse initialized: %d active pairs, max=%d",
            len(active_pairs), self.thresholds["max_pairs"],
        )
    
    def discover_candidates(self, exchange_pairs: List[str]) -> List[str]:
        """
        Discover new candidate pairs from exchange listing.
        
        Filters out already active/blacklisted pairs.
        Returns list of new candidates.
        """
        active_symbols = set(self.pairs.keys())
        new_candidates = []
        
        for symbol in exchange_pairs:
            if symbol not in active_symbols:
                # Check if it's a valid trading pair (has USD/USDT quote)
                if self._is_valid_pair(symbol):
                    self.candidate_queue.append(symbol)
                    new_candidates.append(symbol)
        
        self.total_discovered += len(new_candidates)
        self.last_discovery_time = time.time()
        
        if new_candidates:
            logger.info("Discovered %d new candidate pairs", len(new_candidates))
        
        return new_candidates
    
    def _is_valid_pair(self, symbol: str) -> bool:
        """Check if symbol is a valid trading pair."""
        valid_quotes = ["USD", "USDT", "USDC", "BUSD", "AUD", "EUR"]
        return any(symbol.endswith(f"/{q}") for q in valid_quotes)
    
    def validate_pair(
        self,
        symbol: str,
        volume_usd: float,
        spread_pct: float,
        atr_pct: float,
        orderbook_depth_usd: float,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a candidate pair against thresholds.
        
        Returns (is_valid, rejection_reasons).
        """
        reasons = []
        
        if volume_usd < self.thresholds["min_volume_usd"]:
            reasons.append(f"Low volume: ${volume_usd:,.0f} < ${self.thresholds['min_volume_usd']:,.0f}")
        
        if spread_pct > self.thresholds["max_spread_pct"]:
            reasons.append(f"High spread: {spread_pct*100:.3f}% > {self.thresholds['max_spread_pct']*100:.3f}%")
        
        if atr_pct < self.thresholds["min_atr_pct"]:
            reasons.append(f"Low volatility: {atr_pct*100:.2f}% < {self.thresholds['min_atr_pct']*100:.2f}%")
        
        if atr_pct > self.thresholds["max_atr_pct"]:
            reasons.append(f"High volatility: {atr_pct*100:.2f}% > {self.thresholds['max_atr_pct']*100:.2f}%")
        
        if orderbook_depth_usd < self.thresholds["min_orderbook_depth_usd"]:
            reasons.append(f"Low depth: ${orderbook_depth_usd:,.0f} < ${self.thresholds['min_orderbook_depth_usd']:,.0f}")
        
        # Update pair metrics
        if symbol not in self.pairs:
            self.pairs[symbol] = PairMetrics(symbol=symbol)
        
        pair = self.pairs[symbol]
        pair.avg_volume_usd = volume_usd
        pair.bid_ask_spread_pct = spread_pct
        pair.atr_pct = atr_pct
        pair.orderbook_depth_usd = orderbook_depth_usd
        pair.rejection_reasons = reasons
        pair.last_evaluated = time.time()
        pair.evaluation_count += 1
        
        is_valid = len(reasons) == 0
        
        if is_valid:
            pair.status = PairStatus.PAPER_TESTING
            logger.info("Pair %s validated - starting paper testing", symbol)
        else:
            pair.status = PairStatus.BLACKLISTED
            logger.debug("Pair %s rejected: %s", symbol, "; ".join(reasons))
        
        return is_valid, reasons
    
    def record_paper_trade(
        self,
        symbol: str,
        pnl_pct: float,
    ) -> None:
        """Record a paper trade result for a pair under testing."""
        if symbol not in self.pairs:
            return
        
        pair = self.pairs[symbol]
        pair.paper_trades += 1
        pair.paper_pnl += pnl_pct
        
        if pnl_pct > 0:
            pair.paper_wins += 1
        
        # Update Sharpe
        if pair.paper_trades >= 5:
            # Simplified Sharpe calculation
            pair.paper_sharpe = pair.paper_avg_pnl / max(0.001, abs(pair.paper_avg_pnl))
        
        # Check if ready for promotion
        if self._should_promote(pair):
            self._promote_pair(symbol)
    
    def _should_promote(self, pair: PairMetrics) -> bool:
        """Check if a pair should be promoted to active."""
        if pair.status != PairStatus.PAPER_TESTING:
            return False
        
        # Check minimum trades
        if pair.paper_trades < self.thresholds["paper_trades_required"]:
            return False
        
        # Check win rate
        if pair.paper_win_rate < self.thresholds["paper_min_win_rate"]:
            pair.rejection_reasons.append(
                f"Low win rate: {pair.paper_win_rate*100:.1f}% < {self.thresholds['paper_min_win_rate']*100:.1f}%"
            )
            pair.status = PairStatus.DEMOTED
            return False
        
        # Check Sharpe
        if pair.paper_sharpe < self.thresholds["paper_min_sharpe"]:
            pair.rejection_reasons.append(
                f"Low Sharpe: {pair.paper_sharpe:.2f} < {self.thresholds['paper_min_sharpe']:.2f}"
            )
            pair.status = PairStatus.DEMOTED
            return False
        
        # Check universe size
        active_count = sum(1 for p in self.pairs.values() if p.status == PairStatus.ACTIVE)
        if active_count >= self.thresholds["max_pairs"]:
            logger.info("Universe full (%d/%d), cannot promote %s", 
                       active_count, self.thresholds["max_pairs"], pair.symbol)
            return False
        
        return True
    
    def _promote_pair(self, symbol: str) -> None:
        """Promote a pair to active trading."""
        pair = self.pairs[symbol]
        pair.status = PairStatus.ACTIVE
        self.total_promoted += 1
        
        logger.info(
            "🚀 PROMOTED %s to active universe (win_rate=%.1f%%, sharpe=%.2f, trades=%d)",
            symbol, pair.paper_win_rate * 100, pair.paper_sharpe, pair.paper_trades,
        )
    
    def evaluate_active_pairs(
        self,
        performance_data: Dict[str, Dict[str, float]],
    ) -> List[str]:
        """
        Evaluate active pairs for potential demotion.
        
        Args:
            performance_data: {symbol: {"pnl": float, "trades": int, "win_rate": float}}
        
        Returns:
            List of demoted symbols.
        """
        demoted = []
        
        for symbol, perf in performance_data.items():
            if symbol not in self.pairs:
                continue
            
            pair = self.pairs[symbol]
            if pair.status != PairStatus.ACTIVE:
                continue
            
            avg_pnl = perf.get("pnl", 0) / max(1, perf.get("trades", 1))
            
            # Check demotion threshold
            if avg_pnl < self.thresholds["demotion_threshold"]:
                pair.status = PairStatus.DEMOTED
                pair.rejection_reasons.append(
                    f"Poor performance: avg_pnl={avg_pnl*100:.2f}%"
                )
                demoted.append(symbol)
                
                logger.warning(
                    "📉 DEMOTED %s from active universe (avg_pnl=%.2f%%)",
                    symbol, avg_pnl * 100,
                )
        
        self.total_demoted += len(demoted)
        return demoted
    
    def get_active_pairs(self) -> List[str]:
        """Get list of currently active trading pairs."""
        return [
            symbol for symbol, pair in self.pairs.items()
            if pair.status == PairStatus.ACTIVE
        ]
    
    def get_paper_testing_pairs(self) -> List[str]:
        """Get list of pairs currently in paper testing."""
        return [
            symbol for symbol, pair in self.pairs.items()
            if pair.status == PairStatus.PAPER_TESTING
        ]
    
    def get_universe_stats(self) -> Dict[str, Any]:
        """Get universe statistics."""
        status_counts = {}
        for pair in self.pairs.values():
            status = pair.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_pairs": len(self.pairs),
            "active_pairs": self.get_active_pairs(),
            "active_count": status_counts.get("active", 0),
            "paper_testing_count": status_counts.get("paper_testing", 0),
            "blacklisted_count": status_counts.get("blacklisted", 0),
            "demoted_count": status_counts.get("demoted", 0),
            "total_discovered": self.total_discovered,
            "total_promoted": self.total_promoted,
            "total_demoted": self.total_demoted,
            "status_breakdown": status_counts,
        }
    
    def save_state(self) -> None:
        """Save universe state to disk."""
        state_path = os.path.join(self.data_dir, "universe_state.json")
        
        state = {
            "pairs": {
                symbol: {
                    "status": pair.status.value,
                    "avg_volume_usd": pair.avg_volume_usd,
                    "bid_ask_spread_pct": pair.bid_ask_spread_pct,
                    "atr_pct": pair.atr_pct,
                    "paper_trades": pair.paper_trades,
                    "paper_wins": pair.paper_wins,
                    "paper_pnl": pair.paper_pnl,
                    "paper_sharpe": pair.paper_sharpe,
                    "discovered_at": pair.discovered_at,
                    "rejection_reasons": pair.rejection_reasons,
                }
                for symbol, pair in self.pairs.items()
            },
            "stats": self.get_universe_stats(),
            "thresholds": self.thresholds,
            "saved_at": time.time(),
        }
        
        try:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            logger.debug("Universe state saved to %s", state_path)
        except Exception as e:
            logger.warning("Failed to save universe state: %s", e)
    
    def load_state(self) -> bool:
        """Load universe state from disk."""
        state_path = os.path.join(self.data_dir, "universe_state.json")
        
        if not os.path.exists(state_path):
            return False
        
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            
            for symbol, data in state.get("pairs", {}).items():
                pair = PairMetrics(
                    symbol=symbol,
                    status=PairStatus(data.get("status", "candidate")),
                    avg_volume_usd=data.get("avg_volume_usd", 0),
                    bid_ask_spread_pct=data.get("bid_ask_spread_pct", 0),
                    atr_pct=data.get("atr_pct", 0),
                    paper_trades=data.get("paper_trades", 0),
                    paper_wins=data.get("paper_wins", 0),
                    paper_pnl=data.get("paper_pnl", 0),
                    paper_sharpe=data.get("paper_sharpe", 0),
                    discovered_at=data.get("discovered_at", 0),
                    rejection_reasons=data.get("rejection_reasons", []),
                )
                self.pairs[symbol] = pair
            
            logger.info("Universe state loaded: %d pairs", len(self.pairs))
            return True
            
        except Exception as e:
            logger.warning("Failed to load universe state: %s", e)
            return False
