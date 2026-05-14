"""
Active Opportunity Scanner v1.0

Continuously scans for the BEST opportunities:
1. Monitors multiple symbols/pairs
2. Uses learning to adapt to what works NOW
3. Finds best setups in real-time
4. Auto-tunes to market conditions

Run: py scripts/opp_scanner.py
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np

logger = logging.getLogger(__name__)


class Opportunity:
    """A trading opportunity."""
    
    def __init__(self, symbol: str, signal: str, confidence: float, score: float, features: np.ndarray):
        self.symbol = symbol
        self.signal = signal  # buy/sell/hold
        self.confidence = confidence
        self.score = score
        self.features = features
        self.timestamp = datetime.now(timezone.utc)
        
        # Historical performance
        self.win_count = 0
        self.loss_count = 0
    
    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5
    
    def record_outcome(self, won: bool):
        if won:
            self.win_count += 1
        else:
            self.loss_count += 1


class ActiveScanner:
    """
    Active scanner that finds best opportunities.
    
    How it works:
    1. Scan multiple symbols
    2. Rate each by confidence + learning
    3. Pick best opportunity
    4. Adapt based on results
    """

    def __init__(
        self,
        symbols: List[str] = None,
        min_confidence: float = 0.50,
        max_opportunities: int = 3,
    ):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.min_confidence = min_confidence
        self.max_opportunities = max_opportunities
        
        # Opportunities cache
        self.opportunities: Dict[str, Opportunity] = {}
        
        # Best opportunity tracking
        self.best_opportunity: Optional[Opportunity] = None
        
        # Learning history
        self.history: deque = deque(maxlen=500)
        
        self._lock = threading.Lock()
        
        # Stats
        self.total_scans = 0
        self.best_wins = 0
        self.best_losses = 0
        
        # Feature weights (learned)
        self.weights = np.ones(9)
        
        # What's working NOW
        self.current_winning_signal = None
        
        logger.info("=" * 60)
        logger.info("ACTIVE OPPORTUNITY SCANNER")
        logger.info("=" * 60)
        logger.info("Symbols: {}".format(self.symbols))
        logger.info("Scan for best: {}".format(self.max_opportunities))
        logger.info("=" * 60)

    def scan(
        self,
        symbol_data: Dict[str, np.ndarray]
    ) -> Optional[Opportunity]:
        """
        Scan all symbols and find best opportunity.
        
        Args:
            symbol_data: {symbol: features_array}
        
        Returns:
            Best opportunity or None
        """
        with self._lock:
            self.total_scans += 1
            opportunities = []
            
            for symbol, features in symbol_data.items():
                opp = self._rate_opportunity(symbol, features)
                if opp and opp.confidence >= self.min_confidence:
                    opportunities.append(opp)
            
            if not opportunities:
                return None
            
            # Sort by score
            opportunities.sort(key=lambda x: x.score, reverse=True)
            
            # Take top N
            top = opportunities[:self.max_opportunities]
            
            if top:
                self.best_opportunity = top[0]
                
                # Store for learning
                self.opportunities[symbol] = top[0]
            
            return self.best_opportunity

    def _rate_opportunity(
        self,
        symbol: str,
        features: np.ndarray
    ) -> Optional[Opportunity]:
        """Rate a single opportunity."""
        # Extract signals from features
        r1 = features[0]  # 1-bar return
        r4 = features[1]  # 4-bar return
        r24 = features[3]  # 24-bar return
        v24 = features[5]  # volatility
        
        # Base signal
        if r1 > 0.005 and r4 > 0:
            signal = 'buy'
        elif r1 < -0.005 and r4 < 0:
            signal = 'sell'
        else:
            signal = 'hold'
        
        if signal == 'hold':
            return None
        
        # Confidence from features
        momentum_score = abs(r1) + abs(r4) * 0.5 + abs(r24) * 0.3
        
        # Learn what works
        learning_bonus = 0
        if self.current_winning_signal == signal:
            learning_bonus = 0.1
        
        # Volatility check
        if v24 > 0.05:
            return None  # Too volatile
        
        confidence = min(0.5 + momentum_score + learning_bonus, 0.85)
        
        # Score = confidence * momentum
        score = confidence * momentum_score
        
        opp = Opportunity(symbol, signal, confidence, score, features)
        
        return opp

    def record_result(self, won: bool):
        """Record outcome and adapt."""
        with self._lock:
            if not self.best_opportunity:
                return
            
            # Record for this opportunity
            self.best_opportunity.record_outcome(won)
            
            # Learn what works
            if won:
                self.best_wins += 1
                # This signal is working!
                self.current_winning_signal = self.best_opportunity.signal
                
                # Boost weights for winning features
                features = self.best_opportunity.features
                self.weights += 0.01 * features * np.sign(features)
            else:
                self.best_losses += 1
                # Reduce weights
                features = self.best_opportunity.features
                self.weights -= 0.005 * features * np.sign(features)
            
            # Store history
            self.history.append({
                'symbol': self.best_opportunity.symbol,
                'signal': self.best_opportunity.signal,
                'won': won,
                'timestamp': datetime.now(timezone.utc),
                'weights': self.weights.copy()
            })
            
            # Check for signal change
            if len(self.history) >= 10:
                self._adapt_to_signal()

    def _adapt_to_signal(self):
        """Adapt to what's working now."""
        recent = list(self.history)[-10:]
        
        buys = sum(1 for h in recent if h['signal'] == 'buy' and h['won'])
        sells = sum(1 for h in recent if h['signal'] == 'sell' and h['won'])
        
        if buys > sells:
            self.current_winning_signal = 'buy'
        elif sells > buys:
            self.current_winning_signal = 'sell'
        else:
            self.current_winning_signal = None

    def get_stats(self) -> Dict:
        """Get scanner stats."""
        total = self.best_wins + self.best_losses
        
        # What's working now
        recent = list(self.history)[-20:] if self.history else []
        recent_wins = sum(1 for h in recent if h['won'])
        recent_rate = recent_wins / len(recent) if recent else 0
        
        return {
            'total_scans': self.total_scans,
            'best_wins': self.best_wins,
            'best_losses': self.best_losses,
            'win_rate': self.best_wins / total if total > 0 else 0,
            'recent_win_rate': recent_rate,
            'current_winning_signal': self.current_winning_signal,
            'top_features': self._get_top_features()
        }

    def _get_top_features(self) -> List[str]:
        """Get most important features."""
        feature_names = ['r1', 'r4', 'r12', 'r24', 'v12', 'v24', 'rsi', 'pp', 'vr']
        importance = [(feature_names[i], abs(self.weights[i])) for i in range(9)]
        importance.sort(key=lambda x: x[1], reverse=True)
        return importance[:3]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print()
    print("=" * 60)
    print("ACTIVE OPPORTUNITY SCANNER TEST")
    print("=" * 60)
    print()
    
    # Create scanner
    scanner = ActiveScanner(
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT'],
        min_confidence=0.50,
        max_opportunities=2
    )
    
    # Simulate scanning
    np.random.seed(42)
    
    for cycle in range(50):
        # Generate features for each symbol
        data = {}
        for symbol in scanner.symbols:
            # Random features
            features = np.random.randn(9) * 0.02
            # Add some patterns
            if cycle % 7 == 0:
                features[0] = 0.02
            data[symbol] = features
        
        # Scan
        best = scanner.scan(data)
        
        if best and cycle > 10:
            # Simulate trade
            won = np.random.rand() > 0.5
            scanner.record_result(won)
        
        if cycle % 10 == 0:
            stats = scanner.get_stats()
            print("Cycle {:2d}: Best={} Won={} Signal={} Recent={:.0%}".format(
                cycle,
                best.symbol if best else None,
                stats['best_wins'],
                stats['current_winning_signal'],
                stats['recent_win_rate']))
    
    print()
    stats = scanner.get_stats()
    print("=" * 60)
    print("FINAL")
    print("=" * 60)
    print("Total scans: {}".format(stats['total_scans']))
    print("Wins: {} Losses: {}".format(stats['best_wins'], stats['best_losses']))
    print("Win rate: {:.0%}".format(stats['win_rate']))
    print("Current winning signal: {}".format(stats['current_winning_signal']))
    print("Recent win rate: {:.0%}".format(stats['recent_win_rate']))
    print("Top features: {}".format(stats['top_features']))