"""
Volatility Surface Analyzer — real-time IV surface monitoring and signals.

Extends the base VolatilitySurface with real-time monitoring, regime detection,
and trading signal generation.

Example::

    analyzer = VolSurfaceAnalyzer()
    analyzer.update_option("BTC/USD", strike=50000, expiry_days=30, iv=0.65, option_type="call")
    signal = analyzer.get_signal("BTC/USD")
    print(signal.regime, signal.recommended_strategy)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptionPoint:
    """Single option IV data point."""
    symbol: str
    strike: float
    expiry_days: int
    iv: float
    option_type: str  # "call" or "put"
    volume: float
    open_interest: float
    timestamp: float


@dataclass
class VolSignal:
    """Volatility trading signal."""
    symbol: str
    timestamp: float
    regime: str  # "low", "normal", "elevated", "extreme"
    iv_rank: float  # 0-100
    skew: float  # Put - Call IV
    term_slope: float  # Long - Short IV
    recommended_strategy: str  # "sell_vol", "buy_vol", "sell_skew", "buy_skew"
    confidence: float
    expected_edge_pct: float
    reasoning: List[str] = field(default_factory=list)


@dataclass
class _SymbolState:
    options: Deque[OptionPoint] = field(default_factory=lambda: deque(maxlen=5000))
    price_history: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    iv_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    last_signal: Optional[VolSignal] = None


class VolSurfaceAnalyzer:
    """
    Real-time volatility surface analyzer for trading signals.

    Parameters
    ----------
    iv_history_size : int
        Size of IV history for rank calculation (default 1000).
    skew_threshold : float
        Skew threshold for signal generation (default 0.05).
    regime_boundaries : tuple
        (low, elevated, extreme) IV rank boundaries.
    """

    def __init__(
        self,
        iv_history_size: int = 1000,
        skew_threshold: float = 0.05,
        regime_boundaries: Tuple[float, float, float] = (25.0, 70.0, 85.0),
    ) -> None:
        self._iv_history_size = iv_history_size
        self._skew_threshold = skew_threshold
        self._regime_low, self._regime_elevated, self._regime_extreme = regime_boundaries
        self._states: Dict[str, _SymbolState] = {}

        logger.info(
            "VolSurfaceAnalyzer initialized: skew_thresh=%.1f%% regimes=[%.0f,%.0f,%.0f]",
            skew_threshold * 100, *regime_boundaries,
        )

    def update_option(
        self,
        symbol: str,
        strike: float,
        expiry_days: int,
        iv: float,
        option_type: str,
        volume: float = 0.0,
        open_interest: float = 0.0,
    ) -> None:
        """Update option IV data."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()

        state = self._states[symbol]
        
        point = OptionPoint(
            symbol=symbol,
            strike=strike,
            expiry_days=expiry_days,
            iv=iv,
            option_type=option_type.lower(),
            volume=volume,
            open_interest=open_interest,
            timestamp=time.time(),
        )
        
        state.options.append(point)
        self._analyze(symbol)

    def update_price(self, symbol: str, price: float) -> None:
        """Update underlying price."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        self._states[symbol].price_history.append(price)

    def _analyze(self, symbol: str) -> None:
        """Analyze volatility surface and generate signal."""
        state = self._states[symbol]
        
        if not state.price_history or len(state.options) < 10:
            return

        current_price = state.price_history[-1]
        options = list(state.options)
        
        # Filter to liquid options
        liquid = [o for o in options if o.volume > 0 or o.open_interest > 100]
        if len(liquid) < 5:
            liquid = options

        # Get ATM options (near current price)
        near_atm = [o for o in liquid if abs(o.strike / current_price - 1) < 0.05]
        if not near_atm:
            near_atm = liquid

        # Separate by type and expiry
        calls_30d = [o for o in near_atm if o.option_type == "call" and 20 <= o.expiry_days <= 40]
        puts_30d = [o for o in near_atm if o.option_type == "put" and 20 <= o.expiry_days <= 40]
        
        short_calls = [o for o in near_atm if o.option_type == "call" and o.expiry_days <= 14]
        long_calls = [o for o in near_atm if o.option_type == "call" and o.expiry_days >= 60]

        # Calculate metrics
        atm_call_iv = np.mean([o.iv for o in calls_30d]) if calls_30d else 0.5
        atm_put_iv = np.mean([o.iv for o in puts_30d]) if puts_30d else 0.5
        atm_iv = (atm_call_iv + atm_put_iv) / 2

        skew = atm_put_iv - atm_call_iv
        
        short_iv = np.mean([o.iv for o in short_calls]) if short_calls else atm_iv
        long_iv = np.mean([o.iv for o in long_calls]) if long_calls else atm_iv
        term_slope = long_iv - short_iv

        # IV rank from history
        state.iv_history.append(atm_iv)
        if len(state.iv_history) > self._iv_history_size:
            # Trim to size
            state.iv_history = deque(list(state.iv_history)[-self._iv_history_size:], 
                                     maxlen=self._iv_history_size)
        
        hist = list(state.iv_history)
        if len(hist) > 10:
            iv_rank = (atm_iv - min(hist)) / (max(hist) - min(hist) + 1e-9) * 100
        else:
            iv_rank = 50.0

        # Determine regime
        if iv_rank < self._regime_low:
            regime = "low"
        elif iv_rank < self._regime_elevated:
            regime = "normal"
        elif iv_rank < self._regime_extreme:
            regime = "elevated"
        else:
            regime = "extreme"

        # Generate strategy recommendation
        reasoning = []
        
        if regime in ["elevated", "extreme"]:
            if skew > self._skew_threshold:
                strategy = "sell_skew"
                reasoning.append(f"High IV ({iv_rank:.0f}th percentile) with put skew ({skew:.2%})")
                reasoning.append("Sell expensive puts, buy cheaper calls")
            else:
                strategy = "sell_vol"
                reasoning.append(f"High IV ({iv_rank:.0f}th percentile)")
                reasoning.append("Sell premium via straddles/strangles")
            confidence = min(1.0, iv_rank / 100)
            expected_edge = (iv_rank - 50) / 100 * 2  # 2% edge per 50 IV rank points
            
        elif regime == "low":
            if skew < -self._skew_threshold:
                strategy = "buy_skew"
                reasoning.append(f"Low IV ({iv_rank:.0f}th percentile) with call skew")
                reasoning.append("Buy cheap calls, sell expensive puts")
            else:
                strategy = "buy_vol"
                reasoning.append(f"Low IV ({iv_rank:.0f}th percentile)")
                reasoning.append("Buy options for volatility expansion")
            confidence = min(1.0, (100 - iv_rank) / 100)
            expected_edge = (50 - iv_rank) / 100 * 1.5
            
        else:  # normal
            if abs(skew) > self._skew_threshold * 2:
                strategy = "sell_skew" if skew > 0 else "buy_skew"
                reasoning.append(f"Skew opportunity: {skew:.2%}")
            else:
                strategy = "neutral"
                reasoning.append(f"Normal IV regime ({iv_rank:.0f}th percentile)")
            confidence = 0.5
            expected_edge = 0.5

        signal = VolSignal(
            symbol=symbol,
            timestamp=time.time(),
            regime=regime,
            iv_rank=iv_rank,
            skew=skew,
            term_slope=term_slope,
            recommended_strategy=strategy,
            confidence=confidence,
            expected_edge_pct=expected_edge,
            reasoning=reasoning,
        )

        state.last_signal = signal

    def get_signal(self, symbol: str) -> Optional[VolSignal]:
        """Get current volatility signal."""
        if symbol in self._states:
            return self._states[symbol].last_signal
        return None

    def get_iv_rank(self, symbol: str) -> float:
        """Get current IV rank."""
        signal = self.get_signal(symbol)
        return signal.iv_rank if signal else 50.0

    def get_regime(self, symbol: str) -> str:
        """Get volatility regime."""
        signal = self.get_signal(symbol)
        return signal.regime if signal else "normal"

    def should_sell_vol(self, symbol: str) -> bool:
        """Check if should sell volatility."""
        signal = self.get_signal(symbol)
        if not signal:
            return False
        return signal.recommended_strategy.startswith("sell") and signal.confidence > 0.6

    def should_buy_vol(self, symbol: str) -> bool:
        """Check if should buy volatility."""
        signal = self.get_signal(symbol)
        if not signal:
            return False
        return signal.recommended_strategy.startswith("buy") and signal.confidence > 0.6

    def get_all_signals(self) -> List[VolSignal]:
        """Get all current signals."""
        signals = []
        for state in self._states.values():
            if state.last_signal:
                signals.append(state.last_signal)
        return sorted(signals, key=lambda s: s.confidence, reverse=True)

    def get_all_symbols(self) -> List[str]:
        """Get all tracked symbols."""
        return sorted(self._states.keys())


__all__ = ["VolSurfaceAnalyzer", "VolSignal", "OptionPoint"]
