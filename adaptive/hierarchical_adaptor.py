"""
Hierarchical Adaptation Engine for Argus Ultimate
=================================================
Implements 3-level adaptation:
1. Macro (1-24hr): Quantum + Causal Inference (regime detection)
2. Meso (1-60min): LSTM + RL (parameter tuning)
3. Micro (1-60s): Neuromorphic-inspired logic (execution optimization)

Dependencies:
- numpy
- pandas
- torch (for LSTM)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regimes for macro adaptation."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    PUMP = "pump"
    RANGING_TIGHT = "ranging_tight"
    RANGING_WIDE = "ranging_wide"
    BREAKOUT_PENDING = "breakout_pending"
    REVERSAL_PENDING = "reversal_pending"


@dataclass
class MacroState:
    """Macro-level adaptation state (1-24hr)."""
    regime: MarketRegime
    confidence: float
    predicted_regime: Optional[MarketRegime] = None
    transition_probability: float = 0.0
    time_in_regime: float = 0.0  # in hours


@dataclass
class MesoState:
    """Meso-level adaptation state (1-60min)."""
    position_size: float
    strategy: str
    confidence: float
    volatility: float
    momentum: float


@dataclass
class MicroState:
    """Micro-level adaptation state (1-60s)."""
    execution_style: str  # e.g., "market", "TWAP", "VWAP", "sniper"
    slippage_tolerance: float
    latency: float  # in seconds


@dataclass
class AdaptationDecision:
    """Final adaptation decision."""
    macro: MacroState
    meso: MesoState
    micro: MicroState
    timestamp: float
    reasoning: str


class MacroAdaptor:
    """
    Macro-level adaptation (1-24hr).
    Uses regime detection with classical indicators.
    Can be extended with quantum or causal inference.
    """
    def __init__(self):
        self.current_regime = MarketRegime.RANGING_TIGHT
        self.regime_history: deque = deque(maxlen=100)
        self.transition_matrix: Dict[str, Dict[str, float]] = {}
        self.causal_engine: Optional[Any] = None  # Will be set externally

    def detect_regime(self, market_data: Dict[str, Any]) -> MacroState:
        """
        Detect current market regime using classical indicators.
        Args:
            market_data: Dict with prices, volumes, etc.
        Returns:
            MacroState with detected regime and metadata
        """
        # Extract features
        prices = market_data.get("prices", [])
        if len(prices) < 20:
            return MacroState(
                regime=self.current_regime,
                confidence=0.5,
            )

        prices_arr = np.array(prices[-100:])
        returns = np.diff(np.log(prices_arr))

        # Calculate indicators
        trend_strength = (prices_arr[-1] - prices_arr[-20]) / prices_arr[-20] if prices_arr[-20] != 0 else 0
        volatility = float(np.std(returns) * np.sqrt(252 * 24))  # Annualized
        momentum = (prices_arr[-1] - prices_arr[-5]) / prices_arr[-5] if prices_arr[-5] != 0 else 0
        volume_ratio = market_data.get("volume_ratio", 1.0)

        # Simple regime detection (replace with quantum if available)
        if volatility > 0.8:
            regime = MarketRegime.HIGH_VOLATILITY
        elif trend_strength > 0.05:
            regime = MarketRegime.STRONG_UPTREND
        elif trend_strength > 0.02:
            regime = MarketRegime.WEAK_UPTREND
        elif trend_strength < -0.05:
            regime = MarketRegime.STRONG_DOWNTREND
        elif trend_strength < -0.02:
            regime = MarketRegime.WEAK_DOWNTREND
        elif abs(momentum) < 0.01:
            regime = MarketRegime.RANGING_TIGHT
        else:
            regime = MarketRegime.RANGING_WIDE

        # Update history
        self.regime_history.append({
            "regime": regime,
            "timestamp": pd.Timestamp.now(),
            "trend_strength": trend_strength,
            "volatility": volatility,
        })

        # Update transition matrix
        if len(self.regime_history) >= 2:
            prev = self.regime_history[-2]["regime"]
            curr = self.regime_history[-1]["regime"]
            if prev.value not in self.transition_matrix:
                self.transition_matrix[prev.value] = {}
            if curr.value not in self.transition_matrix[prev.value]:
                self.transition_matrix[prev.value][curr.value] = 0
            self.transition_matrix[prev.value][curr.value] += 1

        # Predict next regime
        predicted_regime = self._predict_next_regime(regime)
        transition_prob = self._get_transition_probability(regime, predicted_regime) if predicted_regime else 0.0

        # Time in current regime
        time_in_regime = 0.0
        if len(self.regime_history) > 1:
            regime_start_idx = next(
                (i for i, r in enumerate(reversed(self.regime_history)) if r["regime"] != regime),
                len(self.regime_history) - 1
            )
            regime_start_time = self.regime_history[-regime_start_idx]["timestamp"]
            time_in_regime = (pd.Timestamp.now() - regime_start_time).total_seconds() / 3600

        self.current_regime = regime

        return MacroState(
            regime=regime,
            confidence=0.8,  # Placeholder (use quantum confidence if available)
            predicted_regime=predicted_regime,
            transition_probability=transition_prob,
            time_in_regime=time_in_regime,
        )

    def _predict_next_regime(self, current_regime: MarketRegime) -> Optional[MarketRegime]:
        """Predict next regime based on transition matrix."""
        if current_regime.value not in self.transition_matrix:
            return None

        transitions = self.transition_matrix[current_regime.value]
        if not transitions:
            return None

        total = sum(transitions.values())
        if total == 0:
            return None

        # Get most likely next regime
        next_regime = max(transitions, key=transitions.get)
        return MarketRegime(next_regime)

    def _get_transition_probability(self, current: MarketRegime, next_regime: MarketRegime) -> float:
        """Get probability of transitioning to next_regime."""
        if current.value not in self.transition_matrix:
            return 0.0
        if next_regime.value not in self.transition_matrix[current.value]:
            return 0.0

        total = sum(self.transition_matrix[current.value].values())
        return self.transition_matrix[current.value][next_regime.value] / total if total > 0 else 0.0


class MesoAdaptor:
    """
    Meso-level adaptation (1-60min).
    Uses classical indicators for parameter tuning.
    Can be extended with LSTM or RL.
    """
    def __init__(self):
        self.position_size = 0.1
        self.strategy = "momentum"
        self.strategy_weights: Dict[str, float] = {
            "momentum": 0.4,
            "mean_reversion": 0.3,
            "breakout": 0.2,
            "scalping": 0.1,
        }
        self.volatility = 0.1
        self.momentum = 0.0

    def tune_parameters(
        self,
        market_data: Dict[str, Any],
        macro_state: MacroState,
    ) -> MesoState:
        """
        Tune meso-level parameters based on market conditions and macro regime.
        Args:
            market_data: Dict with prices, volumes, etc.
            macro_state: Current macro state
        Returns:
            MesoState with tuned parameters
        """
        prices = market_data.get("prices", [])
        if len(prices) < 20:
            return MesoState(
                position_size=self.position_size,
                strategy=self.strategy,
                confidence=0.5,
                volatility=self.volatility,
                momentum=self.momentum,
            )

        prices_arr = np.array(prices[-60:])
        returns = np.diff(np.log(prices_arr))

        # Update volatility and momentum
        self.volatility = float(np.std(returns) * np.sqrt(252 * 24 * 60))  # Hourly annualized
        self.momentum = (prices_arr[-1] - prices_arr[-5]) / prices_arr[-5] if prices_arr[-5] != 0 else 0

        # Adjust position size based on volatility and regime
        if macro_state.regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRASH]:
            self.position_size = max(0.05, min(0.15, 0.2 - self.volatility * 0.1))
        elif macro_state.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.PUMP]:
            self.position_size = min(0.25, 0.1 + self.volatility * 0.05)
        else:
            self.position_size = 0.1

        # Adjust strategy weights based on regime
        if macro_state.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND]:
            self.strategy_weights = {
                "momentum": 0.6,
                "mean_reversion": 0.1,
                "breakout": 0.2,
                "scalping": 0.1,
            }
            self.strategy = "momentum"
        elif macro_state.regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND]:
            self.strategy_weights = {
                "momentum": 0.1,
                "mean_reversion": 0.5,
                "breakout": 0.1,
                "scalping": 0.3,
            }
            self.strategy = "mean_reversion"
        elif macro_state.regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRASH]:
            self.strategy_weights = {
                "momentum": 0.2,
                "mean_reversion": 0.2,
                "breakout": 0.4,
                "scalping": 0.2,
            }
            self.strategy = "breakout"
        else:
            self.strategy_weights = {
                "momentum": 0.3,
                "mean_reversion": 0.3,
                "breakout": 0.2,
                "scalping": 0.2,
            }
            self.strategy = "momentum"

        return MesoState(
            position_size=self.position_size,
            strategy=self.strategy,
            confidence=0.8,
            volatility=self.volatility,
            momentum=self.momentum,
        )


class MicroAdaptor:
    """
    Micro-level adaptation (1-60s).
    Uses neuromorphic-inspired logic for execution optimization.
    """
    def __init__(self):
        self.execution_style = "market"
        self.slippage_tolerance = 0.005  # 0.5%
        self.latency = 0.1  # 100ms
        self.orderbook_history: deque = deque(maxlen=100)

    def optimize_execution(
        self,
        market_data: Dict[str, Any],
        meso_state: MesoState,
    ) -> MicroState:
        """
        Optimize micro-level execution based on order book state and meso parameters.
        Args:
            market_data: Dict with orderbook, etc.
            meso_state: Current meso state
        Returns:
            MicroState with optimized execution parameters
        """
        orderbook = market_data.get("orderbook", {})
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return MicroState(
                execution_style=self.execution_style,
                slippage_tolerance=self.slippage_tolerance,
                latency=self.latency,
            )

        # Calculate spread and depth
        best_bid = bids[0][0] if isinstance(bids[0], list) else bids[0]
        best_ask = asks[0][0] if isinstance(asks[0], list) else asks[0]
        mid_price = (best_bid + best_ask) / 2 if mid_price != 0 else 1.0
        spread = (best_ask - best_bid) / mid_price if mid_price != 0 else 0.0
        depth = sum(size for _, size in bids[:5]) + sum(size for _, size in asks[:5])

        # Update orderbook history
        self.orderbook_history.append({
            "spread": spread,
            "depth": depth,
            "timestamp": pd.Timestamp.now(),
        })

        # Adjust execution style
        if spread > 0.01:  # 1% spread
            self.execution_style = "sniper"  # Find hidden liquidity
            self.slippage_tolerance = 0.002  # 0.2%
        elif meso_state.position_size > 0.15:
            self.execution_style = "TWAP"  # Large order, spread over time
            self.slippage_tolerance = 0.003  # 0.3%
        elif depth < 1000:  # Thin order book
            self.execution_style = "VWAP"  # Volume-weighted
            self.slippage_tolerance = 0.004  # 0.4%
        else:
            self.execution_style = "market"  # Immediate execution
            self.slippage_tolerance = 0.005  # 0.5%

        # Adjust latency based on execution style
        if self.execution_style == "sniper":
            self.latency = 0.5  # 500ms (wait for liquidity)
        elif self.execution_style == "TWAP":
            self.latency = 1.0  # 1s (spread over time)
        else:
            self.latency = 0.1  # 100ms (immediate)

        return MicroState(
            execution_style=self.execution_style,
            slippage_tolerance=self.slippage_tolerance,
            latency=self.latency,
        )


class HierarchicalAdaptor:
    """
    Hierarchical Adaptation Engine.
    Combines:
    1. Macro (1-24hr): Regime detection
    2. Meso (1-60min): Parameter tuning
    3. Micro (1-60s): Execution optimization
    """
    def __init__(self):
        self.macro = MacroAdaptor()
        self.meso = MesoAdaptor()
        self.micro = MicroAdaptor()
        self.decision_history: deque = deque(maxlen=1000)

    def set_causal_engine(self, causal_engine: Any):
        """Set external causal engine for macro adaptation."""
        self.macro.causal_engine = causal_engine

    def adapt(self, market_data: Dict[str, Any]) -> AdaptationDecision:
        """
        Full hierarchical adaptation cycle.
        Args:
            market_data: Dict with prices, orderbook, volumes, etc.
        Returns:
            AdaptationDecision with macro, meso, micro states
        """
        # Macro adaptation (1-24hr)
        macro_state = self.macro.detect_regime(market_data)

        # Meso adaptation (1-60min)
        meso_state = self.meso.tune_parameters(market_data, macro_state)

        # Micro adaptation (1-60s)
        micro_state = self.micro.optimize_execution(market_data, meso_state)

        # Generate reasoning
        reasoning = (
            f"Macro: {macro_state.regime.value} (conf: {macro_state.confidence:.2f}), "
            f"Meso: {meso_state.strategy} (pos: {meso_state.position_size:.2f}), "
            f"Micro: {micro_state.execution_style} (slippage: {micro_state.slippage_tolerance:.4f})"
        )

        # Log decision
        decision = AdaptationDecision(
            macro=macro_state,
            meso=meso_state,
            micro=micro_state,
            timestamp=pd.Timestamp.now().timestamp(),
            reasoning=reasoning,
        )
        self.decision_history.append(decision)

        return decision

    def get_history(self) -> List[AdaptationDecision]:
        """Get history of adaptation decisions."""
        return list(self.decision_history)

    def get_current_state(self) -> Optional[AdaptationDecision]:
        """Get the most recent adaptation decision."""
        return self.decision_history[-1] if self.decision_history else None
