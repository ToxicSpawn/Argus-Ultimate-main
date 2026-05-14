"""
Sandwich Attack Detector — MEV protection for DEX trades.

Detects potential sandwich attacks by monitoring:
- Mempool pending transactions
- Unusual gas price spikes
- Large pending swaps on the same pool
- Front-running patterns

Provides recommendations for:
- Optimal execution timing
- Slippage adjustment
- Private transaction routing (Flashbots Protect)
- Trade size splitting

Example::

    detector = SandwichAttackDetector()
    detector.update_mempool("ETH", pending_large_swap=True, gas_spike=True)
    risk = detector.get_risk("ETH")
    if risk.level == "HIGH":
        recommendation = detector.get_recommendation("ETH")
        print(recommendation.action)  # "use_private_rpc" or "split_trade"
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
class MempoolSignal:
    """Mempool observation for MEV detection."""
    timestamp: float
    symbol: str
    pending_swap_value_usd: float
    gas_price_gwei: float
    gas_spike_ratio: float  # Current gas / average gas
    large_swap_detected: bool
    frontrun_tx_count: int  # Number of similar txs before ours
    pool_liquidity_usd: float


@dataclass
class MEVRisk:
    """MEV risk assessment."""
    symbol: str
    level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float  # 0-1
    sandwich_probability: float  # 0-1
    estimated_loss_pct: float  # Expected loss from MEV
    signals: List[str] = field(default_factory=list)


@dataclass
class MEVRecommendation:
    """MEV protection recommendation."""
    symbol: str
    action: str  # "proceed", "wait", "split_trade", "use_private_rpc", "cancel"
    slippage_adjustment: float  # Additional slippage to add (e.g., 0.01 = +1%)
    wait_seconds: int  # Recommended wait time
    split_into: int  # Number of splits if splitting
    use_private_rpc: bool  # Use Flashbots/private RPC
    reasoning: List[str] = field(default_factory=list)


@dataclass
class _SymbolState:
    mempool_history: Deque[MempoolSignal] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    gas_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    last_risk: Optional[MEVRisk] = None
    last_recommendation: Optional[MEVRecommendation] = None


class SandwichAttackDetector:
    """
    MEV/Sandwich attack detector for DEX trading.

    Parameters
    ----------
    gas_spike_threshold : float
        Gas price ratio to consider a spike (default 2.0x).
    large_swap_threshold_usd : float
        Swap value to consider "large" (default $50000).
    frontrun_threshold : int
        Number of similar txs to indicate frontrunning (default 3).
    risk_decay_minutes : float
        Minutes for risk to decay after signal (default 5).
    """

    def __init__(
        self,
        gas_spike_threshold: float = 2.0,
        large_swap_threshold_usd: float = 50000.0,
        frontrun_threshold: int = 3,
        risk_decay_minutes: float = 5.0,
    ) -> None:
        self._gas_spike_threshold = gas_spike_threshold
        self._large_swap_threshold = large_swap_threshold_usd
        self._frontrun_threshold = frontrun_threshold
        self._risk_decay_minutes = risk_decay_minutes
        self._states: Dict[str, _SymbolState] = {}

        logger.info(
            "SandwichAttackDetector initialized: gas_spike=%.1fx large_swap=$%.0fk "
            "frontrun_thresh=%d",
            gas_spike_threshold, large_swap_threshold_usd / 1000, frontrun_threshold,
        )

    def update_mempool(
        self,
        symbol: str,
        pending_swap_value_usd: float = 0.0,
        gas_price_gwei: float = 0.0,
        gas_spike_ratio: float = 1.0,
        large_swap_detected: bool = False,
        frontrun_tx_count: int = 0,
        pool_liquidity_usd: float = 0.0,
    ) -> None:
        """Update mempool observation for symbol."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()

        state = self._states[symbol]

        signal = MempoolSignal(
            timestamp=time.time(),
            symbol=symbol,
            pending_swap_value_usd=pending_swap_value_usd,
            gas_price_gwei=gas_price_gwei,
            gas_spike_ratio=gas_spike_ratio,
            large_swap_detected=large_swap_detected,
            frontrun_tx_count=frontrun_tx_count,
            pool_liquidity_usd=pool_liquidity_usd,
        )

        state.mempool_history.append(signal)
        if gas_price_gwei > 0:
            state.gas_history.append(gas_price_gwei)

    def update_trade_intent(
        self,
        symbol: str,
        trade_value_usd: float,
        pool_liquidity_usd: float,
    ) -> MEVRisk:
        """
        Assess MEV risk for an intended trade.

        Call this before executing a trade to get risk assessment.
        """
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()

        state = self._states[symbol]
        signals = []
        risk_score = 0.0

        # Check recent mempool signals (last 2 minutes)
        recent_signals = [
            s for s in state.mempool_history
            if time.time() - s.timestamp < 120
        ]

        # Signal 1: Gas spike
        if recent_signals:
            avg_gas = np.mean([s.gas_price_gwei for s in recent_signals if s.gas_price_gwei > 0])
            latest_gas = recent_signals[-1].gas_price_gwei if recent_signals[-1].gas_price_gwei > 0 else 0
            if avg_gas > 0 and latest_gas / avg_gas > self._gas_spike_threshold:
                signals.append("gas_spike")
                risk_score += 0.3

        # Signal 2: Large pending swaps
        large_swaps = [s for s in recent_signals if s.large_swap_detected]
        if large_swaps:
            signals.append("large_pending_swap")
            risk_score += 0.3

        # Signal 3: Frontrun tx count
        high_frontrun = [s for s in recent_signals if s.frontrun_tx_count >= self._frontrun_threshold]
        if high_frontrun:
            signals.append("frontrun_detected")
            risk_score += 0.4

        # Signal 4: Trade size vs pool liquidity (slippage risk)
        if pool_liquidity_usd > 0:
            trade_impact = trade_value_usd / pool_liquidity_usd
            if trade_impact > 0.01:  # >1% of pool
                signals.append("high_impact")
                risk_score += min(0.4, trade_impact * 10)
            if trade_impact > 0.05:  # >5% of pool
                signals.append("very_high_impact")
                risk_score += 0.3

        # Signal 5: Recent large swaps in same direction
        recent_large = [
            s for s in recent_signals
            if s.pending_swap_value_usd > self._large_swap_threshold
        ]
        if len(recent_large) >= 2:
            signals.append("multiple_large_swaps")
            risk_score += 0.2

        # Cap risk score
        risk_score = min(1.0, risk_score)

        # Determine risk level
        if risk_score >= 0.8:
            level = "CRITICAL"
        elif risk_score >= 0.6:
            level = "HIGH"
        elif risk_score >= 0.3:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Estimate potential loss from sandwich
        sandwich_probability = risk_score * 0.8  # Conservative estimate
        estimated_loss_pct = sandwich_probability * 0.02 * (trade_value_usd / max(1, pool_liquidity_usd) * 100)
        estimated_loss_pct = min(0.05, estimated_loss_pct)  # Cap at 5%

        risk = MEVRisk(
            symbol=symbol,
            level=level,
            risk_score=risk_score,
            sandwich_probability=sandwich_probability,
            estimated_loss_pct=estimated_loss_pct,
            signals=signals,
        )

        state.last_risk = risk
        return risk

    def get_risk(self, symbol: str) -> Optional[MEVRisk]:
        """Get last risk assessment for symbol."""
        if symbol in self._states:
            return self._states[symbol].last_risk
        return None

    def get_recommendation(
        self,
        symbol: str,
        trade_value_usd: float,
        pool_liquidity_usd: float,
    ) -> MEVRecommendation:
        """Get MEV protection recommendation."""
        risk = self.update_trade_intent(symbol, trade_value_usd, pool_liquidity_usd)
        state = self._states[symbol]

        reasoning = []
        action = "proceed"
        slippage_adjustment = 0.0
        wait_seconds = 0
        split_into = 1
        use_private_rpc = False

        if risk.level == "CRITICAL":
            action = "cancel"
            reasoning.append("Critical MEV risk detected - consider canceling trade")
            reasoning.append(f"Signals: {', '.join(risk.signals)}")

        elif risk.level == "HIGH":
            action = "use_private_rpc"
            use_private_rpc = True
            slippage_adjustment = 0.01  # +1% slippage
            split_into = 3
            reasoning.append("High MEV risk - use private RPC (Flashbots)")
            reasoning.append(f"Split trade into {split_into} parts")
            reasoning.append(f"Add {slippage_adjustment*100:.0f}% slippage buffer")

        elif risk.level == "MEDIUM":
            if "gas_spike" in risk.signals:
                action = "wait"
                wait_seconds = 60
                reasoning.append("Gas spike detected - wait 60 seconds")
            else:
                action = "split_trade"
                split_into = 2
                reasoning.append("Medium risk - split trade into 2 parts")
            slippage_adjustment = 0.005  # +0.5% slippage

        else:  # LOW
            action = "proceed"
            reasoning.append("Low MEV risk - safe to proceed")
            slippage_adjustment = 0.001  # +0.1% buffer

        # Check if private RPC is generally recommended for this symbol
        recent_risks = [
            s.last_risk for s in [state]
            if s.last_risk is not None
        ]
        high_risk_count = sum(1 for r in recent_risks if r.level in ["HIGH", "CRITICAL"])
        if high_risk_count >= 3:
            use_private_rpc = True
            reasoning.append("Frequent high MEV risk - recommend always using private RPC")

        recommendation = MEVRecommendation(
            symbol=symbol,
            action=action,
            slippage_adjustment=slippage_adjustment,
            wait_seconds=wait_seconds,
            split_into=split_into,
            use_private_rpc=use_private_rpc,
            reasoning=reasoning,
        )

        state.last_recommendation = recommendation
        return recommendation

    def get_safe_slippage(
        self,
        symbol: str,
        base_slippage: float,
        trade_value_usd: float,
        pool_liquidity_usd: float,
    ) -> float:
        """Calculate safe slippage including MEV protection buffer."""
        risk = self.update_trade_intent(symbol, trade_value_usd, pool_liquidity_usd)
        
        # Add risk-based buffer
        mev_buffer = risk.estimated_loss_pct
        
        # Add impact-based buffer
        impact_buffer = 0.0
        if pool_liquidity_usd > 0:
            impact = trade_value_usd / pool_liquidity_usd
            if impact > 0.01:
                impact_buffer = impact * 2  # 2x the expected impact

        safe_slippage = base_slippage + mev_buffer + impact_buffer
        return min(0.10, safe_slippage)  # Cap at 10%

    def get_optimal_timing(self, symbol: str) -> Dict[str, any]:
        """Get optimal trade timing recommendation."""
        if symbol not in self._states:
            return {"recommendation": "no_data", "wait_seconds": 0}

        state = self._states[symbol]
        
        # Analyze gas patterns
        if len(state.gas_history) < 10:
            return {"recommendation": "insufficient_data", "wait_seconds": 0}

        gas_list = list(state.gas_history)
        current_gas = gas_list[-1]
        avg_gas = np.mean(gas_list)
        min_gas = np.min(gas_list)
        max_gas = np.max(gas_list)

        # Determine timing
        if current_gas > avg_gas * 1.5:
            # Gas is high, recommend waiting
            return {
                "recommendation": "wait",
                "wait_seconds": 300,  # 5 minutes
                "reason": f"Gas {current_gas:.1f} gwei is {current_gas/avg_gas:.1f}x average",
                "current_gas": current_gas,
                "average_gas": avg_gas,
            }
        elif current_gas <= avg_gas * 0.8:
            # Gas is low, good time to trade
            return {
                "recommendation": "execute_now",
                "wait_seconds": 0,
                "reason": f"Gas {current_gas:.1f} gwei is below average",
                "current_gas": current_gas,
                "average_gas": avg_gas,
            }
        else:
            # Gas is normal
            return {
                "recommendation": "proceed",
                "wait_seconds": 0,
                "reason": f"Gas {current_gas:.1f} gwei is near average",
                "current_gas": current_gas,
                "average_gas": avg_gas,
            }

    def get_all_symbols(self) -> List[str]:
        """Get all tracked symbols."""
        return sorted(self._states.keys())


__all__ = ["SandwichAttackDetector", "MEVRisk", "MEVRecommendation"]
