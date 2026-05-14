"""
Staking Optimizer — yield optimization for proof-of-stake assets.

Monitors staking APY across validators and protocols, recommends optimal
allocation for staking rewards while maintaining liquidity for trading.

Features:
- Multi-validator APY tracking
- Impermanent loss calculation for LP positions
- Optimal staking/trading capital split
- Validator health monitoring (uptime, commission changes)
- Auto-compounding recommendations

Example::

    optimizer = StakingOptimizer()
    optimizer.update_validator("SOL", "ValidatorA", apy=7.2, commission=0.05)
    optimizer.update_validator("SOL", "ValidatorB", apy=6.8, commission=0.02)
    recommendation = optimizer.get_recommendation("SOL", capital=10000)
    print(recommendation.best_validator, recommendation.recommended_stake)
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
class ValidatorInfo:
    """Validator performance and health metrics."""
    validator_id: str
    symbol: str
    apy: float  # Annual percentage yield
    commission: float  # Validator commission (0-1)
    uptime: float  # Uptime percentage (0-100)
    total_staked: float  # Total stake in USD
    active_stake: float  # Active stake
    last_epoch_rewards: float
    commission_history: List[float] = field(default_factory=list)
    timestamp: float = 0.0
    
    @property
    def net_apy(self) -> float:
        """APY after commission (as percentage, e.g., 6.84 for 6.84%)."""
        return self.apy * (1 - self.commission)
    
    @property
    def risk_score(self) -> float:
        """Risk score (0-1, lower is better)."""
        # Higher commission = higher risk
        # Lower uptime = higher risk
        # Extreme stake concentration = higher risk
        commission_risk = self.commission * 2
        uptime_risk = max(0, (100 - self.uptime) / 100)
        concentration_risk = min(1.0, self.total_staked / 1_000_000_000) * 0.3
        return min(1.0, commission_risk + uptime_risk + concentration_risk)


@dataclass
class StakingRecommendation:
    """Staking allocation recommendation."""
    symbol: str
    best_validator: str
    recommended_stake_pct: float  # % of capital to stake
    recommended_stake_amount: float  # USD amount to stake
    expected_apy: float  # Expected net APY
    expected_monthly_reward: float
    trading_reserve_pct: float  # % to keep for trading
    risk_score: float
    auto_compound: bool  # Whether to auto-compound rewards
    reasoning: List[str] = field(default_factory=list)


@dataclass
class _SymbolState:
    validators: Dict[str, ValidatorInfo] = field(default_factory=dict)
    price_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    last_recommendation: Optional[StakingRecommendation] = None


class StakingOptimizer:
    """
    Multi-symbol staking yield optimizer.

    Parameters
    ----------
    min_uptime : float
        Minimum validator uptime to consider (default 99%).
    max_commission : float
        Maximum validator commission to consider (default 10%).
    target_liquidity_pct : float
        Target % of capital to keep liquid for trading (default 30%).
    auto_compound_threshold : float
        Minimum reward to trigger auto-compound recommendation (default $10).
    """

    def __init__(
        self,
        min_uptime: float = 99.0,
        max_commission: float = 0.10,
        target_liquidity_pct: float = 0.30,
        auto_compound_threshold: float = 10.0,
    ) -> None:
        self._min_uptime = min_uptime
        self._max_commission = max_commission
        self._target_liquidity = target_liquidity_pct
        self._auto_compound_threshold = auto_compound_threshold
        self._states: Dict[str, _SymbolState] = {}

        logger.info(
            "StakingOptimizer initialized: min_uptime=%.1f%% max_commission=%.1f%% "
            "target_liquidity=%.1f%%",
            min_uptime, max_commission * 100, target_liquidity_pct * 100,
        )

    def update_validator(
        self,
        symbol: str,
        validator_id: str,
        apy: float,
        commission: float,
        uptime: float = 100.0,
        total_staked: float = 0.0,
        active_stake: float = 0.0,
        last_epoch_rewards: float = 0.0,
    ) -> None:
        """Update validator information."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()

        state = self._states[symbol]
        
        # Get existing commission history or start new
        existing = state.validators.get(validator_id)
        commission_history = existing.commission_history if existing else []
        commission_history.append(commission)
        if len(commission_history) > 100:
            commission_history = commission_history[-100:]

        state.validators[validator_id] = ValidatorInfo(
            validator_id=validator_id,
            symbol=symbol,
            apy=apy,
            commission=commission,
            uptime=uptime,
            total_staked=total_staked,
            active_stake=active_stake,
            last_epoch_rewards=last_epoch_rewards,
            commission_history=commission_history,
            timestamp=time.time(),
        )

    def update_price(self, symbol: str, price: float) -> None:
        """Update token price for USD calculations."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        self._states[symbol].price_history.append(price)

    def get_validators(self, symbol: str) -> List[ValidatorInfo]:
        """Get all validators for symbol, sorted by net APY."""
        if symbol not in self._states:
            return []
        
        validators = list(self._states[symbol].validators.values())
        
        # Filter by minimum requirements
        validators = [
            v for v in validators
            if v.uptime >= self._min_uptime
            and v.commission <= self._max_commission
        ]
        
        # Sort by net APY (descending)
        validators.sort(key=lambda v: v.net_apy, reverse=True)
        return validators

    def get_best_validator(self, symbol: str) -> Optional[ValidatorInfo]:
        """Get the best validator for symbol."""
        validators = self.get_validators(symbol)
        return validators[0] if validators else None

    def get_recommendation(
        self,
        symbol: str,
        capital: float,
        risk_tolerance: float = 0.5,
    ) -> Optional[StakingRecommendation]:
        """
        Get staking recommendation for symbol.

        Parameters
        ----------
        symbol : str
            Token symbol (e.g., "SOL", "ETH").
        capital : float
            Total capital in USD.
        risk_tolerance : float
            Risk tolerance (0-1, higher = more aggressive).

        Returns
        -------
        StakingRecommendation or None if no validators available.
        """
        validators = self.get_validators(symbol)
        if not validators:
            logger.debug("%s: no suitable validators found", symbol)
            return None

        # Get best validator
        best = validators[0]

        # Calculate optimal stake percentage
        # Higher risk tolerance = more staking, less trading reserve
        base_stake_pct = 1.0 - self._target_liquidity
        risk_adjusted_stake_pct = base_stake_pct * (0.7 + 0.6 * risk_tolerance)
        risk_adjusted_stake_pct = min(0.9, max(0.1, risk_adjusted_stake_pct))

        # Adjust for validator risk
        risk_adjusted_stake_pct *= (1 - best.risk_score * 0.3)

        # Calculate amounts
        stake_amount = capital * risk_adjusted_stake_pct
        trading_reserve = capital * (1 - risk_adjusted_stake_pct)

        # Calculate expected rewards (net_apy is percentage, convert to decimal for calculation)
        expected_monthly = stake_amount * (best.net_apy / 100) / 12

        # Determine auto-compound recommendation
        auto_compound = expected_monthly >= self._auto_compound_threshold

        # Build reasoning
        reasoning = []
        reasoning.append(f"Selected {best.validator_id} with net APY {best.net_apy:.2%}")
        reasoning.append(f"Commission: {best.commission:.1%}, Uptime: {best.uptime:.1f}%")
        if best.risk_score > 0.5:
            reasoning.append(f"Warning: Elevated risk score ({best.risk_score:.2f})")
        reasoning.append(f"Keeping {trading_reserve:.0f} USD liquid for trading")
        if auto_compound:
            reasoning.append(f"Auto-compound recommended (monthly reward > ${self._auto_compound_threshold})")

        recommendation = StakingRecommendation(
            symbol=symbol,
            best_validator=best.validator_id,
            recommended_stake_pct=risk_adjusted_stake_pct,
            recommended_stake_amount=stake_amount,
            expected_apy=best.net_apy,
            expected_monthly_reward=expected_monthly,
            trading_reserve_pct=1 - risk_adjusted_stake_pct,
            risk_score=best.risk_score,
            auto_compound=auto_compound,
            reasoning=reasoning,
        )

        self._states[symbol].last_recommendation = recommendation
        return recommendation

    def get_all_recommendations(
        self,
        total_capital: float,
        allocation_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, StakingRecommendation]:
        """Get staking recommendations for all tracked symbols."""
        if allocation_weights is None:
            # Equal weight by default
            symbols = list(self._states.keys())
            if not symbols:
                return {}
            weight = 1.0 / len(symbols)
            allocation_weights = {s: weight for s in symbols}

        recommendations = {}
        for symbol, weight in allocation_weights.items():
            capital = total_capital * weight
            rec = self.get_recommendation(symbol, capital)
            if rec:
                recommendations[symbol] = rec

        return recommendations

    def get_validator_health(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """Get health metrics for all validators of symbol."""
        if symbol not in self._states:
            return {}

        health = {}
        for vid, validator in self._states[symbol].validators.items():
            # Check for commission changes (potential rug pull indicator)
            commission_stability = 1.0
            if len(validator.commission_history) >= 3:
                recent = validator.commission_history[-3:]
                if max(recent) - min(recent) > 0.02:  # >2% change
                    commission_stability = 0.5

            health[vid] = {
                "uptime": validator.uptime,
                "commission": validator.commission,
                "commission_stability": commission_stability,
                "risk_score": validator.risk_score,
                "net_apy": validator.net_apy,
                "is_active": validator.active_stake > 0,
            }

        return health

    def detect_commission_change(self, symbol: str, validator_id: str) -> Optional[float]:
        """Detect significant commission changes (potential warning)."""
        if symbol not in self._states:
            return None
        
        validator = self._states[symbol].validators.get(validator_id)
        if not validator or len(validator.commission_history) < 2:
            return None

        recent = validator.commission_history[-5:] if len(validator.commission_history) >= 5 else validator.commission_history
        if len(recent) >= 2:
            change = recent[-1] - recent[0]
            if abs(change) > 0.01:  # >1% change
                logger.warning(
                    "%s: validator %s commission changed by %.2f%%",
                    symbol, validator_id, change * 100,
                )
                return change

        return None

    def get_all_symbols(self) -> List[str]:
        """Get all tracked symbols."""
        return sorted(self._states.keys())

    def calculate_il(
        self,
        entry_price: float,
        current_price: float,
        token_a_weight: float = 0.5,
    ) -> float:
        """
        Calculate impermanent loss for a 50/50 LP position.

        Parameters
        ----------
        entry_price : float
            Price when entering LP.
        current_price : float
            Current price.
        token_a_weight : float
            Weight of token A (default 0.5 for 50/50).

        Returns
        -------
        float
            Impermanent loss as percentage (negative = loss).
        """
        if entry_price <= 0 or current_price <= 0:
            return 0.0

        price_ratio = current_price / entry_price
        
        # IL formula for 50/50 LP
        il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        return float(il)


__all__ = ["StakingOptimizer", "ValidatorInfo", "StakingRecommendation"]
