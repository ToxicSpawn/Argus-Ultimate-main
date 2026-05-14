"""
Neuromodulated Risk Manager for Argus Ultimate
===============================================
Dynamically adjusts risk limits based on neuromodulation:
- Dopamine: Reward/motivation (increases risk when winning)
- Serotonin: Mood/risk sensitivity (decreases risk in volatile markets)
- Norepinephrine: Alertness/arousal (freezes trading during shocks)

Dependencies:
- numpy
- pandas
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class NeuromodulatorState:
    """State of a neuromodulator."""
    level: float  # 0-1 scale
    baseline: float = 0.5
    decay_rate: float = 0.01  # Decay per timestep
    max_level: float = 1.0
    min_level: float = 0.0


@dataclass
class RiskLimits:
    """Current risk limits."""
    max_position_size: float  # % of capital
    max_drawdown: float  # % of capital
    stop_loss: float  # % from entry
    take_profit: float  # % from entry
    max_leverage: float
    allowed_strategies: List[str]


class NeuromodulatedRiskManager:
    """
    Neuromodulated Risk Manager.
    Adjusts risk limits based on:
    1. Dopamine (reward/motivation)
    2. Serotonin (mood/risk sensitivity)
    3. Norepinephrine (alertness/arousal)
    """
    def __init__(
        self,
        initial_capital: float = 10000.0,
        baseline_risk: float = 0.1,  # 10% of capital
    ):
        self.initial_capital = initial_capital
        self.baseline_risk = baseline_risk

        # Neuromodulators
        self.dopamine = NeuromodulatorState(level=0.5, baseline=0.5)
        self.serotonin = NeuromodulatorState(level=0.5, baseline=0.5)
        self.norepinephrine = NeuromodulatorState(level=0.1, baseline=0.1)

        # Risk limits
        self.current_limits = RiskLimits(
            max_position_size=0.1,
            max_drawdown=0.2,
            stop_loss=0.02,
            take_profit=0.04,
            max_leverage=2.0,
            allowed_strategies=["momentum", "mean_reversion", "breakout"],
        )

        # History
        self.pnl_history: deque = deque(maxlen=100)
        self.volatility_history: deque = deque(maxlen=100)
        self.shock_history: deque = deque(maxlen=10)

        logger.info("NeuromodulatedRiskManager initialized")

    def update(
        self,
        pnl: float,
        volatility: float,
        is_shock: bool = False,
        win: bool = False,
    ):
        """
        Update neuromodulators and risk limits.
        Args:
            pnl: Recent PnL (positive for gains, negative for losses)
            volatility: Current market volatility (0-1 scale)
            is_shock: Whether a market shock was detected
            win: Whether the last trade was a win
        """
        # Update histories
        self.pnl_history.append(pnl)
        self.volatility_history.append(volatility)
        if is_shock:
            self.shock_history.append(pd.Timestamp.now())

        # Update dopamine (reward)
        if win:
            self.dopamine.level = min(
                self.dopamine.max_level,
                self.dopamine.level + 0.1 * (pnl / self.initial_capital)
            )
        else:
            self.dopamine.level = max(
                self.dopamine.min_level,
                self.dopamine.level - 0.05 * (abs(pnl) / self.initial_capital)
            )

        # Update serotonin (mood/risk sensitivity)
        # Higher volatility -> lower serotonin (more cautious)
        self.serotonin.level = max(
            self.serotonin.min_level,
            self.serotonin.baseline - 0.3 * volatility
        )

        # Update norepinephrine (alertness/arousal)
        if is_shock:
            self.norepinephrine.level = self.norepinephrine.max_level
        else:
            self.norepinephrine.level = max(
                self.norepinephrine.min_level,
                self.norepinephrine.level - self.norepinephrine.decay_rate
            )

        # Decay all neuromodulators
        self._decay_neuromodulators()

        # Update risk limits
        self._update_risk_limits()

    def _decay_neuromodulators(self):
        """Decay neuromodulator levels toward baseline."""
        self.dopamine.level = self.dopamine.baseline + (
            self.dopamine.level - self.dopamine.baseline
        ) * (1 - self.dopamine.decay_rate)

        self.serotonin.level = self.serotonin.baseline + (
            self.serotonin.level - self.serotonin.baseline
        ) * (1 - self.serotonin.decay_rate)

        self.norepinephrine.level = self.norepinephrine.baseline + (
            self.norepinephrine.level - self.norepinephrine.baseline
        ) * (1 - self.norepinephrine.decay_rate)

    def _update_risk_limits(self):
        """Update risk limits based on neuromodulator levels."""
        # Base risk adjustment
        dopamine_effect = self.dopamine.level - self.dopamine.baseline  # -0.5 to 0.5
        serotonin_effect = self.serotonin.level - self.serotonin.baseline  # -0.5 to 0.5
        norepinephrine_effect = self.norepinephrine.level - self.norepinephrine.baseline  # -0.9 to 0.9

        # Calculate risk multiplier (0.1 to 2.0)
        risk_multiplier = 1.0 + (
            0.5 * dopamine_effect -  # More dopamine -> higher risk
            0.3 * serotonin_effect -  # More serotonin -> lower risk
            1.0 * norepinephrine_effect  # More norepinephrine -> much lower risk (shock response)
        )

        # Clamp risk multiplier
        risk_multiplier = max(0.1, min(2.0, risk_multiplier))

        # Update limits
        self.current_limits.max_position_size = max(
            0.01, min(0.5, self.baseline_risk * risk_multiplier * 2)
        )
        self.current_limits.max_drawdown = max(
            0.05, min(0.5, 0.2 * risk_multiplier)
        )
        self.current_limits.stop_loss = max(
            0.01, min(0.1, 0.02 * (1 + norepinephrine_effect * 2))
        )
        self.current_limits.take_profit = max(
            0.02, min(0.2, 0.04 * (1 + dopamine_effect))
        )
        self.current_limits.max_leverage = max(
            1.0, min(5.0, 2.0 * risk_multiplier)
        )

        # Update allowed strategies
        if self.norepinephrine.level > 0.8:  # Shock detected
            self.current_limits.allowed_strategies = ["scalping"]  # Only low-risk strategies
        elif self.serotonin.level < 0.3:  # High volatility
            self.current_limits.allowed_strategies = ["momentum", "scalping"]
        else:
            self.current_limits.allowed_strategies = [
                "momentum", "mean_reversion", "breakout", "scalping"
            ]

    def get_limits(self) -> RiskLimits:
        """Get current risk limits."""
        return self.current_limits

    def get_neuromodulator_levels(self) -> Dict[str, float]:
        """Get current neuromodulator levels."""
        return {
            "dopamine": self.dopamine.level,
            "serotonin": self.serotonin.level,
            "norepinephrine": self.norepinephrine.level,
        }

    def get_risk_multiplier(self) -> float:
        """Get current risk multiplier (0.1 to 2.0)."""
        dopamine_effect = self.dopamine.level - self.dopamine.baseline
        serotonin_effect = self.serotonin.level - self.serotonin.baseline
        norepinephrine_effect = self.norepinephrine.level - self.norepinephrine.baseline

        risk_multiplier = 1.0 + (
            0.5 * dopamine_effect -
            0.3 * serotonin_effect -
            1.0 * norepinephrine_effect
        )
        return max(0.1, min(2.0, risk_multiplier))

    def is_shock_detected(self) -> bool:
        """Check if a shock was recently detected."""
        return self.norepinephrine.level > 0.8

    def reset(self):
        """Reset to initial state."""
        self.dopamine.level = self.dopamine.baseline
        self.serotonin.level = self.serotonin.baseline
        self.norepinephrine.level = self.norepinephrine.baseline
        self.pnl_history.clear()
        self.volatility_history.clear()
        self.shock_history.clear()
        self.current_limits = RiskLimits(
            max_position_size=0.1,
            max_drawdown=0.2,
            stop_loss=0.02,
            take_profit=0.04,
            max_leverage=2.0,
            allowed_strategies=["momentum", "mean_reversion", "breakout"],
        )
        logger.info("NeuromodulatedRiskManager reset")
