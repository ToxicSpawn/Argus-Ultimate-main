"""
Small Capital Manager for Argus-Ultimate v5.0.0 (HFT-Pinnacle)
Fixes -2.3% ROI at $1k by adjusting fee-aware position sizing,
min trade thresholds, and bandit convergence scaling.
"""

import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class SmallCapitalManager:
    """
    Manages position sizing and risk parameters for small capital accounts (~$1k).
    Addresses fee drag and minimum trade threshold issues that cause negative ROI.
    """

    def __init__(self, config_path: str = "config/small_capital_config.yaml"):
        self.config = self._load_config(config_path)
        self.capital = self.config["capital"]["initial"]
        self.min_trade_usd = self.config["thresholds"]["min_trade_usd"]
        self.max_fee_pct = self.config["fees"]["max_fee_pct_of_trade"]
        self.bandit_alpha = self.config["bandit"]["alpha"]
        self.bandit_beta = self.config["bandit"]["beta"]
        self._trade_history: list = []
        self._bandit_counts: Dict[str, int] = {}
        self._bandit_rewards: Dict[str, float] = {}

    def _load_config(self, path: str) -> dict:
        config_file = Path(path)
        if not config_file.exists():
            logger.warning(f"Config not found at {path}, using defaults.")
            return self._default_config()
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict:
        return {
            "capital": {"initial": 1000.0, "min_reserve_pct": 0.05},
            "thresholds": {
                "min_trade_usd": 12.0,
                "min_profit_after_fees_usd": 0.10,
                "max_position_pct": 0.15,
            },
            "fees": {
                "maker_fee": 0.001,
                "taker_fee": 0.001,
                "max_fee_pct_of_trade": 0.003,
                "slippage_estimate": 0.0005,
            },
            "bandit": {
                "alpha": 1.5,
                "beta": 1.0,
                "min_samples": 10,
                "convergence_scale": 0.6,
            },
            "drawdown": {
                "max_daily_drawdown_pct": 0.02,
                "max_total_drawdown_pct": 0.05,
            },
        }

    def update_capital(self, current_capital: float) -> None:
        """Update current capital balance."""
        self.capital = current_capital
        logger.debug(f"Capital updated to ${current_capital:.2f}")

    def fee_aware_position_size(
        self,
        signal_strength: float,
        price: float,
        fee_rate: Optional[float] = None,
        use_taker: bool = True,
    ) -> Tuple[float, float]:
        """
        Calculate fee-aware position size in USD and units.

        Returns (position_usd, position_units)
        Returns (0.0, 0.0) if trade is not viable after fees.
        """
        cfg = self.config
        fee = fee_rate if fee_rate is not None else (
            cfg["fees"]["taker_fee"] if use_taker else cfg["fees"]["maker_fee"]
        )
        slippage = cfg["fees"]["slippage_estimate"]
        total_cost_rate = (fee * 2) + slippage  # round-trip cost

        max_pos_pct = cfg["thresholds"]["max_position_pct"]
        reserve = cfg["capital"]["min_reserve_pct"]
        available_capital = self.capital * (1.0 - reserve)

        # Scale position by signal strength (0..1)
        raw_position_usd = available_capital * max_pos_pct * np.clip(signal_strength, 0.0, 1.0)

        # Enforce minimum trade threshold
        if raw_position_usd < self.min_trade_usd:
            logger.debug(
                f"Position ${raw_position_usd:.2f} below min threshold ${self.min_trade_usd:.2f}, skipping."
            )
            return 0.0, 0.0

        # Fee viability check
        estimated_fee_usd = raw_position_usd * total_cost_rate
        min_profit = cfg["thresholds"]["min_profit_after_fees_usd"]
        expected_profit = raw_position_usd * signal_strength * 0.005  # conservative estimate

        if expected_profit <= estimated_fee_usd + min_profit:
            logger.debug(
                f"Trade not fee-viable: expected profit ${expected_profit:.4f} "
                f"<= fees ${estimated_fee_usd:.4f} + min ${min_profit:.2f}"
            )
            return 0.0, 0.0

        # Cap fee drag
        if total_cost_rate > self.max_fee_pct:
            scale = self.max_fee_pct / total_cost_rate
            raw_position_usd *= scale

        position_units = raw_position_usd / price if price > 0 else 0.0
        return raw_position_usd, position_units

    def bandit_scaled_position(
        self,
        strategy_id: str,
        base_position_usd: float,
    ) -> float:
        """
        Scale position size using bandit convergence for small capital.
        Strategies with insufficient samples are penalized to prevent overconfidence.
        """
        cfg = self.config["bandit"]
        counts = self._bandit_counts.get(strategy_id, 0)
        rewards = self._bandit_rewards.get(strategy_id, 0.0)

        min_samples = cfg["min_samples"]
        convergence_scale = cfg["convergence_scale"]

        if counts < min_samples:
            # Penalize under-explored strategies
            confidence = (counts / min_samples) * convergence_scale
        else:
            # UCB-style confidence
            avg_reward = rewards / counts if counts > 0 else 0.0
            ucb_bonus = np.sqrt((self.bandit_alpha * np.log(max(counts, 1))) / max(counts, 1))
            confidence = np.clip(avg_reward + ucb_bonus, 0.0, 1.0)

        scaled = base_position_usd * confidence
        logger.debug(
            f"Bandit scale for '{strategy_id}': counts={counts}, confidence={confidence:.4f}, "
            f"position=${scaled:.2f}"
        )
        return max(scaled, 0.0)

    def update_bandit(
        self, strategy_id: str, reward: float
    ) -> None:
        """Update bandit statistics after a trade."""
        self._bandit_counts[strategy_id] = self._bandit_counts.get(strategy_id, 0) + 1
        self._bandit_rewards[strategy_id] = self._bandit_rewards.get(strategy_id, 0.0) + reward

    def check_drawdown(
        self, peak_capital: float, daily_start_capital: float
    ) -> Dict[str, bool]:
        """
        Check if drawdown limits are breached.
        Returns dict with 'daily_breach' and 'total_breach' flags.
        """
        cfg = self.config["drawdown"]
        daily_dd = (daily_start_capital - self.capital) / daily_start_capital if daily_start_capital > 0 else 0.0
        total_dd = (peak_capital - self.capital) / peak_capital if peak_capital > 0 else 0.0

        daily_breach = daily_dd > cfg["max_daily_drawdown_pct"]
        total_breach = total_dd > cfg["max_total_drawdown_pct"]

        if daily_breach:
            logger.warning(f"Daily drawdown breach: {daily_dd*100:.2f}% > {cfg['max_daily_drawdown_pct']*100:.2f}%")
        if total_breach:
            logger.warning(f"Total drawdown breach: {total_dd*100:.2f}% > {cfg['max_total_drawdown_pct']*100:.2f}%")

        return {"daily_breach": daily_breach, "total_breach": total_breach}

    def get_stats(self) -> Dict:
        """Return current manager stats."""
        return {
            "capital": self.capital,
            "bandit_strategies": len(self._bandit_counts),
            "bandit_counts": dict(self._bandit_counts),
            "bandit_rewards": dict(self._bandit_rewards),
        }
