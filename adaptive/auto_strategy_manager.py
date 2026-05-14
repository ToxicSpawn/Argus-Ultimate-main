"""
adaptive/auto_strategy_manager.py --- Automatic Strategy Lifecycle Management.

Evaluates all active strategies against a rule-based scorecard and produces
concrete StrategyAction recommendations: keep, reduce, disable, or enable.
Can auto-apply those decisions to the strategy allocator.

Rules
-----
1. Sharpe < 0 for 14 days -> disable
2. Sharpe > 1.5 and rising -> increase weight
3. Regime mismatch (momentum in mean-revert) -> reduce weight 50%
4. No trades in 7 days -> flag for review
5. New strategy passed backtest -> enable at 10% weight
6. Strategy correlation > 0.8 with another -> disable the weaker one

Usage::

    mgr = AutoStrategyManager()
    actions = mgr.evaluate_all_strategies(strategy_metrics, regime="trending")
    mgr.auto_apply(actions, allocator=my_allocator)
    report = mgr.get_strategy_health_report(strategy_metrics)

Standalone --- no hard imports on the rest of the ARGUS tree at module load.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StrategyAction:
    """A lifecycle recommendation for a single strategy."""

    strategy_name: str
    action: str               # "keep" | "reduce" | "disable" | "enable"
    reason: str
    new_weight: float         # suggested allocation weight (0.0 -- 1.0)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Regime compatibility map
# ---------------------------------------------------------------------------

_REGIME_STRATEGY_COMPAT: Dict[str, frozenset] = {
    "trending":     frozenset({"momentum", "trend_following", "breakout", "futures_basis"}),
    "mean_revert":  frozenset({"mean_reversion", "pairs", "stat_arb", "grid", "kalman_pairs"}),
    "volatile":     frozenset({"volatility", "vol_arb", "options", "tail_hedge"}),
    "calm":         frozenset({"dca", "funding_harvest", "grid", "session_effect"}),
    "crisis":       frozenset({"tail_hedge", "hedging"}),
}


# ---------------------------------------------------------------------------
# AutoStrategyManager
# ---------------------------------------------------------------------------

class AutoStrategyManager:
    """Automatic strategy lifecycle management.

    Parameters
    ----------
    config : dict, optional
        ``auto_strategy_manager`` section from unified config.
    """

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._sharpe_disable_threshold: float = float(cfg.get("sharpe_disable_threshold", 0.0))
        self._sharpe_disable_days: int = int(cfg.get("sharpe_disable_days", 14))
        self._sharpe_promote_threshold: float = float(cfg.get("sharpe_promote_threshold", 1.5))
        self._idle_days: int = int(cfg.get("idle_days", 7))
        self._correlation_threshold: float = float(cfg.get("correlation_threshold", 0.8))
        self._new_strategy_weight: float = float(cfg.get("new_strategy_weight", 0.10))
        self._regime_reduce_factor: float = float(cfg.get("regime_reduce_factor", 0.5))
        self._last_evaluation_ts: float = 0.0
        self._action_history: List[StrategyAction] = []

        logger.info(
            "AutoStrategyManager initialised (sharpe_disable=%.2f, promote=%.2f, idle=%dd)",
            self._sharpe_disable_threshold, self._sharpe_promote_threshold, self._idle_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all_strategies(
        self,
        strategy_metrics: Dict[str, Dict[str, Any]],
        *,
        regime: str = "unknown",
        correlations: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> List[StrategyAction]:
        """Evaluate all strategies and return lifecycle actions.

        Parameters
        ----------
        strategy_metrics : dict
            strategy_name -> {
                sharpe, sharpe_14d, sharpe_trend, win_rate, current_weight,
                trades_7d, last_trade_ts, is_active, strategy_type,
                backtest_passed, ...
            }
        regime : str
            Current market regime.
        correlations : dict, optional
            (strategy_a, strategy_b) -> correlation_coefficient.
        """
        if not self._enabled:
            return []

        actions: List[StrategyAction] = []
        self._last_evaluation_ts = time.time()

        for name, m in strategy_metrics.items():
            action = self._evaluate_single(name, m, regime)
            if action:
                actions.append(action)

        # Correlation check: disable weaker of correlated pairs
        if correlations:
            corr_actions = self._check_correlations(strategy_metrics, correlations)
            actions.extend(corr_actions)

        # New strategy promotion
        for name, m in strategy_metrics.items():
            if not m.get("is_active", True) and m.get("backtest_passed", False):
                actions.append(StrategyAction(
                    strategy_name=name,
                    action="enable",
                    reason=(
                        f"Strategy '{name}' passed backtest validation. "
                        f"Enabling at {self._new_strategy_weight:.0%} weight."
                    ),
                    new_weight=self._new_strategy_weight,
                ))

        self._action_history.extend(actions)
        # Prune history
        if len(self._action_history) > 1000:
            self._action_history = self._action_history[-500:]

        if actions:
            logger.info(
                "AutoStrategyManager evaluated %d strategies, produced %d actions",
                len(strategy_metrics), len(actions),
            )
        return actions

    def auto_apply(
        self,
        actions: List[StrategyAction],
        *,
        allocator: Any = None,
    ) -> int:
        """Apply strategy actions to the allocator.

        Parameters
        ----------
        actions : list[StrategyAction]
            Actions to apply.
        allocator : object, optional
            An object with ``set_weight(strategy_name, weight)`` and
            ``set_enabled(strategy_name, enabled)`` methods.

        Returns
        -------
        int
            Number of actions successfully applied.
        """
        if allocator is None:
            logger.warning("auto_apply called with no allocator; actions will be logged only.")
            return 0

        applied = 0
        for a in actions:
            try:
                if a.action == "disable":
                    if hasattr(allocator, "set_enabled"):
                        allocator.set_enabled(a.strategy_name, False)
                    if hasattr(allocator, "set_weight"):
                        allocator.set_weight(a.strategy_name, 0.0)
                    applied += 1
                elif a.action == "enable":
                    if hasattr(allocator, "set_enabled"):
                        allocator.set_enabled(a.strategy_name, True)
                    if hasattr(allocator, "set_weight"):
                        allocator.set_weight(a.strategy_name, a.new_weight)
                    applied += 1
                elif a.action == "reduce":
                    if hasattr(allocator, "set_weight"):
                        allocator.set_weight(a.strategy_name, a.new_weight)
                    applied += 1
                elif a.action == "keep":
                    pass  # no action needed
                else:
                    logger.warning("Unknown strategy action '%s' for '%s'", a.action, a.strategy_name)

                logger.info(
                    "Applied strategy action: %s('%s') -> weight=%.3f | %s",
                    a.action, a.strategy_name, a.new_weight, a.reason,
                )
            except Exception:
                logger.exception("Failed to apply action %s for '%s'", a.action, a.strategy_name)

        return applied

    def get_strategy_health_report(
        self,
        strategy_metrics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Return a health summary per strategy.

        Returns
        -------
        dict
            strategy_name -> {
                health: "healthy" | "degrading" | "critical" | "idle",
                sharpe, win_rate, trades_7d, current_weight, last_action
            }
        """
        report: Dict[str, Dict[str, Any]] = {}
        # Index recent actions by strategy
        recent: Dict[str, StrategyAction] = {}
        for a in reversed(self._action_history):
            if a.strategy_name not in recent:
                recent[a.strategy_name] = a

        for name, m in strategy_metrics.items():
            sharpe = float(m.get("sharpe", 0.0))
            win_rate = float(m.get("win_rate", 0.0))
            trades_7d = int(m.get("trades_7d", 0))
            weight = float(m.get("current_weight", 0.0))

            if trades_7d == 0:
                health = "idle"
            elif sharpe < -0.5:
                health = "critical"
            elif sharpe < 0.5:
                health = "degrading"
            else:
                health = "healthy"

            last = recent.get(name)
            report[name] = {
                "health": health,
                "sharpe": round(sharpe, 3),
                "win_rate": round(win_rate, 3),
                "trades_7d": trades_7d,
                "current_weight": round(weight, 4),
                "last_action": last.to_dict() if last else None,
            }
        return report

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate_single(
        self,
        name: str,
        m: Dict[str, Any],
        regime: str,
    ) -> Optional[StrategyAction]:
        """Evaluate a single strategy against the rule set."""
        sharpe = float(m.get("sharpe", 0.0))
        sharpe_14d = float(m.get("sharpe_14d", sharpe))
        sharpe_trend = float(m.get("sharpe_trend", 0.0))
        current_weight = float(m.get("current_weight", 0.1))
        trades_7d = int(m.get("trades_7d", -1))
        is_active = bool(m.get("is_active", True))
        strategy_type = str(m.get("strategy_type", ""))

        if not is_active:
            return None

        # Rule 1: Sharpe < 0 for 14 days -> disable
        if sharpe_14d < self._sharpe_disable_threshold:
            return StrategyAction(
                strategy_name=name,
                action="disable",
                reason=(
                    f"Sharpe ratio {sharpe_14d:.2f} has been below "
                    f"{self._sharpe_disable_threshold:.2f} for {self._sharpe_disable_days}+ days."
                ),
                new_weight=0.0,
            )

        # Rule 2: Sharpe > 1.5 and rising -> increase weight
        if sharpe > self._sharpe_promote_threshold and sharpe_trend > 0:
            new_w = min(0.4, current_weight * 1.25)
            return StrategyAction(
                strategy_name=name,
                action="keep",  # keep but with increased weight
                reason=(
                    f"Sharpe {sharpe:.2f} above {self._sharpe_promote_threshold:.2f} "
                    f"and rising (trend={sharpe_trend:+.3f}). Increasing weight."
                ),
                new_weight=round(new_w, 4),
            )

        # Rule 3: Regime mismatch -> reduce 50%
        if regime in _REGIME_STRATEGY_COMPAT and strategy_type:
            compatible = _REGIME_STRATEGY_COMPAT.get(regime, frozenset())
            if strategy_type not in compatible and strategy_type != "":
                new_w = max(0.0, current_weight * self._regime_reduce_factor)
                return StrategyAction(
                    strategy_name=name,
                    action="reduce",
                    reason=(
                        f"Strategy type '{strategy_type}' is not compatible with "
                        f"regime '{regime}'. Reducing weight by {(1 - self._regime_reduce_factor):.0%}."
                    ),
                    new_weight=round(new_w, 4),
                )

        # Rule 4: No trades in 7 days -> flag
        if trades_7d == 0:
            return StrategyAction(
                strategy_name=name,
                action="reduce",
                reason=(
                    f"Strategy '{name}' has generated 0 trades in the last "
                    f"{self._idle_days} days. Reducing weight for review."
                ),
                new_weight=round(current_weight * 0.5, 4),
            )

        return None

    def _check_correlations(
        self,
        strategy_metrics: Dict[str, Dict[str, Any]],
        correlations: Dict[Tuple[str, str], float],
    ) -> List[StrategyAction]:
        """Disable the weaker of highly correlated strategy pairs."""
        actions: List[StrategyAction] = []
        disabled: set = set()

        for (a, b), corr in correlations.items():
            if abs(corr) < self._correlation_threshold:
                continue
            if a in disabled or b in disabled:
                continue

            sharpe_a = float((strategy_metrics.get(a) or {}).get("sharpe", 0.0))
            sharpe_b = float((strategy_metrics.get(b) or {}).get("sharpe", 0.0))

            weaker = b if sharpe_a >= sharpe_b else a
            stronger = a if weaker == b else b
            disabled.add(weaker)

            actions.append(StrategyAction(
                strategy_name=weaker,
                action="disable",
                reason=(
                    f"Strategy '{weaker}' has correlation {abs(corr):.2f} with '{stronger}' "
                    f"(>{self._correlation_threshold:.2f}). Disabling weaker strategy "
                    f"(Sharpe {min(sharpe_a, sharpe_b):.2f} vs {max(sharpe_a, sharpe_b):.2f})."
                ),
                new_weight=0.0,
            ))

        return actions
