"""
Rolling performance scorecard — tracks strategy performance over rolling windows
and auto-disables strategies that consistently underperform.

Integrates with StrategyRouter to disable/enable strategies based on metrics.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Rolling metrics for a single strategy."""
    trades: deque = field(default_factory=lambda: deque(maxlen=100))
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    last_trade_time: float = 0.0
    disabled_at: Optional[float] = None
    disable_reason: str = ""

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.5
        wins = sum(1 for t in self.trades if t > 0)
        return wins / len(self.trades)

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(self.trades) / len(self.trades)

    @property
    def sharpe_like(self) -> float:
        """Simple Sharpe-like ratio: mean / std of trade P&L."""
        if len(self.trades) < 5:
            return 0.0
        trades = list(self.trades)
        mean = sum(trades) / len(trades)
        variance = sum((t - mean) ** 2 for t in trades) / len(trades)
        std = max(variance ** 0.5, 1e-9)
        return mean / std


class PerformanceScorecard:
    """
    Tracks rolling strategy performance and auto-disables underperformers.

    Thresholds:
    - Sharpe < -0.5 over 20+ trades → disable
    - 7 consecutive losses → disable (temporary, 200 cycles)
    - Win rate < 0.25 over 30+ trades → disable
    - Recovery: re-enable after cooldown if Sharpe > 0.1
    """

    def __init__(
        self,
        disable_sharpe: float = -0.5,
        disable_win_rate: float = 0.25,
        disable_consec_losses: int = 7,
        cooldown_seconds: float = 7200.0,  # 2 hours
        re_enable_sharpe: float = 0.1,
        min_trades_for_eval: int = 20,
    ):
        self._strategies: Dict[str, StrategyMetrics] = defaultdict(StrategyMetrics)
        self._disabled: Set[str] = set()
        self._disable_sharpe = disable_sharpe
        self._disable_win_rate = disable_win_rate
        self._disable_consec_losses = disable_consec_losses
        self._cooldown_seconds = cooldown_seconds
        self._re_enable_sharpe = re_enable_sharpe
        self._min_trades = min_trades_for_eval
        self._cycle_count = 0

    def record_trade(self, strategy: str, pnl_aud: float) -> None:
        """Record a trade outcome for a strategy."""
        m = self._strategies[strategy]
        m.trades.append(pnl_aud)
        m.total_pnl += pnl_aud
        m.last_trade_time = time.time()

        if pnl_aud < 0:
            m.consecutive_losses += 1
        else:
            m.consecutive_losses = 0

    def evaluate(self, strategy_router: Any = None) -> Dict[str, Any]:
        """
        Evaluate all strategies. Returns advisory dict.
        If strategy_router is provided, actually disables/enables strategies.
        """
        self._cycle_count += 1
        actions: List[Dict[str, str]] = []

        for name, m in self._strategies.items():
            # Check for disable conditions
            if name not in self._disabled:
                reason = self._should_disable(name, m)
                if reason:
                    self._disabled.add(name)
                    m.disabled_at = time.time()
                    m.disable_reason = reason
                    if strategy_router and hasattr(strategy_router, "disable"):
                        try:
                            strategy_router.disable(name)
                        except Exception:
                            pass
                    actions.append({"strategy": name, "action": "disabled", "reason": reason})
                    logger.warning("PerformanceScorecard: DISABLED %s — %s", name, reason)

            # Check for re-enable conditions
            elif name in self._disabled:
                if self._should_reenable(name, m):
                    self._disabled.discard(name)
                    m.disabled_at = None
                    m.disable_reason = ""
                    if strategy_router and hasattr(strategy_router, "enable"):
                        try:
                            strategy_router.enable(name)
                        except Exception:
                            pass
                    actions.append({"strategy": name, "action": "re-enabled"})
                    logger.info("PerformanceScorecard: RE-ENABLED %s (recovered)", name)

        return {
            "cycle": self._cycle_count,
            "disabled_strategies": list(self._disabled),
            "actions": actions,
            "strategy_metrics": {
                name: {
                    "trades": m.trade_count,
                    "win_rate": round(m.win_rate, 3),
                    "sharpe": round(m.sharpe_like, 3),
                    "consec_losses": m.consecutive_losses,
                    "total_pnl": round(m.total_pnl, 2),
                    "disabled": name in self._disabled,
                }
                for name, m in self._strategies.items()
            },
        }

    def _should_disable(self, name: str, m: StrategyMetrics) -> str:
        """Check if strategy should be disabled. Returns reason or empty string."""
        if m.trade_count >= self._min_trades:
            if m.sharpe_like < self._disable_sharpe:
                return f"sharpe={m.sharpe_like:.3f} < {self._disable_sharpe}"
            if m.win_rate < self._disable_win_rate and m.trade_count >= 30:
                return f"win_rate={m.win_rate:.1%} < {self._disable_win_rate:.0%}"
        if m.consecutive_losses >= self._disable_consec_losses:
            return f"consecutive_losses={m.consecutive_losses} >= {self._disable_consec_losses}"
        return ""

    def _should_reenable(self, name: str, m: StrategyMetrics) -> bool:
        """Check if disabled strategy should be re-enabled."""
        if m.disabled_at is None:
            return True
        elapsed = time.time() - m.disabled_at
        if elapsed < self._cooldown_seconds:
            return False
        # After cooldown, check if recent performance improved
        if m.trade_count >= 10 and m.sharpe_like > self._re_enable_sharpe:
            return True
        # Also re-enable if consecutive losses recovered
        if m.consecutive_losses == 0 and elapsed > self._cooldown_seconds * 2:
            return True
        return False

    def is_disabled(self, strategy: str) -> bool:
        """Check if a strategy is currently disabled."""
        return strategy in self._disabled

    def get_ranking(self) -> List[Dict[str, Any]]:
        """Get strategies ranked by Sharpe-like metric."""
        ranked = []
        for name, m in self._strategies.items():
            if m.trade_count >= 5:
                ranked.append({
                    "strategy": name,
                    "sharpe": m.sharpe_like,
                    "win_rate": m.win_rate,
                    "trades": m.trade_count,
                    "disabled": name in self._disabled,
                })
        ranked.sort(key=lambda x: x["sharpe"], reverse=True)
        return ranked
