"""
Strategy Validation Gate — no strategy trades live without proving edge.

Implements the quant fund principle: every strategy must demonstrate
positive expectancy on historical data before being allowed to trade.

Validates via:
1. Walk-forward backtest (in-sample train → out-of-sample test)
2. Minimum Sharpe ratio requirement
3. Minimum number of trades
4. Maximum drawdown requirement
5. Out-of-sample performance must match in-sample within tolerance
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of strategy validation."""
    strategy: str
    passed: bool
    sharpe: float
    win_rate: float
    total_trades: int
    max_drawdown_pct: float
    profit_factor: float
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    oos_degradation_pct: float  # how much worse OOS is vs IS
    reason: str  # why it passed/failed
    validated_at: float = field(default_factory=time.time)


class StrategyValidator:
    """
    Gate that blocks strategies from live trading until validated.

    A strategy must pass ALL of these to trade:
    - min_sharpe: Sharpe ratio >= threshold (default 0.5)
    - min_trades: At least N trades in backtest (default 30)
    - max_drawdown: Max drawdown <= threshold (default 20%)
    - min_win_rate: Win rate >= threshold (default 35%)
    - min_profit_factor: Gross profit / gross loss >= threshold (default 1.2)
    - oos_max_degradation: Out-of-sample Sharpe within X% of in-sample (default 50%)
    """

    def __init__(
        self,
        min_sharpe: float = 0.5,
        min_trades: int = 30,
        max_drawdown_pct: float = 0.20,
        min_win_rate: float = 0.35,
        min_profit_factor: float = 1.2,
        oos_max_degradation_pct: float = 0.50,
        results_path: str = "data/strategy_validations.json",
    ):
        self._min_sharpe = min_sharpe
        self._min_trades = min_trades
        self._max_drawdown = max_drawdown_pct
        self._min_win_rate = min_win_rate
        self._min_profit_factor = min_profit_factor
        self._oos_max_degradation = oos_max_degradation_pct
        self._results_path = results_path
        self._validated: Dict[str, ValidationResult] = {}
        self._blocked: Set[str] = set()
        self._load_results()

    def _load_results(self) -> None:
        """Load previous validation results from disk."""
        try:
            p = Path(self._results_path)
            if p.exists():
                data = json.loads(p.read_text())
                for name, r in data.items():
                    self._validated[name] = ValidationResult(**r)
                    if not r.get("passed", False):
                        self._blocked.add(name)
                logger.info("StrategyValidator: loaded %d validations from %s", len(self._validated), self._results_path)
        except Exception as e:
            logger.debug("StrategyValidator: load failed: %s", e)

    def _save_results(self) -> None:
        """Persist validation results to disk."""
        try:
            p = Path(self._results_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for name, r in self._validated.items():
                data[name] = {
                    "strategy": r.strategy,
                    "passed": r.passed,
                    "sharpe": r.sharpe,
                    "win_rate": r.win_rate,
                    "total_trades": r.total_trades,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "profit_factor": r.profit_factor,
                    "in_sample_sharpe": r.in_sample_sharpe,
                    "out_of_sample_sharpe": r.out_of_sample_sharpe,
                    "oos_degradation_pct": r.oos_degradation_pct,
                    "reason": r.reason,
                    "validated_at": r.validated_at,
                }
            p.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug("StrategyValidator: save failed: %s", e)

    def validate(
        self,
        strategy: str,
        trades: List[Dict[str, Any]],
        in_sample_trades: Optional[List[Dict[str, Any]]] = None,
        out_of_sample_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> ValidationResult:
        """
        Validate a strategy based on backtest results.

        Args:
            strategy: Strategy name
            trades: Full list of backtest trade results
            in_sample_trades: Trades from training period (optional)
            out_of_sample_trades: Trades from test period (optional)

        Returns ValidationResult with passed=True/False and reason.
        """
        if not trades:
            result = ValidationResult(
                strategy=strategy, passed=False, sharpe=0.0, win_rate=0.0,
                total_trades=0, max_drawdown_pct=0.0, profit_factor=0.0,
                in_sample_sharpe=0.0, out_of_sample_sharpe=0.0,
                oos_degradation_pct=1.0, reason="no_trades",
            )
            self._validated[strategy] = result
            self._blocked.add(strategy)
            self._save_results()
            return result

        # Compute metrics
        pnls = [float(t.get("pnl", 0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total = len(pnls)
        win_rate = len(wins) / max(total, 1)
        avg_pnl = sum(pnls) / max(total, 1)

        # Sharpe-like ratio
        if total >= 2:
            mean = sum(pnls) / total
            variance = sum((p - mean) ** 2 for p in pnls) / total
            std = max(variance ** 0.5, 1e-9)
            sharpe = mean / std
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-9
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        # Max drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            dd = (peak - equity) / max(peak, 1e-9) if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # In-sample vs out-of-sample comparison
        is_sharpe = 0.0
        oos_sharpe = 0.0
        oos_degradation = 0.0

        if in_sample_trades and out_of_sample_trades:
            is_pnls = [float(t.get("pnl", 0)) for t in in_sample_trades]
            oos_pnls = [float(t.get("pnl", 0)) for t in out_of_sample_trades]

            if len(is_pnls) >= 2:
                is_mean = sum(is_pnls) / len(is_pnls)
                is_std = max((sum((p - is_mean) ** 2 for p in is_pnls) / len(is_pnls)) ** 0.5, 1e-9)
                is_sharpe = is_mean / is_std

            if len(oos_pnls) >= 2:
                oos_mean = sum(oos_pnls) / len(oos_pnls)
                oos_std = max((sum((p - oos_mean) ** 2 for p in oos_pnls) / len(oos_pnls)) ** 0.5, 1e-9)
                oos_sharpe = oos_mean / oos_std

            if is_sharpe > 0:
                oos_degradation = 1.0 - (oos_sharpe / is_sharpe)
            else:
                oos_degradation = 1.0
        else:
            # No IS/OOS split — use full dataset
            is_sharpe = sharpe
            oos_sharpe = sharpe
            oos_degradation = 0.0

        # Check all gates
        failures = []
        if total < self._min_trades:
            failures.append(f"trades={total}<{self._min_trades}")
        if sharpe < self._min_sharpe:
            failures.append(f"sharpe={sharpe:.3f}<{self._min_sharpe}")
        if win_rate < self._min_win_rate:
            failures.append(f"win_rate={win_rate:.1%}<{self._min_win_rate:.0%}")
        if max_dd > self._max_drawdown:
            failures.append(f"drawdown={max_dd:.1%}>{self._max_drawdown:.0%}")
        if profit_factor < self._min_profit_factor:
            failures.append(f"profit_factor={profit_factor:.2f}<{self._min_profit_factor}")
        if oos_degradation > self._oos_max_degradation and in_sample_trades:
            failures.append(f"oos_degradation={oos_degradation:.1%}>{self._oos_max_degradation:.0%}")

        passed = len(failures) == 0
        reason = "all_gates_passed" if passed else "; ".join(failures)

        result = ValidationResult(
            strategy=strategy, passed=passed, sharpe=sharpe, win_rate=win_rate,
            total_trades=total, max_drawdown_pct=max_dd, profit_factor=profit_factor,
            in_sample_sharpe=is_sharpe, out_of_sample_sharpe=oos_sharpe,
            oos_degradation_pct=oos_degradation, reason=reason,
        )

        self._validated[strategy] = result
        if passed:
            self._blocked.discard(strategy)
            logger.info("StrategyValidator: %s PASSED (sharpe=%.3f, wr=%.1%%, pf=%.2f, dd=%.1%%)",
                        strategy, sharpe, win_rate * 100, profit_factor, max_dd * 100)
        else:
            self._blocked.add(strategy)
            logger.warning("StrategyValidator: %s FAILED — %s", strategy, reason)

        self._save_results()
        return result

    def is_approved(self, strategy: str) -> bool:
        """Check if strategy is approved for live trading."""
        return strategy not in self._blocked and strategy in self._validated

    def is_blocked(self, strategy: str) -> bool:
        """Check if strategy is explicitly blocked."""
        return strategy in self._blocked

    def get_validation(self, strategy: str) -> Optional[ValidationResult]:
        """Get the latest validation result for a strategy."""
        return self._validated.get(strategy)

    def get_all_validations(self) -> Dict[str, ValidationResult]:
        """Get all validation results."""
        return dict(self._validated)

    def approve_without_backtest(self, strategy: str, reason: str = "manual_override") -> None:
        """Manual override — approve a strategy without backtest (for paper trading)."""
        result = ValidationResult(
            strategy=strategy, passed=True, sharpe=0.0, win_rate=0.0,
            total_trades=0, max_drawdown_pct=0.0, profit_factor=0.0,
            in_sample_sharpe=0.0, out_of_sample_sharpe=0.0,
            oos_degradation_pct=0.0, reason=f"manual: {reason}",
        )
        self._validated[strategy] = result
        self._blocked.discard(strategy)
        self._save_results()
        logger.info("StrategyValidator: %s manually approved — %s", strategy, reason)
