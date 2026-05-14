"""Paper trading validator: gate strategies before live deployment.

Validates that a strategy has been paper-traded successfully before
allowing it to go live. Checks:
  - Minimum paper trading duration
  - Minimum number of trades
  - Positive P&L over paper period
  - Maximum drawdown within limits
  - Sharpe ratio above threshold
  - No consecutive losing days beyond limit

Usage:
    validator = PaperTradingValidator(
        min_days=7,
        min_trades=20,
        min_sharpe=0.5,
        max_drawdown_pct=15.0,
        max_consecutive_losses=5,
    )
    result = validator.validate(paper_results)
    if result.passed:
        print("Strategy approved for live trading")
    else:
        print(f"Failed: {result.reasons}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.backtest.metrics import BacktestMetrics, compute_metrics


class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"  # Not enough data yet


@dataclass
class ValidationCheck:
    """Single validation check result."""
    name: str
    passed: bool
    required: float
    actual: float
    message: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "required": self.required,
            "actual": round(self.actual, 4),
            "message": self.message,
        }


@dataclass
class ValidationResult:
    """Full validation result."""
    status: ValidationStatus
    strategy_name: str
    checks: List[ValidationCheck]
    passed_count: int
    failed_count: int
    total_count: int
    paper_duration_days: int
    total_trades: int
    paper_pnl_pct: float
    reasons: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "strategy_name": self.strategy_name,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "paper_duration_days": self.paper_duration_days,
            "total_trades": self.total_trades,
            "paper_pnl_pct": round(self.paper_pnl_pct, 4),
            "checks": [c.to_dict() for c in self.checks],
            "reasons": self.reasons,
        }


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading validation."""
    min_days: int = 7                # Minimum paper trading duration
    min_trades: int = 20             # Minimum number of trades
    min_sharpe: float = 0.5          # Minimum Sharpe ratio
    min_sortino: float = 0.3         # Minimum Sortino ratio
    max_drawdown_pct: float = 15.0   # Maximum allowed drawdown
    min_win_rate: float = 0.4        # Minimum win rate
    max_consecutive_losses: int = 5  # Maximum consecutive losing trades
    min_profit_factor: float = 1.0   # Minimum profit factor
    require_positive_pnl: bool = True  # Require positive P&L

    def to_dict(self) -> dict:
        return {
            "min_days": self.min_days,
            "min_trades": self.min_trades,
            "min_sharpe": self.min_sharpe,
            "min_sortino": self.min_sortino,
            "max_drawdown_pct": self.max_drawdown_pct,
            "min_win_rate": self.min_win_rate,
            "max_consecutive_losses": self.max_consecutive_losses,
            "min_profit_factor": self.min_profit_factor,
            "require_positive_pnl": self.require_positive_pnl,
        }


class PaperTradingValidator:
    """Validates strategy performance on paper trading data.

    Args:
        min_days:          Minimum paper trading duration
        min_trades:        Minimum number of trades
        min_sharpe:        Minimum Sharpe ratio
        min_sortino:       Minimum Sortino ratio
        max_drawdown_pct:  Maximum allowed drawdown %
        min_win_rate:      Minimum win rate
        max_consecutive_losses: Maximum consecutive losing trades
        min_profit_factor: Minimum profit factor
        require_positive_pnl: Require positive P&L
    """

    def __init__(
        self,
        min_days: int = 7,
        min_trades: int = 20,
        min_sharpe: float = 0.5,
        min_sortino: float = 0.3,
        max_drawdown_pct: float = 15.0,
        min_win_rate: float = 0.4,
        max_consecutive_losses: int = 5,
        min_profit_factor: float = 1.0,
        require_positive_pnl: bool = True,
    ):
        self.config = PaperTradingConfig(
            min_days=min_days,
            min_trades=min_trades,
            min_sharpe=min_sharpe,
            min_sortino=min_sortino,
            max_drawdown_pct=max_drawdown_pct,
            min_win_rate=min_win_rate,
            max_consecutive_losses=max_consecutive_losses,
            min_profit_factor=min_profit_factor,
            require_positive_pnl=require_positive_pnl,
        )

    def validate(
        self,
        strategy_name: str,
        equity_curve: Sequence[float],
        trade_pnls: Optional[Sequence[float]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        periods_per_year: int = 252,
    ) -> ValidationResult:
        """Validate paper trading results.

        Args:
            strategy_name:    Name of the strategy
            equity_curve:     Paper trading equity curve
            trade_pnls:       List of per-trade P&L values
            start_time:       Paper trading start time
            end_time:         Paper trading end time
            periods_per_year: For annualization

        Returns:
            ValidationResult with pass/fail status and details
        """
        checks: List[ValidationCheck] = []
        reasons: List[str] = []

        equity = list(equity_curve)
        pnls = list(trade_pnls) if trade_pnls else []

        # Compute duration
        if start_time and end_time:
            duration = (end_time - start_time).days
        else:
            # Estimate from bars
            duration = len(equity)  # Assume 1 bar = 1 day for simplicity

        # Compute metrics
        metrics = compute_metrics(
            equity, pnls if pnls else None,
            periods_per_year=periods_per_year,
        )

        # Check 1: Minimum duration
        check = ValidationCheck(
            name="min_duration",
            passed=duration >= self.config.min_days,
            required=float(self.config.min_days),
            actual=float(duration),
            message=f"Paper trading duration: {duration} days (required: {self.config.min_days})",
        )
        checks.append(check)
        if not check.passed:
            reasons.append(f"Insufficient paper trading duration: {duration} < {self.config.min_days} days")

        # Check 2: Minimum trades
        n_trades = len(pnls) if pnls else metrics.n_trades
        check = ValidationCheck(
            name="min_trades",
            passed=n_trades >= self.config.min_trades,
            required=float(self.config.min_trades),
            actual=float(n_trades),
            message=f"Total trades: {n_trades} (required: {self.config.min_trades})",
        )
        checks.append(check)
        if not check.passed:
            reasons.append(f"Insufficient trades: {n_trades} < {self.config.min_trades}")

        # Check 3: Minimum Sharpe
        check = ValidationCheck(
            name="min_sharpe",
            passed=metrics.sharpe >= self.config.min_sharpe,
            required=self.config.min_sharpe,
            actual=metrics.sharpe,
            message=f"Sharpe ratio: {metrics.sharpe:.4f} (required: {self.config.min_sharpe})",
        )
        checks.append(check)
        if not check.passed:
            reasons.append(f"Sharpe ratio too low: {metrics.sharpe:.4f} < {self.config.min_sharpe}")

        # Check 4: Minimum Sortino
        check = ValidationCheck(
            name="min_sortino",
            passed=metrics.sortino >= self.config.min_sortino,
            required=self.config.min_sortino,
            actual=metrics.sortino,
            message=f"Sortino ratio: {metrics.sortino:.4f} (required: {self.config.min_sortino})",
        )
        checks.append(check)
        if not check.passed:
            reasons.append(f"Sortino ratio too low: {metrics.sortino:.4f} < {self.config.min_sortino}")

        # Check 5: Maximum drawdown
        check = ValidationCheck(
            name="max_drawdown",
            passed=metrics.max_drawdown_pct <= self.config.max_drawdown_pct,
            required=self.config.max_drawdown_pct,
            actual=metrics.max_drawdown_pct,
            message=f"Max drawdown: {metrics.max_drawdown_pct:.2f}% (max allowed: {self.config.max_drawdown_pct}%)",
        )
        checks.append(check)
        if not check.passed:
            reasons.append(f"Drawdown too high: {metrics.max_drawdown_pct:.2f}% > {self.config.max_drawdown_pct}%")

        # Check 6: Minimum win rate
        check = ValidationCheck(
            name="min_win_rate",
            passed=metrics.win_rate >= self.config.min_win_rate,
            required=self.config.min_win_rate,
            actual=metrics.win_rate,
            message=f"Win rate: {metrics.win_rate:.2%} (required: {self.config.min_win_rate:.0%})",
        )
        checks.append(check)
        if not check.passed:
            reasons.append(f"Win rate too low: {metrics.win_rate:.2%} < {self.config.min_win_rate:.0%}")

        # Check 7: Consecutive losses
        if pnls:
            max_consecutive = self._max_consecutive_losses(pnls)
            check = ValidationCheck(
                name="max_consecutive_losses",
                passed=max_consecutive <= self.config.max_consecutive_losses,
                required=float(self.config.max_consecutive_losses),
                actual=float(max_consecutive),
                message=f"Max consecutive losses: {max_consecutive} (max allowed: {self.config.max_consecutive_losses})",
            )
            checks.append(check)
            if not check.passed:
                reasons.append(f"Too many consecutive losses: {max_consecutive} > {self.config.max_consecutive_losses}")
        else:
            # Skip if no trade P&L data
            checks.append(ValidationCheck(
                name="max_consecutive_losses",
                passed=True,
                required=0,
                actual=0,
                message="Skipped (no trade P&L data)",
            ))

        # Check 8: Profit factor
        check = ValidationCheck(
            name="min_profit_factor",
            passed=metrics.profit_factor >= self.config.min_profit_factor,
            required=self.config.min_profit_factor,
            actual=metrics.profit_factor,
            message=f"Profit factor: {metrics.profit_factor:.4f} (required: {self.config.min_profit_factor})",
        )
        checks.append(check)
        if not check.passed:
            reasons.append(f"Profit factor too low: {metrics.profit_factor:.4f} < {self.config.min_profit_factor}")

        # Check 9: Positive P&L
        if self.config.require_positive_pnl:
            check = ValidationCheck(
                name="positive_pnl",
                passed=metrics.total_return_pct > 0,
                required=0.0,
                actual=metrics.total_return_pct,
                message=f"Total return: {metrics.total_return_pct:.2f}% (must be positive)",
            )
            checks.append(check)
            if not check.passed:
                reasons.append(f"Negative P&L: {metrics.total_return_pct:.2f}%")

        # Determine overall status
        passed_count = sum(1 for c in checks if c.passed)
        failed_count = len(checks) - passed_count

        if failed_count == 0:
            status = ValidationStatus.PASSED
        elif duration < self.config.min_days:
            status = ValidationStatus.PENDING  # Not enough data yet
        else:
            status = ValidationStatus.FAILED

        return ValidationResult(
            status=status,
            strategy_name=strategy_name,
            checks=checks,
            passed_count=passed_count,
            failed_count=failed_count,
            total_count=len(checks),
            paper_duration_days=duration,
            total_trades=n_trades,
            paper_pnl_pct=metrics.total_return_pct,
            reasons=reasons,
        )

    def _max_consecutive_losses(self, pnls: Sequence[float]) -> int:
        """Compute maximum consecutive losing trades."""
        max_streak = 0
        current_streak = 0

        for pnl in pnls:
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak


class LiveGate:
    """Gate that prevents strategies from going live without paper validation.

    Usage:
        gate = LiveGate()
        gate.register_strategy("momentum", validator, paper_results)
        if gate.can_go_live("momentum"):
            # Deploy to live
            pass
    """

    def __init__(self):
        self._validations: Dict[str, ValidationResult] = {}
        self._paper_start: Dict[str, datetime] = {}

    def register_strategy(
        self,
        strategy_name: str,
        validator: PaperTradingValidator,
        equity_curve: Sequence[float],
        trade_pnls: Optional[Sequence[float]] = None,
        paper_start: Optional[datetime] = None,
    ) -> ValidationResult:
        """Register and validate a paper-traded strategy."""
        end_time = datetime.now(timezone.utc)
        result = validator.validate(
            strategy_name=strategy_name,
            equity_curve=equity_curve,
            trade_pnls=trade_pnls,
            start_time=paper_start,
            end_time=end_time,
        )
        self._validations[strategy_name] = result
        if paper_start:
            self._paper_start[strategy_name] = paper_start
        return result

    def can_go_live(self, strategy_name: str) -> bool:
        """Check if a strategy is approved for live trading."""
        if strategy_name not in self._validations:
            return False
        return self._validations[strategy_name].passed

    def get_status(self, strategy_name: str) -> Optional[ValidationResult]:
        """Get validation status for a strategy."""
        return self._validations.get(strategy_name)

    def get_all_statuses(self) -> Dict[str, ValidationResult]:
        """Get all strategy validation statuses."""
        return dict(self._validations)

    def get_pending(self) -> List[str]:
        """Get list of strategies pending validation."""
        return [
            name for name, result in self._validations.items()
            if result.status == ValidationStatus.PENDING
        ]

    def get_approved(self) -> List[str]:
        """Get list of strategies approved for live."""
        return [
            name for name, result in self._validations.items()
            if result.passed
        ]

    def get_rejected(self) -> List[str]:
        """Get list of strategies rejected from live."""
        return [
            name for name, result in self._validations.items()
            if result.status == ValidationStatus.FAILED
        ]
