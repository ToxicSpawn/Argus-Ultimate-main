"""
Portfolio-Level Risk Optimizer — Hierarchical Risk Parity + Dynamic Allocation.

Current ARGUS risk management is per-trade. This module optimizes risk
across the ENTIRE portfolio:

1. Hierarchical Risk Parity (HRP): allocate capital based on asset clustering
   and inverse variance, NOT equal-weight or Markowitz (which is unstable)
2. Dynamic Capital Allocation: shift capital toward strategies with positive
   realized Sharpe, away from strategies in drawdown
3. Cross-Strategy Correlation Monitor: reduce exposure when strategies become
   correlated (same bets = concentrated risk)
4. Regime-Conditional Risk Budgets: tighter limits in volatile/crisis regimes
5. Expected Shortfall (CVaR): tail risk beyond VaR

This is what separates institutional risk from retail risk.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioRiskSnapshot:
    """Complete portfolio risk assessment."""
    total_exposure_pct: float           # sum of all position sizes / capital
    var_95: float                       # 95% Value at Risk (% of capital)
    cvar_95: float                      # Expected Shortfall beyond VaR
    max_correlation: float              # highest pairwise correlation
    concentration_hhi: float            # Herfindahl index (0=diversified, 1=concentrated)
    strategy_allocations: Dict[str, float]  # optimal allocation per strategy
    risk_budget_remaining: float        # how much risk budget is unused
    regime_risk_multiplier: float       # 1.0 normal, 0.5 crisis, 1.2 trending
    warnings: List[str]


@dataclass
class StrategyPerformance:
    """Rolling performance tracker for one strategy."""
    name: str
    returns: List[float] = field(default_factory=list)
    sharpe: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    cumulative_pnl: float = 0.0

    def update(self, pnl: float) -> None:
        self.returns.append(pnl)
        if len(self.returns) > 500:
            self.returns = self.returns[-500:]
        self.cumulative_pnl += pnl
        self.peak_equity = max(self.peak_equity, self.cumulative_pnl)
        self.current_drawdown = self.peak_equity - self.cumulative_pnl
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        if len(self.returns) >= 5:
            mean_r = sum(self.returns) / len(self.returns)
            var_r = sum((r - mean_r) ** 2 for r in self.returns) / max(len(self.returns) - 1, 1)
            self.volatility = var_r ** 0.5
            self.sharpe = mean_r / max(self.volatility, 1e-9)


class PortfolioRiskOptimizer:
    """
    Portfolio-level risk optimization.

    Computes optimal allocation across strategies using inverse-volatility
    weighting with correlation adjustment (simplified HRP).
    """

    def __init__(
        self,
        max_total_exposure: float = 1.0,    # max 100% of capital deployed
        max_single_strategy: float = 0.40,  # max 40% in one strategy
        max_correlation: float = 0.80,       # reduce if correlation > 0.80
        var_limit_pct: float = 0.05,        # max 5% VaR
        cvar_multiplier: float = 1.5,       # CVaR ~ 1.5x VaR for normal dist
        rebalance_threshold: float = 0.10,  # rebalance when drift > 10%
    ):
        self._max_exposure = max_total_exposure
        self._max_single = max_single_strategy
        self._max_corr = max_correlation
        self._var_limit = var_limit_pct
        self._cvar_mult = cvar_multiplier
        self._rebal_thresh = rebalance_threshold

        self._strategies: Dict[str, StrategyPerformance] = {}
        self._return_matrix: Dict[str, List[float]] = defaultdict(list)
        self._current_allocations: Dict[str, float] = {}
        self._last_rebalance = 0

    def record_return(self, strategy: str, pnl: float) -> None:
        """Record a strategy return."""
        if strategy not in self._strategies:
            self._strategies[strategy] = StrategyPerformance(name=strategy)
        self._strategies[strategy].update(pnl)
        self._return_matrix[strategy].append(pnl)
        if len(self._return_matrix[strategy]) > 500:
            self._return_matrix[strategy] = self._return_matrix[strategy][-500:]

    def optimize(self, regime: str = "normal") -> PortfolioRiskSnapshot:
        """
        Compute optimal portfolio allocation and risk metrics.

        Returns PortfolioRiskSnapshot with allocations per strategy.
        """
        strategies = {k: v for k, v in self._strategies.items() if len(v.returns) >= 5}
        if not strategies:
            return PortfolioRiskSnapshot(
                total_exposure_pct=0, var_95=0, cvar_95=0, max_correlation=0,
                concentration_hhi=0, strategy_allocations={},
                risk_budget_remaining=self._var_limit,
                regime_risk_multiplier=1.0, warnings=[],
            )

        warnings = []

        # ── 1. Inverse-volatility weighting (simplified HRP) ──
        vol_weights = {}
        for name, perf in strategies.items():
            inv_vol = 1.0 / max(perf.volatility, 1e-6)
            # Sharpe bonus: strategies with positive Sharpe get more weight
            sharpe_bonus = max(0, perf.sharpe) * 0.5
            # Drawdown penalty: strategies in drawdown get less
            dd_penalty = min(1.0, perf.current_drawdown * 2)
            vol_weights[name] = inv_vol * (1 + sharpe_bonus) * (1 - dd_penalty * 0.5)

        total_w = sum(vol_weights.values())
        if total_w > 0:
            allocations = {k: v / total_w for k, v in vol_weights.items()}
        else:
            n = len(strategies)
            allocations = {k: 1.0 / n for k in strategies}

        # ── 2. Cap single-strategy concentration ──
        for name in allocations:
            if allocations[name] > self._max_single:
                allocations[name] = self._max_single

        # Renormalize
        total_alloc = sum(allocations.values())
        if total_alloc > 0:
            allocations = {k: v / total_alloc for k, v in allocations.items()}

        # ── 3. Regime-conditional risk budget ──
        regime_mult = 1.0
        regime_lower = regime.lower()
        if "crisis" in regime_lower or "high_vol" in regime_lower:
            regime_mult = 0.5
            warnings.append("CRISIS regime: risk budget halved")
        elif "volatile" in regime_lower:
            regime_mult = 0.7
        elif "trending" in regime_lower:
            regime_mult = 1.1

        # Scale allocations by regime
        allocations = {k: v * regime_mult for k, v in allocations.items()}

        # ── 4. Correlation check ──
        max_corr = self._compute_max_correlation()
        if max_corr > self._max_corr:
            # Reduce all allocations proportionally
            corr_penalty = 1.0 - (max_corr - self._max_corr) * 2
            corr_penalty = max(0.3, corr_penalty)
            allocations = {k: v * corr_penalty for k, v in allocations.items()}
            warnings.append(f"High correlation ({max_corr:.2f}): exposure reduced")

        # ── 5. VaR and CVaR ──
        var_95 = self._compute_portfolio_var(allocations)
        cvar_95 = var_95 * self._cvar_mult

        if var_95 > self._var_limit:
            scale = self._var_limit / var_95
            allocations = {k: v * scale for k, v in allocations.items()}
            warnings.append(f"VaR breach ({var_95:.3f}): positions scaled to {scale:.0%}")
            var_95 = self._var_limit

        # ── 6. Concentration (HHI) ──
        hhi = sum(v ** 2 for v in allocations.values())

        total_exposure = sum(allocations.values())
        risk_remaining = max(0, self._var_limit - var_95)

        self._current_allocations = allocations

        if allocations:
            logger.debug(
                "PortfolioRisk: %d strategies, exposure=%.1f%%, VaR=%.3f, CVaR=%.3f, HHI=%.3f, corr=%.2f",
                len(allocations), total_exposure * 100, var_95, cvar_95, hhi, max_corr,
            )

        return PortfolioRiskSnapshot(
            total_exposure_pct=total_exposure,
            var_95=var_95,
            cvar_95=cvar_95,
            max_correlation=max_corr,
            concentration_hhi=hhi,
            strategy_allocations=allocations,
            risk_budget_remaining=risk_remaining,
            regime_risk_multiplier=regime_mult,
            warnings=warnings,
        )

    def get_allocation(self, strategy: str) -> float:
        """Get current allocation for a strategy (0.0 to 1.0)."""
        return self._current_allocations.get(strategy, 0.0)

    def _compute_max_correlation(self) -> float:
        """Compute max pairwise correlation between strategy returns."""
        names = [k for k, v in self._return_matrix.items() if len(v) >= 10]
        if len(names) < 2:
            return 0.0

        max_corr = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                corr = self._pearson_corr(
                    self._return_matrix[names[i]][-100:],
                    self._return_matrix[names[j]][-100:],
                )
                max_corr = max(max_corr, abs(corr))
        return max_corr

    def _compute_portfolio_var(self, allocations: Dict[str, float]) -> float:
        """Simplified portfolio VaR: weighted sum of individual VaRs."""
        if not allocations:
            return 0.0
        total_var = 0.0
        for name, weight in allocations.items():
            perf = self._strategies.get(name)
            if perf and perf.volatility > 0:
                # VaR ≈ 1.65 * vol (95% confidence, normal assumption)
                individual_var = 1.65 * perf.volatility * weight
                total_var += individual_var ** 2  # sum of squared (assumes some independence)
        return total_var ** 0.5

    def _pearson_corr(self, a: List[float], b: List[float]) -> float:
        n = min(len(a), len(b))
        if n < 5:
            return 0.0
        a, b = a[-n:], b[-n:]
        ma = sum(a) / n
        mb = sum(b) / n
        cov = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b)) / (n - 1)
        sa = (sum((ai - ma) ** 2 for ai in a) / (n - 1)) ** 0.5
        sb = (sum((bi - mb) ** 2 for bi in b) / (n - 1)) ** 0.5
        return cov / max(sa * sb, 1e-9)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "strategies_tracked": len(self._strategies),
            "current_allocations": dict(self._current_allocations),
            "strategy_sharpes": {k: v.sharpe for k, v in self._strategies.items()},
            "strategy_drawdowns": {k: v.current_drawdown for k, v in self._strategies.items()},
        }
