"""
Push 88 — BanditRouter
=======================
Hard capital allocation layer that dynamically routes available capital
across active strategies using three inputs:

  1. **BanditAllocator** (Thompson-Sampling posteriors)
  2. **RegimeAwareConsensus** (multipliers)
  3. **TradeLedger** (Push 87)

Capital routing rules
---------------------
* Concentration cap  : no single strategy receives > ``max_concentration``
  of total capital (default 40 %).
* Floor              : every active strategy gets at least ``min_alloc_usd``
  unless its regime multiplier is 0.0 (effectively disabled).
* Kill switch        : a strategy whose rolling Sharpe drops below
  ``sharpe_kill_threshold`` is moved to SUSPENDED and receives zero capital
  until manually re-enabled or it recovers above ``sharpe_resume_threshold``.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from strategies.bandit_allocator import BanditAllocator
from strategies.regime_consensus import MarketRegime, REGIME_MULTIPLIERS
from execution.trade_ledger import TradeLedger

logger = logging.getLogger("argus.bandit_router")


class StrategyStatus(str, Enum):
    ACTIVE    = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    DISABLED  = "DISABLED"


@dataclass
class StrategySpec:
    """Registration record for a strategy managed by BanditRouter."""
    name: str
    category: str
    status: StrategyStatus = StrategyStatus.ACTIVE
    min_alloc_usd: float = 0.0
    max_alloc_usd: float = 0.0


@dataclass
class AllocationRecord:
    """Snapshot of a single strategy's allocation at a point in time."""
    strategy: str
    status: StrategyStatus
    bandit_weight: float
    regime_multiplier: float
    final_weight: float
    allocated_usd: float
    rolling_pnl_usd: float
    rolling_sharpe: float


class BanditRouter:
    """
    Dynamic capital router combining Thompson-Sampling bandit allocation
    with regime-aware multipliers and TradeLedger P&L feedback.
    """

    def __init__(
        self,
        total_capital_usd: float,
        strategies: List[StrategySpec],
        ledger: Optional[TradeLedger] = None,
        *,
        max_concentration: float = 0.40,
        min_alloc_usd: float = 50.0,
        decay_halflife_trades: int = 100,
        sharpe_kill_threshold: float = -0.5,
        sharpe_resume_threshold: float = 0.2,
        kill_lookback_h: float = 24.0,
        n_thompson_samples: int = 5,
    ) -> None:
        if total_capital_usd <= 0:
            raise ValueError("total_capital_usd must be positive")
        if not 0 < max_concentration <= 1.0:
            raise ValueError("max_concentration must be in (0, 1]")

        self.total_capital_usd = total_capital_usd
        self._specs: Dict[str, StrategySpec] = {s.name: s for s in strategies}
        self._ledger = ledger
        self.max_concentration = max_concentration
        self.min_alloc_usd = min_alloc_usd
        self.sharpe_kill_threshold = sharpe_kill_threshold
        self.sharpe_resume_threshold = sharpe_resume_threshold
        self.kill_lookback_h = kill_lookback_h
        self._bandit = BanditAllocator(
            decay_halflife_trades=decay_halflife_trades,
            min_weight=0.0,
            n_thompson_samples=n_thompson_samples,
        )
        self._pnl_history: Dict[str, List[Tuple[float, float]]] = {}
        logger.info(
            "BanditRouter init: capital=%.2f strategies=%d max_conc=%.0f%% kill_sharpe=%.2f",
            total_capital_usd, len(strategies), max_concentration * 100, sharpe_kill_threshold,
        )

    def record_outcome(self, strategy_name: str, pnl_usd: float) -> None:
        """Feed a completed trade outcome into the bandit and P&L history."""
        if strategy_name not in self._specs:
            logger.warning("record_outcome: unknown strategy %r — ignoring", strategy_name)
            return
        self._bandit.update(strategy_name, pnl_usd)
        buf = self._pnl_history.setdefault(strategy_name, [])
        buf.append((time.time(), pnl_usd))
        cutoff = time.time() - self.kill_lookback_h * 2 * 3600.0
        self._pnl_history[strategy_name] = [(ts, p) for ts, p in buf if ts >= cutoff]
        self._maybe_update_status(strategy_name)

    def allocations(
        self,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        capital_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute and return USD allocations for all ACTIVE strategies."""
        capital = capital_override if capital_override is not None else self.total_capital_usd
        active = [s for s in self._specs.values() if s.status == StrategyStatus.ACTIVE]
        if not active:
            logger.warning("BanditRouter: no active strategies — returning empty allocation")
            return {s: 0.0 for s in self._specs}
        active_names = [s.name for s in active]
        bandit_weights = self._bandit.weights(active_names)
        regime_table = REGIME_MULTIPLIERS.get(regime, {})
        adjusted: Dict[str, float] = {}
        for spec in active:
            mult = regime_table.get(spec.category, 1.0)
            adjusted[spec.name] = bandit_weights.get(spec.name, 0.0) * mult
        total_adj = sum(adjusted.values())
        if total_adj <= 0:
            equal = 1.0 / len(active)
            adjusted = {n: equal for n in active_names}
            total_adj = 1.0
        norm = {n: w / total_adj for n, w in adjusted.items()}
        capped = self._apply_concentration_cap(norm)
        raw_usd = {n: capped[n] * capital for n in active_names}
        final_usd = self._apply_floors(raw_usd, active, capital)
        return {name: final_usd.get(name, 0.0) for name in self._specs}

    def update_capital(self, new_total_usd: float) -> None:
        """Update total capital."""
        if new_total_usd <= 0:
            raise ValueError("new_total_usd must be positive")
        self.total_capital_usd = new_total_usd
        logger.info("BanditRouter: capital updated to %.2f", new_total_usd)

    def set_status(self, strategy_name: str, status: StrategyStatus) -> None:
        """Manually override a strategy's status."""
        if strategy_name not in self._specs:
            raise KeyError(f"Unknown strategy: {strategy_name!r}")
        old = self._specs[strategy_name].status
        self._specs[strategy_name].status = status
        logger.info("BanditRouter: %s status %s → %s", strategy_name, old.value, status.value)

    def register(self, spec: StrategySpec) -> None:
        """Register a new strategy at runtime."""
        self._specs[spec.name] = spec
        logger.info("BanditRouter: registered strategy %s (category=%s)", spec.name, spec.category)

    def snapshot(self, regime: MarketRegime = MarketRegime.UNKNOWN) -> List[AllocationRecord]:
        """Return a full diagnostic snapshot of the current allocation state."""
        allocs = self.allocations(regime=regime)
        active_names = [s.name for s in self._specs.values() if s.status == StrategyStatus.ACTIVE]
        bandit_weights = self._bandit.weights(active_names) if active_names else {}
        regime_table = REGIME_MULTIPLIERS.get(regime, {})
        records = []
        for spec in self._specs.values():
            records.append(AllocationRecord(
                strategy=spec.name,
                status=spec.status,
                bandit_weight=round(bandit_weights.get(spec.name, 0.0), 4),
                regime_multiplier=regime_table.get(spec.category, 1.0),
                final_weight=round(allocs.get(spec.name, 0.0) / self.total_capital_usd, 4),
                allocated_usd=round(allocs.get(spec.name, 0.0), 2),
                rolling_pnl_usd=round(self._rolling_pnl(spec.name), 4),
                rolling_sharpe=round(self._rolling_sharpe(spec.name), 3),
            ))
        records.sort(key=lambda r: r.allocated_usd, reverse=True)
        return records

    def bandit_snapshot(self) -> List[dict]:
        """Raw Thompson-Sampling arm state."""
        return self._bandit.snapshot()

    def _apply_concentration_cap(self, weights: Dict[str, float]) -> Dict[str, float]:
        cap = self.max_concentration
        w = dict(weights)
        for _ in range(len(w) + 1):
            over = {k: v for k, v in w.items() if v > cap}
            if not over:
                break
            excess = sum(v - cap for v in over.values())
            under = [k for k, v in w.items() if v < cap]
            for k in over:
                w[k] = cap
            if under:
                per = excess / len(under)
                for k in under:
                    w[k] = min(cap, w[k] + per)
            else:
                break
        return w

    def _apply_floors(self, raw_usd: Dict[str, float], active: List[StrategySpec], capital: float) -> Dict[str, float]:
        result = dict(raw_usd)
        for spec in active:
            floor = spec.min_alloc_usd if spec.min_alloc_usd > 0 else self.min_alloc_usd
            if result.get(spec.name, 0.0) < floor:
                result[spec.name] = floor
        total = sum(result.values())
        if total > capital and total > 0:
            scale = capital / total
            result = {k: v * scale for k, v in result.items()}
        return result

    def _rolling_pnl(self, strategy: str) -> float:
        cutoff = time.time() - self.kill_lookback_h * 3600.0
        return sum(p for ts, p in self._pnl_history.get(strategy, []) if ts >= cutoff)

    def _rolling_sharpe(self, strategy: str) -> float:
        cutoff = time.time() - self.kill_lookback_h * 3600.0
        pnls = [p for ts, p in self._pnl_history.get(strategy, []) if ts >= cutoff]
        if len(pnls) < 3:
            return 0.0
        n = len(pnls)
        mean = sum(pnls) / n
        variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        trades_per_day = n / max(self.kill_lookback_h / 24.0, 1e-6)
        return (mean / std) * math.sqrt(252 * trades_per_day)

    def _maybe_update_status(self, strategy_name: str) -> None:
        spec = self._specs[strategy_name]
        sharpe = self._rolling_sharpe(strategy_name)
        if spec.status == StrategyStatus.ACTIVE:
            if sharpe < self.sharpe_kill_threshold and len(self._pnl_history.get(strategy_name, [])) >= 10:
                logger.warning("BanditRouter KILL: %s rolling_sharpe=%.3f < %.3f — suspending", strategy_name, sharpe, self.sharpe_kill_threshold)
                spec.status = StrategyStatus.SUSPENDED
        elif spec.status == StrategyStatus.SUSPENDED:
            if sharpe >= self.sharpe_resume_threshold:
                logger.info("BanditRouter RESUME: %s rolling_sharpe=%.3f >= %.3f — re-activating", strategy_name, sharpe, self.sharpe_resume_threshold)
                spec.status = StrategyStatus.ACTIVE
