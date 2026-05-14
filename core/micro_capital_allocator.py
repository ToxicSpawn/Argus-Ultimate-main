"""
MicroCapitalAllocator — Dynamic capital allocation for $1k AUD (~$620 USD) HFT accounts.

Manages allocation between:
  - Market Making (MM): captures spread on altcoin pairs on Bybit (zero maker fee)
  - Funding Rate Arb: delta-neutral carry on Bybit perpetuals
  - Cash Reserve: always kept > reserve_pct for safety

Rebalances hourly based on rolling 24h Sharpe ratio and funding yield metrics.
Triggers global killswitch if total drawdown exceeds max_drawdown_pct.

Design constraints:
  - Minimum $50 per active strategy (below this, fees and sizing don't work)
  - Never allocate > (100% - reserve_pct) total
  - Annualised Sharpe = mean_hourly_return × sqrt(24×365) / std_hourly_return
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOURS_PER_YEAR: float = 24.0 * 365.0          # 8760
ANNUALISE_FACTOR: float = math.sqrt(HOURS_PER_YEAR)  # sqrt(8760)
MINIMUM_STRATEGY_CAPITAL_USD: float = 50.0     # below this, strategy is disabled
ROLLING_WINDOW_HOURS: int = 24                 # look-back for Sharpe calculation
NS_PER_SECOND: int = 1_000_000_000


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AllocatorConfig:
    """Configuration for MicroCapitalAllocator."""
    total_capital_usd: float = 620.0          # ~$1000 AUD @ 0.62 rate
    base_mm_pct: float = 0.55                 # 55% → ~$341
    base_funding_pct: float = 0.40            # 40% → ~$248
    reserve_pct: float = 0.05                 # 5%  → ~$31
    rebalance_interval_s: float = 3600.0      # hourly rebalance check
    max_drawdown_pct: float = 15.0            # 15% hard stop = ~$93 USD
    mm_sharpe_threshold: float = 0.5          # reduce MM if Sharpe < this
    funding_yield_threshold: float = 0.10     # 10% annualised minimum

    def __post_init__(self) -> None:
        total = self.base_mm_pct + self.base_funding_pct + self.reserve_pct
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"AllocatorConfig: base_mm_pct + base_funding_pct + reserve_pct "
                f"must sum to 1.0, got {total:.6f}"
            )
        if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 100:
            raise ValueError("max_drawdown_pct must be in (0, 100]")
        if self.reserve_pct < 0.01:
            raise ValueError("reserve_pct must be >= 1% for safety")


@dataclass
class Allocation:
    """Snapshot of the current capital allocation."""
    mm_capital_usd: float
    funding_capital_usd: float
    reserve_usd: float
    mm_pct: float
    funding_pct: float
    reserve_pct: float
    total_capital_usd: float
    last_rebalance_ns: int
    reason: str

    @property
    def active_capital_usd(self) -> float:
        """Capital actually deployed (excludes reserve)."""
        return self.mm_capital_usd + self.funding_capital_usd

    def to_dict(self) -> Dict:
        return {
            "mm_capital_usd": round(self.mm_capital_usd, 4),
            "funding_capital_usd": round(self.funding_capital_usd, 4),
            "reserve_usd": round(self.reserve_usd, 4),
            "mm_pct": round(self.mm_pct * 100, 2),
            "funding_pct": round(self.funding_pct * 100, 2),
            "reserve_pct": round(self.reserve_pct * 100, 2),
            "total_capital_usd": round(self.total_capital_usd, 4),
            "last_rebalance_ns": self.last_rebalance_ns,
            "reason": self.reason,
        }


@dataclass
class _HourlyReturn:
    """Single hourly return record for a strategy."""
    timestamp_ns: int
    pnl: float
    sharpe_contribution: float = 0.0  # not used directly; kept for audit


@dataclass
class _StrategyState:
    """Internal state for a single strategy."""
    name: str
    current_capital_usd: float = 0.0
    total_pnl: float = 0.0
    current_sharpe: float = 0.0
    current_yield_annualised: float = 0.0
    hourly_returns: Deque[_HourlyReturn] = field(
        default_factory=lambda: deque(maxlen=ROLLING_WINDOW_HOURS)
    )
    last_update_ns: int = 0
    enabled: bool = True

    def add_return(self, pnl: float, sharpe: float, timestamp_ns: int) -> None:
        self.total_pnl += pnl
        self.current_sharpe = sharpe
        self.last_update_ns = timestamp_ns
        self.hourly_returns.append(_HourlyReturn(timestamp_ns=timestamp_ns, pnl=pnl))

    def compute_rolling_sharpe(self) -> float:
        """
        Compute rolling 24h Sharpe ratio from stored hourly returns.
        Annualised: mean_hourly × sqrt(8760) / std_hourly

        Returns:
            Annualised Sharpe or 0.0 if insufficient data.
        """
        if len(self.hourly_returns) < 2:
            return 0.0

        returns = [r.pnl for r in self.hourly_returns]
        n = len(returns)
        mean_r = sum(returns) / n
        variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0

        if std_r < 1e-12:
            # All returns identical — if positive, infinite Sharpe; treat as high
            return 10.0 if mean_r > 0 else (0.0 if mean_r == 0 else -10.0)

        return (mean_r * ANNUALISE_FACTOR) / std_r

    def compute_annualised_yield(self) -> float:
        """
        Estimate annualised yield from the rolling 24h PnL window
        relative to the capital allocated to this strategy.

        Returns:
            Annualised yield as a fraction (e.g., 0.15 = 15%) or 0.0 if unknown.
        """
        if self.current_capital_usd <= 0 or not self.hourly_returns:
            return 0.0

        # Sum PnL over the window
        window_hours = len(self.hourly_returns)
        window_pnl = sum(r.pnl for r in self.hourly_returns)

        if window_hours == 0:
            return 0.0

        # Annualise: (pnl / capital_per_hour) * hours_per_year
        # This gives the geometric-ish daily yield × 365
        hourly_return = window_pnl / (self.current_capital_usd * window_hours)
        return hourly_return * HOURS_PER_YEAR


# ---------------------------------------------------------------------------
# MicroCapitalAllocator
# ---------------------------------------------------------------------------

class MicroCapitalAllocator:
    """
    Dynamic capital allocator for $620 USD (~$1000 AUD) micro-capital HFT accounts.

    Manages two active strategies:
      * Market Making (MM) on Bybit altcoin pairs
      * Funding Rate Arbitrage (delta-neutral carry)

    Rebalances hourly using rolling 24h Sharpe ratio and funding yield metrics.
    Triggers a global killswitch if total drawdown > max_drawdown_pct.

    Thread-safety note: this class is NOT thread-safe. Wrap with asyncio.Lock
    if calling from multiple coroutines.
    """

    STRATEGY_MM = "mm"
    STRATEGY_FUNDING = "funding"

    def __init__(self, config: Optional[AllocatorConfig] = None) -> None:
        self._cfg = config or AllocatorConfig()
        self._total_capital = self._cfg.total_capital_usd
        self._halted: bool = False
        self._last_rebalance_ns: int = time.time_ns()
        self._next_rebalance_ns: int = (
            self._last_rebalance_ns
            + int(self._cfg.rebalance_interval_s * NS_PER_SECOND)
        )

        # Strategy states
        self._strategies: Dict[str, _StrategyState] = {
            self.STRATEGY_MM: _StrategyState(
                name=self.STRATEGY_MM,
                current_capital_usd=self._cfg.total_capital_usd * self._cfg.base_mm_pct,
                enabled=True,
            ),
            self.STRATEGY_FUNDING: _StrategyState(
                name=self.STRATEGY_FUNDING,
                current_capital_usd=self._cfg.total_capital_usd * self._cfg.base_funding_pct,
                enabled=True,
            ),
        }

        # Current allocation (set at init)
        self._current_allocation: Allocation = self._build_allocation(
            mm_pct=self._cfg.base_mm_pct,
            funding_pct=self._cfg.base_funding_pct,
            reserve_pct=self._cfg.reserve_pct,
            reason="initial_allocation",
        )

        logger.info(
            "MicroCapitalAllocator initialised: $%.2f USD total | MM=$%.2f | "
            "Funding=$%.2f | Reserve=$%.2f",
            self._total_capital,
            self._current_allocation.mm_capital_usd,
            self._current_allocation.funding_capital_usd,
            self._current_allocation.reserve_usd,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_allocation(self) -> Allocation:
        """Return current USD allocation snapshot."""
        return self._current_allocation

    def update_performance(
        self,
        strategy: str,
        pnl: float,
        sharpe: float,
        timestamp_ns: int,
    ) -> None:
        """
        Record a performance update for a strategy.

        Args:
            strategy: "mm" or "funding"
            pnl: PnL in USD for this period (typically 1 hour)
            sharpe: Externally supplied Sharpe ratio (used as override if provided;
                    the internal rolling calculation is also updated)
            timestamp_ns: nanosecond timestamp of this measurement
        """
        if strategy not in self._strategies:
            logger.warning("Unknown strategy '%s' in update_performance", strategy)
            return

        state = self._strategies[strategy]
        state.add_return(pnl=pnl, sharpe=sharpe, timestamp_ns=timestamp_ns)

        # Recompute internal rolling Sharpe (overrides external if we have enough data)
        if len(state.hourly_returns) >= 2:
            state.current_sharpe = state.compute_rolling_sharpe()

        # Recompute annualised yield (used for funding strategy gate)
        state.current_yield_annualised = state.compute_annualised_yield()

        logger.debug(
            "Performance update: strategy=%s pnl=%.6f sharpe=%.3f "
            "yield_ann=%.2f%% total_pnl=%.4f",
            strategy,
            pnl,
            state.current_sharpe,
            state.current_yield_annualised * 100,
            state.total_pnl,
        )

    def rebalance(self) -> Allocation:
        """
        Rebalance capital across strategies based on current performance.

        Logic:
          1. If halted (killswitch): return zero allocation
          2. If MM Sharpe > threshold AND funding yield > threshold → base allocation
          3. If MM Sharpe < threshold only → shift 20% of MM → reserve
          4. If funding yield < threshold only → shift funding → MM
          5. If both poor → 80% reserve, 20% best performer

        Never allocates > (1 - reserve_pct) of total.
        Disables strategies below MINIMUM_STRATEGY_CAPITAL_USD.

        Returns:
            New Allocation dataclass with reason field explaining decision.
        """
        now_ns = time.time_ns()

        if self._halted:
            self._current_allocation = self._zero_allocation(
                reason="halted_killswitch_already_triggered"
            )
            return self._current_allocation

        # Check killswitch before making any new allocation
        if self.check_global_killswitch():
            return self._current_allocation  # already set to zero inside

        mm_state = self._strategies[self.STRATEGY_MM]
        funding_state = self._strategies[self.STRATEGY_FUNDING]

        mm_sharpe = mm_state.current_sharpe
        funding_yield = funding_state.current_yield_annualised

        mm_ok = mm_sharpe >= self._cfg.mm_sharpe_threshold
        funding_ok = funding_yield >= self._cfg.funding_yield_threshold

        reserve_base = self._cfg.reserve_pct

        if mm_ok and funding_ok:
            # Both performing — use base allocation
            new_mm_pct = self._cfg.base_mm_pct
            new_funding_pct = self._cfg.base_funding_pct
            new_reserve_pct = reserve_base
            reason = (
                f"both_performing: mm_sharpe={mm_sharpe:.3f} "
                f"funding_yield={funding_yield*100:.2f}%"
            )

        elif not mm_ok and funding_ok:
            # MM underperforming — shift 20% of MM into reserve
            shift_from_mm = self._cfg.base_mm_pct * 0.20
            new_mm_pct = self._cfg.base_mm_pct - shift_from_mm
            new_funding_pct = self._cfg.base_funding_pct
            new_reserve_pct = min(reserve_base + shift_from_mm, 1.0 - new_funding_pct - new_mm_pct)
            # Normalise
            total = new_mm_pct + new_funding_pct + new_reserve_pct
            if not math.isclose(total, 1.0, rel_tol=1e-6):
                new_reserve_pct = 1.0 - new_mm_pct - new_funding_pct
            reason = (
                f"mm_low_sharpe: mm_sharpe={mm_sharpe:.3f}<{self._cfg.mm_sharpe_threshold}, "
                f"shifted 20% MM→reserve"
            )

        elif mm_ok and not funding_ok:
            # Funding underperforming — shift all funding → MM
            new_mm_pct = self._cfg.base_mm_pct + self._cfg.base_funding_pct
            new_funding_pct = 0.0
            new_reserve_pct = reserve_base
            # Cap at max deployable
            max_deployable = 1.0 - reserve_base
            if new_mm_pct > max_deployable:
                new_mm_pct = max_deployable
            reason = (
                f"funding_low_yield: yield={funding_yield*100:.2f}%<"
                f"{self._cfg.funding_yield_threshold*100:.0f}%, shifted funding→MM"
            )

        else:
            # Both poor — 80% reserve, 20% best performer
            # Determine best performer by total PnL
            if mm_state.total_pnl >= funding_state.total_pnl:
                best = "mm"
            else:
                best = "funding"

            active_pct = 0.20  # 20% to best performer
            new_reserve_pct = 0.80
            if best == "mm":
                new_mm_pct = active_pct
                new_funding_pct = 0.0
            else:
                new_mm_pct = 0.0
                new_funding_pct = active_pct

            reason = (
                f"both_poor: mm_sharpe={mm_sharpe:.3f} "
                f"funding_yield={funding_yield*100:.2f}%, "
                f"best_performer={best}"
            )

        # Enforce minimum allocation — disable strategies below $50
        total_capital = self._total_capital
        if new_mm_pct * total_capital < MINIMUM_STRATEGY_CAPITAL_USD and new_mm_pct > 0:
            logger.warning(
                "MM allocation $%.2f < minimum $%.2f — disabling MM",
                new_mm_pct * total_capital,
                MINIMUM_STRATEGY_CAPITAL_USD,
            )
            new_reserve_pct += new_mm_pct
            new_mm_pct = 0.0
            reason += " [mm_below_minimum_disabled]"

        if new_funding_pct * total_capital < MINIMUM_STRATEGY_CAPITAL_USD and new_funding_pct > 0:
            logger.warning(
                "Funding allocation $%.2f < minimum $%.2f — disabling funding",
                new_funding_pct * total_capital,
                MINIMUM_STRATEGY_CAPITAL_USD,
            )
            new_reserve_pct += new_funding_pct
            new_funding_pct = 0.0
            reason += " [funding_below_minimum_disabled]"

        # Final safety clamp: (mm + funding) <= (1 - reserve_pct)
        max_active = 1.0 - reserve_base
        if new_mm_pct + new_funding_pct > max_active:
            scale = max_active / (new_mm_pct + new_funding_pct)
            new_mm_pct *= scale
            new_funding_pct *= scale
            reason += f" [scaled_down_to_{max_active*100:.0f}%_active]"

        # Re-normalise to ensure sum == 1.0
        total_check = new_mm_pct + new_funding_pct + new_reserve_pct
        if not math.isclose(total_check, 1.0, rel_tol=1e-5):
            delta = 1.0 - total_check
            new_reserve_pct += delta  # absorb rounding into reserve

        self._last_rebalance_ns = now_ns
        self._next_rebalance_ns = now_ns + int(
            self._cfg.rebalance_interval_s * NS_PER_SECOND
        )

        # Update strategy capital state
        self._strategies[self.STRATEGY_MM].current_capital_usd = (
            new_mm_pct * total_capital
        )
        self._strategies[self.STRATEGY_MM].enabled = new_mm_pct > 0

        self._strategies[self.STRATEGY_FUNDING].current_capital_usd = (
            new_funding_pct * total_capital
        )
        self._strategies[self.STRATEGY_FUNDING].enabled = new_funding_pct > 0

        self._current_allocation = self._build_allocation(
            mm_pct=new_mm_pct,
            funding_pct=new_funding_pct,
            reserve_pct=new_reserve_pct,
            reason=reason,
        )

        logger.info(
            "Rebalanced: MM=$%.2f(%.1f%%) Funding=$%.2f(%.1f%%) Reserve=$%.2f | %s",
            self._current_allocation.mm_capital_usd,
            new_mm_pct * 100,
            self._current_allocation.funding_capital_usd,
            new_funding_pct * 100,
            self._current_allocation.reserve_usd,
            reason,
        )

        return self._current_allocation

    def check_global_killswitch(self) -> bool:
        """
        Check if total PnL has breached the max drawdown limit.

        Returns:
            True if killswitch triggered (all allocations set to zero).
            False if within acceptable drawdown.
        """
        if self._halted:
            return True

        total_pnl = (
            self._strategies[self.STRATEGY_MM].total_pnl
            + self._strategies[self.STRATEGY_FUNDING].total_pnl
        )

        drawdown_limit_usd = -(
            self._cfg.max_drawdown_pct / 100.0 * self._cfg.total_capital_usd
        )

        if total_pnl <= drawdown_limit_usd:
            logger.critical(
                "GLOBAL KILLSWITCH TRIGGERED: total_pnl=%.4f <= limit=%.4f USD "
                "(%.1f%% drawdown)",
                total_pnl,
                drawdown_limit_usd,
                abs(total_pnl) / self._cfg.total_capital_usd * 100,
            )
            self._halted = True
            self._current_allocation = self._zero_allocation(
                reason=(
                    f"global_killswitch: pnl={total_pnl:.4f} <= "
                    f"limit={drawdown_limit_usd:.4f}"
                )
            )
            return True

        return False

    def get_stats(self) -> Dict:
        """
        Return a comprehensive stats dictionary for monitoring.

        Keys:
            mm_capital_usd, funding_capital_usd, reserve_usd,
            mm_pnl, funding_pnl, total_pnl,
            mm_sharpe, funding_yield_annualised,
            global_drawdown_pct, max_drawdown_pct,
            halted, next_rebalance_ns, rebalance_in_seconds,
            allocation_reason
        """
        mm = self._strategies[self.STRATEGY_MM]
        funding = self._strategies[self.STRATEGY_FUNDING]
        total_pnl = mm.total_pnl + funding.total_pnl
        drawdown_pct = (
            abs(total_pnl) / self._cfg.total_capital_usd * 100
            if total_pnl < 0
            else 0.0
        )

        now_ns = time.time_ns()
        rebalance_in_s = max(
            0.0,
            (self._next_rebalance_ns - now_ns) / NS_PER_SECOND,
        )

        return {
            # Allocations
            "mm_capital_usd": round(self._current_allocation.mm_capital_usd, 4),
            "funding_capital_usd": round(
                self._current_allocation.funding_capital_usd, 4
            ),
            "reserve_usd": round(self._current_allocation.reserve_usd, 4),
            "total_capital_usd": round(self._cfg.total_capital_usd, 4),
            # Per-strategy PnL
            "mm_pnl": round(mm.total_pnl, 4),
            "funding_pnl": round(funding.total_pnl, 4),
            "total_pnl": round(total_pnl, 4),
            # Performance metrics
            "mm_sharpe": round(mm.current_sharpe, 4),
            "mm_enabled": mm.enabled,
            "funding_yield_annualised_pct": round(
                funding.current_yield_annualised * 100, 4
            ),
            "funding_enabled": funding.enabled,
            # Risk
            "global_drawdown_pct": round(drawdown_pct, 4),
            "max_drawdown_pct": self._cfg.max_drawdown_pct,
            "halted": self._halted,
            # Rebalance timing
            "last_rebalance_ns": self._last_rebalance_ns,
            "next_rebalance_ns": self._next_rebalance_ns,
            "rebalance_in_seconds": round(rebalance_in_s, 1),
            "allocation_reason": self._current_allocation.reason,
        }

    def force_rebalance_now(self) -> Allocation:
        """Force an immediate rebalance regardless of interval timing."""
        self._next_rebalance_ns = time.time_ns()
        return self.rebalance()

    def should_rebalance(self) -> bool:
        """Return True if the rebalance interval has elapsed."""
        return time.time_ns() >= self._next_rebalance_ns

    def set_total_capital(self, capital_usd: float) -> None:
        """
        Update total capital (e.g., after deposit/withdrawal).
        Triggers an immediate rebalance.
        """
        if capital_usd < 0:
            raise ValueError("total_capital_usd cannot be negative")
        logger.info(
            "Capital updated: $%.2f → $%.2f USD", self._total_capital, capital_usd
        )
        self._total_capital = capital_usd
        self._cfg.total_capital_usd = capital_usd
        self.force_rebalance_now()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_allocation(
        self,
        mm_pct: float,
        funding_pct: float,
        reserve_pct: float,
        reason: str,
    ) -> Allocation:
        """Construct an Allocation from percentage fractions."""
        total = self._total_capital
        return Allocation(
            mm_capital_usd=round(mm_pct * total, 6),
            funding_capital_usd=round(funding_pct * total, 6),
            reserve_usd=round(reserve_pct * total, 6),
            mm_pct=mm_pct,
            funding_pct=funding_pct,
            reserve_pct=reserve_pct,
            total_capital_usd=total,
            last_rebalance_ns=self._last_rebalance_ns,
            reason=reason,
        )

    def _zero_allocation(self, reason: str) -> Allocation:
        """Return a halted allocation with all strategies at zero."""
        # Reserve all capital
        return Allocation(
            mm_capital_usd=0.0,
            funding_capital_usd=0.0,
            reserve_usd=self._total_capital,
            mm_pct=0.0,
            funding_pct=0.0,
            reserve_pct=1.0,
            total_capital_usd=self._total_capital,
            last_rebalance_ns=time.time_ns(),
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        a = self._current_allocation
        return (
            f"MicroCapitalAllocator("
            f"total=${a.total_capital_usd:.2f}, "
            f"mm=${a.mm_capital_usd:.2f}, "
            f"funding=${a.funding_capital_usd:.2f}, "
            f"reserve=${a.reserve_usd:.2f}, "
            f"halted={self._halted})"
        )
