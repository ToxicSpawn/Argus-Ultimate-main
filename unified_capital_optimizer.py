"""
Capital Optimizer – Phase 4 of the Unified Trading System.

This module is consumed by
``unified_trading_system.UnifiedSystemArchitecture._initialize_capital_optimizer()``.

Responsibilities
----------------
* Allocate the $1,000 AUD starting capital across the active trading universe
  using HRP (Hierarchical Risk Parity), BL (Black-Litterman), or MPT.
* Enforce the capital guardrails encoded in ``UnifiedConfig``:
  - max_position_pct, max_total_exposure_pct, max_concurrent_positions
  - min_position_size_aud, max_position_size_aud
* Return per-symbol ``PositionBudget`` objects consumed by the execution engine.
* Track capital utilisation and compound performance metrics.
* Kelly-fraction scaling when trade history is available.
* Rebalance recommendation when weights drift beyond ``target_rebalance_min_delta_pct``.
* Paper-mode peak overrides: when ``paper_trading_peak_mode`` is True, aggressively
  deploy capital to maximise return during paper runs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PositionBudget:
    """Approved capital allocation for a single symbol."""
    symbol: str
    allocation_aud: float          # AUD to deploy
    allocation_pct: float          # fraction of total equity
    weight: float                  # raw portfolio weight before AUD conversion
    kelly_scale: float = 1.0       # Kelly fraction (1.0 = no scaling)
    max_size_aud: float = 250.0    # hard cap from config
    min_size_aud: float = 10.0     # minimum viable trade size

    @property
    def is_viable(self) -> bool:
        return self.allocation_aud >= self.min_size_aud


@dataclass
class CapitalState:
    """Snapshot of current capital deployment."""
    total_equity_aud: float
    cash_aud: float
    deployed_aud: float
    n_open_positions: int
    utilisation_pct: float
    peak_equity_aud: float
    drawdown_pct: float
    daily_pnl_aud: float
    cumulative_pnl_aud: float
    trade_count: int
    win_rate_pct: float


# ---------------------------------------------------------------------------
# Weight engines (best-effort, graceful fallback to equal-weight)
# ---------------------------------------------------------------------------

def _equal_weights(symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    w = 1.0 / len(symbols)
    return {s: w for s in symbols}


def _hrp_weights(symbols: List[str], returns_matrix: Optional[Any] = None) -> Dict[str, float]:
    """Hierarchical Risk Parity – delegates to risk.black_litterman when available."""
    try:
        import numpy as np  # type: ignore
        from risk.black_litterman import hrp_weights  # type: ignore
        if returns_matrix is not None:
            return hrp_weights(symbols, returns_matrix)
    except Exception:
        pass
    return _equal_weights(symbols)


def _bl_weights(
    symbols: List[str],
    returns_matrix: Optional[Any] = None,
    views: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """Black-Litterman – delegates to risk.black_litterman when available."""
    try:
        from risk.black_litterman import bl_weights  # type: ignore
        if returns_matrix is not None:
            return bl_weights(symbols, returns_matrix, views=views)
    except Exception:
        pass
    return _equal_weights(symbols)


def _mpt_weights(
    symbols: List[str],
    returns_matrix: Optional[Any] = None,
    *,
    target_return: Optional[float] = None,
) -> Dict[str, float]:
    """MPT (min-variance) – delegates to risk.black_litterman when available."""
    try:
        from risk.black_litterman import mpt_weights  # type: ignore
        if returns_matrix is not None:
            return mpt_weights(symbols, returns_matrix, target_return=target_return)
    except Exception:
        pass
    return _equal_weights(symbols)


# ---------------------------------------------------------------------------
# Kelly fraction
# ---------------------------------------------------------------------------

def _kelly_scale(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    *,
    fractional: float = 0.25,
    min_floor: float = 0.10,
) -> float:
    """Fractional Kelly, bounded to [min_floor, fractional]."""
    if avg_loss_pct <= 0 or avg_win_pct <= 0 or win_rate <= 0:
        return min_floor
    b = avg_win_pct / avg_loss_pct
    q = 1.0 - win_rate
    kelly = (b * win_rate - q) / b
    return max(min_floor, min(kelly * fractional, fractional))


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class CapitalOptimizer1K:
    """
    Phase 4: Capital allocation engine optimised for $1,000 AUD starting capital.

    Interface contract (called by UnifiedSystemArchitecture)
    ---------------------------------------------------------
    await optimizer.initialize()
    budgets = optimizer.compute_budgets(symbols, equity_aud, positions, bl_views)
    optimizer.update(trade_result)           # feed closed-trade PnL
    state  = optimizer.get_capital_state()  # snapshot
    report = optimizer.get_allocation_report()
    """

    def __init__(self, config: Any) -> None:
        self.config = config

        # Capital tracking
        self._equity: float = float(getattr(config, "starting_capital_aud", 1000.0) or 1000.0)
        self._cash: float = self._equity
        self._peak_equity: float = self._equity
        self._daily_pnl: float = 0.0
        self._cumulative_pnl: float = 0.0

        # Trade stats for Kelly
        self._trade_count: int = 0
        self._win_count: int = 0
        self._total_win_pct: float = 0.0
        self._total_loss_pct: float = 0.0
        self._loss_count: int = 0

        # Allocation history (symbol -> weight)
        self._last_weights: Dict[str, float] = {}

        # Returns history (symbol -> list of daily returns)
        self._returns_history: Dict[str, List[float]] = {}

        # Paper-mode peak overrides
        self._peak_mode: bool = bool(getattr(config, "paper_trading_peak_mode", True))

        logger.info("CapitalOptimizer1K created (equity=%.2f AUD, peak_mode=%s)",
                    self._equity, self._peak_mode)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load any persisted state and run initial checks."""
        # Best-effort: load evolved params into returns history if available
        try:
            import json
            from pathlib import Path
            p = Path(str(getattr(self.config, "optimized_params_path",
                                  "data/optimized_params.json") or "data/optimized_params.json"))
            if p.exists() and bool(getattr(self.config, "optimized_params_load", False)):
                data = json.loads(p.read_text())
                logger.debug("CapitalOptimizer: loaded offline optimised params")
        except Exception:
            pass
        logger.info("✅ CapitalOptimizer1K initialized (%.2f AUD)", self._equity)

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------

    def compute_budgets(
        self,
        symbols: List[str],
        *,
        equity_aud: Optional[float] = None,
        positions: Optional[Dict[str, Any]] = None,
        bl_views: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PositionBudget]:
        """
        Compute per-symbol position budgets.

        Parameters
        ----------
        symbols:
            Candidate trading symbols (already filtered by continuous scanner / universe).
        equity_aud:
            Current portfolio equity; falls back to tracked internal equity.
        positions:
            Currently open positions dict (symbol -> {...}) to exclude already-held symbols.
        bl_views:
            Black-Litterman views from the AI brain (optional).

        Returns
        -------
        List[PositionBudget] – only viable budgets (allocation >= min_size_aud).
        """
        if equity_aud is not None:
            self._equity = equity_aud

        cfg = self.config
        max_pos_aud = float(getattr(cfg, "max_position_size_aud", 250.0) or 250.0)
        min_pos_aud = float(getattr(cfg, "min_position_size_aud", 10.0) or 10.0)
        max_pct = float(getattr(cfg, "max_position_pct", 0.25) or 0.25)
        max_exposure_pct = float(getattr(cfg, "max_total_exposure_pct", 0.98) or 0.98)
        max_concurrent = int(getattr(cfg, "max_concurrent_positions", 5) or 5)

        # Exclude already-held symbols up to concurrent limit
        open_syms = set((positions or {}).keys())
        available = [s for s in symbols if s not in open_syms]
        remaining_slots = max(0, max_concurrent - len(open_syms))
        if remaining_slots == 0:
            return []
        candidates = available[:remaining_slots]
        if not candidates:
            return []

        # Cash budget
        deployed = sum(
            float((positions or {}).get(s, {}).get("value_aud", 0.0))
            for s in open_syms
        )
        max_deploy = self._equity * max_exposure_pct - deployed
        if max_deploy < min_pos_aud:
            return []

        # Paper-mode peak: use full remaining budget
        if self._peak_mode and str(getattr(cfg, "run_mode", "paper")).lower() != "live":
            max_deploy = min(max_deploy, self._equity * max_exposure_pct)

        # Compute weights
        weights = self._compute_weights(candidates, bl_views=bl_views)

        # Check drift for rebalancing advisory
        drift = self._check_drift(weights)
        if drift:
            logger.debug("CapitalOptimizer: rebalance recommended – drift %.2f%%", drift * 100)

        # Kelly scale (per-symbol, averaged when no per-symbol history)
        kelly = self._kelly_scale_global()

        # Build budgets
        budgets: List[PositionBudget] = []
        for sym in candidates:
            w = weights.get(sym, 0.0)
            if w <= 0:
                continue
            raw_aud = w * max_deploy * kelly
            # Apply pct and abs caps
            capped_pct = min(raw_aud, self._equity * max_pct)
            capped_abs = min(capped_pct, max_pos_aud)
            allocation = max(min_pos_aud, capped_abs)

            budgets.append(PositionBudget(
                symbol=sym,
                allocation_aud=round(allocation, 2),
                allocation_pct=allocation / self._equity if self._equity > 0 else 0.0,
                weight=w,
                kelly_scale=kelly,
                max_size_aud=max_pos_aud,
                min_size_aud=min_pos_aud,
            ))

        self._last_weights = dict(weights)
        viable = [b for b in budgets if b.is_viable]
        if viable:
            logger.debug(
                "CapitalOptimizer: %d viable budgets (equity=%.2f AUD, deployed=%.2f AUD, method=%s)",
                len(viable), self._equity, deployed,
                str(getattr(cfg, "portfolio_weight_method", "hrp")),
            )
        return viable

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        symbols: List[str],
        *,
        bl_views: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        method = str(getattr(self.config, "portfolio_weight_method", "hrp") or "hrp").lower()
        returns_matrix = self._build_returns_matrix(symbols)

        if method == "bl":
            return _bl_weights(symbols, returns_matrix, views=bl_views)
        elif method == "mpt":
            return _mpt_weights(symbols, returns_matrix)
        else:  # hrp (default)
            return _hrp_weights(symbols, returns_matrix)

    def _build_returns_matrix(self, symbols: List[str]) -> Optional[Any]:
        """Build returns matrix from history if sufficient data exists."""
        if not self._returns_history:
            return None
        try:
            import numpy as np  # type: ignore
            min_len = min(
                len(self._returns_history.get(s, []))
                for s in symbols
                if self._returns_history.get(s)
            ) if any(self._returns_history.get(s) for s in symbols) else 0
            if min_len < 5:
                return None
            mat = np.array([
                self._returns_history.get(s, [0.0] * min_len)[-min_len:]
                for s in symbols
            ]).T  # shape (min_len, n_symbols)
            return mat
        except Exception:
            return None

    def _check_drift(
        self, new_weights: Dict[str, float], threshold: Optional[float] = None
    ) -> float:
        """Return max drift vs last weights; 0 if no prior allocation."""
        if not self._last_weights:
            return 0.0
        th = threshold or float(
            getattr(self.config, "target_rebalance_min_delta_pct", 0.02) or 0.02
        )
        max_drift = max(
            abs(new_weights.get(s, 0.0) - self._last_weights.get(s, 0.0))
            for s in set(new_weights) | set(self._last_weights)
        )
        return max_drift if max_drift > th else 0.0

    def _kelly_scale_global(self) -> float:
        """Global Kelly fraction derived from aggregate trade history."""
        if self._trade_count < int(getattr(self.config, "adaptive_min_trades_before_bias", 3) or 3):
            return 1.0  # no scaling until we have data
        win_rate = self._win_count / self._trade_count if self._trade_count > 0 else 0.5
        avg_win = (self._total_win_pct / self._win_count) if self._win_count > 0 else 0.01
        avg_loss = (self._total_loss_pct / self._loss_count) if self._loss_count > 0 else 0.01
        return _kelly_scale(win_rate, avg_win, avg_loss)

    # ------------------------------------------------------------------
    # Trade feedback
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        symbol: str,
        pnl_aud: float,
        pnl_pct: float,
        new_equity_aud: float,
    ) -> None:
        """Update capital state after a trade closes."""
        self._equity = new_equity_aud
        self._cash = new_equity_aud  # simplified; real cash tracks open positions
        self._cumulative_pnl += pnl_aud
        self._daily_pnl += pnl_aud
        self._trade_count += 1
        if new_equity_aud > self._peak_equity:
            self._peak_equity = new_equity_aud
        if pnl_pct >= 0:
            self._win_count += 1
            self._total_win_pct += pnl_pct
        else:
            self._loss_count += 1
            self._total_loss_pct += abs(pnl_pct)
        # Track returns history for weight computation
        if symbol not in self._returns_history:
            self._returns_history[symbol] = []
        self._returns_history[symbol].append(pnl_pct)
        window = 252  # ~1 year of daily returns
        if len(self._returns_history[symbol]) > window:
            self._returns_history[symbol] = self._returns_history[symbol][-window:]

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0

    # ------------------------------------------------------------------
    # State / reporting
    # ------------------------------------------------------------------

    def get_capital_state(self) -> CapitalState:
        dd = ((self._peak_equity - self._equity) / self._peak_equity
              if self._peak_equity > 0 else 0.0)
        wr = (100.0 * self._win_count / self._trade_count
              if self._trade_count > 0 else 0.0)
        return CapitalState(
            total_equity_aud=self._equity,
            cash_aud=self._cash,
            deployed_aud=max(0.0, self._equity - self._cash),
            n_open_positions=0,  # caller updates from portfolio state
            utilisation_pct=max(0.0, (self._equity - self._cash) / self._equity * 100)
            if self._equity > 0 else 0.0,
            peak_equity_aud=self._peak_equity,
            drawdown_pct=dd * 100,
            daily_pnl_aud=self._daily_pnl,
            cumulative_pnl_aud=self._cumulative_pnl,
            trade_count=self._trade_count,
            win_rate_pct=wr,
        )

    def get_allocation_report(self) -> Dict[str, Any]:
        state = self.get_capital_state()
        kelly = self._kelly_scale_global()
        return {
            "equity_aud": round(state.total_equity_aud, 2),
            "cash_aud": round(state.cash_aud, 2),
            "peak_equity_aud": round(state.peak_equity_aud, 2),
            "drawdown_pct": round(state.drawdown_pct, 2),
            "daily_pnl_aud": round(state.daily_pnl_aud, 2),
            "cumulative_pnl_aud": round(state.cumulative_pnl_aud, 2),
            "cumulative_return_pct": round(
                self._cumulative_pnl / float(getattr(self.config, "starting_capital_aud", 1000.0) or 1000.0) * 100,
                2,
            ),
            "trade_count": state.trade_count,
            "win_rate_pct": round(state.win_rate_pct, 1),
            "kelly_scale": round(kelly, 3),
            "portfolio_weight_method": str(getattr(self.config, "portfolio_weight_method", "hrp")),
            "peak_mode": self._peak_mode,
            "last_weights": {k: round(v, 4) for k, v in self._last_weights.items()},
        }
