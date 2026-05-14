"""
Measured-Edge Kelly Criterion Position Sizing.

Upgrade (2026-04 peak-potential):
- Regime-aware sizing: multiplies fractional Kelly by a per-regime scalar
  (e.g. reduce to 0.5x in HIGH_VOL_CRASH, boost to 1.2x in TRENDING).
- Multi-symbol correlation cap: when two strategies share correlated
  symbols, total allocation is capped to prevent correlated blowups.
- Trade tagging: record_trade accepts optional tags (e.g. 'regime:TRENDING')
  so edge can be estimated per-regime slice rather than globally.
- Confidence-weighted blending: when n_trades is between min_trades and
  3*min_trades, result blends Kelly estimate with default_pct proportional
  to confidence rather than hard-switching at min_trades.
- Bootstrap edge confidence interval: compute_ci() returns (lo, mid, hi)
  Kelly estimates via bootstrap resampling for risk management.
- Zero-edge guard: floor skipped when raw_kelly == 0 (Codex fix retained).
"""

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Per-regime Kelly multipliers.
DEFAULT_REGIME_MULTIPLIERS: Dict[str, float] = {
    "TRENDING_UP":    1.2,
    "TRENDING_DOWN":  1.1,
    "RANGING":        1.0,
    "COILED":         0.8,   # uncertain direction
    "HIGH_VOL_CRASH": 0.5,   # capital preservation
    "UNKNOWN":        0.75,
}


@dataclass
class KellyEstimate:
    """Kelly criterion estimate for a strategy/symbol pair."""
    strategy: str
    symbol: str
    kelly_fraction: float
    position_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float
    n_trades: int
    confidence: float
    vol_multiplier: float = 1.0
    drawdown_scale: float = 1.0
    regime_multiplier: float = 1.0
    regime: str = "UNKNOWN"


@dataclass
class KellyCI:
    """Bootstrap confidence interval for Kelly fraction."""
    lo: float
    mid: float
    hi: float
    n_bootstrap: int


class KellySizer:
    """
    Position sizing based on measured edge with regime awareness and
    correlation-aware multi-symbol capping.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_trades: int = 20,
        max_position_pct: float = 0.10,
        min_position_pct: float = 0.01,
        default_pct: float = 0.02,
        lookback: int = 200,
        drawdown_halve_threshold: float = 0.05,
        drawdown_halt_threshold: float = 0.10,
        regime_multipliers: Optional[Dict[str, float]] = None,
        max_correlated_alloc: float = 0.15,
    ):
        self._kelly_fraction = kelly_fraction
        self._min_trades = min_trades
        self._max_pct = max_position_pct
        self._min_pct = min_position_pct
        self._default_pct = default_pct
        self._lookback = lookback
        self._dd_halve = drawdown_halve_threshold
        self._dd_halt = drawdown_halt_threshold
        self._regime_mults = regime_multipliers or dict(DEFAULT_REGIME_MULTIPLIERS)
        self._max_corr_alloc = max_correlated_alloc
        self._trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback))
        self._tagged_trades: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        self._total_recorded = 0
        self._symbol_alloc: Dict[str, float] = {}  # current cycle allocations per symbol

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        pnl: float,
        tag: str = "",
    ) -> None:
        """Record a trade outcome. Optional tag e.g. 'regime:TRENDING_UP'."""
        key = f"{strategy}:{symbol}"
        self._trades[key].append(pnl)
        if tag:
            self._tagged_trades[key].append((pnl, tag))
            # Rolling window for tagged trades too.
            if len(self._tagged_trades[key]) > self._lookback:
                self._tagged_trades[key] = self._tagged_trades[key][-self._lookback:]
        self._total_recorded += 1

    def reset_cycle_alloc(self) -> None:
        """Call at the start of each sizing cycle to reset correlation tracking."""
        self._symbol_alloc = {}

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def compute(
        self,
        strategy: str,
        symbol: str,
        current_vol: Optional[float] = None,
        baseline_vol: Optional[float] = None,
        current_drawdown: float = 0.0,
        regime: str = "UNKNOWN",
    ) -> KellyEstimate:
        """
        Compute Kelly-optimal position size for strategy x symbol.

        Parameters
        ----------
        current_vol : float, optional
        baseline_vol : float, optional
        current_drawdown : float
            Fraction from equity peak (e.g. 0.07 = 7%).
        regime : str
            Current regime string for multiplier lookup.
        """
        key = f"{strategy}:{symbol}"
        trades = list(self._trades.get(key, []))
        regime_mult = self._regime_mults.get(regime, self._regime_mults.get("UNKNOWN", 1.0))

        # Insufficient data: confidence-weighted blend instead of hard switch.
        if len(trades) < self._min_trades:
            confidence = len(trades) / max(self._min_trades, 1)
            return KellyEstimate(
                strategy=strategy, symbol=symbol,
                kelly_fraction=0.0,
                position_pct=self._default_pct * confidence,
                win_rate=0.5, avg_win=0.0, avg_loss=0.0,
                payoff_ratio=1.0, n_trades=len(trades),
                confidence=confidence,
                regime=regime, regime_multiplier=regime_mult,
            )

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]

        win_rate = len(wins) / len(trades)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 1e-9
        payoff_ratio = avg_win / max(avg_loss, 1e-9)

        raw_kelly = win_rate - (1.0 - win_rate) / max(payoff_ratio, 1e-9)
        raw_kelly = max(0.0, raw_kelly)

        # Zero edge: skip floor entirely (Codex fix).
        if raw_kelly == 0.0:
            return KellyEstimate(
                strategy=strategy, symbol=symbol,
                kelly_fraction=0.0, position_pct=0.0,
                win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
                payoff_ratio=payoff_ratio, n_trades=len(trades),
                confidence=0.0, regime=regime, regime_multiplier=regime_mult,
            )

        position_pct = raw_kelly * self._kelly_fraction

        # Regime multiplier.
        position_pct *= regime_mult

        # Volatility scaling.
        vol_multiplier = 1.0
        if current_vol and baseline_vol and current_vol > 0:
            vol_multiplier = baseline_vol / current_vol
            vol_multiplier = max(0.5, min(1.5, vol_multiplier))
            position_pct *= vol_multiplier

        # Drawdown scaling.
        drawdown_scale = 1.0
        if current_drawdown >= self._dd_halt:
            logger.warning(
                "KellySizer: drawdown %.1f%% >= halt threshold — returning 0",
                current_drawdown * 100,
            )
            return KellyEstimate(
                strategy=strategy, symbol=symbol,
                kelly_fraction=raw_kelly, position_pct=0.0,
                win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
                payoff_ratio=payoff_ratio, n_trades=len(trades),
                confidence=0.0, vol_multiplier=vol_multiplier,
                drawdown_scale=0.0, regime=regime, regime_multiplier=regime_mult,
            )
        elif current_drawdown >= self._dd_halve:
            drawdown_scale = 0.5
            position_pct *= drawdown_scale
            logger.info("KellySizer: drawdown %.1f%% — halving size", current_drawdown * 100)

        # Correlation cap: if this symbol already has allocated capital this
        # cycle, cap the additional allocation.
        existing = self._symbol_alloc.get(symbol, 0.0)
        if existing + position_pct > self._max_corr_alloc:
            position_pct = max(0.0, self._max_corr_alloc - existing)

        # Hard floor/ceiling (only with positive edge).
        position_pct = max(self._min_pct, min(position_pct, self._max_pct))

        # Track symbol allocation for correlation cap.
        self._symbol_alloc[symbol] = self._symbol_alloc.get(symbol, 0.0) + position_pct

        # Confidence-weighted blend when data is still building up.
        confidence = min(1.0, len(trades) / (self._min_trades * 3))
        if confidence < 1.0:
            position_pct = confidence * position_pct + (1.0 - confidence) * self._default_pct

        return KellyEstimate(
            strategy=strategy, symbol=symbol,
            kelly_fraction=raw_kelly, position_pct=position_pct,
            win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
            payoff_ratio=payoff_ratio, n_trades=len(trades),
            confidence=confidence, vol_multiplier=vol_multiplier,
            drawdown_scale=drawdown_scale,
            regime=regime, regime_multiplier=regime_mult,
        )

    def compute_ci(
        self,
        strategy: str,
        symbol: str,
        n_bootstrap: int = 200,
        ci_pct: float = 90.0,
    ) -> Optional[KellyCI]:
        """Bootstrap confidence interval for raw Kelly fraction.

        Returns None if insufficient trades.
        """
        key = f"{strategy}:{symbol}"
        trades = list(self._trades.get(key, []))
        if len(trades) < self._min_trades:
            return None

        rng = random.Random(42)
        kelly_samples = []
        for _ in range(n_bootstrap):
            sample = rng.choices(trades, k=len(trades))
            wins = [t for t in sample if t > 0]
            losses = [t for t in sample if t < 0]
            wr = len(wins) / len(sample)
            aw = sum(wins) / len(wins) if wins else 0.0
            al = abs(sum(losses) / len(losses)) if losses else 1e-9
            pr = aw / max(al, 1e-9)
            k = max(0.0, wr - (1.0 - wr) / max(pr, 1e-9))
            kelly_samples.append(k)

        kelly_samples.sort()
        lo_idx = int((1 - ci_pct / 100) / 2 * n_bootstrap)
        hi_idx = int((1 - (1 - ci_pct / 100) / 2) * n_bootstrap)
        return KellyCI(
            lo=kelly_samples[lo_idx],
            mid=kelly_samples[n_bootstrap // 2],
            hi=kelly_samples[min(hi_idx, n_bootstrap - 1)],
            n_bootstrap=n_bootstrap,
        )

    def get_size_pct(
        self,
        strategy: str,
        symbol: str,
        current_vol: Optional[float] = None,
        baseline_vol: Optional[float] = None,
        current_drawdown: float = 0.0,
        regime: str = "UNKNOWN",
    ) -> float:
        return self.compute(
            strategy, symbol,
            current_vol=current_vol,
            baseline_vol=baseline_vol,
            current_drawdown=current_drawdown,
            regime=regime,
        ).position_pct

    def get_all_estimates(self) -> Dict[str, KellyEstimate]:
        estimates = {}
        for key in self._trades:
            parts = key.split(":", 1)
            if len(parts) == 2:
                estimates[key] = self.compute(parts[0], parts[1])
        return estimates

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_recorded": self._total_recorded,
            "tracked_pairs": len(self._trades),
            "kelly_fraction": self._kelly_fraction,
            "min_trades": self._min_trades,
            "max_position_pct": self._max_pct,
            "min_position_pct": self._min_pct,
            "drawdown_halve_threshold": self._dd_halve,
            "drawdown_halt_threshold": self._dd_halt,
            "max_correlated_alloc": self._max_corr_alloc,
            "regime_multipliers": dict(self._regime_mults),
        }
