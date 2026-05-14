"""
EdgeMonitor — Rolling live-vs-baseline edge erosion detector.

Tracks three metrics across the last N fills and computes a composite
edge_score (0.0 = no edge, 1.0 = full edge).  When edge_score drops
below a threshold the `degraded` flag is set and plain-language reasons
are populated.

Metrics tracked:
  slippage_bps  — avg realised slippage vs baseline assumption
  win_rate      — fraction of profitable trades in the window
  recent_sharpe — annualised Sharpe of P&L series in the window

Edge score formula:
  slippage_score = clamp(1 - max(0, actual/baseline - 1), 0, 1)
  win_rate_score = clamp(actual_win_rate / baseline_win_rate, 0, 1)
  sharpe_score   = clamp((recent_sharpe + 1) / 2, 0, 1)   # maps [-1,1] → [0,1]
  edge_score     = mean(slippage_score, win_rate_score, sharpe_score)
  degraded       = edge_score < degraded_threshold AND n_fills >= min_fills
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class _FillRecord:
    slippage_bps: float
    won: bool
    pnl: float    # net P&L in AUD (used for Sharpe; 0.0 if unknown)
    ts: float


@dataclass
class EdgeSnapshot:
    slippage_bps: float
    win_rate: float
    recent_sharpe: float
    slippage_score: float
    win_rate_score: float
    sharpe_score: float
    edge_score: float
    degraded: bool
    n_fills: int
    reasons: List[str]


class EdgeMonitor:
    """
    Rolling edge erosion detector.  Call ``record_fill`` on every fill,
    then call ``evaluate`` each cycle for the latest snapshot.
    """

    def __init__(
        self,
        baseline_slippage_bps: float = 8.0,
        baseline_win_rate: float = 0.50,
        min_sharpe: float = 0.0,
        lookback_trades: int = 20,
        degraded_threshold: float = 0.35,
        min_fills: int = 5,
    ) -> None:
        self.baseline_slippage_bps = max(0.1, float(baseline_slippage_bps))
        self.baseline_win_rate     = _clamp(float(baseline_win_rate), 0.01, 0.99)
        self.min_sharpe            = float(min_sharpe)
        self.lookback_trades       = max(5, int(lookback_trades))
        self.degraded_threshold    = _clamp(float(degraded_threshold))
        self.min_fills             = max(1, int(min_fills))

        self._fills: Deque[_FillRecord] = deque(maxlen=self.lookback_trades)
        self._last_snapshot: Optional[EdgeSnapshot] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_fill(
        self,
        slippage_bps: float,
        won: bool,
        pnl: float = 0.0,
    ) -> None:
        """
        Record a single fill outcome.

        Parameters
        ----------
        slippage_bps : realised slippage in basis points (0 = no slippage)
        won          : True if the trade was profitable
        pnl          : net P&L in account currency (for Sharpe calculation)
        """
        self._fills.append(_FillRecord(
            slippage_bps=float(slippage_bps),
            won=bool(won),
            pnl=float(pnl),
            ts=time.time(),
        ))

    def evaluate(self) -> EdgeSnapshot:
        """Compute edge snapshot from recent fills."""
        fills = list(self._fills)
        n = len(fills)

        if n < self.min_fills:
            # Not enough data — return neutral (no degradation declared)
            snap = EdgeSnapshot(
                slippage_bps=0.0,
                win_rate=self.baseline_win_rate,
                recent_sharpe=self.min_sharpe,
                slippage_score=1.0,
                win_rate_score=1.0,
                sharpe_score=0.5,
                edge_score=0.83,
                degraded=False,
                n_fills=n,
                reasons=[],
            )
            self._last_snapshot = snap
            return snap

        # ── Compute raw metrics ───────────────────────────────────────────────
        avg_slip   = sum(f.slippage_bps for f in fills) / n
        win_rate   = sum(1 for f in fills if f.won) / n
        pnls       = [f.pnl for f in fills]
        sharpe     = self._rolling_sharpe(pnls)

        # ── Score each metric ─────────────────────────────────────────────────
        slip_ratio      = avg_slip / self.baseline_slippage_bps
        slippage_score  = _clamp(1.0 - max(0.0, slip_ratio - 1.0))

        win_rate_score  = _clamp(win_rate / self.baseline_win_rate)

        # Map sharpe from ~[-1.5, 1.5] → [0, 1] with midpoint at 0
        sharpe_score    = _clamp((sharpe + 1.0) / 2.0)

        edge_score = (slippage_score + win_rate_score + sharpe_score) / 3.0

        # ── Build reasons list ────────────────────────────────────────────────
        reasons: List[str] = []
        if slippage_score < 0.5:
            reasons.append(
                f"slippage {avg_slip:.1f}bps vs baseline {self.baseline_slippage_bps:.1f}bps"
            )
        if win_rate_score < 0.6:
            reasons.append(
                f"win_rate {win_rate*100:.0f}% vs baseline {self.baseline_win_rate*100:.0f}%"
            )
        if sharpe_score < 0.3:
            reasons.append(f"recent_sharpe {sharpe:.2f} below floor {self.min_sharpe:.2f}")

        degraded = edge_score < self.degraded_threshold

        snap = EdgeSnapshot(
            slippage_bps=round(avg_slip, 2),
            win_rate=round(win_rate, 3),
            recent_sharpe=round(sharpe, 3),
            slippage_score=round(slippage_score, 3),
            win_rate_score=round(win_rate_score, 3),
            sharpe_score=round(sharpe_score, 3),
            edge_score=round(edge_score, 3),
            degraded=degraded,
            n_fills=n,
            reasons=reasons,
        )
        self._last_snapshot = snap
        return snap

    def snapshot(self) -> Dict[str, Any]:
        if self._last_snapshot is None:
            self.evaluate()
        s = self._last_snapshot
        return {
            "edge_score":    s.edge_score    if s else 1.0,
            "degraded":      s.degraded      if s else False,
            "slippage_bps":  s.slippage_bps  if s else 0.0,
            "win_rate":      s.win_rate      if s else self.baseline_win_rate,
            "recent_sharpe": s.recent_sharpe if s else 0.0,
            "n_fills":       s.n_fills       if s else 0,
            "reasons":       s.reasons       if s else [],
        }

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rolling_sharpe(pnls: List[float]) -> float:
        """Annualised Sharpe from a list of P&L values (conservative 1-trade-per-day)."""
        if len(pnls) < 2:
            return 0.0
        n = len(pnls)
        mean_p = sum(pnls) / n
        variance = sum((p - mean_p) ** 2 for p in pnls) / (n - 1)
        std_p = math.sqrt(variance) if variance > 1e-12 else 1e-6
        daily_sharpe = mean_p / std_p
        return daily_sharpe * math.sqrt(252)
