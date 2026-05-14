"""Unified Position Sizer — Kelly criterion + Conviction multiplier, one pipeline.

Previously kelly_sizing.py and conviction_sizer.py operated independently,
risking double-scaling or fighting each other. This module unifies them:

    final_size = clip(
        kelly_fraction(strategy, symbol)          # objective measured edge
        * conviction_multiplier(sources)           # subjective signal quality
        * streak_dampener(recent_trades)           # reduces size on losing streaks
        * drawdown_scaler(current_dd)              # reduces size during drawdown
        , min=MIN_PCT, max=MAX_PCT
    )

Key improvements over the originals:
1. Single call-site — callers import PositionSizer and call .size(), done.
2. Losing-streak dampener — after N consecutive losses the multiplier drops to
   1x base until a win is recorded. Prevents "averaging down" spirals.
3. Drawdown scaler — linearly reduces size as current drawdown grows toward
   max_dd_threshold. At max_dd, size floors to MIN_PCT.
4. Conviction still scaled within [0.5x, 1.0x] of Kelly (not above) so
   conviction can sharpen timing but never override the Kelly math.
5. Full audit trail returned in SizeResult for logging and debugging.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
DEFAULT_KELLY_FRACTION   = 0.25   # fractional Kelly safety factor
DEFAULT_MIN_PCT          = 0.005  # 0.5% minimum position
DEFAULT_MAX_PCT          = 0.15   # 15% maximum position
DEFAULT_LOOKBACK         = 200    # trades to track for Kelly estimation
DEFAULT_MIN_TRADES       = 20     # minimum trades before Kelly activates
STREAK_DAMPEN_AFTER      = 4      # consecutive losses before dampening
STREAK_DAMPEN_FACTOR     = 0.5    # multiplier applied when streak triggers
MAX_DD_THRESHOLD         = 0.15   # 15% drawdown => floor to MIN_PCT


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class SizeResult:
    """Full audit trail for a position sizing decision."""
    final_pct:          float   # size to actually use
    kelly_pct:          float   # raw Kelly fraction
    kelly_confidence:   float   # 0-1 based on sample size
    conviction_score:   float   # 0-1 from signal sources
    conviction_mult:    float   # 0.5-1.0 applied to Kelly
    streak_dampened:    bool
    drawdown_scaled:    bool
    current_drawdown:   float
    n_trades:           int
    win_rate:           float
    payoff_ratio:       float
    sources:            Dict[str, float]  # per-source conviction breakdown


# -----------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------

class PositionSizer:
    """
    Unified position sizer: Kelly fraction * Conviction * Streak * Drawdown.

    Usage:
        sizer = PositionSizer()

        # After each trade closes:
        sizer.record_trade(
            strategy="momentum", symbol="BTC/USDT", pnl=120.0
        )

        # Before each new entry:
        result = sizer.size(
            strategy="momentum", symbol="BTC/USDT",
            action="BUY", strategy_type="momentum",
            advisory=advisory_dict, mtf_bias=mtf_dict,
            regime="TRENDING_UP",
        )
        order_size = result.final_pct * portfolio_value
    """

    def __init__(
        self,
        kelly_fraction:    float = DEFAULT_KELLY_FRACTION,
        min_pct:           float = DEFAULT_MIN_PCT,
        max_pct:           float = DEFAULT_MAX_PCT,
        lookback:          int   = DEFAULT_LOOKBACK,
        min_trades:        int   = DEFAULT_MIN_TRADES,
        max_dd_threshold:  float = MAX_DD_THRESHOLD,
    ) -> None:
        self._kf             = kelly_fraction
        self._min            = min_pct
        self._max            = max_pct
        self._lookback       = lookback
        self._min_trades     = min_trades
        self._max_dd         = max_dd_threshold

        # Per-strategy-symbol trade history
        self._trades: Dict[str, deque] = {}

        # Portfolio-level state for drawdown and streak
        self._portfolio_pnl:     List[float] = []
        self._peak_pnl:          float = 0.0
        self._consecutive_losses: int = 0

    # ------------------------------------------------------------------
    # Record outcomes
    # ------------------------------------------------------------------

    def record_trade(
        self,
        strategy: str,
        symbol:   str,
        pnl:      float,
    ) -> None:
        """Record a closed trade for Kelly estimation and streak/DD tracking."""
        key = f"{strategy}:{symbol}"
        if key not in self._trades:
            self._trades[key] = deque(maxlen=self._lookback)
        self._trades[key].append(pnl)

        # Portfolio-level tracking
        self._portfolio_pnl.append(pnl)
        total = sum(self._portfolio_pnl)
        if total > self._peak_pnl:
            self._peak_pnl = total

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    # ------------------------------------------------------------------
    # Main sizing call
    # ------------------------------------------------------------------

    def size(
        self,
        strategy:      str,
        symbol:        str,
        action:        str               = "BUY",
        strategy_type: str               = "momentum",
        advisory:      Optional[Dict[str, Any]] = None,
        mtf_bias:      Optional[Dict[str, str]] = None,
        regime:        str               = "NORMAL",
    ) -> SizeResult:
        """Compute unified position size."""
        advisory  = advisory  or {}
        mtf_bias  = mtf_bias  or {}

        # ---- 1. Kelly fraction ------------------------------------------
        kelly_pct, kelly_conf, win_rate, payoff = self._kelly(
            strategy, symbol
        )

        # ---- 2. Conviction multiplier (0.5 – 1.0 of Kelly) -------------
        conv_score, sources = self._conviction(
            symbol, action, strategy_type, advisory, mtf_bias, regime
        )
        # Map conviction [0,1] -> multiplier [0.5, 1.0]
        conv_mult = 0.5 + 0.5 * conv_score

        # ---- 3. Streak dampener -----------------------------------------
        streak_dampened = self._consecutive_losses >= STREAK_DAMPEN_AFTER
        streak_mult = STREAK_DAMPEN_FACTOR if streak_dampened else 1.0

        # ---- 4. Drawdown scaler -----------------------------------------
        current_dd = self._current_drawdown()
        if current_dd >= self._max_dd:
            dd_mult = 0.0   # at max_dd, floor to min
            dd_scaled = True
        elif current_dd > 0:
            dd_mult = 1.0 - (current_dd / self._max_dd)
            dd_scaled = True
        else:
            dd_mult = 1.0
            dd_scaled = False

        # ---- 5. Combine -------------------------------------------------
        raw = kelly_pct * conv_mult * streak_mult * dd_mult
        final = float(max(self._min, min(self._max, raw)))

        logger.debug(
            "PositionSizer: %s/%s kelly=%.4f conv_mult=%.3f streak=%s "
            "dd=%.3f final=%.4f",
            strategy, symbol, kelly_pct, conv_mult,
            streak_dampened, current_dd, final,
        )

        return SizeResult(
            final_pct        = final,
            kelly_pct        = kelly_pct,
            kelly_confidence = kelly_conf,
            conviction_score = conv_score,
            conviction_mult  = conv_mult,
            streak_dampened  = streak_dampened,
            drawdown_scaled  = dd_scaled,
            current_drawdown = current_dd,
            n_trades         = len(self._trades.get(f"{strategy}:{symbol}", [])),
            win_rate         = win_rate,
            payoff_ratio     = payoff,
            sources          = sources,
        )

    # ------------------------------------------------------------------
    # Kelly estimation
    # ------------------------------------------------------------------

    def _kelly(
        self, strategy: str, symbol: str
    ) -> tuple:
        """Returns (kelly_pct, confidence, win_rate, payoff_ratio)."""
        key    = f"{strategy}:{symbol}"
        trades = list(self._trades.get(key, []))

        if len(trades) < self._min_trades:
            conf = len(trades) / max(self._min_trades, 1)
            return self._min, conf, 0.5, 1.0

        wins   = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]

        win_rate    = len(wins)  / len(trades)
        avg_win     = sum(wins)  / len(wins)   if wins   else 0.0
        avg_loss    = abs(sum(losses) / len(losses)) if losses else 1e-9
        payoff      = avg_win / max(avg_loss, 1e-9)

        raw_kelly   = max(0.0, win_rate - (1.0 - win_rate) / max(payoff, 1e-9))
        kelly_pct   = min(self._max, raw_kelly * self._kf)
        confidence  = min(1.0, len(trades) / (self._min_trades * 3))

        return kelly_pct, confidence, win_rate, payoff

    # ------------------------------------------------------------------
    # Conviction scoring (ported from conviction_sizer.py, simplified)
    # ------------------------------------------------------------------

    def _conviction(
        self,
        symbol:        str,
        action:        str,
        strategy_type: str,
        advisory:      Dict[str, Any],
        mtf_bias:      Dict[str, str],
        regime:        str,
    ) -> tuple:
        """Returns (conviction_score 0-1, sources dict)."""
        sources: Dict[str, float] = {}

        # Scanner confidence
        scanner = advisory.get("strategy_scanner", {})
        if isinstance(scanner, dict):
            opps = scanner.get("top_opportunities", [])
            best_sym = ""
            if opps:
                first = opps[0]
                best_sym = (
                    getattr(first, "symbol", "")
                    if hasattr(first, "symbol")
                    else first.get("symbol", "") if isinstance(first, dict) else ""
                )
            sources["scanner"] = 1.0 if best_sym == symbol else 0.0
        else:
            sources["scanner"] = 0.0

        # Evolver fitness
        evolver = advisory.get("strategy_evolver", {})
        if isinstance(evolver, dict):
            bc = float(evolver.get("best_composite", 0) or 0)
            bs = str(evolver.get("best_symbol", "") or "")
            sources["evolver"] = min(1.0, bc) if bs == symbol and bc > 0.3 else 0.0
        else:
            sources["evolver"] = 0.0

        # Regime alignment
        try:
            from core.strategy_evolver import _REGIME_STRATEGY_AFFINITY
            rl = regime.lower().replace("_", "")
            rk = (
                "trending" if "trend" in rl
                else "volatile" if "vol" in rl or "crisis" in rl
                else "ranging"
            )
            sources["regime"] = 1.0 if strategy_type in _REGIME_STRATEGY_AFFINITY.get(rk, []) else 0.0
        except ImportError:
            sources["regime"] = 0.5

        # MTF agreement
        mtf_score = sum(
            0.5 if bias == action else (0.1 if bias == "NEUTRAL" else 0.0)
            for bias in mtf_bias.values()
        )
        sources["mtf"] = min(1.0, mtf_score)

        # Edge monitor
        edge = advisory.get("edge_monitor", {})
        sources["edge"] = (
            1.0 if isinstance(edge, dict) and float(edge.get("edge_score", 0.5) or 0.5) > 0.4 else 0.5
        )

        # Health score
        health = advisory.get("health_score", {})
        sources["health"] = (
            min(1.0, float(health.get("score", 50) or 50) / 100.0)
            if isinstance(health, dict) else 0.5
        )

        score = sum(sources.values()) / max(1, len(sources))
        return score, sources

    # ------------------------------------------------------------------
    # Drawdown helper
    # ------------------------------------------------------------------

    def _current_drawdown(self) -> float:
        """Current portfolio drawdown as fraction of peak equity."""
        if not self._portfolio_pnl or self._peak_pnl <= 0:
            return 0.0
        current = sum(self._portfolio_pnl)
        dd = (self._peak_pnl - current) / self._peak_pnl
        return max(0.0, dd)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "tracked_pairs":       len(self._trades),
            "consecutive_losses":  self._consecutive_losses,
            "streak_dampened":     self._consecutive_losses >= STREAK_DAMPEN_AFTER,
            "current_drawdown":    round(self._current_drawdown(), 6),
            "peak_pnl":            round(self._peak_pnl, 4),
            "kelly_fraction":      self._kf,
            "max_dd_threshold":    self._max_dd,
        }
