from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import math

from adaptive.regime import MarketRegime

logger = logging.getLogger(__name__)

# Concept drift detection — auto-reset EMA when strategy edge dies
try:
    from ml.online_learning import PageHinkleyDetector, DriftType
    _DRIFT_AVAILABLE = True
except Exception:
    _DRIFT_AVAILABLE = False


@dataclass
class _EmaStat:
    trades: int = 0
    ema_pnl_pct: float = 0.0
    ema_win_rate: float = 0.5

    def update(self, pnl_pct: float, *, alpha: float) -> None:
        self.trades += 1
        win = 1.0 if pnl_pct >= 0 else 0.0
        self.ema_pnl_pct = (1 - alpha) * self.ema_pnl_pct + alpha * float(pnl_pct)
        self.ema_win_rate = (1 - alpha) * self.ema_win_rate + alpha * float(win)


class OnlineStrategyTuner:
    """
    Online tuner that:
    - tracks realized PnL per (symbol, regime, mode)
    - returns confidence multipliers and threshold adjustments

    The goal is *continuous adaptation* without heavy ML dependencies.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.15,
        min_trades_before_bias: int = 20,
    ) -> None:
        self.alpha = float(alpha)
        # Require 20+ trades for statistical significance before biasing confidence.
        # Previously was 3, which caused severe overfitting to noise.
        self.min_trades_before_bias = max(10, int(min_trades_before_bias))

        # stats[symbol][regime][mode] -> _EmaStat
        self._stats: Dict[str, Dict[str, Dict[str, _EmaStat]]] = {}
        # Drift detectors per (symbol, regime, mode) — resets EMA when edge dies
        self._drift_detectors: Dict[str, Any] = {}
        self._drift_resets: int = 0

    def record_trade(
        self,
        *,
        symbol: str,
        regime: MarketRegime,
        mode: str,
        pnl_pct: float,
    ) -> None:
        sym = str(symbol)
        reg = str(regime.value)
        m = str(mode)
        s = self._stats.setdefault(sym, {}).setdefault(reg, {}).setdefault(m, _EmaStat())
        s.update(float(pnl_pct), alpha=self.alpha)

        # Concept drift detection: auto-reset EMA when strategy edge dies
        if _DRIFT_AVAILABLE:
            drift_key = f"{sym}|{reg}|{m}"
            if drift_key not in self._drift_detectors:
                self._drift_detectors[drift_key] = PageHinkleyDetector(
                    delta=0.005, lambda_param=50.0, alpha=0.9999
                )
            result = self._drift_detectors[drift_key].add_sample(float(pnl_pct))
            if result.drift_type == DriftType.DRIFT:
                logger.warning(
                    "Drift detected on %s/%s/%s (stat=%.2f) — resetting tuner EMA",
                    sym, reg, m, result.statistic,
                )
                s.ema_pnl_pct *= 0.3  # dampen rather than zero-out
                s.ema_win_rate = 0.5  # reset to neutral
                self._drift_detectors[drift_key].reset()
                self._drift_resets += 1

    def confidence_multiplier(
        self, *, symbol: str, regime: MarketRegime, mode: str, base: float = 1.0
    ) -> float:
        """
        Return a multiplier in ~[0.7, 1.3] based on EMA PnL and win rate.
        """
        stat = self._get(symbol=symbol, regime=regime, mode=mode)
        if stat is None or stat.trades < self.min_trades_before_bias:
            return float(base)

        # Convert EMA pnl_pct (roughly -10..+10 for most configs) to a bounded bias.
        # Positive pnl increases multiplier; negative decreases.
        pnl = float(stat.ema_pnl_pct)
        win = float(stat.ema_win_rate)

        # Small nonlinear mapping with NaN protection.
        try:
            pnl_term = math.tanh(pnl / 2.5)  # -1..1
            win_term = (win - 0.5) * 2.0  # -1..1
            raw = 1.0 + (0.18 * pnl_term) + (0.08 * win_term)
            # Tighter clamp: ±15% max (was ±30% — too aggressive for small sample sizes)
            result = float(max(0.85, min(1.15, raw)) * float(base))
            if math.isnan(result) or math.isinf(result):
                return float(base)
            return result
        except (ValueError, OverflowError):
            return float(base)

    def threshold_adjustments(
        self,
        *,
        symbol: str,
        regime: MarketRegime,
        mode: str,
    ) -> Dict[str, float]:
        """
        Return small threshold deltas for the StrategyEngine to apply.

        Convention:
        - positive delta makes condition *harder* to trigger
        - negative delta makes condition *easier* to trigger
        """
        stat = self._get(symbol=symbol, regime=regime, mode=mode)
        if stat is None or stat.trades < self.min_trades_before_bias:
            return {}

        # If EMA pnl is negative, tighten (harder entries). If positive, loosen.
        pnl = float(stat.ema_pnl_pct)
        direction = -1.0 if pnl > 0 else 1.0  # pnl>0 => loosen => negative deltas

        mag = min(1.0, abs(pnl) / 3.0)  # scale 0..1
        delta = direction * (0.04 * mag)

        # These are interpreted by StrategyEngine as fractional or absolute shifts.
        return {
            "selectivity": float(delta),
        }

    def status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for sym, by_reg in self._stats.items():
            out[sym] = {}
            for reg, by_mode in by_reg.items():
                out[sym][reg] = {}
                for mode, st in by_mode.items():
                    out[sym][reg][mode] = {
                        "trades": int(st.trades),
                        "ema_pnl_pct": float(st.ema_pnl_pct),
                        "ema_win_rate": float(st.ema_win_rate),
                    }
        return out

    def _get(self, *, symbol: str, regime: MarketRegime, mode: str) -> Optional[_EmaStat]:
        return (
            self._stats.get(str(symbol), {})
            .get(str(regime.value), {})
            .get(str(mode))
        )

