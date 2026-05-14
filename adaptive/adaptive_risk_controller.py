from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AdaptiveRiskProfile:
    # Hard-capped outputs
    max_total_exposure_pct: float
    max_concurrent_signals: int
    min_signal_confidence: float

    # Execution gates
    edge_cost_gate_buffer_mult: float
    edge_cost_gate_min_edge_pct: float

    # Risk geometry (multipliers against config base values)
    stop_loss_mult: float
    take_profit_mult: float

    # Throttles
    strategy_cooldown_cycles: Dict[str, int]

    # Debug
    reason: str


class AdaptiveRiskController:
    """
    Regime + performance aware risk controller.

    This controller adjusts *soft risk knobs* continuously while respecting hard caps.
    It is designed to be cheap enough to run each cycle.
    """

    def __init__(self, *, config: Any) -> None:
        self.config = config

        # Hard caps (balanced profile)
        self.hard_max_total_exposure_pct = float(getattr(config, "max_total_exposure_pct", 0.40) or 0.40)
        self.hard_max_daily_loss_pct = float(getattr(config, "max_daily_loss_pct", 0.02) or 0.02)
        self.hard_max_drawdown_pct = float(getattr(config, "max_drawdown_pct", 0.10) or 0.10)

        # Base values (used as the neutral point)
        self.base_min_signal_confidence = float(getattr(config, "min_signal_confidence", 0.75) or 0.75)
        self.base_max_concurrent_signals = int(getattr(config, "max_concurrent_signals", 2) or 2)

        self.base_edge_buffer = float(getattr(config, "edge_cost_gate_buffer_mult", 1.25) or 1.25)
        self.base_edge_min_edge_pct = float(getattr(config, "edge_cost_gate_min_edge_pct", 0.0) or 0.0)

        self.base_stop_loss_pct = float(getattr(config, "stop_loss_pct", 0.01) or 0.01)
        self.base_take_profit_pct = float(getattr(config, "take_profit_pct", 0.03) or 0.03)

        # State
        self._pnl_ema_pct = 0.0
        self._pnl2_ema = 0.0
        self._dd_ema_pct = 0.0
        self._loss_streak = 0
        self._cooldowns: Dict[str, int] = {}

    def _ema(self, prev: float, x: float, *, alpha: float) -> float:
        a = float(alpha)
        return (1.0 - a) * float(prev) + a * float(x)

    def observe_trade_close(self, *, strategy: str, pnl_pct: float) -> None:
        # lightweight per-strategy cooldown tracking
        s = str(strategy or "unknown").strip() or "unknown"
        if float(pnl_pct) < 0:
            self._loss_streak += 1
            # cooldown increases mildly with streak
            self._cooldowns[s] = int(max(self._cooldowns.get(s, 0), min(3, 1 + self._loss_streak // 3)))
        else:
            self._loss_streak = 0
            # allow cooldown to decay
            if s in self._cooldowns:
                self._cooldowns[s] = int(max(0, self._cooldowns[s] - 1))

        # update performance EMAs (pct)
        p = float(pnl_pct)
        self._pnl_ema_pct = self._ema(self._pnl_ema_pct, p, alpha=0.12)
        self._pnl2_ema = self._ema(self._pnl2_ema, p * p, alpha=0.12)

    def decay(self) -> None:
        # per-cycle cooldown decay
        for k in list(self._cooldowns.keys()):
            v = int(self._cooldowns.get(k, 0) or 0)
            v = max(0, v - 1)
            if v <= 0:
                self._cooldowns.pop(k, None)
            else:
                self._cooldowns[k] = v

    def update_profile(
        self,
        *,
        drawdown_pct: float,
        daily_return_pct: float,
        last_regime_by_symbol: Optional[Dict[str, str]] = None,
        market_volatility: Optional[float] = None,
    ) -> AdaptiveRiskProfile:
        """
        Compute the current adaptive risk profile.
        - drawdown_pct: 0..100 scale (percent)
        - daily_return_pct: percent
        - last_regime_by_symbol: map symbol -> regime string
        - market_volatility: current annualized (or daily scaled) volatility (0.02 = 2%)
        """
        dd = float(max(0.0, drawdown_pct))
        self._dd_ema_pct = self._ema(self._dd_ema_pct, dd, alpha=0.10)

        # Performance Volatility proxy (PnL variance)
        var = max(0.0, float(self._pnl2_ema) - float(self._pnl_ema_pct) * float(self._pnl_ema_pct))
        perf_vol = float(math.sqrt(var))  # pct

        # Use provided market volatility if available, otherwise fallback to perf vol
        vol = float(market_volatility) if market_volatility is not None else perf_vol

        regimes = last_regime_by_symbol or {}
        # simple regime majority (by count)
        r_counts: Dict[str, int] = {}
        for r in regimes.values():
            rr = str(r or "unknown")
            r_counts[rr] = int(r_counts.get(rr, 0) + 1)
        majority = max(r_counts.items(), key=lambda kv: kv[1])[0] if r_counts else "unknown"

        # Start neutral
        max_exposure = float(self.hard_max_total_exposure_pct)
        max_sigs = int(self.base_max_concurrent_signals)
        min_conf = float(self.base_min_signal_confidence)
        edge_buf = float(self.base_edge_buffer)
        edge_min = float(self.base_edge_min_edge_pct)
        sl_mult = 1.0
        tp_mult = 1.0
        reason = "neutral"

        # Tighten aggressively under drawdown pressure
        if dd >= (self.hard_max_drawdown_pct * 100.0) * 0.75:
            max_exposure *= 0.70
            max_sigs = max(1, int(max_sigs * 0.6))
            min_conf = min(0.90, max(min_conf, 0.70))
            edge_buf = min(2.50, max(edge_buf, 1.75))
            sl_mult = 0.85
            tp_mult = 0.90
            reason = "drawdown_tighten"

        # High vol regime: trade less, require more edge
        # Check explicit market volatility (e.g. > 5% daily vol is high)
        is_high_vol = "high_vol" in str(majority).lower() or vol >= 0.05

        if is_high_vol:
            max_exposure *= 0.75
            max_sigs = max(1, int(max_sigs * 0.7))
            min_conf = min(0.92, max(min_conf, 0.75)) # Higher confidence required
            edge_buf = min(3.00, max(edge_buf, 2.00))
            
            # In high vol, we want WIDER stops to avoid noise out, but SMALLER size
            sl_mult = max(sl_mult, 1.5) 
            tp_mult = max(tp_mult, 1.5)
            
            reason = f"{reason}+high_vol_adapt"

        # Low vol regime: allow slightly looser entry, tighter stops
        if vol < 0.015:
            min_conf = max(0.60, min_conf * 0.95)
            sl_mult = min(sl_mult, 0.8)
            reason = f"{reason}+low_vol_adapt"

        # Trend regimes: if performance is positive, allow slightly more
        if "trend_" in str(majority).lower():
            if float(self._pnl_ema_pct) > 0.05 and float(daily_return_pct) > -0.05:
                max_exposure *= 1.05
                max_sigs = min(6, max_sigs + 1)
                edge_buf = max(1.10, edge_buf * 0.92)
                tp_mult = max(tp_mult, 1.05)
                reason = f"{reason}+trend_expand"
            else:
                # trend but not doing well -> be selective
                min_conf = min(0.92, max(min_conf, 0.70))
                edge_buf = max(edge_buf, 1.50)
                reason = f"{reason}+trend_selective"

        # Very noisy pnl -> increase selectivity
        if vol >= 1.25:
            min_conf = min(0.95, max(min_conf, 0.72))
            edge_buf = min(3.00, max(edge_buf, 1.75))
            reason = f"{reason}+noisy"

        # Hard clamps
        max_exposure = float(max(0.05, min(max_exposure, self.hard_max_total_exposure_pct)))
        max_sigs = int(max(1, min(max_sigs, 6)))
        min_conf = float(max(0.45, min(min_conf, 0.95)))
        edge_buf = float(max(1.0, min(edge_buf, 3.0)))
        edge_min = float(max(0.0, min(edge_min, 5.0)))

        # Cooldown snapshot (copy)
        cd = {str(k): int(v) for k, v in (self._cooldowns or {}).items() if int(v) > 0}

        return AdaptiveRiskProfile(
            max_total_exposure_pct=max_exposure,
            max_concurrent_signals=max_sigs,
            min_signal_confidence=min_conf,
            edge_cost_gate_buffer_mult=edge_buf,
            edge_cost_gate_min_edge_pct=edge_min,
            stop_loss_mult=float(max(0.5, min(sl_mult, 1.5))),
            take_profit_mult=float(max(0.5, min(tp_mult, 1.75))),
            strategy_cooldown_cycles=cd,
            reason=str(reason),
        )

