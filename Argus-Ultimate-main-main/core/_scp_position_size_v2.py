"""
core/_scp_position_size_v2.py

Upgrade of _scp_position_size.py:
  • ATR input flows into PositionSizer (ATR-vol targeting)
  • Fractional Kelly overlay on top of existing SCP output
  • CapitalTier heat gate replaces the flat max_pos_pct cap

Drop-in replacement — same monkey-patch interface as v1:

    from core._scp_position_size_v2 import _compute_position_size
    from core._scp_position_size_v2 import _after_fill_hook, _after_close_hook
    UnifiedTradingSystem._compute_position_size = _compute_position_size
    UnifiedTradingSystem._scp_after_fill        = _after_fill_hook
    UnifiedTradingSystem._scp_after_close       = _after_close_hook
"""
from __future__ import annotations

import logging

from core.capital_tier import classify_tier
from core.position_sizing import PositionSizer
from core.tier_config_extension import get_tier_cfg

logger = logging.getLogger(__name__)

_sizer = PositionSizer()   # singleton; aud_to_usd updated per call


# ─────────────────────────────────────────────────────────────────────
# Lifecycle hooks  (unchanged interface from v1)
# ─────────────────────────────────────────────────────────────────────

def _after_fill_hook(self) -> None:
    _scp = _get_scp(self)
    if _scp is not None:
        try:
            _scp.on_position_opened()
        except Exception as exc:
            logger.debug("_after_fill_hook v2: %s", exc)


def _after_close_hook(self, strategy: str, symbol: str, pnl: float) -> None:
    _scp = _get_scp(self)
    if _scp is not None:
        try:
            _scp.on_trade_closed(strategy, symbol, pnl)
        except Exception as exc:
            logger.debug("_after_close_hook v2: %s", exc)
    else:
        _ks = getattr(self.component_registry, "kelly_sizer", None) if self.component_registry else None
        if _ks is not None:
            try:
                _ks.record_trade(strategy, symbol, pnl)
            except Exception as exc:
                logger.debug("_after_close_hook v2 kelly fallback: %s", exc)


def _get_scp(self):
    if self.component_registry is None:
        return None
    return getattr(self.component_registry, "small_capital_pipeline", None)


# ─────────────────────────────────────────────────────────────────────
# Main sizing function
# ─────────────────────────────────────────────────────────────────────

def _compute_position_size(self, sig_fields: dict, ctx: dict) -> tuple:
    """
    ATR + Fractional Kelly + CapitalTier aware position sizing.

    New inputs vs v1:
      ctx["atr"]           – ATR / mid_price (float, optional)
      ctx["open_heat"]     – current portfolio open risk fraction (float, optional)
      ctx["aud_to_usd"]    – FX rate (float, optional, default 0.65)

    Falls back gracefully if new ctx keys are absent.
    """
    symbol          = sig_fields["symbol"]
    action          = sig_fields["action"]
    confidence      = sig_fields["confidence"]
    strength        = sig_fields["strength"]
    source_strategy = sig_fields["source_strategy"]

    regime          = ctx["regime"]
    regime_pos_mult = ctx["regime_pos_mult"]
    session_mult    = ctx["session_mult"]
    macro_event_imminent = ctx["macro_event_imminent"]
    aud_to_usd      = float(ctx.get("aud_to_usd") or 0.65)
    atr             = float(ctx.get("atr") or 0.0)
    open_heat       = float(ctx.get("open_heat") or 0.0)

    equity_aud      = max(float(getattr(self, "portfolio_value_aud", 1.0) or 1.0), 1.0)
    tier            = classify_tier(equity_aud)
    tcfg            = get_tier_cfg(tier)
    max_pos_pct     = float(tcfg["max_position_pct"])
    min_notional    = float(tcfg["min_slice_usd"])
    max_fee_bps     = float(tcfg["max_fee_bps"])

    # ── Fetch strategy stats ──────────────────────────────────────────
    _stats    = self._get_strategy_trade_stats(source_strategy)
    win_rate  = float(_stats.get("win_rate",  0.5) or 0.5)
    avg_win   = float(_stats.get("avg_win",   0.0) or 0.0)
    avg_loss  = float(_stats.get("avg_loss",  0.0) or 0.0)

    # ── ATR fallback from indicator cache ────────────────────────────
    if atr <= 0:
        try:
            current_vol = self._get_current_vol(symbol)
            atr = float(current_vol or 0.0)
        except Exception:
            atr = 0.0

    # ── PositionSizer: ATR-vol + fractional Kelly + heat gate ─────────
    _sizer.aud_to_usd = aud_to_usd
    size_result = _sizer.size(
        equity_aud        = equity_aud,
        atr               = atr,
        win_rate          = win_rate,
        avg_win           = avg_win,
        avg_loss          = avg_loss,
        open_heat         = open_heat,
        min_notional_usd  = min_notional,
    )

    if size_result.blocked:
        logger.info(
            "_compute_position_size v2: BLOCKED %s %s — %s",
            action, symbol, size_result.block_reason,
        )
        return (0.0, f"BLOCKED:{size_result.block_reason}")

    size_pct       = size_result.size_pct
    _sizing_method = size_result.method

    # ── SmallCapitalPipeline override (if registered) ─────────────────
    _scp = _get_scp(self)
    if _scp is not None:
        try:
            _current_vol  = self._get_current_vol(symbol) or None
            _baseline_vol = getattr(self, "_baseline_vol", None) or _current_vol
            _dollar_size  = _scp.get_position_size(
                strategy     = source_strategy,
                symbol       = symbol,
                win_rate     = win_rate,
                avg_win      = avg_win,
                avg_loss     = avg_loss,
                current_vol  = _current_vol,
                baseline_vol = _baseline_vol,
            )
            if _dollar_size > 0:
                _scp_pct = (_dollar_size / max(aud_to_usd, 1e-6)) / equity_aud
                # blend: favour SCP when it gives a tighter size (conservative)
                size_pct       = min(size_pct, _scp_pct)
                _sizing_method += f"+scp_blend({_scp_pct:.4f})"
        except Exception as exc:
            logger.debug("_compute_position_size v2: SCP blend failed: %s", exc)

    size_pct = min(size_pct, max_pos_pct)

    # ── All downstream multipliers (identical to v1) ──────────────────

    current_vol = self._get_current_vol(symbol)
    if current_vol > 0:
        size_pct       = self._vol_adjusted_size(size_pct, current_vol)
        _sizing_method += "+vol_adj"

    sig_quality = self._get_signal_quality()
    if sig_quality is not None:
        sq_rec = sig_quality.get("recommendation", "moderate")
        if sq_rec == "conflicted":
            size_pct *= 0.5; _sizing_method += "+conflict_discount"
        elif sq_rec == "weak":
            size_pct *= 0.7; _sizing_method += "+weak_discount"

    size_pct *= regime_pos_mult
    _sizing_method += f"+regime({regime})*{regime_pos_mult:.2f}"

    size_pct *= session_mult
    if session_mult != 1.0:
        _sizing_method += f"+session*{session_mult:.2f}"

    if macro_event_imminent and action == "SELL":
        size_pct *= 0.7; _sizing_method += "+macro_reduce_30pct"

    try:
        _peak_cap = float(self.peak_equity_aud)
        _curr_cap = float(self.portfolio_value_aud)
        if _peak_cap > 0 and _curr_cap < _peak_cap:
            _dd_ratio = (_peak_cap - _curr_cap) / _peak_cap
            _dd_mult  = max(0.25, 1.0 - _dd_ratio * 2.0)
            size_pct *= _dd_mult
            _sizing_method += f"+dd_adj({_dd_ratio:.3f})*{_dd_mult:.2f}"
    except Exception:
        pass

    try:
        if action == "BUY" and self.positions:
            _corr_reduction = 1.0
            _new_base = symbol.split("/")[0] if "/" in symbol else symbol
            for _pos_sym, _pos_data in (self.positions or {}).items():
                if _pos_data is None or float((_pos_data or {}).get("quantity", 0) or 0) <= 0:
                    continue
                _pos_base = _pos_sym.split("/")[0] if "/" in _pos_sym else _pos_sym
                if _pos_base == _new_base:
                    continue
                _corr_val = 0.0
                _cr = getattr(self, "component_registry", None)
                if _cr and getattr(_cr, "correlation_monitor", None):
                    try:
                        _corr_val = abs(float(getattr(_cr.correlation_monitor, "_last_avg_corr", 0.0) or 0.0))
                    except Exception:
                        pass
                if _corr_val == 0.0:
                    if {_new_base, _pos_sym.split("/")[0] if "/" in _pos_sym else _pos_sym} == {"BTC", "ETH"}:
                        _corr_val = 0.85
                _pos_side = str((_pos_data or {}).get("side", "BUY")).upper()
                if _corr_val > 0.85 and _pos_side == action:
                    _corr_reduction = 0.70
                    _sizing_method += f"+corr_reduce({_new_base}={_corr_val:.2f})"
                    break
            size_pct *= _corr_reduction
    except Exception:
        pass

    try:
        _strat_mult = self._get_strategy_multiplier(source_strategy)
        if _strat_mult != 1.0:
            size_pct *= _strat_mult
            _sizing_method += f"+strat_dampen*{_strat_mult:.2f}"
    except Exception:
        pass

    try:
        _rl_agent = getattr(self.component_registry, "rl_agent", None) if self.component_registry else None
        if _rl_agent and hasattr(_rl_agent, "predict"):
            _rl_state  = [confidence, strength, current_vol, regime_pos_mult, session_mult]
            _rl_action, _ = _rl_agent.predict(_rl_state)
            _rl_factor = float(_rl_action[0]) if hasattr(_rl_action, "__len__") else float(_rl_action)
            _rl_factor = max(0.1, min(2.0, _rl_factor))
            size_pct  *= _rl_factor
            _sizing_method += f"+rl_size*{_rl_factor:.2f}"
    except Exception:
        pass

    try:
        _existing_pos = (self.positions or {}).get(symbol)
        if _existing_pos is not None:
            _existing_qty  = float((_existing_pos or {}).get("quantity", 0) or 0)
            _existing_side = str((_existing_pos or {}).get("side", "")).upper()
            if _existing_qty > 0 and _existing_side:
                _max_pyr = int(getattr(self.config, "max_pyramids_per_position", 2) or 2)
                _pyr_cnt = int((_existing_pos or {}).get("pyramid_count", 0) or 0)
                if action == "BUY" and _existing_side == "BUY" and _pyr_cnt >= _max_pyr:
                    size_pct *= 0.5; _sizing_method += "+pyramid_limit_reduce"
                elif action == "SELL" and _existing_side == "SELL" and _pyr_cnt >= _max_pyr:
                    size_pct *= 0.5; _sizing_method += "+short_pyramid_limit"
                elif (action == "BUY" and _existing_side == "SELL") or \
                     (action == "SELL" and _existing_side == "BUY"):
                    _sizing_method += "+close_opposite"
    except Exception:
        pass

    try:
        _regime_prefs = {
            "TRENDING_UP":   {"momentum", "breakout", "funding_rate", "funding_rate_harvester"},
            "TRENDING_DOWN": {"momentum", "breakout", "funding_rate", "funding_rate_harvester"},
            "RANGE":         {"mean_reversion", "stat_arb", "pairs", "kalman_pairs"},
            "NORMAL":        {"mean_reversion", "stat_arb", "pairs", "kalman_pairs"},
            "HIGH_VOL":      {"funding_rate", "funding_rate_harvester"},
            "CRISIS":        {"funding_rate", "funding_rate_harvester"},
        }
        _preferred = _regime_prefs.get(regime)
        if _preferred is not None:
            _src_lower = source_strategy.lower()
            if not any(p in _src_lower for p in _preferred):
                if regime in ("HIGH_VOL", "CRISIS"):
                    confidence *= 0.5; _sizing_method += "+regime_mismatch_crisis*0.5"
                else:
                    confidence *= 0.7; _sizing_method += "+regime_mismatch*0.7"
    except Exception:
        pass

    try:
        _sss = getattr(self, "_strategy_state_store", None)
        if _sss:
            _state = _sss.get_state(source_strategy)
            if _state:
                _cw = int(_state.get("consecutive_wins", 0) or 0)
                if _cw >= 5:
                    size_pct *= 1.25; _sizing_method += "+hot_hand*1.25"
                elif _cw >= 3:
                    size_pct *= 1.15; _sizing_method += "+hot_hand*1.15"
    except Exception:
        pass

    size_pct = min(size_pct, max_pos_pct)
    return (size_pct, _sizing_method)
