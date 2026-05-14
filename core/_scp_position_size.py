"""
core/_scp_position_size.py

Drop-in replacement for _compute_position_size from execute_signals_helpers.py.

Wires SmallCapitalPipeline (component_registry.small_capital_pipeline) as the
primary position sizing source.  Falls back to KellySizer / legacy Kelly when
the pipeline is not registered.

Upgrade (v2): ATR input + fractional Kelly overlay
--------------------------------------------------
- Reads live ATR from the signal fields or indicator cache
- Feeds ATR into core.position_sizing.compute_position_size
- Fractional Kelly (0.5× full Kelly) used as a secondary ceiling
- Portfolio heat guard applied before returning

Monkey-patch into UnifiedTradingSystem at startup:

    from core._scp_position_size import _compute_position_size
    from core._scp_position_size import _after_fill_hook, _after_close_hook
    UnifiedTradingSystem._compute_position_size = _compute_position_size
    UnifiedTradingSystem._scp_after_fill      = _after_fill_hook
    UnifiedTradingSystem._scp_after_close     = _after_close_hook
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# ATR helper
# ─────────────────────────────────────────────────────────────────────

def _resolve_atr(self, symbol: str, sig_fields: dict) -> Optional[float]:
    """
    Best-effort ATR resolution:
    1. sig_fields["atr"]          — signal already carries it
    2. indicator_cache.get_atr()  — live cache
    3. atr_stops component        — legacy
    Returns None if unavailable.
    """
    # 1. Direct from signal
    atr_val = sig_fields.get("atr") or sig_fields.get("atr14")
    if atr_val and float(atr_val) > 0:
        return float(atr_val)

    cr = getattr(self, "component_registry", None)
    if cr is None:
        return None

    # 2. Indicator cache
    ic = getattr(cr, "indicator_cache", None)
    if ic is not None:
        try:
            atr_val = ic.get(symbol, "atr") or ic.get(symbol, "atr14")
            if atr_val and float(atr_val) > 0:
                return float(atr_val)
        except Exception:
            pass

    # 3. atr_stops component
    atr_comp = getattr(cr, "atr_stops", None)
    if atr_comp is not None:
        try:
            atr_val = atr_comp.get_atr(symbol)
            if atr_val and float(atr_val) > 0:
                return float(atr_val)
        except Exception:
            pass

    return None


def _resolve_open_risk(self) -> float:
    """Total AUD currently at risk across open positions (best effort)."""
    try:
        positions = getattr(self, "positions", None) or {}
        equity    = float(getattr(self, "portfolio_value_aud", 1.0) or 1.0)
        total_risk = 0.0
        for sym, pos in positions.items():
            if pos is None:
                continue
            qty     = float((pos or {}).get("quantity", 0) or 0)
            entry   = float((pos or {}).get("entry_price", 0) or 0)
            sl      = float((pos or {}).get("stop_loss", 0) or 0)
            if qty > 0 and entry > 0 and sl > 0:
                risk_per_unit = abs(entry - sl)
                total_risk   += qty * risk_per_unit
        return total_risk
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────
# Lifecycle hooks
# ─────────────────────────────────────────────────────────────────────

def _after_fill_hook(self) -> None:
    """Call immediately after a BUY order is confirmed filled."""
    _scp = _get_scp(self)
    if _scp is not None:
        try:
            _scp.on_position_opened()
        except Exception as exc:
            logger.debug("_after_fill_hook: SmallCapitalPipeline.on_position_opened failed: %s", exc)


def _after_close_hook(self, strategy: str, symbol: str, pnl: float) -> None:
    """Call immediately after a position is closed with its realised PnL."""
    _scp = _get_scp(self)
    if _scp is not None:
        try:
            _scp.on_trade_closed(strategy, symbol, pnl)
        except Exception as exc:
            logger.debug("_after_close_hook: SmallCapitalPipeline.on_trade_closed failed: %s", exc)
    else:
        _ks = getattr(self.component_registry, "kelly_sizer", None) if self.component_registry else None
        if _ks is not None:
            try:
                _ks.record_trade(strategy, symbol, pnl)
            except Exception as exc:
                logger.debug("_after_close_hook: kelly_sizer.record_trade failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────

def _get_scp(self):
    """Return the SmallCapitalPipeline instance or None."""
    if self.component_registry is None:
        return None
    return getattr(self.component_registry, "small_capital_pipeline", None)


# ─────────────────────────────────────────────────────────────────────
# Replacement _compute_position_size  (v2 — ATR + Kelly overlay)
# ─────────────────────────────────────────────────────────────────────

def _compute_position_size(self, sig_fields: dict, ctx: dict) -> tuple:
    """
    Compute position size percentage.

    Priority stack
    --------------
    1. ATR-vol-targeting  (core.position_sizing.size_from_atr)
    2. Fractional Kelly ceiling  (core.position_sizing.kelly_fraction)
    3. SmallCapitalPipeline dollar size  (legacy SCP primary path)
    4. KellySizer component / legacy internal Kelly  (fallback)

    Portfolio heat guard is applied before the final cap.

    Returns (size_pct: float, sizing_method: str).
    """
    from core.capital_tier   import classify_tier
    from core.position_sizing import (
        size_from_atr, kelly_fraction, apply_portfolio_heat,
    )

    symbol          = sig_fields["symbol"]
    action          = sig_fields["action"]
    confidence      = sig_fields["confidence"]
    strength        = sig_fields["strength"]
    source_strategy = sig_fields["source_strategy"]

    regime               = ctx["regime"]
    regime_pos_mult      = ctx["regime_pos_mult"]
    session_mult         = ctx["session_mult"]
    macro_event_imminent = ctx["macro_event_imminent"]

    equity_aud   = float(getattr(self, "portfolio_value_aud", 1.0) or 1.0)
    max_pos_pct  = float(getattr(self.config, "max_position_pct", 0.25) or 0.25)
    tier         = classify_tier(equity_aud)
    aud_to_usd   = float(ctx.get("aud_to_usd") or 0.65)

    # ── Current price best-effort ─────────────────────────────────────
    price = float(sig_fields.get("price") or sig_fields.get("entry_price") or 0.0)
    if price <= 0:
        try:
            price = float(getattr(self, "last_prices", {}).get(symbol, 0) or 0)
        except Exception:
            price = 0.0

    # ── ATR resolution ────────────────────────────────────────────────
    atr = _resolve_atr(self, symbol, sig_fields)

    # ── Open risk ─────────────────────────────────────────────────────
    open_risk_aud = _resolve_open_risk(self)

    size_pct       = 0.0
    _sizing_method = "default"

    # ── PATH A: ATR-based sizing (new primary when ATR available) ─────
    if atr and atr > 0 and price > 0:
        _stats        = self._get_strategy_trade_stats(source_strategy)
        _win_rate     = float(_stats.get("win_rate",  0.5) or 0.5)
        _avg_win_r    = float(_stats.get("avg_win",   1.5) or 1.5)
        _avg_loss_r   = float(_stats.get("avg_loss",  1.0) or 1.0)

        _atr_size_aud = size_from_atr(
            equity_aud          = equity_aud,
            atr                 = atr,
            price               = price,
            tier                = tier,
        )

        # Kelly ceiling
        _k_frac = kelly_fraction(
            win_rate   = _win_rate,
            avg_win_r  = _avg_win_r,
            avg_loss_r = _avg_loss_r,
            fraction   = 0.5,
            tier       = tier,
        )
        _kelly_size_aud = equity_aud * _k_frac if _k_frac > 0 else _atr_size_aud

        _raw_aud = min(_atr_size_aud, _kelly_size_aud) if _kelly_size_aud > 0 else _atr_size_aud

        # Portfolio heat guard
        _raw_aud = apply_portfolio_heat(
            raw_size_aud  = _raw_aud,
            open_risk_aud = open_risk_aud,
            equity_aud    = equity_aud,
            tier          = tier,
        )

        size_pct       = min(_raw_aud / equity_aud, max_pos_pct)
        _sizing_method = (
            f"atr_kelly(atr={atr:.4f},k_frac={_k_frac:.3f},"
            f"tier={tier.value},heat_used={open_risk_aud:.0f}aud)"
        )

    # ── PATH B: SmallCapitalPipeline ──────────────────────────────────
    if size_pct == 0.0:
        _scp = _get_scp(self)
        if _scp is not None:
            try:
                _stats        = self._get_strategy_trade_stats(source_strategy)
                _current_vol  = self._get_current_vol(symbol) or None
                _baseline_vol = getattr(self, "_baseline_vol", None) or _current_vol

                _dollar_size = _scp.get_position_size(
                    strategy     = source_strategy,
                    symbol       = symbol,
                    win_rate     = float(_stats.get("win_rate",  0.5) or 0.5),
                    avg_win      = float(_stats.get("avg_win",   0.0) or 0.0),
                    avg_loss     = float(_stats.get("avg_loss",  0.0) or 0.0),
                    current_vol  = _current_vol,
                    baseline_vol = _baseline_vol,
                )

                if _dollar_size <= 0:
                    logger.info(
                        "_compute_position_size: SmallCapitalPipeline returned 0 for %s %s — blocked",
                        action, symbol,
                    )
                    return (0.0, "BLOCKED:small_capital_pipeline")

                size_pct       = min((_dollar_size / max(aud_to_usd, 1e-6)) / equity_aud, max_pos_pct)
                _sizing_method = f"small_capital_pipeline(${_dollar_size:.2f})"

            except Exception as _scp_exc:
                logger.warning(
                    "_compute_position_size: SmallCapitalPipeline failed (%s) — Kelly fallback",
                    _scp_exc,
                )

    # ── PATH C: KellySizer component / legacy Kelly ───────────────────
    if size_pct == 0.0:
        _ks = (
            getattr(self.component_registry, "kelly_sizer", None)
            if self.component_registry else None
        )
        if _ks is not None:
            try:
                _ke = _ks.compute(source_strategy, symbol)
                if _ke.n_trades >= 20 and _ke.kelly_fraction > 0:
                    size_pct       = _ke.position_pct
                    _sizing_method = (
                        f"kelly_measured(f={_ke.kelly_fraction:.3f},"
                        f"wr={_ke.win_rate:.1%},n={_ke.n_trades})"
                    )
                elif _ke.n_trades >= 20:
                    size_pct       = min(max_pos_pct, confidence * strength * max_pos_pct * 0.5)
                    _sizing_method = "kelly_no_edge"
                else:
                    size_pct       = min(max_pos_pct, confidence * strength * max_pos_pct)
                    _sizing_method = f"default(kelly_n={_ke.n_trades})"
            except Exception:
                size_pct       = min(max_pos_pct, confidence * strength * max_pos_pct)
                _sizing_method = "default"
        else:
            strategy_stats = self._get_strategy_trade_stats(source_strategy)
            if strategy_stats["n_trades"] >= 20 and strategy_stats["avg_loss"] > 0:
                kelly_pct = self._kelly_size(
                    strategy_stats["win_rate"],
                    strategy_stats["avg_win"],
                    strategy_stats["avg_loss"],
                )
                size_pct       = kelly_pct if kelly_pct > 0 else min(max_pos_pct, confidence * strength * max_pos_pct)
                _sizing_method = "kelly_legacy" if kelly_pct > 0 else "default_no_kelly_edge"
            else:
                size_pct       = min(max_pos_pct, confidence * strength * max_pos_pct)
                _sizing_method = "default"

    # ── Vol adjustment ────────────────────────────────────────────────
    current_vol = self._get_current_vol(symbol)
    if current_vol > 0:
        size_pct       = self._vol_adjusted_size(size_pct, current_vol)
        _sizing_method += "+vol_adj"

    # ── Signal quality discount ───────────────────────────────────────
    sig_quality = self._get_signal_quality()
    if sig_quality is not None:
        sq_rec = sig_quality.get("recommendation", "moderate")
        if sq_rec == "conflicted":
            size_pct *= 0.5;  _sizing_method += "+conflict_discount"
        elif sq_rec == "weak":
            size_pct *= 0.7;  _sizing_method += "+weak_discount"

    # ── Regime scaling ────────────────────────────────────────────────
    size_pct       *= regime_pos_mult
    _sizing_method += f"+regime({regime})*{regime_pos_mult:.2f}"

    # ── Session scaling ───────────────────────────────────────────────
    size_pct       *= session_mult
    if session_mult != 1.0:
        _sizing_method += f"+session*{session_mult:.2f}"

    # ── Macro event ───────────────────────────────────────────────────
    if macro_event_imminent and action == "SELL":
        size_pct *= 0.7;  _sizing_method += "+macro_reduce_30pct"

    # ── Drawdown-adaptive sizing ──────────────────────────────────────
    try:
        _peak_cap = float(self.peak_equity_aud)
        _curr_cap = float(self.portfolio_value_aud)
        if _peak_cap > 0 and _curr_cap < _peak_cap:
            _dd_ratio = (_peak_cap - _curr_cap) / _peak_cap
            _dd_mult  = max(0.25, 1.0 - _dd_ratio * 2.0)
            size_pct       *= _dd_mult
            _sizing_method += f"+dd_adj({_dd_ratio:.3f})*{_dd_mult:.2f}"
    except Exception as _dd_exc:
        logger.debug("_compute_position_size: drawdown sizing failed: %s", _dd_exc)

    # ── Correlation reduction ─────────────────────────────────────────
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
                if _cr is not None and getattr(_cr, "correlation_monitor", None) is not None:
                    try:
                        _corr_val = abs(float(getattr(_cr.correlation_monitor, "_last_avg_corr", 0.0) or 0.0))
                    except Exception:
                        _corr_val = 0.0
                if _corr_val == 0.0 and {_new_base.split("/")[0], _pos_base.split("/")[0]} == {"BTC", "ETH"}:
                    _corr_val = 0.85
                _pos_side = str((_pos_data or {}).get("side", "BUY")).upper()
                if _corr_val > 0.85 and _pos_side == action:
                    _corr_reduction = 0.70
                    _sizing_method += f"+corr_reduce({_new_base}/{_pos_base}={_corr_val:.2f})"
                    break
            size_pct *= _corr_reduction
    except Exception as _corr_exc:
        logger.debug("_compute_position_size: correlation check failed: %s", _corr_exc)

    # ── Strategy dampening ────────────────────────────────────────────
    try:
        _strat_mult = self._get_strategy_multiplier(source_strategy)
        if _strat_mult != 1.0:
            size_pct       *= _strat_mult
            _sizing_method += f"+strat_dampen*{_strat_mult:.2f}"
    except Exception:
        pass

    # ── RL agent sizing ───────────────────────────────────────────────
    try:
        _rl_agent = getattr(self.component_registry, "rl_agent", None) if self.component_registry else None
        if _rl_agent is not None and hasattr(_rl_agent, "predict"):
            _rl_state  = [confidence, strength, current_vol, regime_pos_mult, session_mult]
            _rl_action, _ = _rl_agent.predict(_rl_state)
            _rl_factor = float(_rl_action[0]) if hasattr(_rl_action, "__len__") else float(_rl_action)
            _rl_factor = max(0.1, min(2.0, _rl_factor))
            size_pct       *= _rl_factor
            _sizing_method += f"+rl_size*{_rl_factor:.2f}"
    except Exception as _rl_exc:
        logger.debug("_compute_position_size: RL sizing unavailable: %s", _rl_exc)

    # ── Position conflict ─────────────────────────────────────────────
    try:
        _existing_pos = (self.positions or {}).get(symbol)
        if _existing_pos is not None:
            _existing_qty  = float((_existing_pos or {}).get("quantity", 0) or 0)
            _existing_side = str((_existing_pos or {}).get("side", "")).upper()
            if _existing_qty > 0 and _existing_side:
                if action == "BUY" and _existing_side == "BUY":
                    _pyramid_count = int((_existing_pos or {}).get("pyramid_count", 0) or 0)
                    _max_pyramids  = int(getattr(self.config, "max_pyramids_per_position", 2) or 2)
                    if _pyramid_count >= _max_pyramids:
                        size_pct *= 0.5;  _sizing_method += "+pyramid_limit_reduce"
                elif action == "SELL" and _existing_side == "SELL":
                    _pyramid_count = int((_existing_pos or {}).get("pyramid_count", 0) or 0)
                    _max_pyramids  = int(getattr(self.config, "max_pyramids_per_position", 2) or 2)
                    if _pyramid_count >= _max_pyramids:
                        size_pct *= 0.5;  _sizing_method += "+short_pyramid_limit"
                elif (action == "BUY" and _existing_side == "SELL") or \
                     (action == "SELL" and _existing_side == "BUY"):
                    _sizing_method += "+close_opposite"
    except Exception:
        pass

    # ── Regime whitelist ──────────────────────────────────────────────
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
        if _preferred:
            _src_lower = source_strategy.lower()
            if not any(p in _src_lower for p in _preferred):
                if regime in ("HIGH_VOL", "CRISIS"):
                    confidence *= 0.5;  _sizing_method += "+regime_mismatch_crisis*0.5"
                else:
                    confidence *= 0.7;  _sizing_method += "+regime_mismatch*0.7"
    except Exception:
        pass

    # ── Hot-hand boost ────────────────────────────────────────────────
    try:
        _sss_hot = getattr(self, "_strategy_state_store", None)
        if _sss_hot is not None:
            _hot_state = _sss_hot.get_state(source_strategy)
            if _hot_state is not None:
                _consec_wins = int(_hot_state.get("consecutive_wins", 0) or 0)
                if _consec_wins >= 5:
                    size_pct *= 1.25;  _sizing_method += "+hot_hand*1.25"
                elif _consec_wins >= 3:
                    size_pct *= 1.15;  _sizing_method += "+hot_hand*1.15"
    except Exception:
        pass

    # ── Hard cap ─────────────────────────────────────────────────────
    size_pct = min(size_pct, max_pos_pct)

    return (size_pct, _sizing_method)
