"""
execute_signals_helpers.py — Extracted sub-methods from UnifiedTradingSystem._execute_signals().

These are standalone functions that take `self` as their first argument so they
can be monkey-patched or called as methods on UnifiedTradingSystem instances.
Each function preserves the EXACT logic from the original monolithic method.
"""

import logging
import math
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# (a) _pre_execute_context
# ─────────────────────────────────────────────────────────────────────

def _pre_execute_context(self) -> dict:
    """
    Build the execution context dict containing all system-level state
    needed before iterating over individual signals.

    Returns a dict with keys:
        macro_event_imminent, macro_event_name, macro_event_hours,
        regime, regime_pos_mult, regime_stop_mult, regime_tp_mult,
        session_mult, mode, is_live, aud_to_usd, portfolio_value,
        daily_loss_exceeded, var_breach, _cycle_advisory
    """
    mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
    is_live = mode == "live"
    aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
    portfolio_value = float(self.portfolio_value_aud)

    # --- 0a. Macro calendar check: block BUYs near high-impact events ---
    macro_event_imminent = False
    macro_event_name = ""
    macro_event_hours = None
    try:
        _fred_cal = None
        if self.component_registry is not None:
            _fred_cal = getattr(self.component_registry, "fred_calendar", None)
        if _fred_cal is None:
            from data.macro.fred_calendar import FREDCalendar
            _fred_cal = FREDCalendar()
        snap = _fred_cal.get_upcoming(days=1)
        if snap.hours_to_next_high is not None and snap.hours_to_next_high <= 2.0:
            macro_event_imminent = True
            macro_event_name = snap.next_high_impact.name if snap.next_high_impact else "unknown"
            macro_event_hours = snap.hours_to_next_high
            logger.warning(
                "_execute_signals: MACRO EVENT IMMINENT — '%s' in %.1f hours, blocking new BUY entries",
                macro_event_name, macro_event_hours,
            )
    except Exception as _macro_exc:
        logger.debug("_execute_signals: macro calendar check failed: %s", _macro_exc)

    # --- 0b. Get current regime for sizing adjustments ---
    # FIX 2: Force regime detection — compute fallback if empty/None
    regime = str(getattr(self, "_latest_regime_label", "") or "").upper().strip()
    if not regime:
        _fallback_fn = getattr(self, "_compute_fallback_regime", None)
        if callable(_fallback_fn):
            regime = _fallback_fn()
        else:
            regime = "NORMAL"
        try:
            self._latest_regime_label = regime
        except AttributeError:
            pass
    regime_pos_mult = self.REGIME_POSITION_SCALE.get(regime, 1.0)
    regime_stop_mult = self.REGIME_STOP_SCALE.get(regime, 1.0)
    regime_tp_mult = self.REGIME_TP_SCALE.get(regime, 1.0)

    # --- 0c. Session-based sizing (crypto volume patterns) ---
    from datetime import timezone as _tz
    hour_utc = datetime.now(tz=_tz.utc).hour
    if 13 <= hour_utc <= 17:
        session_mult = 1.1   # Peak volume: NY open
    elif 8 <= hour_utc <= 10:
        session_mult = 1.05  # London open
    elif 1 <= hour_utc <= 5:
        session_mult = 0.8   # Low volume: worse fills
    else:
        session_mult = 1.0

    logger.info(
        "_execute_signals: regime=%s (pos*%.2f, stop*%.2f, tp*%.2f), session_mult=%.2f, macro_imminent=%s",
        regime, regime_pos_mult, regime_stop_mult, regime_tp_mult,
        session_mult, macro_event_imminent,
    )

    # --- 0. System-level risk gates (BLOCKING) ---
    # Daily loss limit: if exceeded, only allow SELL (closing) signals
    daily_loss_exceeded = (
        self.unified_risk_manager is not None
        and self.unified_risk_manager.is_daily_loss_limit_exceeded()
    )
    if daily_loss_exceeded:
        logger.warning("_execute_signals: daily loss limit exceeded — blocking new positions, allowing closes only")

    # VaR/CVaR limit check (portfolio-level)
    var_limit_pct = float(getattr(self.config, "portfolio_var_limit_pct", 0.0) or 0.0)
    cvar_limit_pct = float(getattr(self.config, "portfolio_cvar_limit_pct", 0.0) or 0.0)
    var_breach = False
    if self.unified_risk_manager is not None and (var_limit_pct > 0 or cvar_limit_pct > 0):
        try:
            metrics = self.unified_risk_manager.get_risk_metrics()
            capital = max(metrics.current_capital, 1e-9)
            if var_limit_pct > 0 and abs(metrics.var_95) / capital >= var_limit_pct:
                var_breach = True
                logger.warning(
                    "_execute_signals: VaR breach — 95%% VaR %.2f%% >= limit %.2f%%",
                    abs(metrics.var_95) / capital * 100.0, var_limit_pct * 100.0,
                )
            if cvar_limit_pct > 0 and abs(metrics.var_99) / capital >= cvar_limit_pct:
                var_breach = True
                logger.warning(
                    "_execute_signals: CVaR breach — 99%% VaR %.2f%% >= limit %.2f%%",
                    abs(metrics.var_99) / capital * 100.0, cvar_limit_pct * 100.0,
                )
        except Exception as exc:
            logger.debug("_execute_signals: VaR/CVaR check failed: %s", exc)

    # FIX #1: Make cycle advisory available to all gate blocks
    _cycle_advisory = getattr(self, "_last_cycle_advisory", None) or {}

    return {
        "macro_event_imminent": macro_event_imminent,
        "macro_event_name": macro_event_name,
        "macro_event_hours": macro_event_hours,
        "regime": regime,
        "regime_pos_mult": regime_pos_mult,
        "regime_stop_mult": regime_stop_mult,
        "regime_tp_mult": regime_tp_mult,
        "session_mult": session_mult,
        "mode": mode,
        "is_live": is_live,
        "aud_to_usd": aud_to_usd,
        "portfolio_value": portfolio_value,
        "daily_loss_exceeded": daily_loss_exceeded,
        "var_breach": var_breach,
        "_cycle_advisory": _cycle_advisory,
    }


# ─────────────────────────────────────────────────────────────────────
# (b) _extract_signal_fields
# ─────────────────────────────────────────────────────────────────────

def _extract_signal_fields(self, sig) -> dict | None:
    """
    Extract and validate fields from a single TradingSignal object.

    Returns a dict with keys:
        symbol, action, confidence, strength, entry_price, stop_loss,
        take_profit, reasoning, source_strategy, _sig_age, _age_urgency

    Returns None (with a 'blocked' result dict) if the signal is too stale,
    or None (plain) if the action is invalid.  The caller must check for
    None and handle appropriately (skip/continue).

    When the signal is stale, the returned dict has:
        {"_blocked": True, "result": {...}}
    so the caller can append the result and continue.
    """
    symbol = str(getattr(sig, "symbol", "") or "")
    action = str(getattr(sig, "action", "") or "").upper()
    confidence = float(getattr(sig, "confidence", 0.0) or 0.0)
    strength = float(getattr(sig, "strength", 0.0) or 0.0)
    entry_price = float(getattr(sig, "entry_price", 0.0) or 0.0)
    stop_loss = getattr(sig, "stop_loss", None)
    take_profit = getattr(sig, "take_profit", None)
    reasoning = str(getattr(sig, "reasoning", "") or "")
    source_strategy = str(
        getattr(sig, "strategy", "")
        or getattr(sig, "source_strategy", "")
        or (sig.get("strategy") if isinstance(sig, dict) else "")
        or (sig.get("source_strategy") if isinstance(sig, dict) else "")
        or "unknown"
    )

    # --- FIX 15: Signal staleness decay ---
    _sig_ts = getattr(sig, "timestamp", None)
    if _sig_ts is None:
        _sig_age = 0.0
    elif isinstance(_sig_ts, (int, float)):
        _sig_age = time.time() - float(_sig_ts)
    elif hasattr(_sig_ts, "timestamp"):
        _sig_age = time.time() - _sig_ts.timestamp()  # datetime -> unix
    else:
        _sig_age = 0.0
    if _sig_age > 120.0:
        logger.warning(
            "_execute_signals: signal too stale — age=%.1fs > 120s, rejecting %s %s",
            _sig_age, action, symbol,
        )
        return {
            "_blocked": True,
            "result": {
                "symbol": symbol,
                "side": action,
                "status": "blocked",
                "reason": f"signal_too_stale:age={_sig_age:.1f}s",
            },
        }
    if _sig_age > 0.1:
        import math as _math_staleness
        confidence *= _math_staleness.exp(-_sig_age / 30.0)

    # --- FIX 23: Signal age-based urgency ---
    if _sig_age < 5.0:
        _age_urgency = 0.2   # fresh -> maker, patient
    elif _sig_age < 30.0:
        _age_urgency = 0.5   # balanced
    elif _sig_age < 60.0:
        _age_urgency = 0.8   # aggressive, fill fast
    else:
        _age_urgency = 1.0   # market order or reject

    if action not in ("BUY", "SELL"):
        logger.debug("_execute_signals: skipping signal with action=%s for %s", action, symbol)
        return None

    return {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "strength": strength,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasoning": reasoning,
        "source_strategy": source_strategy,
        "_sig_age": _sig_age,
        "_age_urgency": _age_urgency,
        "_num_confirmations": int(getattr(sig, "num_confirmations", 0) or 0),
        "_sig_obj": sig,  # keep reference for downstream access
    }


# ─────────────────────────────────────────────────────────────────────
# (c) _apply_risk_gates
# ─────────────────────────────────────────────────────────────────────

def _apply_risk_gates(self, sig_fields: dict, ctx: dict) -> tuple:
    """
    Apply all risk gates that were originally `continue` statements in the
    main signal loop.

    Returns (approved: bool, reason: str).
    If approved is False, the caller should append a blocked result and
    continue to the next signal.
    """
    symbol = sig_fields["symbol"]
    action = sig_fields["action"]
    confidence = sig_fields["confidence"]
    strength = sig_fields["strength"]
    entry_price = sig_fields["entry_price"]
    source_strategy = sig_fields["source_strategy"]

    daily_loss_exceeded = ctx["daily_loss_exceeded"]
    macro_event_imminent = ctx["macro_event_imminent"]
    macro_event_name = ctx["macro_event_name"]
    macro_event_hours = ctx["macro_event_hours"]
    var_breach = ctx["var_breach"]
    portfolio_value = ctx["portfolio_value"]
    aud_to_usd = ctx["aud_to_usd"]

    # --- 0b. Strategy cooldown gate (BLOCKING) ---
    _sss = getattr(self, "_strategy_state_store", None)
    if _sss is not None and _sss.check_cooldown(source_strategy):
        remaining = _sss.cooldown_remaining_seconds(source_strategy)
        logger.info(
            "_execute_signals: DROPPED %s %s from strategy '%s' — cooldown active (%.0fs remaining)",
            action, symbol, source_strategy, remaining,
        )
        return (False, f"strategy_cooldown:{source_strategy}")

    if not symbol or entry_price <= 0:
        logger.warning("_execute_signals: invalid signal (symbol=%s, entry_price=%s)", symbol, entry_price)
        return (False, "invalid_signal")

    # --- 1a. Daily loss limit gate (BLOCKING for new positions) ---
    if daily_loss_exceeded and action == "BUY":
        logger.warning(
            "_execute_signals: REJECTED %s %s — daily loss limit exceeded, only closes allowed",
            action, symbol,
        )
        return (False, "daily_loss_limit_exceeded")

    # --- 1a2. Macro event gate (BLOCKING for new BUY entries) ---
    if macro_event_imminent and action == "BUY":
        logger.warning(
            "_execute_signals: REJECTED %s %s — macro event '%s' in %.1f hours, blocking new entries",
            action, symbol, macro_event_name, macro_event_hours or 0.0,
        )
        return (False, f"macro_event_imminent:{macro_event_name}")

    # --- 1b. VaR/CVaR breach gate (BLOCKING for new positions) ---
    if var_breach and action == "BUY":
        logger.warning(
            "_execute_signals: REJECTED %s %s — VaR/CVaR limit breached, only closes allowed",
            action, symbol,
        )
        return (False, "var_limit_breached")

    # --- 1c. UnifiedRiskManager pre-trade check (BLOCKING) ---
    side_for_check = action

    # -- SMART POSITION SIZING (pre-trade estimate) --
    # Base size from confidence * strength
    size_usd_estimate = confidence * strength * portfolio_value * aud_to_usd * 0.1

    # (a) VOLATILITY-ADJUSTED sizing
    _sig_vol = 0.005
    try:
        _sig_vol = self._get_current_vol(symbol) or 0.005
    except Exception:
        pass
    if _sig_vol > 0.01:
        # High vol: reduce size by 30%
        size_usd_estimate *= 0.70
    elif _sig_vol < 0.004:
        # Low vol: increase size by 20%
        size_usd_estimate *= 1.20

    # (b) CONVICTION-WEIGHTED sizing (based on num_confirmations from signal)
    _num_conf = int(sig_fields.get("_num_confirmations", 0) or 0)
    if _num_conf >= 5:
        size_usd_estimate *= 1.0   # 100% of max
    elif _num_conf >= 3:
        size_usd_estimate *= 0.70  # 70%
    elif _num_conf >= 1:
        size_usd_estimate *= 0.40  # 40%
    else:
        size_usd_estimate *= 0.30  # Minimal

    # (c) PORTFOLIO HEAT: reduce size when many positions open
    _open_count = sum(
        1 for p in (self.positions or {}).values()
        if float((p or {}).get("quantity", 0) or 0) > 0
    )
    if _open_count >= 2:
        size_usd_estimate *= 0.60  # Reduce by 40% with 2+ open positions

    # (d) DRAWDOWN PENALTY: reduce all sizes when in deep drawdown
    _current_dd = 0.0
    if self.peak_equity_aud > 0:
        _current_dd = (self.peak_equity_aud - self.portfolio_value_aud) / self.peak_equity_aud
    if _current_dd > 0.05:
        size_usd_estimate *= 0.50  # 50% reduction at >5% drawdown

    # (e) DIVERSIFICATION PREFERENCE: reduce size if already holding this symbol
    _existing_pos_qty = float((self.positions or {}).get(symbol, {}).get("quantity", 0) or 0)
    if _existing_pos_qty > 0 and action == "BUY":
        size_usd_estimate *= 0.50  # Don't double down on same symbol

    exchange_name = str(getattr(self.config, "primary_exchange", "kraken") or "kraken")

    if self.unified_risk_manager is not None and action == "BUY":
        approved, reject_reason = self.unified_risk_manager.pre_trade_risk_check(
            symbol=symbol,
            position_size_usd=size_usd_estimate,
        )
        if not approved:
            logger.warning(
                "_execute_signals: REJECTED %s %s — risk manager: %s",
                action, symbol, reject_reason,
            )
            return (False, reject_reason)

    # --- 1d. Max concurrent positions gate (BLOCKING) ---
    max_positions = int(getattr(self.config, "max_concurrent_positions", 5) or 0)
    if max_positions > 0 and action == "BUY":
        current_positions = sum(
            1 for p in (self.positions or {}).values()
            if float((p or {}).get("quantity", 0) or 0) > 0
        )
        if current_positions >= max_positions:
            logger.warning(
                "_execute_signals: REJECTED %s %s — max concurrent positions (%d/%d)",
                action, symbol, current_positions, max_positions,
            )
            return (False, f"max_concurrent_positions ({current_positions}/{max_positions})")

    if self.component_registry is not None:
        check = self.component_registry.pre_order_check(symbol, side_for_check, size_usd_estimate, exchange_name)
        if not check.get("allow", True):
            logger.info(
                "_execute_signals: risk gate blocked %s %s: %s",
                action, symbol, check.get("reasons", []),
            )
            return (False, str(check.get("reasons", ["risk_gate"])))

    return (True, "")


# ─────────────────────────────────────────────────────────────────────
# (d) _compute_position_size
# ─────────────────────────────────────────────────────────────────────

def _compute_position_size(self, sig_fields: dict, ctx: dict) -> tuple:
    """
    Compute position size percentage using Kelly sizing, volatility
    adjustment, signal quality, regime scaling, session scaling,
    drawdown-adaptive sizing, correlation reduction, strategy dampening,
    RL agent, position conflict check, regime whitelist, and hot hand boost.

    Returns (size_pct: float, sizing_method: str).
    """
    symbol = sig_fields["symbol"]
    action = sig_fields["action"]
    confidence = sig_fields["confidence"]
    strength = sig_fields["strength"]
    source_strategy = sig_fields["source_strategy"]

    regime = ctx["regime"]
    regime_pos_mult = ctx["regime_pos_mult"]
    session_mult = ctx["session_mult"]
    macro_event_imminent = ctx["macro_event_imminent"]

    # --- 2. Position sizing (measured-edge Kelly + volatility-adjusted) ---
    max_pos_pct = float(getattr(self.config, "max_position_pct", 0.25) or 0.25)
    min_pos_aud = float(getattr(self.config, "min_position_size_aud", 10.0) or 10.0)
    strategy_stats = {"n_trades": 0, "win_rate": 0.5, "avg_win": 0.0, "avg_loss": 0.0}

    # Prefer KellySizer component (per strategy x symbol) if available
    _sizing_method = "default"
    _ks = getattr(self.component_registry, "kelly_sizer", None) if self.component_registry else None
    if _ks is not None:
        try:
            _ke = _ks.compute(source_strategy, symbol)
            if _ke.n_trades >= 20 and _ke.kelly_fraction > 0:
                size_pct = _ke.position_pct
                _sizing_method = f"kelly_measured(f={_ke.kelly_fraction:.3f},wr={_ke.win_rate:.1%},n={_ke.n_trades})"
            elif _ke.n_trades >= 20:
                # Kelly says no edge -- use minimal
                size_pct = min(max_pos_pct, confidence * strength * max_pos_pct * 0.5)
                _sizing_method = "kelly_no_edge"
            else:
                size_pct = min(max_pos_pct, confidence * strength * max_pos_pct)
                _sizing_method = f"default(kelly_n={_ke.n_trades})"
        except Exception:
            size_pct = min(max_pos_pct, confidence * strength * max_pos_pct)
    else:
        # Fallback to legacy internal Kelly
        _kelly_min_trades = 20
        strategy_stats = self._get_strategy_trade_stats(source_strategy)
        if strategy_stats["n_trades"] >= _kelly_min_trades and strategy_stats["avg_loss"] > 0:
            kelly_pct = self._kelly_size(
                strategy_stats["win_rate"],
                strategy_stats["avg_win"],
                strategy_stats["avg_loss"],
            )
            if kelly_pct > 0:
                size_pct = kelly_pct
                _sizing_method = "kelly_legacy"
            else:
                size_pct = min(max_pos_pct, confidence * strength * max_pos_pct)
                _sizing_method = "default_no_kelly_edge"
        else:
            size_pct = min(max_pos_pct, confidence * strength * max_pos_pct)

    # Apply volatility adjustment
    current_vol = self._get_current_vol(symbol)
    if current_vol > 0:
        size_pct = self._vol_adjusted_size(size_pct, current_vol)
        _sizing_method += "+vol_adj"

    # Apply signal quality discount if available
    sig_quality = self._get_signal_quality()
    if sig_quality is not None:
        sq_recommendation = sig_quality.get("recommendation", "moderate")
        if sq_recommendation == "conflicted":
            size_pct *= 0.5  # halve size on conflicted signals
            _sizing_method += "+conflict_discount"
        elif sq_recommendation == "weak":
            size_pct *= 0.7
            _sizing_method += "+weak_discount"

    # --- 2b. Regime-adaptive scaling ---
    size_pct *= regime_pos_mult
    _sizing_method += f"+regime({regime})*{regime_pos_mult:.2f}"

    # --- 2c. Session-based sizing ---
    size_pct *= session_mult
    if session_mult != 1.0:
        _sizing_method += f"+session*{session_mult:.2f}"

    # --- 2d. Macro event: reduce size by 30% for existing exits ---
    if macro_event_imminent and action == "SELL":
        size_pct *= 0.7
        _sizing_method += "+macro_reduce_30pct"

    # --- FIX 10: Drawdown-adaptive sizing ---
    try:
        _peak_cap = float(self.peak_equity_aud)
        _curr_cap = float(self.portfolio_value_aud)
        if _peak_cap > 0 and _curr_cap < _peak_cap:
            _dd_ratio = (_peak_cap - _curr_cap) / _peak_cap
            _dd_mult = max(0.25, 1.0 - _dd_ratio * 2.0)
            size_pct *= _dd_mult
            _sizing_method += f"+dd_adj({_dd_ratio:.3f})*{_dd_mult:.2f}"
            logger.info(
                "_execute_signals: drawdown adjustment — dd_ratio=%.3f, size_mult=%.2f",
                _dd_ratio, _dd_mult,
            )
    except Exception as _dd_exc:
        logger.debug("_execute_signals: drawdown sizing failed: %s", _dd_exc)

    # --- FIX 11: Correlation-based position reduction ---
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
                # Check correlation via component registry
                _corr_val = 0.0
                _cr = getattr(self, "component_registry", None)
                if _cr is not None and getattr(_cr, "correlation_monitor", None) is not None:
                    try:
                        _cm = _cr.correlation_monitor
                        _corr_val = abs(float(getattr(_cm, "_last_avg_corr", 0.0) or 0.0))
                    except Exception:
                        _corr_val = 0.0
                # Fallback: hardcoded BTC/ETH correlation
                if _corr_val == 0.0:
                    _btc_eth = {"BTC", "ETH"}
                    if {_new_base, _pos_base} == _btc_eth:
                        _corr_val = 0.85
                _pos_side = str((_pos_data or {}).get("side", "BUY")).upper()
                if _corr_val > 0.85 and _pos_side == action:
                    _corr_reduction = 0.70
                    _sizing_method += f"+corr_reduce({_new_base}/{_pos_base}={_corr_val:.2f})"
                    logger.info(
                        "_execute_signals: correlation reduction — %s/%s corr=%.2f, size*=0.70",
                        _new_base, _pos_base, _corr_val,
                    )
                    break
            
            # Also apply global correlation scalar from CorrelationMonitor
            if _cr is not None and getattr(_cr, "correlation_monitor", None) is not None:
                try:
                    _global_corr_scalar = _cr.correlation_monitor.get_position_scalar()
                    if _global_corr_scalar < 1.0:
                        _corr_reduction *= _global_corr_scalar
                        _sizing_method += f"*global_corr*{_global_corr_scalar:.2f}"
                        logger.info(
                            "_execute_signals: global correlation scalar applied — scalar=%.2f",
                            _global_corr_scalar,
                        )
                except Exception:
                    pass
            
            size_pct *= _corr_reduction
    except Exception as _corr_exc:
        logger.debug("_execute_signals: correlation check failed: %s", _corr_exc)

    # --- FIX 16: Aggressive strategy dampening ---
    try:
        _strat_mult = self._get_strategy_multiplier(source_strategy)
        if _strat_mult != 1.0:
            size_pct *= _strat_mult
            _sizing_method += f"+strat_dampen*{_strat_mult:.2f}"
    except Exception as _sd_exc:
        logger.debug("_execute_signals: strategy dampening failed: %s", _sd_exc)

    # --- FIX 18: Wire RL agent for execution sizing ---
    try:
        _rl_agent = None
        if self.component_registry is not None:
            _rl_agent = getattr(self.component_registry, "rl_agent", None)
        if _rl_agent is not None and hasattr(_rl_agent, "predict"):
            _rl_state = [confidence, strength, current_vol, regime_pos_mult, session_mult]
            _rl_action, _ = _rl_agent.predict(_rl_state)
            _rl_size_factor = float(_rl_action[0]) if hasattr(_rl_action, "__len__") else float(_rl_action)
            _rl_size_factor = max(0.1, min(2.0, _rl_size_factor))
            size_pct *= _rl_size_factor
            _sizing_method += f"+rl_size*{_rl_size_factor:.2f}"
    except Exception as _rl_exc:
        logger.debug("_execute_signals: RL sizing unavailable: %s", _rl_exc)

    # --- FIX 19: Position conflict check ---
    try:
        _existing_pos = (self.positions or {}).get(symbol)
        if _existing_pos is not None:
            _existing_qty = float((_existing_pos or {}).get("quantity", 0) or 0)
            _existing_side = str((_existing_pos or {}).get("side", "")).upper()
            if _existing_qty > 0 and _existing_side:
                if action == "BUY" and _existing_side == "BUY":
                    # Pyramid: check limits
                    _pyramid_count = int((_existing_pos or {}).get("pyramid_count", 0) or 0)
                    _max_pyramids = int(getattr(self.config, "max_pyramids_per_position", 2) or 2)
                    if _pyramid_count >= _max_pyramids:
                        logger.info(
                            "_execute_signals: CONFLICT — pyramid limit reached for %s (%d/%d)",
                            symbol, _pyramid_count, _max_pyramids,
                        )
                        size_pct *= 0.5
                        _sizing_method += "+pyramid_limit_reduce"
                elif action == "SELL" and _existing_side == "SELL":
                    _pyramid_count = int((_existing_pos or {}).get("pyramid_count", 0) or 0)
                    _max_pyramids = int(getattr(self.config, "max_pyramids_per_position", 2) or 2)
                    if _pyramid_count >= _max_pyramids:
                        size_pct *= 0.5
                        _sizing_method += "+short_pyramid_limit"
                elif (action == "BUY" and _existing_side == "SELL") or (action == "SELL" and _existing_side == "BUY"):
                    # Opposite direction -- this is a close, allow it
                    logger.info(
                        "_execute_signals: CONFLICT — %s signal for %s with existing %s position (closing)",
                        action, symbol, _existing_side,
                    )
                    _sizing_method += "+close_opposite"
    except Exception as _pc_exc:
        logger.debug("_execute_signals: position conflict check failed: %s", _pc_exc)

    # --- FIX 24: Regime-specific strategy whitelist ---
    try:
        _regime_prefs = {
            "TRENDING_UP": {"momentum", "breakout", "funding_rate", "funding_rate_harvester"},
            "TRENDING_DOWN": {"momentum", "breakout", "funding_rate", "funding_rate_harvester"},
            "RANGE": {"mean_reversion", "stat_arb", "pairs", "kalman_pairs"},
            "NORMAL": {"mean_reversion", "stat_arb", "pairs", "kalman_pairs"},
            "HIGH_VOL": {"funding_rate", "funding_rate_harvester"},
            "CRISIS": {"funding_rate", "funding_rate_harvester"},
        }
        _preferred = _regime_prefs.get(regime)
        if _preferred is not None:
            _src_lower = source_strategy.lower()
            _matches_regime = any(p in _src_lower for p in _preferred)
            if not _matches_regime:
                if regime in ("HIGH_VOL", "CRISIS"):
                    confidence *= 0.5
                    _sizing_method += "+regime_mismatch_crisis*0.5"
                else:
                    confidence *= 0.7
                    _sizing_method += "+regime_mismatch*0.7"
    except Exception as _rw_exc:
        logger.debug("_execute_signals: regime whitelist check failed: %s", _rw_exc)

    # --- FIX 25: Hot hand strategy boost ---
    try:
        _sss_hot = getattr(self, "_strategy_state_store", None)
        if _sss_hot is not None:
            _hot_state = _sss_hot.get_state(source_strategy)
            if _hot_state is not None:
                _consec_wins = int(_hot_state.get("consecutive_wins", 0) or 0)
                if _consec_wins >= 5:
                    _hot_boost = min(1.30, 1.25)
                    size_pct *= _hot_boost
                    _sizing_method += f"+hot_hand*{_hot_boost:.2f}"
                elif _consec_wins >= 3:
                    _hot_boost = 1.15
                    size_pct *= _hot_boost
                    _sizing_method += f"+hot_hand*{_hot_boost:.2f}"
    except Exception as _hh_exc:
        logger.debug("_execute_signals: hot hand boost failed: %s", _hh_exc)

    # Hard cap at max_position_pct
    size_pct = min(size_pct, max_pos_pct)

    return (size_pct, _sizing_method)


# ─────────────────────────────────────────────────────────────────────
# (e) _apply_intelligence_gates
# ─────────────────────────────────────────────────────────────────────

def _apply_intelligence_gates(self, sig_fields: dict, size_pct: float, advisory: dict, sizing_method: str, ctx: dict | None = None) -> tuple:
    """
    Apply ALL advisory-based intelligence gates (Batches G, K, L, M, N, O,
    and wire blocks).  Also applies NaN guard and gate floor.

    Returns (final_size_pct: float, sizing_method: str).

    When a gate would have caused a `continue` (skip) in the original code,
    this function returns size_pct = 0.0 with a special sizing_method prefix
    of "BLOCKED:" so the caller can detect it and append the appropriate
    blocked result.
    """
    if ctx is None:
        ctx = {}
    symbol = sig_fields["symbol"]
    action = sig_fields["action"]
    entry_price = sig_fields["entry_price"]
    source_strategy = sig_fields["source_strategy"]
    _cycle_advisory = advisory

    max_pos_pct = float(getattr(self.config, "max_position_pct", 0.25) or 0.25)

    current_vol = 0.005
    try:
        current_vol = self._get_current_vol(symbol) or 0.005
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════════
    # LEAN MODE: skip all 80+ intelligence gates — only enforce basic risk
    # ══════════════════════════════════════════════════════════════════════
    import os as _os_lean
    _lean = _os_lean.environ.get("ARGUS_LEAN_MODE", "").lower() in ("1", "true", "yes")
    if _lean:
        # In lean mode: cap at max_pos_pct, apply NaN guard, return
        import math as _m
        if not isinstance(size_pct, (int, float)) or _m.isnan(size_pct) or _m.isinf(size_pct):
            size_pct = 0.01
        size_pct = min(size_pct, max_pos_pct)
        size_pct = max(size_pct, max_pos_pct * 0.15)  # gate floor
        return (size_pct, sizing_method + "+lean_mode")

    # ══════════════════════════════════════════════════════════════════════
    # Batch P: Signal Quality Pre-Gates (Phase Y — self-awareness layer)
    # These run FIRST, before all other intelligence gates. They can BLOCK
    # trades entirely or reduce sizing based on system health signals.
    # ══════════════════════════════════════════════════════════════════════

    # P1: MetaGate — regime confidence + staleness + drawdown trade gate
    try:
        _tg = (_cycle_advisory or {}).get("trade_gate")
        if isinstance(_tg, dict):
            _tg_decision = str(_tg.get("decision", "allow")).lower()
            if _tg_decision == "halt":
                return (0.0, "BLOCKED:trade_gate_halt")
            if _tg_decision == "pause" and action == "BUY":
                return (0.0, "BLOCKED:trade_gate_pause")
            if _tg_decision == "reduce":
                size_pct *= 0.50
                sizing_method += "+trade_gate_reduce*0.50"
    except Exception:
        pass

    # P2: EscalatingGate — temporal MetaGate memory with escalation
    try:
        _eg = (_cycle_advisory or {}).get("escalating_gate")
        if isinstance(_eg, dict) and _eg.get("escalated"):
            _eg_decision = str(_eg.get("decision", "allow")).lower()
            if _eg_decision == "halt":
                return (0.0, "BLOCKED:escalating_gate_halt")
            if _eg_decision == "pause" and action == "BUY":
                return (0.0, "BLOCKED:escalating_gate_pause")
            if _eg_decision == "reduce":
                size_pct *= 0.50
                sizing_method += "+escalating_gate_reduce*0.50"
    except Exception:
        pass

    # P3: ConsecutiveLossGuard — per-strategy loss streaks + daily budget
    try:
        _clg = (_cycle_advisory or {}).get("consecutive_loss_guard")
        if isinstance(_clg, dict) and action == "BUY":
            _paused = _clg.get("paused_strategies") or {}
            if isinstance(_paused, dict) and source_strategy in _paused:
                return (0.0, f"BLOCKED:consecutive_loss_paused:{source_strategy}")
            if _clg.get("daily_budget_exhausted"):
                return (0.0, "BLOCKED:daily_budget_exhausted")
    except Exception:
        pass

    # P4: EdgeMonitor — live edge erosion detector
    try:
        _em = (_cycle_advisory or {}).get("edge_monitor")
        if isinstance(_em, dict) and _em.get("degraded"):
            _edge_score = float(_em.get("edge_score", 0.5) or 0.5)
            _em_factor = max(0.30, _edge_score)
            size_pct *= _em_factor
            sizing_method += f"+edge_degraded*{_em_factor:.2f}"
    except Exception:
        pass

    # P5: HealthScore — composite 0-100 system health
    try:
        _hs = (_cycle_advisory or {}).get("health_score")
        if isinstance(_hs, dict):
            _hs_label = str(_hs.get("label", "GOOD")).upper()
            if _hs_label == "CRITICAL" and action == "BUY":
                return (0.0, "BLOCKED:health_critical")
            if _hs_label == "POOR":
                size_pct *= 0.50
                sizing_method += "+health_poor*0.50"
            elif _hs_label in ("MARGINAL", "FAIR"):
                size_pct *= 0.75
                sizing_method += "+health_marginal*0.75"
    except Exception:
        pass

    # P6: StrategyRegimeMatrix — regime x strategy fitness [0.25, 1.5]
    try:
        _srm = (_cycle_advisory or {}).get("strategy_regime_matrix")
        if isinstance(_srm, dict):
            _srm_weights = _srm.get("current_regime_weights") or {}
            if isinstance(_srm_weights, dict) and source_strategy in _srm_weights:
                _srm_fitness = float(_srm_weights[source_strategy])
                _srm_fitness = max(0.25, min(1.5, _srm_fitness))
                size_pct *= _srm_fitness
                sizing_method += f"+srm_fitness*{_srm_fitness:.2f}"
    except Exception:
        pass

    # P7: RegimeTransitionMonitor — forward-looking regime risk
    try:
        _rt = (_cycle_advisory or {}).get("regime_transition")
        if isinstance(_rt, dict) and _rt.get("pre_hedge_signal"):
            _rt_risk = float(_rt.get("transition_risk_score", 0.0) or 0.0)
            _rt_factor = max(0.30, 1.0 - _rt_risk)
            size_pct *= _rt_factor
            sizing_method += f"+regime_transition*{_rt_factor:.2f}"
    except Exception:
        pass

    # P8: SelfDiagnosis — trend-based early warning
    try:
        _sd = (_cycle_advisory or {}).get("self_diagnosis")
        if isinstance(_sd, dict):
            _sd_critical = _sd.get("critical") or []
            _sd_warnings = _sd.get("warnings") or []
            if len(_sd_critical) > 0 and action == "BUY":
                return (0.0, "BLOCKED:self_diagnosis_critical")
            if len(_sd_warnings) >= 3:
                size_pct *= 0.70
                sizing_method += "+self_diag_warnings*0.70"
    except Exception:
        pass

    # ── Batch G: Wire previously-unused advisory keys into sizing ───────

    # G1: Ensemble composite -- multi-source signal strength
    # (on_cycle key: "ensemble" with sub-key "composite")
    try:
        _ens = (_cycle_advisory or {}).get("ensemble")
        if _ens and isinstance(_ens, dict):
            _composite = float(_ens.get("composite", 0.0) or 0.0)
            if _composite != 0.0:
                _stk_mult = max(0.50, min(1.30, 1.0 + _composite * 0.30))
                size_pct *= _stk_mult
                sizing_method += f"+ensemble_composite*{_stk_mult:.2f}"
    except Exception:
        pass

    # G2: Antifragile multiplier -- fragility-aware sizing
    # (on_cycle key: "antifragile_multiplier" as float scalar)
    try:
        _af_mult = (_cycle_advisory or {}).get("antifragile_multiplier")
        if _af_mult is not None:
            _af_mult = float(_af_mult)
            _af_mult = max(0.50, min(1.50, _af_mult))
            if _af_mult != 1.0:
                size_pct *= _af_mult
                sizing_method += f"+antifragile*{_af_mult:.2f}"
    except Exception:
        pass

    # G3: Bleeders -- halve size for strategies in losing streak
    # (on_cycle key: "bleeders" as LIST of dicts with "name" field)
    try:
        _bldr = (_cycle_advisory or {}).get("bleeders")
        if _bldr and isinstance(_bldr, list):
            _bldr_names = [str(b.get("name", "")) for b in _bldr if isinstance(b, dict)]
            if source_strategy in _bldr_names:
                size_pct *= 0.50
                sizing_method += "+bleeder*0.50"
    except Exception:
        pass

    # G4: TCA score -- high transaction costs reduce size
    # (on_cycle key: "tca_score" as float scalar, 0-100)
    try:
        _tca_score = (_cycle_advisory or {}).get("tca_score")
        if _tca_score is not None:
            _tca_score = float(_tca_score)
            if _tca_score > 50:
                _tca_mult = max(0.50, 1.0 - (_tca_score - 50) / 100.0)
                size_pct *= _tca_mult
                sizing_method += f"+tca*{_tca_mult:.2f}"
            # Phase Y: TCA > 70 → hint TWAP execution for slippage reduction
            if _tca_score > 70 and ctx is not None:
                ctx["_order_type_hint"] = "twap"
                sizing_method += "+tca_twap_hint"
    except Exception:
        pass

    # G5: System status -- CRITICAL blocks, DEGRADED reduces
    try:
        _sys = (_cycle_advisory or {}).get("system_status")
        if _sys and isinstance(_sys, dict):
            _status = str(_sys.get("overall", "HEALTHY")).upper()
            if _status == "CRITICAL" and action == "BUY":
                logger.warning("_execute_signals: system CRITICAL — blocking BUY %s", symbol)
                return (0.0, "BLOCKED:system_critical")
            elif _status == "DEGRADED":
                size_pct *= 0.50
                sizing_method += "+sys_degraded*0.70"
    except Exception:
        pass

    # G6: Whale activity -- on-chain flow direction
    try:
        _whale = (_cycle_advisory or {}).get("whale_activity")
        if _whale and isinstance(_whale, dict):
            _whale_bias = float(_whale.get("net_flow_bias", 0.0) or 0.0)
            if _whale_bias != 0.0:
                _whale_mult = max(0.70, min(1.30, 1.0 + _whale_bias * 0.25))
                size_pct *= _whale_mult
                sizing_method += f"+whale*{_whale_mult:.2f}"
    except Exception:
        pass

    # G7: Causal graph -- funding->vol->regime chain
    try:
        _cg = (_cycle_advisory or {}).get("causal_graph")
        if _cg and isinstance(_cg, dict):
            _cg_conf = float(_cg.get("confidence", 0.0) or 0.0)
            _cg_dir = str(_cg.get("direction", "neutral")).lower()
            if _cg_conf > 0.60:
                if _cg_dir == "bearish" and action == "BUY":
                    size_pct *= 0.70
                    sizing_method += "+causal_bearish*0.70"
                elif _cg_dir == "bullish" and action == "BUY":
                    size_pct *= 1.20
                    sizing_method += "+causal_bullish*1.20"
    except Exception:
        pass

    # G8: Outcome correlator -- reduce in historically unfavorable conditions
    try:
        _oc = (_cycle_advisory or {}).get("outcome_correlator")
        if _oc and isinstance(_oc, dict):
            _oc_score = float(_oc.get("favorability", 0.5) or 0.5)
            if _oc_score < 0.30:
                size_pct *= 0.60
                sizing_method += "+unfavorable*0.60"
            elif _oc_score > 0.70:
                size_pct *= 1.15
                sizing_method += "+favorable*1.15"
    except Exception:
        pass

    # G9: Alpha decay -- age signal by strategy-level decay factor
    try:
        _ad = (_cycle_advisory or {}).get("alpha_decay")
        if _ad and isinstance(_ad, dict):
            _decays = _ad.get("strategy_decays", {})
            if isinstance(_decays, dict) and source_strategy in _decays:
                _decay_factor = float(_decays[source_strategy])
                _decay_factor = max(0.30, min(1.0, _decay_factor))
                if _decay_factor < 1.0:
                    size_pct *= _decay_factor
                    sizing_method += f"+alpha_decay*{_decay_factor:.2f}"
    except Exception:
        pass

    # G10: Liquidation cascade -- boost/block based on cascade direction
    try:
        _lc = (_cycle_advisory or {}).get("liquidation_cascade")
        if _lc and isinstance(_lc, dict):
            _lc_signals = _lc.get("signals", [])
            for _lc_sig in (_lc_signals if isinstance(_lc_signals, list) else []):
                if isinstance(_lc_sig, dict) and _lc_sig.get("symbol") == symbol:
                    _lc_dir = str(_lc_sig.get("direction", "")).lower()
                    _lc_conf = float(_lc_sig.get("confidence", 0.0) or 0.0)
                    if _lc_dir == "long_squeeze" and action == "BUY" and _lc_conf > 0.60:
                        size_pct *= 0.40
                        sizing_method += "+lc_squeeze*0.40"
                    elif _lc_dir == "short_squeeze" and action == "BUY" and _lc_conf > 0.60:
                        size_pct *= 1.30
                        sizing_method += "+lc_short_squeeze*1.30"
                    # Phase Y: cascade detected → widen stops to avoid shakeout
                    if _lc_conf > 0.50 and ctx is not None:
                        ctx["_stop_widen_pct"] = max(
                            ctx.get("_stop_widen_pct", 0.0), 0.50
                        )
                        sizing_method += "+lc_stop_widen"
                    break
    except Exception:
        pass

    # Re-apply hard cap after batch G gates
    size_pct = min(size_pct, max_pos_pct)

    # ── Batch K: ML Intelligence gates ──────────────────────────────

    # K1: vol_forecasts -> scale by forecasted vol vs current
    try:
        _vf = (_cycle_advisory or {}).get("vol_forecasts")
        if _vf and isinstance(_vf, dict) and symbol in _vf:
            _sym_vf = _vf[symbol]
            if isinstance(_sym_vf, dict):
                _fvol = float(_sym_vf.get("forecast_vol_1d", 0.0) or 0.0)
                if _fvol > 0 and current_vol > 0:
                    _vol_ratio = _fvol / max(current_vol, 0.001)
                    _vf_mult = max(0.50, min(1.15, 1.0 - (_vol_ratio - 1.0) * 0.30))
                    if abs(_vf_mult - 1.0) > 0.01:
                        size_pct *= _vf_mult
                        sizing_method += f"+vol_forecast*{_vf_mult:.2f}"
    except Exception:
        pass

    # K2: alpha_scores -> alpha model direction weighting
    try:
        _als = (_cycle_advisory or {}).get("alpha_scores")
        if _als and isinstance(_als, dict) and symbol in _als:
            _sym_alpha = _als[symbol]
            if isinstance(_sym_alpha, dict):
                _alpha_comp = float(_sym_alpha.get("composite", 0.0) or 0.0)
                if abs(_alpha_comp) > 0.3:
                    _aligns = (_alpha_comp > 0 and action == "BUY") or (_alpha_comp < 0 and action == "SELL")
                    if _aligns:
                        size_pct *= 1.15
                        sizing_method += "+alpha_align*1.15"
                    else:
                        size_pct *= 0.70
                        sizing_method += "+alpha_oppose*0.70"
    except Exception:
        pass

    # K3: pretrained_regime -> ML regime gate
    try:
        _pr = (_cycle_advisory or {}).get("pretrained_regime")
        if _pr and isinstance(_pr, dict):
            _pr_pred = _pr.get("prediction")
            if _pr_pred is not None:
                _pr_label = str(_pr_pred[0] if hasattr(_pr_pred, '__len__') and len(_pr_pred) > 0 else _pr_pred).upper()
                if "CRISIS" in _pr_label and action == "BUY":
                    size_pct *= 0.50
                    sizing_method += "+ml_crisis*0.50"
                elif "TRENDING_UP" in _pr_label and action == "BUY":
                    size_pct *= 1.15
                    sizing_method += "+ml_trending_up*1.15"
    except Exception:
        pass

    # K4: pretrained_vol_forecast -> tighten sizing if vol predicted high
    try:
        _pvf = (_cycle_advisory or {}).get("pretrained_vol_forecast")
        if _pvf and isinstance(_pvf, dict):
            _next5d = float(_pvf.get("next_5d_vol", 0.0) or 0.0)
            if _next5d > 0.05:
                size_pct *= 0.80
                sizing_method += f"+high_vol_pred*0.80"
    except Exception:
        pass

    # K5: pretrained_alpha -> ML direction confidence
    try:
        _pa = (_cycle_advisory or {}).get("pretrained_alpha")
        if _pa and isinstance(_pa, dict):
            _pa_dir = str(_pa.get("direction", "")).upper()
            _pa_conf = float(_pa.get("confidence", 0.0) or 0.0)
            if _pa_conf > 0.70:
                _pa_aligns = (_pa_dir == "UP" and action == "BUY") or (_pa_dir == "DOWN" and action == "SELL")
                if _pa_aligns:
                    size_pct *= 1.20
                    sizing_method += "+ml_alpha_align*1.20"
                else:
                    size_pct *= 0.60
                    sizing_method += "+ml_alpha_oppose*0.60"
    except Exception:
        pass

    # K6: inference -> inference service confidence gate
    try:
        _inf = (_cycle_advisory or {}).get("inference")
        if _inf and isinstance(_inf, dict):
            _inf_conf = float(_inf.get("confidence", 0.5) or 0.5)
            if _inf_conf < 0.30:
                size_pct *= 0.70
                sizing_method += "+low_inference*0.70"
            elif _inf_conf > 0.70:
                size_pct *= 1.10
                sizing_method += "+high_inference*1.10"
    except Exception:
        pass

    # ── Batch L: Sentiment, Pattern & LLM ───────────────────────────

    # L1: fear_greed -> contrarian sizing
    try:
        _fg = (_cycle_advisory or {}).get("fear_greed")
        if _fg is not None:
            _fg_val = float(_fg)
            _fg_bias = (50.0 - _fg_val) / 250.0  # +/-0.20 range
            if action == "BUY":
                _fg_mult = 1.0 + _fg_bias
            else:
                _fg_mult = 1.0 - _fg_bias
            _fg_mult = max(0.80, min(1.20, _fg_mult))
            if abs(_fg_mult - 1.0) > 0.01:
                size_pct *= _fg_mult
                sizing_method += f"+fear_greed*{_fg_mult:.2f}"
    except Exception:
        pass

    # L2: llm_analysis -> LLM direction confidence
    try:
        _llm = (_cycle_advisory or {}).get("llm_analysis")
        if _llm and isinstance(_llm, dict):
            _llm_dir = str(_llm.get("direction", "")).upper()
            _llm_conf = float(_llm.get("confidence", 0.0) or 0.0)
            if _llm_conf > 0.60:
                _llm_aligns = (_llm_dir in ("UP", "BULLISH", "BUY") and action == "BUY") or \
                              (_llm_dir in ("DOWN", "BEARISH", "SELL") and action == "SELL")
                if _llm_aligns:
                    size_pct *= 1.10
                    sizing_method += "+llm_align*1.10"
                elif _llm_dir not in ("NEUTRAL", "UNKNOWN", ""):
                    size_pct *= 0.90
                    sizing_method += "+llm_oppose*0.90"
    except Exception:
        pass

    # L3: sentiment_stats -> aggregate sentiment bias
    try:
        _sent = (_cycle_advisory or {}).get("sentiment_stats")
        if _sent and isinstance(_sent, dict):
            _sent_score = float(_sent.get("avg_sentiment", 0.0) or 0.0)
            if _sent_score != 0.0:
                _sent_aligns = (_sent_score > 0 and action == "BUY") or (_sent_score < 0 and action == "SELL")
                if _sent_aligns:
                    size_pct *= 1.10
                    sizing_method += "+sentiment_align*1.10"
                else:
                    size_pct *= 0.90
                    sizing_method += "+sentiment_oppose*0.90"
    except Exception:
        pass

    # L4: chart_patterns -> pattern confirmation
    try:
        _cp = (_cycle_advisory or {}).get("chart_patterns")
        if _cp and isinstance(_cp, dict):
            _cp_bias = float(_cp.get("bias", 0.0) or 0.0)
            _cp_conf = float(_cp.get("confidence", 0.0) or 0.0)
            if _cp_conf > 0.60 and abs(_cp_bias) > 0.1:
                _cp_aligns = (_cp_bias > 0 and action == "BUY") or (_cp_bias < 0 and action == "SELL")
                if _cp_aligns:
                    size_pct *= 1.10
                    sizing_method += "+pattern_confirm*1.10"
                else:
                    size_pct *= 0.75
                    sizing_method += "+pattern_oppose*0.75"
    except Exception:
        pass

    # ── Batch M: Quantum Intelligence ────────────────────────────────

    # M1: quantum_prediction -> price direction
    try:
        _qp = (_cycle_advisory or {}).get("quantum_prediction")
        if _qp and isinstance(_qp, dict):
            _qp_next = float(_qp.get("next_value", 0.0) or 0.0)
            _qp_conf = float(_qp.get("confidence", 0.0) or 0.0)
            if _qp_next > 0 and _qp_conf > 0.50 and entry_price > 0:
                _qp_pct = (_qp_next - entry_price) / entry_price
                if _qp_pct > 0.001 and action == "BUY":
                    size_pct *= 1.10
                    sizing_method += "+qpred_up*1.10"
                elif _qp_pct < -0.001 and action == "BUY":
                    size_pct *= 0.85
                    sizing_method += "+qpred_down*0.85"
    except Exception:
        pass

    # M2: quantum_regime -> regime confirmation
    try:
        _qr = (_cycle_advisory or {}).get("quantum_regime")
        if _qr and isinstance(_qr, dict):
            _qr_entropy = float(_qr.get("entropy", 0.0) or 0.0)
            if _qr_entropy > 0.80:
                size_pct *= 0.90
                sizing_method += "+q_high_entropy*0.90"
    except Exception:
        pass

    # M3: quantum_anomaly_score -> circuit breaker
    try:
        _qa = (_cycle_advisory or {}).get("quantum_anomaly_score")
        if _qa is not None:
            _qa_val = float(_qa)
            if _qa_val > 0.90 and action == "BUY":
                logger.warning("_execute_signals: quantum anomaly %.2f > 0.90 — blocking BUY %s", _qa_val, symbol)
                return (0.0, "BLOCKED:quantum_anomaly")
            elif _qa_val > 0.75:
                size_pct *= 0.60
                sizing_method += f"+q_anomaly*0.60"
    except Exception:
        pass

    # M4: quantum_signal_quality -> quality filter
    try:
        _qsq = (_cycle_advisory or {}).get("quantum_signal_quality")
        if _qsq and isinstance(_qsq, dict):
            _q_quality = float(_qsq.get("quality", 0.5) or 0.5)
            if _q_quality > 0.80:
                size_pct *= 1.15
                sizing_method += "+high_q_quality*1.15"
            elif _q_quality < 0.30:
                size_pct *= 0.70
                sizing_method += "+low_q_quality*0.70"
    except Exception:
        pass

    # M5: quantum_portfolio -> bias toward optimal weights
    try:
        _qpw = (_cycle_advisory or {}).get("quantum_portfolio")
        if _qpw and isinstance(_qpw, dict) and _qpw.get("method") != "insufficient_data":
            _qp_weights = _qpw.get("weights")
            if _qp_weights and hasattr(_qp_weights, '__len__') and len(_qp_weights) > 0:
                # Advisory only for now -- log optimal weights
                sizing_method += "+qportfolio_available"
    except Exception:
        pass

    # M6: quantum_risk_check -> VaR gate
    try:
        _qrc = (_cycle_advisory or {}).get("quantum_risk_check")
        if _qrc and isinstance(_qrc, dict):
            _q_cvar = float(_qrc.get("cvar_95", 0.0) or 0.0)
            if _q_cvar > 0.05:
                size_pct *= 0.75
                sizing_method += f"+q_var_high*0.75"
    except Exception:
        pass

    # ── Batch N: Portfolio & Risk ────────────────────────────────────

    # N1: correlation_penalty -> reduce correlated positions
    try:
        _corr = (_cycle_advisory or {}).get("correlation_penalty")
        if _corr is not None:
            _corr_val = float(_corr)
            if _corr_val > 0.30:
                _corr_mult = max(0.50, 1.0 - _corr_val)
                size_pct *= _corr_mult
                sizing_method += f"+corr_penalty*{_corr_mult:.2f}"
    except Exception:
        pass

    # N2: tail_hedge -> reduce exposure when hedging recommended
    try:
        _th = (_cycle_advisory or {}).get("tail_hedge")
        if _th and isinstance(_th, dict):
            if _th.get("should_hedge") is True and action == "BUY":
                size_pct *= 0.70
                sizing_method += "+tail_hedge*0.70"
    except Exception:
        pass

    # N3: stress_test -> reduce if stress test flagged
    try:
        _st = (_cycle_advisory or {}).get("stress_test")
        if _st is not None and _st != "ok":
            size_pct *= 0.80
            sizing_method += "+stress_warning*0.80"
    except Exception:
        pass

    # N4: adaptive_risk -> enforce adjusted limits
    try:
        _ar = (_cycle_advisory or {}).get("adaptive_risk")
        if _ar and isinstance(_ar, dict):
            _ar_max = float(_ar.get("max_position_pct", 0.0) or 0.0)
            if _ar_max > 0 and size_pct > _ar_max:
                size_pct = _ar_max
                sizing_method += f"+adaptive_cap={_ar_max:.3f}"
    except Exception:
        pass

    # N5: risk_score -> portfolio risk gate
    try:
        _rs = (_cycle_advisory or {}).get("risk_score")
        if _rs is not None:
            _rs_val = float(_rs)
            if _rs_val > 0.95 and action == "BUY":
                logger.warning("_execute_signals: risk_score %.2f > 0.95 — blocking BUY %s", _rs_val, symbol)
                return (0.0, "BLOCKED:risk_score_extreme")
            elif _rs_val > 0.80:
                size_pct *= 0.70
                sizing_method += "+high_risk*0.70"
    except Exception:
        pass

    # N6: market_anomaly -> anomaly circuit breaker
    try:
        _ma = (_cycle_advisory or {}).get("market_anomaly")
        if _ma and isinstance(_ma, dict) and _ma.get("is_anomaly"):
            _ma_sev = str(_ma.get("severity", "low")).lower()
            if _ma_sev == "high" and action == "BUY":
                logger.warning("_execute_signals: HIGH anomaly detected — blocking BUY %s", symbol)
                return (0.0, "BLOCKED:market_anomaly_high")
            elif _ma_sev == "medium":
                size_pct *= 0.70
                sizing_method += "+anomaly_medium*0.70"
    except Exception:
        pass

    # ── Batch O: Advanced AI + Strategy Intelligence ─────────────────

    # O1: gnn_asset_flow -> lead-lag sizing
    try:
        _gnn = (_cycle_advisory or {}).get("gnn_asset_flow")
        if _gnn and isinstance(_gnn, dict) and symbol in _gnn:
            _flow = _gnn[symbol]
            if isinstance(_flow, dict):
                _flow_sig = float(_flow.get("flow_signal", 0.0) or 0.0)
                if _flow_sig > 0 and action == "BUY":
                    size_pct *= 1.10
                    sizing_method += "+gnn_positive*1.10"
                elif _flow_sig < 0 and action == "BUY":
                    size_pct *= 0.85
                    sizing_method += "+gnn_negative*0.85"
    except Exception:
        pass

    # O2: autoencoder_regime -> transition caution
    try:
        _ae = (_cycle_advisory or {}).get("autoencoder_regime")
        if _ae and isinstance(_ae, dict):
            if _ae.get("is_transition") is True:
                size_pct *= 0.75
                sizing_method += "+ae_transition*0.75"
    except Exception:
        pass

    # O3: rl_portfolio_allocation -> RL weight modifier
    try:
        _rl = (_cycle_advisory or {}).get("rl_portfolio_allocation")
        if _rl and isinstance(_rl, dict):
            _rl_weight = float(_rl.get(symbol, 0.0) or 0.0)
            if _rl_weight > 0:
                _rl_mult = max(0.50, min(1.50, _rl_weight / max(size_pct, 0.001)))
                _rl_mult = max(0.85, min(1.15, _rl_mult))  # cap influence to +/-15%
                if abs(_rl_mult - 1.0) > 0.01:
                    size_pct *= _rl_mult
                    sizing_method += f"+rl_weight*{_rl_mult:.2f}"
    except Exception:
        pass

    # O4: attention_orderflow -> direction confirmation
    try:
        _attn = (_cycle_advisory or {}).get("attention_orderflow")
        if _attn and isinstance(_attn, dict):
            _attn_dir = str(_attn.get("direction", "")).upper()
            _attn_conf = float(_attn.get("confidence", 0.0) or 0.0)
            if _attn_conf > 0.60:
                _attn_aligns = (_attn_dir in ("UP", "BUY") and action == "BUY") or \
                               (_attn_dir in ("DOWN", "SELL") and action == "SELL")
                if _attn_aligns:
                    size_pct *= 1.15
                    sizing_method += "+attn_align*1.15"
                elif _attn_dir not in ("NEUTRAL", ""):
                    size_pct *= 0.80
                    sizing_method += "+attn_oppose*0.80"
    except Exception:
        pass

    # O5: regime_rotation -> strategy weights per regime
    try:
        _rr = (_cycle_advisory or {}).get("regime_rotation")
        if _rr and isinstance(_rr, dict):
            _rr_weights = _rr.get("strategy_weights", {})
            if isinstance(_rr_weights, dict) and source_strategy in _rr_weights:
                _rr_w = float(_rr_weights[source_strategy])
                _rr_w = max(0.30, min(1.50, _rr_w))
                if abs(_rr_w - 1.0) > 0.01:
                    size_pct *= _rr_w
                    sizing_method += f"+regime_rot*{_rr_w:.2f}"
    except Exception:
        pass

    # O6: regime_prediction -> next regime forecast
    try:
        _rp = (_cycle_advisory or {}).get("regime_prediction")
        if _rp and isinstance(_rp, dict):
            _rp_next = str(_rp.get("predicted_regime", _rp.get("next_regime", ""))).upper()
            if "CRISIS" in _rp_next and action == "BUY":
                size_pct *= 0.70
                sizing_method += "+pred_crisis*0.70"
    except Exception:
        pass

    # O7: regime_pre_transition_signals -> pre-transition warning
    try:
        _rpt = (_cycle_advisory or {}).get("regime_pre_transition_signals")
        if _rpt:
            size_pct *= 0.80
            sizing_method += "+pre_transition*0.80"
    except Exception:
        pass

    # O8: funding_prediction -> predicted funding rate
    try:
        _fp = (_cycle_advisory or {}).get("funding_prediction")
        if _fp and isinstance(_fp, dict):
            _fp_rate = float(_fp.get("predicted_rate_pct", 0.0) or 0.0)
            if _fp_rate < -0.03 and action == "BUY":
                size_pct *= 0.85
                sizing_method += "+neg_funding*0.85"
            elif _fp_rate > 0.03 and action == "SELL":
                size_pct *= 0.85
                sizing_method += "+pos_funding*0.85"
    except Exception:
        pass

    # O9: session_effect -> time-of-day sizing bias
    try:
        _se = (_cycle_advisory or {}).get("session_effect")
        if _se is not None:
            _se_val = float(_se) if not isinstance(_se, dict) else float(_se.get("bias", 0.0) or 0.0)
            _se_mult = max(0.85, min(1.15, 1.0 + _se_val * 0.15))
            if abs(_se_mult - 1.0) > 0.01:
                size_pct *= _se_mult
                sizing_method += f"+session*{_se_mult:.2f}"
    except Exception:
        pass

    # O10: bandit_rankings -> strategy allocation weighting
    try:
        _br = (_cycle_advisory or {}).get("bandit_rankings")
        if _br and isinstance(_br, list):
            for _br_entry in _br:
                if isinstance(_br_entry, dict) and _br_entry.get("strategy") == source_strategy:
                    _br_wr = float(_br_entry.get("expected_win_rate", 0.5) or 0.5)
                    if _br_wr < 0.40:
                        size_pct *= 0.75
                        sizing_method += "+bandit_low*0.75"
                    elif _br_wr > 0.60:
                        size_pct *= 1.15
                        sizing_method += "+bandit_high*1.15"
                    break
    except Exception:
        pass

    # O11: orderbook_prediction -> OBI direction confirmation
    try:
        _obp = (_cycle_advisory or {}).get("orderbook_prediction")
        if _obp and isinstance(_obp, dict):
            _obp_dir = str(_obp.get("direction", "")).upper()
            _obp_conf = float(_obp.get("confidence", 0.0) or 0.0)
            if _obp_conf > 0.60:
                _obp_opposes = (_obp_dir in ("DOWN", "SELL") and action == "BUY") or \
                               (_obp_dir in ("UP", "BUY") and action == "SELL")
                if _obp_opposes:
                    size_pct *= 0.80
                    sizing_method += "+obi_oppose*0.80"
    except Exception:
        pass

    # ── Wire remaining advisory keys ────────────────────────────────

    # online_learner drift -> reduce sizing when model drift detected
    try:
        _ol = (_cycle_advisory or {}).get("online_learner")
        if _ol and isinstance(_ol, dict) and _ol.get("drift_detected"):
            _drift_mag = float(_ol.get("drift_magnitude", 0.0) or 0.0)
            _drift_mult = max(0.50, 1.0 - _drift_mag * 0.50)
            size_pct *= _drift_mult
            sizing_method += f"+ol_drift*{_drift_mult:.2f}"
    except Exception:
        pass

    # genetic_evolver -> use evolved position scale if available
    try:
        _ge = (_cycle_advisory or {}).get("genetic_evolver")
        if _ge and isinstance(_ge, dict):
            _ge_best = _ge.get("best_fitness", 0.0)
            if isinstance(_ge_best, (int, float)) and _ge_best > 0:
                _ge_scale = _ge.get("best_params", {})
                if isinstance(_ge_scale, dict):
                    _ge_pos = float(_ge_scale.get("position_scale", 1.0) or 1.0)
                    _ge_pos = max(0.50, min(1.50, _ge_pos))
                    if abs(_ge_pos - 1.0) > 0.01:
                        size_pct *= _ge_pos
                        sizing_method += f"+evolved*{_ge_pos:.2f}"
    except Exception:
        pass

    # strategy_optimization -> reduce sizing for strategies flagged for param adjustment
    try:
        _so = (_cycle_advisory or {}).get("strategy_optimization")
        if _so and isinstance(_so, dict):
            _so_strats = _so.get("needs_adjustment", [])
            if isinstance(_so_strats, list) and source_strategy in _so_strats:
                size_pct *= 0.80
                sizing_method += "+needs_optim*0.80"
    except Exception:
        pass

    # feature_discovery -> boost if new high-IC features discovered
    try:
        _fd = (_cycle_advisory or {}).get("feature_discovery")
        if _fd and isinstance(_fd, dict):
            _fd_count = int(_fd.get("total_discovered", 0) or 0)
            if _fd_count > 10:
                size_pct *= 1.05  # slight boost -- new features = more signal
                sizing_method += "+feature_rich*1.05"
    except Exception:
        pass

    # ── Strategy scanner -- boost symbols the scanner recommends ──
    try:
        _scan = (_cycle_advisory or {}).get("strategy_scanner")
        if _scan and isinstance(_scan, dict):
            _top_syms = _scan.get("top_symbols", [])
            _top_strats = _scan.get("top_strategies", [])
            if isinstance(_top_syms, list) and symbol in _top_syms:
                # This symbol is in the scanner's top opportunities
                _scan_boost = 1.20
                # Check if the strategy also matches
                for _ts in (_top_strats if isinstance(_top_strats, list) else []):
                    if isinstance(_ts, dict) and _ts.get("symbol") == symbol:
                        _ts_sharpe = float(_ts.get("sharpe", 0) or 0)
                        if _ts_sharpe > 0.5:
                            _scan_boost = 1.30  # strong scanner match
                        break
                size_pct *= _scan_boost
                sizing_method += f"+scanner_top*{_scan_boost:.2f}"
            elif isinstance(_top_syms, list) and _top_syms and symbol not in _top_syms:
                # Scanner has recommendations but this symbol isn't in them
                size_pct *= 0.70
                sizing_method += "+scanner_not_top*0.70"
    except Exception:
        pass

    # ── Market impact estimation -- reduce size if impact too high ──
    try:
        _mim = getattr(self.component_registry, "market_impact", None) if self.component_registry else None
        if _mim is not None and hasattr(_mim, "estimate") and entry_price > 0:
            _est_usd = ctx["portfolio_value"] * size_pct * ctx["aud_to_usd"]
            _impact = _mim.estimate(
                symbol=symbol, side=action.lower(),
                quantity_usd=_est_usd, price=entry_price,
            )
            _impact_bps = float(getattr(_impact, "total_impact_bps", 0.0) or 0.0)
            _impact_threshold = 15.0  # bps
            if _impact_bps > _impact_threshold:
                _impact_mult = max(0.30, 1.0 - (_impact_bps - _impact_threshold) / 100.0)
                size_pct *= _impact_mult
                sizing_method += f"+mkt_impact*{_impact_mult:.2f}({_impact_bps:.1f}bps)"
    except Exception:
        pass

    # Re-apply hard cap after all gates
    size_pct = min(size_pct, max_pos_pct)

    # FIX #25: NaN/Inf guard -- prevent corrupted sizing from propagating
    import math as _math_check
    if not isinstance(size_pct, (int, float)) or _math_check.isnan(size_pct) or _math_check.isinf(size_pct):
        logger.error("_execute_signals: size_pct corrupted to %s for %s — using 1%% default", size_pct, symbol)
        size_pct = 0.01
    elif size_pct < 0:
        size_pct = 0.0

    # PEAK: Dynamic gate floor — shrinks as more gates fire warnings
    # Count how many gate tags were added (proxy for risk accumulation)
    _gate_tags = sizing_method.count("+")
    if _gate_tags <= 3:
        _gate_floor = max_pos_pct * 0.20  # few warnings → generous floor (20%)
    elif _gate_tags <= 8:
        _gate_floor = max_pos_pct * 0.12  # moderate warnings → standard floor
    else:
        _gate_floor = max_pos_pct * 0.05  # many warnings → tight floor (5%)
    if size_pct > 0 and size_pct < _gate_floor:
        size_pct = _gate_floor

    # PEAK: Conviction bonus — reward high-confirmation signals with size boost
    # If confidence > 0.80 AND the signal has 3+ confirmations, boost size 20%
    _sig_conf = float(sig_fields.get("confidence", 0) or 0)
    _sig_str = float(sig_fields.get("strength", 0) or 0)
    if _sig_conf >= 0.80 and _sig_str >= 0.70:
        _conviction_boost = min(1.30, 1.0 + (_sig_conf - 0.80) * 1.5)  # up to 1.30x
        size_pct *= _conviction_boost
        sizing_method += f"+conviction*{_conviction_boost:.2f}"
    size_pct = min(size_pct, max_pos_pct)  # re-cap after boost

    # ── Latency compensator: adjust for 280ms AU -> Kraken RTT ──
    try:
        _lc = getattr(self.component_registry, "latency_compensator", None) if self.component_registry else None
        if _lc is not None:
            _sig_obj = sig_fields.get("_sig_obj")
            _sig_ts = float(getattr(_sig_obj, "timestamp", 0) or 0)
            if _sig_ts == 0:
                _sig_ts = time.time() * 1000
            elif _sig_ts < 1e12:  # seconds not ms
                _sig_ts *= 1000
            _comp = _lc.compensate(_sig_ts, current_vol)
            if _comp.is_stale:
                logger.debug("SKIP stale signal for %s (age=%.0fms)", symbol, _comp.stale_age_ms)
                return (0.0, "BLOCKED:latency_stale")
            size_pct *= _comp.size_multiplier
    except Exception:
        pass

    # ── Manipulation detector: BLOCK trades on manipulated symbols ──
    try:
        _manip = (_cycle_advisory or {}).get("manipulation_detector", {})
        if isinstance(_manip, dict):
            _blocked = _manip.get("blocked_symbols", [])
            if isinstance(_blocked, list) and symbol in _blocked:
                logger.warning("BLOCKED %s %s — manipulation detected", action, symbol)
                return (0.0, "BLOCKED:manipulation_detected")
    except Exception:
        pass

    # ── Portfolio risk: respect allocation limits per strategy ──
    try:
        _pr = (_cycle_advisory or {}).get("portfolio_risk", {})
        if isinstance(_pr, dict):
            _allocs = _pr.get("allocations", {})
            if isinstance(_allocs, dict) and source_strategy in _allocs:
                _target_alloc = float(_allocs[source_strategy])
                if _target_alloc < 0.05:
                    size_pct *= 0.3  # strategy has minimal allocation
                elif _target_alloc < 0.15:
                    size_pct *= 0.7
            _warnings = _pr.get("warnings", [])
            if isinstance(_warnings, list) and len(_warnings) >= 2:
                size_pct *= 0.8  # multiple risk warnings
    except Exception:
        pass

    # ── ML feedback: reduce confidence for drifting models ──
    try:
        _mlf = (_cycle_advisory or {}).get("ml_feedback", {})
        if isinstance(_mlf, dict):
            _drifting = _mlf.get("drifting", [])
            if isinstance(_drifting, list) and _drifting:
                size_pct *= 0.7  # some models are drifting
    except Exception:
        pass

    # ── Strategy attribution: boost winners, reduce losers ──
    try:
        _attr = (_cycle_advisory or {}).get("strategy_attribution", {})
        if isinstance(_attr, dict):
            _top = str(_attr.get("top_contributor", "") or "")
            _worst = str(_attr.get("worst_contributor", "") or "")
            if source_strategy == _top and _top:
                size_pct *= 1.2  # this strategy is the top performer
            elif source_strategy == _worst and _worst:
                size_pct *= 0.6  # this strategy is the worst performer
    except Exception:
        pass

    # ── Causal engine: adjust based on predicted downstream effects ──
    try:
        _ce = getattr(self.component_registry, "causal_engine", None) if self.component_registry else None
        if _ce is not None and action == "BUY":
            _latest_regime_label = getattr(self, "_latest_regime_label", "normal")
            _effects = _ce.predict_effects(f"regime_{str(_latest_regime_label or 'normal')}")
            for _eff_name, _eff_prob, _eff_lag in _effects:
                if "dump" in _eff_name and _eff_prob > 0.3:
                    size_pct *= max(0.5, 1.0 - _eff_prob)
                elif "squeeze" in _eff_name and _eff_prob > 0.3:
                    size_pct *= min(1.5, 1.0 + _eff_prob * 0.5)
    except Exception:
        pass

    # ── Counterfactual: override if we have a systematic bias ──
    try:
        _cf = getattr(self.component_registry, "counterfactual", None) if self.component_registry else None
        if _cf is not None:
            _latest_regime_label = getattr(self, "_latest_regime_label", "normal")
            _regime_str = str(_latest_regime_label or "normal")
            _override = _cf.should_override(action, _regime_str)
            if _override == "SKIP" and action == "BUY":
                size_pct *= 0.3  # strong bias says we're usually wrong here
    except Exception:
        pass

    # ── Meta-cognition: skip trades when ARGUS knows it doesn't know ──
    try:
        _mc_adv = (_cycle_advisory or {}).get("meta_cognition", {})
        if isinstance(_mc_adv, dict):
            _mc_rec = str(_mc_adv.get("recommendation", "TRADE") or "TRADE")
            _mc_conf = float(_mc_adv.get("confidence", 1.0) or 1.0)
            if _mc_rec == "SKIP" and action == "BUY":
                size_pct = 0.0
            elif _mc_rec == "WAIT" and action == "BUY":
                size_pct *= 0.3
            elif _mc_rec == "REDUCE_SIZE":
                size_pct *= max(0.5, _mc_conf)
    except Exception:
        pass

    # ── Temporal abstraction: boost when all time scales align ──
    try:
        _ta_adv = (_cycle_advisory or {}).get("temporal_abstraction", {})
        if isinstance(_ta_adv, dict) and symbol in _ta_adv:
            _ta = _ta_adv[symbol]
            _alignment = float(_ta.get("alignment", 0.5) or 0.5)
            _macro = float(_ta.get("macro", 0) or 0)
            if action == "BUY" and _alignment > 0.75 and _macro > 0:
                size_pct *= 1.2  # all scales agree bullish
            elif action == "BUY" and _macro < -0.5:
                size_pct *= 0.6  # macro trend is against us
    except Exception:
        pass

    # ── Universal Data Brain: 17-source market intelligence ──
    try:
        _udb_adv = (_cycle_advisory or {}).get("universal_data_brain", {})
        if isinstance(_udb_adv, dict) and symbol in _udb_adv:
            _intel = _udb_adv[symbol]
            _composite = float(_intel.get("composite", 0) or 0)
            _conviction = str(_intel.get("conviction", "LOW") or "LOW")
            _n_signals = int(_intel.get("signals", 0) or 0)
            if _n_signals >= 5:
                if action == "BUY" and _composite > 0.3 and _conviction in ("HIGH", "EXTREME"):
                    size_pct *= 1.3  # strong multi-source agreement = size up
                elif action == "BUY" and _composite < -0.3:
                    size_pct *= 0.5  # market intelligence says bearish
    except Exception:
        pass

    # ── Price prediction: boost/reduce based on predicted direction ──
    try:
        _pred_adv = (_cycle_advisory or {}).get("price_predictions", {})
        if isinstance(_pred_adv, dict) and symbol in _pred_adv:
            _pred = _pred_adv[symbol]
            _pred_dir = _pred.get("direction", "FLAT")
            _pred_conf = float(_pred.get("confidence", 0) or 0)
            _pred_agree = int(_pred.get("models_agree", 0) or 0)
            if action == "BUY" and _pred_dir == "UP" and _pred_agree >= 2:
                size_pct *= (1 + _pred_conf * 0.3)  # boost up to 30%
            elif action == "BUY" and _pred_dir == "DOWN" and _pred_agree >= 2:
                size_pct *= max(0.5, 1 - _pred_conf * 0.4)  # reduce up to 40%
    except Exception:
        pass

    # ── Entropy filter: suppress new positions when market is pure noise ──
    try:
        _ef_adv = (_cycle_advisory or {}).get("entropy_filter", {})
        if isinstance(_ef_adv, dict) and not _ef_adv.get("should_trade", True):
            if action == "BUY":
                size_pct *= 0.3  # reduce heavily but don't block entirely
    except Exception:
        pass

    # ── Market memory: adjust size based on similar historical outcomes ──
    try:
        _mm_adv = (_cycle_advisory or {}).get("market_memory", {})
        if isinstance(_mm_adv, dict) and _mm_adv.get("similar_count", 0) >= 5:
            _exp_pnl = float(_mm_adv.get("expected_pnl", 0) or 0)
            _similarity = float(_mm_adv.get("similarity", 0) or 0)
            if _exp_pnl < -0.5 and _similarity > 0.3:
                size_pct *= 0.5  # similar past conditions lost money
            elif _exp_pnl > 0.5 and _similarity > 0.3:
                size_pct *= 1.2  # similar past conditions made money
    except Exception:
        pass

    # ── Conviction-based sizing: size UP on high-conviction setups ──
    try:
        _cs = getattr(self.component_registry, "conviction_sizer", None) if self.component_registry else None
        if _cs is not None and size_pct > 0:
            _mtf_bias = getattr(self, "_mtf_bias", None) or {}
            _latest_regime_label = getattr(self, "_latest_regime_label", "NORMAL")
            _conv = _cs.compute(
                base_size_pct=size_pct, symbol=symbol, action=action,
                strategy_type=source_strategy,
                advisory=_cycle_advisory, mtf_bias=_mtf_bias,
                regime=str(_latest_regime_label or "NORMAL"),
                max_pos_pct=max_pos_pct,
            )
            size_pct = _conv.final_size_pct
            if _conv.conviction_score > 0.7:
                logger.info("Conviction %.2f (%.1fx) for %s %s: %s",
                            _conv.conviction_score, _conv.multiplier,
                            action, symbol, _conv.sources)
    except Exception:
        pass

    # ── Phase C4: Quantum portfolio weight consumer ─────────────────────
    # Read advisory["quantum_portfolio"]["weights_by_symbol"] (written in
    # component_registry.on_cycle every 100 cycles by _quantum_portfolio_optimizer)
    # OR self._quantum_portfolio_weights (written in unified_trading_system.py
    # by optimize_portfolio_with_quantum every 10 cycles).
    # Multiply size_pct by the symbol's normalized weight when enabled.
    try:
        if getattr(self.config, "use_quantum_portfolio_weights", False):
            _qp_weight: float | None = None
            # Path 1: live ARGUSQuantumSimulator weights
            _qp_dict = getattr(self, "_quantum_portfolio_weights", None)
            if _qp_dict and isinstance(_qp_dict, dict):
                _qp_weight = _qp_dict.get(symbol)
            # Path 2: on_cycle advisory weights
            if _qp_weight is None:
                _qp_adv = (_cycle_advisory or {}).get("quantum_portfolio")
                if _qp_adv and isinstance(_qp_adv, dict):
                    _qp_by_sym = _qp_adv.get("weights_by_symbol")
                    if _qp_by_sym and isinstance(_qp_by_sym, dict):
                        _qp_weight = _qp_by_sym.get(symbol)
            if _qp_weight is not None:
                _qp_weight = float(_qp_weight)
                # Normalize: target weight is in [0, 1]; we use 2x as the
                # multiplier when at the equal-weight baseline (1/N) and
                # scale linearly. Cap at [0.3, 1.5].
                # Use a multiplier of weight * n where n = number of assets, so
                # the average weight (1/n) gives 1.0x, double-overweight 2x, etc.
                _qp_full = (_cycle_advisory or {}).get("quantum_portfolio") or {}
                _n_assets = max(int(_qp_full.get("n_assets", 1)), 1)
                _qp_mult = max(0.3, min(1.5, _qp_weight * _n_assets))
                if abs(_qp_mult - 1.0) > 0.01:
                    size_pct *= _qp_mult
                    sizing_method += f"+qportfolio*{_qp_mult:.2f}"
    except Exception:
        pass

    # ── Phase C5: Quantum annealer strategy selection mask ──────────────
    # If the annealer has deselected this strategy for the current cycle,
    # block the trade entirely.
    try:
        _strat_mask = getattr(self, "_strategy_active_mask", None)
        if _strat_mask is not None and source_strategy is not None:
            if str(source_strategy) not in _strat_mask:
                return (0.0, "BLOCKED:quantum_annealer_deselected")
    except Exception:
        pass

    return (size_pct, sizing_method)


# ─────────────────────────────────────────────────────────────────────
# (f) _compute_stops_and_quantity
# ─────────────────────────────────────────────────────────────────────

def _compute_stops_and_quantity(self, sig_fields: dict, size_pct: float, ctx: dict) -> dict | None:
    """
    Compute stop-loss, take-profit, position value, and quantity from the
    sized position percentage.

    Returns a dict with keys:
        stop_loss, take_profit, quantity, position_value_aud,
        position_value_usd, sizing_method_suffix

    Returns None if position is too small or quantity is zero.
    When too small, returns a dict with {"_too_small": True, "result": {...}}
    so the caller can append the result and continue.
    """
    symbol = sig_fields["symbol"]
    action = sig_fields["action"]
    entry_price = sig_fields["entry_price"]
    stop_loss = sig_fields["stop_loss"]
    take_profit = sig_fields["take_profit"]

    regime_stop_mult = ctx["regime_stop_mult"]
    regime_tp_mult = ctx["regime_tp_mult"]
    portfolio_value = ctx["portfolio_value"]
    aud_to_usd = ctx["aud_to_usd"]

    min_pos_aud = float(getattr(self.config, "min_position_size_aud", 10.0) or 10.0)

    current_vol = 0.0
    try:
        current_vol = self._get_current_vol(symbol) or 0.0
    except Exception:
        pass

    sizing_method_suffix = ""

    # --- 2e. Dynamic stop/TP: regime + volatility-adjusted exits ---
    _base_stop_pct = float(getattr(self.config, "stop_loss_pct", 0.02) or 0.02)
    _base_tp_pct = float(getattr(self.config, "take_profit_pct", 0.04) or 0.04)
    _adj_stop_pct = _base_stop_pct * regime_stop_mult
    _adj_tp_pct = _base_tp_pct * regime_tp_mult

    # Volatility-adjusted exits: ATR-based when vol data available
    if current_vol > 0:
        # Use 1.5x vol as stop, 3.0x vol as TP (maintains ~2:1 R:R)
        _atr_stop = current_vol * 1.5
        _atr_tp = current_vol * 3.0
        # Blend: use ATR if it's meaningfully different from base
        if _atr_stop > 0.001:
            _adj_stop_pct = _atr_stop
            _adj_tp_pct = _atr_tp
            sizing_method_suffix += f"+atr_exit(sl={_adj_stop_pct:.4f},tp={_adj_tp_pct:.4f})"

    # Apply the computed stop/TP to the signal
    if stop_loss is None and entry_price > 0:
        if action == "BUY":
            stop_loss = entry_price * (1.0 - _adj_stop_pct)
        else:
            stop_loss = entry_price * (1.0 + _adj_stop_pct)
    if take_profit is None and entry_price > 0:
        if action == "BUY":
            take_profit = entry_price * (1.0 + _adj_tp_pct)
        else:
            take_profit = entry_price * (1.0 - _adj_tp_pct)

    position_value_aud = portfolio_value * size_pct
    if position_value_aud < min_pos_aud:
        logger.debug(
            "_execute_signals: position too small for %s (%.2f AUD < %.2f AUD min)",
            symbol, position_value_aud, min_pos_aud,
        )
        return {
            "_too_small": True,
            "result": {
                "symbol": symbol,
                "side": action,
                "status": "skipped",
                "reason": "position_too_small",
            },
        }

    # Convert to base currency quantity
    position_value_usd = position_value_aud * aud_to_usd
    quantity = position_value_usd / entry_price if entry_price > 0 else 0.0
    if quantity <= 0:
        return None

    return {
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "quantity": quantity,
        "position_value_aud": position_value_aud,
        "position_value_usd": position_value_usd,
        "sizing_method_suffix": sizing_method_suffix,
    }


# ─────────────────────────────────────────────────────────────────────
# (g) _log_cycle_summary
# ─────────────────────────────────────────────────────────────────────

def _log_cycle_summary(self, results: list, ctx: dict) -> None:
    """
    Log a structured one-line summary of the execution cycle:
        "Cycle N: X signals -> Y passed -> Z filled | +$P&L | regime=R | blocked: reason1:count reason2:count"
    """
    regime = ctx.get("regime", "UNKNOWN")
    cycle_num = int(getattr(self, "_cycle_number", 0) or 0)

    total = len(results)
    filled = sum(1 for r in results if r.get("status") == "filled")
    blocked = sum(1 for r in results if r.get("status") in ("blocked", "skipped"))
    errors = sum(1 for r in results if r.get("status") == "error")

    # Aggregate block reasons
    block_reasons: dict = {}
    for r in results:
        if r.get("status") in ("blocked", "skipped"):
            reason = r.get("reason", "unknown")
            if isinstance(reason, list):
                reason = ",".join(str(x) for x in reason)
            # Normalise to first token for grouping
            key = str(reason).split(":")[0].split("(")[0].strip()
            block_reasons[key] = block_reasons.get(key, 0) + 1

    block_str = " ".join(f"{k}:{v}" for k, v in sorted(block_reasons.items(), key=lambda x: -x[1]))

    # Approximate cycle P&L from filled trades
    pnl_str = ""
    try:
        _daily_pnl = float(getattr(self, "daily_pnl", 0.0) or 0.0)
        pnl_str = f" | pnl=${_daily_pnl:+.2f}"
    except Exception:
        pass

    logger.info(
        "Cycle %d: %d signals -> %d passed -> %d filled%s | regime=%s%s",
        cycle_num,
        total,
        total - blocked - errors,
        filled,
        pnl_str,
        regime,
        f" | blocked: {block_str}" if block_str else "",
    )
