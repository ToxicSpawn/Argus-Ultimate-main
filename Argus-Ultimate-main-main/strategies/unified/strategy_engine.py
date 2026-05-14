"""
Canonical StrategyEngine for the Unified System.

This is intentionally conservative and production-oriented:
- Uses real OHLCV via MarketDataService.
- Generates a small number of high-confidence signals.
- Avoids overtrading (especially in small-capital mode).
"""

from __future__ import annotations

import asyncio
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from unified_types import TradingSignal

from adaptive.online_tuner import OnlineStrategyTuner
from adaptive.regime import MarketRegime, RegimeDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategySignal:
    symbol: str
    action: str
    confidence: float
    strength: float
    entry_price: float
    reasoning: str


class StrategyEngine:
    def __init__(self, config: Any) -> None:
        self.config = config
        self._opt_params: Dict[str, Any] = {}
        self._load_opt_params()

        # Sentiment engine — feeds volatility + momentum readings into fear/greed
        try:
            from alpha.sentiment.sentiment_engine import SentimentEngine
            self._sentiment = SentimentEngine()
        except Exception:
            self._sentiment = None

        # Adaptive components (dependency-light)
        self._adaptive_enabled = bool(getattr(config, "adaptive_enabled", True))
        minutes_per_bar = float(getattr(config, "adaptive_minutes_per_bar", 60.0) or 60.0)
        self._regime = RegimeDetector(minutes_per_bar=minutes_per_bar)
        self._tuner = OnlineStrategyTuner(
            alpha=float(getattr(config, "adaptive_tuner_alpha", 0.15) or 0.15),
            min_trades_before_bias=int(getattr(config, "adaptive_min_trades_before_bias", 3) or 3),
        )
        self._last_regime: Dict[str, MarketRegime] = {}
        self._entry_mode: Dict[str, str] = {}
        # Drawdown-adaptive confidence
        self._current_drawdown_pct: float = 0.0
        self._max_drawdown_limit: float = float(
            (getattr(config, "risk", None) or {}).get("max_drawdown_pct", 0.12)
            if isinstance(getattr(config, "risk", None), dict)
            else getattr(config, "max_drawdown_pct", 0.12) or 0.12
        )
        self._dd_adaptive_enabled: bool = bool(getattr(config, "dd_adaptive_confidence_enabled", False))
        self._dd_floor_conf: float = float(getattr(config, "dd_adaptive_conf_floor", 0.50) or 0.50)
        self._dd_ceiling_conf: float = float(getattr(config, "dd_adaptive_conf_ceiling", 0.85) or 0.85)

        # Signal intelligence state
        self._signal_accuracy_tracker: Dict[str, Dict[str, Any]] = {}  # "symbol:action" -> {"wins": int, "total": int}
        self._signal_birth_cycle: Dict[str, int] = {}  # signal_key -> cycle when first generated
        self._global_cycle: int = 0

    def update_drawdown(self, current_drawdown_pct: float) -> None:
        """Update the current drawdown ratio (0.0 = no drawdown, 0.12 = 12%).
        Used by drawdown-adaptive confidence to auto-adjust signal threshold."""
        self._current_drawdown_pct = max(0.0, float(current_drawdown_pct))

    def _load_opt_params(self) -> None:
        """
        Load optional tuned parameters for `unified_engine` from config dicts.
        Supports:
        - config.optimized_params_by_timeframe[timeframe]["unified_engine"]["best_params"]
        - config.optimized_params["unified_engine"]["best_params"]
        - same, but with params directly (no "best_params" wrapper)
        """
        try:
            # Optional: load from file if requested and not already present.
            try:
                if (
                    not getattr(self.config, "optimized_params_by_timeframe", None)
                    and bool(getattr(self.config, "optimized_params_load", False))
                ):
                    p = str(getattr(self.config, "optimized_params_path", "data/optimized_params.json") or "").strip()
                    if p:
                        data = json.loads(Path(p).read_text(encoding="utf-8"))
                        if isinstance(data, dict):
                            if isinstance(data.get("timeframes"), dict):
                                setattr(self.config, "optimized_params_by_timeframe", data.get("timeframes"))
                            elif isinstance(data.get("params"), dict):
                                tf0 = str(data.get("timeframe", "") or "")
                                setattr(self.config, "optimized_params_by_timeframe", {tf0: data.get("params")})
            except Exception as _e:
                logger.warning("StrategyEngine: failed to load optimized params from %s: %s", p, _e)

            opt = None
            by_tf = getattr(self.config, "optimized_params_by_timeframe", None)
            tf = str(getattr(self.config, "optimized_params_timeframe", "") or "")
            if isinstance(by_tf, dict) and tf and isinstance(by_tf.get(tf), dict):
                opt = by_tf.get(tf)
            if opt is None:
                opt = getattr(self.config, "optimized_params", None)
            if not isinstance(opt, dict):
                return
            ue = opt.get("unified_engine") or opt.get("UnifiedEngine") or opt.get("unified") or opt.get("unified_engine_v1")
            if isinstance(ue, dict) and isinstance(ue.get("best_params"), dict):
                ue = ue.get("best_params")
            if isinstance(ue, dict):
                self._opt_params = dict(ue)
        except Exception as _e:
            logger.warning("StrategyEngine: error applying optimized params: %s", _e)
            return

    def on_realized_pnl(self, *, symbol: str, pnl_pct: float) -> None:
        """
        Feedback hook: call this when a trade is closed (realized PnL known).
        Used to adapt thresholds/weights online.
        """
        sym = str(symbol)
        if not self._adaptive_enabled:
            return
        reg = self._last_regime.get(sym, MarketRegime.RANGE)
        mode = self._entry_mode.get(sym) or (
            "trend" if reg in (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN) else "mean_reversion"
        )
        self._tuner.record_trade(symbol=sym, regime=reg, mode=mode, pnl_pct=float(pnl_pct))

    def get_adaptation_status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self._adaptive_enabled),
            "last_regime": {k: str(v.value) for k, v in self._last_regime.items()},
            "tuner": self._tuner.status(),
        }

    async def generate_signals(
        self,
        market_data_service: Any,
        timeframe_override: Optional[str] = None,
        strategy_suffix: Optional[str] = None,
    ) -> List[TradingSignal]:
        """
        Generate TradingSignal objects for all configured trading pairs.
        timeframe_override: e.g. "1m", "15m", "1h" for multi-timeframe; None = use config or "1m".
        strategy_suffix: appended to strategy name (e.g. "_1h") for multi-TF attribution.
        """
        # Hot-reload optimized params (lets the system evolve without restart).
        self._load_opt_params()

        symbols = []
        try:
            # Scanner can set a per-scan window to rotate through full universe
            window = getattr(self.config, "_scanner_symbol_window", None)
            if window is not None and isinstance(window, (list, tuple)) and len(window) > 0:
                symbols = list(window)
            else:
                symbols = list(getattr(self.config, "trading_pairs", []) or [])
        except Exception:
            symbols = []
        if not symbols:
            symbols = ["BTC/USD"]
        


        # In paper/backtest, allow more signals; in live use higher bar when set (push beyond).
        min_conf_cfg = float(getattr(self.config, "min_signal_confidence", 0.75) or 0.75)
        mode = str(getattr(self.config, "run_mode", "paper") or "paper").lower()
        if mode == "live":
            live_min = getattr(self.config, "live_min_signal_confidence", None)
            if live_min is not None and float(live_min) > 0:
                min_conf_cfg = float(live_min)
        # Drawdown-adaptive: interpolate between floor and ceiling based on current DD
        if self._dd_adaptive_enabled:
            dd_ratio = self._current_drawdown_pct / max(self._max_drawdown_limit, 1e-9)
            dd_ratio = max(0.0, min(1.0, dd_ratio))
            min_conf = self._dd_floor_conf + (self._dd_ceiling_conf - self._dd_floor_conf) * dd_ratio
        else:
            min_conf = float(min_conf_cfg)
        max_signals = int(getattr(self.config, "max_concurrent_signals", 2) or 2)

        out: List[TradingSignal] = []
        tf = (timeframe_override or getattr(self.config, "signal_primary_timeframe", "1h"))
        if isinstance(tf, (int, float)):
            tf = "1m" if tf == 1 else f"{int(tf)}m"

        # Rate limit protection: only process 8 pairs per cycle (rotate through full list)
        # This prevents Kraken rate limiting which causes 240s timeout cycles
        _max_per_cycle = 8
        if not hasattr(self, "_symbol_rotation_idx"):
            self._symbol_rotation_idx = 0
        if len(symbols) > _max_per_cycle:
            # Always include BTC and ETH (Tier 1), rotate the rest
            _tier1 = [s for s in symbols if s in ("BTC/USD", "ETH/USD")]
            _rest = [s for s in symbols if s not in ("BTC/USD", "ETH/USD")]
            _remaining_slots = _max_per_cycle - len(_tier1)
            _start = self._symbol_rotation_idx % max(len(_rest), 1)
            _rotated = _rest[_start:_start + _remaining_slots]
            if len(_rotated) < _remaining_slots:
                _rotated += _rest[:_remaining_slots - len(_rotated)]
            symbols = _tier1 + _rotated
            self._symbol_rotation_idx += _remaining_slots

        # Process all symbols in parallel (was sequential — saves 50%+ on cycle time)
        async def _process_sym(sym):
            try:
                return await self._signal_for_symbol(sym, market_data_service, timeframe=str(tf))
            except Exception as e:
                logger.debug("StrategyEngine failed for %s: %s", sym, e)
                return []
        logger.info(f"StrategyEngine: processing {len(symbols)} symbols with min_conf={min_conf:.2f}")
        _all_signals = await asyncio.gather(*[_process_sym(s) for s in symbols], return_exceptions=True)
        for signals in _all_signals:
            if isinstance(signals, Exception) or signals is None:
                continue
            for sig in signals:
                if sig and sig.confidence >= min_conf:
                    if strategy_suffix and hasattr(sig, "strategy"):
                        setattr(sig, "strategy", (getattr(sig, "strategy", "unified_engine") or "unified_engine") + strategy_suffix)
                    out.append(sig)
        logger.debug(f"StrategyEngine: generated {len(out)} signals (min_conf={min_conf:.2f})")

        # Increment global cycle counter (used for signal decay)
        self._global_cycle += 1

        # Resolve conflicting signals: when multiple strategies disagree on
        # the same symbol, use regime to break the tie before deduplication.
        if len(out) > 1:
            out = self._resolve_conflicting_signals(out)

        # Deduplicate: keep only the best signal per symbol (prevents 5x over-concentration)
        best_per_symbol: Dict[str, TradingSignal] = {}
        for sig in out:
            sym = str(sig.symbol)
            score = float(sig.confidence) * float(sig.strength)
            existing = best_per_symbol.get(sym)
            if existing is None or score > float(existing.confidence) * float(existing.strength):
                best_per_symbol[sym] = sig
        out = list(best_per_symbol.values())

        # ── Inject scanner/evolver-discovered opportunities ──
        try:
            _scanner_adv = getattr(self.config, "_last_cycle_advisory", None) or {}
            if isinstance(_scanner_adv, dict):
                _scanner_adv = _scanner_adv.get("strategy_scanner", {})
            _evolver_adv = getattr(self.config, "_last_cycle_advisory", None) or {}
            if isinstance(_evolver_adv, dict):
                _evolver_adv = _evolver_adv.get("strategy_evolver", {})

            # Boost confidence of signals that match scanner's top recommendations
            _scan_best_sym = str(_scanner_adv.get("best_symbol", "") if isinstance(_scanner_adv, dict) else "")
            _scan_best_type = str(_scanner_adv.get("best_strategy", "") if isinstance(_scanner_adv, dict) else "")
            _evo_best_sym = str(_evolver_adv.get("best_symbol", "") if isinstance(_evolver_adv, dict) else "")
            for sig in out:
                _sym = str(sig.symbol)
                # Scanner-recommended symbol gets +10% confidence boost
                if _scan_best_sym and _sym == _scan_best_sym:
                    sig = sig._replace(confidence=min(0.99, float(sig.confidence) * 1.10)) if hasattr(sig, '_replace') else sig
                # Evolver-recommended symbol gets +5% boost
                if _evo_best_sym and _sym == _evo_best_sym:
                    sig = sig._replace(confidence=min(0.99, float(sig.confidence) * 1.05)) if hasattr(sig, '_replace') else sig
        except Exception:
            pass

        # Rank by confidence * strength and cap
        max_signals = max(max_signals, 6)
        out.sort(key=lambda s: float(s.confidence) * float(s.strength), reverse=True)
        return out[: max(1, max_signals)]

    async def _signal_for_symbol(
        self, symbol: str, market_data_service: Any, timeframe: str = "1m"
    ) -> List[TradingSignal]:
        # Get OHLCV for indicator calculations (multi-timeframe: pass 1h, 15m, etc.)
        df = await market_data_service.fetch_ohlcv_df(symbol, timeframe=timeframe, limit=200)
        if df is None or df.empty or "close" not in df.columns:
            # Fallback to ticker-only signal generation: no trade (production-safe)
            await market_data_service.fetch_ticker(symbol)
            return []

        # Minimum bars required: MACD slow(26) + signal(9) + buffer = 40 bars
        _MIN_BARS_FOR_INDICATORS = 40
        if len(df) < _MIN_BARS_FOR_INDICATORS:
            logger.debug(
                "Skipping %s: only %d bars (need %d for reliable indicators)",
                symbol, len(df), _MIN_BARS_FOR_INDICATORS,
            )
            return []

        df = df.copy()
        close = df["close"].astype(float)

        snap = self._regime.detect(df) if self._adaptive_enabled else None
        regime = snap.regime if snap is not None else MarketRegime.RANGE
        self._last_regime[str(symbol)] = regime

        rsi = self._rsi(close, period=14)
        macd_hist = self._macd_hist(close)
        bb_pos = self._bollinger_position(close, period=20, std=2.0)

        # New indicators: ADX (trend strength), VWAP, divergence
        adx_series = self._adx(df, period=14) if "high" in df.columns else None
        last_adx = float(adx_series.iloc[-1]) if adx_series is not None else 20.0
        vwap_series = self._vwap(df) if "volume" in df.columns and "high" in df.columns else None
        last_vwap = float(vwap_series.iloc[-1]) if vwap_series is not None and not vwap_series.empty else None
        divergence = self._detect_divergence(close, rsi, lookback=20)

        last_price = float(close.iloc[-1])
        last_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50.0
        last_macd = float(macd_hist.iloc[-1]) if not macd_hist.empty else 0.0
        last_bbpos = float(bb_pos.iloc[-1]) if not bb_pos.empty else 0.5

        # ── MULTI-TIMEFRAME CONFIRMATION ──────────────────────────────
        # Check if 1h and 4h timeframes agree with this signal direction.
        # If higher timeframe disagrees, reduce confidence by 50%.
        mtf_discount = 1.0  # 1.0 = no discount
        mtf_reasons: List[str] = []
        try:
            htf_checks = {"1h": None, "4h": None}
            # Only fetch MTF data every 5 cycles to save API calls
            if not hasattr(self, "_mtf_cache"):
                self._mtf_cache = {}
            _mtf_cache_key = f"{symbol}_{self._global_cycle}"
            _use_cached_mtf = (self._global_cycle % 5 != 0) and f"{symbol}_{self._global_cycle - (self._global_cycle % 5)}" in self._mtf_cache
            if _use_cached_mtf:
                htf_checks = dict(self._mtf_cache.get(f"{symbol}_{self._global_cycle - (self._global_cycle % 5)}", {}))
            else:
              for htf in htf_checks:
                if timeframe == htf:
                    continue
                htf_df = await market_data_service.fetch_ohlcv_df(symbol, timeframe=htf, limit=60)
                if htf_df is not None and not htf_df.empty and "close" in htf_df.columns and len(htf_df) >= 30:
                    htf_close = htf_df["close"].astype(float)
                    htf_rsi = self._rsi(htf_close, period=14)
                    htf_macd = self._macd_hist(htf_close)
                    htf_rsi_val = float(htf_rsi.iloc[-1]) if not htf_rsi.empty else 50.0
                    htf_macd_val = float(htf_macd.iloc[-1]) if not htf_macd.empty else 0.0
                    # Determine higher-TF bias: bullish if RSI > 50 and MACD > 0
                    if htf_rsi_val > 55 and htf_macd_val > 0:
                        htf_checks[htf] = "BUY"
                    elif htf_rsi_val < 45 and htf_macd_val < 0:
                        htf_checks[htf] = "SELL"
                    else:
                        htf_checks[htf] = "NEUTRAL"
              self._mtf_cache[_mtf_cache_key] = dict(htf_checks)
            # Count disagreements with eventual signal direction (applied after action decided)
            self._mtf_bias = htf_checks  # Store for later use
        except Exception as _mtf_err:
            logger.debug("MTF confirmation failed for %s: %s", symbol, _mtf_err)
            self._mtf_bias = {}

        # ── VOLUME CONFIRMATION ───────────────────────────────────────
        # Track whether current volume is above the 20-period average
        volume_confirmed = False
        volume_ratio = 1.0
        if "volume" in df.columns:
            vol_series = df["volume"].astype(float)
            vol_sma_20 = vol_series.rolling(20).mean()
            if not pd.isna(vol_sma_20.iloc[-1]) and float(vol_sma_20.iloc[-1]) > 0:
                volume_ratio = float(vol_series.iloc[-1]) / float(vol_sma_20.iloc[-1])
                volume_confirmed = volume_ratio >= 1.0

        # ── TREND FILTER (SMA 50 + SMA 200) ──────────────────────────
        # Strong downtrend: only SELL. Strong uptrend: only BUY. Ranging: both.
        sma_50 = close.rolling(50).mean() if len(close) >= 50 else None
        sma_200 = close.rolling(200).mean() if len(close) >= 200 else None
        trend_filter = "RANGING"  # default: allow both directions
        if sma_50 is not None and sma_200 is not None:
            last_sma50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None
            last_sma200 = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None
            if last_sma50 is not None and last_sma200 is not None:
                if last_price > last_sma50 and last_price > last_sma200:
                    trend_filter = "UPTREND"
                elif last_price < last_sma50 and last_price < last_sma200:
                    trend_filter = "DOWNTREND"

        # Optional order book confirmation
        ob_bias = 0.0
        ob = None
        try:
            ob = await market_data_service.fetch_order_book(symbol, limit=20)
            if isinstance(ob, dict):
                ob_bias = self._order_book_bias(ob)
        except Exception:
            ob_bias = 0.0

        # Conservative-but-usable rules:
        # - BUY: RSI oversold OR price near lower Bollinger band
        # - SELL: RSI overbought OR price near upper Bollinger band
        # Confirmations (MACD/order book) improve confidence.
        action = "HOLD"
        confidence = 0.0
        strength = 0.0
        reasons: List[str] = []
        _num_confirmations = 0  # Track for weighted confidence scoring

        # Slightly relaxed thresholds so paper/backtests generate trades.
        base_buy_rsi = float(getattr(self.config, "se_buy_rsi", self._opt_params.get("se_buy_rsi", 35.0)) or 35.0)
        base_sell_rsi = float(getattr(self.config, "se_sell_rsi", self._opt_params.get("se_sell_rsi", 65.0)) or 65.0)
        base_buy_bb = float(getattr(self.config, "se_buy_bb", self._opt_params.get("se_buy_bb", 0.30)) or 0.30)
        base_sell_bb = float(getattr(self.config, "se_sell_bb", self._opt_params.get("se_sell_bb", 0.70)) or 0.70)

        # Regime-aware bias:
        # - RANGE: mean-reversion entries are fine
        # - TREND: favor trend-following entries, make mean-reversion slightly more selective
        # - HIGH_VOL: be more selective overall
        selectivity = 0.0
        mode_hint = "mean_reversion"
        if regime in (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN):
            selectivity += 0.10
            mode_hint = "trend"
        if regime == MarketRegime.HIGH_VOL:
            selectivity += 0.18

        # Online tuner adjustment (based on realized PnL in this regime)
        adj = self._tuner.threshold_adjustments(symbol=str(symbol), regime=regime, mode=mode_hint)
        selectivity += float(adj.get("selectivity", 0.0) or 0.0)

        # ── REGIME-CONDITIONAL RSI THRESHOLDS ─────────────────────────
        # Shift buy/sell RSI thresholds based on regime to make entries
        # easier in trend direction and harder against it.
        if regime == MarketRegime.TREND_UP:
            base_buy_rsi = 40.0   # easier to trigger BUY in uptrend
            base_sell_rsi = 80.0  # harder to trigger SELL in uptrend
        elif regime == MarketRegime.TREND_DOWN:
            base_buy_rsi = 20.0   # harder to trigger BUY in downtrend
            base_sell_rsi = 60.0  # easier to trigger SELL in downtrend
        # RANGE / HIGH_VOL: keep original base values (already loaded from config)

        # Apply selectivity: make oversold/overbought thresholds more extreme, and BB edges tighter.
        buy_rsi = float(max(10.0, base_buy_rsi - (selectivity * 10.0)))
        sell_rsi = float(min(90.0, base_sell_rsi + (selectivity * 10.0)))
        buy_bb = float(max(0.05, base_buy_bb - (selectivity * 0.15)))
        sell_bb = float(min(0.95, base_sell_bb + (selectivity * 0.15)))

        if last_rsi < buy_rsi or last_bbpos < buy_bb:
            action = "BUY"
            if last_rsi < buy_rsi:
                confidence += min(0.55, (buy_rsi - last_rsi) / max(buy_rsi, 1e-9) * 0.55)
                strength += 0.45
                reasons.append(f"RSI low ({last_rsi:.1f})")
            if last_bbpos < buy_bb:
                confidence += min(0.45, (buy_bb - last_bbpos) / max(buy_bb, 1e-9) * 0.45)
                strength += 0.35
                reasons.append(f"BB low ({last_bbpos:.2f})")
        elif last_rsi > sell_rsi or last_bbpos > sell_bb:
            action = "SELL"
            if last_rsi > sell_rsi:
                confidence += min(0.55, (last_rsi - sell_rsi) / max(100.0 - sell_rsi, 1e-9) * 0.55)
                strength += 0.45
                reasons.append(f"RSI high ({last_rsi:.1f})")
            if last_bbpos > sell_bb:
                confidence += min(0.45, (last_bbpos - sell_bb) / max(1.0 - sell_bb, 1e-9) * 0.45)
                strength += 0.35
                reasons.append(f"BB high ({last_bbpos:.2f})")


        # Trend-following entry/exit (helps long-only systems actually enter).
        # If we didn't already trigger a mean-reversion action:
        if action == "HOLD":
            # Regime-aware trend thresholds
            trend_rsi_buy = float(
                getattr(self.config, "se_trend_rsi_buy", self._opt_params.get("se_trend_rsi_buy", 55.0)) or 55.0
            )
            trend_rsi_sell = float(
                getattr(self.config, "se_trend_rsi_sell", self._opt_params.get("se_trend_rsi_sell", 45.0)) or 45.0
            )
            trend_bb_buy = 0.55
            trend_bb_sell = 0.45
            if regime == MarketRegime.HIGH_VOL:
                # In very high vol, require a bit more confirmation
                trend_rsi_buy = 58.0
                trend_rsi_sell = 42.0
                trend_bb_buy = 0.60
                trend_bb_sell = 0.40

            # Trend entry: requires 2 of 3 confirmations (MACD + RSI + BB)
            _trend_buy_votes = int(last_macd > 0) + int(last_rsi >= trend_rsi_buy) + int(last_bbpos >= trend_bb_buy)
            _trend_sell_votes = int(last_macd < 0) + int(last_rsi <= trend_rsi_sell) + int(last_bbpos <= trend_bb_sell)
            if _trend_buy_votes >= 2:
                action = "BUY"
                confidence = 0.55 + 0.12 * _trend_buy_votes  # 0.67-0.91
                strength = 0.60 + 0.10 * _trend_buy_votes
                reasons.append(f"Trend buy ({_trend_buy_votes}/3 confirms: MACD={'+'if last_macd>0 else '-'}, RSI={last_rsi:.0f}, BB={last_bbpos:.2f})")
            elif _trend_sell_votes >= 2:
                action = "SELL"
                confidence = 0.55 + 0.12 * _trend_sell_votes
                strength = 0.60 + 0.10 * _trend_sell_votes
                reasons.append(f"Trend sell ({_trend_sell_votes}/3 confirms: MACD={'+'if last_macd>0 else '-'}, RSI={last_rsi:.0f}, BB={last_bbpos:.2f})")

        # ── TREND FILTER: reduce confidence against macro trend (don't block) ──
        # Previously this BLOCKED signals entirely — but that meant in uptrend
        # only mean-reversion sells could fire (which never happens). Now we
        # reduce confidence by 40% instead of blocking, so trend-following
        # entries still work and the gate cascade handles the rest.
        if action == "BUY" and trend_filter == "DOWNTREND":
            confidence *= 0.80
            strength *= 0.80
            reasons.append("Trend discount: BUY in downtrend (conf*0.80)")
        elif action == "SELL" and trend_filter == "UPTREND":
            confidence *= 0.80
            strength *= 0.80
            reasons.append("Trend discount: SELL in uptrend (conf*0.80)")

        # ── SMA CROSSOVER FALLBACK ─────────────────────────────────────
        # If no signal from mean-reversion or trend, check simple SMA20
        # crossover. This ensures at least one strategy fires in any market.
        if action == "HOLD" and sma_50 is not None:
            sma_20 = close.rolling(20).mean()
            if sma_20 is not None and len(sma_20) >= 2:
                prev_sma20 = float(sma_20.iloc[-2]) if not pd.isna(sma_20.iloc[-2]) else 0
                curr_sma20 = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else 0
                prev_close = float(close.iloc[-2])
                curr_close = float(close.iloc[-1])
                # PEAK: Dynamic confidence based on trend alignment
                _sma_base_conf = 0.55
                if trend_filter == "UPTREND":
                    _sma_buy_conf = 0.72  # strong in uptrend
                    _sma_sell_conf = 0.50  # weak against trend
                elif trend_filter == "DOWNTREND":
                    _sma_buy_conf = 0.50
                    _sma_sell_conf = 0.72
                else:
                    _sma_buy_conf = 0.62
                    _sma_sell_conf = 0.62

                if prev_close < prev_sma20 and curr_close > curr_sma20:
                    action = "BUY"
                    confidence = _sma_buy_conf
                    strength = _sma_buy_conf * 0.9
                    reasons.append(f"SMA20 bullish crossover ({curr_close:.0f} > {curr_sma20:.0f})")
                elif prev_close > prev_sma20 and curr_close < curr_sma20:
                    action = "SELL"
                    confidence = _sma_sell_conf
                    strength = _sma_sell_conf * 0.9
                    reasons.append(f"SMA20 bearish crossover ({curr_close:.0f} < {curr_sma20:.0f})")

        # Minimum actionable confidence: use config threshold, not hardcoded floor.
        if action in ("BUY", "SELL"):
            min_actionable = float(getattr(self.config, "se_min_actionable_confidence", 0.35) or 0.35)
            confidence = max(confidence, min_actionable)
            strength = max(strength, min_actionable * 0.8)
            _num_confirmations += 1  # Base signal counts as 1 confirmation

        # MACD histogram confirmation
        if action == "BUY" and last_macd > 0:
            confidence += 0.15
            strength += 0.18
            reasons.append("MACD bullish")
            _num_confirmations += 1
        if action == "SELL" and last_macd < 0:
            confidence += 0.15
            strength += 0.18
            reasons.append("MACD bearish")
            _num_confirmations += 1

        # ── VOLUME CONFIRMATION ───────────────────────────────────────
        if action in ("BUY", "SELL"):
            if volume_confirmed:
                confidence += 0.12
                strength += 0.10
                reasons.append(f"Volume confirmed ({volume_ratio:.1f}x avg)")
                _num_confirmations += 1
            else:
                # Low volume: small penalty (not 30% — too aggressive)
                confidence *= 0.90
                reasons.append(f"Low volume ({volume_ratio:.1f}x avg)")

        # ── TREND ALIGNMENT BONUS ─────────────────────────────────────
        if action in ("BUY", "SELL"):
            if (action == "BUY" and trend_filter == "UPTREND") or (action == "SELL" and trend_filter == "DOWNTREND"):
                confidence += 0.15
                strength += 0.10
                reasons.append(f"Trend aligned ({trend_filter})")
                _num_confirmations += 1

        # Order book bias (if available)
        if action == "BUY" and ob_bias > 0.1:
            confidence += 0.1
            reasons.append("Order book bid imbalance")
        if action == "SELL" and ob_bias < -0.1:
            confidence += 0.1
            reasons.append("Order book ask imbalance")

        # ── Divergence confirmation ──────────────────────────────────
        if action == "BUY" and divergence == "bullish":
            confidence += 0.12
            strength += 0.10
            reasons.append("Bullish RSI divergence")
        elif action == "SELL" and divergence == "bearish":
            confidence += 0.12
            strength += 0.10
            reasons.append("Bearish RSI divergence")
        # Contra-divergence penalty: PEAK stronger penalty (was 0.88)
        elif action == "BUY" and divergence == "bearish":
            confidence *= 0.75
            reasons.append("Bearish divergence opposes buy (*0.75)")
        elif action == "SELL" and divergence == "bullish":
            confidence *= 0.75  # PEAK: stronger contra-divergence penalty (was 0.88)
            reasons.append("Bullish divergence opposes sell (*0.75)")

        # ── ADX trend strength filter (PEAK: force HOLD in choppy markets) ─
        if action in ("BUY", "SELL") and last_adx is not None:
            if mode_hint == "trend" and last_adx >= 25.0:
                adx_boost = min(0.15, (last_adx - 25.0) / 50.0 * 0.15)  # PEAK: +0.15 max (was 0.10)
                confidence += adx_boost
                strength += adx_boost * 0.5
                reasons.append(f"ADX confirms trend ({last_adx:.0f})")
                _num_confirmations += 1
            elif mode_hint == "trend" and last_adx < 20.0:
                # PEAK: force HOLD in choppy market for trend signals (was just 0.90 penalty)
                action = "HOLD"
                confidence = 0.0
                reasons.append(f"ADX too weak for trend ({last_adx:.0f}) — forced HOLD")
            elif mode_hint == "mean_reversion" and last_adx < 20.0:
                confidence += 0.08  # PEAK: +0.08 (was 0.05)
                reasons.append(f"ADX supports mean-reversion ({last_adx:.0f})")
                _num_confirmations += 1
            elif mode_hint == "mean_reversion" and last_adx >= 30.0:
                # PEAK: penalize mean-reversion in strong trends
                confidence *= 0.85
                reasons.append(f"ADX too strong for mean-reversion ({last_adx:.0f})")

        # ── VWAP deviation signal (PEAK: wider threshold, higher cap) ────
        if action in ("BUY", "SELL") and last_vwap is not None and last_vwap > 0:
            vwap_dev = (last_price - last_vwap) / last_vwap
            if action == "BUY" and vwap_dev < -0.003:  # PEAK: tighter threshold (was -0.005)
                _vwap_boost = min(0.15, abs(vwap_dev) * 8.0)  # PEAK: +0.15 max (was 0.08)
                confidence += _vwap_boost
                strength += _vwap_boost * 0.5
                reasons.append(f"Price below VWAP ({vwap_dev:+.3f})")
                _num_confirmations += 1
            elif action == "SELL" and vwap_dev > 0.003:
                _vwap_boost = min(0.15, abs(vwap_dev) * 8.0)
                confidence += _vwap_boost
                strength += _vwap_boost * 0.5
                reasons.append(f"Price above VWAP ({vwap_dev:+.3f})")
                _num_confirmations += 1

        # ── Order book depth pressure (PEAK: higher cap) ────────────────
        if action in ("BUY", "SELL") and isinstance(ob, dict):
            try:
                ob_pressure = self._order_book_pressure(ob, depth=10)
                if action == "BUY" and ob_pressure > 0.10:  # PEAK: lower threshold (was 0.15)
                    _ob_boost = min(0.12, ob_pressure * 0.30)  # PEAK: +0.12 max (was 0.06)
                    confidence += _ob_boost
                    reasons.append(f"OB depth pressure bullish ({ob_pressure:+.2f})")
                    _num_confirmations += 1
                elif action == "SELL" and ob_pressure < -0.10:
                    _ob_boost = min(0.12, abs(ob_pressure) * 0.30)
                    confidence += _ob_boost
                    reasons.append(f"OB depth pressure bearish ({ob_pressure:+.2f})")
                    _num_confirmations += 1
            except Exception as _e:
                logger.debug("strategy_engine error: %s", _e)

        # ── MULTI-TIMEFRAME CONFIRMATION (apply discount) ──────────────
        if action in ("BUY", "SELL"):
            mtf_bias = getattr(self, "_mtf_bias", {})
            mtf_disagree = 0
            for htf_name, htf_dir in mtf_bias.items():
                if htf_dir is None or htf_dir == "NEUTRAL":
                    continue
                if htf_dir != action:
                    mtf_disagree += 1
                    reasons.append(f"MTF {htf_name} disagrees ({htf_dir})")
            if mtf_disagree > 0:
                # Each disagreeing timeframe reduces confidence by 25% (two = 50%)
                mtf_discount = max(0.50, 1.0 - mtf_disagree * 0.25)
                confidence *= mtf_discount
                reasons.append(f"MTF discount {mtf_discount:.0%}")
            elif any(v == action for v in mtf_bias.values() if v is not None):
                # Higher TF confirms — small boost
                confidence *= 1.08
                _num_confirmations += 1
                reasons.append("MTF confirms direction")

        # ══════════════════════════════════════════════════════════════════
        # PEAK SIGNALS — 12 additional signal sources for maximum alpha
        # ══════════════════════════════════════════════════════════════════

        # ── 1. Momentum ROC (Rate of Change over 10 and 20 bars) ──────
        if action in ("BUY", "SELL") and len(close) >= 20:
            roc_10 = (float(close.iloc[-1]) - float(close.iloc[-11])) / max(float(close.iloc[-11]), 1e-9)
            roc_20 = (float(close.iloc[-1]) - float(close.iloc[-21])) / max(float(close.iloc[-21]), 1e-9)
            if action == "BUY" and roc_10 > 0.01 and roc_20 > 0.005:
                confidence += 0.10
                strength += 0.08
                reasons.append(f"Momentum ROC+ (10={roc_10:.3f}, 20={roc_20:.3f})")
                _num_confirmations += 1
            elif action == "SELL" and roc_10 < -0.01 and roc_20 < -0.005:
                confidence += 0.10
                strength += 0.08
                reasons.append(f"Momentum ROC- (10={roc_10:.3f}, 20={roc_20:.3f})")
                _num_confirmations += 1

        # ── 2. Volume-weighted momentum ───────────────────────────────
        if action in ("BUY", "SELL") and "volume" in df.columns and len(close) >= 5:
            _vol_s = df["volume"].astype(float)
            _vol_avg = float(_vol_s.rolling(20).mean().iloc[-1]) if len(_vol_s) >= 20 else float(_vol_s.mean())
            _vol_now = float(_vol_s.iloc[-1])
            _price_chg = (float(close.iloc[-1]) - float(close.iloc[-2])) / max(float(close.iloc[-2]), 1e-9)
            _vol_mom = _price_chg * (_vol_now / max(_vol_avg, 1e-9))
            if action == "BUY" and _vol_mom > 0.01:
                confidence += min(0.10, _vol_mom * 3.0)
                reasons.append(f"Volume-weighted momentum+ ({_vol_mom:.3f})")
                _num_confirmations += 1
            elif action == "SELL" and _vol_mom < -0.01:
                confidence += min(0.10, abs(_vol_mom) * 3.0)
                reasons.append(f"Volume-weighted momentum- ({_vol_mom:.3f})")
                _num_confirmations += 1

        # ── 3. Market structure (higher-highs/higher-lows) ────────────
        if action in ("BUY", "SELL") and "high" in df.columns and "low" in df.columns and len(df) >= 10:
            _highs = df["high"].astype(float).values[-10:]
            _lows = df["low"].astype(float).values[-10:]
            _hh = _highs[-1] > _highs[-5] and _highs[-5] > _highs[-9]
            _hl = _lows[-1] > _lows[-5] and _lows[-5] > _lows[-9]
            _lh = _highs[-1] < _highs[-5] and _highs[-5] < _highs[-9]
            _ll = _lows[-1] < _lows[-5] and _lows[-5] < _lows[-9]
            if action == "BUY" and _hh and _hl:
                confidence += 0.10
                reasons.append("Market structure: HH+HL (uptrend)")
                _num_confirmations += 1
            elif action == "SELL" and _lh and _ll:
                confidence += 0.10
                reasons.append("Market structure: LH+LL (downtrend)")
                _num_confirmations += 1

        # ── 4. Session/time-of-day bias ──────────────────────────────
        import time as _time_mod
        _utc_hour = int(_time_mod.gmtime().tm_hour)
        # London open (7-9 UTC) = volatile, often trend start
        # NY open (13-15 UTC) = volatile, reversals common
        # Asian (0-5 UTC) = quiet, mean-reversion
        if action in ("BUY", "SELL"):
            if 7 <= _utc_hour <= 9 or 13 <= _utc_hour <= 15:
                confidence += 0.05
                reasons.append(f"Active session ({_utc_hour}h UTC)")
            elif 0 <= _utc_hour <= 5:
                if mode_hint == "mean_reversion":
                    confidence += 0.05
                    reasons.append(f"Asian session favors MR ({_utc_hour}h UTC)")

        # ── 5. Volatility regime entry ───────────────────────────────
        if action in ("BUY", "SELL") and regime is not None:
            if regime == MarketRegime.HIGH_VOL and action == "SELL":
                confidence += 0.08
                reasons.append("HIGH_VOL regime → sell bias")
            elif regime == MarketRegime.RANGE and mode_hint == "mean_reversion":
                confidence += 0.05
                reasons.append("RANGE regime → MR bias")

        # ── 6. Adaptive ATR-based stop distance ─────────────────────
        # (doesn't change confidence but adjusts the signal's stop_loss field)
        if action in ("BUY", "SELL") and "high" in df.columns and "low" in df.columns:
            try:
                _atr_series = self._atr(df, period=14) if hasattr(self, "_atr") else None
                if _atr_series is not None and not _atr_series.empty:
                    _atr_val = float(_atr_series.iloc[-1])
                    # Store ATR for stop-loss computation downstream
                    if not hasattr(self, "_last_atr"):
                        self._last_atr = {}
                    self._last_atr[symbol] = _atr_val
            except Exception:
                pass

        # ── CONVICTION-WEIGHTED CONFIDENCE (0.3-1.0 based on confirmations) ──
        if action in ("BUY", "SELL") and _num_confirmations > 0:
            # Map confirmations to confidence range:
            # 1 confirmation -> 0.35, 2 -> 0.50, 3 -> 0.65, 4 -> 0.80, 5+ -> 1.0
            conviction_floor = 0.30 + _num_confirmations * 0.14
            conviction_floor = min(1.0, conviction_floor)
            # Use the higher of computed confidence and conviction floor
            confidence = max(confidence, conviction_floor)

        # Online tuner: scale confidence/strength based on recent realized performance
        mult = self._tuner.confidence_multiplier(symbol=str(symbol), regime=regime, mode=mode_hint, base=1.0)
        confidence *= float(mult)
        strength *= float(mult)

        confidence = float(min(1.0, max(0.0, confidence)))
        strength = float(min(1.0, max(0.0, strength)))
        # Optional: regime/next-bar boost from recent closes (one better alpha source)
        if getattr(self.config, "use_regime_lstm_boost", False):
            try:
                from ml.regime_boost import apply_regime_boost
                confidence = apply_regime_boost(confidence, close.tolist(), lookback=20)
            except Exception as _e:
                logger.debug("strategy_engine error: %s", _e)

        # Optional: scale down confidence in very high volatility regime (realized vol from closes)
        if getattr(self.config, "use_volatility_regime_scale", False) and len(close) >= 10:
            try:
                rets = np.diff(close.values) / np.maximum(close.values[:-1], 1e-12)
                vol = float(np.std(rets))
                if vol > float(getattr(self.config, "volatility_regime_high_threshold", 0.02) or 0.02):
                    confidence *= 0.85
                    confidence = max(0.0, min(1.0, confidence))
            except Exception as _e:
                logger.debug("strategy_engine error: %s", _e)

        # ── Sentiment-adjusted confidence ──────────────────────────────
        if action in ("BUY", "SELL") and confidence > 0 and self._sentiment is not None:
            try:
                # Feed latest volatility + returns into sentiment engine
                if len(close) >= 10:
                    rets = close.pct_change().dropna()
                    vol_now = float(rets.iloc[-20:].std()) if len(rets) >= 20 else float(rets.std())
                    self._sentiment.on_volatility(vol_now)
                    self._sentiment.on_return(float(rets.iloc[-1]))
                sent_score = self._sentiment.fear_greed.get_score()  # -1 (fear) to 1 (greed)
                if action == "BUY" and sent_score > 0.3:
                    confidence *= 1.08  # bullish sentiment boost
                elif action == "BUY" and sent_score < -0.3:
                    confidence *= 0.90  # fearful market penalty
                elif action == "SELL" and sent_score < -0.3:
                    confidence *= 1.08  # fear confirms sell
                elif action == "SELL" and sent_score > 0.3:
                    confidence *= 0.90  # greed penalizes sell
                confidence = max(0.0, min(1.0, confidence))
            except Exception as _e:
                logger.debug("strategy_engine error: %s", _e)

        # ── Microstructure boost from order book ──────────────────────
        if action in ("BUY", "SELL") and confidence > 0:
            try:
                from alpha.order_book_signals import microstructure_boost
                # ob was fetched earlier for ob_bias
                if isinstance(ob, dict):
                    ms_boost = microstructure_boost(ob, spread_bps_max=20.0, flow_boost=0.1)
                    if action == "SELL":
                        # Invert: negative flow = better for sell
                        ms_boost = 2.0 - ms_boost  # flip around 1.0
                    confidence *= ms_boost
                    confidence = max(0.0, min(1.0, confidence))
            except Exception as _e:
                logger.debug("strategy_engine error: %s", _e)

        # ── 7. Fear/Greed index as contrarian signal ────────────────
        if action in ("BUY", "SELL"):
            try:
                _fg = getattr(self.config, "_last_cycle_advisory", None) or {}
                if isinstance(_fg, dict):
                    _fg_data = _fg.get("fear_greed")
                    if isinstance(_fg_data, dict):
                        _fg_val = float(_fg_data.get("value", 50) or 50)
                        # Contrarian: extreme fear (< 25) = buy, extreme greed (> 75) = sell
                        if action == "BUY" and _fg_val < 25:
                            confidence += 0.10
                            reasons.append(f"Extreme fear ({_fg_val:.0f}) → contrarian buy")
                            _num_confirmations += 1
                        elif action == "SELL" and _fg_val > 75:
                            confidence += 0.10
                            reasons.append(f"Extreme greed ({_fg_val:.0f}) → contrarian sell")
                            _num_confirmations += 1
            except Exception:
                pass

        # ── 8. Funding rate as directional signal ───────────────────
        if action in ("BUY", "SELL"):
            try:
                _adv = getattr(self.config, "_last_cycle_advisory", None) or {}
                _fund = _adv.get("funding_prediction") if isinstance(_adv, dict) else None
                if isinstance(_fund, dict):
                    _rate = float(_fund.get("predicted_rate_pct", 0.0) or 0.0)
                    # Positive funding → longs pay → bearish bias
                    if action == "SELL" and _rate > 0.01:
                        confidence += 0.08
                        reasons.append(f"Funding rate bullish pressure ({_rate:.4f}%) → sell")
                    elif action == "BUY" and _rate < -0.01:
                        confidence += 0.08
                        reasons.append(f"Funding rate bearish pressure ({_rate:.4f}%) → buy")
            except Exception:
                pass

        # ── ATR-based adaptive stop-loss and take-profit ────────────
        stop_loss_pct = float(getattr(self.config, "stop_loss_pct", 0.01) or 0.01)
        take_profit_pct = float(getattr(self.config, "take_profit_pct", 0.03) or 0.03)
        # Override with ATR-based stops if available (2x ATR for SL, 3x for TP)
        _sym_atr = getattr(self, "_last_atr", {}).get(symbol)
        if _sym_atr is not None and last_price > 0:
            _atr_pct = _sym_atr / last_price
            stop_loss_pct = max(0.005, min(0.05, _atr_pct * 2.0))
            take_profit_pct = max(0.01, min(0.10, _atr_pct * 3.0))
            reasons.append(f"ATR stops: SL={stop_loss_pct:.3%} TP={take_profit_pct:.3%}")
        out_signals: List[TradingSignal] = []

        # Optional: skip long when funding rate is above threshold (e.g. avoid paying high funding)
        if action == "BUY" and getattr(self.config, "use_funding_rate_filter", False):
            try:
                from ml.funding_rates import fetch_funding_rate, should_skip_long
                url = getattr(self.config, "funding_rates_url", None) or None
                rate = fetch_funding_rate(str(symbol), exchange_id=str(getattr(self.config, "primary_exchange", "kraken") or "kraken"), url_override=url)
                threshold = float(getattr(self.config, "funding_rate_skip_long_threshold", 0.0001) or 0.0001)
                if should_skip_long(symbol, rate, threshold=threshold):
                    logger.info(
                        "Funding rate filter: skipping BUY on %s — rate=%.6f >= threshold=%.6f (confidence was %.3f)",
                        symbol, float(rate or 0), threshold, confidence,
                    )
                    action = "HOLD"
                    confidence = 0.0
            except Exception as exc:
                logger.debug("Funding rate fetch failed for %s: %s", symbol, exc)

        # ── MOMENTUM IGNITION FILTER ─────────────────────────────────
        # Skip signal if price moved >2% in 3 candles without proportional
        # volume increase (likely manipulation / wash trading).
        if action in ("BUY", "SELL") and confidence > 0:
            if self._momentum_ignition_filter(df):
                logger.info(
                    "Momentum ignition filter: skipping %s on %s (conf=%.3f)",
                    action, symbol, confidence,
                )
                action = "HOLD"
                confidence = 0.0
                reasons.append("BLOCKED: momentum ignition (price/vol mismatch)")

        # ── SIGNAL QUALITY GATE ──────────────────────────────────────
        # Reject signals where historical accuracy for this pattern < 45%
        if action in ("BUY", "SELL") and confidence > 0:
            if not self._signal_quality_gate(symbol, action, confidence):
                action = "HOLD"
                confidence = 0.0
                reasons.append("BLOCKED: signal quality gate (win rate < 45%)")

        # ── BREAKOUT CONFIRMATION ────────────────────────────────────
        # For trend-following signals, require candle CLOSE above/below
        # Bollinger band, not just a wick. Prevents false breakout entries.
        if action in ("BUY", "SELL") and mode_hint == "trend" and confidence > 0:
            try:
                close_series = df["close"].astype(float)
                bb_sma = close_series.rolling(window=20).mean()
                bb_std = close_series.rolling(window=20).std()
                if not pd.isna(bb_sma.iloc[-1]) and not pd.isna(bb_std.iloc[-1]):
                    bb_upper_val = float(bb_sma.iloc[-1]) + 2.0 * float(bb_std.iloc[-1])
                    bb_lower_val = float(bb_sma.iloc[-1]) - 2.0 * float(bb_std.iloc[-1])
                    candle_close = float(close_series.iloc[-1])
                    if action == "BUY" and last_bbpos >= 0.85:
                        # Trend buy near upper band: require close above band
                        if candle_close <= bb_upper_val:
                            confidence *= 0.70  # penalize wick-only breakout
                            reasons.append("Breakout not confirmed (close <= BB upper)")
                    elif action == "SELL" and last_bbpos <= 0.15:
                        # Trend sell near lower band: require close below band
                        if candle_close >= bb_lower_val:
                            confidence *= 0.70
                            reasons.append("Breakout not confirmed (close >= BB lower)")
            except Exception as _bb_err:
                logger.debug("Breakout confirmation check failed: %s", _bb_err)

        if action in ("BUY", "SELL") and confidence > 0:
            # Track which mode produced the entry so we can learn from realized PnL.
            if action == "BUY":
                self._entry_mode[str(symbol)] = str(mode_hint)
            elif action == "SELL":
                # Reset entry mode on SELL so regime learning isn't biased by stale mode
                self._entry_mode.pop(str(symbol), None)
            if action == "BUY":
                stop_loss = last_price * (1 - stop_loss_pct)
                take_profit = last_price * (1 + take_profit_pct)
            else:
                stop_loss = last_price * (1 + stop_loss_pct)
                take_profit = last_price * (1 - take_profit_pct)
            sig = TradingSignal(
                symbol=str(symbol),
                action=action,
                confidence=confidence,
                strength=strength,
                entry_price=last_price,
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                reasoning="; ".join(reasons) if reasons else "StrategyEngine signal",
                agent_consensus=confidence,
            )
            setattr(sig, "strategy", "unified_engine")
            setattr(sig, "num_confirmations", _num_confirmations)
            setattr(sig, "volume_ratio", volume_ratio)
            setattr(sig, "trend_filter", trend_filter)

            # ── SCALED ENTRY LOGIC ───────────────────────────────────
            # Confidence tiers determine how aggressively to enter:
            #   > 0.7  -> full entry (100% of position size)
            #   0.5-0.7 -> scaled entry (50% now, 50% on pullback)
            #   < 0.5  -> wait mode (only enter on confirmation candle close)
            if confidence > 0.70:
                entry_mode_label = "full"
            elif confidence >= 0.50:
                entry_mode_label = "scaled"
            else:
                entry_mode_label = "wait"
            setattr(sig, "entry_mode", entry_mode_label)
            setattr(sig, "entry_scale_pct", 1.0 if entry_mode_label == "full" else 0.5 if entry_mode_label == "scaled" else 0.0)

            out_signals.append(sig)

            # Regime-aligned: only emit when direction matches regime (reduces wrong-regime trades)
            reg_sig = self._make_regime_aligned_signal(
                symbol=symbol,
                action=action,
                regime=regime,
                last_price=last_price,
                confidence=confidence,
                strength=strength,
                reasons=reasons,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            if reg_sig is not None:
                out_signals.append(reg_sig)

        # Volume-confirmed momentum: EMA cross + volume above average (reduces false breakouts)
        vol_sig = self._make_volume_momentum_signal(
            df=df,
            symbol=symbol,
            last_price=last_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        if vol_sig is not None:
            out_signals.append(vol_sig)

        # Mean reversion: RSI oversold/overbought only in RANGE regime
        mr_sig = self._make_mean_reversion_signal(
            symbol=symbol,
            regime=regime,
            last_rsi=last_rsi,
            last_price=last_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        if mr_sig is not None:
            out_signals.append(mr_sig)

        # Bollinger breakout: price at band extreme with volume confirmation
        bb_sig = self._make_breakout_bb_signal(
            df=df,
            symbol=symbol,
            last_price=last_price,
            last_bbpos=last_bbpos,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        if bb_sig is not None:
            out_signals.append(bb_sig)

        # MACD trend: trend regime + MACD direction confirmation (rising/falling)
        macd_sig = self._make_macd_trend_signal(
            symbol=symbol,
            regime=regime,
            macd_hist=macd_hist,
            last_price=last_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        if macd_sig is not None:
            out_signals.append(macd_sig)

        return out_signals

    # ── SIGNAL INTELLIGENCE ─────────────────────────────────────────

    def _signal_decay(self, signal_age_cycles: int, base_confidence: float) -> float:
        """Reduce confidence by 10% per cycle of age.

        A fresh signal (age 0) keeps full confidence. A 5-cycle-old signal
        retains only 50% of the original confidence. Clamps at 10% minimum
        so very stale signals are not entirely invisible to downstream logic.
        """
        decay_rate = 0.10  # 10% per cycle
        decay_factor = max(0.10, 1.0 - decay_rate * int(max(0, signal_age_cycles)))
        return float(base_confidence * decay_factor)

    def _resolve_conflicting_signals(self, signals: list) -> list:
        """When multiple strategies disagree on the same symbol, use the current
        regime to break the tie.

        * Trending regime (TREND_UP / TREND_DOWN): favour the trend direction.
        * Ranging regime (RANGE): favour mean-reversion (opposite of recent move).
        * HIGH_VOL: favour the signal with highest raw confidence.

        Returns a de-duplicated list with at most one signal per symbol.
        """
        from collections import defaultdict
        by_symbol: Dict[str, List[Any]] = defaultdict(list)
        for sig in signals:
            by_symbol[str(sig.symbol)].append(sig)

        resolved: list = []
        for sym, sym_sigs in by_symbol.items():
            actions = {str(getattr(s, "action", "")).upper() for s in sym_sigs}
            # No conflict — all agree (or single signal)
            if len(actions) <= 1:
                # Pick highest confidence
                best = max(sym_sigs, key=lambda s: float(s.confidence))
                resolved.append(best)
                continue

            regime = self._last_regime.get(sym, MarketRegime.RANGE)
            if regime == MarketRegime.TREND_UP:
                favoured = "BUY"
            elif regime == MarketRegime.TREND_DOWN:
                favoured = "SELL"
            elif regime == MarketRegime.RANGE:
                # Mean-reversion: favour opposite of majority direction
                buy_count = sum(1 for s in sym_sigs if str(getattr(s, "action", "")).upper() == "BUY")
                sell_count = sum(1 for s in sym_sigs if str(getattr(s, "action", "")).upper() == "SELL")
                favoured = "SELL" if buy_count >= sell_count else "BUY"
            else:
                # HIGH_VOL or unknown — pick highest confidence
                best = max(sym_sigs, key=lambda s: float(s.confidence))
                resolved.append(best)
                continue

            # Pick the best signal matching the favoured direction
            matching = [s for s in sym_sigs if str(getattr(s, "action", "")).upper() == favoured]
            if matching:
                resolved.append(max(matching, key=lambda s: float(s.confidence)))
            else:
                # No signal matches favoured direction — take highest confidence anyway
                resolved.append(max(sym_sigs, key=lambda s: float(s.confidence)))

        return resolved

    def _signal_quality_gate(self, symbol: str, action: str, confidence: float) -> bool:
        """Reject signals where the historical accuracy for the (symbol, action) pattern
        is below 45%.  Returns True if the signal passes the gate.

        If fewer than 10 samples have been recorded, the signal is allowed through
        (insufficient data to judge).
        """
        key = f"{symbol}:{action}"
        stats = self._signal_accuracy_tracker.get(key)
        if stats is None or int(stats.get("total", 0)) < 10:
            return True  # not enough history to reject
        win_rate = float(stats.get("wins", 0)) / max(1, int(stats["total"]))
        if win_rate < 0.45:
            logger.debug(
                "Signal quality gate REJECTED %s %s (win_rate=%.2f%%, conf=%.3f)",
                symbol, action, win_rate * 100, confidence,
            )
            return False
        return True

    def record_signal_outcome(self, symbol: str, action: str, won: bool) -> None:
        """Called externally when a trade closes so we can track per-pattern accuracy."""
        key = f"{symbol}:{action}"
        if key not in self._signal_accuracy_tracker:
            self._signal_accuracy_tracker[key] = {"wins": 0, "total": 0}
        self._signal_accuracy_tracker[key]["total"] += 1
        if won:
            self._signal_accuracy_tracker[key]["wins"] += 1

    def _momentum_ignition_filter(self, df: pd.DataFrame) -> bool:
        """Return True if the last 3 candles look like momentum ignition (manipulation).

        Detects when price moved >2% in the last 3 candles but volume did NOT
        increase proportionally (price_change / volume_change ratio > 3).
        """
        if df is None or len(df) < 4 or "close" not in df.columns or "volume" not in df.columns:
            return False
        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        price_3_ago = float(close.iloc[-4])
        price_now = float(close.iloc[-1])
        if price_3_ago <= 0:
            return False
        price_change_pct = abs(price_now - price_3_ago) / price_3_ago

        if price_change_pct <= 0.02:
            return False  # move is small, not suspicious

        vol_3_ago = float(vol.iloc[-4])
        vol_now = float(vol.iloc[-1])
        if vol_3_ago <= 0:
            return False
        vol_change_ratio = vol_now / vol_3_ago if vol_3_ago > 0 else 1.0
        if vol_change_ratio <= 0:
            vol_change_ratio = 0.01

        price_vol_ratio = price_change_pct / max(vol_change_ratio, 0.01)
        if price_vol_ratio > 3.0:
            logger.debug(
                "Momentum ignition filter triggered: price_chg=%.2f%%, vol_ratio=%.2f, pv_ratio=%.2f",
                price_change_pct * 100, vol_change_ratio, price_vol_ratio,
            )
            return True
        return False

    def _make_regime_aligned_signal(
        self,
        symbol: str,
        action: str,
        regime: MarketRegime,
        last_price: float,
        confidence: float,
        strength: float,
        reasons: List[str],
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Optional[TradingSignal]:
        """Emit a signal only when direction aligns with regime (BUY in RANGE/TREND_UP, SELL in RANGE/TREND_DOWN)."""
        if action == "BUY" and regime in (MarketRegime.RANGE, MarketRegime.TREND_UP):
            pass  # aligned
        elif action == "SELL" and regime in (MarketRegime.RANGE, MarketRegime.TREND_DOWN):
            pass  # aligned
        else:
            return None
        if action == "BUY":
            stop_loss = last_price * (1 - stop_loss_pct)
            take_profit = last_price * (1 + take_profit_pct)
        else:
            stop_loss = last_price * (1 + stop_loss_pct)
            take_profit = last_price * (1 - take_profit_pct)
        # Profitability: stronger bump for regime-aligned (prioritize when ranking vs other signals)
        conf = min(1.0, confidence * 1.08)
        sig = TradingSignal(
            symbol=str(symbol),
            action=action,
            confidence=conf,
            strength=strength,
            entry_price=last_price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            reasoning=("regime_aligned: " + "; ".join(reasons)) if reasons else "Regime aligned with direction",
            agent_consensus=conf,
        )
        setattr(sig, "strategy", "regime_aligned")
        return sig

    def _make_volume_momentum_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        last_price: float,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Optional[TradingSignal]:
        """Momentum (EMA cross) plus volume above recent average to reduce false breakouts."""
        if df is None or df.empty or len(df) < 30:
            return None
        close = df["close"] if "close" in df.columns else df.get("Close", pd.Series(dtype=float))
        if "volume" in df.columns:
            vol = df["volume"]
        elif "Volume" in df.columns:
            vol = df["Volume"]
        else:
            return None
        close = pd.Series(close, dtype=float)
        vol = pd.Series(vol, dtype=float)
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        vol_sma = vol.rolling(20).mean()
        if pd.isna(vol_sma.iloc[-1]) or vol_sma.iloc[-1] <= 0:
            return None
        last_vol = float(vol.iloc[-1])
        if last_vol < float(vol_sma.iloc[-1]) * 1.1:
            return None  # require volume at least 10% above average
        c, ef, es = float(close.iloc[-1]), float(ema_fast.iloc[-1]), float(ema_slow.iloc[-1])
        if c > ef > es:
            action = "BUY"
        elif c < ef < es:
            action = "SELL"
        else:
            return None
        confidence = 0.72
        strength = 0.7
        if action == "BUY":
            stop_loss = last_price * (1 - stop_loss_pct)
            take_profit = last_price * (1 + take_profit_pct)
        else:
            stop_loss = last_price * (1 + stop_loss_pct)
            take_profit = last_price * (1 - take_profit_pct)
        sig = TradingSignal(
            symbol=str(symbol),
            action=action,
            confidence=confidence,
            strength=strength,
            entry_price=last_price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            reasoning="volume_momentum: EMA cross with volume above average",
            agent_consensus=confidence,
        )
        setattr(sig, "strategy", "volume_momentum")
        return sig

    def _make_mean_reversion_signal(
        self,
        symbol: str,
        regime: MarketRegime,
        last_rsi: float,
        last_price: float,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Optional[TradingSignal]:
        """RSI oversold/overbought signals only in RANGE regime (reduces trend-against entries)."""
        if regime != MarketRegime.RANGE:
            return None
        if last_rsi < 32:
            action = "BUY"
            confidence = 0.70 + min(0.12, (32 - last_rsi) / 32 * 0.12)
        elif last_rsi > 68:
            action = "SELL"
            confidence = 0.70 + min(0.12, (last_rsi - 68) / 32 * 0.12)
        else:
            return None
        strength = 0.65
        if action == "BUY":
            stop_loss = last_price * (1 - stop_loss_pct)
            take_profit = last_price * (1 + take_profit_pct)
        else:
            stop_loss = last_price * (1 + stop_loss_pct)
            take_profit = last_price * (1 - take_profit_pct)
        sig = TradingSignal(
            symbol=str(symbol),
            action=action,
            confidence=min(1.0, confidence),
            strength=strength,
            entry_price=last_price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            reasoning=f"mean_reversion: RSI {last_rsi:.1f} in range regime",
            agent_consensus=confidence,
        )
        setattr(sig, "strategy", "mean_reversion")
        return sig

    def _make_breakout_bb_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        last_price: float,
        last_bbpos: float,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Optional[TradingSignal]:
        """Price at Bollinger band extreme with volume above average (breakout confirmation)."""
        if df is None or df.empty or len(df) < 25:
            return None
        if "volume" in df.columns:
            vol = pd.Series(df["volume"], dtype=float)
        elif "Volume" in df.columns:
            vol = pd.Series(df["Volume"], dtype=float)
        else:
            return None
        vol_sma = vol.rolling(20).mean()
        if pd.isna(vol_sma.iloc[-1]) or vol_sma.iloc[-1] <= 0:
            return None
        if float(vol.iloc[-1]) < float(vol_sma.iloc[-1]) * 1.05:
            return None  # require volume at least 5% above average
        if last_bbpos >= 0.92:
            action = "BUY"
        elif last_bbpos <= 0.08:
            action = "SELL"
        else:
            return None
        confidence = 0.72
        strength = 0.68
        if action == "BUY":
            stop_loss = last_price * (1 - stop_loss_pct)
            take_profit = last_price * (1 + take_profit_pct)
        else:
            stop_loss = last_price * (1 + stop_loss_pct)
            take_profit = last_price * (1 - take_profit_pct)
        sig = TradingSignal(
            symbol=str(symbol),
            action=action,
            confidence=confidence,
            strength=strength,
            entry_price=last_price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            reasoning=f"breakout_bb: BB position {last_bbpos:.2f} with volume confirmation",
            agent_consensus=confidence,
        )
        setattr(sig, "strategy", "breakout_bb")
        return sig

    def _make_macd_trend_signal(
        self,
        symbol: str,
        regime: MarketRegime,
        macd_hist: pd.Series,
        last_price: float,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Optional[TradingSignal]:
        """Trend regime + MACD histogram direction (rising in TREND_UP, falling in TREND_DOWN)."""
        if macd_hist is None or len(macd_hist) < 3:
            return None
        cur = float(macd_hist.iloc[-1])
        prev = float(macd_hist.iloc[-2])
        if regime == MarketRegime.TREND_UP and cur > 0 and cur > prev:
            action = "BUY"
        elif regime == MarketRegime.TREND_DOWN and cur < 0 and cur < prev:
            action = "SELL"
        else:
            return None
        confidence = 0.74
        strength = 0.70
        if action == "BUY":
            stop_loss = last_price * (1 - stop_loss_pct)
            take_profit = last_price * (1 + take_profit_pct)
        else:
            stop_loss = last_price * (1 + stop_loss_pct)
            take_profit = last_price * (1 - take_profit_pct)
        sig = TradingSignal(
            symbol=str(symbol),
            action=action,
            confidence=confidence,
            strength=strength,
            entry_price=last_price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            reasoning="macd_trend: MACD confirming trend regime",
            agent_consensus=confidence,
        )
        setattr(sig, "strategy", "macd_trend")
        return sig

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        # Optional: ensure close is on GPU for downstream batch (use_gpu_indicators); RSI computed in NumPy/pandas.
        if bool(getattr(self.config, "use_gpu_indicators", False)):
            try:
                try:
                    from utils.gpu_inference import is_gpu_available, array
                except ImportError:
                    is_gpu_available = lambda: False
                    array = None
                if is_gpu_available() and array is not None:
                    array(close.values)  # touch GPU for data path; full GPU RSI can use CuPy kernels later
            except Exception as _e:
                logger.debug("strategy_engine error: %s", _e)
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1 / float(period), adjust=False).mean()
        roll_down = down.ewm(alpha=1 / float(period), adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # Keep NaN for the first `period` bars (warmup) so they never trigger signals.
        # Only fill NaN AFTER warmup with 50.0 as a neutral default.
        rsi.iloc[:period] = np.nan
        return rsi.fillna(50.0)

    def _macd_hist(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - sig
        # Mark warmup period as NaN so no spurious signals fire during convergence
        warmup = slow + signal
        hist.iloc[:warmup] = np.nan
        return hist.fillna(0.0)

    def _bollinger_position(self, close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
        sma = close.rolling(window=period).mean()
        sd = close.rolling(window=period).std()
        upper = sma + (sd * std)
        lower = sma - (sd * std)
        denom = (upper - lower).replace(0, np.nan)
        pos = (close - lower) / denom
        # Mark warmup period as NaN (rolling window needs `period` bars)
        pos.iloc[:period] = np.nan
        return pos.clip(lower=0.0, upper=1.0).fillna(0.5)

    def _order_book_bias(self, ob: Dict[str, Any]) -> float:
        """
        Return imbalance in [-1, 1] based on top-of-book volumes.
        """
        try:
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            bid_vol = float(sum(float(x[1]) for x in bids[:10] if len(x) >= 2))
            ask_vol = float(sum(float(x[1]) for x in asks[:10] if len(x) >= 2))
            tot = bid_vol + ask_vol
            if tot <= 0:
                return 0.0
            return (bid_vol - ask_vol) / tot
        except Exception:
            return 0.0

    # ── NEW INDICATORS ────────────────────────────────────────────────

    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index — trend strength 0-100."""
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr = self._atr(df, period)
        atr_safe = atr.replace(0, np.nan)
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_safe)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_safe)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.ewm(span=period, adjust=False).mean()
        adx.iloc[:period * 2] = np.nan
        return adx.fillna(20.0)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def _vwap(self, df: pd.DataFrame) -> pd.Series:
        """Volume-Weighted Average Price (session VWAP)."""
        typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
        vol = df["volume"].astype(float).replace(0, np.nan)
        cum_tp_vol = (typical * vol).cumsum()
        cum_vol = vol.cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    def _detect_divergence(self, close: pd.Series, rsi: pd.Series, lookback: int = 20) -> str:
        """
        Detect RSI-price divergence over last `lookback` bars.
        Returns 'bullish', 'bearish', or 'none'.

        Bullish: price makes lower low, RSI makes higher low
        Bearish: price makes higher high, RSI makes lower high
        """
        if len(close) < lookback or len(rsi) < lookback:
            return "none"
        recent_close = close.iloc[-lookback:]
        recent_rsi = rsi.iloc[-lookback:]
        half = lookback // 2

        # Find lows/highs in first and second half
        c_first_low = float(recent_close.iloc[:half].min())
        c_second_low = float(recent_close.iloc[half:].min())
        r_first_low = float(recent_rsi.iloc[:half].min())
        r_second_low = float(recent_rsi.iloc[half:].min())

        c_first_high = float(recent_close.iloc[:half].max())
        c_second_high = float(recent_close.iloc[half:].max())
        r_first_high = float(recent_rsi.iloc[:half].max())
        r_second_high = float(recent_rsi.iloc[half:].max())

        # Bullish divergence: lower low in price, higher low in RSI
        if c_second_low < c_first_low * 0.998 and r_second_low > r_first_low * 1.02:
            return "bullish"
        # Bearish divergence: higher high in price, lower high in RSI
        if c_second_high > c_first_high * 1.002 and r_second_high < r_first_high * 0.98:
            return "bearish"
        return "none"

    def _order_book_pressure(self, ob: Dict[str, Any], depth: int = 10) -> float:
        """Depth-weighted order book pressure in [-1, 1]. Positive = bid pressure."""
        try:
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            bid_p = sum(float(x[1]) * (1 - i / depth) for i, x in enumerate(bids[:depth]) if len(x) >= 2)
            ask_p = sum(float(x[1]) * (1 - i / depth) for i, x in enumerate(asks[:depth]) if len(x) >= 2)
            total = bid_p + ask_p
            if total <= 0:
                return 0.0
            return (bid_p - ask_p) / total
        except Exception:
            return 0.0

