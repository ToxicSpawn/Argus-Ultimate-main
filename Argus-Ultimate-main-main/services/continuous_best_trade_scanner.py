"""
Continuous Best-Trade Scanner (Peak Mode)
=========================================

Runs in the background and continuously scans the market for the best
trading opportunities. Aggregates signals from the strategy engine and
optional HFT/arbitrage sources, ranks them by a unified "best trade"
score, and exposes the top opportunities for the main trading loop.

Peak enhancements:
- Parallel source gathering (AI brain, strategy engine, HFT) with per-source timeouts
- Source budgets: cap candidates per source to avoid single-source domination
- Richer scoring: confidence * strength + edge bonus + liquidity boost + strategy/recency
- Diversity selection: cap per-symbol and per-strategy so top N is not all one pair
- Optional liquidity boost from order book spread (tighter spread = higher score)
- Adaptive interval: scan more frequently when market is volatile or when cache was empty
- Metrics: scan duration, per-source timing/counts, cache freshness for tuning
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RankedOpportunity:
    """A trading opportunity with a unified 'best trade' score."""
    symbol: str
    action: str
    confidence: float
    strength: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    strategy: str = ""
    score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_signal: Any = None


class ContinuousBestTradeScanner:
    """
    Continuously scans the market, ranks opportunities by a unified score,
    and exposes the top N for the main trading loop. Peak mode: parallel
    sources, liquidity boost, diversity, adaptive interval, metrics.
    """

    def __init__(
        self,
        config: Any,
        *,
        ai_brain: Any = None,
        market_data_service: Any = None,
        strategy_engine: Any = None,
        hft_engine: Any = None,
        interval_seconds: float = 10.0,
        top_n: int = 5,
        min_score: float = 0.0,
        parallel_sources: bool = True,
        use_liquidity_boost: bool = True,
        liquidity_spread_pct_cap: float = 0.05,
        diversity_max_per_symbol: int = 2,
        diversity_max_per_strategy: int = 2,
        adaptive_interval_enabled: bool = True,
        min_interval_seconds: float = 5.0,
        max_interval_seconds: float = 30.0,
    ) -> None:
        self.config = config
        self.ai_brain = ai_brain
        self.market_data_service = market_data_service
        self.strategy_engine = strategy_engine
        self.hft_engine = hft_engine
        self._base_interval = max(0.5, float(interval_seconds))
        self.interval_seconds = self._base_interval
        self.top_n = max(1, int(top_n))
        self.min_score = float(min_score)
        self.parallel_sources = bool(parallel_sources)
        self.use_liquidity_boost = bool(use_liquidity_boost)
        self.liquidity_spread_pct_cap = max(0.001, float(liquidity_spread_pct_cap))
        self.diversity_max_per_symbol = max(1, int(diversity_max_per_symbol))
        self.diversity_max_per_strategy = max(1, int(diversity_max_per_strategy))
        self.adaptive_interval_enabled = bool(adaptive_interval_enabled)
        self.min_interval_seconds = max(1.0, float(min_interval_seconds))
        self.max_interval_seconds = max(self.min_interval_seconds, float(max_interval_seconds))

        self._lock = asyncio.Lock()
        self._best_opportunities: List[RankedOpportunity] = []
        self._last_scan_ts: float = 0.0
        self._scan_count: int = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_scan_duration_s: float = 0.0
        self._last_candidate_count: int = 0
        self._full_pairs: List[str] = []
        self._rotation_offset: int = 0
        self._max_symbols_per_scan = max(10, min(50, int(getattr(config, "continuous_scan_max_symbols_per_scan", 25) or 25)))
        self.net_edge_weight = float(getattr(config, "continuous_scan_net_edge_weight", 1.0) or 1.0)
        self.cost_penalty_weight = float(getattr(config, "continuous_scan_cost_penalty_weight", 0.75) or 0.75)
        self.fill_prob_weight = float(getattr(config, "continuous_scan_fill_prob_weight", 0.50) or 0.50)
        self.slippage_penalty_weight = float(
            getattr(config, "continuous_scan_slippage_penalty_weight", 0.50) or 0.50
        )
        self.min_expected_net_edge_bps = float(
            getattr(config, "continuous_scan_min_expected_net_edge_bps", 0.0) or 0.0
        )
        self.symbol_quality_overrides = dict(
            getattr(config, "continuous_scan_symbol_quality_overrides", {}) or {}
        )
        self.session_hour_multipliers = dict(
            getattr(config, "continuous_scan_session_hour_multipliers", {}) or {}
        )
        self.strategy_multipliers = dict(
            getattr(config, "continuous_scan_strategy_multipliers", {}) or {}
        )
        self.source_timeouts_seconds = dict(
            getattr(config, "continuous_scan_source_timeouts_seconds", {}) or {}
        )
        self.source_max_candidates = dict(
            getattr(config, "continuous_scan_source_max_candidates", {}) or {}
        )
        self.exposure_penalty_weight = float(
            getattr(config, "continuous_scan_exposure_penalty_weight", 0.0) or 0.0
        )
        self.apply_runtime_risk_scale = bool(
            getattr(config, "continuous_scan_apply_runtime_risk_scale", True)
        )
        self.runtime_risk_floor = float(
            getattr(config, "continuous_scan_runtime_risk_floor", 0.25) or 0.25
        )
        self._last_source_timings_ms: Dict[str, float] = {}
        self._last_source_candidate_counts: Dict[str, int] = {}
        self._last_source_capped_counts: Dict[str, int] = {}

    def _session_multiplier(self) -> float:
        if not self.session_hour_multipliers:
            return 1.0
        hour = str(datetime.utcnow().hour)
        try:
            val = float(self.session_hour_multipliers.get(hour, 1.0) or 1.0)
            return max(0.1, min(3.0, val))
        except Exception:
            return 1.0

    def _symbol_quality_multiplier(self, symbol: str) -> float:
        if not symbol:
            return 1.0
        try:
            val = float(self.symbol_quality_overrides.get(symbol, 1.0) or 1.0)
            return max(0.1, min(3.0, val))
        except Exception:
            return 1.0

    def _strategy_multiplier(self, strategy: str) -> float:
        if not strategy:
            return 1.0
        try:
            val = float(self.strategy_multipliers.get(strategy, 1.0) or 1.0)
            return max(0.1, min(3.0, val))
        except Exception:
            return 1.0

    def _source_timeout(self, source_name: str, default_value: float) -> float:
        try:
            raw = self.source_timeouts_seconds.get(source_name, default_value)
            return max(0.1, float(raw))
        except Exception:
            return float(default_value)

    def _apply_source_cap(self, source_name: str, items: List[Any]) -> List[Any]:
        if not items:
            return []
        raw_cap = self.source_max_candidates.get(source_name)
        if raw_cap is None:
            self._last_source_capped_counts[source_name] = len(items)
            return list(items)
        try:
            cap = max(0, int(raw_cap))
        except Exception:
            cap = len(items)
        if cap <= 0:
            self._last_source_capped_counts[source_name] = 0
            return []
        capped = list(items)[:cap]
        self._last_source_capped_counts[source_name] = len(capped)
        return capped

    def _estimate_net_edge_bps(self, signal: Any) -> float:
        direct = getattr(signal, "expected_net_edge_bps", None)
        if direct is not None:
            try:
                return float(direct)
            except Exception as _e:
                logger.debug("continuous_best_trade_scanner error: %s", _e)

        edge_pct = float(getattr(signal, "expected_profit_pct", 0.0) or 0.0)
        gross_edge_bps = max(0.0, edge_pct * 10000.0)
        spread_bps = float(getattr(signal, "spread_bps", 0.0) or 0.0)
        slippage_pct = float(getattr(self.config, "slippage_pct", 0.001) or 0.001)
        slippage_bps = slippage_pct * 2.0 * 10000.0
        fee = max(
            float(getattr(self.config, "kraken_taker_fee", 0.0026) or 0.0026),
            float(getattr(self.config, "coinbase_taker_fee", 0.005) or 0.005),
        )
        fee_bps = fee * 2.0 * 10000.0
        return float(gross_edge_bps - (spread_bps + slippage_bps + fee_bps))

    def _score_signal(self, signal: Any, liquidity_mult: float = 1.0) -> float:
        """
        How the scanner knows what is 'best' at any given time:

        - Base: confidence * strength * liquidity_mult
          (confidence/strength come from strategy engine: RSI, Bollinger, MACD, regime, etc.)
        - Liquidity: tighter order-book spread -> mult up to 1.25 (fewer slippage/cost)
        - Strategy: arbitrage +15%, unified_engine +5%
        - Edge: expected_profit_pct (e.g. from HFT) adds up to +50%
        - Recency: signal < 60s old +2%

        Higher score = better. We rank by this score, then apply diversity (max per symbol/strategy), then take top N.
        """
        try:
            conf = float(getattr(signal, "confidence", 0.0) or 0.0)
            strength = float(getattr(signal, "strength", 0.0) or 0.0)
            base = conf * strength * liquidity_mult
            if base <= 0:
                return 0.0
            # Strategy bonus
            strategy = str(getattr(signal, "strategy", "") or "").strip()
            if "arbitrage" in strategy.lower():
                base *= 1.15
            if "unified_engine" in strategy:
                base *= 1.05
            # Edge bonus (expected profit %)
            edge = float(getattr(signal, "expected_profit_pct", 0.0) or 0.0)
            if edge > 0:
                base *= (1.0 + min(0.5, edge * 10.0))
            # Slight recency bonus (newer = slightly better)
            ts = getattr(signal, "timestamp", None)
            if ts is not None:
                age_s = (datetime.now() - ts).total_seconds() if hasattr(ts, "__sub__") else 0.0
                if age_s < 60.0:
                    base *= 1.02
            net_edge_bps = self._estimate_net_edge_bps(signal)
            if net_edge_bps < float(self.min_expected_net_edge_bps):
                return 0.0

            expected_fill_prob = float(
                getattr(signal, "expected_fill_probability", getattr(signal, "maker_fill_ratio", 0.5)) or 0.5
            )
            expected_fill_prob = max(0.0, min(1.0, expected_fill_prob))
            expected_slippage_bps = float(
                getattr(
                    signal,
                    "expected_slippage_bps",
                    float(getattr(self.config, "slippage_pct", 0.001) or 0.001) * 10000.0,
                )
                or 0.0
            )
            expected_slippage_bps = max(0.0, expected_slippage_bps)
            sym = str(getattr(signal, "symbol", "") or "")
            strategy = str(getattr(signal, "strategy", "") or "")
            current_exposure_pct = abs(float(getattr(signal, "current_exposure_pct", 0.0) or 0.0))
            delta_exposure_pct = abs(float(getattr(signal, "delta_exposure_pct", 0.0) or 0.0))
            score = float(base)
            score += float(self.net_edge_weight) * max(0.0, net_edge_bps) / 100.0
            score -= float(self.cost_penalty_weight) * max(0.0, -net_edge_bps) / 100.0
            score *= 1.0 + float(self.fill_prob_weight) * expected_fill_prob
            score /= 1.0 + float(self.slippage_penalty_weight) * (expected_slippage_bps / 25.0)
            score *= self._strategy_multiplier(strategy)
            if self.exposure_penalty_weight > 0.0:
                score /= 1.0 + float(self.exposure_penalty_weight) * (
                    current_exposure_pct + 0.5 * delta_exposure_pct
                )
            if self.apply_runtime_risk_scale:
                runtime_scale = float(
                    getattr(
                        signal,
                        "risk_scale_realtime",
                        getattr(self.config, "_runtime_risk_scale_realtime", 1.0),
                    )
                    or 1.0
                )
                floor = max(0.0, min(1.0, float(self.runtime_risk_floor)))
                runtime_scale = max(floor, min(1.0, runtime_scale))
                score *= runtime_scale
            score *= self._session_multiplier()
            score *= self._symbol_quality_multiplier(sym)
            return max(0.0, float(score))
        except Exception:
            return 0.0

    def _signal_to_ranked(self, signal: Any, score: float) -> RankedOpportunity:
        """Convert a TradingSignal (or similar) to RankedOpportunity."""
        return RankedOpportunity(
            symbol=str(getattr(signal, "symbol", "") or ""),
            action=str(getattr(signal, "action", "HOLD") or "HOLD").upper(),
            confidence=float(getattr(signal, "confidence", 0.0) or 0.0),
            strength=float(getattr(signal, "strength", 0.0) or 0.0),
            entry_price=float(getattr(signal, "entry_price", 0.0) or 0.0),
            stop_loss=getattr(signal, "stop_loss", None),
            take_profit=getattr(signal, "take_profit", None),
            reasoning=str(getattr(signal, "reasoning", "") or ""),
            strategy=str(getattr(signal, "strategy", "") or ""),
            score=score,
            timestamp=datetime.now(),
            raw_signal=signal,
        )

    async def _gather_ai_brain(self) -> List[Any]:
        """Gather signals from AI brain (return list, empty on error)."""
        if self.ai_brain is None:
            return []
        try:
            signals = await asyncio.wait_for(
                self.ai_brain.generate_trading_signals(),
                timeout=self._source_timeout("ai_brain", 28.0),
            )
            return list(signals or [])
        except asyncio.TimeoutError:
            logger.debug("Scanner: AI brain timeout")
            return []
        except Exception as e:
            logger.debug("Scanner: AI brain error: %s", e)
            return []

    async def _gather_strategy_engine(self) -> List[Any]:
        """Gather signals from strategy engine only."""
        if self.strategy_engine is None or self.market_data_service is None:
            return []
        try:
            signals = await asyncio.wait_for(
                self.strategy_engine.generate_signals(self.market_data_service),
                timeout=self._source_timeout("strategy_engine", 18.0),
            )
            return list(signals or [])
        except Exception as e:
            logger.debug("Scanner: strategy engine error: %s", e)
            return []

    async def _gather_strategy_engine_multi_tf(self) -> List[Any]:
        """Multi-timeframe: primary TF (e.g. 1h) for trend, entry TF (e.g. 15m) for entries; keep only when both agree."""
        if self.strategy_engine is None or self.market_data_service is None:
            return []
        enabled = bool(getattr(self.config, "signal_multi_timeframe_enabled", False))
        primary_tf = str(getattr(self.config, "signal_primary_timeframe", "1h") or "1h")
        entry_tf = str(getattr(self.config, "signal_entry_timeframe", "15m") or "15m")
        if not enabled or primary_tf == entry_tf:
            return []
        try:
            primary_signals = await asyncio.wait_for(
                self.strategy_engine.generate_signals(
                    self.market_data_service,
                    timeframe_override=primary_tf,
                    strategy_suffix=f"_{primary_tf}",
                ),
                timeout=self._source_timeout("strategy_multi_tf", 20.0),
            )
            entry_signals = await asyncio.wait_for(
                self.strategy_engine.generate_signals(
                    self.market_data_service,
                    timeframe_override=entry_tf,
                    strategy_suffix=f"_{entry_tf}",
                ),
                timeout=self._source_timeout("strategy_multi_tf", 20.0),
            )
            primary_by_sym = {}
            for s in primary_signals or []:
                sym = str(getattr(s, "symbol", "") or "")
                act = str(getattr(s, "action", "") or "").upper()
                if sym and act in ("BUY", "SELL"):
                    primary_by_sym[sym] = act
            out = []
            for s in entry_signals or []:
                sym = str(getattr(s, "symbol", "") or "")
                act = str(getattr(s, "action", "") or "").upper()
                if sym and act in ("BUY", "SELL") and primary_by_sym.get(sym) == act:
                    out.append(s)
            return out
        except Exception as e:
            logger.debug("Scanner: strategy engine multi-TF error: %s", e)
            return []

    async def _gather_external_alpha(self) -> List[Any]:
        """Fetch external signals from URL (webhook/API); merge into candidates."""
        url = str(getattr(self.config, "external_alpha_url", "") or "").strip()
        enabled = bool(getattr(self.config, "external_alpha_enabled", False))
        timeout_s = self._source_timeout(
            "external_alpha",
            float(getattr(self.config, "external_alpha_timeout_seconds", 5.0) or 5.0),
        )
        if not enabled or not url:
            return []
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
            items = data if isinstance(data, list) else (data.get("signals", data.get("signals_list", [])) or [])
            out = []
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                sym = str(item.get("symbol") or item.get("pair") or "BTC/USD")
                action = str(item.get("action") or item.get("side") or "").strip().upper()
                if action not in ("BUY", "SELL"):
                    continue
                conf = float(item.get("confidence", 0.7) or 0.7)
                price = float(item.get("price") or item.get("entry_price") or 0.0)
                sig = type("ExtSignal", (), {
                    "symbol": sym,
                    "action": action,
                    "confidence": min(1.0, max(0.0, conf)),
                    "strength": min(1.0, conf * 0.9),
                    "entry_price": price,
                    "strategy": "external_alpha",
                    "reasoning": str(item.get("reason", item.get("reasoning", "")) or ""),
                })()
                out.append(sig)
            return out
        except Exception as e:
            logger.debug("Scanner: external alpha fetch error: %s", e)
            return []

    async def _gather_strategy_plugins(self) -> List[Any]:
        """Load strategy plugins from config (strategy_plugin_modules); each module provides get_strategies(config) -> {name: strategy}."""
        modules = list(getattr(self.config, "strategy_plugin_modules", None) or [])
        if not modules or not self.market_data_service:
            return []
        out: List[Any] = []
        window = getattr(self.config, "_scanner_symbol_window", None) or list(getattr(self.config, "trading_pairs", []) or ["BTC/USD"])[:5]
        for mod_path in modules:
            if not isinstance(mod_path, str) or not mod_path.strip():
                continue
            try:
                import importlib
                mod = importlib.import_module(mod_path.strip())
                get_strategies = getattr(mod, "get_strategies", None) or getattr(mod, "register_strategies", None)
                if not callable(get_strategies):
                    continue
                try:
                    registry = get_strategies(self.config) if get_strategies.__code__.co_argcount >= 1 else get_strategies()
                except TypeError:
                    registry = get_strategies()
                if not isinstance(registry, dict):
                    continue
                for name, strat in registry.items():
                    if not hasattr(strat, "analyze"):
                        continue
                    for symbol in window:
                        try:
                            df = await asyncio.wait_for(
                                self.market_data_service.fetch_ohlcv_df(symbol, timeframe="1m", limit=200),
                                timeout=self._source_timeout("strategy_plugins", 5.0),
                            )
                            if df is None or df.empty or "close" not in df.columns or len(df) < 50:
                                continue
                            price = float(df["close"].iloc[-1])
                            market_data = {"symbol": symbol, "price": price, "ohlcv_df": df, "tickers": {symbol: price}}
                            result = strat.analyze(market_data)
                            if not result or not isinstance(result, dict):
                                continue
                            action = str(result.get("action", "") or "").strip().upper()
                            if action not in ("BUY", "SELL"):
                                continue
                            conf = float(result.get("confidence", 0.5) or 0.5)
                            if conf < 0.5:
                                continue
                            sig = type("PluginSignal", (), {
                                "symbol": symbol,
                                "action": action,
                                "confidence": conf,
                                "strength": min(1.0, conf * 0.9),
                                "entry_price": price,
                                "strategy": f"plugin_{name}",
                                "reasoning": str(result.get("source", "") or ""),
                            })()
                            out.append(sig)
                        except Exception:
                            continue
            except Exception as e:
                logger.debug("Scanner: strategy plugin %s error: %s", mod_path, e)
        return out

    async def _gather_strategy_library(self) -> List[Any]:
        """Gather signals from strategy library (tier + algorithmic strategies) when enabled."""
        if not getattr(self.config, "strategy_library_enabled", False):
            return []
        modes = list(getattr(self.config, "strategy_library_modes", []) or [])
        run_mode = (getattr(self.config, "run_mode", None) or "paper").lower()
        if run_mode not in [str(m).lower() for m in modes]:
            return []
        if not self.market_data_service:
            return []
        enabled = list(getattr(self.config, "strategy_library_strategies_enabled", []) or [])
        if not enabled:
            return []
        try:
            from strategies.strategy_library_impl import get_library_strategies_for_names
            registry = get_library_strategies_for_names(enabled)
        except Exception as e:
            logger.debug("Scanner: strategy library registry error: %s", e)
            return []
        if not registry:
            return []
        window = getattr(self.config, "_scanner_symbol_window", None) or list(getattr(self.config, "trading_pairs", []) or ["BTC/USD"])[:5]
        out: List[Any] = []
        for symbol in window:
            try:
                df = await asyncio.wait_for(
                    self.market_data_service.fetch_ohlcv_df(symbol, timeframe="1m", limit=200),
                    timeout=self._source_timeout("strategy_library", 5.0),
                )
            except Exception:
                continue
            if df is None or df.empty or "close" not in df.columns or len(df) < 50:
                continue
            price = float(df["close"].iloc[-1]) if df is not None else 0.0
            market_data = {"symbol": symbol, "price": price, "ohlcv_df": df, "tickers": {symbol: price}}
            for name, strat in registry.items():
                try:
                    result = strat.analyze(market_data)
                except Exception:
                    continue
                if not result or not isinstance(result, dict):
                    continue
                action = str(result.get("action", "") or "").strip().upper()
                if action not in ("BUY", "SELL"):
                    continue
                conf = float(result.get("confidence", 0) or 0)
                if conf < 0.5:
                    continue
                # Object compatible with scanner scoring (symbol, strategy, action, confidence, strength, entry_price)
                sig = type("LibSignal", (), {
                    "symbol": symbol,
                    "action": action,
                    "confidence": conf,
                    "strength": min(1.0, conf * 0.9),
                    "entry_price": price,
                    "strategy": name,
                    "reasoning": str(result.get("source", "") or ""),
                })()
                out.append(sig)
        return out

    async def _gather_hft(self) -> List[Any]:
        """Gather HFT/arbitrage opportunities (OBI from order book when market_data_service available)."""
        if self.hft_engine is None or not hasattr(self.hft_engine, "scan_for_opportunities"):
            return []
        try:
            pairs = list(getattr(self.config, "trading_pairs", []) or [])
            if not pairs:
                pairs = ["BTC/USD", "ETH/USD"]
            request = {"symbols": pairs}
            if self.market_data_service is not None:
                request["market_data_service"] = self.market_data_service
            opps = await asyncio.wait_for(
                self.hft_engine.scan_for_opportunities(request),
                timeout=self._source_timeout("hft", 8.0),
            )
            out = []
            for opp in (opps or []):
                edge = float(getattr(opp, "expected_profit_pct", 0.0) or 0.0)
                if edge > 0:
                    out.append(opp)
            return out
        except Exception as e:
            logger.debug("Scanner: HFT scan error: %s", e)
            return []

    async def _fetch_liquidity_multipliers(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch order book spread for each symbol; return multiplier per symbol.
        Tighter spread = higher multiplier (max 1.25). Uses cache if available.
        """
        mult: Dict[str, float] = {}
        if not self.use_liquidity_boost or not self.market_data_service or not symbols:
            return {s: 1.0 for s in symbols}
        cap = self.liquidity_spread_pct_cap
        unique = list(dict.fromkeys(s for s in symbols if s))

        async def one(sym: str) -> Tuple[str, float]:
            try:
                ob = await asyncio.wait_for(
                    self.market_data_service.fetch_order_book(sym, limit=10),
                    timeout=self._source_timeout("liquidity", 3.0),
                )
                if not ob or not isinstance(ob, dict):
                    return (sym, 1.0)
                bids = ob.get("bids") or []
                asks = ob.get("asks") or []
                if not bids or not asks:
                    return (sym, 1.0)
                best_bid = float(bids[0][0]) if bids else 0.0
                best_ask = float(asks[0][0]) if asks else 0.0
                if best_bid <= 0:
                    return (sym, 1.0)
                mid = (best_bid + best_ask) / 2.0
                spread = best_ask - best_bid
                spread_pct = (spread / mid * 100.0) if mid else 1.0
                # Tighter spread -> mult up to 1.25
                if spread_pct <= cap:
                    m = 1.0 + 0.25 * (1.0 - min(1.0, spread_pct / max(1e-9, cap)))
                else:
                    m = 1.0
                return (sym, m)
            except Exception:
                return (sym, 1.0)

        results = await asyncio.gather(*[one(s) for s in unique], return_exceptions=True)
        for r in results:
            if isinstance(r, tuple) and len(r) == 2:
                mult[r[0]] = r[1]
            else:
                pass
        for s in unique:
            if s not in mult:
                mult[s] = 1.0
        return mult

    def _apply_diversity(
        self,
        scored: List[tuple],
        max_per_symbol: int,
        max_per_strategy: int,
        top_n: int,
    ) -> List[tuple]:
        """Select top by score with diversity caps per symbol and per strategy."""
        symbol_count: Dict[str, int] = {}
        strategy_count: Dict[str, int] = {}
        out: List[tuple] = []
        for s, score in scored:
            if len(out) >= top_n:
                break
            sym = str(getattr(s, "symbol", "") or "")
            strat = str(getattr(s, "strategy", "") or "unknown")
            if symbol_count.get(sym, 0) >= max_per_symbol:
                continue
            if strategy_count.get(strat, 0) >= max_per_strategy:
                continue
            out.append((s, score))
            symbol_count[sym] = symbol_count.get(sym, 0) + 1
            strategy_count[strat] = strategy_count.get(strat, 0) + 1
        return out

    def _get_scan_window(self) -> List[str]:
        """Return this scan's symbol window (rotate through full list so we cover all pairs)."""
        pairs = list(getattr(self.config, "trading_pairs", []) or [])
        if not pairs:
            return ["BTC/USD", "ETH/USD"]
        if len(pairs) <= self._max_symbols_per_scan:
            return pairs
        # Rotate: each scan we take next window
        n = len(pairs)
        start = self._rotation_offset % n
        window = []
        for i in range(self._max_symbols_per_scan):
            window.append(pairs[(start + i) % n])
        self._rotation_offset = (start + self._max_symbols_per_scan) % n
        return window

    async def _run_scan(self) -> List[RankedOpportunity]:
        """Perform one scan: parallel gather, liquidity boost, rank, diversity, top N."""
        candidates: List[Any] = []
        window = self._get_scan_window()
        # Strategy engine / AI brain read _scanner_symbol_window when set
        setattr(self.config, "_scanner_symbol_window", window)
        source_results: Dict[str, List[Any]] = {}
        source_timings_ms: Dict[str, float] = {}
        source_counts: Dict[str, int] = {}

        async def _timed_collect(source_name: str, coro: Any) -> Tuple[str, List[Any], float]:
            t0 = time.perf_counter()
            try:
                res = await coro
                items = list(res or [])
            except Exception as exc:
                logger.debug("Scanner source %s error: %s", source_name, exc)
                items = []
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return source_name, items, elapsed_ms

        try:
            if self.parallel_sources:
                named_tasks = [
                    asyncio.create_task(_timed_collect("ai_brain", self._gather_ai_brain())),
                    asyncio.create_task(_timed_collect("strategy_engine", self._gather_strategy_engine())),
                    asyncio.create_task(_timed_collect("strategy_multi_tf", self._gather_strategy_engine_multi_tf())),
                    asyncio.create_task(_timed_collect("external_alpha", self._gather_external_alpha())),
                    asyncio.create_task(_timed_collect("hft", self._gather_hft())),
                    asyncio.create_task(_timed_collect("strategy_library", self._gather_strategy_library())),
                    asyncio.create_task(_timed_collect("strategy_plugins", self._gather_strategy_plugins())),
                ]
                for name, items, elapsed_ms in await asyncio.gather(*named_tasks):
                    source_results[name] = items
                    source_timings_ms[name] = elapsed_ms
                    source_counts[name] = len(items)
                ai_list = self._apply_source_cap("ai_brain", source_results.get("ai_brain", []))
                strat_list = self._apply_source_cap("strategy_engine", source_results.get("strategy_engine", []))
                multi_tf_list = self._apply_source_cap("strategy_multi_tf", source_results.get("strategy_multi_tf", []))
                ext_alpha_list = self._apply_source_cap("external_alpha", source_results.get("external_alpha", []))
                hft_list = self._apply_source_cap("hft", source_results.get("hft", []))
                lib_list = self._apply_source_cap("strategy_library", source_results.get("strategy_library", []))
                plugin_list = self._apply_source_cap("strategy_plugins", source_results.get("strategy_plugins", []))
                candidates.extend(ai_list)
                if not candidates:
                    candidates.extend(strat_list)
                candidates.extend(multi_tf_list)
                candidates.extend(ext_alpha_list)
                candidates.extend(hft_list)
                candidates.extend(lib_list)
                candidates.extend(plugin_list)
            else:
                name, ai_items, elapsed_ms = await _timed_collect("ai_brain", self._gather_ai_brain())
                source_results[name] = ai_items
                source_timings_ms[name] = elapsed_ms
                source_counts[name] = len(ai_items)
                ai_list = self._apply_source_cap("ai_brain", ai_items)
                if ai_list:
                    candidates.extend(ai_list)
                else:
                    name, strat_items, elapsed_ms = await _timed_collect("strategy_engine", self._gather_strategy_engine())
                    source_results[name] = strat_items
                    source_timings_ms[name] = elapsed_ms
                    source_counts[name] = len(strat_items)
                    candidates.extend(self._apply_source_cap("strategy_engine", strat_items))

                for source_name, coro in [
                    ("strategy_multi_tf", self._gather_strategy_engine_multi_tf()),
                    ("external_alpha", self._gather_external_alpha()),
                    ("hft", self._gather_hft()),
                    ("strategy_library", self._gather_strategy_library()),
                    ("strategy_plugins", self._gather_strategy_plugins()),
                ]:
                    name, items, elapsed_ms = await _timed_collect(source_name, coro)
                    source_results[name] = items
                    source_timings_ms[name] = elapsed_ms
                    source_counts[name] = len(items)
                    candidates.extend(self._apply_source_cap(source_name, items))

            self._last_source_timings_ms = dict(source_timings_ms)
            self._last_source_candidate_counts = dict(source_counts)
            # Pinnacle: when strategy_whitelist is set, only keep signals from whitelisted strategies
            whitelist = list(getattr(self.config, "strategy_whitelist", None) or [])
            if whitelist:
                allowed = set(str(x).strip().lower() for x in whitelist if x)
                # Treat missing/empty strategy as "unified_engine" (strategy_engine signals)
                filtered = []
                for s in candidates:
                    strat = (getattr(s, "strategy", None) or "").strip() or "unified_engine"
                    if strat.lower() in allowed:
                        filtered.append(s)
                candidates = filtered
        finally:
            # Clear so main loop and others use full trading_pairs again
            if hasattr(self.config, "_scanner_symbol_window"):
                try:
                    delattr(self.config, "_scanner_symbol_window")
                except Exception:
                    setattr(self.config, "_scanner_symbol_window", None)

        symbols = list(dict.fromkeys(str(getattr(s, "symbol", "") or "") for s in candidates if getattr(s, "symbol", None)))
        liquidity = await self._fetch_liquidity_multipliers(symbols)

        scored: List[tuple] = []
        for s in candidates:
            sym = str(getattr(s, "symbol", "") or "")
            mult = liquidity.get(sym, 1.0)
            score = self._score_signal(s, liquidity_mult=mult)
            if score >= self.min_score:
                scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Diversity-aware selection
        diverse = self._apply_diversity(
            scored,
            max_per_symbol=self.diversity_max_per_symbol,
            max_per_strategy=self.diversity_max_per_strategy,
            top_n=self.top_n * 2,
        )
        ranked: List[RankedOpportunity] = []
        for s, score in diverse[: self.top_n]:
            ro = self._signal_to_ranked(s, score)
            if ro.action in ("BUY", "SELL") and ro.entry_price > 0:
                ranked.append(ro)
        ranked = ranked[: self.top_n]

        return ranked

    async def _loop(self) -> None:
        """Background loop: scan every interval_seconds; first scan immediately; adaptive interval."""
        self._running = True
        logger.info(
            "Continuous best-trade scanner (peak) started (base_interval=%.1fs, top_n=%s, parallel=%s, liquidity=%s)",
            self._base_interval,
            self.top_n,
            self.parallel_sources,
            self.use_liquidity_boost,
        )
        first = True
        while self._running:
            try:
                t0 = time.perf_counter()
                opportunities = await self._run_scan()
                elapsed = time.perf_counter() - t0
                async with self._lock:
                    self._best_opportunities = opportunities
                    self._last_scan_ts = time.time()
                    self._scan_count += 1
                    self._last_scan_duration_s = elapsed
                    self._last_candidate_count = len(opportunities)
                if opportunities:
                    logger.debug(
                        "Scanner cycle %s: %s best (top: %s %s @ %.2f) in %.2fs",
                        self._scan_count,
                        len(opportunities),
                        opportunities[0].symbol,
                        opportunities[0].action,
                        opportunities[0].score,
                        elapsed,
                    )
                if self._last_source_candidate_counts:
                    logger.debug(
                        "Scanner source telemetry counts=%s capped=%s timings_ms=%s",
                        self._last_source_candidate_counts,
                        self._last_source_capped_counts,
                        {k: round(v, 1) for k, v in self._last_source_timings_ms.items()},
                    )
                # Adaptive interval: shorten when we found opportunities, lengthen when empty
                if self.adaptive_interval_enabled:
                    if opportunities:
                        self.interval_seconds = max(
                            self.min_interval_seconds,
                            min(self._base_interval * 0.9, self.interval_seconds * 0.95),
                        )
                    else:
                        self.interval_seconds = min(
                            self.max_interval_seconds,
                            self.interval_seconds * 1.05,
                        )
                    self.interval_seconds = max(
                        self.min_interval_seconds,
                        min(self.max_interval_seconds, self.interval_seconds),
                    )
                else:
                    self.interval_seconds = self._base_interval
                sleep = max(0.0, self.interval_seconds - elapsed)
                if not first:
                    await asyncio.sleep(sleep)
                first = False
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Continuous scanner cycle error: %s", e)
                self.interval_seconds = min(self.max_interval_seconds, self.interval_seconds * 1.1)
                await asyncio.sleep(min(5.0, self.interval_seconds))
        self._running = False
        logger.info("Continuous best-trade scanner stopped")

    def start(self) -> None:
        """Start the background scan loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        """Stop the background scan loop."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def get_best_opportunities(
        self,
        *,
        max_age_seconds: Optional[float] = None,
        convert_to_signals: bool = True,
    ) -> List[Any]:
        """
        Return the current best opportunities (from last scan).
        If max_age_seconds is set and the last scan is older, returns [] unless
        convert_to_signals is False (then still returns cached list).
        If convert_to_signals is True, returns TradingSignal-like objects for
        the execution layer; otherwise returns RankedOpportunity objects.
        """
        async with self._lock:
            opportunities = list(self._best_opportunities)
            last_ts = self._last_scan_ts
        if max_age_seconds is not None and (time.time() - last_ts) > max_age_seconds:
            if convert_to_signals:
                return []
            return opportunities
        if not convert_to_signals:
            return opportunities
        # Convert to signals the main loop expects (use raw_signal when available)
        out = []
        for ro in opportunities:
            if ro.raw_signal is not None:
                out.append(ro.raw_signal)
            else:
                from unified_types import TradingSignal
                out.append(
                    TradingSignal(
                        symbol=ro.symbol,
                        action=ro.action,
                        confidence=ro.confidence,
                        strength=ro.strength,
                        entry_price=ro.entry_price,
                        stop_loss=ro.stop_loss,
                        take_profit=ro.take_profit,
                        reasoning=ro.reasoning,
                        timestamp=ro.timestamp,
                    )
                )
        return out

    def status(self) -> dict:
        """Return scanner status for observability (peak: duration, candidate count, interval)."""
        return {
            "running": self._running,
            "scan_count": self._scan_count,
            "last_scan_ts": self._last_scan_ts,
            "last_scan_duration_s": self._last_scan_duration_s,
            "last_candidate_count": self._last_candidate_count,
            "best_opportunities_count": len(self._best_opportunities),
            "interval_seconds": self.interval_seconds,
            "base_interval_seconds": self._base_interval,
            "top_n": self.top_n,
            "parallel_sources": self.parallel_sources,
            "use_liquidity_boost": self.use_liquidity_boost,
            "adaptive_interval": self.adaptive_interval_enabled,
            "source_candidate_counts": dict(self._last_source_candidate_counts),
            "source_capped_counts": dict(self._last_source_capped_counts),
            "source_timings_ms": {k: round(v, 2) for k, v in self._last_source_timings_ms.items()},
        }
