"""
ARGUS Strategy Router — central dispatcher for all trading strategies.

Manages registration, enable/disable, and signal collection from all 15+
strategy implementations.  Each strategy call is wrapped in try/except with
a configurable timeout so one failure never blocks others.

The router also handles:
  - Signal type conversion (Signal, dict, ArbitrageOpportunity, etc. -> TradingSignal)
  - MacroEventFilter as a position-size multiplier (not a signal source)
  - MTFConfluence as a confirmation filter (boost/reduce confidence)
  - Signal deduplication and conflict resolution
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default timeout per strategy call (seconds)
_DEFAULT_TIMEOUT_S = 5.0


@dataclass
class StrategyStats:
    """Per-strategy runtime statistics."""
    calls: int = 0
    signals_produced: int = 0
    errors: int = 0
    timeouts: int = 0
    total_latency_ms: float = 0.0
    last_call_ts: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.calls, 1)


class StrategyRouter:
    """
    Central router that manages all ARGUS strategies.

    Usage::

        router = StrategyRouter(config)
        router.register("peak_alpha", peak_alpha_instance)
        signals = await router.generate_all_signals(symbol, ohlcv, regime, market_data)
    """

    # Strategy interface categories
    GENERATE_SIGNAL_STRATEGIES = frozenset({
        "peak_alpha", "mean_reversion", "momentum", "breakout", "scalping",
    })
    ANALYZE_STRATEGIES = frozenset({
        "stat_arb_cointegration", "market_maker",
    })
    # These have unique signatures handled individually
    SPECIAL_STRATEGIES = frozenset({
        "cross_exchange_arb", "futures_basis_arb", "delta_neutral_perp_arb",
        "volatility_arb", "liquidation_cascade", "deribit_options",
    })
    FILTER_STRATEGIES = frozenset({
        "macro_event_filter", "mtf_confluence",
    })

    def __init__(self, config: Any = None, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self.config = config
        self._timeout_s = timeout_s
        self._strategies: Dict[str, Any] = {}
        self._enabled: Dict[str, bool] = {}
        self._stats: Dict[str, StrategyStats] = {}
        self._macro_filter: Any = None  # MacroEventFilter instance
        self._mtf_filter: Any = None    # MTFConfluenceFilter instance

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, name: str, strategy_instance: Any, enabled: bool = True) -> None:
        """Register a strategy by name."""
        self._strategies[name] = strategy_instance
        self._enabled[name] = enabled
        self._stats[name] = StrategyStats()
        # Keep references to filter strategies for special handling
        if name == "macro_event_filter":
            self._macro_filter = strategy_instance
        elif name == "mtf_confluence":
            self._mtf_filter = strategy_instance
        logger.debug("StrategyRouter: registered '%s' (enabled=%s)", name, enabled)

    def enable(self, name: str) -> None:
        """Enable a strategy at runtime."""
        if name in self._strategies:
            self._enabled[name] = True

    def disable(self, name: str) -> None:
        """Disable a strategy at runtime."""
        if name in self._strategies:
            self._enabled[name] = False

    def get_active_strategies(self) -> List[str]:
        """Return list of enabled, non-filter strategy names."""
        return [
            n for n, enabled in self._enabled.items()
            if enabled and n not in self.FILTER_STRATEGIES
        ]

    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return per-strategy runtime statistics."""
        return {
            name: {
                "enabled": self._enabled.get(name, False),
                "calls": s.calls,
                "signals_produced": s.signals_produced,
                "errors": s.errors,
                "timeouts": s.timeouts,
                "avg_latency_ms": round(s.avg_latency_ms, 2),
            }
            for name, s in self._stats.items()
        }

    # ── Main entry point ─────────────────────────────────────────────────

    async def generate_all_signals(
        self,
        symbol: str,
        ohlcv: Any,
        regime: str,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Call every enabled strategy, collect signals, return unified list.

        Each strategy call is wrapped in try/except with a timeout.
        Returns list of TradingSignal (from unified_types).
        """
        from unified_types import TradingSignal

        market_data = market_data or {}
        raw_signals: List[Any] = []

        # Collect signals from all enabled non-filter strategies
        for name, strategy in self._strategies.items():
            if not self._enabled.get(name, False):
                continue
            if name in self.FILTER_STRATEGIES:
                continue

            stats = self._stats[name]
            t0 = time.perf_counter()
            stats.calls += 1
            stats.last_call_ts = time.time()

            try:
                result = await asyncio.wait_for(
                    self._call_strategy(name, strategy, symbol, ohlcv, regime, market_data),
                    timeout=self._timeout_s,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                stats.total_latency_ms += elapsed_ms

                if result is not None:
                    converted = self._convert_to_trading_signals(name, result, symbol, market_data)
                    stats.signals_produced += len(converted)
                    raw_signals.extend(converted)

            except asyncio.TimeoutError:
                stats.timeouts += 1
                logger.debug("StrategyRouter: '%s' timed out (%.1fs)", name, self._timeout_s)
            except Exception as exc:
                stats.errors += 1
                logger.debug("StrategyRouter: '%s' error: %s", name, exc)

        # Apply MacroEventFilter as size multiplier
        raw_signals = self._apply_macro_filter(raw_signals)

        # Apply MTFConfluence as confidence modifier
        raw_signals = self._apply_mtf_confluence(raw_signals, symbol, market_data)

        # Deduplicate and resolve conflicts
        raw_signals = self._deduplicate_signals(raw_signals)

        return raw_signals

    # ── Strategy dispatching ─────────────────────────────────────────────

    async def _call_strategy(
        self,
        name: str,
        strategy: Any,
        symbol: str,
        ohlcv: Any,
        regime: str,
        market_data: Dict[str, Any],
    ) -> Any:
        """Dispatch to the correct strategy method based on its interface."""

        # Standard generate_signal(symbol, ohlcv, regime) strategies
        if name in self.GENERATE_SIGNAL_STRATEGIES:
            method = getattr(strategy, "generate_signal", None)
            if method is not None:
                result = method(symbol, ohlcv, regime)
                if asyncio.iscoroutine(result):
                    return await result
                return result

        # analyze(market_data) strategies
        if name in self.ANALYZE_STRATEGIES:
            method = getattr(strategy, "analyze", None)
            if method is not None:
                md = dict(market_data)
                md.setdefault("symbol", symbol)
                md.setdefault("price", market_data.get("price", 0.0))
                result = method(md)
                if asyncio.iscoroutine(result):
                    return await result
                return result

        # Special strategies with unique signatures
        if name == "cross_exchange_arb":
            method = getattr(strategy, "generate_signals", None)
            if method is not None:
                return method()

        if name == "futures_basis_arb":
            method = getattr(strategy, "generate_signal", None)
            if method is not None:
                return method(symbol)

        if name == "delta_neutral_perp_arb":
            method = getattr(strategy, "evaluate", None)
            if method is not None:
                spot = float(market_data.get("price", 0.0))
                perp = float(market_data.get("perp_price", spot))
                funding = float(market_data.get("predicted_funding_bps", 0.0))
                time_h = float(market_data.get("funding_time_to_hours", 8.0))
                return method(symbol, spot, perp, funding, time_h)

        if name == "volatility_arb":
            method = getattr(strategy, "evaluate", None)
            if method is not None:
                return method(symbol)

        if name == "liquidation_cascade":
            method = getattr(strategy, "generate_signal", None)
            if method is not None:
                return method(symbol)

        if name == "deribit_options":
            method = getattr(strategy, "generate_signal", None)
            if method is not None:
                result = method()
                if asyncio.iscoroutine(result):
                    return await result
                return result

        # Fallback: try generate_signal(symbol, ohlcv, regime) then analyze(market_data)
        for mname in ("generate_signal", "analyze", "evaluate"):
            method = getattr(strategy, mname, None)
            if method is not None:
                try:
                    if mname == "generate_signal":
                        result = method(symbol, ohlcv, regime)
                    elif mname == "analyze":
                        result = method(market_data)
                    else:
                        result = method(symbol)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                except TypeError:
                    continue

        return None

    # ── Signal conversion ────────────────────────────────────────────────

    def _convert_to_trading_signals(
        self,
        strategy_name: str,
        result: Any,
        default_symbol: str,
        market_data: Dict[str, Any],
    ) -> List[Any]:
        """Convert strategy-specific return types to TradingSignal."""
        from unified_types import TradingSignal

        if result is None:
            return []

        # Handle lists (some strategies return multiple signals)
        if isinstance(result, list):
            signals = []
            for item in result:
                signals.extend(self._convert_to_trading_signals(strategy_name, item, default_symbol, market_data))
            return signals

        # core.types.Signal (from strategies like peak_alpha, mean_reversion, etc.)
        if hasattr(result, "action") and hasattr(result, "confidence") and hasattr(result, "entry_price"):
            action_str = str(getattr(result, "action", "HOLD"))
            # Handle SignalAction enum
            if hasattr(action_str, "value"):
                action_str = str(action_str.value)
            action_str = action_str.upper()
            if action_str in ("HOLD",):
                return []
            symbol = str(getattr(result, "symbol", default_symbol))
            return [TradingSignal(
                symbol=symbol,
                action=action_str,
                confidence=float(getattr(result, "confidence", 0.5)),
                strength=float(getattr(result, "strength", 0.5)),
                entry_price=float(getattr(result, "entry_price", 0.0)),
                stop_loss=getattr(result, "stop_loss", None),
                take_profit=getattr(result, "take_profit", None),
                reasoning=f"{strategy_name}: {getattr(result, 'reasoning', '')}",
            )]

        # Dict-based signals
        if isinstance(result, dict):
            # Market-maker dicts: may have bid_price/ask_price (check before action filter)
            if "bid_price" in result and "ask_price" in result:
                # Market maker produces two signals: bid (buy) and ask (sell)
                signals = []
                bid_price = float(result.get("bid_price", 0.0))
                ask_price = float(result.get("ask_price", 0.0))
                symbol = str(result.get("symbol", default_symbol))
                if bid_price > 0:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        action="BUY",
                        confidence=0.55,
                        strength=0.4,
                        entry_price=bid_price,
                        reasoning=f"{strategy_name}: market-maker bid",
                    ))
                if ask_price > 0:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        action="SELL",
                        confidence=0.55,
                        strength=0.4,
                        entry_price=ask_price,
                        reasoning=f"{strategy_name}: market-maker ask",
                    ))
                return signals
            # Generic dict signal
            action_str = str(result.get("action", "HOLD")).upper()
            if action_str in ("HOLD", "NEUTRAL"):
                return []
            symbol = str(result.get("symbol", default_symbol))
            price = float(result.get("price", 0.0) or result.get("entry_price", 0.0) or 0.0)
            return [TradingSignal(
                symbol=symbol,
                action=action_str,
                confidence=float(result.get("confidence", 0.5)),
                strength=float(result.get("strength", 0.5)),
                entry_price=price,
                stop_loss=result.get("stop_loss"),
                take_profit=result.get("take_profit"),
                reasoning=f"{strategy_name}: {result.get('reason', result.get('reasoning', ''))}",
            )]

        # ArbitrageOpportunity (cross_exchange_arb)
        if hasattr(result, "cheap_price") and hasattr(result, "expensive_price"):
            symbol = str(getattr(result, "symbol", default_symbol))
            # Buy on cheap exchange
            cheap_price = float(getattr(result, "cheap_price", 0.0))
            spread_bps = float(getattr(result, "net_spread_bps", 0.0))
            confidence = min(0.85, 0.50 + spread_bps / 100.0)
            return [TradingSignal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                strength=min(1.0, spread_bps / 50.0),
                entry_price=cheap_price,
                reasoning=f"{strategy_name}: arb spread {spread_bps:.1f}bps",
            )]

        # BasisOpportunity (futures_basis_arb)
        if hasattr(result, "annual_basis_pct") and hasattr(result, "spot_price"):
            symbol = str(getattr(result, "symbol", default_symbol))
            action_str = str(getattr(result, "action", "NEUTRAL")).upper()
            if action_str in ("NEUTRAL", "HOLD"):
                return []
            side = "BUY" if action_str in ("BUY_SPOT", "BUY", "LONG") else "SELL"
            basis_pct = float(getattr(result, "annual_basis_pct", 0.0))
            confidence = min(0.80, 0.50 + abs(basis_pct) / 30.0)
            return [TradingSignal(
                symbol=symbol,
                action=side,
                confidence=confidence,
                strength=min(1.0, abs(basis_pct) / 20.0),
                entry_price=float(getattr(result, "spot_price", 0.0)),
                reasoning=f"{strategy_name}: basis {basis_pct:.1f}% annualised",
            )]

        # ArbSignal (delta_neutral_perp_arb)
        if hasattr(result, "predicted_funding_bps") and hasattr(result, "basis_bps"):
            action_str = str(getattr(result, "action", "HOLD")).upper()
            if action_str in ("HOLD",):
                return []
            symbol = str(getattr(result, "symbol", default_symbol))
            side = "BUY" if action_str == "ENTER" else "SELL"
            funding_bps = float(getattr(result, "predicted_funding_bps", 0.0))
            confidence = min(0.75, 0.50 + abs(funding_bps) / 20.0)
            price = float(market_data.get("price", 0.0))
            return [TradingSignal(
                symbol=symbol,
                action=side,
                confidence=confidence,
                strength=0.5,
                entry_price=price,
                reasoning=f"{strategy_name}: {getattr(result, 'reason', '')}",
            )]

        # VolArbSignal (volatility_arb)
        if hasattr(result, "vol_premium_pct") and hasattr(result, "iv_pct"):
            action_str = str(getattr(result, "action", "HOLD")).upper()
            if action_str in ("HOLD",):
                return []
            symbol = str(getattr(result, "symbol", default_symbol))
            side = "SELL" if "SELL" in action_str else "BUY"
            premium = float(getattr(result, "vol_premium_pct", 0.0))
            confidence = min(0.75, 0.50 + abs(premium) / 30.0)
            price = float(market_data.get("price", 0.0))
            return [TradingSignal(
                symbol=symbol,
                action=side,
                confidence=confidence,
                strength=min(1.0, abs(premium) / 15.0),
                entry_price=price,
                reasoning=f"{strategy_name}: vol premium {premium:.1f}%",
            )]

        # LiquidationSignal (liquidation_cascade)
        if hasattr(result, "oi_drop_pct") and hasattr(result, "direction"):
            symbol = str(getattr(result, "symbol", default_symbol))
            direction = str(getattr(result, "direction", "")).upper()
            if direction not in ("BUY", "SELL"):
                return []
            price = float(market_data.get("price", 0.0))
            return [TradingSignal(
                symbol=symbol,
                action=direction,
                confidence=float(getattr(result, "confidence", 0.6)),
                strength=min(1.0, float(getattr(result, "oi_drop_pct", 0.0)) * 10.0),
                entry_price=price,
                reasoning=f"{strategy_name}: liquidation cascade detected",
            )]

        # OptionsSignal (deribit_options)
        if hasattr(result, "direction") and hasattr(result, "iv_percentile"):
            direction = str(getattr(result, "direction", "NEUTRAL")).upper()
            if direction == "NEUTRAL":
                return []
            symbol = str(getattr(result, "symbol", default_symbol))
            if "/" not in symbol:
                symbol = f"{symbol}/USD"
            side = "BUY" if direction == "BULLISH" else "SELL"
            price = float(market_data.get("price", 0.0))
            return [TradingSignal(
                symbol=symbol,
                action=side,
                confidence=float(getattr(result, "confidence", 0.5)),
                strength=0.5,
                entry_price=price,
                reasoning=f"{strategy_name}: {getattr(result, 'rationale', '')}",
            )]

        logger.debug(
            "StrategyRouter: unrecognised return type from '%s': %s",
            strategy_name, type(result).__name__,
        )
        return []

    # ── Filters ──────────────────────────────────────────────────────────

    def _apply_macro_filter(self, signals: List[Any]) -> List[Any]:
        """Apply MacroEventFilter as a position-size multiplier."""
        if self._macro_filter is None or not self._enabled.get("macro_event_filter", False):
            return signals
        try:
            multiplier = self._macro_filter.get_position_multiplier()
            if multiplier >= 1.0:
                return signals
            # multiplier = 0.0 means halt — drop all signals
            if multiplier <= 0.0:
                logger.debug("StrategyRouter: macro_event_filter halting all signals (multiplier=0)")
                return []
            # Apply as confidence reduction (proxy for size reduction)
            for sig in signals:
                current_conf = float(getattr(sig, "confidence", 0.5))
                new_conf = max(0.0, current_conf * multiplier)
                setattr(sig, "confidence", new_conf)
                old_reasoning = str(getattr(sig, "reasoning", ""))
                setattr(sig, "reasoning", f"{old_reasoning} [macro_mult={multiplier:.2f}]")
            logger.debug("StrategyRouter: macro filter applied multiplier=%.2f to %d signals", multiplier, len(signals))
        except Exception as exc:
            logger.debug("StrategyRouter: macro_event_filter error: %s", exc)
        return signals

    def _apply_mtf_confluence(
        self,
        signals: List[Any],
        symbol: str,
        market_data: Dict[str, Any],
    ) -> List[Any]:
        """Apply MTFConfluence as a confidence modifier."""
        if self._mtf_filter is None or not self._enabled.get("mtf_confluence", False):
            return signals
        for sig in signals:
            try:
                sig_action = str(getattr(sig, "action", "")).upper()
                if sig_action not in ("BUY", "SELL"):
                    continue
                direction = sig_action.lower()
                sig_symbol = str(getattr(sig, "symbol", symbol))
                # MTFConfluenceFilter.check() returns (passes, confluence_score, reason)
                passes, score, reason = self._mtf_filter.check(
                    sig_symbol, direction, market_data,
                )
                current_conf = float(getattr(sig, "confidence", 0.5))
                if passes:
                    # Boost confidence by 15% when MTF agrees
                    new_conf = min(1.0, current_conf * 1.15)
                else:
                    # Reduce confidence by 20% when MTF disagrees
                    new_conf = max(0.0, current_conf * 0.80)
                setattr(sig, "confidence", new_conf)
                old_reasoning = str(getattr(sig, "reasoning", ""))
                setattr(sig, "reasoning", f"{old_reasoning} [mtf:{'+' if passes else '-'}{score:.2f}]")
            except Exception as exc:
                logger.debug("StrategyRouter: mtf_confluence error: %s", exc)
        return signals

    # ── Deduplication & conflict resolution ──────────────────────────────

    def _deduplicate_signals(self, signals: List[Any]) -> List[Any]:
        """
        Deduplicate and resolve conflicts among signals for the same symbol.

        Rules:
        - If 2+ strategies say BUY same symbol: take highest confidence, boost +10%
        - If strategies disagree (BUY vs SELL): cancel both unless confidence gap > 0.3
        """
        if not signals:
            return signals

        from unified_types import TradingSignal

        # Group by symbol
        by_symbol: Dict[str, Dict[str, List[Any]]] = {}
        for sig in signals:
            sym = str(getattr(sig, "symbol", ""))
            action = str(getattr(sig, "action", "")).upper()
            if sym not in by_symbol:
                by_symbol[sym] = {}
            if action not in by_symbol[sym]:
                by_symbol[sym][action] = []
            by_symbol[sym][action].append(sig)

        result: List[Any] = []
        for sym, actions in by_symbol.items():
            buys = actions.get("BUY", [])
            sells = actions.get("SELL", [])

            # Handle conflict: both BUY and SELL for same symbol
            if buys and sells:
                best_buy = max(buys, key=lambda s: float(getattr(s, "confidence", 0)))
                best_sell = max(sells, key=lambda s: float(getattr(s, "confidence", 0)))
                buy_conf = float(getattr(best_buy, "confidence", 0))
                sell_conf = float(getattr(best_sell, "confidence", 0))
                gap = abs(buy_conf - sell_conf)
                if gap > 0.3:
                    # Keep the stronger signal
                    winner = best_buy if buy_conf > sell_conf else best_sell
                    result.append(winner)
                    logger.debug(
                        "StrategyRouter: conflict on %s — keeping %s (gap=%.2f)",
                        sym, getattr(winner, "action", ""), gap,
                    )
                else:
                    # Cancel both
                    logger.debug(
                        "StrategyRouter: conflict on %s — cancelling both (gap=%.2f)",
                        sym, gap,
                    )
                continue

            # Same-direction dedup: merge and boost
            for direction_signals in [buys, sells]:
                if not direction_signals:
                    continue
                if len(direction_signals) == 1:
                    result.append(direction_signals[0])
                else:
                    # Multiple strategies agree — take best confidence and boost +10%
                    best = max(direction_signals, key=lambda s: float(getattr(s, "confidence", 0)))
                    current_conf = float(getattr(best, "confidence", 0.5))
                    boosted = min(1.0, current_conf * 1.10)
                    setattr(best, "confidence", boosted)
                    sources = ", ".join(
                        str(getattr(s, "reasoning", "")).split(":")[0]
                        for s in direction_signals
                    )
                    old_reasoning = str(getattr(best, "reasoning", ""))
                    setattr(best, "reasoning", f"{old_reasoning} [consensus:{len(direction_signals)} sources: {sources}]")
                    result.append(best)

            # Pass through CLOSE and other actions
            for action, sigs in actions.items():
                if action not in ("BUY", "SELL"):
                    result.extend(sigs)

        return result
