"""
Pinnacle AI Brain – unified AI layer for the Argus trading system.

Delegates to the canonical StrategyEngine (RSI/BB/MACD + regime + tuner) for signal
generation. When the continuous scanner has no cached opportunities, the main loop
calls generate_trading_signals() here. Multi-agent and quantum extensions can be
added by subclassing or composing this brain.

BL View Bridge
--------------
After each signal-generation cycle, ``get_bl_views()`` maps the AI confidence
scores to the Black-Litterman view schema consumed by ``risk.black_litterman``:

    view = {
        "assets":     [symbol],
        "coeffs":     [1.0],
        "return":     expected_return,   # AI score * annualised scale factor
        "confidence": ai_confidence,     # clipped to (0.05, 0.95)
    }

The execution engine calls ``brain.get_bl_views()`` once per rebalance cycle
and passes the result to ``bl_weights(symbols, returns, views=views)``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BL View Bridge constants
# ---------------------------------------------------------------------------
# Scale factor: maps AI score [-1, +1] to an annualised expected-return view.
# 0.20 means a full-conviction (+1) signal is expressed as a +20 % p.a. view.
_BL_RETURN_SCALE: float = 0.20
# Minimum absolute AI score needed to include an asset as a BL view.
_BL_MIN_SCORE: float = 0.10


class PinnacleAIBrain:
    """
    Production AI brain: strategy engine + optional future agents.
    - generate_trading_signals(): returns list of TradingSignal (from strategy engine).
    - get_adaptation_status(): returns dict with strategy_engine.last_regime, tuner.
    - on_trade_closed(): feeds realized PnL back to strategy engine tuner.
    - get_bl_views(): converts latest signals to Black-Litterman view dicts.
    """

    def __init__(self, config: Any, *, market_data_service: Any = None) -> None:
        self.config = config
        self.market_data_service = market_data_service
        self.strategy_engine: Optional[Any] = None  # set in initialize()
        # Cache of last generated signals for BL view conversion
        self._last_signals: List[Any] = []

    async def initialize(self) -> None:
        """Create and wire the strategy engine."""
        try:
            from strategies.unified.strategy_engine import StrategyEngine
            self.strategy_engine = StrategyEngine(self.config)
            logger.debug("Pinnacle AI brain: strategy engine ready")
        except Exception as e:
            logger.warning("Pinnacle AI brain: strategy engine unavailable: %s", e)
            self.strategy_engine = None

    async def generate_trading_signals(self) -> List[Any]:
        """Generate trading signals via the strategy engine (unified_engine)."""
        if self.strategy_engine is None or self.market_data_service is None:
            return []
        try:
            signals = await self.strategy_engine.generate_signals(self.market_data_service)
            self._last_signals = signals  # cache for BL view bridge
            return signals
        except Exception as e:
            logger.debug("Pinnacle AI generate_trading_signals: %s", e)
            return []

    def get_adaptation_status(self) -> Dict[str, Any]:
        """Return adaptation status for allocator/risk (includes strategy_engine.last_regime)."""
        if self.strategy_engine is None:
            return {"enabled": False, "strategy_engine": None}
        return {
            "enabled": True,
            "strategy_engine": self.strategy_engine.get_adaptation_status(),
        }

    def on_trade_closed(
        self,
        *,
        symbol: str,
        pnl_pct: float,
        strategy: str,
        regime: str,
    ) -> None:
        """Feed realized PnL back to the strategy engine tuner."""
        if self.strategy_engine is None:
            return
        try:
            from adaptive.regime import MarketRegime
            reg = getattr(MarketRegime, regime.upper(), None) if regime else None
            if reg is None:
                logger.warning(
                    "unified_ai_brain.on_trade_closed: unknown regime string %r for symbol %s "
                    "(strategy=%s); defaulting to MarketRegime.RANGE. Check regime pipeline.",
                    regime, symbol, strategy,
                )
                reg = MarketRegime.RANGE
            self.strategy_engine.on_realized_pnl(symbol=symbol, pnl_pct=pnl_pct)
        except Exception as e:
            logger.warning("Pinnacle AI on_trade_closed failed for %s: %s", symbol, e)

    # -----------------------------------------------------------------------
    # Black-Litterman View Bridge
    # -----------------------------------------------------------------------

    def get_bl_views(self, *, return_scale: float = _BL_RETURN_SCALE) -> List[Dict[str, Any]]:
        """Convert cached AI signals into Black-Litterman view dicts.

        Called by the execution engine once per rebalance cycle, immediately
        before calling ``risk.black_litterman.bl_weights()``.

        The view schema matches ``BlackLittermanOptimizer.weights()``:

        .. code-block:: python

            {
                "assets":     ["BTC/USDT"],   # single-asset absolute view
                "coeffs":     [1.0],
                "return":     0.12,           # annualised expected return
                "confidence": 0.75,           # (0.05, 0.95)
            }

        Parameters
        ----------
        return_scale:
            Multiplier that maps a normalised AI score in [-1, +1] to an
            annualised expected return.  Default 0.20 → max view = ±20 % p.a.

        Returns
        -------
        list[dict]
            BL view dicts, one per symbol that has a signal above
            ``_BL_MIN_SCORE``.  Empty list when no signals are cached.
        """
        views: List[Dict[str, Any]] = []

        for sig in self._last_signals:
            try:
                symbol = getattr(sig, "symbol", None) or sig.get("symbol")
                if not symbol:
                    continue

                # Prefer a normalised score field; fall back to side * confidence
                score: float = 0.0
                if hasattr(sig, "score") and sig.score is not None:
                    score = float(sig.score)
                elif hasattr(sig, "confidence") and hasattr(sig, "side"):
                    side_mult = 1.0 if str(getattr(sig, "side", "buy")).lower() in (
                        "buy", "long", "1"
                    ) else -1.0
                    score = side_mult * float(getattr(sig, "confidence", 0.5))
                elif isinstance(sig, dict):
                    side_mult = 1.0 if str(sig.get("side", "buy")).lower() in (
                        "buy", "long", "1"
                    ) else -1.0
                    score = side_mult * float(sig.get("confidence", sig.get("score", 0.0)))

                if abs(score) < _BL_MIN_SCORE:
                    continue

                # Confidence is the *magnitude* of the AI score, clipped to
                # (0.05, 0.95) so Omega stays finite and non-degenerate.
                confidence = float(min(max(abs(score), 0.05), 0.95))
                expected_return = float(score * return_scale)

                views.append({
                    "assets":     [symbol],
                    "coeffs":     [1.0],
                    "return":     expected_return,
                    "confidence": confidence,
                })
            except Exception as exc:
                logger.debug("get_bl_views: skipping signal (%s): %s", sig, exc)

        logger.debug("BL view bridge: produced %d views from %d signals", len(views), len(self._last_signals))
        return views


class FallbackAIBrain(PinnacleAIBrain):
    """
    Optional-module fallback: when PinnacleAIBrain cannot be loaded (e.g. missing deps),
    the unified system uses this. Same interface; only StrategyEngine (no extra agents).
    """

    async def initialize(self) -> None:
        """Create and wire the strategy engine only."""
        try:
            from strategies.unified.strategy_engine import StrategyEngine
            self.strategy_engine = StrategyEngine(self.config)
            logger.info("Fallback AI brain: strategy engine ready (optional Pinnacle not loaded)")
        except Exception as e:
            logger.warning("Fallback AI brain: strategy engine unavailable: %s", e)
            self.strategy_engine = None
