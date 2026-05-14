"""
core/argus_ai_adapter.py
========================
ArgusAI → UnifiedExecutionEngine wiring bridge.

Responsibilities:
- Accept AI-generated signal dicts (from autonomous_brain / cognitive_engine)
  and normalise them into ExecutionRequest objects.
- Apply confidence gating before handing off to ExecutionEngine.
- Expose a single async entry-point: ArgusAIAdapter.dispatch(signal).
- Emit structured events onto the audit/event bus.

This module owns *no* ML logic and *no* order-routing logic.
Those concerns belong to autonomous_brain.py / cognitive_engine.py and
execution_engine.py respectively.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.execution_engine import ExecutionEngine, ExecutionRequest, ExecutionResult

logger = logging.getLogger("argus.core.argus_ai_adapter")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_CONFIDENCE: float = 0.55   # below this, signal is discarded
DEFAULT_MAX_CONCURRENCY: int = 4


# ---------------------------------------------------------------------------
# Signal schema (normalised AI output)
# ---------------------------------------------------------------------------

@dataclass
class AISignal:
    """
    Normalised signal produced by ArgusAI components.

    Fields map 1-to-1 to ExecutionRequest so the adapter conversion is
    explicit and traceable.
    """
    symbol: str
    side: str                           # "buy" | "sell"
    quantity: float
    confidence: float                   # 0.0 – 1.0
    strategy_name: str = "argus_ai"
    price: Optional[float] = None       # None → market order
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory: build from raw AI output dict
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "AISignal":
        """
        Build an AISignal from a raw dict produced by autonomous_brain /
        cognitive_engine / any other AI component.

        Tolerant of missing optional keys; raises ValueError for required ones.
        """
        required = ("symbol", "side", "quantity", "confidence")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"AISignal.from_dict: missing required keys {missing}")

        side = str(d["side"]).lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"AISignal.from_dict: invalid side '{side}' — must be 'buy' or 'sell'")

        return cls(
            symbol=str(d["symbol"]).upper(),
            side=side,
            quantity=float(d["quantity"]),
            confidence=float(d["confidence"]),
            strategy_name=str(d.get("strategy_name", "argus_ai")),
            price=float(d["price"]) if d.get("price") is not None else None,
            stop_loss=float(d["stop_loss"]) if d.get("stop_loss") is not None else None,
            take_profit=float(d["take_profit"]) if d.get("take_profit") is not None else None,
            meta=dict(d.get("meta") or {}),
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ArgusAIAdapter:
    """
    Bridge between ArgusAI signal producers and ExecutionEngine.

    Parameters
    ----------
    execution_engine:
        Configured ExecutionEngine instance (with router, risk, tracker).
    min_confidence:
        Signals below this threshold are discarded before reaching the engine.
    event_bus:
        Optional object with ``publish(topic: str, payload: dict)``.
        If provided, execution outcomes are broadcast for downstream consumers
        (dashboard, audit chain, etc.).
    max_concurrency:
        Max simultaneous orders when dispatching a batch.
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        event_bus: Any = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self._engine = execution_engine
        self._min_confidence = min_confidence
        self._bus = event_bus
        self._max_concurrency = max_concurrency

        # simple counters
        self._signals_received: int = 0
        self._signals_gated: int = 0
        self._signals_dispatched: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(self, signal: AISignal | dict) -> Optional[ExecutionResult]:
        """
        Normalise, gate, and execute a single AI signal.

        Parameters
        ----------
        signal:
            Either an AISignal dataclass or a raw dict that will be
            converted via AISignal.from_dict().

        Returns
        -------
        ExecutionResult if the signal passed gating, else None.
        """
        self._signals_received += 1

        # --- normalise ---
        if isinstance(signal, dict):
            try:
                signal = AISignal.from_dict(signal)
            except (ValueError, TypeError) as exc:
                logger.error("ArgusAIAdapter.dispatch: malformed signal dict — %s", exc)
                self._publish("ai.signal.malformed", {"error": str(exc)})
                return None

        # --- confidence gate ---
        if signal.confidence < self._min_confidence:
            self._signals_gated += 1
            logger.debug(
                "Signal GATED [conf=%.3f < threshold=%.3f] %s %s",
                signal.confidence, self._min_confidence, signal.side, signal.symbol,
            )
            self._publish("ai.signal.gated", {
                "symbol": signal.symbol,
                "side": signal.side,
                "confidence": signal.confidence,
                "threshold": self._min_confidence,
            })
            return None

        # --- build ExecutionRequest ---
        req = self._to_request(signal)

        # --- execute ---
        self._signals_dispatched += 1
        logger.info(
            "Dispatching AI signal → engine: %s %s qty=%.6f conf=%.3f [%s]",
            signal.side, signal.symbol, signal.quantity,
            signal.confidence, signal.strategy_name,
        )
        result = await self._engine.execute(req)

        # --- publish outcome ---
        self._publish_result(signal, result)
        return result

    async def dispatch_batch(
        self, signals: list[AISignal | dict]
    ) -> list[Optional[ExecutionResult]]:
        """
        Dispatch multiple signals concurrently with bounded concurrency.
        Preserves input order in returned results.
        """
        sem = asyncio.Semaphore(self._max_concurrency)

        async def _bounded(sig: AISignal | dict) -> Optional[ExecutionResult]:
            async with sem:
                return await self.dispatch(sig)

        return list(await asyncio.gather(*[_bounded(s) for s in signals]))

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_confidence_threshold(self, threshold: float) -> None:
        """Hot-update the minimum confidence gate."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        old = self._min_confidence
        self._min_confidence = threshold
        logger.info("Confidence threshold updated: %.3f → %.3f", old, threshold)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "signals_received": self._signals_received,
            "signals_gated": self._signals_gated,
            "signals_dispatched": self._signals_dispatched,
            "engine_stats": self._engine.stats,
        }

    def reset_stats(self) -> None:
        self._signals_received = 0
        self._signals_gated = 0
        self._signals_dispatched = 0
        self._engine.reset_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_request(signal: AISignal) -> ExecutionRequest:
        return ExecutionRequest(
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
            price=signal.price,
            strategy_name=signal.strategy_name,
            signal_confidence=signal.confidence,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            meta={**signal.meta, "_ai_dispatched": True, "_dispatch_ts": time.time()},
        )

    def _publish(self, topic: str, payload: dict) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(topic, payload)
        except Exception:
            logger.exception("event_bus.publish failed for topic '%s'", topic)

    def _publish_result(self, signal: AISignal, result: ExecutionResult) -> None:
        topic = "ai.execution.fill" if result.success else "ai.execution.failed"
        self._publish(topic, {
            "symbol": signal.symbol,
            "side": signal.side,
            "strategy": signal.strategy_name,
            "confidence": signal.confidence,
            "filled_qty": result.filled_quantity,
            "filled_price": result.filled_price,
            "fee": result.fee,
            "order_id": result.order_id,
            "latency_ms": result.latency_ms,
            "error": result.error,
        })


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_argus_ai_adapter(
    order_router: Any,
    risk_facade: Any = None,
    position_tracker: Any = None,
    event_bus: Any = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    dry_run: bool = False,
) -> ArgusAIAdapter:
    """
    One-shot factory used in full_wiring.py and tests.

    Creates an ExecutionEngine and wraps it in an ArgusAIAdapter.
    """
    engine = ExecutionEngine(
        order_router=order_router,
        risk_facade=risk_facade,
        position_tracker=position_tracker,
        dry_run=dry_run,
    )
    return ArgusAIAdapter(
        execution_engine=engine,
        min_confidence=min_confidence,
        event_bus=event_bus,
    )
