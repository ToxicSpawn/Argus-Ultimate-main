"""SignalPipeline — generate -> filter -> rank -> size signals.

Extracted from unified_trading_system.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A trading signal produced by a strategy."""
    symbol: str
    action: str  # 'BUY' | 'SELL' | 'HOLD'
    confidence: float  # 0.0 - 1.0
    quantity: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = self.action.upper()
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


SignalFilter = Callable[[Signal], bool]


class SignalGenerator(Protocol):
    """Protocol for anything that can generate signals."""
    async def generate_signals(self, market_data: Any) -> List[Signal]: ...


class SignalPipeline:
    """
    Composable signal pipeline: generate -> filter -> rank -> size.

    Usage:
        pipeline = SignalPipeline(config)
        pipeline.add_generator(my_strategy)
        pipeline.add_filter(lambda s: s.confidence >= 0.6)
        signals = await pipeline.run(market_data)
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._generators: List[SignalGenerator] = []
        self._filters: List[SignalFilter] = []
        self._min_confidence: float = float(getattr(config, "min_signal_confidence", 0.5) or 0.5)

    def add_generator(self, generator: SignalGenerator) -> "SignalPipeline":
        self._generators.append(generator)
        return self

    def add_filter(self, f: SignalFilter) -> "SignalPipeline":
        self._filters.append(f)
        return self

    async def run(self, market_data: Any) -> List[Signal]:
        """Run the full pipeline and return filtered, ranked signals."""
        raw: List[Signal] = []

        for gen in self._generators:
            try:
                signals = await gen.generate_signals(market_data)
                raw.extend(signals or [])
            except Exception as exc:
                logger.exception("Signal generator %s error: %s", gen.__class__.__name__, exc)

        # Built-in confidence floor
        filtered = [s for s in raw if s.confidence >= self._min_confidence]

        # User-defined filters
        for f in self._filters:
            filtered = [s for s in filtered if self._safe_filter(f, s)]

        # Rank by confidence descending
        ranked = sorted(filtered, key=lambda s: s.confidence, reverse=True)

        logger.debug(
            "SignalPipeline: %d raw -> %d filtered -> %d ranked",
            len(raw), len(filtered), len(ranked),
        )
        return ranked

    def _safe_filter(self, f: SignalFilter, s: Signal) -> bool:
        try:
            return bool(f(s))
        except Exception as exc:
            logger.exception("Signal filter error: %s", exc)
            return True  # fail-open: don't drop signal on filter crash
