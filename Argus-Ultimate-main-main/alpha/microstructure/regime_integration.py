"""
alpha/microstructure/regime_integration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RegimeIntegration — wires RegimeScheduler into the Argus startup sequence
and provides a single helper that any executor/strategy can call to get
full regime-aware quoting parameters.

This is the missing integration layer between RegimeScheduler (which
existed standalone) and the actual bot execution loop.

Usage (in run_ultimate.py / main.py startup)
--------------------------------------------
    from alpha.microstructure.regime_integration import RegimeIntegration

    regime_integration = RegimeIntegration.build(
        symbols=["BTC", "ETH", "SOL"],
        spread_schedule=session_spread_schedule,
    )

    # In strategy / executor before quoting:
    params = regime_integration.quote_params("BTC", base_spread_bps=10.0)
    if not params.should_quote:
        return  # halt quoting this tick

    spread_bps = params.effective_spread_bps  # regime + session adjusted

The RegimeDetector is automatically updated on every call to
`regime_integration.update(symbol, ohlcv_data)` which should be called
from the OHLCV/candle feed handler.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from alpha.regime_detector import MarketRegime, RegimeDetector
from alpha.regime_scheduler import QuoteParams, RegimeScheduler

logger = logging.getLogger(__name__)


class RegimeIntegration:
    """
    Single-entry-point facade for all regime-aware decisions.

    Owns a RegimeDetector and RegimeScheduler, keeps them synchronised,
    and provides the simple API that executors/strategies need.
    """

    def __init__(
        self,
        regime_detector: RegimeDetector,
        regime_scheduler: RegimeScheduler,
        symbols: List[str],
    ) -> None:
        self._detector = regime_detector
        self._scheduler = regime_scheduler
        self._symbols = [s.upper() for s in symbols]
        logger.info("RegimeIntegration initialised: symbols=%s", self._symbols)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        symbols: List[str],
        spread_schedule: Any,
    ) -> "RegimeIntegration":
        """Build a RegimeIntegration with default-configured sub-components.

        Parameters
        ----------
        symbols : list[str]
            Trading instruments to track.
        spread_schedule : SessionSpreadSchedule
            Clock-based spread schedule from execution layer.
        """
        detector = RegimeDetector()
        scheduler = RegimeScheduler(
            regime_detector=detector,
            spread_schedule=spread_schedule,
        )
        return cls(
            regime_detector=detector,
            regime_scheduler=scheduler,
            symbols=symbols,
        )

    # ── Update (call from OHLCV feed) ─────────────────────────────────────────

    def update(self, symbol: str, ohlcv: Any) -> None:
        """Feed new OHLCV data to the RegimeDetector.

        Call this from your candle/OHLCV feed handler so that regime
        classification stays current with market conditions.

        Parameters
        ----------
        symbol : str
            Trading symbol.
        ohlcv : Any
            OHLCV data in whatever format RegimeDetector.update() accepts.
        """
        try:
            self._detector.update(symbol, ohlcv)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RegimeIntegration.update error for %s: %s", symbol, exc)

    # ── Query ─────────────────────────────────────────────────────────────────

    def quote_params(self, symbol: str, base_spread_bps: float = 10.0) -> QuoteParams:
        """Return full regime-aware quoting parameters for *symbol*.

        This is the primary hot-path API.  Call before placing any quote.

        Parameters
        ----------
        symbol : str
            Trading symbol.
        base_spread_bps : float
            Base spread before regime/session multipliers.

        Returns
        -------
        QuoteParams
            .should_quote     : bool — False = halt all quoting this tick
            .effective_spread_bps : float — adjusted spread to use
            .regime           : MarketRegime
            .confidence       : float
            .multiplier       : float
            .reason           : str
        """
        return self._scheduler.get_quote_params(symbol, base_spread_bps)

    def should_quote(self, symbol: str) -> bool:
        """Quick boolean check — False = halt quoting immediately."""
        return self._scheduler.should_quote(symbol)

    def get_regime(self, symbol: str) -> MarketRegime:
        """Return current detected regime for *symbol*."""
        return self._detector.get_regime(symbol)

    def get_spread_multiplier(self, symbol: str) -> float:
        """Return total spread multiplier (session × regime)."""
        return self._scheduler.get_effective_spread_multiplier(symbol)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def session_stats(self) -> dict:
        """Return regime scheduler session statistics."""
        return self._scheduler.get_session_stats()

    def current_regimes(self) -> Dict[str, str]:
        """Return dict of symbol → regime name for all tracked symbols."""
        return {
            sym: self._detector.get_regime(sym).value
            for sym in self._symbols
        }
