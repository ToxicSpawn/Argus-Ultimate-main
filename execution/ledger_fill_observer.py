"""
Push 87 — LedgerFillObserver
=============================
Mixin / wrapper that attaches a TradeLedger to a FillTracker so that every
call to FillTracker.record_fill() is automatically forwarded to the ledger.

Two integration patterns are supported:

1. **Mixin** — subclass FillTracker and mix in LedgerFillObserver::

       class InstrumentedFillTracker(LedgerFillObserverMixin, FillTracker):
           pass

       tracker = InstrumentedFillTracker(ledger=TradeLedger())

2. **Wrapper** (preferred for existing instances)::

       tracker  = FillTracker()
       ledger   = TradeLedger()
       observer = LedgerFillObserver(tracker, ledger)

       # Use observer.record_fill() everywhere; it delegates to tracker
       # and then posts to the ledger transparently.

Fee estimation
--------------
If the caller does not supply fee_usd, a conservative flat rate of
``DEFAULT_FEE_BPS`` basis points is applied to the notional.
"""
from __future__ import annotations

import logging
from typing import Optional

from execution.fill_tracker import FillRecord, FillTracker
from execution.trade_ledger import TradeLedger

logger = logging.getLogger(__name__)

# Conservative default taker fee (3 bps = 0.03 %)
DEFAULT_FEE_BPS: float = 3.0


# ---------------------------------------------------------------------------
# Standalone wrapper (preferred)
# ---------------------------------------------------------------------------

class LedgerFillObserver:
    """
    Thin wrapper around FillTracker that posts every confirmed fill to a
    TradeLedger immediately after recording slippage.

    All FillTracker methods are proxied transparently so this can be used
    as a drop-in replacement wherever a FillTracker is expected.
    """

    def __init__(
        self,
        tracker: FillTracker,
        ledger: TradeLedger,
        default_fee_bps: float = DEFAULT_FEE_BPS,
    ) -> None:
        self._tracker = tracker
        self._ledger = ledger
        self._default_fee_bps = default_fee_bps
        logger.info(
            "LedgerFillObserver attached: tracker=%r ledger=%r fee_bps=%.1f",
            tracker, ledger, default_fee_bps,
        )

    # ------------------------------------------------------------------
    # Intercepted method
    # ------------------------------------------------------------------

    def record_fill(
        self,
        strategy: str,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
        quantity_usd: float,
        exchange: str = "kraken",
        fee_usd: Optional[float] = None,
    ) -> FillRecord:
        """
        Record fill in FillTracker then post to TradeLedger.

        Parameters are identical to FillTracker.record_fill() with one
        optional addition: ``fee_usd``.  If omitted, a fee is estimated
        from DEFAULT_FEE_BPS applied to quantity_usd.
        """
        record = self._tracker.record_fill(
            strategy=strategy,
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            actual_price=actual_price,
            quantity_usd=quantity_usd,
            exchange=exchange,
        )

        estimated_fee = (
            fee_usd
            if fee_usd is not None
            else quantity_usd * self._default_fee_bps / 10_000.0
        )

        try:
            self._ledger.post(
                fill_id=record.fill_id,
                strategy=strategy,
                symbol=symbol,
                side=side,
                quantity_usd=quantity_usd,
                fill_price=actual_price,
                fee_usd=estimated_fee,
                exchange=exchange,
                timestamp=record.timestamp,
            )
        except Exception:
            # Ledger write failure must never break the trading path
            logger.exception(
                "LedgerFillObserver: ledger.post failed for fill %s — "
                "fill is recorded in FillTracker but NOT in ledger",
                record.fill_id,
            )

        return record

    # ------------------------------------------------------------------
    # Proxy everything else to the underlying tracker
    # ------------------------------------------------------------------

    def is_within_budget(self, strategy: str) -> bool:
        return self._tracker.is_within_budget(strategy)

    def get_budget(self, strategy: str):
        return self._tracker.get_budget(strategy)

    def reset_budgets(self) -> None:
        self._tracker.reset_budgets()

    def get_strategy_stats(self, strategy: str, lookback_hours: float = 24):
        return self._tracker.get_strategy_stats(strategy, lookback_hours)

    def get_all_stats(self, lookback_hours: float = 24):
        return self._tracker.get_all_stats(lookback_hours)

    # Expose underlying objects for introspection
    @property
    def tracker(self) -> FillTracker:
        return self._tracker

    @property
    def ledger(self) -> TradeLedger:
        return self._ledger


# ---------------------------------------------------------------------------
# Mixin (for subclassing FillTracker)
# ---------------------------------------------------------------------------

class LedgerFillObserverMixin:
    """
    Mixin to be used with FillTracker subclasses.  Requires the subclass
    to pass ``ledger`` as a keyword argument to __init__.

    Example::

        class InstrumentedFillTracker(LedgerFillObserverMixin, FillTracker):
            pass

        tracker = InstrumentedFillTracker(
            ledger=TradeLedger(),
            db_path="data/fills.db",
        )
    """

    def __init__(self, *args, ledger: TradeLedger, default_fee_bps: float = DEFAULT_FEE_BPS, **kwargs) -> None:  # type: ignore[override]
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]
        self._ledger: TradeLedger = ledger
        self._default_fee_bps: float = default_fee_bps

    def record_fill(  # type: ignore[override]
        self,
        strategy: str,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
        quantity_usd: float,
        exchange: str = "kraken",
        fee_usd: Optional[float] = None,
    ) -> FillRecord:
        record: FillRecord = super().record_fill(  # type: ignore[misc]
            strategy=strategy,
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            actual_price=actual_price,
            quantity_usd=quantity_usd,
            exchange=exchange,
        )
        estimated_fee = (
            fee_usd
            if fee_usd is not None
            else quantity_usd * self._default_fee_bps / 10_000.0
        )
        try:
            self._ledger.post(
                fill_id=record.fill_id,
                strategy=strategy,
                symbol=symbol,
                side=side,
                quantity_usd=quantity_usd,
                fill_price=actual_price,
                fee_usd=estimated_fee,
                exchange=exchange,
                timestamp=record.timestamp,
            )
        except Exception:
            logger.exception(
                "LedgerFillObserverMixin: ledger.post failed for fill %s",
                record.fill_id,
            )
        return record
