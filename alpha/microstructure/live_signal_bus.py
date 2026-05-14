"""
alpha/microstructure/live_signal_bus.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LiveSignalBus — singleton aggregator that continuously feeds
OFI, VPIN, MicropriceDrift and DeepLOB signals from the live
WebSocket feed into a single shared in-memory dict.

This is the missing glue that activates the entire dormant
microstructure stack.  Before this module, all four signals
were always 0.0 at execution time despite the stream classes
existing.  After this module, every place_order() call has
real live values for all four signal dimensions.

Architecture
------------
  LiveSignalBus.create(symbols, hl_client)
    └─ starts background coroutine: _run_loop()
         ├─ WS trade tick  → OFI.on_trade()  + VPIN.on_trade()
         ├─ WS book update → OFI.on_book_update() + MicropriceDrift.on_book_update()
         │                 → DeepLOBLiveBridge.on_book_update()
         └─ every update   → _snapshot dict updated (O(1) read)

Usage
-----
    bus = await LiveSignalBus.create(symbols=["BTC", "ETH"], client=hl_client)
    await bus.start()

    # In executor / strategy:
    sig = bus.get("BTC")
    result = await executor.place_order(
        "BTC", "buy", 0.001,
        ofi_signal=sig.ofi_zscore,
        vpin=sig.vpin,
        microprice_drift=sig.microprice_drift_magnitude,
    )
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from alpha.microstructure.live_ofi_stream import LiveOFIStream, OFISignal
from alpha.microstructure.live_vpin_stream import LiveVPINStream
from alpha.microstructure.microprice_drift import MicropriceDriftSignal

logger = logging.getLogger(__name__)

# Optional DeepLOB — gracefully absent if model weights not loaded
try:
    from alpha.microstructure.deeplob_live_bridge import DeepLOBLiveBridge
    _DEEPLOB_AVAILABLE = True
except Exception:  # noqa: BLE001
    _DEEPLOB_AVAILABLE = False
    DeepLOBLiveBridge = None  # type: ignore[assignment,misc]


# ─── Live signal snapshot ─────────────────────────────────────────────────────


@dataclass
class LiveSignal:
    """Complete live microstructure signal snapshot for one symbol.

    All fields are continuously updated by LiveSignalBus on every WS tick.
    Reads are O(1) and lock-free (dataclass replace semantics).
    """
    symbol: str

    # OFI
    ofi: float = 0.0                    # [-1, 1] buy/sell volume imbalance
    ofi_zscore: float = 0.0             # z-score of OFI over recent history
    signed_imbalance: float = 0.5       # [0,1] fraction of buy volume
    aggressive_ratio: float = 0.0       # aggressive buy/sell ratio

    # VPIN
    vpin: float = 0.0                   # [0, 1] flow toxicity
    vpin_alert: bool = False            # True when vpin > 0.7
    vpin_trend: str = "stable"          # "rising" | "falling" | "stable"

    # MicropriceDrift
    microprice_drift: float = 0.5       # [0,1] fraction of snapshots micro > mid
    microprice_drift_magnitude: float = 0.0  # [-1, 1] signed drift
    microprice_quote_skew: float = 0.0  # [-2, +2] ticks skew recommendation
    microprice_signal: str = "neutral"  # "bullish" | "bearish" | "neutral"

    # DeepLOB (optional)
    deeplob_prediction: float = 0.0     # [-1=down, 0=neutral, 1=up]
    deeplob_confidence: float = 0.0     # model confidence [0, 1]
    deeplob_available: bool = False

    # Meta
    last_updated_ns: int = field(default_factory=time.time_ns)
    ticks_processed: int = 0
    book_updates_processed: int = 0

    @property
    def is_stale(self, max_age_ms: float = 5_000.0) -> bool:
        """True if this snapshot has not been updated in max_age_ms milliseconds."""
        age_ms = (time.time_ns() - self.last_updated_ns) / 1_000_000.0
        return age_ms > max_age_ms

    @property
    def composite_toxicity(self) -> float:
        """Combined flow toxicity score [0, 1].

        Weights:
          VPIN           0.40  (direct toxicity measure)
          |OFI z-score|  0.30  (normalised flow pressure)
          |DeepLOB|      0.30  (if available)
        """
        vpin_component = self.vpin * 0.40
        ofi_component = min(1.0, abs(self.ofi_zscore) / 3.0) * 0.30
        if self.deeplob_available:
            deep_component = abs(self.deeplob_prediction) * self.deeplob_confidence * 0.30
        else:
            deep_component = min(1.0, abs(self.ofi_zscore) / 3.0) * 0.30
        return min(1.0, vpin_component + ofi_component + deep_component)


# ─── Null signal (safe default before any data arrives) ──────────────────────


_NULL_SIGNAL_CACHE: Dict[str, LiveSignal] = {}


def _null_signal(symbol: str) -> LiveSignal:
    """Return a cached zero-state LiveSignal for *symbol*."""
    if symbol not in _NULL_SIGNAL_CACHE:
        _NULL_SIGNAL_CACHE[symbol] = LiveSignal(symbol=symbol)
    return _NULL_SIGNAL_CACHE[symbol]


# ─── LiveSignalBus ────────────────────────────────────────────────────────────


class LiveSignalBus:
    """
    Singleton that aggregates all live microstructure streams.

    Creates and owns:
      - LiveOFIStream
      - LiveVPINStream
      - MicropriceDriftSignal
      - DeepLOBLiveBridge (optional)

    Provides:
      - bus.get(symbol)  → LiveSignal (O(1), no lock)
      - bus.start()      → launches background update loop
      - bus.stop()       → graceful shutdown
      - bus.health()     → dict with per-symbol staleness + counts
    """

    _instance: Optional["LiveSignalBus"] = None

    def __init__(
        self,
        symbols: List[str],
        ofi_window: int = 500,
        ofi_alpha: float = 0.94,
        vpin_bucket_volume: float = 1.0,
        vpin_n_buckets: int = 50,
        drift_window: int = 20,
        drift_threshold: float = 0.6,
        enable_deeplob: bool = True,
    ) -> None:
        self._symbols = [s.upper() for s in symbols]

        # Initialise all streams
        self._ofi = LiveOFIStream(window_trades=ofi_window, alpha=ofi_alpha)
        self._vpin = LiveVPINStream(bucket_volume=vpin_bucket_volume, n_buckets=vpin_n_buckets)
        self._drift = MicropriceDriftSignal(window=drift_window, threshold=drift_threshold)

        self._deeplob: Optional[object] = None
        if enable_deeplob and _DEEPLOB_AVAILABLE:
            try:
                self._deeplob = DeepLOBLiveBridge()
                logger.info("LiveSignalBus: DeepLOB bridge loaded")
            except Exception as exc:  # noqa: BLE001
                logger.warning("LiveSignalBus: DeepLOB unavailable — %s", exc)

        # Snapshot dict: symbol → LiveSignal (atomic replace on update)
        self._signals: Dict[str, LiveSignal] = {
            sym: LiveSignal(symbol=sym) for sym in self._symbols
        }

        self._running = False
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

        logger.info(
            "LiveSignalBus created: symbols=%s deeplob=%s",
            self._symbols,
            self._deeplob is not None,
        )

    # ── Singleton factory ─────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "LiveSignalBus":
        """Return the singleton instance; raises if not yet created."""
        if cls._instance is None:
            raise RuntimeError(
                "LiveSignalBus not initialised. "
                "Call LiveSignalBus.create() first."
            )
        return cls._instance

    @classmethod
    async def create(
        cls,
        symbols: List[str],
        **kwargs,
    ) -> "LiveSignalBus":
        """Create singleton, start background loop, return bus."""
        if cls._instance is not None:
            logger.warning("LiveSignalBus.create() called twice — returning existing instance")
            return cls._instance
        bus = cls(symbols=symbols, **kwargs)
        cls._instance = bus
        await bus.start()
        return bus

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background signal maintenance loop."""
        if self._running:
            return
        self._running = True
        logger.info("LiveSignalBus: started (symbols=%s)", self._symbols)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("LiveSignalBus: stopped")

    # ── Feed ingestion (call from WS handlers) ────────────────────────────────

    def on_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Ingest a trade tick — call from WS trade handler.

        This is synchronous and O(1) on the hot path.
        Updates OFI and VPIN streams, then refreshes the snapshot.
        """
        sym = symbol.upper()
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()

        self._ofi.on_trade(sym, side, size, price, ts)
        self._vpin.on_trade(sym, side, size, price, ts)

        # Refresh snapshot
        prev = self._signals.get(sym) or _null_signal(sym)
        self._signals[sym] = self._build_snapshot(
            sym,
            ticks=prev.ticks_processed + 1,
            books=prev.book_updates_processed,
        )

    def on_book_update(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Ingest a LOB snapshot — call from WS book handler.

        Updates OFI book state, MicropriceDrift, and optionally DeepLOB.
        """
        sym = symbol.upper()
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()

        self._ofi.on_book_update(sym, bids, asks, ts)

        # Compute microprice for drift signal
        if bids and asks:
            best_bid_p, best_bid_s = float(bids[0][0]), float(bids[0][1])
            best_ask_p, best_ask_s = float(asks[0][0]), float(asks[0][1])
            microprice = MicropriceDriftSignal.compute_microprice(
                best_bid_p, best_ask_p, best_bid_s, best_ask_s
            )
            mid = (best_bid_p + best_ask_p) / 2.0
            self._drift.on_book_update(sym, microprice, mid, ts)

        # Feed DeepLOB if available
        if self._deeplob is not None:
            try:
                self._deeplob.on_book_update(sym, bids, asks, ts)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.debug("DeepLOB on_book_update error: %s", exc)

        prev = self._signals.get(sym) or _null_signal(sym)
        self._signals[sym] = self._build_snapshot(
            sym,
            ticks=prev.ticks_processed,
            books=prev.book_updates_processed + 1,
        )

    # ── Signal reads (O(1) no-lock) ───────────────────────────────────────────

    def get(self, symbol: str) -> LiveSignal:
        """Return the latest LiveSignal for *symbol*.

        Returns a zero-state signal if the symbol is unknown or no data yet.
        This is the hot-path read method — O(1), no locks, safe from any
        async context or thread.
        """
        sym = symbol.upper()
        return self._signals.get(sym) or _null_signal(sym)

    def get_all(self) -> Dict[str, LiveSignal]:
        """Return a shallow copy of the full signal dict."""
        return dict(self._signals)

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Return health dict for monitoring / dashboard."""
        now_ns = time.time_ns()
        result = {}
        for sym, sig in self._signals.items():
            age_ms = (now_ns - sig.last_updated_ns) / 1_000_000.0
            result[sym] = {
                "age_ms": round(age_ms, 2),
                "stale": age_ms > 5_000.0,
                "ticks": sig.ticks_processed,
                "books": sig.book_updates_processed,
                "ofi_zscore": round(sig.ofi_zscore, 4),
                "vpin": round(sig.vpin, 4),
                "vpin_alert": sig.vpin_alert,
                "drift_signal": sig.microprice_signal,
                "composite_toxicity": round(sig.composite_toxicity, 4),
                "deeplob_available": sig.deeplob_available,
            }
        return result

    # ── Private snapshot builder ──────────────────────────────────────────────

    def _build_snapshot(self, symbol: str, ticks: int, books: int) -> LiveSignal:
        """Construct a fresh LiveSignal from all stream states."""
        ofi_sig: OFISignal = self._ofi.get_signal(symbol)
        vpin_val = self._vpin.get_vpin(symbol)
        vpin_stats = self._vpin.get_stats(symbol)
        drift_state = self._drift.get_drift_state(symbol)

        # DeepLOB
        deep_pred = 0.0
        deep_conf = 0.0
        deep_avail = False
        if self._deeplob is not None:
            try:
                pred = self._deeplob.get_prediction(symbol)  # type: ignore[attr-defined]
                if pred is not None:
                    deep_pred = float(getattr(pred, "direction", 0.0))
                    deep_conf = float(getattr(pred, "confidence", 0.0))
                    deep_avail = True
            except Exception:  # noqa: BLE001
                pass

        return LiveSignal(
            symbol=symbol,
            # OFI
            ofi=ofi_sig.ofi,
            ofi_zscore=ofi_sig.ofi_zscore,
            signed_imbalance=ofi_sig.signed_imbalance,
            aggressive_ratio=ofi_sig.aggressive_ratio,
            # VPIN
            vpin=vpin_val,
            vpin_alert=vpin_stats.get("alert", False),
            vpin_trend=vpin_stats.get("trend", "stable"),
            # MicropriceDrift
            microprice_drift=drift_state.drift,
            microprice_drift_magnitude=drift_state.magnitude,
            microprice_quote_skew=drift_state.quote_skew,
            microprice_signal=drift_state.signal,
            # DeepLOB
            deeplob_prediction=deep_pred,
            deeplob_confidence=deep_conf,
            deeplob_available=deep_avail,
            # Meta
            last_updated_ns=time.time_ns(),
            ticks_processed=ticks,
            book_updates_processed=books,
        )
