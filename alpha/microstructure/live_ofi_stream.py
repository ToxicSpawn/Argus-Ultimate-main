"""LiveOFIStream — Push 40.

Computes Order Flow Imbalance (OFI) from a rolling trade/quote tape
and exposes a z-scored OFI value for the OFI_STREAM SignalGateway source.

OFI Definition
--------------
For each bar, OFI = sum of signed volume:
  +vol  when trade is a buy  (aggressor = buyer)
  -vol  when trade is a sell (aggressor = seller)

An alternative LOB-delta formulation is also supported:
  OFI_lob = (bid_size_delta) - (ask_size_delta)

Z-score
-------
  ofi_zscore = (ofi_current - mean(window)) / std(window)
  window default = 20 bars.

Usage
-----
  stream = LiveOFIStream(window=20)
  stream.on_trade({"side": "buy", "amount": 1.5})   # from WebSocket feed
  stream.on_book_delta({"bid_delta": 0.5, "ask_delta": -0.3})
  stream.close_bar()   # call once per candle
  z = stream.ofi_zscore
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OFISignal:
    """Order Flow Imbalance signal snapshot.
    
    Attributes
    ----------
    timestamp : float
        Unix timestamp of the signal
    ofi_value : float
        Raw OFI value (positive = buy pressure, negative = sell pressure)
    ofi_zscore : float
        Z-scored OFI value normalized over rolling window
    symbol : str
        Trading pair symbol (e.g., "BTC/USDT")
    confidence : float
        Signal confidence in [0, 1]
    """
    timestamp: float = field(default_factory=time.time)
    ofi_value: float = 0.0
    ofi_zscore: float = 0.0
    symbol: str = ""
    confidence: float = 0.0
    
    @property
    def direction(self) -> str:
        """Return 'buy', 'sell', or 'neutral' based on OFI z-score."""
        if self.ofi_zscore > 0.5:
            return "buy"
        elif self.ofi_zscore < -0.5:
            return "sell"
        return "neutral"

_DEFAULT_WINDOW = 20
_DEFAULT_LOB_WEIGHT = 0.5   # blend weight for LOB-delta OFI vs trade-tape OFI


class LiveOFIStream:
    """Real-time OFI z-score stream.

    Parameters
    ----------
    window      : Rolling window for z-score normalisation (default 20)
    lob_weight  : Blend weight for LOB-delta OFI [0=tape only, 1=LOB only]
    min_bars    : Minimum bars before emitting non-zero z-score (default 5)
    """

    def __init__(
        self,
        window: int = _DEFAULT_WINDOW,
        lob_weight: float = _DEFAULT_LOB_WEIGHT,
        min_bars: int = 5,
    ) -> None:
        self._window     = window
        self._lob_weight = float(np.clip(lob_weight, 0.0, 1.0))
        self._min_bars   = min_bars

        # Intra-bar accumulators
        self._bar_tape_ofi: float = 0.0   # signed volume from trade tape
        self._bar_lob_ofi:  float = 0.0   # LOB bid/ask delta
        self._bar_trade_count: int = 0

        # Rolling bar history
        self._ofi_history: Deque[float] = deque(maxlen=window)
        self._last_ofi:    float = 0.0
        self._ofi_zscore:  float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ofi_zscore(self) -> float:
        """Latest OFI z-score. Positive = buy pressure, negative = sell pressure."""
        return self._ofi_zscore

    @property
    def last_ofi(self) -> float:
        """Raw OFI value for the last closed bar."""
        return self._last_ofi

    @property
    def bar_trade_count(self) -> int:
        """Number of trades ingested in the current open bar."""
        return self._bar_trade_count

    def on_trade(self, trade: Dict) -> None:
        """Ingest a trade tick.

        Parameters
        ----------
        trade : dict with keys:
            'side'   : 'buy' or 'sell'
            'amount' : float (base currency volume)
        """
        side   = str(trade.get("side", "")).lower()
        amount = float(trade.get("amount", 0.0))
        if side == "buy":
            self._bar_tape_ofi += amount
        elif side == "sell":
            self._bar_tape_ofi -= amount
        self._bar_trade_count += 1

    def on_book_delta(self, delta: Dict) -> None:
        """Ingest an order book top-of-book delta.

        Parameters
        ----------
        delta : dict with keys:
            'bid_delta' : change in best bid size
            'ask_delta' : change in best ask size
        """
        bid_d = float(delta.get("bid_delta", 0.0))
        ask_d = float(delta.get("ask_delta", 0.0))
        self._bar_lob_ofi += bid_d - ask_d

    def close_bar(self) -> float:
        """Finalise the current bar, update z-score, reset accumulators.

        Returns
        -------
        float : OFI z-score for the closed bar.
        """
        # Blend tape and LOB OFI
        w = self._lob_weight
        bar_ofi = (1.0 - w) * self._bar_tape_ofi + w * self._bar_lob_ofi

        self._last_ofi = bar_ofi
        self._ofi_history.append(bar_ofi)

        if len(self._ofi_history) >= self._min_bars:
            arr  = np.array(self._ofi_history, dtype=float)
            mean = arr.mean()
            std  = arr.std()
            self._ofi_zscore = float((bar_ofi - mean) / std) if std > 1e-10 else 0.0
        else:
            self._ofi_zscore = 0.0

        logger.debug(
            "OFI bar closed | tape=%.4f lob=%.4f blended=%.4f zscore=%.4f trades=%d",
            self._bar_tape_ofi, self._bar_lob_ofi, bar_ofi,
            self._ofi_zscore, self._bar_trade_count,
        )

        # Reset accumulators
        self._bar_tape_ofi    = 0.0
        self._bar_lob_ofi     = 0.0
        self._bar_trade_count = 0

        return self._ofi_zscore

    def reset(self) -> None:
        """Full reset of all state."""
        self._bar_tape_ofi    = 0.0
        self._bar_lob_ofi     = 0.0
        self._bar_trade_count = 0
        self._ofi_history.clear()
        self._last_ofi   = 0.0
        self._ofi_zscore = 0.0

    def get_history(self) -> np.ndarray:
        """Return a copy of the OFI bar history array."""
        return np.array(self._ofi_history, dtype=float)
