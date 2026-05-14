"""LiveVPINStream — Push 40.

Computes Volume-synchronised Probability of Informed Trading (VPIN)
from an incoming trade tape using the Easley-Lopez de Prado-O'Hara
bucket-based methodology.

VPIN Algorithm
--------------
1. Classify each trade as buy or sell using the tick rule:
     price > prev_price  -> buy
     price < prev_price  -> sell
     price == prev_price -> carry forward previous side
2. Accumulate trades into volume buckets of size V_bucket.
3. Per bucket:  tau = |buy_vol - sell_vol| / V_bucket
4. VPIN = rolling mean of tau over last n_buckets (default 50).

VPIN Interpretation
-------------------
  VPIN close to 1.0 -> high toxic / informed order flow -> volatile / directional
  VPIN close to 0.0 -> balanced order flow -> mean-reversion / low vol

For the VPIN_STREAM gateway source:
  VPIN > 0.65 -> short bias (informed sellers dominate)
  VPIN < 0.35 -> long bias  (informed buyers dominate)
  else        -> flat

Usage
-----
  stream = LiveVPINStream(bucket_size=50, n_buckets=50)
  stream.on_trade({"price": 50000.5, "amount": 0.1})
  v = stream.vpin
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET_SIZE = 50.0   # volume per bucket (BTC equivalent)
_DEFAULT_N_BUCKETS   = 50     # rolling window of buckets for VPIN
_DEFAULT_INIT_VPIN   = 0.5    # neutral starting VPIN before enough buckets


class LiveVPINStream:
    """Real-time VPIN calculator from trade tape.

    Parameters
    ----------
    bucket_size : Volume per bucket in base currency (default 50.0 BTC)
    n_buckets   : Rolling window of buckets for VPIN mean (default 50)
    """

    def __init__(
        self,
        bucket_size: float = _DEFAULT_BUCKET_SIZE,
        n_buckets:   int   = _DEFAULT_N_BUCKETS,
    ) -> None:
        self._bucket_size  = float(bucket_size)
        self._n_buckets    = n_buckets

        # Bucket accumulators
        self._bucket_buy_vol:  float = 0.0
        self._bucket_sell_vol: float = 0.0
        self._bucket_vol:      float = 0.0

        # History
        self._tau_history: Deque[float] = deque(maxlen=n_buckets)
        self._vpin_value:  float = _DEFAULT_INIT_VPIN

        # Tick rule state
        self._prev_price:  float = 0.0
        self._prev_side:   str   = "buy"   # carry-forward

        self._total_trades: int = 0
        self._total_buckets: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vpin(self) -> float:
        """Current VPIN in [0, 1]. 0.5 = neutral."""
        return self._vpin_value

    @property
    def total_buckets(self) -> int:
        return self._total_buckets

    @property
    def total_trades(self) -> int:
        return self._total_trades

    def on_trade(self, trade: Dict) -> None:
        """Ingest a trade tick.

        Parameters
        ----------
        trade : dict with keys:
            'price'  : float
            'amount' : float (base currency volume)
            'side'   : optional str 'buy'|'sell' (overrides tick rule if present)
        """
        price  = float(trade.get("price", 0.0))
        amount = float(trade.get("amount", 0.0))
        if amount <= 0:
            return

        # Classify trade direction
        side = trade.get("side", None)
        if side not in ("buy", "sell"):
            side = self._tick_rule(price)
        self._prev_price = price
        self._prev_side  = side
        self._total_trades += 1

        # Distribute across buckets (a single trade may fill multiple)
        remaining = amount
        while remaining > 0:
            space = self._bucket_size - self._bucket_vol
            fill  = min(remaining, space)

            if side == "buy":
                self._bucket_buy_vol  += fill
            else:
                self._bucket_sell_vol += fill
            self._bucket_vol += fill
            remaining -= fill

            if self._bucket_vol >= self._bucket_size - 1e-9:
                self._close_bucket()

    def reset(self) -> None:
        """Full reset."""
        self._bucket_buy_vol  = 0.0
        self._bucket_sell_vol = 0.0
        self._bucket_vol      = 0.0
        self._tau_history.clear()
        self._vpin_value   = _DEFAULT_INIT_VPIN
        self._prev_price   = 0.0
        self._prev_side    = "buy"
        self._total_trades = 0
        self._total_buckets = 0

    def get_tau_history(self) -> np.ndarray:
        return np.array(self._tau_history, dtype=float)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _close_bucket(self) -> None:
        tau = abs(self._bucket_buy_vol - self._bucket_sell_vol) / self._bucket_size
        tau = float(np.clip(tau, 0.0, 1.0))
        self._tau_history.append(tau)
        self._total_buckets += 1

        if len(self._tau_history) >= 1:
            self._vpin_value = float(np.mean(self._tau_history))

        logger.debug(
            "VPIN bucket #%d | buy=%.4f sell=%.4f tau=%.4f vpin=%.4f",
            self._total_buckets,
            self._bucket_buy_vol, self._bucket_sell_vol,
            tau, self._vpin_value,
        )

        self._bucket_buy_vol  = 0.0
        self._bucket_sell_vol = 0.0
        self._bucket_vol      = 0.0

    def _tick_rule(self, price: float) -> str:
        if self._prev_price == 0.0:
            return self._prev_side
        if price > self._prev_price:
            return "buy"
        if price < self._prev_price:
            return "sell"
        return self._prev_side   # uptick/downtick carry
