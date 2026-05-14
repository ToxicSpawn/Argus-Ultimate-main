"""
CrossAssetRegime — BTC dominance + ETH/BTC ratio regime signal.

Derives a market-regime signal from two cross-asset relationships:

1. BTC Dominance (BTC.D)
   - Rising dominance  -> risk-off / BTC season -> favour BTC longs
   - Falling dominance -> alt season / risk-on  -> favour alt exposure

2. ETH/BTC Ratio
   - Rising ETH/BTC    -> ETH outperformance   -> alt-season indicator
   - Falling ETH/BTC   -> BTC outperformance   -> BTC-season indicator

Combined output
---------------
  regime_signal : float in [-1, 1]
      +1  =>  strong BTC-season / risk-off
      -1  =>  strong alt-season / risk-on
       0  =>  neutral / transitioning

  regime_label  : str  one of BTC_SEASON | ALT_SEASON | NEUTRAL

Usage
-----
    ca = CrossAssetRegime()
    asyncio.run(ca.fetch())
    print(ca.regime_label, ca.regime_signal)
    scalar = ca.get_scalar(candles)   # regime-adaptive risk scalar
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable endpoints (CoinGecko — no API key for basic usage)
# ---------------------------------------------------------------------------
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
COINGECKO_PRICE_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=ethereum,bitcoin&vs_currencies=usd"
)
DEFAULT_TIMEOUT_SEC = 10
STALE_THRESHOLD_SEC = 300
HISTORY_LEN = 100

# Dominance thresholds
BTC_DOM_HIGH = 55.0     # above this -> BTC season
BTC_DOM_LOW  = 45.0     # below this -> alt season

# ETH/BTC ratio momentum window (bars)
MOMENTUM_WINDOW = 10

# Regime-adaptive risk scalars
_SCALAR_HIGH_VOL = 0.5   # BTC_SEASON  -> cut risk 50%
_SCALAR_TRENDING = 1.5   # ALT_SEASON  -> scale risk up 1.5x
_SCALAR_NEUTRAL  = 1.0   # NEUTRAL     -> no adjustment


class RegimeLabel:
    BTC_SEASON = "BTC_SEASON"
    ALT_SEASON = "ALT_SEASON"
    NEUTRAL    = "NEUTRAL"


@dataclass
class CrossAssetSnapshot:
    btc_dominance: float            # BTC market-cap dominance (%)
    eth_btc_ratio: float            # ETH price / BTC price
    regime_signal: float            # combined signal [-1, 1]
    regime_label: str
    fetched_at: float = field(default_factory=time.time)


class CrossAssetRegime:
    """
    Computes a cross-asset regime signal from BTC dominance and ETH/BTC ratio.

    Parameters
    ----------
    global_url      : CoinGecko /global endpoint
    price_url       : CoinGecko /simple/price endpoint
    timeout_sec     : HTTP timeout
    stale_threshold : seconds before cached snapshot is considered stale
    btc_dom_high    : dominance level above which BTC season is declared
    btc_dom_low     : dominance level below which alt season is declared
    momentum_window : bars used to compute ETH/BTC momentum
    """

    def __init__(
        self,
        global_url: str = COINGECKO_GLOBAL_URL,
        price_url: str = COINGECKO_PRICE_URL,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        stale_threshold: int = STALE_THRESHOLD_SEC,
        btc_dom_high: float = BTC_DOM_HIGH,
        btc_dom_low: float = BTC_DOM_LOW,
        momentum_window: int = MOMENTUM_WINDOW,
    ) -> None:
        self._global_url = global_url
        self._price_url = price_url
        self._timeout_sec = timeout_sec
        self._stale_threshold = stale_threshold
        self._btc_dom_high = btc_dom_high
        self._btc_dom_low = btc_dom_low
        self._momentum_window = momentum_window

        self._latest: Optional[CrossAssetSnapshot] = None
        self._eth_btc_history: Deque[float] = deque(maxlen=HISTORY_LEN)
        self._dom_history: Deque[float] = deque(maxlen=HISTORY_LEN)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def regime_signal(self) -> float:
        return self._latest.regime_signal if self._latest else 0.0

    @property
    def regime_label(self) -> str:
        return self._latest.regime_label if self._latest else RegimeLabel.NEUTRAL

    @property
    def btc_dominance(self) -> Optional[float]:
        return self._latest.btc_dominance if self._latest else None

    @property
    def eth_btc_ratio(self) -> Optional[float]:
        return self._latest.eth_btc_ratio if self._latest else None

    def is_stale(self) -> bool:
        if self._latest is None:
            return True
        return (time.time() - self._latest.fetched_at) > self._stale_threshold

    # ------------------------------------------------------------------
    # Regime scalar (Push 31)
    # ------------------------------------------------------------------

    def get_scalar(self, candles: Optional[np.ndarray] = None) -> float:
        """
        Return a regime-adaptive risk scalar for PositionSizer.

        Mapping
        -------
          BTC_SEASON  (high volatility / risk-off)  -> 0.5
          ALT_SEASON  (trending / risk-on)           -> 1.5
          NEUTRAL                                    -> 1.0

        Parameters
        ----------
        candles : np.ndarray, optional
            Passed for API symmetry with other regime methods; not used
            internally (regime is derived from cross-asset data, not OHLCV).

        Returns
        -------
        float  risk scalar to multiply into base position fraction
        """
        label = self.regime_label
        if label == RegimeLabel.BTC_SEASON:
            scalar = _SCALAR_HIGH_VOL
        elif label == RegimeLabel.ALT_SEASON:
            scalar = _SCALAR_TRENDING
        else:
            scalar = _SCALAR_NEUTRAL
        logger.debug("CrossAssetRegime.get_scalar: label=%s -> scalar=%.2f", label, scalar)
        return scalar

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    async def fetch(self) -> CrossAssetSnapshot:
        """Fetch latest data and update the internal snapshot."""
        btc_dom, eth_btc = await asyncio.gather(
            self._fetch_btc_dominance(),
            self._fetch_eth_btc_ratio(),
            return_exceptions=False,
        )
        self._dom_history.append(btc_dom)
        self._eth_btc_history.append(eth_btc)

        signal = self._compute_signal(btc_dom, eth_btc)
        label  = self._classify(signal)

        self._latest = CrossAssetSnapshot(
            btc_dominance=btc_dom,
            eth_btc_ratio=eth_btc,
            regime_signal=signal,
            regime_label=label,
        )
        logger.info(
            "CrossAsset: BTC.D=%.2f%% ETH/BTC=%.6f signal=%.3f [%s] scalar=%.2f",
            btc_dom, eth_btc, signal, label, self.get_scalar(),
        )
        return self._latest

    async def fetch_if_stale(self) -> CrossAssetSnapshot:
        if self.is_stale():
            return await self.fetch()
        return self._latest  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_btc_dominance(self) -> float:
        """Return BTC dominance as a percentage (e.g. 52.4)."""
        if not _AIOHTTP_AVAILABLE:
            return 50.0
        timeout = aiohttp.ClientTimeout(total=self._timeout_sec)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self._global_url) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
            dom = data["data"]["market_cap_percentage"]["btc"]
            return float(dom)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BTC dominance fetch failed: %s", exc)
            return float(self._dom_history[-1]) if self._dom_history else 50.0

    async def _fetch_eth_btc_ratio(self) -> float:
        """Return ETH/BTC price ratio."""
        if not _AIOHTTP_AVAILABLE:
            return 0.05
        timeout = aiohttp.ClientTimeout(total=self._timeout_sec)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self._price_url) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
            eth_usd = float(data["ethereum"]["usd"])
            btc_usd = float(data["bitcoin"]["usd"])
            return eth_usd / btc_usd if btc_usd > 0 else 0.05
        except Exception as exc:  # noqa: BLE001
            logger.warning("ETH/BTC ratio fetch failed: %s", exc)
            return float(self._eth_btc_history[-1]) if self._eth_btc_history else 0.05

    def _compute_signal(self, btc_dom: float, eth_btc: float) -> float:
        """
        Combine dominance level and ETH/BTC momentum into a single signal.

        BTC-season (positive) components:
          - BTC dominance above btc_dom_high
          - ETH/BTC ratio declining (negative momentum)

        Alt-season (negative) components:
          - BTC dominance below btc_dom_low
          - ETH/BTC ratio rising (positive momentum)
        """
        # --- Dominance signal ---
        if btc_dom >= self._btc_dom_high:
            dom_signal = min(1.0, (btc_dom - self._btc_dom_high) / 10.0)
        elif btc_dom <= self._btc_dom_low:
            dom_signal = max(-1.0, -(self._btc_dom_low - btc_dom) / 10.0)
        else:
            mid = (self._btc_dom_high + self._btc_dom_low) / 2.0
            dom_signal = (btc_dom - mid) / ((self._btc_dom_high - self._btc_dom_low) / 2.0)

        # --- ETH/BTC momentum signal ---
        hist = list(self._eth_btc_history)
        if len(hist) >= self._momentum_window:
            window = hist[-self._momentum_window:]
            old = window[0]
            new = window[-1]
            if old > 0:
                pct_change = (new - old) / old
                mom_signal = -math.tanh(pct_change / 0.1)
            else:
                mom_signal = 0.0
        else:
            mom_signal = 0.0

        combined = (dom_signal + mom_signal) / 2.0
        return max(-1.0, min(1.0, combined))

    @staticmethod
    def _classify(signal: float) -> str:
        if signal > 0.2:
            return RegimeLabel.BTC_SEASON
        if signal < -0.2:
            return RegimeLabel.ALT_SEASON
        return RegimeLabel.NEUTRAL
