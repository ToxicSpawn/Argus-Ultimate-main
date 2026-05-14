"""FundingRateScanner — Push 39.

Polls perpetual funding rates from Binance, Bybit, OKX, and Kraken
(where available) and produces a directional signal for the FUNDING_ARB
SignalGateway source.

Funding rate interpretation
---------------------------
  rate > 0  →  longs paying shorts  →  crowded long / bearish lean
  rate < 0  →  shorts paying longs  →  crowded short / bullish lean

Stability
---------
  Consistency of sign across last N samples (default 5).
  stability = count(sign == majority_sign) / N

Usage
-----
  scanner = FundingRateScanner(symbols=["BTCUSDT"], poll_interval=60)
  await scanner.start()
  signal = scanner.latest_signal()  # {"direction": "long", "rate": -0.0003, "stability": 0.8}
  await scanner.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

logger = logging.getLogger(__name__)

_BINANCE_URL  = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
_BYBIT_URL    = "https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
_OKX_URL      = "https://www.okx.com/api/v5/public/funding-rate?instId={inst_id}"

_DEFAULT_SYMBOLS = ["BTCUSDT"]
_DEFAULT_POLL_INTERVAL = 60  # seconds
_STABILITY_WINDOW = 5
_RATE_THRESHOLD   = 0.0001   # rates below this magnitude treated as neutral


@dataclass
class FundingRateSample:
    ts: float
    rate: float
    source: str


class FundingRateScanner:
    """Async funding rate poller. Soft-degrades if aiohttp unavailable.

    Parameters
    ----------
    symbols         : List of symbol strings (Binance format, e.g. 'BTCUSDT')
    poll_interval   : Seconds between polls (default 60)
    stability_window: Number of samples used to compute stability (default 5)
    preferred_source: 'binance' | 'bybit' | 'okx' (default 'binance')
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        poll_interval: int = _DEFAULT_POLL_INTERVAL,
        stability_window: int = _STABILITY_WINDOW,
        preferred_source: str = "binance",
    ) -> None:
        self._symbols         = symbols or list(_DEFAULT_SYMBOLS)
        self._poll_interval   = poll_interval
        self._stability_window = stability_window
        self._preferred_source = preferred_source
        self._samples: Dict[str, Deque[FundingRateSample]] = {
            s: deque(maxlen=stability_window) for s in self._symbols
        }
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._session: Optional[object] = None  # aiohttp.ClientSession
        self._last_rate: Dict[str, float] = {s: 0.0 for s in self._symbols}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not _HAS_AIOHTTP:
            logger.warning("FundingRateScanner: aiohttp not available — using mock rates")
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("FundingRateScanner started | symbols=%s poll_interval=%ds",
                    self._symbols, self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
        logger.info("FundingRateScanner stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def latest_signal(
        self, symbol: Optional[str] = None
    ) -> Dict[str, object]:
        """Return latest FUNDING_ARB signal dict.

        Returns
        -------
        {"direction": str, "rate": float, "stability": float}
        direction is 'long' (rate < 0), 'short' (rate > 0), or 'flat'.
        """
        sym = symbol or self._symbols[0]
        samples = list(self._samples.get(sym, deque()))
        if not samples:
            return {"direction": "flat", "rate": 0.0, "stability": 0.0}

        latest_rate = samples[-1].rate
        stability   = self._compute_stability(samples)
        direction   = self._rate_to_direction(latest_rate)

        return {
            "direction":  direction,
            "rate":       latest_rate,
            "stability":  stability,
        }

    def get_all_rates(self) -> Dict[str, float]:
        """Return latest raw rate per symbol."""
        return dict(self._last_rate)

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        if _HAS_AIOHTTP:
            import aiohttp as _aiohttp
            self._session = _aiohttp.ClientSession(
                timeout=_aiohttp.ClientTimeout(total=10)
            )
        try:
            while self._running:
                for symbol in self._symbols:
                    await self._fetch_and_store(symbol)
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("FundingRateScanner poll loop error: %s", exc, exc_info=True)
        finally:
            if self._session:
                try:
                    await self._session.close()
                except Exception:
                    pass

    async def _fetch_and_store(self, symbol: str) -> None:
        rate = await self._fetch_rate(symbol)
        if rate is None:
            return
        sample = FundingRateSample(ts=time.time(), rate=rate, source=self._preferred_source)
        self._samples[symbol].append(sample)
        self._last_rate[symbol] = rate
        logger.debug("FundingRate [%s] %s = %.6f", self._preferred_source, symbol, rate)

    async def _fetch_rate(self, symbol: str) -> Optional[float]:
        if not _HAS_AIOHTTP or self._session is None:
            return self._mock_rate(symbol)
        try:
            if self._preferred_source == "binance":
                return await self._fetch_binance(symbol)
            elif self._preferred_source == "bybit":
                return await self._fetch_bybit(symbol)
            elif self._preferred_source == "okx":
                return await self._fetch_okx(symbol)
            else:
                return await self._fetch_binance(symbol)
        except Exception as exc:
            logger.debug("FundingRate fetch failed (%s): %s", symbol, exc)
            return None

    async def _fetch_binance(self, symbol: str) -> Optional[float]:
        url = _BINANCE_URL.format(symbol=symbol)
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return float(data.get("lastFundingRate", 0.0))

    async def _fetch_bybit(self, symbol: str) -> Optional[float]:
        url = _BYBIT_URL.format(symbol=symbol)
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            items = data.get("result", {}).get("list", [])
            if not items:
                return None
            return float(items[0].get("fundingRate", 0.0))

    async def _fetch_okx(self, symbol: str) -> Optional[float]:
        # OKX uses instId format e.g. BTC-USDT-SWAP
        inst_id = symbol.replace("USDT", "-USDT-SWAP").replace("BTC", "BTC")
        url = _OKX_URL.format(inst_id=inst_id)
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            items = data.get("data", [])
            if not items:
                return None
            return float(items[0].get("fundingRate", 0.0))

    @staticmethod
    def _mock_rate(symbol: str) -> float:
        """Return a synthetic funding rate (used when aiohttp unavailable)."""
        import math
        t = time.time()
        # Oscillates between -0.001 and +0.001 on an 8h cycle
        return 0.001 * math.sin(2 * math.pi * t / (8 * 3600))

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _compute_stability(self, samples: List[FundingRateSample]) -> float:
        if not samples:
            return 0.0
        signs = [1 if s.rate > _RATE_THRESHOLD else (-1 if s.rate < -_RATE_THRESHOLD else 0)
                 for s in samples]
        non_zero = [s for s in signs if s != 0]
        if not non_zero:
            return 0.0
        majority = max(set(non_zero), key=non_zero.count)
        return float(sum(1 for s in non_zero if s == majority) / len(non_zero))

    @staticmethod
    def _rate_to_direction(rate: float) -> str:
        if rate > _RATE_THRESHOLD:
            return "short"   # longs paying = crowded long = fade
        elif rate < -_RATE_THRESHOLD:
            return "long"    # shorts paying = crowded short = fade
        return "flat"
