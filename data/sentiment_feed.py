"""
SentimentFeed — external sentiment data ingestion.

Sources
-------
1. Fear & Greed Index  — alternative.me public API (no key required)
   https://api.alternative.me/fng/

2. BTC on-chain net flow — CryptoQuant-style endpoint or Glassnode
   (configurable base URL; falls back to a mock value if unavailable)

Usage
-----
    feed = SentimentFeed()
    asyncio.run(feed.fetch_all())
    print(feed.fear_greed_value)   # 0-100
    print(feed.fear_greed_label)   # e.g. "Extreme Fear"
    print(feed.btc_net_flow)       # USD net flow (positive = inflow)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"
BTC_NETFLOW_URL = "https://api.cryptoquant.com/v1/btc/exchange-flows/netflow"  # placeholder
DEFAULT_TIMEOUT_SEC = 10
STALE_THRESHOLD_SEC = 300   # re-fetch if data is older than 5 min


@dataclass
class FearGreedSnapshot:
    value: int                  # 0 (extreme fear) … 100 (extreme greed)
    label: str                  # human-readable classification
    fetched_at: float = field(default_factory=time.time)

    @property
    def normalised(self) -> float:
        """Return value scaled to [0.0, 1.0]."""
        return self.value / 100.0

    @property
    def signal(self) -> float:
        """
        Map Fear & Greed to a directional signal in [-1, 1].
        Extreme fear  (<20)  -> contrarian BUY  (+1)
        Extreme greed (>80)  -> contrarian SELL (-1)
        Middle range         -> neutral (0)
        """
        if self.value <= 20:
            return 1.0
        if self.value >= 80:
            return -1.0
        # linear interpolation between 20 and 80
        return 1.0 - ((self.value - 20) / 60.0) * 2.0


@dataclass
class NetFlowSnapshot:
    net_flow_usd: float         # positive = net inflow to exchanges (bearish)
    fetched_at: float = field(default_factory=time.time)

    @property
    def signal(self) -> float:
        """
        Large inflow  -> selling pressure -> bearish (-1)
        Large outflow -> accumulation     -> bullish (+1)
        Clamped to [-1, 1] via tanh-like scaling.
        """
        import math
        # Scale: every $100M inflow maps roughly to -0.5 signal
        scaled = self.net_flow_usd / 2e8
        return max(-1.0, min(1.0, -math.tanh(scaled)))


class SentimentFeed:
    """
    Async sentiment data feed.

    Parameters
    ----------
    netflow_url      : override for BTC net-flow endpoint
    netflow_api_key  : bearer token / API key for net-flow provider
    timeout_sec      : HTTP request timeout
    stale_threshold  : seconds before cached data is considered stale
    """

    def __init__(
        self,
        netflow_url: str = BTC_NETFLOW_URL,
        netflow_api_key: Optional[str] = None,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        stale_threshold: int = STALE_THRESHOLD_SEC,
    ) -> None:
        self._netflow_url = netflow_url
        self._netflow_api_key = netflow_api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec) if _AIOHTTP_AVAILABLE else None
        self._stale_threshold = stale_threshold

        self._fg: Optional[FearGreedSnapshot] = None
        self._nf: Optional[NetFlowSnapshot] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fear_greed_value(self) -> Optional[int]:
        return self._fg.value if self._fg else None

    @property
    def fear_greed_label(self) -> Optional[str]:
        return self._fg.label if self._fg else None

    @property
    def fear_greed_signal(self) -> float:
        return self._fg.signal if self._fg else 0.0

    @property
    def btc_net_flow(self) -> Optional[float]:
        return self._nf.net_flow_usd if self._nf else None

    @property
    def net_flow_signal(self) -> float:
        return self._nf.signal if self._nf else 0.0

    @property
    def combined_signal(self) -> float:
        """Equal-weight average of both signals."""
        return (self.fear_greed_signal + self.net_flow_signal) / 2.0

    def is_stale(self) -> bool:
        now = time.time()
        fg_stale = self._fg is None or (now - self._fg.fetched_at) > self._stale_threshold
        nf_stale = self._nf is None or (now - self._nf.fetched_at) > self._stale_threshold
        return fg_stale or nf_stale

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    async def fetch_all(self) -> None:
        """Concurrently fetch both data sources."""
        await asyncio.gather(
            self._fetch_fear_greed(),
            self._fetch_btc_netflow(),
            return_exceptions=True,
        )

    async def fetch_if_stale(self) -> None:
        if self.is_stale():
            await self.fetch_all()

    async def _fetch_fear_greed(self) -> None:
        if not _AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not installed — skipping Fear & Greed fetch")
            return
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(FEAR_GREED_URL) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
            entry = data["data"][0]
            self._fg = FearGreedSnapshot(
                value=int(entry["value"]),
                label=entry["value_classification"],
            )
            logger.info("Fear & Greed: %d (%s)", self._fg.value, self._fg.label)
        except Exception as exc:  # noqa: BLE001
            logger.error("Fear & Greed fetch failed: %s", exc)

    async def _fetch_btc_netflow(self) -> None:
        if not _AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not installed — skipping BTC net-flow fetch")
            return
        headers: dict = {"Accept": "application/json"}
        if self._netflow_api_key:
            headers["Authorization"] = f"Bearer {self._netflow_api_key}"
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(self._netflow_url, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
            # Normalise across common provider response shapes
            net_flow = self._parse_netflow(data)
            self._nf = NetFlowSnapshot(net_flow_usd=net_flow)
            logger.info("BTC net flow: $%.2fM", net_flow / 1e6)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BTC net-flow fetch failed (%s) — using 0.0", exc)
            self._nf = NetFlowSnapshot(net_flow_usd=0.0)

    @staticmethod
    def _parse_netflow(data: dict) -> float:
        """
        Extract net flow USD from provider response.
        Supports CryptoQuant and Glassnode response shapes.
        Falls back to 0.0 if structure is unrecognised.
        """
        # CryptoQuant: {"result": {"data": [{"netflow_total": ...}]}}
        try:
            return float(data["result"]["data"][0]["netflow_total"])
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        # Glassnode: [{"t": ..., "v": ...}]
        try:
            return float(data[0]["v"])
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        # Direct scalar
        try:
            return float(data)
        except (TypeError, ValueError):
            pass
        return 0.0
