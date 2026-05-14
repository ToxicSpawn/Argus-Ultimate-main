"""
MarketDataService (canonical unified market data).

The Unified System originally used mocked prices inside `unified_ai_brain.py`.
This service provides a real, best-effort market data surface for:
- ticker (last/bid/ask/volume)
- order book (L2)
- OHLCV (for indicators)

Design goals:
- Work in both paper and live modes (public endpoints preferred).
- Prefer Kraken (CCXT-backed) for OHLCV/order book, with fallbacks for ticker.
- Keep the interface small and import-safe.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ticker:
    symbol: str
    last: float
    bid: Optional[float]
    ask: Optional[float]
    volume: Optional[float]
    timestamp: float
    source: str
    raw: Dict[str, Any]


class MarketDataService:
    def __init__(
        self,
        exchanges: Dict[str, Any],
        *,
        primary: str = "kraken",
        secondary: str = "coinbase_advanced",
        ticker_ttl_s: float = 2.0,
        ohlcv_ttl_s: float = 120.0,
        ohlcv_poll_interval_s: float = 30.0,
        ohlcv_retry_attempts: int = 2,
        order_book_ttl_s: float = 1.0,
        request_timeout_s: float = 20.0,
        persist_tick_store: bool = False,
    ) -> None:
        self.exchanges = exchanges
        self.primary = str(primary)
        self.secondary = str(secondary)

        self._ticker_ttl_s = float(ticker_ttl_s)
        self._persist_tick_store = bool(persist_tick_store)
        self._ohlcv_ttl_s = float(ohlcv_ttl_s)
        self._ohlcv_poll_interval_s = float(max(0.0, ohlcv_poll_interval_s))
        self._ohlcv_retry_attempts = int(max(0, ohlcv_retry_attempts))
        self._order_book_ttl_s = float(order_book_ttl_s)
        self._request_timeout_s = float(request_timeout_s)

        self._ticker_cache: Dict[str, Tuple[float, Ticker]] = {}
        self._ohlcv_cache: Dict[Tuple[str, str, int], Tuple[float, pd.DataFrame]] = {}
        self._order_book_cache: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}
        # Rate limiting: track last OHLCV request timestamp to avoid Kraken rate limits
        self._last_ohlcv_request_ts: float = 0.0
        self._ohlcv_min_interval_s: float = 1.0  # ~1 request/sec max (Kraken-friendly)
        self._recent_trades_ttl_s = 0.5  # Short TTL for HFT tick momentum
        self._recent_trades_cache: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]]]] = {}

        # TickEngine reference for real-time candle bypass (set via set_tick_engine)
        self._tick_engine: Any = None
        # Semaphore for parallel OHLCV fetches (limit concurrent requests to stay under rate limits)
        self._ohlcv_semaphore = asyncio.Semaphore(2)

        try:
            from utils.circuit_breaker import CircuitBreaker
            # Increased thresholds for better resilience during API rate limits
            self._cb_ticker = CircuitBreaker(failure_threshold=50, cooldown_s=30.0, name="market_data_ticker")
            self._cb_ohlcv = CircuitBreaker(failure_threshold=50, cooldown_s=30.0, name="market_data_ohlcv")
            self._cb_order_book = CircuitBreaker(failure_threshold=50, cooldown_s=30.0, name="market_data_order_book")
            self._cb_trades = CircuitBreaker(failure_threshold=50, cooldown_s=30.0, name="market_data_trades")
        except Exception:
            self._cb_ticker = self._cb_ohlcv = self._cb_order_book = self._cb_trades = None

    def set_tick_engine(self, tick_engine: Any) -> None:
        """Attach a TickEngine for real-time candle bypass (avoids REST OHLCV fetches)."""
        self._tick_engine = tick_engine
        logger.info("MarketDataService: TickEngine attached for real-time candle bypass")

    def _now(self) -> float:
        return float(time.time())

    def _get_exchange(self, name: str) -> Optional[Any]:
        return self.exchanges.get(str(name))

    @staticmethod
    def _normalize_timeframe(timeframe: str) -> str:
        tf = str(timeframe or "1m").strip().lower()
        tf_map = {
            "60m": "1h",
            "120m": "2h",
            "180m": "3h",
            "240m": "4h",
            "360m": "6h",
            "720m": "12h",
            "1440m": "1d",
        }
        return tf_map.get(tf, tf)

    async def fetch_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        Best-effort ticker fetch.

        Uses primary exchange first; falls back to secondary if needed.
        """
        sym = str(symbol)
        cached = self._ticker_cache.get(sym)
        if cached and (self._now() - cached[0]) <= self._ticker_ttl_s:
            return cached[1]

        cb = getattr(self, "_cb_ticker", None)
        if cb is not None and not cb.allow():
            return None

        for ex_name in (self.primary, self.secondary):
            ex = self._get_exchange(ex_name)
            if ex is None:
                continue
            try:
                raw = await asyncio.wait_for(ex.fetch_ticker(sym), timeout=self._request_timeout_s)
                t = self._normalize_ticker(sym, ex_name, raw)
                self._ticker_cache[sym] = (self._now(), t)
                if self._persist_tick_store:
                    try:
                        from data.paper_data_hooks import push_ticker_to_tick_store
                        push_ticker_to_tick_store(sym, ex_name, t.last, t.volume, t.timestamp)
                    except Exception as _e:
                        logger.debug("market_data_service error: %s", _e)
                if cb is not None:
                    cb.record_success()
                return t
            except Exception as e:
                logger.debug("Ticker fetch failed (%s, %s): %s", ex_name, sym, e)
                if cb is not None:
                    cb.record_failure()
                continue

        return None

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """
        Best-effort L2 order book.

        Prefers CCXT-backed clients (Kraken) since CoinbaseAdvancedClient does not
        expose public order book in this build.
        """
        sym = str(symbol)
        lim = int(limit)
        key = (sym, lim)
        cached = self._order_book_cache.get(key)
        if cached and (self._now() - cached[0]) <= self._order_book_ttl_s:
            return cached[1]

        cb = getattr(self, "_cb_order_book", None)
        if cb is not None and not cb.allow():
            return None

        ex = self._get_exchange(self.primary)
        if ex is None:
            return None

        try:
            if callable(getattr(ex, "fetch_order_book", None)):
                ob = await asyncio.wait_for(ex.fetch_order_book(sym, limit=lim), timeout=self._request_timeout_s)
            else:
                ccxt_ex = getattr(ex, "_exchange", None)
                if ccxt_ex is None or not callable(getattr(ccxt_ex, "fetch_order_book", None)):
                    return None
                ob = await asyncio.wait_for(ccxt_ex.fetch_order_book(sym, limit=lim), timeout=self._request_timeout_s)
            if not isinstance(ob, dict):
                return None
            self._order_book_cache[key] = (self._now(), ob)
            if self._persist_tick_store:
                try:
                    from data.paper_data_hooks import push_order_book_to_tick_store
                    bids = ob.get("bids") or []
                    asks = ob.get("asks") or []
                    push_order_book_to_tick_store(sym, self.primary, bids, asks, self._now())
                except Exception as _e:
                    logger.debug("market_data_service error: %s", _e)
            if cb is not None:
                cb.record_success()
            return ob
        except Exception as e:
            logger.debug("Order book fetch failed (%s): %s", sym, e)
            if cb is not None:
                cb.record_failure()
            return None

    async def fetch_recent_trades(self, symbol: str, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """
        Best-effort recent trades for HFT tick momentum (CCXT fetch_trades).

        Returns list of dicts with at least: side, price, amount, timestamp (ms).
        Short TTL so HFT sees fresh flow.
        """
        sym = str(symbol)
        lim = int(limit)
        key = (sym, lim)
        cached = self._recent_trades_cache.get(key)
        if cached and (self._now() - cached[0]) <= self._recent_trades_ttl_s:
            return cached[1]

        cb = getattr(self, "_cb_trades", None)
        if cb is not None and not cb.allow():
            return None

        ex = self._get_exchange(self.primary)
        if ex is None:
            return None

        try:
            fetcher = getattr(ex, "fetch_trades", None)
            if not callable(fetcher):
                ccxt_ex = getattr(ex, "_exchange", None)
                fetcher = getattr(ccxt_ex, "fetch_trades", None) if ccxt_ex else None
            if not callable(fetcher):
                return None
            raw = await asyncio.wait_for(fetcher(sym, limit=lim), timeout=min(self._request_timeout_s, 5.0))
            if not isinstance(raw, list):
                return None
            # Normalize: ensure side, price, amount, timestamp
            out = []
            for t in raw[:lim]:
                if not isinstance(t, dict):
                    continue
                out.append({
                    "side": t.get("side", "buy"),
                    "price": float(t.get("price") or t.get("rate") or 0),
                    "amount": float(t.get("amount") or t.get("cost") or 0),
                    "timestamp": t.get("timestamp"),
                })
            self._recent_trades_cache[key] = (self._now(), out)
            if cb is not None:
                cb.record_success()
            return out
        except Exception as e:
            logger.debug("Recent trades fetch failed (%s): %s", sym, e)
            if cb is not None:
                cb.record_failure()
            return None

    async def fetch_ohlcv_df(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 200,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV and return as a pandas DataFrame with columns:
        ['timestamp','open','high','low','close','volume'].

        Checks TickEngine first for real-time candles built from L2 WebSocket
        data.  Falls back to REST API if TickEngine data is insufficient.
        """
        sym = str(symbol)
        tf = self._normalize_timeframe(timeframe)
        lim = int(limit)
        key = (sym, tf, lim)
        cached = self._ohlcv_cache.get(key)
        now_ts = self._now()
        if cached:
            age_s = now_ts - cached[0]
            if age_s <= self._ohlcv_ttl_s:
                return cached[1]
            if age_s <= self._ohlcv_poll_interval_s:
                return cached[1]

        # --- TickEngine bypass: use real-time candles if available ---
        te = self._tick_engine
        if te is not None:
            try:
                te_df = te.get_ohlcv_df(sym, tf, limit=lim)
                if te_df is not None and len(te_df) >= 2:
                    self._ohlcv_cache[key] = (self._now(), te_df)
                    logger.debug("OHLCV from TickEngine: %s %s (%d bars)", sym, tf, len(te_df))
                    return te_df
            except Exception as e:
                logger.debug("TickEngine OHLCV bypass failed for %s %s: %s", sym, tf, e)

        cb = getattr(self, "_cb_ohlcv", None)
        if cb is not None and not cb.allow():
            return cached[1] if cached else None

        ex = self._get_exchange(self.primary)
        if ex is None:
            return cached[1] if cached else None

        max_attempts = 1 + int(max(0, self._ohlcv_retry_attempts))
        timeout_s = max(2.0, min(10.0, float(self._request_timeout_s or 10.0)))
        last_exc: Optional[Exception] = None

        # Rate limiting: wait if we recently made an OHLCV request to avoid exchange rate limits
        now_rl = self._now()
        elapsed = now_rl - self._last_ohlcv_request_ts
        if elapsed < self._ohlcv_min_interval_s:
            await asyncio.sleep(self._ohlcv_min_interval_s - elapsed)
        self._last_ohlcv_request_ts = self._now()

        for attempt in range(1, max_attempts + 1):
            try:
                if callable(getattr(ex, "fetch_ohlcv", None)):
                    ohlcv = await asyncio.wait_for(
                        ex.fetch_ohlcv(sym, timeframe=tf, limit=lim),
                        timeout=timeout_s,
                    )
                else:
                    ccxt_ex = getattr(ex, "_exchange", None)
                    if ccxt_ex is None or not callable(getattr(ccxt_ex, "fetch_ohlcv", None)):
                        return cached[1] if cached else None
                    ohlcv = await asyncio.wait_for(
                        ccxt_ex.fetch_ohlcv(sym, timeframe=tf, limit=lim),
                        timeout=timeout_s,
                    )

                if not isinstance(ohlcv, list) or not ohlcv:
                    logger.warning(f"MarketDataService: empty OHLCV for {symbol} {timeframe}")
                    raise ValueError("empty_ohlcv")

                logger.info(f"MarketDataService: fetched {len(ohlcv)} candles for {symbol} {timeframe}")
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                # CCXT timestamps are ms
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)
                self._ohlcv_cache[key] = (self._now(), df)
                if cb is not None:
                    cb.record_success()
                return df
            except Exception as e:
                last_exc = e
                log_fn = logger.debug if attempt < max_attempts else logger.warning
                log_fn(
                    "OHLCV fetch failed (exchange=%s symbol=%s timeframe=%s attempt=%s/%s): %s",
                    str(self.primary),
                    sym,
                    tf,
                    attempt,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(min(1.0, 0.25 * attempt))

        if cb is not None:
            cb.record_failure()
        if last_exc is not None:
            logger.warning(
                "OHLCV fetch exhausted retries (exchange=%s symbol=%s timeframe=%s): %s",
                str(self.primary),
                sym,
                tf,
                last_exc,
            )
        return cached[1] if cached else None

    async def fetch_ohlcv_multi(
        self,
        symbols: List[str],
        timeframe: str = "1m",
        limit: int = 200,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch OHLCV for multiple symbols in parallel with a semaphore.

        Uses ``asyncio.gather`` with a concurrency limit of 2 to stay under
        exchange rate limits while eliminating sequential 0.5s delays.

        Returns a dict mapping symbol -> DataFrame (or None on failure).
        """
        async def _fetch_one(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
            async with self._ohlcv_semaphore:
                df = await self.fetch_ohlcv_df(sym, timeframe=timeframe, limit=limit)
                return (sym, df)

        results = await asyncio.gather(
            *[_fetch_one(s) for s in symbols],
            return_exceptions=True,
        )
        out: Dict[str, Optional[pd.DataFrame]] = {}
        for r in results:
            if isinstance(r, Exception):
                logger.debug("fetch_ohlcv_multi exception: %s", r)
                continue
            if isinstance(r, tuple) and len(r) == 2:
                out[r[0]] = r[1]
        return out

    def _normalize_ticker(self, symbol: str, source: str, raw: Dict[str, Any]) -> Ticker:
        def _f(v: Any) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        # CCXT uses last/close; our Coinbase client uses last/bid/ask too
        last = _f(raw.get("last") or raw.get("close") or raw.get("price"))
        if last is None:
            last = 0.0

        bid = _f(raw.get("bid") or raw.get("best_bid"))
        ask = _f(raw.get("ask") or raw.get("best_ask"))
        volume = _f(raw.get("volume") or raw.get("baseVolume") or raw.get("quoteVolume"))
        ts = raw.get("timestamp")
        try:
            # CCXT timestamp is ms; normalize to seconds
            timestamp_s = float(ts) / 1000.0 if ts is not None and float(ts) > 1e12 else float(ts or self._now())
        except Exception:
            timestamp_s = self._now()

        return Ticker(
            symbol=symbol,
            last=float(last),
            bid=bid,
            ask=ask,
            volume=volume,
            timestamp=float(timestamp_s),
            source=source,
            raw=raw if isinstance(raw, dict) else {"raw": str(raw)},
        )

    def get_order_book_mid_spread_from_cache(self, symbol: str, limit: int = 20) -> Optional[Tuple[float, float]]:
        """Local order book cache: return (mid, spread) from cached L2 if valid; avoids extra request."""
        key = (str(symbol), int(limit))
        cached = self._order_book_cache.get(key)
        if not cached or (self._now() - cached[0]) > self._order_book_ttl_s:
            return None
        ob = cached[1]
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return None
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        return (mid, spread)

    @staticmethod
    def is_ticker_stale(ticker: Ticker, max_age_s: float = 5.0) -> bool:
        """Stale data check: reject or flag data older than N ms."""
        return (time.time() - ticker.timestamp) > max_age_s

    @staticmethod
    def is_ticker_outlier(last: float, recent_mean: float, recent_std: float, n_std: float = 4.0) -> bool:
        """Outlier detection: reject or dampen ticks that are X std devs from recent mean."""
        if recent_std <= 0:
            return False
        return abs(last - recent_mean) > n_std * recent_std
