"""
Connection Pool — persistent HTTPS sessions with keep-alive for each exchange.

Pre-warms TCP connections on startup (not first trade), monitors health,
auto-reconnects, and caches DNS resolutions.  Reduces per-request latency
by reusing established connections instead of opening new ones.

Usage:
    pool = ConnectionPool()
    await pool.warm_up()                  # pre-connect all exchanges
    session = pool.get_session("kraken")  # returns aiohttp.ClientSession
    await pool.close_all()                # graceful shutdown
"""
from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default exchange REST base URLs
# ---------------------------------------------------------------------------
_EXCHANGE_URLS: Dict[str, str] = {
    "kraken": "https://api.kraken.com",
    "coinbase": "https://api.coinbase.com",
    "bybit": "https://api.bybit.com",
    "okx": "https://www.okx.com",
    "deribit": "https://www.deribit.com",
}


@dataclass
class _PoolEntry:
    """Internal bookkeeping for one exchange session."""
    exchange: str
    base_url: str
    session: Any = None           # aiohttp.ClientSession once created
    connector: Any = None         # aiohttp.TCPConnector
    last_health_check: float = 0.0
    healthy: bool = True
    latency_ms: float = 0.0
    dns_cache: Dict[str, str] = field(default_factory=dict)


class ConnectionPool:
    """
    Persistent HTTPS connection pool for exchange REST APIs.

    Each exchange gets its own ``aiohttp.ClientSession`` backed by a
    ``TCPConnector`` with keep-alive, configurable limits, and DNS caching.
    """

    def __init__(
        self,
        exchanges: Optional[Dict[str, str]] = None,
        limit_per_host: int = 10,
        keepalive_timeout: int = 30,
        total_limit: int = 50,
        health_check_interval_s: float = 30.0,
        connect_timeout: float = 5.0,
    ) -> None:
        self._exchange_urls: Dict[str, str] = dict(exchanges) if exchanges else dict(_EXCHANGE_URLS)
        self._limit_per_host = limit_per_host
        self._keepalive_timeout = keepalive_timeout
        self._total_limit = total_limit
        self._health_check_interval_s = health_check_interval_s
        self._connect_timeout = connect_timeout
        self._pools: Dict[str, _PoolEntry] = {}
        self._dns_cache: Dict[str, str] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # DNS caching
    # ------------------------------------------------------------------

    def resolve_dns(self, hostname: str) -> Optional[str]:
        """
        Resolve hostname once and cache the IP address.

        Returns cached IP or None if resolution fails.
        """
        if hostname in self._dns_cache:
            return self._dns_cache[hostname]
        try:
            ip = socket.gethostbyname(hostname)
            self._dns_cache[hostname] = ip
            logger.debug("DNS cached: %s -> %s", hostname, ip)
            return ip
        except socket.gaierror as exc:
            logger.warning("DNS resolution failed for %s: %s", hostname, exc)
            return None

    def get_dns_cache(self) -> Dict[str, str]:
        """Return a copy of the current DNS cache."""
        return dict(self._dns_cache)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _create_session(self, exchange: str) -> Any:
        """Create a new aiohttp session with keep-alive and connection pooling."""
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed; ConnectionPool returns stub sessions")
            return None

        base_url = self._exchange_urls.get(exchange, "")

        # Pre-resolve DNS for the exchange
        if base_url:
            from urllib.parse import urlparse
            hostname = urlparse(base_url).hostname
            if hostname:
                self.resolve_dns(hostname)

        connector = aiohttp.TCPConnector(
            limit=self._total_limit,
            limit_per_host=self._limit_per_host,
            keepalive_timeout=self._keepalive_timeout,
            enable_cleanup_closed=True,
            force_close=False,
            ttl_dns_cache=300,  # 5 min DNS TTL in aiohttp layer
        )

        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=self._connect_timeout,
            sock_connect=self._connect_timeout,
        )

        session = aiohttp.ClientSession(
            base_url=base_url if base_url else None,
            connector=connector,
            timeout=timeout,
            headers={
                "Connection": "keep-alive",
                "Accept": "application/json",
            },
        )

        entry = _PoolEntry(
            exchange=exchange,
            base_url=base_url,
            session=session,
            connector=connector,
        )
        self._pools[exchange] = entry
        logger.info("ConnectionPool: session created for %s (%s)", exchange, base_url)
        return session

    def get_session(self, exchange: str) -> Any:
        """
        Get or create a persistent aiohttp.ClientSession for *exchange*.

        Returns None if aiohttp is not installed.
        """
        if self._closed:
            raise RuntimeError("ConnectionPool is closed")

        entry = self._pools.get(exchange)
        if entry is not None and entry.session is not None:
            return entry.session

        return self._create_session(exchange)

    # ------------------------------------------------------------------
    # Warm-up: pre-connect all exchanges on startup
    # ------------------------------------------------------------------

    async def warm_up(self) -> Dict[str, bool]:
        """
        Pre-warm TCP connections for every configured exchange.

        Performs a lightweight HEAD/GET to establish the TCP + TLS handshake
        before the first real trade.  Returns ``{exchange: success_bool}``.
        """
        results: Dict[str, bool] = {}
        for exchange in self._exchange_urls:
            session = self.get_session(exchange)
            if session is None:
                results[exchange] = False
                continue
            try:
                t0 = time.perf_counter()
                async with session.get("/") as resp:
                    _ = resp.status  # consume response
                latency_ms = (time.perf_counter() - t0) * 1000.0
                entry = self._pools.get(exchange)
                if entry:
                    entry.latency_ms = latency_ms
                    entry.healthy = True
                    entry.last_health_check = time.monotonic()
                results[exchange] = True
                logger.info(
                    "ConnectionPool: warm-up %s OK (%.1fms)",
                    exchange, latency_ms,
                )
            except Exception as exc:
                logger.warning("ConnectionPool: warm-up %s failed: %s", exchange, exc)
                results[exchange] = False
                entry = self._pools.get(exchange)
                if entry:
                    entry.healthy = False
        return results

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    async def health_check(self, exchange: str) -> Dict[str, Any]:
        """
        Run a health check on a single exchange connection.

        Returns dict with healthy, latency_ms, error.
        """
        session = self.get_session(exchange)
        if session is None:
            return {"exchange": exchange, "healthy": False, "error": "no_session"}
        try:
            t0 = time.perf_counter()
            async with session.get("/") as resp:
                _ = resp.status
            latency_ms = (time.perf_counter() - t0) * 1000.0
            entry = self._pools.get(exchange)
            if entry:
                entry.latency_ms = latency_ms
                entry.healthy = True
                entry.last_health_check = time.monotonic()
            return {"exchange": exchange, "healthy": True, "latency_ms": latency_ms}
        except Exception as exc:
            entry = self._pools.get(exchange)
            if entry:
                entry.healthy = False
            return {"exchange": exchange, "healthy": False, "error": str(exc)}

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Run health checks on all configured exchanges."""
        tasks = {ex: self.health_check(ex) for ex in self._exchange_urls}
        results = {}
        for exchange, coro in tasks.items():
            results[exchange] = await coro
        return results

    async def _auto_reconnect(self, exchange: str) -> bool:
        """Tear down and recreate a session for *exchange*."""
        entry = self._pools.pop(exchange, None)
        if entry and entry.session:
            try:
                await entry.session.close()
            except Exception as _e:
                logger.debug("connection_pool error: %s", _e)
        session = self._create_session(exchange)
        return session is not None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return per-exchange pool statistics."""
        stats: Dict[str, Dict[str, Any]] = {}
        for exchange, entry in self._pools.items():
            stats[exchange] = {
                "healthy": entry.healthy,
                "latency_ms": entry.latency_ms,
                "base_url": entry.base_url,
                "dns_cached": bool(self._dns_cache.get(exchange)),
            }
        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close_all(self) -> None:
        """Close all sessions and connectors."""
        self._closed = True
        for exchange, entry in list(self._pools.items()):
            if entry.session is not None:
                try:
                    await entry.session.close()
                except Exception as exc:
                    logger.debug("ConnectionPool: error closing %s: %s", exchange, exc)
        self._pools.clear()
        logger.info("ConnectionPool: all sessions closed")
