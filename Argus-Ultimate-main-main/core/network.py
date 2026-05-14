"""
Argus Trading System - Network Optimizations
============================================

Low-latency network utilities and socket optimizations.

Features:
- TCP socket tuning for trading
- Optimized aiohttp connector
- Connection pooling
- Latency measurement
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Network configuration for low-latency trading."""
    # TCP settings
    tcp_nodelay: bool = True          # Disable Nagle's algorithm
    tcp_quickack: bool = True         # Immediate ACKs (Linux only)
    tcp_keepalive: bool = True        # Keep connections alive
    keepalive_time: int = 10          # Seconds before keepalive probes
    keepalive_interval: int = 5       # Seconds between probes
    keepalive_probes: int = 3         # Number of probes before drop

    # Connection pool
    pool_size: int = 10               # Connections per host
    pool_ttl: int = 300               # Connection TTL seconds

    # Timeouts
    connect_timeout: float = 5.0
    read_timeout: float = 10.0
    total_timeout: float = 30.0

    # Buffer sizes
    recv_buffer_size: int = 262144    # 256KB
    send_buffer_size: int = 262144    # 256KB


def optimize_socket(sock: socket.socket, config: Optional[NetworkConfig] = None) -> None:
    """
    Apply low-latency optimizations to a socket.

    Args:
        sock: Socket to optimize
        config: Network configuration
    """
    config = config or NetworkConfig()

    try:
        # Disable Nagle's algorithm (send immediately, don't buffer)
        if config.tcp_nodelay:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Set buffer sizes
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, config.recv_buffer_size)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, config.send_buffer_size)

        # Enable keepalive
        if config.tcp_keepalive:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            # Linux-specific keepalive tuning
            if hasattr(socket, 'TCP_KEEPIDLE'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, config.keepalive_time)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, config.keepalive_interval)
            if hasattr(socket, 'TCP_KEEPCNT'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, config.keepalive_probes)

        # Linux: Enable TCP_QUICKACK (disable delayed ACKs)
        if config.tcp_quickack and hasattr(socket, 'TCP_QUICKACK'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)

        logger.debug("Socket optimized: nodelay=%s, buffers=%d", config.tcp_nodelay, config.recv_buffer_size)

    except (OSError, AttributeError) as e:
        logger.warning("Could not apply all socket optimizations: %s", e)


def create_optimized_connector(config: Optional[NetworkConfig] = None) -> aiohttp.TCPConnector:
    """
    Create an optimized aiohttp TCP connector.

    Args:
        config: Network configuration

    Returns:
        Configured TCPConnector for low-latency connections
    """
    config = config or NetworkConfig()

    connector = aiohttp.TCPConnector(
        # Connection pool
        limit=config.pool_size * 10,          # Total connections
        limit_per_host=config.pool_size,      # Per host
        ttl_dns_cache=300,                    # DNS cache TTL
        use_dns_cache=True,

        # TCP settings
        force_close=False,                    # Keep connections alive
        enable_cleanup_closed=True,

        # Timeouts handled at session level
    )

    return connector


def create_optimized_session(config: Optional[NetworkConfig] = None) -> aiohttp.ClientSession:
    """
    Create an optimized aiohttp client session.

    Args:
        config: Network configuration

    Returns:
        Configured ClientSession for trading API calls
    """
    config = config or NetworkConfig()

    connector = create_optimized_connector(config)

    timeout = aiohttp.ClientTimeout(
        total=config.total_timeout,
        connect=config.connect_timeout,
        sock_read=config.read_timeout,
    )

    session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        # Skip auto-decompress for slightly lower latency
        # auto_decompress=False,
    )

    return session


class LatencyTracker:
    """
    Track and report network latency statistics.

    Useful for monitoring exchange connectivity.
    """

    def __init__(self, name: str, window_size: int = 100) -> None:
        self.name = name
        self.window_size = window_size
        self._samples: list[float] = []
        self._last_sample: float = 0.0

    def record(self, latency_ms: float) -> None:
        """Record a latency sample."""
        self._samples.append(latency_ms)
        self._last_sample = latency_ms

        # Keep window size
        if len(self._samples) > self.window_size:
            self._samples.pop(0)

    @property
    def last(self) -> float:
        """Last recorded latency in ms."""
        return self._last_sample

    @property
    def avg(self) -> float:
        """Average latency in ms."""
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def min(self) -> float:
        """Minimum latency in ms."""
        return min(self._samples) if self._samples else 0.0

    @property
    def max(self) -> float:
        """Maximum latency in ms."""
        return max(self._samples) if self._samples else 0.0

    @property
    def p99(self) -> float:
        """99th percentile latency in ms."""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def stats(self) -> Dict[str, float]:
        """Get all statistics."""
        return {
            "last_ms": self.last,
            "avg_ms": self.avg,
            "min_ms": self.min,
            "max_ms": self.max,
            "p99_ms": self.p99,
            "samples": len(self._samples),
        }


class ConnectionManager:
    """
    Manage connections to multiple exchanges with health monitoring.
    """

    def __init__(self, config: Optional[NetworkConfig] = None) -> None:
        self.config = config or NetworkConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._latency_trackers: Dict[str, LatencyTracker] = {}

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create optimized session."""
        if self._session is None or self._session.closed:
            self._session = create_optimized_session(self.config)
        return self._session

    async def close(self) -> None:
        """Close session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def get_latency_tracker(self, name: str) -> LatencyTracker:
        """Get or create latency tracker for an endpoint."""
        if name not in self._latency_trackers:
            self._latency_trackers[name] = LatencyTracker(name)
        return self._latency_trackers[name]

    async def timed_request(
        self,
        method: str,
        url: str,
        tracker_name: Optional[str] = None,
        **kwargs,
    ) -> tuple[aiohttp.ClientResponse, float]:
        """
        Make a timed HTTP request.

        Returns:
            Tuple of (response, latency_ms)
        """
        session = await self.get_session()

        start = time.perf_counter()
        response = await session.request(method, url, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        if tracker_name:
            self.get_latency_tracker(tracker_name).record(latency_ms)

        return response, latency_ms

    def all_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency stats for all tracked endpoints."""
        return {
            name: tracker.stats()
            for name, tracker in self._latency_trackers.items()
        }


async def measure_endpoint_latency(
    url: str,
    samples: int = 5,
    config: Optional[NetworkConfig] = None,
) -> Dict[str, float]:
    """
    Measure latency to an endpoint.

    Args:
        url: URL to test
        samples: Number of samples to take
        config: Network configuration

    Returns:
        Dict with latency statistics
    """
    config = config or NetworkConfig()
    tracker = LatencyTracker("test", window_size=samples)

    async with create_optimized_session(config) as session:
        for _ in range(samples):
            try:
                start = time.perf_counter()
                async with session.get(url) as response:
                    await response.read()
                latency_ms = (time.perf_counter() - start) * 1000
                tracker.record(latency_ms)
            except Exception as e:
                logger.warning("Latency test failed: %s", e)

            await asyncio.sleep(0.1)  # Small delay between samples

    return tracker.stats()


# Pre-configured settings for known exchanges
EXCHANGE_CONFIGS = {
    "kraken": NetworkConfig(
        tcp_nodelay=True,
        tcp_quickack=True,
        pool_size=5,
        connect_timeout=5.0,
        read_timeout=10.0,
    ),
    "coinbase": NetworkConfig(
        tcp_nodelay=True,
        tcp_quickack=True,
        pool_size=5,
        connect_timeout=5.0,
        read_timeout=10.0,
    ),
}


def get_exchange_config(exchange: str) -> NetworkConfig:
    """Get optimized network config for an exchange."""
    return EXCHANGE_CONFIGS.get(exchange.lower(), NetworkConfig())
