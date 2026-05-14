#!/usr/bin/env python3
"""
OpenOnload Socket Factory — creates sockets with optimal EF_* environment
variables and socket options pre-applied for minimum latency.

When OpenOnload kernel module is present, sockets created here are
automatically accelerated via the onload user-space stack. When absent,
standard kernel sockets are returned transparently.

Key optimisations applied:
  - TCP_NODELAY (disable Nagle)
  - SO_BUSY_POLL (spin-poll instead of sleep-wait)
  - IP_TOS DSCP expedited forwarding
  - SO_RCVBUF / SO_SNDBUF maximised
  - TCP_QUICKACK
  - EF_POLL_USEC, EF_SPIN_USEC via os.environ pre-fork
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ EF_ env vars
# These must be set before the onload-accelerated process forks or before
# the first socket() call if running under `onload` wrapper.

_ONLOAD_ENV: dict[str, str] = {
    # Spin-poll for up to 100ms before sleeping — eliminates context-switch latency
    "EF_POLL_USEC":             "100000",
    "EF_SPIN_USEC":             "100000",
    # Use huge pages for the DMA buffers
    "EF_HUGE_PAGES":            "1",
    # Disable Nagle at the OpenOnload stack level
    "EF_TCP_SEND_SPIN":         "1",
    # Interrupt moderation — deliver packets immediately
    "EF_INT_DRIVEN":            "1",
    # Use receive-side spin
    "EF_RECV_SPIN":             "1",
    # Disable delayed ACKs at stack level
    "EF_DELACK_THRESH":         "1",
    # Use the fast-path TX path
    "EF_TX_PUSH":               "1",
    # Busy-wait on connect
    "EF_TCP_CONNECT_SPIN":      "1",
    # Maximise socket buffer — 16MB
    "EF_SOCKET_RECV_BUFFER":    "16777216",
    "EF_SOCKET_SEND_BUFFER":    "16777216",
    # Cluster sockets to same VI for cache locality
    "EF_CLUSTER_SIZE":          "1",
}


def apply_onload_env() -> None:
    """Apply EF_* environment variables into current process environment."""
    for k, v in _ONLOAD_ENV.items():
        if k not in os.environ:
            os.environ[k] = v
    logger.info("OnloadSocket: applied %d EF_* environment variables", len(_ONLOAD_ENV))


def make_onload_socket(
    family: int = socket.AF_INET,
    type_: int = socket.SOCK_STREAM,
    *,
    no_delay: bool = True,
    busy_poll_us: int = 50,
    rcvbuf: int = 16 * 1024 * 1024,
    sndbuf: int = 16 * 1024 * 1024,
    tos_ef: bool = True,
) -> socket.socket:
    """
    Create a socket with all low-latency options pre-applied.

    Works with or without OpenOnload installed — falls back to a
    standard kernel socket with the same socket-level options.
    """
    sock = socket.socket(family, type_)
    _apply_socket_opts(sock, no_delay=no_delay, busy_poll_us=busy_poll_us,
                       rcvbuf=rcvbuf, sndbuf=sndbuf, tos_ef=tos_ef)
    return sock


def _apply_socket_opts(
    sock: socket.socket,
    *,
    no_delay: bool,
    busy_poll_us: int,
    rcvbuf: int,
    sndbuf: int,
    tos_ef: bool,
) -> None:
    """Apply all low-latency socket options; swallow errors gracefully."""
    opts: list[tuple[str, Any]] = []
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1 if no_delay else 0)
        opts.append(("TCP_NODELAY", no_delay))
    except OSError:
        pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        opts.append(("SO_RCVBUF", rcvbuf))
    except OSError:
        pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
        opts.append(("SO_SNDBUF", sndbuf))
    except OSError:
        pass
    try:
        # SO_BUSY_POLL: spin for busy_poll_us microseconds before sleeping
        SO_BUSY_POLL = 46  # Linux socket option number
        sock.setsockopt(socket.SOL_SOCKET, SO_BUSY_POLL, busy_poll_us)
        opts.append(("SO_BUSY_POLL", busy_poll_us))
    except OSError:
        pass
    try:
        # TCP_QUICKACK: immediately ACK received data
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        opts.append(("TCP_QUICKACK", 1))
    except (OSError, AttributeError):
        pass
    try:
        if tos_ef:
            # DSCP Expedited Forwarding (EF) = 0xB8 (46 << 2)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0xB8)
            opts.append(("IP_TOS", "EF/0xB8"))
    except OSError:
        pass
    try:
        # SO_REUSEADDR + SO_REUSEPORT for fast rebind
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except OSError:
        pass
    logger.debug("OnloadSocket: applied options: %s", opts)


class OnloadSocketFactory:
    """
    Factory that creates pre-configured sockets and tracks their lifecycle.

    Usage::
        factory = OnloadSocketFactory()
        factory.apply_env()  # call once at startup
        sock = factory.tcp_client()  # TCP outbound
        sock = factory.tcp_server(port=9000)  # TCP server
        sock = factory.udp_multicast(group="239.0.0.1", port=4000)  # UDP mcast
    """

    def __init__(
        self,
        *,
        busy_poll_us: int = 50,
        rcvbuf: int = 16 * 1024 * 1024,
        sndbuf: int = 16 * 1024 * 1024,
    ):
        self.busy_poll_us = int(busy_poll_us)
        self.rcvbuf = int(rcvbuf)
        self.sndbuf = int(sndbuf)
        self._sockets: list[socket.socket] = []

    def apply_env(self) -> None:
        """Apply EF_* env vars. Call once before any socket creation."""
        apply_onload_env()

    def tcp_client(self, *, bind_iface: Optional[str] = None) -> socket.socket:
        """Create a TCP client socket optimised for low-latency outbound."""
        sock = make_onload_socket(
            socket.AF_INET, socket.SOCK_STREAM,
            busy_poll_us=self.busy_poll_us,
            rcvbuf=self.rcvbuf,
            sndbuf=self.sndbuf,
        )
        if bind_iface:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
                                bind_iface.encode())
            except OSError as e:
                logger.debug("SO_BINDTODEVICE %s: %s", bind_iface, e)
        self._sockets.append(sock)
        return sock

    def tcp_server(self, port: int, *, backlog: int = 128) -> socket.socket:
        """Create a TCP server socket bound to port."""
        sock = make_onload_socket(
            socket.AF_INET, socket.SOCK_STREAM,
            busy_poll_us=self.busy_poll_us,
            rcvbuf=self.rcvbuf,
            sndbuf=self.sndbuf,
        )
        sock.bind(("", port))
        sock.listen(backlog)
        self._sockets.append(sock)
        return sock

    def udp_socket(self, *, multicast: bool = False) -> socket.socket:
        """Create a UDP socket for market data (unicast or multicast)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _apply_socket_opts(sock, no_delay=False, busy_poll_us=self.busy_poll_us,
                           rcvbuf=self.rcvbuf, sndbuf=self.sndbuf, tos_ef=True)
        if multicast:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 3)
            try:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 0)
            except OSError:
                pass
        self._sockets.append(sock)
        return sock

    def close_all(self) -> None:
        for s in self._sockets:
            try:
                s.close()
            except Exception:
                pass
        self._sockets.clear()
