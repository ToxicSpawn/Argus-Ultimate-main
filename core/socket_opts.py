"""Low-level socket option helpers — Push 99 RegimeGrafana.

Applies TCP_NODELAY (and optionally SO_SNDBUF / SO_RCVBUF) to any
async / sync socket used by exchange connectors to eliminate Nagle
buffering on order-submission paths.
"""
from __future__ import annotations

import socket
import logging
from typing import Union

log = logging.getLogger(__name__)

# Default buffer sizes (bytes). 0 = keep OS default.
DEFAULT_SNDBUF: int = 131_072   # 128 KiB
DEFAULT_RCVBUF: int = 131_072   # 128 KiB


def apply_tcp_nodelay(sock: socket.socket) -> None:
    """Set TCP_NODELAY on *sock*, disabling Nagle's algorithm."""
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.debug("TCP_NODELAY set on fd=%d", sock.fileno())
    except OSError as exc:
        log.warning("Failed to set TCP_NODELAY on fd=%d: %s", sock.fileno(), exc)


def apply_socket_opts(
    sock: socket.socket,
    *,
    tcp_nodelay: bool = True,
    sndbuf: int = DEFAULT_SNDBUF,
    rcvbuf: int = DEFAULT_RCVBUF,
    keepalive: bool = True,
    keepalive_idle:     int = 10,
    keepalive_interval: int = 3,
    keepalive_count:    int = 5,
) -> None:
    """Apply a full suite of latency-optimised socket options.

    Parameters
    ----------
    sock:               Target socket (must be SOCK_STREAM / TCP).
    tcp_nodelay:        Disable Nagle algorithm (default True).
    sndbuf:             Send-buffer size in bytes (0 = OS default).
    rcvbuf:             Recv-buffer size in bytes (0 = OS default).
    keepalive:          Enable TCP keepalive probes.
    keepalive_idle:     Seconds before first keepalive probe.
    keepalive_interval: Seconds between keepalive probes.
    keepalive_count:    Number of unanswered probes before drop.
    """
    fd = sock.fileno()

    if tcp_nodelay:
        apply_tcp_nodelay(sock)

    if sndbuf > 0:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
            log.debug("SO_SNDBUF=%d on fd=%d", sndbuf, fd)
        except OSError as exc:
            log.warning("SO_SNDBUF failed fd=%d: %s", fd, exc)

    if rcvbuf > 0:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
            log.debug("SO_RCVBUF=%d on fd=%d", rcvbuf, fd)
        except OSError as exc:
            log.warning("SO_RCVBUF failed fd=%d: %s", fd, exc)

    if keepalive:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # Linux-specific fine-grained keepalive tuning
            if hasattr(socket, "TCP_KEEPIDLE"):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE,     keepalive_idle)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL,    keepalive_interval)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT,      keepalive_count)
            log.debug("SO_KEEPALIVE set on fd=%d", fd)
        except OSError as exc:
            log.warning("SO_KEEPALIVE failed fd=%d: %s", fd, exc)


def patch_ssl_socket(ssl_sock: object) -> None:
    """Best-effort TCP_NODELAY patch for an ssl.SSLSocket wrapper."""
    raw: Union[socket.socket, None] = getattr(ssl_sock, "_sock", None)
    if raw is None:
        raw = getattr(ssl_sock, "socket", None)
    if isinstance(raw, socket.socket):
        apply_tcp_nodelay(raw)
    else:
        log.debug("patch_ssl_socket: no raw socket found on %r", ssl_sock)
