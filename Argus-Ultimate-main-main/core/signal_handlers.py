"""M08 — SIGTERM / SIGINT graceful shutdown handler.

Usage::

    from core.signal_handlers import install_signal_handlers, request_shutdown, is_shutdown_requested

    install_signal_handlers()  # call once at startup

    # In your main loop:
    while not is_shutdown_requested():
        ...
"""
from __future__ import annotations

import asyncio
import logging
import signal
import threading

logger = logging.getLogger(__name__)

_shutdown_event: threading.Event = threading.Event()
_async_shutdown_event: asyncio.Event | None = None
_registered: bool = False


def install_signal_handlers(
    *,
    extra_cleanup: list | None = None,
) -> None:
    """Register SIGTERM and SIGINT handlers.

    Args:
        extra_cleanup: Optional list of zero-arg callables invoked on signal
                       receipt before the shutdown event is set.
    """
    global _registered
    if _registered:
        return

    _cleanup_callbacks: list = list(extra_cleanup or [])

    def _handle(signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning("Received %s — initiating graceful shutdown", sig_name)
        for cb in _cleanup_callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001
                logger.exception("Cleanup callback raised during signal handling")
        _shutdown_event.set()
        # Wake any async loop that is listening
        if _async_shutdown_event is not None:
            try:
                _async_shutdown_event.set()
            except RuntimeError:
                pass  # event loop already closed

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    _registered = True
    logger.debug("Signal handlers installed (SIGTERM + SIGINT)")


def request_shutdown() -> None:
    """Programmatically trigger a shutdown (useful in tests)."""
    _shutdown_event.set()


def is_shutdown_requested() -> bool:
    """Return True if a shutdown has been signalled."""
    return _shutdown_event.is_set()


def reset_shutdown() -> None:  # for test teardown
    """Clear the shutdown flag (test helper — do not use in production)."""
    global _registered
    _shutdown_event.clear()
    _registered = False


def get_async_shutdown_event(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Event:
    """Return (or create) the asyncio.Event that mirrors the thread-level flag."""
    global _async_shutdown_event
    if _async_shutdown_event is None:
        _async_shutdown_event = asyncio.Event()
    return _async_shutdown_event
