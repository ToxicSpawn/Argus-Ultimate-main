"""
lan_signal_bridge.py — LAN signal bridge between PC (GPU sender) and R7525 (receiver).

Architecture
------------
PC side  (role="sender"):
    GPUInferenceServer publishes msgpack signals over ZMQ PUB on tcp://*:9200.
    LANSignalSender is a thin wrapper used by GPUInferenceServer internally.

R7525 side (role="receiver"):
    LANSignalReceiver connects a ZMQ SUB socket to tcp://argus-pc:9200,
    decodes signals, fires registered callbacks, and maintains per-symbol cache.
    If no signal arrives for signal_timeout_ms, is_stale() returns True and
    get_latest_signal() returns None — triggering CPU OBI fallback in
    alpha/microstructure/deeplob_live_bridge.py.

High-level API
--------------
    # On PC
    bridge = LANSignalBridge(role="sender", publish_address="tcp://*:9200")
    asyncio.run(bridge.start())

    # On R7525
    bridge = LANSignalBridge(role="receiver", subscribe_address="tcp://argus-pc:9200")
    asyncio.run(bridge.start())
    signal = bridge.get_signal("BTC/USDT")
"""
from __future__ import annotations

import asyncio
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Optional msgpack ───────────────────────────────────────────────────────────
try:
    import msgpack
    _MSGPACK_AVAILABLE = True
except ImportError:
    import json as _json_fallback  # type: ignore
    _MSGPACK_AVAILABLE = False
    logger.warning("lan_signal_bridge: msgpack not available — falling back to JSON")

# ── Optional ZMQ ──────────────────────────────────────────────────────────────
try:
    import zmq
    import zmq.asyncio as zmq_asyncio
    _ZMQ_AVAILABLE = True
except ImportError:
    _ZMQ_AVAILABLE = False
    logger.warning("lan_signal_bridge: pyzmq not available — bridge will be no-op")


# ─── Serialisation helpers ────────────────────────────────────────────────────


def _pack(data: dict) -> bytes:
    if _MSGPACK_AVAILABLE:
        return msgpack.packb(data, use_bin_type=True)
    return _json_fallback.dumps(data).encode("utf-8")


def _unpack(raw: bytes) -> dict:
    if _MSGPACK_AVAILABLE:
        return msgpack.unpackb(raw, raw=False)
    return _json_fallback.loads(raw.decode("utf-8"))


# ─── ReceiverConfig ───────────────────────────────────────────────────────────


@dataclass
class ReceiverConfig:
    """Configuration for LANSignalReceiver (R7525 side)."""

    subscribe_address: str = "tcp://argus-pc:9200"  # PC's LAN hostname or IP
    signal_timeout_ms: float = 5000.0               # stale threshold
    fallback_to_cpu: bool = True                     # emit None when stale
    recv_timeout_ms: int = 100                       # ZMQ RCVTIMEO


# ─── SenderConfig ─────────────────────────────────────────────────────────────


@dataclass
class SenderConfig:
    """Configuration for LANSignalSender (PC side)."""

    publish_address: str = "tcp://*:9200"


# ─── LANSignalReceiver ────────────────────────────────────────────────────────


class LANSignalReceiver:
    """ZMQ SUB receiver that listens for DeepLOB signals published by the PC.

    Runs on the R7525 server. Decodes msgpack messages, caches the latest
    signal per symbol, and fires registered callbacks.

    Stale detection: if no signal has arrived within signal_timeout_ms the
    bridge is considered stale; get_latest_signal() returns None so that the
    caller can activate the CPU OBI fallback path.

    Example
    -------
        cfg = ReceiverConfig(subscribe_address="tcp://192.168.1.10:9200")
        recv = LANSignalReceiver(cfg)
        recv.on_signal(lambda sig: print(sig))
        asyncio.run(recv.start())
    """

    def __init__(self, config: Optional[ReceiverConfig] = None) -> None:
        self.config = config or ReceiverConfig()
        self._callbacks: List[Callable[[dict], None]] = []
        self._latest: Dict[str, dict] = {}        # symbol → last signal
        self._signals_received: int = 0
        self._last_signal_time_ns: int = 0
        self._latencies: List[float] = []         # one-way latency estimates (us)
        self._running: bool = False
        self._connected: bool = False
        self._ctx: Optional[object] = None
        self._sub: Optional[object] = None

    # ── Connection ────────────────────────────────────────────────────────────

    def _open_socket(self) -> None:
        if not _ZMQ_AVAILABLE:
            logger.warning("lan_signal_bridge: ZMQ not available — receiver disabled")
            return
        self._ctx = zmq_asyncio.Context.instance()
        self._sub = self._ctx.socket(zmq.SUB)
        # Subscribe to all messages (empty topic filter)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
        self._sub.connect(self.config.subscribe_address)
        self._connected = True
        logger.info(
            "LANSignalReceiver: connected to %s", self.config.subscribe_address
        )

    def _close_socket(self) -> None:
        if self._sub is not None:
            try:
                self._sub.close()
            except Exception:
                pass
            self._sub = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None
        self._connected = False

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect ZMQ SUB socket and start the receive loop."""
        self._open_socket()
        self._running = True
        await self._receive_loop()

    async def stop(self) -> None:
        """Stop the receive loop and close the socket."""
        logger.info("LANSignalReceiver: stopping")
        self._running = False
        self._close_socket()

    def on_signal(self, callback: Callable[[dict], None]) -> None:
        """Register a callback that will be called for every received signal.

        Parameters
        ----------
        callback : callable(signal: dict) → None
        """
        self._callbacks.append(callback)

    def get_latest_signal(self, symbol: str) -> Optional[dict]:
        """Return the most recent signal for *symbol*.

        Returns None if:
          • No signal has ever been received for this symbol
          • The bridge is stale AND config.fallback_to_cpu is True
        """
        if self.is_stale() and self.config.fallback_to_cpu:
            warnings.warn(
                f"LANSignalReceiver: stale (no signal for >"
                f"{self.config.signal_timeout_ms:.0f}ms) — "
                "returning None, CPU fallback active",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return self._latest.get(symbol)

    def is_connected(self) -> bool:
        """True if the ZMQ socket is open and connected."""
        return self._connected

    def is_stale(self) -> bool:
        """True if the last received signal is older than signal_timeout_ms."""
        if self._last_signal_time_ns == 0:
            # Never received anything; consider stale only if we have been running a while
            return self._running
        elapsed_ms = (time.time_ns() - self._last_signal_time_ns) / 1_000_000
        return elapsed_ms > self.config.signal_timeout_ms

    def get_stats(self) -> Dict:
        """Return receiver telemetry."""
        import numpy as np
        avg_lat = float(np.mean(self._latencies)) if self._latencies else 0.0
        return {
            "connected": self._connected,
            "last_signal_time_ns": self._last_signal_time_ns,
            "signals_received": self._signals_received,
            "avg_latency_us": avg_lat,
            "stale": self.is_stale(),
            "using_fallback": self.is_stale() and self.config.fallback_to_cpu,
            "symbols_tracked": list(self._latest.keys()),
            "subscribe_address": self.config.subscribe_address,
        }

    # ── Receive loop ──────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """Async loop: poll ZMQ socket, decode messages, update cache."""
        logger.info("LANSignalReceiver: receive loop started")
        while self._running:
            if self._sub is None:
                await asyncio.sleep(0.1)
                continue
            try:
                raw: bytes = await self._sub.recv()
                self._handle_raw(raw)
            except zmq.Again:
                # Timeout — normal, just continue
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("LANSignalReceiver: recv error: %s", exc)
                await asyncio.sleep(0.01)

    def _handle_raw(self, raw: bytes) -> None:
        """Decode a raw ZMQ message and dispatch it."""
        try:
            signal = _unpack(raw)
        except Exception as exc:
            logger.debug("LANSignalReceiver: decode error: %s", exc)
            return

        if not isinstance(signal, dict):
            return

        symbol = signal.get("symbol")
        if symbol:
            self._latest[symbol] = signal

        now_ns = time.time_ns()
        self._last_signal_time_ns = now_ns
        self._signals_received += 1

        # Estimate one-way latency from embedded timestamp
        ts_ns = signal.get("timestamp_ns")
        if ts_ns and isinstance(ts_ns, int):
            latency_us = (now_ns - ts_ns) / 1_000
            if 0 <= latency_us < 10_000_000:   # sanity: < 10 seconds
                self._latencies.append(latency_us)
                if len(self._latencies) > 10_000:
                    self._latencies = self._latencies[-5_000:]

        # Fire callbacks
        for cb in self._callbacks:
            try:
                cb(signal)
            except Exception as exc:
                logger.warning("LANSignalReceiver: callback error: %s", exc)


# ─── LANSignalSender ─────────────────────────────────────────────────────────


class LANSignalSender:
    """Thin ZMQ PUB wrapper used by GPUInferenceServer.

    Serialises signal dicts to msgpack and publishes over the PUB socket.
    Also usable standalone for testing or injection of manually crafted signals.

    Example
    -------
        sender = LANSignalSender("tcp://*:9200")
        sender.bind()
        sender.send_signal({"type": "deeplob_signal", "symbol": "BTC/USDT", ...})
        sender.close()
    """

    def __init__(self, publish_address: str = "tcp://*:9200") -> None:
        self.publish_address = publish_address
        self._ctx: Optional[object] = None
        self._pub: Optional[object] = None
        self._bound: bool = False
        self._sent: int = 0

    def bind(self) -> None:
        """Bind the ZMQ PUB socket to publish_address."""
        if not _ZMQ_AVAILABLE:
            logger.warning("LANSignalSender: ZMQ not available — send is a no-op")
            return
        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(self.publish_address)
        self._bound = True
        logger.info("LANSignalSender: PUB socket bound to %s", self.publish_address)

    def send_signal(self, signal: dict) -> None:
        """Msgpack-serialise *signal* and publish it.

        Parameters
        ----------
        signal : dict matching the DeepLOB signal schema
        """
        if self._pub is None:
            logger.debug("LANSignalSender: socket not bound — dropping signal")
            return
        try:
            payload = _pack(signal)
            self._pub.send(payload, zmq.NOBLOCK)
            self._sent += 1
        except Exception as exc:
            logger.debug("LANSignalSender: send error: %s", exc)

    def close(self) -> None:
        """Close the PUB socket and terminate the ZMQ context."""
        if self._pub is not None:
            try:
                self._pub.close()
            except Exception:
                pass
            self._pub = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None
        self._bound = False

    @property
    def sent_count(self) -> int:
        return self._sent

    def __enter__(self) -> "LANSignalSender":
        self.bind()
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ─── LANSignalBridge ─────────────────────────────────────────────────────────


class LANSignalBridge:
    """High-level coordinator that starts either sender (PC) or receiver (R7525).

    Parameters
    ----------
    role : "sender" or "receiver"
    publish_address  : ZMQ bind address for sender (default tcp://*:9200)
    subscribe_address: ZMQ connect address for receiver (default tcp://argus-pc:9200)

    Example
    -------
        # PC side
        bridge = LANSignalBridge(role="sender")
        asyncio.run(bridge.start())

        # R7525 side
        bridge = LANSignalBridge(role="receiver")
        asyncio.run(bridge.start())
        sig = bridge.get_signal("BTC/USDT")
    """

    VALID_ROLES = ("sender", "receiver")

    def __init__(
        self,
        role: str,
        publish_address: str = "tcp://*:9200",
        subscribe_address: str = "tcp://argus-pc:9200",
        receiver_config: Optional[ReceiverConfig] = None,
        sender_config: Optional[SenderConfig] = None,
    ) -> None:
        if role not in self.VALID_ROLES:
            raise ValueError(
                f"LANSignalBridge: role must be one of {self.VALID_ROLES}, got '{role}'"
            )
        self.role = role
        self._sender: Optional[LANSignalSender] = None
        self._receiver: Optional[LANSignalReceiver] = None

        if role == "sender":
            cfg = sender_config or SenderConfig(publish_address=publish_address)
            self._sender = LANSignalSender(publish_address=cfg.publish_address)

        elif role == "receiver":
            cfg_r = receiver_config or ReceiverConfig(subscribe_address=subscribe_address)
            self._receiver = LANSignalReceiver(config=cfg_r)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the appropriate component based on role.

        For "sender": binds ZMQ PUB socket (non-blocking — call send_signal to publish).
        For "receiver": connects ZMQ SUB socket and enters the receive loop (blocking
        until stop() is called or the task is cancelled).
        """
        if self.role == "sender":
            if self._sender is not None:
                self._sender.bind()
                logger.info("LANSignalBridge[sender]: ready on %s", self._sender.publish_address)

        elif self.role == "receiver":
            if self._receiver is not None:
                await self._receiver.start()

    async def stop(self) -> None:
        """Stop and clean up."""
        if self.role == "sender" and self._sender is not None:
            self._sender.close()
        elif self.role == "receiver" and self._receiver is not None:
            await self._receiver.stop()

    # ── Unified signal interface ──────────────────────────────────────────────

    def get_signal(self, symbol: str) -> Optional[dict]:
        """Unified signal accessor regardless of role.

        For "receiver": returns the latest cached GPU signal (or None if stale/absent).
        For "sender"  : always returns None (the PC publishes, it does not subscribe).

        Returns
        -------
        dict or None
        """
        if self.role == "receiver" and self._receiver is not None:
            return self._receiver.get_latest_signal(symbol)
        return None

    def send_signal(self, signal: dict) -> None:
        """Publish a signal (sender role only).

        Parameters
        ----------
        signal : dict conforming to the DeepLOB signal schema
        """
        if self.role == "sender" and self._sender is not None:
            self._sender.send_signal(signal)
        else:
            logger.debug("LANSignalBridge.send_signal: no-op for role='%s'", self.role)

    def on_signal(self, callback: Callable[[dict], None]) -> None:
        """Register a callback for incoming signals (receiver role only)."""
        if self.role == "receiver" and self._receiver is not None:
            self._receiver.on_signal(callback)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        if self.role == "sender":
            return self._sender is not None and self._sender._bound
        if self.role == "receiver":
            return self._receiver is not None and self._receiver.is_connected()
        return False

    def is_stale(self) -> bool:
        """Only meaningful for receiver role."""
        if self.role == "receiver" and self._receiver is not None:
            return self._receiver.is_stale()
        return False

    def get_stats(self) -> Dict:
        """Return diagnostics for the active component."""
        if self.role == "sender" and self._sender is not None:
            return {
                "role": "sender",
                "publish_address": self._sender.publish_address,
                "bound": self._sender._bound,
                "sent_count": self._sender.sent_count,
            }
        if self.role == "receiver" and self._receiver is not None:
            stats = self._receiver.get_stats()
            stats["role"] = "receiver"
            return stats
        return {"role": self.role, "active": False}
