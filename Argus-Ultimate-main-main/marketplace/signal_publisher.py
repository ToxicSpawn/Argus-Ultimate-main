"""
Copy-Trading Signal Publisher & Subscriber.

Directly targets 3Commas SmartTrade copy-trading and Cryptohopper signal marketplace.

How it works:
  Publisher: runs inside Argus main bot, publishes every order to a ZMQ PUB socket.
  Subscriber: any other Argus instance (second account, paper trading, shadow bot)
    connects and mirrors all orders with configurable lot-size scaling.

This means:
  - Run Argus on Account A with $1K -> auto-mirror to Account B with $500
  - Run a shadow paper-trading subscriber to validate before going live
  - No platform fees (vs 3Commas $15-79/mo for SmartTrade copy)

Dependency: pyzmq (pip install pyzmq)
Fallback: queue-based in-process pub/sub if pyzmq not installed.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PUB_PORT = 5555
DEFAULT_SUB_PORT = 5555
SIGNAL_TOPIC = b"argus_signal"


@dataclass
class TradeSignal:
    strategy_id: str
    action: str          # 'buy' | 'sell'
    symbol: str
    qty: float
    price: float
    order_type: str      # 'market' | 'limit'
    reason: str
    timestamp: float
    source_account: str  # identifies origin bot


class SignalPublisher:
    """
    Publishes Argus trade signals over ZMQ PUB socket.

    Usage:
        publisher = SignalPublisher(port=5555)
        publisher.start()
        publisher.publish(TradeSignal(...))
    """

    def __init__(self, port: int = DEFAULT_PUB_PORT,
                 account_id: str = "argus_primary") -> None:
        self.port = port
        self.account_id = account_id
        self._socket = None
        self._context = None
        self._fallback_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._using_zmq = False

    def start(self) -> None:
        try:
            import zmq
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.bind(f"tcp://*:{self.port}")
            self._context = context
            self._socket = socket
            self._using_zmq = True
            logger.info("SignalPublisher ZMQ PUB socket bound on port %d", self.port)
        except ImportError:
            logger.warning("pyzmq not installed — using in-process fallback queue")

    def publish(self, signal: TradeSignal) -> None:
        # Build a new signal dict with source_account and timestamp stamped in,
        # without mutating the caller's original TradeSignal object.
        data = asdict(signal)
        data["source_account"] = self.account_id
        data["timestamp"] = time.time()
        payload = json.dumps(data).encode()
        if self._using_zmq and self._socket:
            self._socket.send_multipart([SIGNAL_TOPIC, payload])
        else:
            try:
                self._fallback_queue.put_nowait(payload)
            except queue.Full:
                logger.warning("SignalPublisher fallback queue full — dropping signal")

    def stop(self) -> None:
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()


class SignalSubscriber:
    """
    Subscribes to Argus signals and mirrors them on a second account.

    Usage:
        subscriber = SignalSubscriber(
            host='127.0.0.1', port=5555,
            lot_scale=0.5,     # mirror at 50% of original size
            on_signal=my_executor.execute,
        )
        subscriber.start()  # runs in background thread
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_SUB_PORT,
        lot_scale: float = 1.0,
        on_signal: Optional[Callable[[TradeSignal], None]] = None,
        publisher: Optional[SignalPublisher] = None,  # for in-process fallback
    ) -> None:
        self.host = host
        self.port = port
        self.lot_scale = lot_scale
        self._on_signal = on_signal
        self._publisher = publisher
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._received_signals: List[TradeSignal] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("SignalSubscriber started (scale=%.2f)", self.lot_scale)

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        try:
            import zmq
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.connect(f"tcp://{self.host}:{self.port}")
            socket.setsockopt(zmq.SUBSCRIBE, SIGNAL_TOPIC)
            socket.setsockopt(zmq.RCVTIMEO, 500)  # 500ms timeout
            logger.info("SignalSubscriber ZMQ connected to %s:%d", self.host, self.port)
            while not self._stop_event.is_set():
                try:
                    _, payload = socket.recv_multipart()
                    self._handle(payload)
                except Exception:
                    pass
            socket.close()
            context.term()
        except ImportError:
            self._run_fallback()

    def _run_fallback(self) -> None:
        """In-process queue fallback when pyzmq is not installed."""
        if self._publisher is None:
            return
        while not self._stop_event.is_set():
            try:
                payload = self._publisher._fallback_queue.get(timeout=0.5)
                self._handle(payload)
            except queue.Empty:
                pass

    def _handle(self, payload: bytes) -> None:
        data = json.loads(payload.decode())
        signal = TradeSignal(**data)
        signal.qty = signal.qty * self.lot_scale
        self._received_signals.append(signal)
        if self._on_signal:
            try:
                self._on_signal(signal)
            except Exception as exc:
                logger.error("SignalSubscriber on_signal error: %s", exc)
