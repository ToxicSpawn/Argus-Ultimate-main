"""
gpu_inference_server.py — GPU inference server for DeepLOB signals on RTX 5080.

Runs on the PC side of the Argus architecture:
  • Loads trained DeepLOB weights to GPU
  • Maintains per-symbol LOB sliding windows
  • Runs batched fp16 inference every inference_interval_ms milliseconds
  • Broadcasts results over ZMQ PUB socket on tcp://*:9200 (msgpack serialised)

R7525 server subscribes via ml/lan_signal_bridge.py:LANSignalReceiver.

Signal format (msgpack)
-----------------------
{
    "type": "deeplob_signal",
    "symbol": "BTC/USDT",
    "direction": "up",          # "up" / "down" / "neutral"
    "confidence": 0.85,
    "logits": [0.05, 0.10, 0.85],
    "timestamp_ns": 1234567890,
    "inference_latency_us": 450,
    "model_version": "v1"
}
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional torch ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("gpu_inference_server: torch not available")

# ── Optional msgpack ───────────────────────────────────────────────────────────
try:
    import msgpack
    _MSGPACK_AVAILABLE = True
except ImportError:
    import json as _json_fallback
    _MSGPACK_AVAILABLE = False
    logger.warning("gpu_inference_server: msgpack not available — falling back to JSON")

# ── Optional ZMQ ──────────────────────────────────────────────────────────────
try:
    import zmq
    import zmq.asyncio as zmq_asyncio
    _ZMQ_AVAILABLE = True
except ImportError:
    _ZMQ_AVAILABLE = False
    logger.warning("gpu_inference_server: pyzmq not available — ZMQ disabled")


# ─── InferenceServerConfig ────────────────────────────────────────────────────


@dataclass
class InferenceServerConfig:
    """Configuration for the GPU inference server."""

    device: str = "cuda:0"
    model_path: str = "models/deeplob_weights.pt"
    publish_address: str = "tcp://*:9200"   # ZMQ PUB socket — any LAN client can subscribe
    inference_interval_ms: float = 100.0    # run inference every 100 ms
    batch_size: int = 64
    mixed_precision: bool = True            # fp16 on RTX 5080 Tensor Cores
    warmup_batches: int = 5                 # warm up CUDA before real inference
    sequence_length: int = 100             # LOB snapshots per inference window
    feature_dim: int = 40                  # 10 levels × 4
    n_classes: int = 3
    model_version: str = "v1"
    n_levels: int = 10                     # LOB depth levels


# ─── LOB feature extractor (mirrors deep_lob.py) ──────────────────────────────


class _LOBFeatureExtractor:
    """Extract normalised 40-d feature vector from raw LOB levels."""

    @staticmethod
    def extract(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        n_levels: int = 10,
    ) -> Optional[np.ndarray]:
        """
        Parameters
        ----------
        bids : list of (price, size) sorted descending (best bid first)
        asks : list of (price, size) sorted ascending  (best ask first)

        Returns
        -------
        np.ndarray shape (4*n_levels,) or None on bad data
        """
        if not bids or not asks:
            return None
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
            return None
        mid = (best_bid + best_ask) / 2.0

        def _pad(levels: list, n: int) -> List[Tuple[float, float]]:
            lvls = [(float(p), float(s)) for p, s in levels[:n]]
            while len(lvls) < n:
                lvls.append((0.0, 0.0))
            return lvls

        bp = _pad(bids, n_levels)
        ap = _pad(asks, n_levels)
        bid_prices = np.array([p for p, _ in bp], dtype=np.float32)
        ask_prices = np.array([p for p, _ in ap], dtype=np.float32)
        bid_sizes  = np.array([s for _, s in bp], dtype=np.float32)
        ask_sizes  = np.array([s for _, s in ap], dtype=np.float32)

        bid_prices = (bid_prices - mid) / (mid + 1e-12)
        ask_prices = (ask_prices - mid) / (mid + 1e-12)
        total_depth = bid_sizes.sum() + ask_sizes.sum() + 1e-12
        bid_sizes /= total_depth
        ask_sizes /= total_depth

        return np.concatenate([bid_prices, ask_prices, bid_sizes, ask_sizes]).astype(np.float32)


# ─── DeepLOB model (mirrors ml/deep_lob.py, GPU-ready) ───────────────────────


if _TORCH_AVAILABLE:
    class _InferenceDeepLOBNet(nn.Module):
        """Conv1D → LSTM → FC — identical architecture to ml/deep_lob.py.

        Accepts single snapshots: input shape (B, feature_dim).
        """

        def __init__(self, input_dim: int = 40, n_classes: int = 3) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=2)
            self.conv2 = nn.Conv1d(32, 32, kernel_size=2)
            self.pool  = nn.MaxPool1d(kernel_size=2)
            self.leaky = nn.LeakyReLU(0.01)

            with torch.no_grad():
                probe = torch.zeros(1, 1, input_dim)
                probe = self.pool(self.leaky(self.conv2(self.leaky(self.conv1(probe)))))
                self._lstm_seq = probe.shape[2]

            self.lstm = nn.LSTM(32, 64, num_layers=1, batch_first=True)
            self.fc   = nn.Linear(64, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x.unsqueeze(1)                 # (B, 1, D)
            x = self.leaky(self.conv1(x))      # (B, 32, D-1)
            x = self.leaky(self.conv2(x))      # (B, 32, D-3)
            x = self.pool(x)                   # (B, 32, ...)
            x = x.transpose(1, 2)             # (B, L, 32)
            _, (h, _) = self.lstm(x)
            h = h.squeeze(0)                  # (B, 64)
            return self.fc(h)                  # (B, 3)


# ─── Per-symbol state ─────────────────────────────────────────────────────────


@dataclass
class _SymbolBuffer:
    """Sliding window buffer for one symbol."""
    window: Deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=100)
    )

    def update_maxlen(self, seq_len: int) -> None:
        if self.window.maxlen != seq_len:
            new_window: Deque[np.ndarray] = deque(
                list(self.window)[-seq_len:], maxlen=seq_len
            )
            self.window = new_window

    def is_full(self) -> bool:
        return len(self.window) == self.window.maxlen

    def get_array(self) -> np.ndarray:
        """Return window as (seq_len, feature_dim) float32 array."""
        return np.stack(list(self.window), axis=0).astype(np.float32)


# ─── Serialisation helpers ───────────────────────────────────────────────────


def _pack(data: dict) -> bytes:
    if _MSGPACK_AVAILABLE:
        return msgpack.packb(data, use_bin_type=True)
    return _json_fallback.dumps(data).encode("utf-8")


def _unpack(raw: bytes) -> dict:
    if _MSGPACK_AVAILABLE:
        return msgpack.unpackb(raw, raw=False)
    return _json_fallback.loads(raw.decode("utf-8"))


# ─── Latency tracker ─────────────────────────────────────────────────────────


class _LatencyTracker:
    """Rolling percentile tracker for inference latency (microseconds)."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._buf: Deque[float] = deque(maxlen=maxlen)

    def record(self, us: float) -> None:
        self._buf.append(us)

    def stats(self) -> Dict:
        if not self._buf:
            return {"p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0, "avg_us": 0.0}
        arr = np.array(self._buf)
        return {
            "avg_us": float(arr.mean()),
            "p50_us": float(np.percentile(arr, 50)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
        }


# ─── GPUInferenceServer ───────────────────────────────────────────────────────


class GPUInferenceServer:
    """GPU inference server: batches LOB windows → DeepLOB forward pass → ZMQ PUB.

    Designed to run on the PC (RTX 5080) and broadcast signals over LAN to the
    R7525 server.

    Example
    -------
        server = GPUInferenceServer()
        asyncio.run(server.start())
    """

    DIRECTION_MAP = {0: "up", 1: "neutral", 2: "down"}

    def __init__(self, config: Optional[InferenceServerConfig] = None) -> None:
        self.config = config or InferenceServerConfig()
        self.model: Optional["nn.Module"] = None
        self._model_loaded = False
        self._device: Optional["torch.device"] = None
        self._extractor = _LOBFeatureExtractor()
        self._buffers: Dict[str, _SymbolBuffer] = {}
        self._latency = _LatencyTracker()
        self._inference_count = 0
        self._running = False
        self._zmq_ctx: Optional[object] = None
        self._pub_socket: Optional[object] = None
        self._use_amp = False

        # Resolve device
        self._device_str = self._resolve_device(self.config.device)

    # ── Device ────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(req: str) -> str:
        if not _TORCH_AVAILABLE:
            return "cpu"
        if req.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                f"gpu_inference_server: {req} requested but CUDA unavailable — using cpu",
                RuntimeWarning,
            )
            return "cpu"
        return req

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load DeepLOB weights from disk to GPU."""
        if not _TORCH_AVAILABLE:
            logger.warning("gpu_inference_server: torch unavailable — running without model")
            return

        self._device = torch.device(self._device_str)
        self._use_amp = self.config.mixed_precision and self._device_str.startswith("cuda")

        net = _InferenceDeepLOBNet(
            input_dim=self.config.feature_dim,
            n_classes=self.config.n_classes,
        ).to(self._device)

        model_path = self.config.model_path
        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=self._device)
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    net.load_state_dict(ckpt["state_dict"])
                elif isinstance(ckpt, dict):
                    net.load_state_dict(ckpt)
                logger.info(
                    "gpu_inference_server: model loaded from %s on %s",
                    model_path, self._device_str,
                )
                self._model_loaded = True
            except Exception as exc:
                logger.warning(
                    "gpu_inference_server: failed to load model from %s: %s "
                    "— running with random weights",
                    model_path, exc,
                )
                self._model_loaded = False
        else:
            logger.warning(
                "gpu_inference_server: model file not found at %s "
                "— running with uninitialised weights",
                model_path,
            )
            self._model_loaded = False

        net.eval()
        self.model = net

    def _warmup(self) -> None:
        """Run dummy forward passes to trigger CUDA JIT compilation."""
        if not _TORCH_AVAILABLE or self.model is None:
            return
        dummy = torch.zeros(
            self.config.batch_size,
            self.config.feature_dim,
            device=self._device,
        )
        for i in range(self.config.warmup_batches):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                with torch.no_grad():
                    _ = self.model(dummy)
        if self._device_str.startswith("cuda"):
            torch.cuda.synchronize()
        logger.info(
            "gpu_inference_server: GPU warmup complete (%d batches)",
            self.config.warmup_batches,
        )

    # ── ZMQ socket ────────────────────────────────────────────────────────────

    def _open_zmq(self) -> None:
        if not _ZMQ_AVAILABLE:
            logger.warning("gpu_inference_server: pyzmq unavailable — ZMQ disabled")
            return
        self._zmq_ctx = zmq_asyncio.Context.instance()
        self._pub_socket = self._zmq_ctx.socket(zmq.PUB)
        self._pub_socket.bind(self.config.publish_address)
        logger.info(
            "gpu_inference_server: ZMQ PUB socket bound to %s",
            self.config.publish_address,
        )

    def _close_zmq(self) -> None:
        if self._pub_socket is not None:
            try:
                self._pub_socket.close()
            except Exception:
                pass
            self._pub_socket = None
        if self._zmq_ctx is not None:
            try:
                self._zmq_ctx.term()
            except Exception:
                pass
            self._zmq_ctx = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Load model, warm up CUDA, open ZMQ socket, start inference loop."""
        logger.info("gpu_inference_server: starting on device=%s", self._device_str)
        self._load_model()
        self._warmup()
        self._open_zmq()
        self._running = True
        await self.run_inference_loop()

    async def stop(self) -> None:
        """Gracefully stop the inference loop and close resources."""
        logger.info("gpu_inference_server: stopping")
        self._running = False
        self._close_zmq()

    def on_book_update(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp_ns: int,
    ) -> None:
        """Ingest a new LOB snapshot and update the symbol's sliding window.

        Parameters
        ----------
        symbol       : trading symbol, e.g. "BTC/USDT"
        bids         : list of (price, size) sorted descending
        asks         : list of (price, size) sorted ascending
        timestamp_ns : UNIX timestamp in nanoseconds
        """
        feat = self._extractor.extract(bids, asks, n_levels=self.config.n_levels)
        if feat is None:
            return

        if symbol not in self._buffers:
            buf = _SymbolBuffer()
            buf.update_maxlen(self.config.sequence_length)
            self._buffers[symbol] = buf

        buf = self._buffers[symbol]
        buf.update_maxlen(self.config.sequence_length)
        buf.window.append(feat)

    async def run_inference_loop(self) -> None:
        """Main loop: batch all ready symbols → GPU forward pass → ZMQ publish."""
        interval_s = self.config.inference_interval_ms / 1000.0
        logger.info(
            "gpu_inference_server: inference loop started (interval=%.1fms)",
            self.config.inference_interval_ms,
        )

        while self._running:
            loop_start = time.perf_counter()

            ready_symbols = [
                sym for sym, buf in self._buffers.items() if buf.is_full()
            ]

            if ready_symbols and _TORCH_AVAILABLE and self.model is not None:
                # Build batch
                batch_arrays = []
                for sym in ready_symbols:
                    arr = self._buffers[sym].get_array()  # (seq_len, feat_dim)
                    # Use last snapshot for single-step inference (matches CPU bridge)
                    batch_arrays.append(arr[-1])          # (feat_dim,)

                batch_np = np.stack(batch_arrays, axis=0)  # (N, feat_dim)
                t_inf_start = time.perf_counter()

                with torch.no_grad():
                    t_in = torch.tensor(
                        batch_np, dtype=torch.float32, device=self._device
                    )
                    with torch.cuda.amp.autocast(enabled=self._use_amp):
                        logits = self.model(t_in)           # (N, 3)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()  # (N, 3)

                if self._device_str.startswith("cuda"):
                    torch.cuda.synchronize()

                latency_us = (time.perf_counter() - t_inf_start) * 1_000_000
                per_symbol_latency = latency_us / max(len(ready_symbols), 1)
                self._latency.record(per_symbol_latency)
                self._inference_count += len(ready_symbols)

                ts_ns = time.time_ns()
                for i, sym in enumerate(ready_symbols):
                    p = probs[i]
                    cls_idx = int(np.argmax(p))
                    signal = {
                        "type": "deeplob_signal",
                        "symbol": sym,
                        "direction": self.DIRECTION_MAP[cls_idx],
                        "confidence": float(p[cls_idx]),
                        "logits": p.tolist(),
                        "timestamp_ns": ts_ns,
                        "inference_latency_us": int(per_symbol_latency),
                        "model_version": self.config.model_version,
                    }
                    await self._publish(signal)

            # Sleep for remainder of interval
            elapsed = time.perf_counter() - loop_start
            sleep_s = max(0.0, interval_s - elapsed)
            await asyncio.sleep(sleep_s)

    async def _publish(self, signal: dict) -> None:
        """Serialise and publish a signal dict over ZMQ PUB."""
        if self._pub_socket is None:
            return
        try:
            payload = _pack(signal)
            await self._pub_socket.send(payload)
        except Exception as exc:
            logger.debug("gpu_inference_server: ZMQ send error: %s", exc)

    def get_stats(self) -> Dict:
        """Return server telemetry.

        Returns
        -------
        dict with keys:
            inference_count, avg_latency_us, p50/p95/p99 latency,
            symbols_tracked, gpu_utilisation_pct, model_loaded
        """
        lat = self._latency.stats()

        gpu_util: Optional[float] = None
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                idx = 0
                if self._device_str.startswith("cuda:"):
                    idx = int(self._device_str.split(":")[1])
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = float(util.gpu)
            except Exception:
                pass

        return {
            "inference_count": self._inference_count,
            "avg_latency_us": lat["avg_us"],
            "p50_latency_us": lat["p50_us"],
            "p95_latency_us": lat["p95_us"],
            "p99_latency_us": lat["p99_us"],
            "symbols_tracked": len(self._buffers),
            "gpu_utilisation_pct": gpu_util,
            "model_loaded": self._model_loaded,
            "device": self._device_str,
            "running": self._running,
        }
