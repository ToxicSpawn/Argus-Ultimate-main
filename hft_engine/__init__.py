"""Argus HFT Engine package — canonical HFT implementation.

Consolidates hft/ and hft_engine/ into one authoritative package.
The hft/ directory is deprecated — all imports should use hft_engine/.

Pinnacle HFT components (v5.0.0):
  - HFTScalpingEngine        — OBI/VPIN/CVD-driven scalping signals
  - OrderBookProcessor       — L2/L3 book with microprice, Kyle's lambda
  - OrderBookSignals         — alpha signals from book microstructure
  - LatencyTelemetry         — tick-to-trade ns telemetry (singleton)
  - HotPathProfiler          — context-manager hot-path profiler
  - JitterMonitor            — WebSocket tick arrival jitter
  - RustLOBBridge            — zero-copy bridge to Rust LOB worker
"""
from hft_engine.hft_scalping_engine import HFTScalpingEngine
from hft_engine.advanced_realtime_hft_infrastructure import AdvancedRealtimeHFTInfrastructure
from hft_engine.order_book_processor import (
    L2OrderBook,
    L3OrderBook,
    OrderBookSignals,
    PriceLevel,
)
from hft_engine.latency_telemetry import (
    LatencyTelemetry,
    LatencyStage,
    HotPathProfiler,
    JitterMonitor,
    generate_report,
)
from hft_engine.rust_lob_bridge import RustLOBBridge

__all__ = [
    # Core scalping
    "HFTScalpingEngine",
    "AdvancedRealtimeHFTInfrastructure",
    # Order book
    "L2OrderBook",
    "L3OrderBook",
    "OrderBookSignals",
    "PriceLevel",
    # Latency
    "LatencyTelemetry",
    "LatencyStage",
    "HotPathProfiler",
    "JitterMonitor",
    "generate_report",
    # Rust bridge
    "RustLOBBridge",
]
