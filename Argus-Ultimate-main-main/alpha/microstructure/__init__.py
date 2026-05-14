"""
alpha/microstructure/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public API for the microstructure package.

All key classes are importable directly from `alpha.microstructure`:

    from alpha.microstructure import (
        LiveSignalBus, LiveSignal,
        WSFeedAdapter,
        RegimeIntegration,
        LiveOFIStream, OFISignal,
        LiveVPINStream,
        MicropriceDriftSignal,
        LatencyTelemetry, LatencyStage,
    )
"""
from __future__ import annotations

from alpha.microstructure.live_ofi_stream import LiveOFIStream, OFISignal
from alpha.microstructure.live_signal_bus import LiveSignal, LiveSignalBus
from alpha.microstructure.live_vpin_stream import LiveVPINStream
from alpha.microstructure.microprice_drift import MicropriceDriftSignal
from alpha.microstructure.regime_integration import RegimeIntegration
from alpha.microstructure.ws_feed_adapter import WSFeedAdapter
from hft_engine.latency_telemetry import LatencyStage, LatencyTelemetry

__all__ = [
    "LiveSignalBus",
    "LiveSignal",
    "WSFeedAdapter",
    "RegimeIntegration",
    "LiveOFIStream",
    "OFISignal",
    "LiveVPINStream",
    "MicropriceDriftSignal",
    "LatencyTelemetry",
    "LatencyStage",
]
