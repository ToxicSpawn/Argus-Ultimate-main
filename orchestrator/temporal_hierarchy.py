"""
orchestrator/temporal_hierarchy.py
===================================
Multi-timescale coordination — the key to true adaptation.

Coordinates decisions across 6 timescales:
  - Microsecond (μs): Queue position, latency arbitrage
  - Millisecond (ms): Order flow momentum, spread capture
  - Second (1s): Micro-regime, signal generation
  - Minute (1m): Strategy allocation, position management
  - Hour (1h): Portfolio construction, risk budgeting
  - Day (1d): Regime forecasting, meta-strategy

Slower layers constrain faster layers (top-down).
Faster layers inform slower layers (bottom-up).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Timescale(Enum):
    """Timescale levels."""
    MICROSECOND    = "microsecond"    # 100μs
    MILLISECOND    = "millisecond"    # 10ms
    SECOND         = "second"         # 1s
    MINUTE         = "minute"         # 60s
    HOUR           = "hour"           # 3600s
    DAY            = "day"            # 86400s


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimescaleLayer:
    """Configuration for a timescale layer."""
    timescale       : Timescale
    update_interval : float  # seconds
    agents          : List[str] = field(default_factory=list)
    constraints     : Dict[str, Any] = field(default_factory=dict)
    last_update     : float = 0.0
    update_count    : int = 0

    def should_update(self, now: float) -> bool:
        """Check if this layer should update now."""
        return now - self.last_update >= self.update_interval


@dataclass
class CrossTimescaleSignal:
    """A signal that propagates between timescales."""
    source_timescale: Timescale
    target_timescale: Timescale
    signal_type     : str  # "constraint" (top-down) or "inform" (bottom-up)
    data            : Dict[str, Any]
    priority        : int = 5
    timestamp       : float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Temporal Hierarchy
# ---------------------------------------------------------------------------

class TemporalHierarchy:
    """
    Coordinates decisions across timescales.

    Key principles:
      1. Top-down constraints: Slower layers set boundaries for faster layers
      2. Bottom-up signals: Faster layers inform slower layers of changes
      3. Bidirectional communication: Layers influence each other
      4. Emergent synchronization: Layers naturally align over time
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Define timescale layers (fastest to slowest)
        self.layers: Dict[Timescale, TimescaleLayer] = {
            Timescale.MICROSECOND: TimescaleLayer(
                timescale=Timescale.MICROSECOND,
                update_interval=0.0001,  # 100μs
            ),
            Timescale.MILLISECOND: TimescaleLayer(
                timescale=Timescale.MILLISECOND,
                update_interval=0.01,    # 10ms
            ),
            Timescale.SECOND: TimescaleLayer(
                timescale=Timescale.SECOND,
                update_interval=1.0,     # 1s
            ),
            Timescale.MINUTE: TimescaleLayer(
                timescale=Timescale.MINUTE,
                update_interval=60.0,    # 1m
            ),
            Timescale.HOUR: TimescaleLayer(
                timescale=Timescale.HOUR,
                update_interval=3600.0,  # 1h
            ),
            Timescale.DAY: TimescaleLayer(
                timescale=Timescale.DAY,
                update_interval=86400.0, # 1d
            ),
        }

        # Signal queues between layers
        self._top_down_signals: Deque[CrossTimescaleSignal] = deque(maxlen=1000)
        self._bottom_up_signals: Deque[CrossTimescaleSignal] = deque(maxlen=1000)

        # Layer state
        self._layer_state: Dict[Timescale, Dict[str, Any]] = {
            ts: {} for ts in Timescale
        }

        logger.info("TemporalHierarchy: initialised with %d layers", len(self.layers))

    # ------------------------------------------------------------------ Layer management

    def register_agent(self, timescale: Timescale, agent_name: str) -> None:
        """Register an agent to a timescale layer."""
        with self._lock:
            layer = self.layers.get(timescale)
            if layer and agent_name not in layer.agents:
                layer.agents.append(agent_name)
                logger.debug("TemporalHierarchy: registered '%s' to %s", agent_name, timescale.value)

    def unregister_agent(self, timescale: Timescale, agent_name: str) -> None:
        """Unregister an agent from a timescale layer."""
        with self._lock:
            layer = self.layers.get(timescale)
            if layer and agent_name in layer.agents:
                layer.agents.remove(agent_name)

    # ------------------------------------------------------------------ Signal propagation

    def send_top_down(self, signal: CrossTimescaleSignal) -> None:
        """Send a top-down constraint signal (slower → faster)."""
        with self._lock:
            self._top_down_signals.append(signal)
            # Update target layer state
            target_state = self._layer_state.get(signal.target_timescale, {})
            target_state[f"constraint_{signal.signal_type}"] = signal.data
            self._layer_state[signal.target_timescale] = target_state

    def send_bottom_up(self, signal: CrossTimescaleSignal) -> None:
        """Send a bottom-up inform signal (faster → slower)."""
        with self._lock:
            self._bottom_up_signals.append(signal)
            # Update target layer state
            target_state = self._layer_state.get(signal.target_timescale, {})
            target_state[f"inform_{signal.signal_type}"] = signal.data
            self._layer_state[signal.target_timescale] = target_state

    def get_constraints(self, timescale: Timescale) -> Dict[str, Any]:
        """Get current constraints for a timescale (from slower layers)."""
        with self._lock:
            state = self._layer_state.get(timescale, {})
            return {k: v for k, v in state.items() if k.startswith("constraint_")}

    def get_incoming_signals(self, timescale: Timescale) -> List[CrossTimescaleSignal]:
        """Get incoming signals for a timescale (from faster layers)."""
        with self._lock:
            return [
                s for s in self._bottom_up_signals
                if s.target_timescale == timescale
            ]

    # ------------------------------------------------------------------ Update cycle

    def tick(self, timescale: Timescale, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process one tick at a given timescale.

        Returns layer output if update was due, None otherwise.
        """
        now = time.time()
        layer = self.layers.get(timescale)

        if layer is None or not layer.should_update(now):
            return None

        # Gather constraints from slower layers
        constraints = self.get_constraints(timescale)

        # Gather signals from faster layers
        signals = self.get_incoming_signals(timescale)

        # Process
        output = self._process_layer(timescale, market_data, constraints, signals)

        # Update layer state
        layer.last_update = now
        layer.update_count += 1

        # Generate cross-timescale signals
        self._generate_cross_signals(timescale, output)

        return output

    def _process_layer(
        self,
        timescale: Timescale,
        market_data: Dict[str, Any],
        constraints: Dict[str, Any],
        signals: List[CrossTimescaleSignal],
    ) -> Dict[str, Any]:
        """Process a single layer update."""
        # This is a framework — actual processing is done by registered agents
        return {
            "timescale"  : timescale.value,
            "timestamp"  : time.time(),
            "constraints": constraints,
            "n_signals"  : len(signals),
            "market_data_keys": list(market_data.keys()) if market_data else [],
        }

    def _generate_cross_signals(self, source: Timescale, output: Dict[str, Any]) -> None:
        """Generate cross-timescale signals based on layer output."""
        if output is None:
            return

        timescales = list(Timescale)
        source_idx = timescales.index(source)

        # Bottom-up: inform slower layers of significant events
        if output.get("significant_event"):
            for ts in timescales[source_idx + 1:]:
                self.send_bottom_up(CrossTimescaleSignal(
                    source_timescale=source,
                    target_timescale=ts,
                    signal_type="significant_event",
                    data=output,
                    priority=output.get("priority", 5),
                ))

        # Top-down: faster layers respect slower layer constraints
        # (constraints are already applied in _process_layer)

    # ------------------------------------------------------------------ Layer state

    def set_layer_state(self, timescale: Timescale, state: Dict[str, Any]) -> None:
        """Set state for a layer."""
        with self._lock:
            self._layer_state[timescale] = state

    def get_layer_state(self, timescale: Timescale) -> Dict[str, Any]:
        """Get state for a layer."""
        with self._lock:
            return dict(self._layer_state.get(timescale, {}))

    # ------------------------------------------------------------------ Status

    def get_status(self) -> Dict[str, Any]:
        """Get hierarchy status."""
        with self._lock:
            return {
                ts.value: {
                    "update_interval" : layer.update_interval,
                    "agents"          : layer.agents,
                    "last_update"     : layer.last_update,
                    "update_count"    : layer.update_count,
                }
                for ts, layer in self.layers.items()
            }

    def get_layer_utilization(self) -> Dict[str, float]:
        """Get how busy each layer is."""
        with self._lock:
            now = time.time()
            utilization = {}
            for ts, layer in self.layers.items():
                if layer.last_update > 0:
                    time_since = now - layer.last_update
                    utilization[ts.value] = min(1.0, time_since / layer.update_interval)
                else:
                    utilization[ts.value] = 0.0
            return utilization
