"""Signal Gateway package — Push 35.

Unified broker between all signal sources and the order execution layer.
Normalises, validates, deduplicates, and consensus-gates every inbound
signal before it reaches the executor.
"""

from core.signal_gateway.signal_envelope import SignalEnvelope
from core.signal_gateway.signal_source import SignalSource
from core.signal_gateway.gateway_config import GatewayConfig
from core.signal_gateway.signal_validator import SignalValidator, ValidationResult
from core.signal_gateway.signal_deduplicator import SignalDeduplicator
from core.signal_gateway.consensus_engine import ConsensusEngine, ConsensusResult
from core.signal_gateway.signal_gateway import SignalGateway
from core.signal_gateway.gateway_wiring import wire_gateway_to_argus_bot

__all__ = [
    "SignalEnvelope",
    "SignalSource",
    "GatewayConfig",
    "SignalValidator",
    "ValidationResult",
    "SignalDeduplicator",
    "ConsensusEngine",
    "ConsensusResult",
    "SignalGateway",
    "wire_gateway_to_argus_bot",
]
