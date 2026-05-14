"""SignalSource enum — all registered signal producers in Argus."""

from __future__ import annotations

from enum import Enum
from typing import Dict


class SignalSource(str, Enum):
    """Enum of every signal source wired into the Signal Gateway.

    String values are used as Prometheus label values and JSON keys.
    """

    VOID_BREAKER = "void_breaker"
    RL_AGENT = "rl_agent"
    LLM_OVERLAY = "llm_overlay"
    DEEPLOB = "deeplob"
    CROSS_ASSET = "cross_asset"
    OFI_STREAM = "ofi_stream"
    VPIN_STREAM = "vpin_stream"
    FUNDING_ARB = "funding_arb"


# Priority weights used by ConsensusEngine.
# Higher weight = more influence on weighted vote.
# Weights must be positive; they are normalised internally.
DEFAULT_SOURCE_WEIGHTS: Dict[SignalSource, float] = {
    SignalSource.VOID_BREAKER: 2.0,   # Tier-5 multi-signal consensus engine
    SignalSource.RL_AGENT: 1.8,       # PPO execution agent
    SignalSource.DEEPLOB: 1.6,        # Microstructure ML
    SignalSource.LLM_OVERLAY: 1.2,    # GPT-4o sentiment
    SignalSource.OFI_STREAM: 1.4,     # Real-time order flow imbalance
    SignalSource.VPIN_STREAM: 1.3,    # Streaming VPIN toxicity
    SignalSource.CROSS_ASSET: 1.1,    # Regime / cross-asset alpha
    SignalSource.FUNDING_ARB: 1.0,    # Funding rate carry signal
}
