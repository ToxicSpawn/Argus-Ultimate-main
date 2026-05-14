"""
Theory of Mind — trader archetype profiling.

Clusters market participants by their observed microstructure footprint
(spreads, cancel rates, fill patterns) into archetypes:
    - market_maker
    - whale
    - retail
    - hft
    - arbitrageur

Per Plan-agent review, retail_sentiment and adversarial_opponent modules
were DROPPED; whale_tracker already exists in ``data/onchain/``.
"""

from .trader_profiler import (
    TraderProfiler,
    TraderArchetype,
    ProfilerSnapshot,
    OrderFootprint,
)

__all__ = [
    "TraderProfiler",
    "TraderArchetype",
    "ProfilerSnapshot",
    "OrderFootprint",
]
