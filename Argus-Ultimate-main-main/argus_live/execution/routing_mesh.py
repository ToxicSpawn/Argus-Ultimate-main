from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VenueScore:
    venue_id: str
    score: float
    slippage_bps: float
    latency_ms: float
    fee_bps: float
    liquidity: float
    reason: str


@dataclass(frozen=True)
class RoutingMeshDecision:
    ranked_venues: tuple[VenueScore, ...]
    primary_venue: str
    backup_venue: str | None


def rank_venues(
    symbol: str,
    venue_inputs: list[VenueScore],
) -> RoutingMeshDecision:
    """Rank venues by composite score (desc), break ties by slippage,
    latency, fee, then descending liquidity.

    Returns a RoutingMeshDecision with the best venue as primary and
    the second-best as backup (or None if only one venue).
    Raises ValueError if the venue list is empty.
    """
    if not venue_inputs:
        raise ValueError("No venues to rank")

    ranked = sorted(
        venue_inputs,
        key=lambda v: (
            -v.score,
            v.slippage_bps,
            v.latency_ms,
            v.fee_bps,
            -v.liquidity,
        ),
    )
    return RoutingMeshDecision(
        ranked_venues=tuple(ranked),
        primary_venue=ranked[0].venue_id,
        backup_venue=ranked[1].venue_id if len(ranked) > 1 else None,
    )
