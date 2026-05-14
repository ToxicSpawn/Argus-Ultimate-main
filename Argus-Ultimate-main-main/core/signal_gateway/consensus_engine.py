"""ConsensusEngine — weighted vote aggregation across signal sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.signal_gateway.signal_envelope import SignalEnvelope
from core.signal_gateway.signal_source import SignalSource
from core.signal_gateway.gateway_config import GatewayConfig


@dataclass
class ConsensusResult:
    """Output of a consensus evaluation pass.

    Attributes
    ----------
    fired:                  True if consensus threshold was met.
    winning_direction:      'long', 'short', or 'flat' (None if not fired).
    aggregate_confidence:   Weighted-average confidence of winning direction.
    participating_sources:  Sources whose votes counted toward the winner.
    dissenting_sources:     Sources that voted for a different direction.
    vote_breakdown:         Raw weighted votes per direction.
    """

    fired: bool
    winning_direction: Optional[str] = None
    aggregate_confidence: float = 0.0
    participating_sources: List[SignalSource] = field(default_factory=list)
    dissenting_sources: List[SignalSource] = field(default_factory=list)
    vote_breakdown: Dict[str, float] = field(default_factory=dict)


class ConsensusEngine:
    """Evaluates a batch of SignalEnvelopes and returns a ConsensusResult.

    Algorithm
    ---------
    1. For each envelope, compute weighted vote = source_weight × confidence.
    2. Accumulate weighted votes per direction.
    3. Winning direction = direction with highest total weighted vote.
    4. Consensus fires if:
       a. total_vote_winning / total_vote_all >= consensus_threshold, AND
       b. number of distinct sources voting for winner >= min_sources.
    5. aggregate_confidence = weighted-average confidence of winning sources.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    def evaluate(self, envelopes: List[SignalEnvelope]) -> ConsensusResult:
        """Evaluate *envelopes* and return a ConsensusResult."""
        if not envelopes:
            return ConsensusResult(fired=False)

        weights = self._config.source_weights

        # Accumulate weighted votes per direction.
        direction_votes: Dict[str, float] = {}
        direction_sources: Dict[str, List[SignalEnvelope]] = {}

        for env in envelopes:
            w = weights.get(env.source, 1.0)
            vote = w * env.confidence
            direction_votes[env.direction] = (
                direction_votes.get(env.direction, 0.0) + vote
            )
            direction_sources.setdefault(env.direction, []).append(env)

        total_vote = sum(direction_votes.values())
        if total_vote == 0:
            return ConsensusResult(fired=False, vote_breakdown=direction_votes)

        winning_dir = max(direction_votes, key=lambda d: direction_votes[d])
        winning_vote = direction_votes[winning_dir]
        vote_fraction = winning_vote / total_vote

        winning_envs = direction_sources.get(winning_dir, [])
        n_winning_sources = len({e.source for e in winning_envs})

        # Quorum check.
        if (
            vote_fraction < self._config.consensus_threshold
            or n_winning_sources < self._config.min_sources
        ):
            return ConsensusResult(
                fired=False,
                winning_direction=winning_dir,
                aggregate_confidence=vote_fraction,
                vote_breakdown=direction_votes,
            )

        # Compute weighted-average confidence of participating sources.
        weight_sum = sum(
            weights.get(e.source, 1.0) for e in winning_envs
        )
        agg_conf = (
            sum(weights.get(e.source, 1.0) * e.confidence for e in winning_envs)
            / weight_sum
            if weight_sum > 0
            else 0.0
        )

        participating = list({e.source for e in winning_envs})
        dissenting = [
            e.source
            for d, envs in direction_sources.items()
            if d != winning_dir
            for e in envs
        ]

        return ConsensusResult(
            fired=True,
            winning_direction=winning_dir,
            aggregate_confidence=agg_conf,
            participating_sources=participating,
            dissenting_sources=list(set(dissenting)),
            vote_breakdown=direction_votes,
        )
