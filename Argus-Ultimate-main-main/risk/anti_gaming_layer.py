"""Anti-gaming / pattern obfuscation layer for order execution.

Adversarial market participants (front-runners, sandwich bots, other algos)
can detect predictable execution patterns.  This module introduces controlled
randomisation to order size, timing, venue selection, and splitting — making
the strategy's footprint harder to fingerprint.

All randomness uses the ``secrets`` module for cryptographic quality.
"""

from __future__ import annotations

import logging
import math
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExecutionMask:
    """A bundle of randomised execution parameters applied to a single order."""

    size_multiplier: float  # e.g. 0.92 means 92 % of base size
    delay_s: float  # seconds to wait before submitting
    venue: str  # selected execution venue
    split_count: int  # number of child orders (1 = no split)
    use_iceberg: bool  # whether to use iceberg / hidden qty


# ---------------------------------------------------------------------------
# AntiGamingLayer
# ---------------------------------------------------------------------------

class AntiGamingLayer:
    """Apply controlled randomisation to execution parameters.

    All public methods are stateless — they can be called from any context
    without side-effects.

    Parameters
    ----------
    seed_entropy : int | None
        Not used for the actual RNG (which is always ``secrets``), but can
        be logged for audit traceability.
    iceberg_probability : float
        Base probability that ``use_iceberg`` is True in an ExecutionMask.
        Default 0.3 (30 %).
    """

    def __init__(
        self,
        seed_entropy: Optional[int] = None,
        iceberg_probability: float = 0.3,
    ) -> None:
        self._iceberg_prob = max(0.0, min(1.0, iceberg_probability))
        if seed_entropy is not None:
            logger.debug("AntiGamingLayer initialised (entropy tag=%d)", seed_entropy)

    # ------------------------------------------------------------------
    # Size randomisation
    # ------------------------------------------------------------------

    def randomize_order_size(
        self,
        base_size: float,
        max_deviation_pct: float = 10.0,
    ) -> float:
        """Return a slightly randomised order size.

        The returned value is within ``[base_size * (1 - dev), base_size * (1 + dev)]``
        where ``dev = max_deviation_pct / 100``.

        Uses cryptographically secure randomness.

        Parameters
        ----------
        base_size : float
            The intended order size (units or USD).
        max_deviation_pct : float
            Maximum percentage deviation in either direction.
            Default 10 %.

        Returns
        -------
        float
            Randomised size, always > 0.
        """
        if base_size <= 0:
            return base_size

        dev = abs(max_deviation_pct) / 100.0
        # Uniform random in [-dev, +dev]
        rand_factor = self._secure_uniform(-dev, dev)
        result = base_size * (1.0 + rand_factor)
        return max(result, base_size * 0.01)  # floor at 1% to avoid zero

    # ------------------------------------------------------------------
    # Timing randomisation
    # ------------------------------------------------------------------

    def randomize_timing(
        self,
        base_delay_s: float,
        max_jitter_s: float = 5.0,
    ) -> float:
        """Return a jittered delay in seconds.

        The jitter is added on top of ``base_delay_s`` and is uniformly
        distributed in ``[0, max_jitter_s]``.

        Parameters
        ----------
        base_delay_s : float
            Minimum delay before order submission.
        max_jitter_s : float
            Maximum additional random delay.  Default 5 s.

        Returns
        -------
        float
            Total delay (always >= 0).
        """
        jitter = self._secure_uniform(0.0, abs(max_jitter_s))
        return max(0.0, base_delay_s + jitter)

    # ------------------------------------------------------------------
    # Venue selection
    # ------------------------------------------------------------------

    def get_random_venue(
        self,
        preferred_venues: List[str],
        weights: Optional[List[float]] = None,
    ) -> str:
        """Select a venue using weighted random sampling.

        Parameters
        ----------
        preferred_venues : list[str]
            Available venue identifiers (e.g. ``["kraken", "coinbase"]``).
        weights : list[float] | None
            Relative weights for each venue.  If ``None``, uniform selection.

        Returns
        -------
        str
            Selected venue name.

        Raises
        ------
        ValueError
            If *preferred_venues* is empty.
        """
        if not preferred_venues:
            raise ValueError("preferred_venues must not be empty")

        if len(preferred_venues) == 1:
            return preferred_venues[0]

        if weights is None:
            weights = [1.0] * len(preferred_venues)

        if len(weights) != len(preferred_venues):
            raise ValueError("weights length must match preferred_venues length")

        # Weighted selection using cumulative distribution
        total = sum(weights)
        if total <= 0:
            # Fall back to uniform if all weights are zero/negative
            idx = secrets.randbelow(len(preferred_venues))
            return preferred_venues[idx]

        threshold = self._secure_uniform(0.0, total)
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if threshold <= cumulative:
                return preferred_venues[i]

        # Shouldn't reach here, but just in case
        return preferred_venues[-1]

    # ------------------------------------------------------------------
    # Order splitting
    # ------------------------------------------------------------------

    def should_split_order(
        self,
        size_usd: float,
        threshold_usd: float = 500.0,
    ) -> Tuple[bool, int]:
        """Decide whether to split an order and into how many pieces.

        Splitting logic:
        * Below threshold: no split (1 order).
        * 1-5x threshold: 2-3 splits.
        * 5-20x threshold: 3-5 splits.
        * Above 20x: 5-8 splits.

        A small random element is added so the split count is not perfectly
        predictable from order size alone.

        Parameters
        ----------
        size_usd : float
            Total order size in USD.
        threshold_usd : float
            Below this value, no splitting occurs.

        Returns
        -------
        tuple[bool, int]
            ``(should_split, num_splits)`` where ``num_splits >= 1``.
        """
        if size_usd < threshold_usd:
            return False, 1

        ratio = size_usd / threshold_usd

        if ratio < 5.0:
            lo, hi = 2, 3
        elif ratio < 20.0:
            lo, hi = 3, 5
        else:
            lo, hi = 5, 8

        num_splits = lo + secrets.randbelow(hi - lo + 1)
        return True, num_splits

    # ------------------------------------------------------------------
    # Full execution mask
    # ------------------------------------------------------------------

    def get_execution_mask(
        self,
        base_size: float = 100.0,
        base_delay_s: float = 0.0,
        preferred_venues: Optional[List[str]] = None,
        venue_weights: Optional[List[float]] = None,
        size_usd: Optional[float] = None,
        split_threshold_usd: float = 500.0,
        max_size_deviation_pct: float = 10.0,
        max_jitter_s: float = 5.0,
    ) -> ExecutionMask:
        """Generate a complete randomised execution mask.

        Combines all randomisation methods into a single dataclass for
        convenient consumption by the execution layer.

        Parameters
        ----------
        base_size : float
            Intended order size.
        base_delay_s : float
            Base delay before submission.
        preferred_venues : list[str] | None
            Venue choices.  Default ``["primary"]``.
        venue_weights : list[float] | None
            Weights for venue selection.
        size_usd : float | None
            USD value for split decision.  Defaults to *base_size*.
        split_threshold_usd : float
            Split threshold.
        max_size_deviation_pct : float
            Max size jitter percentage.
        max_jitter_s : float
            Max timing jitter.

        Returns
        -------
        ExecutionMask
        """
        if preferred_venues is None:
            preferred_venues = ["primary"]

        randomised_size = self.randomize_order_size(base_size, max_size_deviation_pct)
        size_multiplier = randomised_size / base_size if base_size > 0 else 1.0

        delay = self.randomize_timing(base_delay_s, max_jitter_s)
        venue = self.get_random_venue(preferred_venues, venue_weights)

        effective_usd = size_usd if size_usd is not None else base_size
        _, split_count = self.should_split_order(effective_usd, split_threshold_usd)

        use_iceberg = self._secure_uniform(0.0, 1.0) < self._iceberg_prob

        mask = ExecutionMask(
            size_multiplier=round(size_multiplier, 6),
            delay_s=round(delay, 3),
            venue=venue,
            split_count=split_count,
            use_iceberg=use_iceberg,
        )
        logger.debug(
            "ExecutionMask: size_mult=%.4f delay=%.2fs venue=%s splits=%d iceberg=%s",
            mask.size_multiplier, mask.delay_s, mask.venue, mask.split_count, mask.use_iceberg,
        )
        return mask

    # ------------------------------------------------------------------
    # Secure RNG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _secure_uniform(lo: float, hi: float) -> float:
        """Cryptographically secure uniform random float in [lo, hi].

        Uses ``secrets.randbelow`` with 2**53 resolution (same as IEEE 754
        double-precision mantissa) to avoid modular bias.
        """
        if lo >= hi:
            return lo
        resolution = 2**53
        rand_int = secrets.randbelow(resolution)
        fraction = rand_int / resolution
        return lo + fraction * (hi - lo)
