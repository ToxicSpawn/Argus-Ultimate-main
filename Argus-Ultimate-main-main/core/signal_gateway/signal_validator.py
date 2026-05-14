"""SignalValidator — validates every inbound SignalEnvelope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.signal_gateway.signal_envelope import SignalEnvelope
from core.signal_gateway.gateway_config import GatewayConfig

VALID_DIRECTIONS = frozenset({"long", "short", "flat"})


@dataclass
class ValidationResult:
    """Result of a single envelope validation pass."""

    valid: bool
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.valid


class SignalValidator:
    """Validates SignalEnvelopes against gateway config rules.

    Checks performed (in order):
    1. Direction must be 'long', 'short', or 'flat'.
    2. Confidence must be in [0.0, 1.0].
    3. Source must be in enabled_sources (if configured).
    4. Envelope must not have expired (TTL check).
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    def validate(self, envelope: SignalEnvelope) -> ValidationResult:
        """Return ValidationResult for *envelope*."""
        if envelope.direction not in VALID_DIRECTIONS:
            return ValidationResult(
                False,
                f"invalid direction {envelope.direction!r}; "
                f"must be one of {sorted(VALID_DIRECTIONS)}",
            )

        if not (0.0 <= envelope.confidence <= 1.0):
            return ValidationResult(
                False,
                f"confidence {envelope.confidence} out of [0.0, 1.0]",
            )

        if self._config.enabled_sources is not None:
            if envelope.source not in self._config.enabled_sources:
                return ValidationResult(
                    False,
                    f"source {envelope.source.value!r} not in enabled_sources",
                )

        # Apply default TTL if envelope carries none.
        ttl = envelope.ttl_ms if envelope.ttl_ms > 0 else self._config.ttl_ms
        if ttl > 0 and envelope.is_expired():
            return ValidationResult(
                False,
                f"envelope expired (age={envelope.age_ms():.1f}ms, ttl={ttl}ms)",
            )

        return ValidationResult(True)
