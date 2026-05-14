"""Push 70 — ReconnectPolicy: configurable backoff + circuit breaker.

Backoff strategies:
  linear:      delay = base_delay * attempt
  exponential: delay = base_delay * (factor ** attempt)
  fibonacci:   delay follows Fibonacci sequence

Circuit breaker: halts reconnect after max_consecutive_failures
consecutive failures. Requires manual reset().
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReconnectPolicy:
    """Configurable reconnect backoff policy.

    Args:
        strategy:              "linear" | "exponential" | "fibonacci"
        base_delay_secs:       Base delay for first retry
        max_delay_secs:        Cap on delay
        factor:                Multiplier for exponential strategy
        max_retries:           Max attempts before None returned (None = infinite)
        max_consecutive_failures: Circuit breaker threshold
        jitter_fraction:       Random fraction added to delay [0, 1)
    """
    strategy: str = "exponential"
    base_delay_secs: float = 1.0
    max_delay_secs: float = 60.0
    factor: float = 2.0
    max_retries: Optional[int] = 8
    max_consecutive_failures: int = 10
    jitter_fraction: float = 0.1

    # Runtime state
    _attempt: int = field(default=0, repr=False)
    _consecutive_failures: int = field(default=0, repr=False)
    _fib_a: int = field(default=1, repr=False)
    _fib_b: int = field(default=1, repr=False)
    _tripped: bool = field(default=False, repr=False)   # circuit breaker

    def next_delay(self) -> Optional[float]:
        """Return next delay in seconds, or None if should stop."""
        if self._tripped:
            return None
        if self.max_retries is not None and self._attempt >= self.max_retries:
            return None

        self._attempt += 1
        self._consecutive_failures += 1

        if self._consecutive_failures >= self.max_consecutive_failures:
            self._tripped = True
            return None

        if self.strategy == "linear":
            delay = self.base_delay_secs * self._attempt
        elif self.strategy == "exponential":
            delay = self.base_delay_secs * (self.factor ** (self._attempt - 1))
        elif self.strategy == "fibonacci":
            delay = float(self._fib_a) * self.base_delay_secs
            self._fib_a, self._fib_b = self._fib_b, self._fib_a + self._fib_b
        else:
            delay = self.base_delay_secs

        delay = min(delay, self.max_delay_secs)
        if self.jitter_fraction > 0:
            delay += delay * self.jitter_fraction * random.random()
        return delay

    def reset(self) -> None:
        """Reset after successful connection."""
        self._attempt = 0
        self._consecutive_failures = 0
        self._fib_a = 1
        self._fib_b = 1
        # Note: does NOT reset circuit breaker (_tripped). Use reset_circuit().

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker."""
        self._tripped = False
        self.reset()

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def is_tripped(self) -> bool:
        return self._tripped
