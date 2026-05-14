"""Helpers for retired quantum surfaces.

ARGUS keeps a few import shims so older optional scripts fail with a clear
message instead of accidentally running unsupported hype-era implementations.
The working quantum API is :func:`quantum.get_quantum_facade`.
"""

from __future__ import annotations

from typing import Any


RETIREMENT_MESSAGE = (
    "This legacy quantum surface has been retired. Use quantum.get_quantum_facade() "
    "for the supported QAOA, Sobol QMC, MLQAE, and status APIs."
)


class RetiredQuantumFeature:
    """Callable placeholder that raises an explicit retirement error on use."""

    def __init__(self, name: str = "legacy quantum feature") -> None:
        self.name = name

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{self.name}: {RETIREMENT_MESSAGE}")

    def __getattr__(self, item: str) -> "RetiredQuantumFeature":
        return RetiredQuantumFeature(f"{self.name}.{item}")

    def __repr__(self) -> str:
        return f"<RetiredQuantumFeature {self.name!r}>"


def retired_error(name: str) -> RuntimeError:
    """Build a consistent error for retired modules/classes/functions."""
    return RuntimeError(f"{name}: {RETIREMENT_MESSAGE}")


def retired_placeholder(name: str) -> RetiredQuantumFeature:
    """Return a callable placeholder for compatibility imports."""
    return RetiredQuantumFeature(name)


__all__ = [
    "RETIREMENT_MESSAGE",
    "RetiredQuantumFeature",
    "retired_error",
    "retired_placeholder",
]
