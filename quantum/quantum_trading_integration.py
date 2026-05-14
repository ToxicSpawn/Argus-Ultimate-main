"""Retired quantum trading integration shim.

The old module name is kept only for import compatibility. Trading code should
consume advisory data from ``quantum.get_quantum_facade()`` and route any output
through normal risk gates.
"""

from __future__ import annotations

from typing import Any

from .retired import retired_placeholder


def __getattr__(name: str) -> Any:
    return retired_placeholder(f"quantum.quantum_trading_integration.{name}")


__all__: list[str] = []
