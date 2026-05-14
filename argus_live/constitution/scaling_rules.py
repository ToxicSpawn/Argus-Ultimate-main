from __future__ import annotations


def assert_aum_safe(current_aum: float, max_safe_aum: float) -> None:
    """Raise RuntimeError if current AUM exceeds the safe ceiling."""
    if current_aum > max_safe_aum:
        raise RuntimeError(
            f"AUM {current_aum:.2f} exceeds max safe AUM {max_safe_aum:.2f}"
        )
