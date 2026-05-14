"""
Helper utilities (import-safe).
"""

from __future__ import annotations


def format_currency(amount: float) -> str:
    """Format a number as USD-like currency."""
    try:
        return f"${float(amount):,.2f}"
    except Exception:
        return f"${amount}"


def calculate_percentage(value: float, total: float) -> float:
    """Calculate value/total * 100, guarding divide-by-zero."""
    try:
        total_f = float(total)
        return (float(value) / total_f * 100.0) if total_f > 0 else 0.0
    except Exception:
        return 0.0
