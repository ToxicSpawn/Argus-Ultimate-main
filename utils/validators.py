"""
Validation utilities (import-safe).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a minimal config shape."""
    errors: List[str] = []
    required = ["starting_capital", "exchanges"]
    for key in required:
        if key not in config:
            errors.append(f"Missing key: {key}")
    return (len(errors) == 0), errors
