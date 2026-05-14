"""
Pytest configuration and shared fixtures.

This repo contains many optional components (databases, Redis, GPU/quantum stacks).
These fixtures intentionally keep dependencies minimal so the test suite can run
in a lightweight environment.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def rng() -> np.random.Generator:
    """Deterministic RNG for tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def sample_tickers() -> Dict[str, Dict[str, Any]]:
    """Simple ticker snapshots used by unit tests."""
    return {
        "BTC/USD": {"last": 50000.0, "bid": 49990.0, "ask": 50010.0, "volume": 1000.0},
        "ETH/USD": {"last": 3000.0, "bid": 2995.0, "ask": 3005.0, "volume": 5000.0},
    }

