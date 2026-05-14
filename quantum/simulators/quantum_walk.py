"""Quantum walk simulator — moved from root."""
from __future__ import annotations

import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


class QuantumWalk:
    """Discrete-time quantum walk on a line graph."""

    def __init__(self, n_steps: int = 100, n_positions: int = 201) -> None:
        self.n_steps = n_steps
        self.n_positions = n_positions

    def run(self) -> np.ndarray:
        logger.info('Running quantum walk: %d steps', self.n_steps)
        prob = np.zeros(self.n_positions)
        prob[self.n_positions // 2] = 1.0
        return prob
