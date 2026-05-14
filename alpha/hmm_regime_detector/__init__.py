"""
alpha/hmm_regime_detector/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HMM-based regime detection package.

Exports:
- RegimeState: Enum of HMM regime states
- RegimeResult: Result dataclass
- HMMRegimeDetector: HMM-based regime detector
- GaussianHMM: Core HMM implementation
"""
from __future__ import annotations

from alpha.hmm_regime_detector.regime_detector import (
    RegimeState,
    RegimeResult,
    GaussianHMM,
    HMMRegimeDetector,
)

__all__ = ["RegimeState", "RegimeResult", "GaussianHMM", "HMMRegimeDetector"]
