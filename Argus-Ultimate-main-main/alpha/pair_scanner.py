"""
alpha/pair_scanner.py
~~~~~~~~~~~~~~~~~~~~~
Compatibility shim — the canonical pair scanner is pair_scanner_extended.py.

This module re-exports everything from pair_scanner_extended so that all
existing `from alpha.pair_scanner import PairScanner` calls transparently
get the full extended implementation (with cointegration, VECM, and
extended filtering) without any code changes at the call site.

The extended scanner is a strict superset of the original:
  pair_scanner.py          — basic OLS-based pair scoring
  pair_scanner_extended.py — + cointegration tests, VECM spread model,
                               extended volatility/liquidity filters,
                               async batch scanning
"""
from __future__ import annotations

from alpha.pair_scanner_extended import *  # noqa: F401, F403

try:
    from alpha.pair_scanner_extended import PairScannerExtended as PairScanner  # type: ignore[attr-defined]
except ImportError:
    pass

try:
    from alpha.pair_scanner_extended import PairScannerExtended  # noqa: F401
except ImportError:
    pass
