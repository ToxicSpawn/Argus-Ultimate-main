"""
Adaptive components for ARGUS Unified.

These modules are dependency-light and designed to be safe to import even in
minimal environments. They implement:
- market regime detection (trend vs range vs high-vol)
- online tuning / weighting of strategy modes based on realized PnL
"""

