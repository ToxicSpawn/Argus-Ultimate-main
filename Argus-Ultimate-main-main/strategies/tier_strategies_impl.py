"""
Tier strategies (restored, dependency-light).

The repo historically had many "tier" strategy modules (absolute/omega/etc) that
were marketing-heavy and often got stubbed during compile repair.

This module provides practical, import-safe tier strategies as ensembles of the
already-restored strategy library components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strategies.strategy_library_impl import (
    CandlestickPatternStrategy,
    HighFreqGridStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    QuantumBreakoutEliteStrategy,
    TrendFollowingStrategy,
)


@dataclass
class _Ensemble:
    members: List[Any]
    consensus: int = 1

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sigs: List[Dict[str, Any]] = []
        for s in self.members:
            try:
                r = s.analyze(market_data)
            except Exception:
                r = None
            if isinstance(r, dict) and r.get("action"):
                sigs.append(r)
        if not sigs:
            return None

        if self.consensus <= 1:
            return max(sigs, key=lambda d: float(d.get("confidence", 0.0) or 0.0))

        # Consensus: require N strategies to agree on action
        by_action: Dict[str, List[Dict[str, Any]]] = {}
        for d in sigs:
            a = str(d.get("action") or "").upper()
            by_action.setdefault(a, []).append(d)
        best_action = None
        best_count = 0
        for a, items in by_action.items():
            if len(items) > best_count:
                best_action = a
                best_count = len(items)
        if best_action is None or best_count < int(self.consensus):
            return None
        # Choose the highest-confidence signal among agreeing ones
        agreeing = by_action.get(best_action, [])
        return max(agreeing, key=lambda d: float(d.get("confidence", 0.0) or 0.0)) if agreeing else None


class AbsoluteTierStrategy:
    """Aggressive ensemble: momentum + breakout + trend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "absolute_tier"
        self._ens = _Ensemble(
            members=[
                MomentumStrategy(cfg.get("momentum") if isinstance(cfg.get("momentum"), dict) else {}),
                TrendFollowingStrategy(cfg.get("trend") if isinstance(cfg.get("trend"), dict) else {}),
                QuantumBreakoutEliteStrategy(cfg.get("breakout") if isinstance(cfg.get("breakout"), dict) else {}),
            ],
            consensus=int(cfg.get("consensus", 1) or 1),
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_absolute:{r.get('source','')}"
        return r


class AkashicTierStrategy:
    """Pattern + trend ensemble."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "akashic_tier"
        self._ens = _Ensemble(
            members=[
                CandlestickPatternStrategy(cfg.get("candles") if isinstance(cfg.get("candles"), dict) else {}),
                TrendFollowingStrategy(cfg.get("trend") if isinstance(cfg.get("trend"), dict) else {}),
            ],
            consensus=int(cfg.get("consensus", 1) or 1),
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_akashic:{r.get('source','')}"
        return r


class ApeironTierStrategy:
    """Mean reversion + grid ensemble."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "apeiron_tier"
        self._ens = _Ensemble(
            members=[
                MeanReversionStrategy(cfg.get("mean_reversion") if isinstance(cfg.get("mean_reversion"), dict) else {}),
                HighFreqGridStrategy(cfg.get("grid") if isinstance(cfg.get("grid"), dict) else {}),
            ],
            consensus=int(cfg.get("consensus", 1) or 1),
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_apeiron:{r.get('source','')}"
        return r


class ChronosTierStrategy:
    """Trend + breakout ensemble (time/continuation bias)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "chronos_tier"
        self._ens = _Ensemble(
            members=[
                TrendFollowingStrategy(cfg.get("trend") if isinstance(cfg.get("trend"), dict) else {}),
                QuantumBreakoutEliteStrategy(cfg.get("breakout") if isinstance(cfg.get("breakout"), dict) else {}),
            ],
            consensus=int(cfg.get("consensus", 1) or 1),
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_chronos:{r.get('source','')}"
        return r


class OmegaTierStrategy:
    """Balanced ensemble with consensus (more conservative)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "omega_tier"
        self._ens = _Ensemble(
            members=[
                MomentumStrategy(cfg.get("momentum") if isinstance(cfg.get("momentum"), dict) else {}),
                MeanReversionStrategy(cfg.get("mean_reversion") if isinstance(cfg.get("mean_reversion"), dict) else {}),
                TrendFollowingStrategy(cfg.get("trend") if isinstance(cfg.get("trend"), dict) else {}),
            ],
            consensus=int(cfg.get("consensus", 2) or 2),
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_omega:{r.get('source','')}"
        return r


class ParadoxTierStrategy:
    """Mean reversion + momentum ensemble."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "paradox_tier"
        self._ens = _Ensemble(
            members=[
                MeanReversionStrategy(cfg.get("mean_reversion") if isinstance(cfg.get("mean_reversion"), dict) else {}),
                MomentumStrategy(cfg.get("momentum") if isinstance(cfg.get("momentum"), dict) else {}),
            ],
            consensus=int(cfg.get("consensus", 1) or 1),
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_paradox:{r.get('source','')}"
        return r


class SingularityTierStrategy:
    """High-confidence ensemble with consensus."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "singularity_tier"
        self._ens = _Ensemble(
            members=[
                CandlestickPatternStrategy(cfg.get("candles") if isinstance(cfg.get("candles"), dict) else {}),
                MomentumStrategy(cfg.get("momentum") if isinstance(cfg.get("momentum"), dict) else {}),
                TrendFollowingStrategy(cfg.get("trend") if isinstance(cfg.get("trend"), dict) else {}),
                QuantumBreakoutEliteStrategy(cfg.get("breakout") if isinstance(cfg.get("breakout"), dict) else {}),
            ],
            consensus=int(cfg.get("consensus", 2) or 2),
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_singularity:{r.get('source','')}"
        return r


class SourceTierStrategy:
    """Minimal ensemble (fast)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "source_tier"
        self._ens = _Ensemble(
            members=[TrendFollowingStrategy(cfg.get("trend") if isinstance(cfg.get("trend"), dict) else {})],
            consensus=1,
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_source:{r.get('source','')}"
        return r


class ThanatosTierStrategy:
    """Breakout-biased ensemble."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "thanatos_tier"
        self._ens = _Ensemble(
            members=[QuantumBreakoutEliteStrategy(cfg.get("breakout") if isinstance(cfg.get("breakout"), dict) else {})],
            consensus=1,
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_thanatos:{r.get('source','')}"
        return r


class VoidTierStrategy:
    """Very conservative ensemble (requires consensus)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.name = "void_tier"
        self._ens = _Ensemble(
            members=[
                CandlestickPatternStrategy(cfg.get("candles") if isinstance(cfg.get("candles"), dict) else {}),
                MeanReversionStrategy(cfg.get("mean_reversion") if isinstance(cfg.get("mean_reversion"), dict) else {}),
                TrendFollowingStrategy(cfg.get("trend") if isinstance(cfg.get("trend"), dict) else {}),
            ],
            consensus=int(cfg.get("consensus", 2) or 2),
        )

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        r = self._ens.analyze(market_data)
        if r:
            r["source"] = f"tier_void:{r.get('source','')}"
        return r


__all__ = [
    "AbsoluteTierStrategy",
    "AkashicTierStrategy",
    "ApeironTierStrategy",
    "ChronosTierStrategy",
    "OmegaTierStrategy",
    "ParadoxTierStrategy",
    "SingularityTierStrategy",
    "SourceTierStrategy",
    "ThanatosTierStrategy",
    "VoidTierStrategy",
]

