"""
Cross-Strategy Signal Consensus Filter — only trade when multiple
strategies agree on direction.

Dramatically reduces losing trades by requiring agreement between
independent strategy algorithms before acting.

Modes:
  - MAJORITY: 3+ strategies agree on direction
  - WEIGHTED: confidence-weighted vote exceeds threshold
  - UNANIMOUS: all active strategies agree (very selective)

Batch 1 addition: regime-specific strategy weight multipliers.
Pass `regime` (str or MarketRegime) to filter_signals() to activate.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Consensus modes
MODE_MAJORITY = "majority"
MODE_WEIGHTED = "weighted"
MODE_UNANIMOUS = "unanimous"

_VALID_MODES = {MODE_MAJORITY, MODE_WEIGHTED, MODE_UNANIMOUS}

# ---------------------------------------------------------------------------
# Regime-specific weight multipliers per strategy name keyword
# Keys match lowercase fragments of strategy_name field on signals.
# ---------------------------------------------------------------------------
REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "TREND_UP": {
        "momentum": 1.5,
        "breakout": 1.3,
        "mean_reversion": 0.5,
        "mean_rev": 0.5,
        "scalp": 0.8,
    },
    "TREND_DOWN": {
        "momentum": 1.3,
        "breakout": 0.6,
        "mean_reversion": 0.7,
        "mean_rev": 0.7,
        "scalp": 0.9,
    },
    "RANGE": {
        "momentum": 0.6,
        "breakout": 0.3,
        "mean_reversion": 1.5,
        "mean_rev": 1.5,
        "scalp": 1.2,
    },
    "HIGH_VOL": {
        "momentum": 0.8,
        "breakout": 1.2,
        "mean_reversion": 0.4,
        "mean_rev": 0.4,
        "scalp": 0.5,
    },
    "LOW_VOL": {
        "momentum": 1.1,
        "breakout": 0.7,
        "mean_reversion": 1.3,
        "mean_rev": 1.3,
        "scalp": 1.1,
    },
}


def _regime_key(regime: Any) -> Optional[str]:
    """Normalise a regime value to a REGIME_WEIGHTS key, or None."""
    if regime is None:
        return None
    s = str(regime).upper().replace(" ", "_")
    for k in REGIME_WEIGHTS:
        if k in s:
            return k
    return None


def _strategy_multiplier(strategy_name: str, regime_weights: Dict[str, float]) -> float:
    """Return the weight multiplier for a strategy given regime weights."""
    name_lower = strategy_name.lower()
    for fragment, mult in regime_weights.items():
        if fragment in name_lower:
            return mult
    return 1.0


@dataclass
class ConsensusStats:
    """Running statistics for the consensus filter."""
    signals_in: int = 0
    signals_out: int = 0
    symbols_evaluated: int = 0
    consensus_reached: int = 0
    consensus_missed: int = 0
    last_evaluation_ts: float = 0.0

    @property
    def filter_rate(self) -> float:
        if self.signals_in == 0:
            return 0.0
        return 1.0 - (self.signals_out / self.signals_in)


class SignalConsensus:
    """
    Only trade when multiple strategies agree on direction.

    Parameters
    ----------
    mode : str
        "majority" — 3+ strategies agree.
        "weighted" — confidence-weighted vote exceeds min_agreement.
        "unanimous" — all active strategies agree.
    min_agreement : float
        For weighted/majority mode: minimum weighted agreement threshold (0-1).
    min_strategies : int
        Minimum number of strategies that must have signals for a symbol
        before consensus is evaluated.
    """

    def __init__(
        self,
        mode: str = MODE_WEIGHTED,
        min_agreement: float = 0.6,
        min_strategies: int = 2,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got '{mode}'")
        self._mode = mode
        self._min_agreement = min_agreement
        self._min_strategies = max(1, min_strategies)
        self._stats = ConsensusStats()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def min_agreement(self) -> float:
        return self._min_agreement

    # ------------------------------------------------------------------
    # Main filtering
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signals: List[Dict[str, Any]],
        regime: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter signals by consensus, optionally applying regime-based
        strategy weight multipliers before voting.

        Parameters
        ----------
        signals : list of dict
            Each dict must have at least 'symbol' and 'action' keys.
            'confidence' and 'strategy_name' used for weighting.
        regime : str | MarketRegime | None
            Current market regime. When provided, per-strategy multipliers
            from REGIME_WEIGHTS are applied to confidence before voting.

        Returns
        -------
        list of dict
            Filtered signals — at most one per symbol.
        """
        self._stats.signals_in += len(signals)
        self._stats.last_evaluation_ts = time.time()

        if not signals:
            return []

        rkey = _regime_key(regime)
        regime_weights = REGIME_WEIGHTS.get(rkey, {}) if rkey else {}

        # Apply regime multipliers to a working copy
        working: List[Dict[str, Any]] = []
        for sig in signals:
            s = dict(sig)
            if regime_weights:
                strat = str(s.get("strategy_name", ""))
                mult = _strategy_multiplier(strat, regime_weights)
                s["confidence"] = min(1.0, float(s.get("confidence", 0.5)) * mult)
                s["_regime_mult"] = mult
            working.append(s)

        # Group by symbol
        by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sig in working:
            sym = sig.get("symbol", "UNKNOWN")
            by_symbol[sym].append(sig)

        result: List[Dict[str, Any]] = []

        for sym, sym_signals in by_symbol.items():
            self._stats.symbols_evaluated += 1

            if len(sym_signals) < self._min_strategies:
                result.extend(sym_signals)
                self._stats.signals_out += len(sym_signals)
                continue

            consensus_signal = self._evaluate_consensus(sym, sym_signals)
            if consensus_signal is not None:
                result.append(consensus_signal)
                self._stats.signals_out += 1
                self._stats.consensus_reached += 1
            else:
                self._stats.consensus_missed += 1
                logger.debug(
                    "SignalConsensus: no consensus for %s (%d signals, mode=%s, regime=%s)",
                    sym, len(sym_signals), self._mode, rkey,
                )

        return result

    def _evaluate_consensus(
        self,
        symbol: str,
        signals: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        buy_signals: List[Dict[str, Any]] = []
        sell_signals: List[Dict[str, Any]] = []

        for sig in signals:
            action = str(sig.get("action", "HOLD")).upper()
            if action == "BUY":
                buy_signals.append(sig)
            elif action == "SELL":
                sell_signals.append(sig)

        total_directional = len(buy_signals) + len(sell_signals)
        if total_directional == 0:
            return None

        buy_weight = sum(float(s.get("confidence", 0.5)) for s in buy_signals)
        sell_weight = sum(float(s.get("confidence", 0.5)) for s in sell_signals)
        total_weight = buy_weight + sell_weight

        if total_weight < 1e-10:
            return None

        buy_ratio = buy_weight / total_weight
        sell_ratio = sell_weight / total_weight

        if self._mode == MODE_UNANIMOUS:
            if len(buy_signals) == total_directional:
                winning_signals, winning_action, agreement = buy_signals, "BUY", 1.0
            elif len(sell_signals) == total_directional:
                winning_signals, winning_action, agreement = sell_signals, "SELL", 1.0
            else:
                return None
        elif self._mode == MODE_MAJORITY:
            if buy_ratio >= self._min_agreement:
                winning_signals, winning_action, agreement = buy_signals, "BUY", buy_ratio
            elif sell_ratio >= self._min_agreement:
                winning_signals, winning_action, agreement = sell_signals, "SELL", sell_ratio
            else:
                return None
        else:  # MODE_WEIGHTED
            if buy_ratio >= self._min_agreement:
                winning_signals, winning_action, agreement = buy_signals, "BUY", buy_ratio
            elif sell_ratio >= self._min_agreement:
                winning_signals, winning_action, agreement = sell_signals, "SELL", sell_ratio
            else:
                return None

        best = max(winning_signals, key=lambda s: float(s.get("confidence", 0.0)))
        boosted_confidence = min(1.0, float(best.get("confidence", 0.5)) * (1.0 + agreement * 0.3))

        consensus = dict(best)
        consensus["action"] = winning_action
        consensus["confidence"] = boosted_confidence
        consensus["consensus_agreement"] = round(agreement, 3)
        consensus["consensus_mode"] = self._mode
        consensus["consensus_signals_count"] = len(winning_signals)
        consensus["consensus_total_signals"] = len(signals)

        reasoning = consensus.get("reasoning", "")
        consensus["reasoning"] = (
            f"[Consensus {self._mode}: {len(winning_signals)}/{len(signals)} "
            f"agree={agreement:.0%}] {reasoning}"
        )
        return consensus

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_consensus_stats(self) -> Dict[str, Any]:
        s = self._stats
        return {
            "signals_in": s.signals_in,
            "signals_out": s.signals_out,
            "filter_rate": round(s.filter_rate, 3),
            "symbols_evaluated": s.symbols_evaluated,
            "consensus_reached": s.consensus_reached,
            "consensus_missed": s.consensus_missed,
            "mode": self._mode,
            "min_agreement": self._min_agreement,
        }

    def reset_stats(self) -> None:
        self._stats = ConsensusStats()
