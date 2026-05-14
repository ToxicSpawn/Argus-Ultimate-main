"""
Post-Trade Shapley Attribution Engine.

After every closed trade, computes approximate Shapley values for each
signal source's contribution to the outcome.  Uses Monte Carlo permutation
sampling over the coalition of active signals.

Shapley value for signal i = E_π [ v(S∪{i}) - v(S) ]
where S is the set of signals before i in random permutation π, and
v(S) is the "predicted outcome" using only signals in S.

The value function v(S) approximates: weighted_average(signal_values[S]).
The outcome is the realised P&L (binary: win or loss, or the P&L magnitude).

Attribution weights are then fed back to EnsembleSignalHub to adjust
forward-looking signal weights.

Usage::

    engine = TradeAttributionEngine(n_samples=200)

    # At trade entry — snapshot current signal values
    engine.record_entry(
        trade_id="t-001",
        signal_snapshot={"fear_greed": -0.3, "llm": 0.6, "whale": 0.2, ...},
        strategy="momentum",
        regime="TREND_UP",
    )

    # At trade exit — compute Shapley and get weight adjustments
    adjustments = engine.record_exit(
        trade_id="t-001",
        pnl=12.5,
    )
    # adjustments: {"fear_greed": 0.05, "llm": -0.02, ...}
    # Feed to ensemble_hub.update_source_weights(adjustments)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeEntry:
    """Stores signal state at trade entry."""
    trade_id: str
    signal_snapshot: Dict[str, float]   # signal_name → value in [-1, 1]
    strategy: str
    regime: str
    entry_ts: float = field(default_factory=time.time)


@dataclass
class Attribution:
    """Shapley attribution result for a single closed trade."""
    trade_id: str
    strategy: str
    regime: str
    pnl: float
    shapley_values: Dict[str, float]    # signal_name → Shapley value
    signal_snapshot: Dict[str, float]
    closed_ts: float = field(default_factory=time.time)

    @property
    def top_contributor(self) -> Optional[str]:
        if not self.shapley_values:
            return None
        return max(self.shapley_values, key=lambda k: abs(self.shapley_values[k]))

    def summary(self) -> str:
        lines = [
            f"Trade {self.trade_id} ({self.strategy} | {self.regime}): P&L={self.pnl:+.4f}",
        ]
        for sig, sv in sorted(self.shapley_values.items(), key=lambda x: -abs(x[1])):
            bar = "+" if sv >= 0 else "-"
            lines.append(f"  {bar} {sig}: {sv:+.4f}")
        return "\n".join(lines)


class TradeAttributionEngine:
    """
    Approximate Shapley value attribution engine for closed trades.

    Maintains a pending-entry dict of open trades keyed by trade_id.
    On exit, computes Shapley values via Monte Carlo permutation sampling
    and returns per-signal weight adjustment recommendations.

    Parameters
    ----------
    n_samples : int
        Number of random permutations for MC Shapley estimation.
        200 is accurate enough for <= 10 signals. Use 500+ for 15+.
    ema_alpha : float
        EMA decay for rolling per-signal Shapley accumulator.
        Used to produce smoothed weight_adjustments over time.
    history_len : int
        Maximum closed-trade attributions to retain for analysis.
    """

    def __init__(
        self,
        n_samples: int = 200,
        ema_alpha: float = 0.1,
        history_len: int = 500,
    ) -> None:
        self._n_samples = int(n_samples)
        self._ema_alpha = float(ema_alpha)
        self._history_len = int(history_len)

        self._pending: Dict[str, TradeEntry] = {}
        self._history: Deque[Attribution] = deque(maxlen=history_len)

        # Rolling EMA of per-signal mean Shapley value (running estimate)
        self._ema_shapley: Dict[str, float] = defaultdict(float)
        self._signal_trade_count: Dict[str, int] = defaultdict(int)

        # Per-signal cumulative P&L attribution
        self._cumulative_attribution: Dict[str, float] = defaultdict(float)

    # ── Entry / Exit recording ─────────────────────────────────────────────

    def record_entry(
        self,
        trade_id: str,
        signal_snapshot: Dict[str, float],
        strategy: str = "unknown",
        regime: str = "UNKNOWN",
    ) -> None:
        """
        Snapshot signal values at the moment a trade is entered.

        Parameters
        ----------
        trade_id : str
            Unique identifier for this trade (e.g. order_id).
        signal_snapshot : dict
            Current signal values, each in [-1, 1].
            Keys should match the EnsembleSignalHub source names.
        strategy : str
            Strategy that generated the signal.
        regime : str
            Current market regime label.
        """
        if not signal_snapshot:
            return
        self._pending[trade_id] = TradeEntry(
            trade_id=trade_id,
            signal_snapshot={k: float(v) for k, v in signal_snapshot.items()},
            strategy=strategy,
            regime=regime,
        )
        logger.debug("TradeAttribution: entry recorded trade_id=%s signals=%d", trade_id, len(signal_snapshot))

    def record_exit(
        self,
        trade_id: str,
        pnl: float,
    ) -> Dict[str, float]:
        """
        Compute Shapley attribution for a closed trade.

        Parameters
        ----------
        trade_id : str
            Must match a previously recorded entry.
        pnl : float
            Realised P&L of the trade.

        Returns
        -------
        dict[str, float]
            Per-signal weight adjustment recommendations.
            Positive = increase weight, negative = reduce weight.
            Magnitudes are small (~0.01-0.05) suitable for EMA blending.
            Returns empty dict if trade_id not found in pending.
        """
        entry = self._pending.pop(trade_id, None)
        if entry is None:
            logger.debug("TradeAttribution: no pending entry for trade_id=%s", trade_id)
            return {}

        shapley = self._compute_shapley(entry.signal_snapshot, pnl)

        attribution = Attribution(
            trade_id=trade_id,
            strategy=entry.strategy,
            regime=entry.regime,
            pnl=float(pnl),
            shapley_values=shapley,
            signal_snapshot=entry.signal_snapshot,
        )
        self._history.append(attribution)

        # Update EMA accumulators
        for sig, sv in shapley.items():
            self._signal_trade_count[sig] += 1
            self._cumulative_attribution[sig] += sv
            prev = self._ema_shapley[sig]
            self._ema_shapley[sig] = (1.0 - self._ema_alpha) * prev + self._ema_alpha * sv

        # Build weight adjustment recommendations
        adjustments = self._compute_weight_adjustments(shapley, pnl)
        logger.debug(
            "TradeAttribution: exit trade_id=%s pnl=%+.4f top=%s",
            trade_id, pnl, attribution.top_contributor,
        )
        return adjustments

    # ── Weight adjustments ────────────────────────────────────────────────

    def _compute_weight_adjustments(
        self,
        shapley: Dict[str, float],
        pnl: float,
    ) -> Dict[str, float]:
        """
        Translate Shapley values into small weight adjustment signals.

        Logic:
        - A signal with a large positive Shapley value on a winning trade
          → increase its weight.
        - A signal with a large positive Shapley value on a losing trade
          → decrease its weight (it confidently pushed in the wrong direction
            and was a big contributor to the loss).
        - The adjustment magnitude is scaled by |pnl| and Shapley magnitude,
          capped at ±0.05 per trade to prevent runaway updates.
        """
        if not shapley:
            return {}

        pnl_sign = 1.0 if pnl >= 0 else -1.0
        scale = min(1.0, abs(pnl) / 50.0)  # normalise: $50 pnl → full scale
        adjustments: Dict[str, float] = {}

        for sig, sv in shapley.items():
            # If signal contribution aligned with outcome → positive adjustment
            # If signal contribution opposed outcome → negative adjustment
            raw = pnl_sign * sv * scale
            adjustments[sig] = float(np.clip(raw, -0.05, 0.05))

        return adjustments

    def get_weight_adjustments(self) -> Dict[str, float]:
        """
        Return smoothed (EMA) per-signal weight adjustments based on
        all attributed trades so far. Suitable for periodic ensemble reweighting.
        """
        if not self._ema_shapley:
            return {}
        # Normalise EMA Shapley values to [-0.10, +0.10] adjustment range
        vals = np.array(list(self._ema_shapley.values()), dtype=float)
        if np.all(vals == 0):
            return {sig: 0.0 for sig in self._ema_shapley}
        max_abs = max(np.abs(vals).max(), 1e-9)
        return {
            sig: float(np.clip(v / max_abs * 0.10, -0.10, 0.10))
            for sig, v in self._ema_shapley.items()
        }

    # ── Analytics ─────────────────────────────────────────────────────────

    def top_signals(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Return the n signals with highest cumulative attribution magnitude.
        """
        rows = [
            {
                "signal": sig,
                "cumulative_attribution": round(self._cumulative_attribution[sig], 4),
                "ema_shapley": round(self._ema_shapley[sig], 6),
                "trade_count": self._signal_trade_count[sig],
            }
            for sig in self._cumulative_attribution
        ]
        rows.sort(key=lambda r: abs(r["cumulative_attribution"]), reverse=True)
        return rows[:n]

    def per_regime_attribution(self) -> Dict[str, Dict[str, float]]:
        """
        Return mean Shapley value per signal per regime across all history.
        Useful for understanding which signals work in which regimes.
        """
        regime_sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        regime_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for a in self._history:
            for sig, sv in a.shapley_values.items():
                regime_sums[a.regime][sig] += sv
                regime_counts[a.regime][sig] += 1

        result: Dict[str, Dict[str, float]] = {}
        for regime, sigs in regime_sums.items():
            result[regime] = {
                sig: round(total / regime_counts[regime][sig], 6)
                for sig, total in sigs.items()
            }
        return result

    def snapshot(self) -> Dict[str, Any]:
        return {
            "pending_trades": len(self._pending),
            "attributed_trades": len(self._history),
            "top_signals": self.top_signals(5),
            "ema_shapley": {k: round(v, 6) for k, v in self._ema_shapley.items()},
        }

    # ── Shapley computation ───────────────────────────────────────────────

    def _compute_shapley(
        self,
        signal_snapshot: Dict[str, float],
        pnl: float,
    ) -> Dict[str, float]:
        """
        Approximate Shapley values using Monte Carlo permutation sampling.

        Value function v(S): for a coalition S of signals, the predicted
        directional outcome is the weighted average of signal_values in S
        (equal weights — attribution is about contribution, not magnitude).
        The outcome alignment is: sign(v(S)) == sign(pnl) → 1 else 0.

        Shapley(i) = E_π [ v(S∪{i}) - v(S) ]
        where S is the prefix of π before i.
        """
        signals = list(signal_snapshot.keys())
        n = len(signals)
        if n == 0:
            return {}
        if n == 1:
            return {signals[0]: 1.0 if pnl > 0 else -1.0}

        pnl_sign = np.sign(pnl)
        shapley: Dict[str, float] = {sig: 0.0 for sig in signals}
        rng = np.random.default_rng(seed=42)

        for _ in range(self._n_samples):
            perm = rng.permutation(n)
            coalition_signals: List[str] = []
            v_prev = 0.0

            for idx in perm:
                sig = signals[int(idx)]
                coalition_signals.append(sig)

                # v(S): mean of signal values in S, sign-compared to pnl
                vals = np.array([signal_snapshot[s] for s in coalition_signals])
                pred = float(np.mean(vals))
                # Value = 1 if prediction aligned with outcome, else 0
                v_curr = 1.0 if (np.sign(pred) == pnl_sign or pnl == 0) else 0.0

                shapley[sig] += (v_curr - v_prev)
                v_prev = v_curr

        # Normalise by number of samples
        for sig in shapley:
            shapley[sig] /= self._n_samples

        return shapley
