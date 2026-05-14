#!/usr/bin/env python3
"""
Dynamic Attention Allocation — decides what ARGUS should focus on RIGHT NOW.

Prevents wasting compute on irrelevant symbols/strategies by ranking everything
by importance.  Factors: volatility, open positions, stop proximity, signal
strength, volume anomalies, regime relevance.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AttentionItem:
    """A single item in the attention ranking."""
    name: str
    category: str  # "symbol", "strategy", "risk"
    score: float  # 0..1
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttentionMap:
    """Full attention allocation for the current cycle."""
    focus_symbols: List[AttentionItem] = field(default_factory=list)
    focus_strategies: List[AttentionItem] = field(default_factory=list)
    focus_risks: List[AttentionItem] = field(default_factory=list)
    ignore_list: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focus_symbols": [i.to_dict() for i in self.focus_symbols],
            "focus_strategies": [i.to_dict() for i in self.focus_strategies],
            "focus_risks": [i.to_dict() for i in self.focus_risks],
            "ignore_list": self.ignore_list,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# AttentionSystem
# ---------------------------------------------------------------------------

class AttentionSystem:
    """
    Computes dynamic attention allocation across symbols, strategies, and risks.

    Parameters
    ----------
    config : dict, optional
        ``attention_system`` section from unified_config.yaml.
    """

    _DEFAULTS: Dict[str, Any] = {
        "enabled": True,
        "high_vol_weight": 0.25,
        "open_position_weight": 0.30,
        "stop_proximity_weight": 0.20,
        "signal_strength_weight": 0.15,
        "volume_anomaly_weight": 0.10,
        "ignore_threshold": 0.10,
        "max_focus_symbols": 10,
        "max_focus_strategies": 5,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(self._DEFAULTS)
        if config:
            cfg.update(config)
        self._cfg = cfg
        self._enabled = bool(cfg.get("enabled", True))

        # Weights
        self._w_vol = float(cfg.get("high_vol_weight", 0.25))
        self._w_pos = float(cfg.get("open_position_weight", 0.30))
        self._w_stop = float(cfg.get("stop_proximity_weight", 0.20))
        self._w_sig = float(cfg.get("signal_strength_weight", 0.15))
        self._w_vol_anom = float(cfg.get("volume_anomaly_weight", 0.10))
        self._ignore_thresh = float(cfg.get("ignore_threshold", 0.10))
        self._max_syms = int(cfg.get("max_focus_symbols", 10))
        self._max_strats = int(cfg.get("max_focus_strategies", 5))

        # History for adaptive weighting
        self._attention_history: List[AttentionMap] = []

        logger.info("AttentionSystem initialised (enabled=%s)", self._enabled)

    # ------------------------------------------------------------------
    # Core: compute attention
    # ------------------------------------------------------------------

    def compute_attention(self, market_state: Dict[str, Any]) -> AttentionMap:
        """
        Compute what ARGUS should focus on right now.

        Parameters
        ----------
        market_state : dict
            Expected keys:
            - symbols: dict[symbol, dict] with vol, price, change_pct, volume_ratio
            - positions: dict[symbol, dict] with entry_price, stop_loss, size, pnl
            - signals: dict[symbol, dict] with score/strength
            - strategies: dict[name, dict] with recent_pnl, win_rate, regime_fit
            - risk_metrics: dict with drawdown_pct, var, exposure_pct

        Returns
        -------
        AttentionMap
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        # --- Symbol attention ---
        symbol_items = self._rank_symbols(market_state)

        # --- Strategy attention ---
        strategy_items = self._rank_strategies(market_state)

        # --- Risk attention ---
        risk_items = self._rank_risks(market_state)

        # --- Ignore list ---
        all_syms = set()
        symbols_data = market_state.get("symbols", {})
        if isinstance(symbols_data, dict):
            all_syms = set(symbols_data.keys())
        focus_sym_names = {i.name for i in symbol_items}
        ignore = sorted(all_syms - focus_sym_names)

        attention = AttentionMap(
            focus_symbols=symbol_items,
            focus_strategies=strategy_items,
            focus_risks=risk_items,
            ignore_list=ignore,
            timestamp=now,
        )

        # Keep history (bounded)
        self._attention_history.append(attention)
        if len(self._attention_history) > 100:
            self._attention_history = self._attention_history[-100:]

        return attention

    # ------------------------------------------------------------------
    # Symbol ranking
    # ------------------------------------------------------------------

    def _rank_symbols(self, state: Dict[str, Any]) -> List[AttentionItem]:
        """Score and rank symbols by attention priority."""
        symbols = state.get("symbols", {})
        positions = state.get("positions", {})
        signals = state.get("signals", {})

        items: List[AttentionItem] = []
        for sym, data in symbols.items():
            if not isinstance(data, dict):
                continue
            score = 0.0
            reasons: List[str] = []

            # Volatility factor
            vol = data.get("volatility", data.get("vol", data.get("atr_pct", 0.0)))
            if vol > 0:
                vol_score = min(vol / 5.0, 1.0)  # normalise: 5% vol -> 1.0
                score += self._w_vol * vol_score
                if vol_score > 0.5:
                    reasons.append(f"high volatility ({vol:.1f}%)")

            # Open position factor
            pos = positions.get(sym, {})
            if pos:
                score += self._w_pos
                reasons.append("open position")

                # Stop proximity factor
                stop_loss = pos.get("stop_loss")
                current_price = data.get("price", data.get("last", 0))
                if stop_loss and current_price and current_price > 0:
                    dist_pct = abs(current_price - stop_loss) / current_price * 100
                    if dist_pct < 2.0:
                        stop_score = 1.0
                        reasons.append(f"NEAR STOP ({dist_pct:.1f}%)")
                    elif dist_pct < 5.0:
                        stop_score = 0.6
                        reasons.append(f"approaching stop ({dist_pct:.1f}%)")
                    else:
                        stop_score = 0.2
                    score += self._w_stop * stop_score

            # Signal strength factor
            sig = signals.get(sym, {})
            sig_val = 0.0
            if isinstance(sig, dict):
                sig_val = abs(sig.get("score", sig.get("strength", 0.0)))
            elif isinstance(sig, (int, float)):
                sig_val = abs(float(sig))
            if sig_val > 0:
                sig_score = min(sig_val / 1.0, 1.0)
                score += self._w_sig * sig_score
                if sig_score > 0.5:
                    reasons.append(f"strong signal ({sig_val:.3f})")

            # Volume anomaly factor
            vol_ratio = data.get("volume_ratio", 1.0)
            if vol_ratio > 1.5:
                anom_score = min((vol_ratio - 1.0) / 3.0, 1.0)
                score += self._w_vol_anom * anom_score
                reasons.append(f"volume {vol_ratio:.1f}x average")

            if score >= self._ignore_thresh:
                items.append(AttentionItem(
                    name=sym, category="symbol",
                    score=round(min(score, 1.0), 4),
                    reasons=reasons,
                ))

        # Sort descending by score
        items.sort(key=lambda x: x.score, reverse=True)
        return items[:self._max_syms]

    # ------------------------------------------------------------------
    # Strategy ranking
    # ------------------------------------------------------------------

    def _rank_strategies(self, state: Dict[str, Any]) -> List[AttentionItem]:
        """Rank strategies by relevance."""
        strategies = state.get("strategies", {})
        items: List[AttentionItem] = []

        for name, data in strategies.items():
            if not isinstance(data, dict):
                continue
            score = 0.0
            reasons: List[str] = []

            # Recent performance
            pnl = data.get("recent_pnl", 0.0)
            if pnl > 0:
                score += 0.3
                reasons.append(f"profitable (PnL={pnl:.2f})")
            elif pnl < 0:
                score += 0.4  # needs attention because it's losing
                reasons.append(f"losing money (PnL={pnl:.2f})")

            # Win rate
            wr = data.get("win_rate", 0.5)
            if wr < 0.4:
                score += 0.2
                reasons.append(f"low win rate ({wr:.0%})")
            elif wr > 0.6:
                score += 0.2
                reasons.append(f"high win rate ({wr:.0%})")

            # Regime fit
            regime_fit = data.get("regime_fit", 0.5)
            if regime_fit > 0.7:
                score += 0.3
                reasons.append("good regime fit")
            elif regime_fit < 0.3:
                score += 0.1
                reasons.append("poor regime fit")

            if score > 0.1:
                items.append(AttentionItem(
                    name=name, category="strategy",
                    score=round(min(score, 1.0), 4),
                    reasons=reasons,
                ))

        items.sort(key=lambda x: x.score, reverse=True)
        return items[:self._max_strats]

    # ------------------------------------------------------------------
    # Risk ranking
    # ------------------------------------------------------------------

    def _rank_risks(self, state: Dict[str, Any]) -> List[AttentionItem]:
        """Identify top risk concerns."""
        risk = state.get("risk_metrics", {})
        items: List[AttentionItem] = []

        dd = risk.get("drawdown_pct", 0.0)
        if dd > 3.0:
            severity = min(dd / 10.0, 1.0)
            items.append(AttentionItem(
                name="drawdown", category="risk",
                score=round(severity, 4),
                reasons=[f"drawdown at {dd:.1f}%"],
            ))

        var = risk.get("var", 0.0)
        if var > 2.0:
            items.append(AttentionItem(
                name="value_at_risk", category="risk",
                score=round(min(var / 5.0, 1.0), 4),
                reasons=[f"VaR at {var:.1f}%"],
            ))

        exposure = risk.get("exposure_pct", 0.0)
        if exposure > 80.0:
            items.append(AttentionItem(
                name="overexposure", category="risk",
                score=round(min(exposure / 100.0, 1.0), 4),
                reasons=[f"exposure at {exposure:.0f}%"],
            ))

        corr = risk.get("correlation_risk", 0.0)
        if corr > 0.7:
            items.append(AttentionItem(
                name="correlation_risk", category="risk",
                score=round(corr, 4),
                reasons=[f"portfolio correlation at {corr:.2f}"],
            ))

        items.sort(key=lambda x: x.score, reverse=True)
        return items

    # ------------------------------------------------------------------
    # Processing priority
    # ------------------------------------------------------------------

    def get_processing_priority(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Return an ordered list of what to process first.

        Returns list of dicts: [{"name": ..., "category": ..., "score": ...}, ...]
        """
        attention = self.compute_attention(market_state)
        all_items: List[AttentionItem] = []
        all_items.extend(attention.focus_risks)  # risks first
        all_items.extend(attention.focus_symbols)
        all_items.extend(attention.focus_strategies)

        # Re-sort by score globally, but bias risks higher
        for item in all_items:
            if item.category == "risk":
                item.score = min(item.score + 0.2, 1.0)  # risk boost

        all_items.sort(key=lambda x: x.score, reverse=True)
        return [item.to_dict() for item in all_items]

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_attention_trend(self, symbol: str, lookback: int = 20) -> List[float]:
        """Return recent attention scores for a symbol."""
        scores: List[float] = []
        for am in self._attention_history[-lookback:]:
            found = False
            for item in am.focus_symbols:
                if item.name == symbol:
                    scores.append(item.score)
                    found = True
                    break
            if not found:
                scores.append(0.0)
        return scores
