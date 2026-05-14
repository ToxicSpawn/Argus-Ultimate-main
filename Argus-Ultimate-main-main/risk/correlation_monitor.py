"""
Correlation Breakdown Monitor.

Detects when pairwise asset correlations spike toward 1.0 — the hallmark
of risk-off panic selling (March 2020, May 2021, Nov 2022 style events).

When all assets move together, diversification fails and all positions
lose simultaneously. This module signals when to reduce all positions.

Thresholds:
  avg_pairwise > 0.80 → alert, reduce positions by 50%
  avg_pairwise > 0.92 → crisis, reduce positions by 90%
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ALERT_THRESHOLD  = 0.80   # avg pairwise correlation → alert + halve positions
CRISIS_THRESHOLD = 0.92   # avg pairwise correlation → near-full risk-off
LOOKBACK         = 20     # rolling correlation window in bars
COOLDOWN_SEC     = 3600   # 1 hour between repeated alerts


class CorrelationMonitor:
    """
    Rolling pairwise correlation tracker across a portfolio of symbols.

    Feed prices via update() after each bar close. Call check_and_alert()
    to receive an alert dict when correlations are dangerously high.
    """

    def __init__(
        self,
        symbols: List[str],
        lookback: int = LOOKBACK,
        alert_threshold: float = ALERT_THRESHOLD,
        crisis_threshold: float = CRISIS_THRESHOLD,
    ):
        self.symbols = symbols
        self.lookback = lookback
        self.alert_threshold = alert_threshold
        self.crisis_threshold = crisis_threshold

        self._price_history: Dict[str, Deque[float]] = {
            s: deque(maxlen=lookback + 5) for s in symbols
        }
        self._last_alert_ts: float = 0.0
        self._last_matrix: Optional[np.ndarray] = None
        self._last_avg_corr: float = 0.0

    def update(self, symbol: str, price: float) -> None:
        """Feed a new price observation for a symbol."""
        if symbol in self._price_history:
            self._price_history[symbol].append(float(price))

    def compute_correlation_matrix(self) -> Optional[np.ndarray]:
        """
        Compute the rolling pairwise Pearson correlation matrix.

        Returns (n_symbols, n_symbols) array or None if insufficient data.
        """
        eligible = [
            s for s in self.symbols
            if len(self._price_history[s]) >= max(self.lookback // 2, 5)
        ]
        if len(eligible) < 2:
            return None

        # Build returns matrix
        returns_cols = []
        used_symbols = []
        for sym in eligible:
            prices = np.array(list(self._price_history[sym]))
            if len(prices) < 2:
                continue
            log_ret = np.diff(np.log(np.maximum(prices, 1e-10)))
            returns_cols.append(log_ret)
            used_symbols.append(sym)

        if len(returns_cols) < 2:
            return None

        # Align lengths
        min_len = min(len(r) for r in returns_cols)
        matrix = np.column_stack([r[-min_len:] for r in returns_cols])  # (T, n)

        try:
            corr = np.corrcoef(matrix.T)  # (n, n)
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            self._last_matrix = corr
            return corr
        except Exception as exc:
            logger.debug("Correlation matrix computation failed: %s", exc)
            return None

    def get_avg_pairwise_correlation(self) -> float:
        """
        Compute mean of upper-triangle pairwise correlations (excluding diagonal).

        Returns 0.0 if insufficient data.
        """
        corr = self.compute_correlation_matrix()
        if corr is None:
            return 0.0

        n = corr.shape[0]
        upper_idx = np.triu_indices(n, k=1)
        upper = corr[upper_idx]
        upper = upper[~np.isnan(upper)]

        if len(upper) == 0:
            return 0.0

        avg = float(np.mean(upper))
        self._last_avg_corr = avg
        return avg

    def get_position_scalar(self) -> float:
        """
        Position size scalar based on current correlation level.

        Returns:
            1.0 — normal (no correlation concern)
            0.5 — moderate (correlations rising, halve positions)
            0.1 — crisis (near-full risk-off, minimal exposure)
        """
        avg = self.get_avg_pairwise_correlation()

        if avg >= self.crisis_threshold:
            return 0.10
        elif avg >= self.alert_threshold:
            # Linear interpolation from 0.5 (at alert) to 0.1 (at crisis)
            ratio = (avg - self.alert_threshold) / max(self.crisis_threshold - self.alert_threshold, 1e-6)
            return float(np.clip(0.5 - ratio * 0.4, 0.1, 0.5))
        else:
            return 1.0

    def check_and_alert(self) -> Optional[Dict[str, Any]]:
        """
        Check current correlation level and return an alert dict if warranted.

        Applies a cooldown to avoid repeated alerts within 1 hour.

        Returns:
            Alert dict or None.
        """
        now = time.time()
        avg_corr = self.get_avg_pairwise_correlation()

        if avg_corr < self.alert_threshold:
            return None

        if (now - self._last_alert_ts) < COOLDOWN_SEC:
            return None  # Suppress repeat alerts

        is_crisis = avg_corr >= self.crisis_threshold
        scalar = self.get_position_scalar()

        self._last_alert_ts = now
        level = "CRISIS" if is_crisis else "WARNING"

        logger.warning(
            "CorrelationMonitor %s: avg_pairwise=%.3f scalar=%.2f",
            level, avg_corr, scalar,
        )

        return {
            "alert": True,
            "level": level,
            "avg_correlation": round(avg_corr, 4),
            "position_scalar": scalar,
            "crisis": is_crisis,
            "symbols_monitored": list(self.symbols),
            "lookback_bars": self.lookback,
            "recommendation": (
                "Reduce ALL positions immediately — correlation spike indicates panic selling"
                if is_crisis else
                "Consider reducing positions — cross-asset correlations rising"
            ),
        }

    def get_correlation_penalty(self, symbols: Optional[List[str]] = None) -> float:
        """
        Return a position-limit multiplier (0.5–1.0) based on average pairwise
        correlation among *symbols* (defaults to all monitored symbols).

        Mapping (linear interpolation):
          avg_corr >= 0.8  →  0.5  (halve position limits)
          avg_corr <= 0.3  →  1.0  (no reduction)
          between          →  linear from 1.0 down to 0.5

        This is a gentler, always-on scaling intended for position sizing,
        distinct from get_position_scalar() which handles crisis/alert states.
        """
        LOW = 0.3
        HIGH = 0.8

        # If a subset of symbols is requested, compute pairwise corr for just those
        if symbols is not None:
            if len(symbols) < 2:
                return 1.0  # Can't compute pairwise correlation with < 2 symbols
            avg = self._avg_pairwise_for_subset(symbols)
        else:
            avg = self.get_avg_pairwise_correlation()

        if avg <= LOW:
            return 1.0
        if avg >= HIGH:
            return 0.5
        # Linear interpolation: 1.0 at LOW → 0.5 at HIGH
        ratio = (avg - LOW) / (HIGH - LOW)
        return float(np.clip(1.0 - ratio * 0.5, 0.5, 1.0))

    def _avg_pairwise_for_subset(self, symbols: List[str]) -> float:
        """Compute average pairwise correlation for a subset of tracked symbols."""
        eligible = [s for s in symbols if s in self._price_history
                    and len(self._price_history[s]) >= max(self.lookback // 2, 5)]
        if len(eligible) < 2:
            return 0.0

        returns_cols = []
        for sym in eligible:
            prices = np.array(list(self._price_history[sym]))
            if len(prices) < 2:
                continue
            log_ret = np.diff(np.log(np.maximum(prices, 1e-10)))
            returns_cols.append(log_ret)

        if len(returns_cols) < 2:
            return 0.0

        min_len = min(len(r) for r in returns_cols)
        matrix = np.column_stack([r[-min_len:] for r in returns_cols])

        try:
            corr = np.corrcoef(matrix.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            n = corr.shape[0]
            upper_idx = np.triu_indices(n, k=1)
            upper = corr[upper_idx]
            upper = upper[~np.isnan(upper)]
            return float(np.mean(upper)) if len(upper) > 0 else 0.0
        except Exception:
            return 0.0

    def get_most_correlated_pair(self) -> Optional[Tuple[str, str, float]]:
        """Return the most correlated symbol pair (sym1, sym2, corr_value)."""
        corr = self._last_matrix
        if corr is None or len(self.symbols) < 2:
            return None

        n = min(corr.shape[0], len(self.symbols))
        best_val = -1.0
        best_pair: Optional[Tuple[str, str, float]] = None

        for i in range(n):
            for j in range(i + 1, n):
                val = float(corr[i, j])
                if val > best_val:
                    best_val = val
                    best_pair = (self.symbols[i], self.symbols[j], val)

        return best_pair

    def get_status(self) -> Dict[str, Any]:
        """Return current correlation status."""
        avg = self._last_avg_corr
        return {
            "avg_pairwise_correlation": round(avg, 4),
            "position_scalar": self.get_position_scalar(),
            "alert_threshold": self.alert_threshold,
            "crisis_threshold": self.crisis_threshold,
            "symbols": self.symbols,
            "n_symbols_with_data": sum(
                1 for s in self.symbols
                if len(self._price_history[s]) >= 5
            ),
        }
