"""
Alpha Decay Tracker.

Measures how quickly a strategy's trading edge (alpha) decays
after signal generation. This determines the optimal execution window.

If alpha peaks at 5 minutes but TWAP runs over 2 hours,
you are trading against your own signal decay.

Usage:
    tracker = AlphaDecayTracker("momentum")
    tracker.record_signal("BTC/USD", "long", 65000.0)
    tracker.update_price("BTC/USD", 65200.0)   # called on each new bar
    decay = tracker.compute_alpha_decay()
    logger.info(decay["peak_horizon"])  # e.g. 900 (15 minutes)
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Measurement horizons in seconds
HORIZONS = [300, 900, 1800, 3600, 14400]  # 5m, 15m, 30m, 1h, 4h
HORIZON_LABELS = {300: "5m", 900: "15m", 1800: "30m", 3600: "1h", 14400: "4h"}
MAX_SIGNALS = 1000    # cap stored signals to avoid memory bloat
MAX_HORIZON_RETURNS = 10_000  # max per-horizon return samples kept in memory
PRICE_HISTORY_TTL = 6 * 3600  # keep price history for 6 hours


class AlphaDecayTracker:
    """
    Tracks signal alpha at multiple forward horizons.

    Feed price updates after each bar, and call compute_alpha_decay()
    periodically to see where your edge is concentrated.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        # signal_id -> {ts, symbol, direction, entry_price, returns: {horizon: float|None}}
        self._signals: Dict[str, Dict[str, Any]] = {}
        # symbol -> [(ts, price)]
        self._price_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        # Aggregated returns per horizon — bounded deque to prevent memory leaks
        self._horizon_returns: Dict[int, Deque[float]] = {
            h: deque(maxlen=MAX_HORIZON_RETURNS) for h in HORIZONS
        }
        self._last_cleanup: float = time.time()

    def record_signal(
        self, symbol: str, direction: str, entry_price: float
    ) -> str:
        """
        Record a new signal for alpha decay tracking.

        Args:
            symbol:      e.g. "BTC/USD"
            direction:   "long" or "short"
            entry_price: Price at signal generation time

        Returns:
            signal_id (use for manual tagging if needed)
        """
        signal_id = f"{symbol}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self._signals[signal_id] = {
            "ts": time.time(),
            "symbol": symbol,
            "direction": direction.lower(),
            "entry_price": float(entry_price),
            "returns": {h: None for h in HORIZONS},
            "evaluated": {h: False for h in HORIZONS},
        }

        # Trim old signals
        if len(self._signals) > MAX_SIGNALS:
            oldest = sorted(self._signals.items(), key=lambda x: x[1]["ts"])
            for sid, _ in oldest[:MAX_SIGNALS // 4]:
                del self._signals[sid]

        return signal_id

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update live price for a symbol. Triggers horizon evaluation.

        Call this every bar (or every minute minimum).
        """
        now = time.time()
        self._price_history[symbol].append((now, float(price)))

        # Trim old price history
        cutoff = now - PRICE_HISTORY_TTL
        self._price_history[symbol] = [
            (ts, p) for ts, p in self._price_history[symbol] if ts >= cutoff
        ]

        # Evaluate pending signals
        self._evaluate_pending(symbol, price, now)

        # Periodic cleanup
        if now - self._last_cleanup > 3600:
            self._cleanup_old_signals(now)
            self._last_cleanup = now

    def _evaluate_pending(self, symbol: str, current_price: float, now: float) -> None:
        """Check each pending signal against its horizon checkpoints."""
        for sig_id, sig in self._signals.items():
            if sig["symbol"] != symbol:
                continue
            entry_ts = sig["ts"]
            entry_price = sig["entry_price"]
            direction = sig["direction"]

            for horizon in HORIZONS:
                if sig["evaluated"][horizon]:
                    continue
                if now >= entry_ts + horizon:
                    # Find the closest price to this horizon timestamp
                    target_ts = entry_ts + horizon
                    future_price = self._find_price_near(symbol, target_ts)
                    if future_price is None:
                        future_price = current_price

                    raw_ret = (future_price - entry_price) / max(entry_price, 1e-10)
                    signed_ret = raw_ret if direction == "long" else -raw_ret
                    sig["returns"][horizon] = signed_ret
                    sig["evaluated"][horizon] = True
                    self._horizon_returns[horizon].append(signed_ret)

    def _find_price_near(self, symbol: str, target_ts: float) -> Optional[float]:
        """Find the closest available price to a target timestamp."""
        history = self._price_history.get(symbol, [])
        if not history:
            return None
        closest = min(history, key=lambda x: abs(x[0] - target_ts))
        return float(closest[1])

    def _cleanup_old_signals(self, now: float) -> None:
        """Remove signals that have been fully evaluated or are too old."""
        max_age = HORIZONS[-1] + 3600  # 1 hour past the longest horizon
        to_delete = [
            sid for sid, sig in self._signals.items()
            if now - sig["ts"] > max_age
        ]
        for sid in to_delete:
            del self._signals[sid]
        if to_delete:
            logger.debug("AlphaDecay %s: cleaned up %d old signals", self.strategy_name, len(to_delete))

    def compute_alpha_decay(self) -> Dict[str, Any]:
        """
        Compute the alpha decay curve across all measurement horizons.

        Returns:
            {
              "strategy": str,
              "horizons": {300: avg_return, 900: ..., ...},
              "horizon_labels": {300: "5m", 900: "15m", ...},
              "peak_horizon": int,        # horizon with maximum alpha
              "half_life_estimate": float, # seconds until alpha halves from peak
              "optimal_execution_window": int,  # recommended TWAP duration
              "n_signals": int,
              "n_evaluated_per_horizon": Dict[int, int],
            }
        """
        horizon_avgs: Dict[int, float] = {}
        horizon_counts: Dict[int, int] = {}

        for horizon in HORIZONS:
            # _horizon_returns[horizon] is a deque — iterate directly
            returns = [r for r in self._horizon_returns[horizon] if r is not None]
            # Also check signals that have been evaluated
            extra = [
                sig["returns"][horizon]
                for sig in self._signals.values()
                if sig["evaluated"][horizon] and sig["returns"][horizon] is not None
            ]
            all_returns = returns + extra

            horizon_avgs[horizon] = float(np.mean(all_returns)) if all_returns else 0.0
            horizon_counts[horizon] = len(all_returns)

        # Find peak horizon (where |alpha| is largest)
        peak_horizon = max(HORIZONS, key=lambda h: abs(horizon_avgs.get(h, 0.0)))
        peak_alpha = horizon_avgs.get(peak_horizon, 0.0)

        # Estimate half-life: find where alpha drops to half its peak
        half_life = float(peak_horizon)
        if abs(peak_alpha) > 1e-6:
            half_target = abs(peak_alpha) / 2.0
            for h in sorted(HORIZONS):
                if h > peak_horizon and abs(horizon_avgs.get(h, 0.0)) <= half_target:
                    half_life = float(h)
                    break

        # Optimal execution window = use ~80% of the peak horizon
        optimal_window = max(int(peak_horizon * 0.8), 60)

        total_signals = len(self._signals) + sum(
            len(returns) for returns in self._horizon_returns.values()
        )

        return {
            "strategy": self.strategy_name,
            "horizons": horizon_avgs,
            "horizon_labels": {h: HORIZON_LABELS.get(h, f"{h}s") for h in HORIZONS},
            "peak_horizon": peak_horizon,
            "peak_alpha": round(peak_alpha, 6),
            "half_life_estimate_seconds": round(half_life, 0),
            "optimal_execution_window_seconds": optimal_window,
            "n_signals_total": len(self._signals),
            "n_evaluated_per_horizon": horizon_counts,
        }

    def get_optimal_execution_horizon(self) -> int:
        """Returns recommended TWAP/VWAP execution window in seconds."""
        decay = self.compute_alpha_decay()
        return decay["optimal_execution_window_seconds"]

    def reset(self) -> None:
        """Clear all signal history (use when strategy is reset or reloaded)."""
        self._signals.clear()
        self._price_history.clear()
        self._horizon_returns = {
            h: deque(maxlen=MAX_HORIZON_RETURNS) for h in HORIZONS
        }
        logger.info("AlphaDecayTracker %s: reset", self.strategy_name)
