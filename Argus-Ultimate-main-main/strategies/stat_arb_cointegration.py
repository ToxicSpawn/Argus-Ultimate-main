"""
Statistical Arbitrage via Cointegration (Engle-Granger & Johansen).

Trades the spread between cointegrated cryptocurrency pairs using
z-score entry/exit signals with adaptive hedge ratios.

Pairs tracked: BTC/ETH, ETH/SOL, BTC/LTC, ETH/LINK
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from strategies.strategy_base import BaseStrategy, TradingSignal

logger = logging.getLogger(__name__)


class CointegrationPairsTrader(BaseStrategy):
    """
    Pairs trading strategy based on the Engle-Granger cointegration framework.

    For each configured pair (sym1, sym2):
      1. Estimate hedge ratio via OLS regression (prices2 ~ prices1).
      2. Construct the spread: spread = prices1 - hedge_ratio * prices2.
      3. Test stationarity of the spread with an Augmented Dickey-Fuller test.
      4. If cointegrated, compute the rolling z-score of the spread.
      5. Enter long-spread when z < -Z_ENTRY, short-spread when z > +Z_ENTRY.
      6. Exit when |z| < Z_EXIT; stop-loss when |z| > Z_STOP.
    """

    PAIRS: List[Tuple[str, str]] = [
        ("BTC/USD", "ETH/USD"),
        ("ETH/USD", "SOL/USD"),
        ("BTC/USD", "LTC/USD"),
    ]
    PRICE_HISTORY_LEN: int = 200  # bars needed for cointegration test
    Z_ENTRY: float = 2.0           # enter when |z-score| exceeds this
    Z_EXIT: float = 0.5            # exit when |z-score| drops below this
    Z_STOP: float = 4.0            # stop-loss when |z-score| exceeds this
    _COINT_RETEST_INTERVAL: int = 50  # re-run cointegration test every N updates
    _ADF_CRITICAL_5PCT: float = -2.86  # 5 % critical value for ADF (n ≈ 100)

    # Rolling cointegration monitoring
    _COINT_FAIL_PAUSE_THRESHOLD: int = 3  # consecutive failures before pausing
    _ADF_PVALUE_THRESHOLD: float = 0.10   # ADF stat must be more negative than -2.86 (p<0.05)

    # Spread acceleration — enter on deceleration, not peak
    USE_SPREAD_ACCELERATION: bool = True

    # Pair rotation — trade only the best pairs
    MAX_ACTIVE_PAIRS: int = 2  # only trade the top N pairs by score

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(name="stat_arb_cointegration", config=config)
        # symbol -> list of recent close prices (capped at PRICE_HISTORY_LEN)
        self._price_history: Dict[str, List[float]] = {}
        # (sym1, sym2) -> position info dict
        self._active_pairs: Dict[Tuple[str, str], Dict] = {}
        # (sym1, sym2) -> OLS hedge ratio β so that spread = p1 - β * p2
        self._hedge_ratios: Dict[Tuple[str, str], float] = {}
        # (sym1, sym2) -> True if latest ADF test passed cointegration
        self._cointegrated: Dict[Tuple[str, str], bool] = {}
        # bar counter per symbol, used to schedule re-tests
        self._bar_counts: Dict[str, int] = {}
        # cumulative update counter (any symbol) for scheduling
        self._update_count: int = 0

        # Rolling cointegration failure tracking
        self._coint_fail_counts: Dict[Tuple[str, str], int] = {}
        # Paused pairs (consecutive failures exceeded threshold)
        self._paused_pairs: Dict[Tuple[str, str], bool] = {}
        # Previous z-scores for spread acceleration
        self._prev_z: Dict[Tuple[str, str], List[float]] = {}
        # Pair rotation scores: (sym1, sym2) -> score
        self._pair_scores: Dict[Tuple[str, str], float] = {}
        # ADF t-statistics for pair scoring
        self._adf_stats: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def get_required_indicators(self) -> List[str]:
        return ["close", "volume"]

    async def analyze(  # type: ignore[override]
        self, market_data: Dict
    ) -> Optional[List[Dict]]:
        """
        Consume one price tick, update internal state, and emit signals.

        Parameters
        ----------
        market_data:
            Must contain at minimum ``symbol`` (str) and ``price`` (float).
            Optional keys: ``bid``, ``ask``, ``volume``.

        Returns
        -------
        A list of signal dicts (possibly empty), or ``None`` on error.
        """
        symbol: Optional[str] = market_data.get("symbol")
        price_raw = market_data.get("price")

        if symbol is None or price_raw is None:
            logger.debug("stat_arb: incomplete market_data, skipping")
            return None

        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            logger.warning("stat_arb: non-numeric price for %s", symbol)
            return None

        if price <= 0.0:
            return None

        self.update_prices(symbol, price)
        self._update_count += 1

        signals: List[Dict] = []

        # --- Pair rotation: score all pairs and select the best ---
        eligible_pairs: List[Tuple[str, str]] = []

        for sym1, sym2 in self.PAIRS:
            h1 = self._price_history.get(sym1, [])
            h2 = self._price_history.get(sym2, [])

            if len(h1) < self.PRICE_HISTORY_LEN or len(h2) < self.PRICE_HISTORY_LEN:
                continue  # not enough history yet

            p1 = np.asarray(h1, dtype=float)
            p2 = np.asarray(h2, dtype=float)

            pair_key = (sym1, sym2)

            # Re-run cointegration test every COINT_RETEST_INTERVAL updates
            needs_retest = (
                pair_key not in self._cointegrated
                or self._update_count % self._COINT_RETEST_INTERVAL == 0
            )
            if needs_retest:
                coint, hedge = self._is_cointegrated(p1, p2)

                # Rolling cointegration monitoring
                if not coint:
                    self._coint_fail_counts[pair_key] = self._coint_fail_counts.get(pair_key, 0) + 1
                    if self._coint_fail_counts[pair_key] >= self._COINT_FAIL_PAUSE_THRESHOLD:
                        self._paused_pairs[pair_key] = True
                        logger.warning(
                            "stat_arb: PAUSING pair %s/%s — %d consecutive "
                            "cointegration failures",
                            sym1, sym2, self._coint_fail_counts[pair_key],
                        )
                else:
                    # Reset failure counter on success
                    self._coint_fail_counts[pair_key] = 0
                    self._paused_pairs[pair_key] = False

                self._cointegrated[pair_key] = coint
                self._hedge_ratios[pair_key] = hedge

                # Store ADF stat for pair scoring
                if coint:
                    spread_test = p1 - hedge * p2
                    adf_stat = self._adf_test(spread_test)
                    self._adf_stats[pair_key] = adf_stat

                logger.debug(
                    "stat_arb: pair %s/%s cointegrated=%s hedge=%.4f paused=%s",
                    sym1, sym2, coint, hedge,
                    self._paused_pairs.get(pair_key, False),
                )

            # Skip paused pairs
            if self._paused_pairs.get(pair_key, False):
                continue

            if not self._cointegrated.get(pair_key, False):
                continue

            # Score the pair for rotation: |ADF stat| * |z-score| (higher = better)
            hedge_ratio = self._hedge_ratios[pair_key]
            spread = p1 - hedge_ratio * p2
            z = self._compute_zscore(spread)
            adf_stat = self._adf_stats.get(pair_key, 0.0)
            self._pair_scores[pair_key] = abs(adf_stat) * abs(z)

            eligible_pairs.append((sym1, sym2))

        # Sort by pair score descending, take only top MAX_ACTIVE_PAIRS
        eligible_pairs.sort(
            key=lambda pair: self._pair_scores.get(pair, 0.0),
            reverse=True,
        )
        active_pairs = eligible_pairs[:self.MAX_ACTIVE_PAIRS]

        for sym1, sym2 in active_pairs:
            pair_key = (sym1, sym2)
            h1 = self._price_history[sym1]
            h2 = self._price_history[sym2]
            p1 = np.asarray(h1, dtype=float)
            p2 = np.asarray(h2, dtype=float)

            hedge_ratio = self._hedge_ratios[pair_key]
            spread = p1 - hedge_ratio * p2
            z = self._compute_zscore(spread)

            # --- Spread acceleration filter ---
            if self.USE_SPREAD_ACCELERATION and pair_key not in self._active_pairs:
                if not self._check_spread_deceleration(pair_key, z):
                    continue  # only enter on deceleration

            # Track z-score history for acceleration
            if pair_key not in self._prev_z:
                self._prev_z[pair_key] = []
            self._prev_z[pair_key].append(z)
            if len(self._prev_z[pair_key]) > 5:
                self._prev_z[pair_key] = self._prev_z[pair_key][-5:]

            pair_signals = self._generate_signals(
                sym1=sym1,
                sym2=sym2,
                pair_key=pair_key,
                z=z,
                hedge_ratio=hedge_ratio,
                price1=float(p1[-1]),
                price2=float(p2[-1]),
            )
            signals.extend(pair_signals)

        self.performance_metrics["signals_generated"] += len(signals)
        return signals if signals else None

    # ------------------------------------------------------------------
    # Price history management
    # ------------------------------------------------------------------

    def update_prices(self, symbol: str, price: float) -> None:
        """Append *price* to the history for *symbol*, respecting the max length."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        buf = self._price_history[symbol]
        buf.append(price)
        if len(buf) > self.PRICE_HISTORY_LEN:
            # Trim from the front (oldest observations)
            del buf[: len(buf) - self.PRICE_HISTORY_LEN]
        self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_hedge_ratio(
        prices1: np.ndarray, prices2: np.ndarray
    ) -> float:
        """
        OLS estimate of β in the regression:  prices1 = β * prices2 + ε

        We regress prices1 (y) on prices2 with an intercept (X = [prices2, 1]).
        The first coefficient is the hedge ratio β.
        """
        n = len(prices2)
        X = np.column_stack([prices2, np.ones(n)])
        y = prices1
        result = np.linalg.lstsq(X, y, rcond=None)
        beta = float(result[0][0])
        return beta

    @staticmethod
    def _adf_test(series: np.ndarray) -> float:
        """
        Simplified Augmented Dickey-Fuller test.

        Tests the null hypothesis that *series* has a unit root.

        Procedure (1-lag ADF):
          1. First-difference: dy[t] = y[t] - y[t-1]
          2. Regress dy on y_lagged (with intercept) plus one lagged difference.
          3. Return the t-statistic on y_lagged.
             More negative => stronger evidence of stationarity.

        ADF 5 % critical value (n ≈ 100): −2.86
        """
        y = series
        n = len(y)
        if n < 10:
            return 0.0  # cannot compute meaningfully

        # dy = first differences
        dy = np.diff(y)  # length n-1

        # Build regressors: y_lagged (t-1), lagged_dy (t-2..t), intercept
        # Align indices for 1 lag:
        #   dy[1:]       = dy[t],   t = 2..n-1   (length n-2)
        #   y[1:-1]      = y[t-1],  t = 2..n-1
        #   dy[:-1]      = dy[t-1], t = 2..n-1  (lagged difference)
        y_lag = y[1:-1]       # y_{t-1}
        dy_lag1 = dy[:-1]     # Δy_{t-1}
        dep = dy[1:]          # Δy_t

        m = len(dep)
        if m < 5:
            return 0.0

        X = np.column_stack([y_lag, dy_lag1, np.ones(m)])
        coeffs, residuals, rank, _ = np.linalg.lstsq(X, dep, rcond=None)

        if rank < X.shape[1]:
            return 0.0

        beta_ylag = coeffs[0]
        if len(residuals) == 0:
            # Compute residuals manually when lstsq does not return them
            res = dep - X @ coeffs
        else:
            res = dep - X @ coeffs

        sigma2 = float(np.sum(res ** 2)) / max(m - X.shape[1], 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_beta = np.sqrt(sigma2 * float(XtX_inv[0, 0]))

        if se_beta < 1e-12:
            return 0.0

        t_stat = beta_ylag / se_beta
        return float(t_stat)

    def _is_cointegrated(
        self, prices1: np.ndarray, prices2: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Engle-Granger cointegration test.

        Steps:
          1. Estimate hedge ratio β via OLS.
          2. Compute residual spread = prices1 − β * prices2.
          3. Apply ADF to the spread.
          4. If ADF t-stat < critical value (-2.86), the pair is cointegrated.

        Returns
        -------
        (is_cointegrated, hedge_ratio)
        """
        try:
            hedge = self._estimate_hedge_ratio(prices1, prices2)
            if not np.isfinite(hedge) or abs(hedge) > 1000:
                return False, 1.0
            spread = prices1 - hedge * prices2
            adf_stat = self._adf_test(spread)
            is_coint = adf_stat < self._ADF_CRITICAL_5PCT
            return is_coint, hedge
        except Exception:
            logger.exception("stat_arb: cointegration test failed")
            return False, 1.0

    @staticmethod
    def _compute_zscore(spread: np.ndarray, lookback: int = 60) -> float:
        """
        Compute the z-score of the most recent spread observation.

        Uses the last *lookback* bars to estimate mean and standard deviation.
        Clamps result to [-6, 6].  Returns 0.0 when std is near-zero.
        """
        window = spread[-lookback:] if len(spread) >= lookback else spread
        if len(window) < 2:
            return 0.0
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        if sigma < 1e-10:
            return 0.0
        z = (float(spread[-1]) - mu) / sigma
        return float(np.clip(z, -6.0, 6.0))

    # ------------------------------------------------------------------
    # Spread acceleration / deceleration
    # ------------------------------------------------------------------

    def _check_spread_deceleration(
        self, pair_key: Tuple[str, str], current_z: float
    ) -> bool:
        """
        Check if spread is decelerating (rate of change slowing).

        For SHORT_SPREAD entries (z > 0): z should be decreasing or flattening
        (second derivative negative — spread widening is slowing).

        For LONG_SPREAD entries (z < 0): z should be increasing or flattening
        (second derivative positive — spread narrowing is slowing).

        Enter on deceleration, not at peak — catches the turn, avoids tops.
        Returns True if deceleration detected or insufficient history.
        """
        hist = self._prev_z.get(pair_key, [])
        if len(hist) < 3:
            return True  # not enough data, allow entry

        # Compute first and second derivatives of z-score
        dz_prev = hist[-1] - hist[-2]  # recent velocity
        dz_before = hist[-2] - hist[-3]

        # Second derivative (acceleration)
        ddz = dz_prev - dz_before

        if current_z > 0:
            # For short spread: we want z to be decelerating (ddz < 0)
            return ddz < 0
        elif current_z < 0:
            # For long spread: we want z to be decelerating upward (ddz > 0)
            return ddz > 0

        return True

    def get_pair_scores(self) -> Dict[str, float]:
        """Return current pair rotation scores (higher = stronger pair)."""
        return {
            f"{s1}/{s2}": score
            for (s1, s2), score in self._pair_scores.items()
        }

    def get_paused_pairs(self) -> List[str]:
        """Return list of paused pair labels."""
        return [
            f"{s1}/{s2}"
            for (s1, s2), paused in self._paused_pairs.items()
            if paused
        ]

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signals(
        self,
        sym1: str,
        sym2: str,
        pair_key: Tuple[str, str],
        z: float,
        hedge_ratio: float,
        price1: float,
        price2: float,
    ) -> List[Dict]:
        """
        Translate z-score crossings into entry / exit signals.

        Convention:
          * "Long spread"  = buy sym1 / sell sym2  (z is too negative)
          * "Short spread" = sell sym1 / buy sym2  (z is too positive)
        """
        signals: List[Dict] = []
        position = self._active_pairs.get(pair_key)

        abs_z = abs(z)

        # ---- Stop-loss -----------------------------------------------
        if position is not None and abs_z > self.Z_STOP:
            exit_signals = self._build_exit_signals(
                sym1=sym1,
                sym2=sym2,
                position=position,
                z=z,
                hedge_ratio=hedge_ratio,
                price1=price1,
                price2=price2,
                reason="stop_loss",
            )
            signals.extend(exit_signals)
            del self._active_pairs[pair_key]
            return signals

        # ---- Normal exit (mean reversion) ----------------------------
        if position is not None and abs_z < self.Z_EXIT:
            exit_signals = self._build_exit_signals(
                sym1=sym1,
                sym2=sym2,
                position=position,
                z=z,
                hedge_ratio=hedge_ratio,
                price1=price1,
                price2=price2,
                reason="mean_reversion",
            )
            signals.extend(exit_signals)
            del self._active_pairs[pair_key]
            return signals

        # ---- Entry ---------------------------------------------------
        if position is None and abs_z > self.Z_ENTRY:
            confidence = min(0.95, 0.5 + (abs_z - self.Z_ENTRY) * 0.1)

            if z > self.Z_ENTRY:
                # Spread too high: short spread (sell sym1, buy sym2)
                action1, action2 = "SELL", "BUY"
                direction = "short_spread"
            else:
                # Spread too low: long spread (buy sym1, sell sym2)
                action1, action2 = "BUY", "SELL"
                direction = "long_spread"

            self._active_pairs[pair_key] = {
                "direction": direction,
                "entry_z": z,
                "hedge_ratio": hedge_ratio,
            }

            base = {
                "source": "stat_arb_cointegration",
                "pair": f"{sym1}/{sym2}",
                "z_score": z,
                "hedge_ratio": hedge_ratio,
                "confidence": confidence,
            }
            signals.append({
                **base,
                "symbol": sym1,
                "symbol2": sym2,
                "action": action1,
                "price": price1,
            })
            signals.append({
                **base,
                "symbol": sym2,
                "symbol2": sym1,
                "action": action2,
                "price": price2,
            })
            logger.info(
                "stat_arb: entering %s on %s/%s  z=%.3f  hedge=%.4f",
                direction,
                sym1,
                sym2,
                z,
                hedge_ratio,
            )

        return signals

    @staticmethod
    def _build_exit_signals(
        sym1: str,
        sym2: str,
        position: Dict,
        z: float,
        hedge_ratio: float,
        price1: float,
        price2: float,
        reason: str,
    ) -> List[Dict]:
        direction = position.get("direction", "long_spread")
        if direction == "long_spread":
            action1, action2 = "SELL", "BUY"
        else:
            action1, action2 = "BUY", "SELL"

        base = {
            "source": "stat_arb_cointegration",
            "pair": f"{sym1}/{sym2}",
            "z_score": z,
            "hedge_ratio": hedge_ratio,
            "confidence": 0.8,
            "exit_reason": reason,
        }
        return [
            {**base, "symbol": sym1, "symbol2": sym2, "action": action1, "price": price1},
            {**base, "symbol": sym2, "symbol2": sym1, "action": action2, "price": price2},
        ]

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    def generate_orders(
        self, signal: Dict, portfolio_value: float
    ) -> List[Dict]:
        """Convert a cointegration signal dict to pairs orders.

        Each signal dict (as emitted by ``_generate_signals``) contains
        ``symbol``, ``action``, ``price``, ``confidence``, ``hedge_ratio``,
        and ``pair``.  This method converts the raw signal into order dicts
        suitable for execution.

        Parameters
        ----------
        signal : dict
            A single signal dict from ``analyze()`` output.
        portfolio_value : float
            Current total portfolio value in USD.

        Returns
        -------
        List of order dicts with keys: symbol, side, quantity, order_type, reason.
        """
        action = signal.get("action", "")
        symbol = signal.get("symbol", "")
        price = signal.get("price", 0.0)
        confidence = signal.get("confidence", 0.5)
        pair = signal.get("pair", "")
        hedge_ratio = signal.get("hedge_ratio", 1.0)
        z_score = signal.get("z_score", 0.0)

        if not action or not symbol or price <= 0:
            return []

        # Size: fraction of portfolio scaled by confidence
        size_fraction = min(0.10, confidence * 0.12)
        notional = portfolio_value * size_fraction
        quantity = notional / price

        exit_reason = signal.get("exit_reason", "")
        if exit_reason:
            reason = f"stat_arb_exit_{exit_reason}_pair_{pair}_z_{z_score:.2f}"
        else:
            reason = f"stat_arb_entry_pair_{pair}_z_{z_score:.2f}_hedge_{hedge_ratio:.3f}"

        return [{
            "symbol": symbol,
            "side": action,
            "quantity": quantity,
            "order_type": "limit",
            "reason": reason,
        }]

    # ------------------------------------------------------------------
    # Status / reporting
    # ------------------------------------------------------------------

    def get_pair_status(self) -> Dict:
        """
        Return a snapshot of z-scores, cointegration flags, hedge ratios,
        and active positions for every configured pair.
        """
        status: Dict = {}
        for sym1, sym2 in self.PAIRS:
            pair_key = (sym1, sym2)
            h1 = self._price_history.get(sym1, [])
            h2 = self._price_history.get(sym2, [])

            z: Optional[float] = None
            if (
                len(h1) >= self.PRICE_HISTORY_LEN
                and len(h2) >= self.PRICE_HISTORY_LEN
                and self._cointegrated.get(pair_key, False)
            ):
                hedge = self._hedge_ratios.get(pair_key, 1.0)
                p1 = np.asarray(h1, dtype=float)
                p2 = np.asarray(h2, dtype=float)
                spread = p1 - hedge * p2
                z = self._compute_zscore(spread)

            status[f"{sym1}/{sym2}"] = {
                "cointegrated": self._cointegrated.get(pair_key, False),
                "hedge_ratio": self._hedge_ratios.get(pair_key),
                "z_score": z,
                "active_position": self._active_pairs.get(pair_key),
                "history_len_sym1": len(h1),
                "history_len_sym2": len(h2),
            }
        return status
