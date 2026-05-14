"""
Avellaneda-Stoikov v2 — Enhanced Optimal Market Making Strategy.

Extends the original Avellaneda-Stoikov (2008) model with:
  - Guéant-Lehalle-Fernandez-Tapia (2013) closed-form optimal spread including
    Poisson order arrival intensity:  λ(δ) = A * exp(-k*δ)
  - Inventory penalty and asymmetric quote skewing
  - Regime-aware quoting (NORMAL / ADVERSE / TRENDING / ILLIQUID)
  - Online Bayesian updating of intensity parameters (k, A) from fill history

References:
  Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
    https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf
  Guéant, Lehalle & Fernandez-Tapia (2013), "Dealing with the Inventory Risk"
    https://doi.org/10.1007/s11579-012-0090-2
"""
from __future__ import annotations

import logging
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from strategies.market_maker_avellaneda import (
    AvellanedaStoikovMM,
    GAMMA_DEFAULT,
    K_DEFAULT,
    T_SESSION,
    SIGMA_WINDOW,
    MAX_INVENTORY,
    MIN_SPREAD_BPS,
    MAX_SPREAD_BPS,
)

logger = logging.getLogger(__name__)

# ── intensity model defaults ──────────────────────────────────────────────────
A_DEFAULT    = 1.0    # baseline order arrival rate (orders per unit time)
VPIN_ADVERSE = 0.70   # VPIN threshold that triggers ADVERSE regime
OBI_TRENDING = 0.40   # OBI magnitude that triggers TRENDING regime
SPREAD_ILLIQ = 50.0   # market spread > 50 bps → ILLIQUID


# ─────────────────────────────────────────────────────────────────────────────
# Enums and Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class QuoteRegime(Enum):
    NORMAL   = "NORMAL"
    ADVERSE  = "ADVERSE"    # high VPIN — toxic flow likely
    TRENDING = "TRENDING"   # strong order book imbalance
    ILLIQUID = "ILLIQUID"   # market spread too wide — halt quoting


@dataclass
class QuoteResult:
    """Output of AvellanedaStoikovV2.compute_quotes_v2()."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    regime: QuoteRegime
    expected_pnl_per_fill_bps: float   # gross edge per round-trip in bps
    recommended_action: str            # "QUOTE", "WIDEN", "SKEW", "HALT"


# ─────────────────────────────────────────────────────────────────────────────
# Guéant-Lehalle intensity model — module-level pure functions
# ─────────────────────────────────────────────────────────────────────────────

def intensity_model(delta: float, k: float, A: float) -> float:
    """
    Poisson arrival intensity for limit orders at spread δ.

        λ(δ) = A · exp(−k · δ)

    Args:
        delta: half-spread in price units (positive)
        k:     order book depth parameter (decay rate of arrivals)
        A:     baseline arrival rate (orders per unit time)

    Returns:
        Arrival intensity λ ≥ 0.
    """
    if k <= 0 or A <= 0:
        return 0.0
    return float(A * math.exp(-k * delta))


def fill_probability(delta: float, k: float) -> float:
    """
    Probability that a limit order posted at spread δ is filled.

        P_fill(δ) = exp(−k · δ)

    Derived from the Poisson intensity model (unit-time horizon).

    Args:
        delta: half-spread in price units
        k:     depth parameter

    Returns:
        Probability in [0, 1].
    """
    if k <= 0 or delta < 0:
        return 1.0
    return float(np.clip(math.exp(-k * delta), 0.0, 1.0))


def optimal_spread_with_intensity(
    vol: float,
    gamma: float,
    k: float,
    A: float,
    T_remaining: float,
) -> float:
    """
    Closed-form optimal half-spread including Poisson arrival intensity.

    From Guéant-Lehalle-Fernandez-Tapia (2013), equation (7):

        δ*(t) = (1/k) · ln(1 + γ/k) + √(γσ²(T-t)/2 · (2k+γ)/(k(k+γ)))

    When A is small the spread widens to compensate for infrequent fills.
    An A-dependent correction term (1/k) · ln(1 + γ/k) is already implicit
    in the Avellaneda formula; the Guéant extension adds a fill-rate
    scaling via the 1/k factor and the √(…) volatility term.

    Args:
        vol:         instantaneous volatility σ (per second)
        gamma:       risk-aversion coefficient γ
        k:           depth parameter
        A:           baseline arrival rate
        T_remaining: fraction of session remaining ∈ (0, 1]

    Returns:
        Optimal half-spread in price units.
    """
    if k <= 0 or gamma <= 0 or T_remaining <= 0 or vol <= 0:
        return 0.0

    try:
        # Guéant-Lehalle closed form
        base_term = (1.0 / k) * math.log(1.0 + gamma / k)

        vol_term_inner = (
            gamma * vol ** 2 * T_remaining / 2.0
            * (2.0 * k + gamma) / (k * (k + gamma))
        )
        vol_term = math.sqrt(max(vol_term_inner, 0.0))

        half_spread = base_term + vol_term

        # A-scaling: lower arrival rate → wider spread (more risk per fill)
        if A > 0:
            a_scale = 1.0 + 0.5 * max(0.0, 1.0 - A)
            half_spread *= a_scale
    except (ValueError, ZeroDivisionError):
        half_spread = 0.001

    return max(half_spread, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Inventory risk — module-level pure functions
# ─────────────────────────────────────────────────────────────────────────────

def inventory_penalty(
    q: float,
    gamma: float,
    sigma: float,
    T_remaining: float,
) -> float:
    """
    Quadratic inventory penalty on the value function.

        ψ(q) = γ · σ² · (T - t) · q²

    Measures the utility cost of holding inventory q given risk aversion γ,
    volatility σ and remaining session time (T-t).

    Args:
        q:           current inventory (signed)
        gamma:       risk aversion
        sigma:       instantaneous volatility (per second)
        T_remaining: fraction of session remaining

    Returns:
        Penalty ≥ 0 in the same units as variance (price² / time).
    """
    return gamma * sigma ** 2 * T_remaining * q ** 2


def skew_quotes(
    mid: float,
    q: float,
    gamma: float,
    sigma: float,
    T_remaining: float,
    max_inv: float,
) -> Tuple[float, float]:
    """
    Compute asymmetric bid/ask to drive inventory toward zero.

    The shift in the reservation price equals:
        Δr = −q · γ · σ² · (T-t)

    We additionally reduce the half-spread on the contra side by the
    magnitude of the inventory-to-max ratio so the unwind side is posted
    more aggressively.

    Args:
        mid:         current mid price
        q:           signed inventory
        gamma:       risk aversion
        sigma:       volatility
        T_remaining: session fraction remaining
        max_inv:     maximum inventory limit

    Returns:
        (bid_adjustment, ask_adjustment) — signed shifts to add to the
        reservation bid and ask quotes respectively.
    """
    if max_inv == 0 or mid <= 0:
        return 0.0, 0.0

    inv_ratio = np.clip(q / max_inv, -1.0, 1.0)

    # Reservation price shift (A-S equation)
    r_shift = -q * gamma * sigma ** 2 * T_remaining

    # Extra aggressive skew when inventory is heavy
    aggressiveness = abs(inv_ratio) ** 1.5  # nonlinear: heavier skew when fuller

    if inv_ratio > 0:
        # Long inventory: lower ask (sell cheaper), keep bid flat
        bid_adj = r_shift
        ask_adj = r_shift - aggressiveness * mid * 0.0005
    elif inv_ratio < 0:
        # Short inventory: raise bid (buy more), keep ask flat
        bid_adj = r_shift + aggressiveness * mid * 0.0005
        ask_adj = r_shift
    else:
        bid_adj = ask_adj = 0.0

    return float(bid_adj), float(ask_adj)


def dynamic_max_inventory(base_max: float, session_elapsed_frac: float) -> float:
    """
    Reduce the maximum allowed inventory as session approaches end.

    At session start (frac=0): full limit applies.
    At 80% of session (frac=0.8): limit shrinks to 50% of base.
    At session end (frac=1.0): limit is 10% of base (nearly closed).

    Args:
        base_max:             full-session maximum inventory
        session_elapsed_frac: fraction of session elapsed ∈ [0, 1]

    Returns:
        Effective maximum inventory for this point in time.
    """
    frac = np.clip(session_elapsed_frac, 0.0, 1.0)
    # Exponential decay: limit * exp(-3 * elapsed_frac)  →  ~0.05 at end
    scale = math.exp(-3.0 * frac)
    scale = max(scale, 0.05)   # hard floor at 5%
    return base_max * scale


# ─────────────────────────────────────────────────────────────────────────────
# Regime detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_quote_regime(
    obi: float,
    vpin: float,
    spread_percentile: float,
) -> QuoteRegime:
    """
    Classify current market microstructure regime.

    Priority (highest first):
        1. ILLIQUID  — market spread > 50bps percentile
        2. ADVERSE   — VPIN > 0.7 (toxic, informed-flow dominated)
        3. TRENDING  — |OBI| > 0.4 (strong directional order imbalance)
        4. NORMAL    — everything else

    Args:
        obi:               order book imbalance ∈ [-1, 1]
                           (bid_qty − ask_qty) / (bid_qty + ask_qty)
        vpin:              volume-synchronised probability of informed trading
                           ∈ [0, 1]
        spread_percentile: current market spread expressed as bps percentile
                           (e.g. 75 = 75th percentile of recent spread dist.)

    Returns:
        QuoteRegime enum value.
    """
    if spread_percentile >= SPREAD_ILLIQ:
        return QuoteRegime.ILLIQUID
    if vpin >= VPIN_ADVERSE:
        return QuoteRegime.ADVERSE
    if abs(obi) >= OBI_TRENDING:
        return QuoteRegime.TRENDING
    return QuoteRegime.NORMAL


# ─────────────────────────────────────────────────────────────────────────────
# Online Bayesian update helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bayesian_update_k(
    k_prior: float,
    observed_spread: float,
    filled: bool,
    learning_rate: float = 0.05,
) -> float:
    """
    Simple gradient-step Bayesian-like update for the depth parameter k.

    If a fill occurred at spread δ: k should decrease (more liquidity than assumed).
    If no fill occurred at spread δ: k should increase (market thinner than assumed).

    Args:
        k_prior:         current k estimate
        observed_spread: half-spread at which the order was posted
        filled:          whether the order was filled
        learning_rate:   gradient step size

    Returns:
        Updated k estimate.
    """
    if observed_spread <= 0:
        return k_prior
    grad = observed_spread if not filled else -observed_spread
    k_new = k_prior + learning_rate * grad
    return float(np.clip(k_new, 0.1, 20.0))


def _bayesian_update_A(
    A_prior: float,
    elapsed_since_last_fill: float,
    learning_rate: float = 0.05,
) -> float:
    """
    Update the baseline arrival rate A from observed inter-fill time.

    If fill came quickly: A was underestimated → increase.
    If fill took long:   A was overestimated → decrease.

    Args:
        A_prior:                  current A estimate
        elapsed_since_last_fill:  seconds since prior fill
        learning_rate:            step size

    Returns:
        Updated A estimate (clamped > 0).
    """
    if elapsed_since_last_fill <= 0:
        return A_prior
    observed_rate = 1.0 / elapsed_since_last_fill
    A_new = A_prior + learning_rate * (observed_rate - A_prior)
    return float(np.clip(A_new, 1e-4, 1000.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main v2 class
# ─────────────────────────────────────────────────────────────────────────────

class AvellanedaStoikovV2(AvellanedaStoikovMM):
    """
    Enhanced Avellaneda-Stoikov market maker with Guéant-Lehalle intensity
    model, regime detection, and online parameter learning.

    All original AvellanedaStoikovMM methods are preserved and unchanged.
    New functionality is layered on top via compute_quotes_v2().
    """

    def __init__(
        self,
        symbol: str = "BTC/USD",
        gamma: float = GAMMA_DEFAULT,
        k: float = K_DEFAULT,
        A: float = A_DEFAULT,
        session_seconds: float = T_SESSION,
        max_inventory: float = MAX_INVENTORY,
        min_spread_bps: float = MIN_SPREAD_BPS,
        base_quote_size: float = 1.0,
        learning_rate: float = 0.05,
    ):
        super().__init__(
            symbol=symbol,
            gamma=gamma,
            k=k,
            session_seconds=session_seconds,
            max_inventory=max_inventory,
            min_spread_bps=min_spread_bps,
        )
        # Guéant-Lehalle intensity parameters (learned online)
        self.A: float = A
        self._k_estimate: float = k       # may diverge from self.k after learning
        self._A_estimate: float = A

        self.base_quote_size: float = base_quote_size
        self.learning_rate: float = learning_rate

        # Fill tracking for online learning
        self._last_fill_ts: float = time.time()
        self._v2_fills: List[Dict[str, Any]] = []

        # Regime distribution counter
        self._regime_counts: Counter = Counter()

        # Running spread tracker for summary
        self._spread_history: Deque[float] = deque(maxlen=500)

        # Per-session PnL in bps
        self._pnl_bps: List[float] = []

    # ── Regime-aware quote computation ────────────────────────────────────────

    def compute_quotes_v2(
        self,
        mid: float,
        inventory: float,
        obi_signal: float,
        vpin: float,
        spread_pct: float,
        session_elapsed_frac: float,
    ) -> Optional[QuoteResult]:
        """
        Compute optimal bid/ask using the full Guéant-Lehalle model with
        regime awareness and dynamic inventory limits.

        Args:
            mid:                   current mid price
            inventory:             current signed inventory (positive = long)
            obi_signal:            order book imbalance ∈ [-1, 1]
            vpin:                  VPIN ∈ [0, 1]
            spread_pct:            current market half-spread in bps
                                   (used as spread_percentile proxy)
            session_elapsed_frac:  fraction of session elapsed ∈ [0, 1]

        Returns:
            QuoteResult or None if quoting should be halted.
        """
        if mid <= 0:
            return None

        # Sync inventory state
        self.inventory = inventory
        self._mid_prices.append(mid)

        # ── Determine regime ────────────────────────────────────────────────
        regime = detect_quote_regime(obi_signal, vpin, spread_pct)
        self._regime_counts[regime.value] += 1

        if regime == QuoteRegime.ILLIQUID:
            return QuoteResult(
                bid_price=0.0, ask_price=0.0,
                bid_size=0.0, ask_size=0.0,
                regime=regime,
                expected_pnl_per_fill_bps=0.0,
                recommended_action="HALT",
            )

        # ── Dynamic inventory limit ──────────────────────────────────────────
        eff_max_inv = dynamic_max_inventory(self.max_inventory, session_elapsed_frac)
        if abs(inventory) >= eff_max_inv:
            side_to_close = "SELL" if inventory > 0 else "BUY"
            logger.warning(
                "MM v2 %s: inventory %.3f >= dynamic limit %.3f — need %s",
                self.symbol, inventory, eff_max_inv, side_to_close,
            )
            return QuoteResult(
                bid_price=0.0, ask_price=0.0,
                bid_size=0.0, ask_size=0.0,
                regime=regime,
                expected_pnl_per_fill_bps=0.0,
                recommended_action="HALT",
            )

        # ── Compute volatility and time-remaining ────────────────────────────
        sigma = self._estimate_sigma()
        T_remaining = max(1.0 - session_elapsed_frac, 0.001)

        # ── Guéant-Lehalle optimal half-spread ───────────────────────────────
        half_spread = optimal_spread_with_intensity(
            vol=sigma,
            gamma=self.gamma,
            k=self._k_estimate,
            A=self._A_estimate,
            T_remaining=T_remaining,
        )

        # Fallback to A-S formula if vol is zero
        if half_spread == 0.0:
            half_spread = self.optimal_spread(mid)

        # Clamp to [min, max] bps
        min_half = (self.min_spread_bps / 20000.0) * mid
        max_half = (MAX_SPREAD_BPS / 20000.0) * mid
        half_spread = float(np.clip(half_spread, min_half, max_half))

        # ── Inventory-aware reservation price and quote skew ─────────────────
        bid_adj, ask_adj = skew_quotes(
            mid=mid,
            q=inventory,
            gamma=self.gamma,
            sigma=sigma,
            T_remaining=T_remaining,
            max_inv=eff_max_inv,
        )
        r_price = mid + (bid_adj + ask_adj) / 2.0  # approximate reservation price

        # ── Apply regime-specific adjustments ────────────────────────────────
        bid_size = self.base_quote_size
        ask_size = self.base_quote_size
        action = "QUOTE"

        if regime == QuoteRegime.ADVERSE:
            # Widen spread 2× and reduce size 50%
            half_spread *= 2.0
            bid_size *= 0.5
            ask_size *= 0.5
            action = "WIDEN"

        elif regime == QuoteRegime.TRENDING:
            # Skew strongly in trend direction; reduce contra-side size
            action = "SKEW"
            if obi_signal > 0:
                # Buy pressure: raise ask, reduce ask size (don't sell cheap)
                half_spread_ask = half_spread * 1.5
                half_spread_bid = half_spread * 0.8
                ask_size *= 0.4
            else:
                # Sell pressure: lower bid, reduce bid size
                half_spread_bid = half_spread * 1.5
                half_spread_ask = half_spread * 0.8
                bid_size *= 0.4
        else:
            half_spread_bid = half_spread
            half_spread_ask = half_spread

        # Re-assign for non-TRENDING
        if regime != QuoteRegime.TRENDING:
            half_spread_bid = half_spread
            half_spread_ask = half_spread

        # ── Final quote prices ───────────────────────────────────────────────
        bid_quote = (r_price + bid_adj) - half_spread_bid
        ask_quote = (r_price + ask_adj) + half_spread_ask

        if bid_quote >= ask_quote or bid_quote <= 0:
            return None

        # ── Expected PnL per fill (gross edge in bps) ────────────────────────
        gross_spread_bps = (ask_quote - bid_quote) / mid * 10000.0
        p_fill_bid = fill_probability(half_spread_bid, self._k_estimate)
        p_fill_ask = fill_probability(half_spread_ask, self._k_estimate)
        expected_pnl_bps = gross_spread_bps * p_fill_bid * p_fill_ask

        self._spread_history.append(gross_spread_bps)
        self._pnl_bps.append(expected_pnl_bps)

        return QuoteResult(
            bid_price=round(bid_quote, 8),
            ask_price=round(ask_quote, 8),
            bid_size=round(bid_size, 6),
            ask_size=round(ask_size, 6),
            regime=regime,
            expected_pnl_per_fill_bps=round(expected_pnl_bps, 4),
            recommended_action=action,
        )

    # ── Online learning from fills ────────────────────────────────────────────

    def update_from_fill(
        self,
        side: str,
        price: float,
        size: float,
        fill_timestamp: float,
    ) -> None:
        """
        Learn from an actual fill — update k and A via online Bayesian step.

        Updates:
          - k estimate: filled at the posted spread → k was too high (market
            thinner in reality); converge toward observed fill spread.
          - A estimate: inter-fill time observed → update arrival rate.
          - Inventory and PnL via the parent record_fill() method.

        Args:
            side:           "buy" or "sell"
            price:          fill price
            size:           quantity filled
            fill_timestamp: Unix timestamp (seconds) of the fill event
        """
        # Observe inter-fill time for A update
        elapsed = fill_timestamp - self._last_fill_ts
        self._last_fill_ts = fill_timestamp

        # Infer observed spread from fill price vs last mid
        last_mid = self._last_mid if self._last_mid > 0 else price
        observed_half_spread = abs(price - last_mid)

        # Bayesian-like parameter updates
        self._k_estimate = _bayesian_update_k(
            self._k_estimate,
            observed_spread=observed_half_spread,
            filled=True,
            learning_rate=self.learning_rate,
        )
        self._A_estimate = _bayesian_update_A(
            self._A_estimate,
            elapsed_since_last_fill=max(elapsed, 0.001),
            learning_rate=self.learning_rate,
        )

        # Track fill metadata
        self._v2_fills.append({
            "side": side,
            "price": price,
            "size": size,
            "ts": fill_timestamp,
            "k_after": self._k_estimate,
            "A_after": self._A_estimate,
            "inter_fill_sec": elapsed,
        })

        # Delegate inventory/PnL tracking to parent
        self.record_fill(side=side, price=price, amount=size)

        logger.info(
            "MM v2 fill: %s %.6f @ %.2f | k=%.3f A=%.4f inter=%.1fs",
            side, size, price, self._k_estimate, self._A_estimate, elapsed,
        )

    # ── Session summary ───────────────────────────────────────────────────────

    def session_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the current/last session's activity.

        Returns:
            Dict containing fill statistics, PnL estimate, average spread,
            regime distribution, and learned parameters.
        """
        fills = self._v2_fills or self._fills
        n_fills = len(fills)
        avg_spread = float(np.mean(list(self._spread_history))) if self._spread_history else 0.0
        avg_pnl_bps = float(np.mean(self._pnl_bps)) if self._pnl_bps else 0.0

        # Regime distribution as percentage
        total_regime_ticks = sum(self._regime_counts.values()) or 1
        regime_dist = {
            k: round(v / total_regime_ticks * 100, 1)
            for k, v in self._regime_counts.items()
        }

        # Simple PnL estimate: sum of signed cash flows from fills
        pnl_estimate = self.total_pnl

        return {
            "symbol": self.symbol,
            "n_fills": n_fills,
            "pnl_estimate_usd": round(pnl_estimate, 4),
            "avg_spread_bps": round(avg_spread, 3),
            "avg_expected_pnl_per_fill_bps": round(avg_pnl_bps, 4),
            "regime_distribution_pct": regime_dist,
            "current_inventory": self.inventory,
            "k_learned": round(self._k_estimate, 4),
            "A_learned": round(self._A_estimate, 4),
            "session_t_remaining": round(self._time_remaining(), 3),
        }
