"""
Lightweight strategy library implementations.

The repo historically contained many strategy modules under:
- strategies/algorithmic/
- strategies/advanced/
- strategies/quantum_custom/

Some of those files were previously auto-stubbed to satisfy repo-wide compileability.
This module provides clean, dependency-light implementations and a consistent API
so those strategy modules can be restored as thin wrappers.

API contract for integration with `unified_ai_brain.PinnacleAIBrain`:
- Each strategy exposes an `analyze(market_data: dict) -> dict | None`
- Returned dict is normalized by `_normalize_pack_signal()`
"""

from __future__ import annotations
import logging

import logging

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

def _safe_price(md: Dict[str, Any]) -> float:
    try:
        return float(md.get("price") or 0.0)
    except Exception:
        return 0.0


def _safe_symbol(md: Dict[str, Any]) -> str:
    return str(md.get("symbol") or "BTC/USD")


def _get_ohlcv(md: Dict[str, Any]) -> Optional[pd.DataFrame]:
    df = md.get("ohlcv_df")
    if df is None:
        return None
    try:
        if hasattr(df, "empty") and df.empty:
            return None
        return df
    except Exception:
        return None


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(window=max(2, int(n))).mean()


def _zscore(x: pd.Series, n: int) -> pd.Series:
    w = max(5, int(n))
    mu = x.rolling(window=w).mean()
    sd = x.rolling(window=w).std().replace(0, np.nan)
    return ((x - mu) / sd).fillna(0.0)


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=max(2, int(n)), adjust=False).mean()


def _clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


@dataclass
class StrategyBaseLite:
    config: Dict[str, Any]
    name: str

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def _emit(self, *, symbol: str, action: str, confidence: float, price: float, source: str) -> Dict[str, Any]:
        return {
            "symbol": str(symbol),
            "action": str(action).upper(),
            "confidence": float(_clamp01(confidence)),
            "price": float(price) if price > 0 else 0.0,
            "source": str(source),
        }


class MomentumStrategy(StrategyBaseLite):
    """Simple momentum: SMA cross + recent return filter."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="momentum")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or "close" not in df.columns or len(df) < 60:
            return None
        close = df["close"].astype(float)
        fast = int(self.config.get("fast", 20) or 20)
        slow = int(self.config.get("slow", 50) or 50)
        ret_n = int(self.config.get("return_window", 20) or 20)

        sma_f = _sma(close, fast)
        sma_s = _sma(close, slow)
        if math.isnan(float(sma_f.iloc[-1])) or math.isnan(float(sma_s.iloc[-1])):
            return None

        r = float(close.iloc[-1] / max(float(close.iloc[-ret_n]), 1e-12) - 1.0) if len(close) > ret_n else 0.0
        sym = _safe_symbol(market_data)
        px = float(close.iloc[-1])

        if float(sma_f.iloc[-1]) > float(sma_s.iloc[-1]) and r > 0:
            conf = 0.55 + min(0.35, abs(r) * 5.0)
            return self._emit(symbol=sym, action="BUY", confidence=conf, price=px, source="algorithmic_momentum")
        if float(sma_f.iloc[-1]) < float(sma_s.iloc[-1]) and r < 0:
            conf = 0.55 + min(0.35, abs(r) * 5.0)
            return self._emit(symbol=sym, action="SELL", confidence=conf, price=px, source="algorithmic_momentum")
        return None


class MeanReversionStrategy(StrategyBaseLite):
    """Mean reversion: z-score vs rolling mean."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="mean_reversion")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or "close" not in df.columns or len(df) < 80:
            return None
        close = df["close"].astype(float)
        window = int(self.config.get("window", 50) or 50)
        entry_z = float(self.config.get("entry_z", 2.0) or 2.0)
        z = _zscore(close, window)
        last_z = float(z.iloc[-1])
        sym = _safe_symbol(market_data)
        px = float(close.iloc[-1])

        if last_z <= -abs(entry_z):
            conf = 0.55 + min(0.35, abs(last_z) / 5.0)
            return self._emit(symbol=sym, action="BUY", confidence=conf, price=px, source="algorithmic_mean_reversion")
        if last_z >= abs(entry_z):
            conf = 0.55 + min(0.35, abs(last_z) / 5.0)
            return self._emit(symbol=sym, action="SELL", confidence=conf, price=px, source="algorithmic_mean_reversion")
        return None


class TrendFollowingStrategy(StrategyBaseLite):
    """Trend following: EMA cross."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="trend_following")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or "close" not in df.columns or len(df) < 80:
            return None
        close = df["close"].astype(float)
        fast = int(self.config.get("fast", 20) or 20)
        slow = int(self.config.get("slow", 60) or 60)
        ef = _ema(close, fast)
        es = _ema(close, slow)
        sym = _safe_symbol(market_data)
        px = float(close.iloc[-1])

        if float(ef.iloc[-1]) > float(es.iloc[-1]):
            conf = 0.55 + min(0.25, float((ef.iloc[-1] - es.iloc[-1]) / max(px, 1e-9)))
            return self._emit(symbol=sym, action="BUY", confidence=conf, price=px, source="algorithmic_trend_following")
        if float(ef.iloc[-1]) < float(es.iloc[-1]):
            conf = 0.55 + min(0.25, float((es.iloc[-1] - ef.iloc[-1]) / max(px, 1e-9)))
            return self._emit(symbol=sym, action="SELL", confidence=conf, price=px, source="algorithmic_trend_following")
        return None


class PairsTradingStrategy(StrategyBaseLite):
    """
    Simple pairs heuristic.
    Expects config:
      - pair: "BTC/USD,ETH/USD"
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="pairs_trading")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tickers = market_data.get("tickers") or {}
        pair = str(self.config.get("pair", "") or "")
        if "," not in pair:
            return None
        a, b = [x.strip() for x in pair.split(",", 1)]
        try:
            pa = float(tickers.get(a, 0.0))
            pb = float(tickers.get(b, 0.0))
        except Exception:
            return None
        if pa <= 0 or pb <= 0:
            return None
        ratio = pa / pb
        # Track ratio mean via config-less rolling not available; use soft band around 1.0.
        # This is intentionally conservative: only emit on extreme dislocations.
        if ratio > 1.10:
            return self._emit(symbol=a, action="SELL", confidence=0.65, price=pa, source="algorithmic_pairs_trading")
        if ratio < 0.90:
            return self._emit(symbol=a, action="BUY", confidence=0.65, price=pa, source="algorithmic_pairs_trading")
        return None


class MarketMakingStrategy(StrategyBaseLite):
    """
    Active Market Making Strategy.
    Places LIMIT orders on both sides of the spread to capture the spread + rebate.
    Uses volatility to widen/tighten spreads dynamically.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="market_making")
        self.spread_bps = float(self.config.get("spread_bps", 20) or 20) / 10000 # 0.2% default
        self.order_refresh_time = 60 # seconds

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 1. Get current mid-price
        df = _get_ohlcv(market_data)
        if df is None or df.empty:
            return None
            
        last_close = float(df["close"].iloc[-1])
        
        # 2. Calculate dynamic spread based on volatility (ATR)
        atr_series = _atr(df, 14)
        if not atr_series.empty:
            current_atr = float(atr_series.iloc[-1])
            # If volatility is high, widen spread to protect against adverse flow
            vol_adjustment = (current_atr / last_close) * 0.5 
            final_spread = max(self.spread_bps, vol_adjustment)
        else:
            final_spread = self.spread_bps

        # 3. Generate Signals
        # Market Makers emit BOTH buy and sell signals at different levels
        # The executor handles placing them as Limit orders
        
        buy_price = last_close * (1 - final_spread)
        sell_price = last_close * (1 + final_spread)
        
        sym = _safe_symbol(market_data)
        
        # We return a compound signal (the Router needs to handle list, or we return "PROVIDE_LIQUIDITY")
        # For compatibility with StrategyBaseLite, we emit a special action
        
        return {
            "symbol": sym,
            "action": "MARKET_MAKE",
            "buy_limit": buy_price,
            "sell_limit": sell_price,
            "confidence": 0.9, # MM is always active
            "price": last_close,
            "source": "market_making_dynamic"
        }


class ArbitrageStrategy(StrategyBaseLite):
    """
    Arbitrage placeholder.
    True arb requires multi-venue quotes; in unified paper/backtest we emit none.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="arbitrage")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None


class RegimeSwitchingStrategy(StrategyBaseLite):
    """Switch between trend vs mean reversion based on volatility."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="regime_switching")
        self._trend = TrendFollowingStrategy(self.config.get("trend", {}) if isinstance(self.config.get("trend"), dict) else {})
        self._mr = MeanReversionStrategy(self.config.get("mean_reversion", {}) if isinstance(self.config.get("mean_reversion"), dict) else {})

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or "close" not in df.columns or len(df) < 80:
            return None
        close = df["close"].astype(float)
        rets = close.pct_change().dropna()
        vol = float(rets.tail(50).std() or 0.0)
        vol_thresh = float(self.config.get("vol_threshold", 0.01) or 0.01)
        if vol >= vol_thresh:
            return self._mr.analyze(market_data)
        return self._trend.analyze(market_data)


class FactorInvestingStrategy(StrategyBaseLite):
    """
    Factor investing placeholder.
    True factor models require cross-asset data; in per-symbol callback we emit none.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="factor_investing")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None


class StatArbStrategy(StrategyBaseLite):
    """Simple stat arb = mean reversion wrapper."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="stat_arb")
        self._mr = MeanReversionStrategy(self.config)

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sig = self._mr.analyze(market_data)
        if sig:
            sig["source"] = "advanced_stat_arb"
        return sig


class CrossExchangeArbStrategy(StrategyBaseLite):
    """Cross-exchange arb placeholder (requires multiple venue quotes)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="cross_exchange_arb")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None


# ════════════════════════════════════════════════════════════════════════════
# Real quantum strategies (Phase G5).
#
# The previous "quantum_*_elite" strategies were thin wrappers around classical
# logic with marketing names. These are now real quantum signal generators
# that route through quantum/algorithms/* and emit signals based on the
# quantum output.
#
# Each strategy keeps its original name and constructor signature so the
# strategy registry doesn't need updates.
# ════════════════════════════════════════════════════════════════════════════


class QuantumMomentumEliteStrategy(StrategyBaseLite):
    """
    Quantum momentum strategy: VQE on the recent returns covariance matrix
    extracts the dominant momentum eigenmode. The sign of the projection of
    current returns onto this eigenvector determines BUY/SELL.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="quantum_momentum_elite")
        self._base = MomentumStrategy(self.config.get("base", {}) if isinstance(self.config.get("base"), dict) else {})

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or "close" not in df.columns or len(df) < 30:
            # Fall back to classical when not enough data
            sig = self._base.analyze(market_data)
            if not sig:
                return None
            sig["source"] = "quantum_momentum_elite_fallback"
            return sig
        try:
            import numpy as _np
            from quantum.algorithms.vqe import VQESolver

            close = df["close"].astype(float).values[-30:]
            returns = _np.diff(_np.log(close + 1e-10))
            # Build a tiny 2x2 Pauli Hamiltonian whose ground state encodes
            # the recent returns mean direction.
            mean_ret = float(_np.mean(returns))
            std_ret = float(_np.std(returns)) or 1e-6
            # H = -mean Z + std X (anti-ferromagnet captures sign+volatility)
            pauli_terms = [("Z", -mean_ret * 100.0), ("X", std_ret * 50.0)]
            solver = VQESolver(n_qubits=1, n_layers=2)
            result = solver.solve_hamiltonian(pauli_terms, max_iter=20, shots=512, n_restarts=1)
            # Use the ground bit to determine direction
            sym = _safe_symbol(market_data)
            px = float(close[-1])
            if result["ground_state_bits"][0] == 0 and mean_ret > 0:
                return self._emit(
                    symbol=sym, action="BUY", confidence=0.65,
                    price=px, source="quantum_momentum_elite",
                )
            elif result["ground_state_bits"][0] == 1 and mean_ret < 0:
                return self._emit(
                    symbol=sym, action="SELL", confidence=0.65,
                    price=px, source="quantum_momentum_elite",
                )
            return None
        except Exception as exc:
            logger.debug("QuantumMomentumElite quantum path failed: %s", exc)
            sig = self._base.analyze(market_data)
            if not sig:
                return None
            sig["source"] = "quantum_momentum_elite_fallback"
            return sig


class QuantumMeanReversionEliteStrategy(StrategyBaseLite):
    """
    Quantum mean-reversion strategy: QFT on the recent return series gives
    a spectral density. Signals fire when the dominant frequency suggests
    we're at a mean-reversion turning point.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="quantum_mean_reversion_elite")
        self._base = MeanReversionStrategy(self.config.get("base", {}) if isinstance(self.config.get("base"), dict) else {})

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or "close" not in df.columns or len(df) < 16:
            sig = self._base.analyze(market_data)
            if not sig:
                return None
            sig["source"] = "quantum_mean_reversion_elite_fallback"
            return sig
        try:
            import numpy as _np
            from quantum.algorithms.qft import qft_matrix

            close = df["close"].astype(float).values[-16:]
            mean = float(_np.mean(close))
            centered = close - mean
            # Pad/truncate to power of 2
            n_qubits = 4
            dim = 1 << n_qubits
            signal = _np.zeros(dim, dtype=complex)
            signal[: min(dim, len(centered))] = centered[: dim]
            # Normalize
            norm = float(_np.linalg.norm(signal))
            if norm > 1e-9:
                signal = signal / norm
            # Apply QFT
            M = qft_matrix(n_qubits)
            spectrum = M @ signal
            power = _np.abs(spectrum) ** 2
            # Dominant frequency (skip DC)
            dom_freq = int(_np.argmax(power[1:]) + 1)
            # Mean-reversion: BUY if oversold (current below mean) and dom freq high
            sym = _safe_symbol(market_data)
            px = float(close[-1])
            if px < mean * 0.99 and dom_freq > 2:
                return self._emit(
                    symbol=sym, action="BUY", confidence=0.60,
                    price=px, source="quantum_mean_reversion_elite",
                )
            if px > mean * 1.01 and dom_freq > 2:
                return self._emit(
                    symbol=sym, action="SELL", confidence=0.60,
                    price=px, source="quantum_mean_reversion_elite",
                )
            return None
        except Exception as exc:
            logger.debug("QuantumMeanReversionElite quantum path failed: %s", exc)
            sig = self._base.analyze(market_data)
            if not sig:
                return None
            sig["source"] = "quantum_mean_reversion_elite_fallback"
            return sig


class QuantumTrendFollowingEliteStrategy(StrategyBaseLite):
    """
    Quantum trend following: Grover search over candidate lookback windows
    for the strongest trend strength.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="quantum_trend_following_elite")
        self._base = TrendFollowingStrategy(self.config.get("base", {}) if isinstance(self.config.get("base"), dict) else {})

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or "close" not in df.columns or len(df) < 60:
            sig = self._base.analyze(market_data)
            if not sig:
                return None
            sig["source"] = "quantum_trend_following_elite_fallback"
            return sig
        try:
            import numpy as _np
            from quantum.algorithms.grover import GroverSearch

            close = df["close"].astype(float).values
            # Score each lookback in {5, 10, 15, 20, 25, 30, 40, 60}
            lookbacks = [5, 10, 15, 20, 25, 30, 40, 60]
            scores = []
            for lb in lookbacks:
                if len(close) < lb:
                    scores.append(0.0)
                    continue
                window = close[-lb:]
                slope = float(_np.polyfit(range(lb), window, 1)[0])
                scores.append(abs(slope) / max(float(_np.std(window)), 1e-6))
            # Use Grover to find indices with score > threshold
            threshold = float(_np.median(scores))
            grover = GroverSearch(n_qubits=3)  # 8 lookbacks → 3 qubits
            result = grover.search(
                lambda i: i < len(scores) and scores[i] > threshold,
                n_items=8,
                n_solutions=max(1, sum(1 for s in scores if s > threshold)),
            )
            best_idx = max(result["found_indices"], key=lambda i: scores[i]) if result["found_indices"] else int(_np.argmax(scores))
            best_lb = lookbacks[best_idx]
            window = close[-best_lb:]
            slope = float(_np.polyfit(range(best_lb), window, 1)[0])
            sym = _safe_symbol(market_data)
            px = float(close[-1])
            if slope > 0 and scores[best_idx] > 0.1:
                return self._emit(
                    symbol=sym, action="BUY", confidence=min(0.85, 0.55 + scores[best_idx] / 5.0),
                    price=px, source=f"quantum_trend_following_elite_lb{best_lb}",
                )
            if slope < 0 and scores[best_idx] > 0.1:
                return self._emit(
                    symbol=sym, action="SELL", confidence=min(0.85, 0.55 + scores[best_idx] / 5.0),
                    price=px, source=f"quantum_trend_following_elite_lb{best_lb}",
                )
            return None
        except Exception as exc:
            logger.debug("QuantumTrendFollowingElite quantum path failed: %s", exc)
            sig = self._base.analyze(market_data)
            if not sig:
                return None
            sig["source"] = "quantum_trend_following_elite_fallback"
            return sig


class QuantumBreakoutEliteStrategy(StrategyBaseLite):
    """
    Quantum breakout: Quantum counting estimates how many recent bars are
    above/below the breakout threshold. High count → strong breakout.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="quantum_breakout_elite")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or "close" not in df.columns or len(df) < 16:
            return None
        try:
            import numpy as _np
            from quantum.algorithms.quantum_counting import quantum_counting

            close = df["close"].astype(float).values[-16:]
            lookback = 16
            px = float(close[-1])
            hi = float(_np.max(close[:-1]))
            lo = float(_np.min(close[:-1]))
            # Use quantum counting to count bars above 95th percentile
            high_threshold = float(_np.percentile(close[:-1], 95))
            low_threshold = float(_np.percentile(close[:-1], 5))

            n_qubits = 4  # 16 bars
            count_high = quantum_counting(
                lambda i: i < len(close) and close[i] >= high_threshold,
                n_search_qubits=n_qubits, n_count_qubits=4,
            )
            count_low = quantum_counting(
                lambda i: i < len(close) and close[i] <= low_threshold,
                n_search_qubits=n_qubits, n_count_qubits=4,
            )

            sym = _safe_symbol(market_data)
            if px >= hi and count_high["count_estimate"] >= 1:
                conf = 0.55 + min(0.30, count_high["fraction_estimate"])
                return self._emit(
                    symbol=sym, action="BUY", confidence=conf,
                    price=px, source="quantum_breakout_elite",
                )
            if px <= lo and count_low["count_estimate"] >= 1:
                conf = 0.55 + min(0.30, count_low["fraction_estimate"])
                return self._emit(
                    symbol=sym, action="SELL", confidence=conf,
                    price=px, source="quantum_breakout_elite",
                )
            return None
        except Exception as exc:
            logger.debug("QuantumBreakoutElite quantum path failed: %s", exc)
            return None


class QuantumPortfolioRotationEliteStrategy(StrategyBaseLite):
    """
    Quantum portfolio rotation: emits signals based on the QAOA-optimized
    portfolio weights computed in component_registry.on_cycle. Reads from
    advisory["quantum_portfolio"]["weights_by_symbol"] when available.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="quantum_portfolio_rotation_elite")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Look for the quantum_portfolio advisory written by ComponentRegistry
        try:
            advisory = market_data.get("advisory", {}) if isinstance(market_data, dict) else {}
            qp = advisory.get("quantum_portfolio") if isinstance(advisory, dict) else None
            if not qp or not isinstance(qp, dict):
                return None
            weights = qp.get("weights_by_symbol", {})
            if not weights:
                return None
            sym = _safe_symbol(market_data)
            if sym not in weights:
                return None
            # Use the QAOA weight as a confidence signal: high weight → BUY
            n_assets = qp.get("n_assets", len(weights))
            equal_weight = 1.0 / max(n_assets, 1)
            w = float(weights[sym])
            if w > equal_weight * 1.5:
                df = _get_ohlcv(market_data)
                if df is not None and "close" in df.columns and len(df) > 0:
                    px = float(df["close"].iloc[-1])
                    return self._emit(
                        symbol=sym, action="BUY",
                        confidence=min(0.85, 0.55 + (w - equal_weight)),
                        price=px, source="quantum_portfolio_rotation_elite",
                    )
            return None
        except Exception as exc:
            logger.debug("QuantumPortfolioRotationElite failed: %s", exc)
            return None


class QuantumArbitrageEliteStrategy(StrategyBaseLite):
    """
    Quantum arbitrage: thin wrapper around the Grover-driven arbitrage
    searcher (strategies/quantum_arb_search.py). Reads multi-venue prices
    from market_data when available.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="quantum_arbitrage_elite")
        self._searcher = None

    def _ensure_searcher(self):
        if self._searcher is None:
            from strategies.quantum_arb_search import QuantumArbSearcher
            self._searcher = QuantumArbSearcher(
                threshold_multiplier=1.2,
                min_edge_bps=3.0,
            )
        return self._searcher

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            from strategies.quantum_arb_search import VenuePrice
            # Multi-venue prices may be in market_data["venues"]
            venues = market_data.get("venues") if isinstance(market_data, dict) else None
            if not venues or not isinstance(venues, dict) or len(venues) < 2:
                return None
            prices_by_venue: Dict[str, Dict[str, VenuePrice]] = {}
            for venue_name, venue_data in venues.items():
                quotes = {}
                for sym, sym_data in (venue_data or {}).items():
                    if isinstance(sym_data, dict) and "bid" in sym_data and "ask" in sym_data:
                        quotes[sym] = VenuePrice(
                            venue=venue_name,
                            symbol=sym,
                            bid=float(sym_data["bid"]),
                            ask=float(sym_data["ask"]),
                            fee_bps=float(sym_data.get("fee_bps", 5.0)),
                        )
                if quotes:
                    prices_by_venue[venue_name] = quotes

            if len(prices_by_venue) < 2:
                return None

            searcher = self._ensure_searcher()
            signals = searcher.find_opportunities(prices_by_venue)
            if not signals:
                return None
            top = signals[0]
            return self._emit(
                symbol=top.symbol, action="BUY",
                confidence=top.confidence,
                price=top.metadata.get("buy_price", 0.0),
                source="quantum_arbitrage_elite",
                metadata={
                    "venue_buy": top.venue_buy,
                    "venue_sell": top.venue_sell,
                    "edge_bps": top.expected_edge_bps,
                },
            )
        except Exception as exc:
            logger.debug("QuantumArbitrageElite failed: %s", exc)
            return None


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if not {"high", "low", "close"}.issubset(set(df.columns)):
        return pd.Series(dtype=float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(window=max(3, int(n))).mean().fillna(0.0)


class CandlestickPatternStrategy(StrategyBaseLite):
    """
    Dependency-light candlestick pattern signals.

    Detects a small set of high-signal patterns:
    - bullish/bearish engulfing
    - hammer / shooting star
    - doji (no signal, used to reduce confidence)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="candlestick_patterns")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or len(df) < 5:
            return None
        if not {"open", "high", "low", "close"}.issubset(set(df.columns)):
            return None

        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)

        o1, c1 = float(o.iloc[-2]), float(c.iloc[-2])
        o2, c2 = float(o.iloc[-1]), float(c.iloc[-1])
        h2, l2 = float(h.iloc[-1]), float(l.iloc[-1])
        px = float(c2)
        sym = _safe_symbol(market_data)

        body = abs(c2 - o2)
        rng = max(1e-12, h2 - l2)
        upper = h2 - max(o2, c2)
        lower = min(o2, c2) - l2

        doji = body / rng <= float(self.config.get("doji_body_ratio", 0.10) or 0.10)

        prev_bear = c1 < o1
        prev_bull = c1 > o1
        curr_bull = c2 > o2
        curr_bear = c2 < o2

        # Engulfing
        bullish_engulf = prev_bear and curr_bull and (c2 >= o1) and (o2 <= c1)
        bearish_engulf = prev_bull and curr_bear and (o2 >= c1) and (c2 <= o1)

        # Hammer / shooting star (simple shape heuristic)
        hammer = (lower / rng >= float(self.config.get("hammer_lower_ratio", 0.55) or 0.55)) and (upper / rng <= 0.20)
        shooting_star = (upper / rng >= float(self.config.get("star_upper_ratio", 0.55) or 0.55)) and (lower / rng <= 0.20)

        if doji:
            return None

        if bullish_engulf or hammer:
            conf = 0.60 + min(0.30, float(body / rng))
            src = "candlestick_bullish_engulf" if bullish_engulf else "candlestick_hammer"
            return self._emit(symbol=sym, action="BUY", confidence=conf, price=px, source=src)

        if bearish_engulf or shooting_star:
            conf = 0.60 + min(0.30, float(body / rng))
            src = "candlestick_bearish_engulf" if bearish_engulf else "candlestick_shooting_star"
            return self._emit(symbol=sym, action="SELL", confidence=conf, price=px, source=src)

        return None


class HighFreqGridStrategy(StrategyBaseLite):
    """
    Grid-style mean-reversion strategy (OHLCV-only approximation).

    In real HFT/grid trading you'd place multiple resting orders and manage inventory.
    Here, we approximate the *decision logic* using deviation from a rolling mean.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config or {}, name="high_freq_grid")

    def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _get_ohlcv(market_data)
        if df is None or "close" not in df.columns or len(df) < 80:
            return None
        close = df["close"].astype(float)
        px = float(close.iloc[-1])
        sym = _safe_symbol(market_data)

        window = int(self.config.get("window", 50) or 50)
        band = float(self.config.get("grid_spacing_pct", 0.003) or 0.003)
        mean = _sma(close, window)
        m = float(mean.iloc[-1]) if len(mean) else 0.0
        if m <= 0:
            return None

        dev = (px - m) / m
        if abs(dev) < band:
            return None

        # Optional "sideways" filter using ATR / price
        atr_n = int(self.config.get("atr_window", 14) or 14)
        atr = _atr(df, atr_n)
        atrp = float(atr.iloc[-1] / max(px, 1e-9)) if len(atr) else 0.0
        max_atr_pct = float(self.config.get("max_atr_pct", 0.03) or 0.03)
        if atrp > max_atr_pct:
            return None

        conf = 0.55 + min(0.35, abs(dev) / max(band, 1e-9) * 0.10)
        if dev <= -band:
            return self._emit(symbol=sym, action="BUY", confidence=conf, price=px, source="grid_mean_reversion")
        return self._emit(symbol=sym, action="SELL", confidence=conf, price=px, source="grid_mean_reversion")


def get_library_strategies_for_names(enabled_names: List[str]) -> Dict[str, Any]:
    """
    Return a dict name -> strategy instance for each enabled name.
    Used by the continuous scanner to run strategy library as an optional signal source.
    """
    from strategies import tier_strategies_impl

    _logger = logging.getLogger(__name__)

    allowed = set((x or "").strip().lower() for x in enabled_names if x)
    if not allowed or "__all__" in allowed:
        allowed = {
            "momentum", "mean_reversion", "trend_following", "candlestick_patterns", "high_freq_grid",
            "regime_switching", "stat_arb", "quantum_momentum_elite", "quantum_mean_reversion_elite",
            "quantum_trend_following_elite", "quantum_breakout_elite",
            "absolute_tier", "akashic_tier", "apeiron_tier", "chronos_tier", "omega_tier",
            "paradox_tier", "singularity_tier", "source_tier", "thanatos_tier", "void_tier",
        }
    out: Dict[str, Any] = {}

    # Algorithmic + quantum custom (this module)
    for name, cls in [
        ("momentum", MomentumStrategy),
        ("mean_reversion", MeanReversionStrategy),
        ("trend_following", TrendFollowingStrategy),
        ("candlestick_patterns", CandlestickPatternStrategy),
        ("high_freq_grid", HighFreqGridStrategy),
        ("regime_switching", RegimeSwitchingStrategy),
        ("stat_arb", StatArbStrategy),
        ("quantum_momentum_elite", QuantumMomentumEliteStrategy),
        ("quantum_mean_reversion_elite", QuantumMeanReversionEliteStrategy),
        ("quantum_trend_following_elite", QuantumTrendFollowingEliteStrategy),
        ("quantum_breakout_elite", QuantumBreakoutEliteStrategy),
    ]:
        if name.lower() in allowed:
            try:
                out[name] = cls({})
            except Exception as _e:
                logger.debug("strategy_library_impl error: %s", _e)

    # Tier ensembles
    _tier = tier_strategies_impl
    for name, cls in [
        ("absolute_tier", _tier.AbsoluteTierStrategy),
        ("akashic_tier", _tier.AkashicTierStrategy),
        ("apeiron_tier", _tier.ApeironTierStrategy),
        ("chronos_tier", _tier.ChronosTierStrategy),
        ("omega_tier", _tier.OmegaTierStrategy),
        ("paradox_tier", _tier.ParadoxTierStrategy),
        ("singularity_tier", _tier.SingularityTierStrategy),
        ("source_tier", _tier.SourceTierStrategy),
        ("thanatos_tier", _tier.ThanatosTierStrategy),
        ("void_tier", _tier.VoidTierStrategy),
    ]:
        if name.lower() in allowed:
            try:
                out[name] = cls({})
            except Exception as _e:
                logger.debug("strategy_library_impl error: %s", _e)

    return out

