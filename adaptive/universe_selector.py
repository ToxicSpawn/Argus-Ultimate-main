from __future__ import annotations
import logging

import logging

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

@dataclass
class SymbolStats:
    trades: int = 0
    wins: int = 0
    pnl_ema: float = 0.0  # pct
    pnl2_ema: float = 0.0  # pct^2

    def update(self, pnl_pct: float, *, alpha: float = 0.12) -> None:
        self.trades += 1
        if pnl_pct >= 0:
            self.wins += 1
        a = float(alpha)
        self.pnl_ema = (1.0 - a) * float(self.pnl_ema) + a * float(pnl_pct)
        self.pnl2_ema = (1.0 - a) * float(self.pnl2_ema) + a * float(pnl_pct) * float(pnl_pct)

    def win_rate(self) -> float:
        return float(self.wins / max(1, self.trades))

    def std_proxy(self) -> float:
        v = max(0.0, float(self.pnl2_ema) - float(self.pnl_ema) * float(self.pnl_ema))
        return float(math.sqrt(v))

    def sharpe_proxy(self, risk_free_pct: float = 0.0) -> float:
        """Approx Sharpe: (mean - rf) / std; use 0 when std too small."""
        std = self.std_proxy()
        if std < 1e-6:
            return 0.0
        return float((float(self.pnl_ema) - risk_free_pct) / std)

    def sortino_proxy(self, risk_free_pct: float = 0.0) -> float:
        """Downside deviation proxy: use std of losses only; here we use std_proxy as proxy."""
        std = self.std_proxy()
        if std < 1e-6:
            return 0.0
        return float((float(self.pnl_ema) - risk_free_pct) / std)


class AdaptiveUniverseSelector:
    """
    Adaptive universe selection with promotion/demotion guards.

    - Tracks realized PnL per symbol via EMA.
    - Chooses a stable active subset of symbols from a candidate pool.
    - Uses UCB-ish scoring to allow exploration while exploiting winners.
    """

    def __init__(
        self,
        *,
        persist_path: str = "data/adaptive_universe.json",
        max_active: int = 5,
        min_trades_before_rank: int = 3,
        min_hold_cycles: int = 20,
        ema_alpha: float = 0.12,
        exploration_c: float = 1.0,
        min_liquidity: float = 0.0,
        max_vol_proxy: float = 999.0,
        liquidity_vol_provider: Any = None,
        correlation_provider: Any = None,
    ) -> None:
        self.persist_path = str(persist_path)
        self.max_active = int(max_active)
        self.min_trades_before_rank = int(min_trades_before_rank)
        self.min_hold_cycles = int(min_hold_cycles)
        self.ema_alpha = float(ema_alpha)
        self.exploration_c = float(exploration_c)
        self.min_liquidity = float(min_liquidity)
        self.max_vol_proxy = float(max_vol_proxy) if max_vol_proxy is not None else 999.0
        self._liquidity_vol_provider = liquidity_vol_provider
        self._correlation_provider = correlation_provider
        self._event_calendar: Any = None  # optional EventCalendar for event-driven exclude/include

        self.stats: Dict[str, SymbolStats] = {}
        self.active: List[str] = []
        self._hold_until_cycle: Dict[str, int] = {}
        self._last_effective_max_active: int = max(1, int(max_active))
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            p = Path(self.persist_path)
            if not p.exists():
                return
            d = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                return
            self.active = [str(x) for x in (d.get("active") or []) if str(x).strip()]
            st = d.get("stats") or {}
            if isinstance(st, dict):
                for k, v in st.items():
                    if not isinstance(v, dict):
                        continue
                    self.stats[str(k)] = SymbolStats(
                        trades=int(v.get("trades", 0) or 0),
                        wins=int(v.get("wins", 0) or 0),
                        pnl_ema=float(v.get("pnl_ema", 0.0) or 0.0),
                        pnl2_ema=float(v.get("pnl2_ema", 0.0) or 0.0),
                    )
        except Exception:
            return

    def save(self) -> None:
        try:
            p = Path(self.persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "active": list(self.active),
                "stats": {
                    k: {"trades": v.trades, "wins": v.wins, "pnl_ema": v.pnl_ema, "pnl2_ema": v.pnl2_ema}
                    for k, v in self.stats.items()
                },
            }
            p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            return

    def observe_trade_close(self, *, symbol: str, pnl_pct: float) -> None:
        self.load()
        sym = str(symbol or "").strip()
        if not sym:
            return
        s = self.stats.get(sym) or SymbolStats()
        s.update(float(pnl_pct), alpha=float(self.ema_alpha))
        self.stats[sym] = s

    def _score(self, sym: str) -> float:
        s = self.stats.get(sym)
        if s is None:
            return 0.0
        if s.trades < self.min_trades_before_rank:
            # small positive bias to allow exploration
            return 0.05
        pnl_term = math.tanh(float(s.pnl_ema) / 1.5)
        win_term = (float(s.win_rate()) - 0.5) * 2.0
        vol = float(s.std_proxy())
        vol_term = 1.0 / (1.0 + vol / 2.0) if s.trades >= 10 else 1.0
        # Sharpe/Sortino: prefer risk-adjusted return
        sharpe = s.sharpe_proxy(0.0)
        sharpe_term = math.tanh(sharpe / 2.0)  # cap influence
        raw = (0.5 * pnl_term + 0.25 * win_term + 0.25 * sharpe_term) * vol_term

        total = 1 + sum(int(v.trades) for v in self.stats.values())
        bonus = float(self.exploration_c) * math.sqrt(math.log(float(total)) / max(1.0, float(s.trades)))
        return float(raw + 0.10 * bonus)

    def select_active(
        self,
        *,
        candidate_symbols: List[str],
        cycle_id: int,
        risk_scale: float = 1.0,
        symbol_scores: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Pick the active universe for the next cycle.
        """
        self.load()
        cands = [str(x) for x in candidate_symbols if str(x).strip()]
        if not cands:
            return list(self.active) if self.active else []
        score_overrides = dict(symbol_scores or {})
        clamped_risk_scale = max(0.0, min(1.0, float(risk_scale or 0.0)))
        effective_max_active = max(1, min(int(self.max_active), int(round(float(self.max_active) * clamped_risk_scale)) or 1))
        self._last_effective_max_active = int(effective_max_active)

        # Optional event-driven: exclude symbols in event window, add force-include
        cal = getattr(self, "_event_calendar", None)
        if cal is not None:
            try:
                exclude = set()
                if callable(getattr(cal, "get_exclude_symbols", None)):
                    exclude = cal.get_exclude_symbols()
                cands = [s for s in cands if s not in exclude]
                if callable(getattr(cal, "get_include_symbols", None)):
                    include = cal.get_include_symbols()
                    for sym in include:
                        if sym not in cands:
                            cands.append(sym)
            except Exception as _e:
                logger.debug("universe_selector error: %s", _e)

        # Optional liquidity/vol filters (dynamic by regime if provider gives per-symbol data)
        p = getattr(self, "_liquidity_vol_provider", None)
        if p is not None and (self.min_liquidity > 0 or (self.max_vol_proxy or 999) < 999):
            filtered = []
            for s in cands:
                try:
                    liq, vol = (0.0, 999.0)
                    if callable(getattr(p, "get", None)):
                        liq = float(p.get(s, {}).get("liquidity", 0) or 0)
                        vol = float(p.get(s, {}).get("vol", 999) or 999)
                    elif callable(p):
                        liq, vol = p(s)
                    if (liq >= self.min_liquidity) and (vol <= (self.max_vol_proxy or 999)):
                        filtered.append(s)
                except Exception:
                    filtered.append(s)
            cands = filtered if filtered else cands

        def _combined_score(sym: str) -> float:
            base = float(self._score(sym))
            bonus = max(-1.0, min(1.0, float(score_overrides.get(sym, 0.0) or 0.0)))
            return float(base + 0.20 * bonus)

        # Ensure active is a subset of candidates
        cands_set = set(cands)
        self.active = [s for s in self.active if s in cands_set]

        # Score all candidates
        scored: List[Tuple[float, str]] = [(_combined_score(s), s) for s in cands]
        scored.sort(key=lambda t: float(t[0]), reverse=True)

        # Bootstrap: if empty, pick top-scored candidates up to effective cap.
        if not self.active:
            self.active = [s for _, s in scored[:effective_max_active]]
            for s in self.active:
                self._hold_until_cycle[s] = int(cycle_id + self.min_hold_cycles)
            return list(self.active)

        # Keep currently held symbols until hold expires
        held = []
        for s in list(self.active):
            if int(self._hold_until_cycle.get(s, 0) or 0) > int(cycle_id):
                held.append(s)

        # Fill remaining slots from best-scoring candidates; optional correlation/diversification bonus
        out = list(dict.fromkeys(held))  # preserve order, unique
        corr_provider = getattr(self, "_correlation_provider", None)
        for _, s in scored:
            if len(out) >= effective_max_active:
                break
            if s in out:
                continue
            # Optional: prefer symbols that reduce correlation with current active
            if corr_provider is not None and out:
                try:
                    avg_corr = 0.0
                    if callable(getattr(corr_provider, "avg_correlation_with", None)):
                        avg_corr = float(corr_provider.avg_correlation_with(s, out) or 0.0)
                    elif callable(corr_provider):
                        avg_corr = float(corr_provider(s, out) or 0.0)
                    if avg_corr > 0.85:  # skip if highly correlated with existing
                        continue
                except Exception as _e:
                    logger.debug("universe_selector error: %s", _e)
            out.append(s)
            self._hold_until_cycle[s] = int(cycle_id + self.min_hold_cycles)

        # If we have too many (e.g., held > max_active), trim worst-scoring among held
        if len(out) > effective_max_active:
            out_sc = sorted([(_combined_score(s), s) for s in out], key=lambda t: float(t[0]), reverse=True)
            out = [s for _, s in out_sc[:effective_max_active]]

        self.active = out
        return list(self.active)
