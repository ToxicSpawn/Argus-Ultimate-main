from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class TargetPosition:
    symbol: str
    current_qty: float
    target_qty: float
    delta_qty: float
    price: float
    vol_scale: float
    cluster_scale: float


class TargetPortfolioEngine:
    """
    Signals -> targets -> execution deltas.
    Keeps output deterministic and converges gradually toward targets.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    @staticmethod
    def _strategy_name(sig: Any) -> str:
        return str(getattr(sig, "strategy", "") or getattr(sig, "source_strategy", "") or "")

    @staticmethod
    def _signal_fields(sig: Any) -> Tuple[str, str, float, float, float]:
        symbol = str(getattr(sig, "symbol", "") or "")
        action = str(getattr(sig, "action", "") or "").upper()
        confidence = float(getattr(sig, "confidence", 0.0) or 0.0)
        qty = float(getattr(sig, "quantity", 0.0) or 0.0)
        entry = float(getattr(sig, "entry_price", 0.0) or 0.0)
        return symbol, action, confidence, qty, entry

    def _vol_scale(self) -> float:
        target_vol_pct = float(getattr(self.config, "target_vol_pct", 2.0) or 2.0)
        realized_vol_pct = float(getattr(self.config, "realized_vol_pct", 0.0) or 0.0)
        if realized_vol_pct <= 0:
            return 1.0
        return max(0.2, min(1.0, target_vol_pct / max(realized_vol_pct, 1e-9)))

    def build_targets(
        self,
        *,
        signals: Iterable[Any],
        current_positions: Dict[str, Dict[str, Any]],
        equity_aud: float,
    ) -> List[TargetPosition]:
        aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
        max_pos_pct = float(getattr(self.config, "max_position_pct", 0.1) or 0.1)
        max_pos_aud = float(getattr(self.config, "max_position_size_aud", 0.0) or 0.0)
        smoothing = float(getattr(self.config, "target_convergence_alpha", 0.35) or 0.35)
        smoothing = max(0.05, min(1.0, smoothing))
        vol_scale = self._vol_scale()

        # Desired target notional per symbol from input signals.
        desired_notional_aud: Dict[str, float] = {}
        price_by_symbol: Dict[str, float] = {}
        strategy_by_symbol: Dict[str, str] = {}

        for sig in sorted(signals, key=lambda s: (str(getattr(s, "symbol", "") or ""), self._strategy_name(s))):
            symbol, action, confidence, qty, entry = self._signal_fields(sig)
            if not symbol or entry <= 0:
                continue
            quote = symbol.split("/")[-1].upper() if "/" in symbol else "USD"
            notional_quote = qty * entry
            if notional_quote <= 0:
                base_cap = min(max_pos_aud if max_pos_aud > 0 else equity_aud, equity_aud * max_pos_pct)
                notional_aud = max(0.0, base_cap * max(0.0, min(1.0, confidence)))
            elif quote == "AUD":
                notional_aud = notional_quote
            else:
                notional_aud = notional_quote / max(aud_to_usd, 1e-9)
            sign = 1.0 if action == "BUY" else -1.0
            desired_notional_aud[symbol] = desired_notional_aud.get(symbol, 0.0) + sign * notional_aud * vol_scale
            price_by_symbol[symbol] = entry
            strategy_by_symbol[symbol] = self._strategy_name(sig)

        # Cluster exposure caps (simple).
        cluster_map = dict(getattr(self.config, "target_cluster_map", {}) or {})
        cluster_cap_pct = float(getattr(self.config, "target_cluster_cap_pct", 0.4) or 0.4)
        cluster_cap_aud = max(0.0, equity_aud * cluster_cap_pct)
        cluster_totals: Dict[str, float] = {}
        for sym, notion in desired_notional_aud.items():
            cluster = str(cluster_map.get(sym, sym))
            cluster_totals[cluster] = cluster_totals.get(cluster, 0.0) + abs(float(notion))
        cluster_scale_by_symbol: Dict[str, float] = {}
        for sym, notion in desired_notional_aud.items():
            cluster = str(cluster_map.get(sym, sym))
            total = max(cluster_totals.get(cluster, 0.0), 1e-9)
            scale = min(1.0, cluster_cap_aud / total) if cluster_cap_aud > 0 else 1.0
            cluster_scale_by_symbol[sym] = scale
            desired_notional_aud[sym] = notion * scale

        # Build targets and smooth convergence.
        targets: List[TargetPosition] = []
        for symbol in sorted(desired_notional_aud.keys()):
            target_notional = float(desired_notional_aud[symbol])
            entry = float(price_by_symbol.get(symbol, 0.0) or 0.0)
            if entry <= 0:
                continue
            quote = symbol.split("/")[-1].upper() if "/" in symbol else "USD"
            px_aud = entry if quote == "AUD" else (entry / max(aud_to_usd, 1e-9))
            current_qty = float((current_positions.get(symbol) or {}).get("quantity", 0.0) or 0.0)
            raw_target_qty = target_notional / max(px_aud, 1e-9)
            smoothed_target = current_qty + (raw_target_qty - current_qty) * smoothing
            delta = smoothed_target - current_qty
            targets.append(
                TargetPosition(
                    symbol=symbol,
                    current_qty=current_qty,
                    target_qty=smoothed_target,
                    delta_qty=delta,
                    price=entry,
                    vol_scale=vol_scale,
                    cluster_scale=float(cluster_scale_by_symbol.get(symbol, 1.0)),
                )
            )
        return targets

    def build_execution_signals(
        self,
        *,
        targets: Iterable[TargetPosition],
        min_qty: float = 1e-9,
        regime_label: str | None = None,
    ) -> List[Any]:
        out: List[Any] = []
        for t in targets:
            if abs(float(t.delta_qty)) <= min_qty:
                continue
            action = "BUY" if t.delta_qty > 0 else "SELL"
            out.append(
                SimpleNamespace(
                    symbol=t.symbol,
                    action=action,
                    quantity=abs(float(t.delta_qty)),
                    confidence=1.0,
                    entry_price=float(t.price),
                    strategy="target_rebalance",
                    source_strategy="target_rebalance",
                    regime_label=regime_label or "",
                )
            )
        return out
