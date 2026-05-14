from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class TargetPosition:
    # Required v1 normalized target fields.
    symbol: str
    target_exposure_pct: float
    current_exposure_pct: float
    delta_exposure_pct: float
    priority_score: float
    expected_net_edge_bps: float
    regime_label: Optional[str]
    reasons: List[str] = field(default_factory=list)
    # Compatibility/runtime helpers (kept additive).
    price: float = 0.0
    reference_price: float = 0.0
    current_qty: float = 0.0
    target_qty: float = 0.0
    delta_qty: float = 0.0
    # Liquidity-aware risk metadata (optional, populated by liquidity risk engine).
    liquidity_score: float = 0.0
    liquidity_state: str = ""
    max_safe_trade_size: float = 0.0
    adjusted_target_exposure_pct: float = 0.0
    liquidity_clamp_flag: bool = False
    slippage_estimate_bps: float = 0.0
    # Meta-weighting and microstructure context (optional passthrough).
    strategy_weight: float = 0.0
    meta_priority_adjustment: float = 0.0
    weighting_reason: str = ""
    order_book_imbalance: float = 0.0
    spread_bps: float = 0.0
    microprice: float = 0.0
    trade_velocity: float = 0.0
    liquidity_vacuum_flag: bool = False
    adverse_selection_risk: float = 0.0
    microstructure_bias: str = ""


class PortfolioTargetEngine:
    """
    Signals -> target exposure -> execution deltas.
    Deterministic v1 implementation focused on correctness and auditability.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    @staticmethod
    def _signal_get(sig: Any, name: str, default: Any = None) -> Any:
        if isinstance(sig, dict):
            return sig.get(name, default)
        return getattr(sig, name, default)

    @staticmethod
    def _strategy_name(sig: Any) -> str:
        return str(
            PortfolioTargetEngine._signal_get(sig, "source_strategy", "")
            or PortfolioTargetEngine._signal_get(sig, "strategy", "")
            or ""
        )

    @staticmethod
    def _signal_fields(sig: Any) -> Tuple[str, str, float, float, float]:
        symbol = str(PortfolioTargetEngine._signal_get(sig, "symbol", "") or "")
        action = str(
            PortfolioTargetEngine._signal_get(sig, "action", "")
            or PortfolioTargetEngine._signal_get(sig, "side", "")
            or ""
        ).upper()
        confidence = float(PortfolioTargetEngine._signal_get(sig, "confidence", 0.0) or 0.0)
        qty = float(PortfolioTargetEngine._signal_get(sig, "quantity", 0.0) or 0.0)
        entry = float(
            PortfolioTargetEngine._signal_get(sig, "entry_price", 0.0)
            or PortfolioTargetEngine._signal_get(sig, "price", 0.0)
            or 0.0
        )
        return symbol, action, confidence, qty, entry

    def _expected_net_edge_bps(self, sig: Any, confidence: float) -> float:
        edge = self._signal_get(sig, "expected_net_edge_bps", None)
        if edge is None:
            edge = self._signal_get(sig, "net_edge_bps", None)
        if edge is None:
            edge = self._signal_get(sig, "edge_bps", None)
        if edge is None:
            # Conservative deterministic fallback when strategy does not provide edge.
            edge = max(0.0, confidence) * 10.0
        try:
            return float(edge)
        except (TypeError, ValueError):
            return 0.0

    def _liquidity_quality(self, sig: Any) -> float:
        direct = self._signal_get(sig, "liquidity_quality", None)
        if isinstance(direct, (int, float)):
            return max(0.5, min(1.5, float(direct)))
        spread_bps = float(self._signal_get(sig, "spread_bps", 0.0) or 0.0)
        depth = float(self._signal_get(sig, "depth", 0.0) or 0.0)
        q = 1.0
        if spread_bps > 25.0:
            q *= 0.8
        elif spread_bps > 0.0 and spread_bps < 5.0:
            q *= 1.1
        if depth > 100.0:
            q *= 1.1
        elif depth > 0.0 and depth < 10.0:
            q *= 0.85
        return max(0.5, min(1.5, q))

    def _regime_factor(self, sig: Any, regime_label: str | None) -> float:
        if not bool(getattr(self.config, "target_regime_boost_enabled", True)):
            return 1.0
        regime = str(regime_label or "")
        if not regime:
            return 1.0
        strategy = self._strategy_name(sig)
        mapping = dict(getattr(self.config, "regime_strategy_map", {}) or {})
        allowed = mapping.get(regime)
        if allowed is None:
            return 1.0
        allowed_set = {str(x) for x in list(allowed or []) if str(x)}
        if strategy in allowed_set:
            return 1.1
        return 0.75

    @staticmethod
    def _px_aud(symbol: str, price: float, aud_to_usd: float) -> float:
        quote = symbol.split("/")[-1].upper() if "/" in symbol else "USD"
        return float(price) if quote == "AUD" else (float(price) / max(float(aud_to_usd), 1e-9))

    def _effective_risk_scale(self, risk_scale: float | None = None) -> float:
        # v1 deterministic reduced-risk scaler from runtime + caller.
        scales = [1.0]
        runtime_scale = getattr(self.config, "_runtime_risk_scale", None)
        if isinstance(runtime_scale, (int, float)):
            scales.append(float(runtime_scale))
        if isinstance(risk_scale, (int, float)):
            scales.append(float(risk_scale))
        scale = min(scales)
        return max(0.0, min(1.0, scale))

    def build_targets(
        self,
        *,
        signals: Iterable[Any],
        current_positions: Dict[str, Dict[str, Any]],
        equity_aud: float,
        regime_label: str | None = None,
        risk_scale: float | None = None,
        rebalance_min_delta_pct: float | None = None,
    ) -> List[TargetPosition]:
        eq = max(0.0, float(equity_aud or 0.0))
        if eq <= 0:
            return []

        aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
        max_pos_pct = float(getattr(self.config, "max_position_pct", 0.10) or 0.10)
        max_total_pct = float(getattr(self.config, "max_total_exposure_pct", 0.50) or 0.50)
        rebalance_min = (
            float(rebalance_min_delta_pct)
            if rebalance_min_delta_pct is not None
            else float(getattr(self.config, "target_rebalance_min_delta_pct", 0.005) or 0.005)
        )
        conf_w = float(getattr(self.config, "target_score_confidence_weight", 1.0) or 1.0)
        edge_w = float(getattr(self.config, "target_score_net_edge_weight", 1.0) or 1.0)
        effective_scale = self._effective_risk_scale(risk_scale)

        by_symbol_score: Dict[str, float] = {}
        by_symbol_edge_weighted: Dict[str, float] = {}
        by_symbol_weight_sum: Dict[str, float] = {}
        by_symbol_price: Dict[str, float] = {}
        by_symbol_reasons: Dict[str, List[str]] = {}

        ordered = sorted(
            list(signals or []),
            key=lambda s: (
                str(self._signal_get(s, "symbol", "") or ""),
                self._strategy_name(s),
            ),
        )
        for sig in ordered:
            symbol, action, confidence, _qty, entry = self._signal_fields(sig)
            if not symbol:
                continue
            confidence = max(0.0, min(1.0, float(confidence)))
            if entry > 0:
                by_symbol_price[symbol] = float(entry)
            edge_bps = self._expected_net_edge_bps(sig, confidence=confidence)
            pos_edge = max(float(edge_bps), 0.0)
            regime_factor = self._regime_factor(sig, regime_label)
            liq_factor = self._liquidity_quality(sig)
            base_score = confidence * pos_edge
            score = base_score * conf_w * edge_w * regime_factor * liq_factor
            sign = -1.0 if action == "SELL" else 1.0
            by_symbol_score[symbol] = by_symbol_score.get(symbol, 0.0) + (score * sign)
            by_symbol_edge_weighted[symbol] = by_symbol_edge_weighted.get(symbol, 0.0) + (pos_edge * max(confidence, 1e-9))
            by_symbol_weight_sum[symbol] = by_symbol_weight_sum.get(symbol, 0.0) + max(confidence, 1e-9)
            r = by_symbol_reasons.setdefault(symbol, [])
            r.append(f"signal:{self._strategy_name(sig) or 'unknown'}")

        # Build current exposure snapshot first (for symbols with or without new signals).
        current_exposure: Dict[str, float] = {}
        symbols = set(by_symbol_score.keys()) | set((current_positions or {}).keys())
        for symbol in symbols:
            pos = (current_positions or {}).get(symbol) or {}
            qty = float(pos.get("quantity", 0.0) or 0.0)
            px = float(pos.get("current_price", 0.0) or by_symbol_price.get(symbol, 0.0) or 0.0)
            if qty <= 0 or px <= 0:
                current_exposure[symbol] = 0.0
                continue
            px_aud = self._px_aud(symbol, px, aud_to_usd)
            current_notional = max(0.0, qty * px_aud)
            current_exposure[symbol] = max(0.0, current_notional / eq)

        positive_symbols = [s for s, v in sorted(by_symbol_score.items()) if float(v) > 0.0]
        score_sum = sum(max(0.0, float(by_symbol_score.get(s, 0.0))) for s in positive_symbols)

        raw_targets: Dict[str, float] = {}
        total_cap = max(0.0, max_total_pct * effective_scale)
        if score_sum > 0.0 and total_cap > 0.0:
            for symbol in positive_symbols:
                w = max(0.0, float(by_symbol_score.get(symbol, 0.0))) / score_sum
                raw_targets[symbol] = total_cap * w
        # Symbols with non-positive score implicitly target to 0.
        for symbol in symbols:
            raw_targets.setdefault(symbol, 0.0)

        # Symbol cap then portfolio cap normalization.
        clamped: Dict[str, float] = {
            symbol: max(0.0, min(float(v), max(0.0, max_pos_pct * effective_scale)))
            for symbol, v in raw_targets.items()
        }
        cluster_map = dict(getattr(self.config, "target_cluster_map", {}) or {})
        cluster_cap_pct = float(getattr(self.config, "target_cluster_cap_pct", 1.0) or 1.0)
        if cluster_cap_pct > 0:
            cluster_cap = max(0.0, cluster_cap_pct * effective_scale)
            cluster_totals: Dict[str, float] = {}
            for symbol, exp in clamped.items():
                cluster = str(cluster_map.get(symbol, symbol))
                cluster_totals[cluster] = cluster_totals.get(cluster, 0.0) + float(max(0.0, exp))
            for symbol in list(clamped.keys()):
                cluster = str(cluster_map.get(symbol, symbol))
                total = float(cluster_totals.get(cluster, 0.0))
                if total > cluster_cap > 0:
                    clamped[symbol] = float(clamped[symbol]) * (cluster_cap / max(total, 1e-9))
        clamped_total = sum(clamped.values())
        if clamped_total > total_cap > 0.0:
            ratio = total_cap / max(clamped_total, 1e-9)
            for symbol in clamped:
                clamped[symbol] = float(clamped[symbol]) * ratio

        targets: List[TargetPosition] = []
        for symbol in sorted(symbols):
            target_pct = float(clamped.get(symbol, 0.0))
            current_pct = float(current_exposure.get(symbol, 0.0))
            delta_pct = float(target_pct - current_pct)
            score = max(0.0, float(by_symbol_score.get(symbol, 0.0)))
            wsum = max(float(by_symbol_weight_sum.get(symbol, 0.0)), 1e-9)
            edge = float(by_symbol_edge_weighted.get(symbol, 0.0) / wsum)
            reasons = list(by_symbol_reasons.get(symbol, []))
            if abs(delta_pct) < rebalance_min:
                reasons.append("suppressed:small_delta")
            if effective_scale < 0.999:
                reasons.append(f"risk_scale:{effective_scale:.4f}")

            # Qty conversion for downstream execution compatibility.
            px = float(by_symbol_price.get(symbol, 0.0) or (current_positions.get(symbol) or {}).get("current_price", 0.0) or 0.0)
            qty = float((current_positions.get(symbol) or {}).get("quantity", 0.0) or 0.0)
            if px > 0:
                px_aud = self._px_aud(symbol, px, aud_to_usd)
                target_qty = (target_pct * eq) / max(px_aud, 1e-9)
            else:
                target_qty = qty
            delta_qty = target_qty - qty

            targets.append(
                TargetPosition(
                    symbol=symbol,
                    target_exposure_pct=target_pct,
                    current_exposure_pct=current_pct,
                    delta_exposure_pct=delta_pct,
                    priority_score=score,
                    expected_net_edge_bps=edge,
                    regime_label=str(regime_label) if regime_label else None,
                    reasons=reasons,
                    price=px,
                    reference_price=px,
                    current_qty=qty,
                    target_qty=target_qty,
                    delta_qty=delta_qty,
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
        for t in sorted(list(targets or []), key=lambda x: x.symbol):
            delta_pct = float(t.delta_exposure_pct)
            if abs(delta_pct) <= 0.0:
                continue
            if any(str(r).startswith("suppressed:small_delta") for r in list(t.reasons or [])):
                continue
            qty = abs(float(t.delta_qty))
            if qty <= min_qty:
                continue
            action = "BUY" if delta_pct > 0 else "SELL"
            out.append(
                SimpleNamespace(
                    symbol=t.symbol,
                    action=action,
                    quantity=qty,
                    confidence=1.0,
                    entry_price=float(t.reference_price),
                    strategy="target_rebalance",
                    source_strategy="target_rebalance",
                    regime_label=str(regime_label or t.regime_label or ""),
                    target_exposure_pct=float(t.target_exposure_pct),
                    current_exposure_pct=float(t.current_exposure_pct),
                    delta_exposure_pct=float(t.delta_exposure_pct),
                    priority_score=float(t.priority_score),
                    expected_net_edge_bps=float(t.expected_net_edge_bps),
                    target_reasons=list(t.reasons or []),
                    liquidity_score=float(getattr(t, "liquidity_score", 0.0) or 0.0),
                    liquidity_state=str(getattr(t, "liquidity_state", "") or ""),
                    max_safe_trade_size=float(getattr(t, "max_safe_trade_size", 0.0) or 0.0),
                    adjusted_target_exposure_pct=float(
                        getattr(t, "adjusted_target_exposure_pct", getattr(t, "target_exposure_pct", 0.0))
                        or 0.0
                    ),
                    liquidity_clamp_flag=bool(getattr(t, "liquidity_clamp_flag", False)),
                    slippage_estimate_bps=float(getattr(t, "slippage_estimate_bps", 0.0) or 0.0),
                    strategy_weight=float(getattr(t, "strategy_weight", 0.0) or 0.0),
                    meta_priority_adjustment=float(getattr(t, "meta_priority_adjustment", 0.0) or 0.0),
                    weighting_reason=str(getattr(t, "weighting_reason", "") or ""),
                    order_book_imbalance=float(getattr(t, "order_book_imbalance", 0.0) or 0.0),
                    spread_bps=float(getattr(t, "spread_bps", 0.0) or 0.0),
                    microprice=float(getattr(t, "microprice", 0.0) or 0.0),
                    trade_velocity=float(getattr(t, "trade_velocity", 0.0) or 0.0),
                    liquidity_vacuum_flag=bool(getattr(t, "liquidity_vacuum_flag", False)),
                    adverse_selection_risk=float(getattr(t, "adverse_selection_risk", 0.0) or 0.0),
                    microstructure_bias=str(getattr(t, "microstructure_bias", "") or ""),
                )
            )
        return out


# Backwards-compatible name used in runtime/tests.
TargetPortfolioEngine = PortfolioTargetEngine
